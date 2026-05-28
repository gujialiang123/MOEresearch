#!/usr/bin/env python3
"""Run ONE workload end-to-end: launch server → wait → benchmark → parse → cleanup.

This is the unit of work shared by Stage 1 (run_regime_suite.py) and Stage 2/3.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from utils import (
    PROJECT_ROOT,
    SGLANG_CONDA_ENV,
    conda_run_argv,
    env_for_config,
    free_port_or_die,
    kill_process_group,
    load_yaml,
    now_compact,
    save_json,
    yaml_config_to_argv,
    read_text_safe,
)


SCRIPTS_DIR = Path(__file__).resolve().parent


def launch_server(server_cfg_path: Path, log_path: Path, conda_env: str) -> subprocess.Popen:
    cfg = load_yaml(server_cfg_path)
    argv = conda_run_argv(
        ["-m", "sglang.launch_server", *yaml_config_to_argv(cfg)],
        conda_env=conda_env,
    )
    env = env_for_config(cfg, conda_env=conda_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run_experiment] launching: {' '.join(argv)}", flush=True)
    print(f"[run_experiment] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','(unset)')}",
          flush=True)
    print(f"[run_experiment] CUDA_HOME={env.get('CUDA_HOME','(unset)')}", flush=True)
    logf = open(log_path, "ab")
    proc = subprocess.Popen(
        argv,
        stdout=logf,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
        start_new_session=True,
    )
    return proc


def wait_ready(host: str, port: int, timeout: int, log_path: Path) -> bool:
    rc = subprocess.call([
        sys.executable, str(SCRIPTS_DIR / "wait_ready.py"),
        "--host", host, "--port", str(port),
        "--timeout", str(timeout),
    ])
    return rc == 0


def run_bench(workload: Path, server_cfg: Path, raw_out: Path, log: Path,
              timeout: int, conda_env: str) -> int:
    return subprocess.call([
        sys.executable, str(SCRIPTS_DIR / "run_benchmark.py"),
        "--workload", str(workload),
        "--server-config", str(server_cfg),
        "--raw-out", str(raw_out),
        "--log", str(log),
        "--timeout", str(timeout),
        "--conda-env", conda_env,
    ])


def parse(raw: Path, bench_log: Path, server_log: Path, out: Path,
          mode: str, expected: int | None) -> int:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "parse_metrics.py"),
        "--raw", str(raw),
        "--log", str(bench_log),
        "--server-log", str(server_log),
        "--out", str(out),
        "--mode", mode,
    ]
    if expected is not None:
        cmd.extend(["--expected-requests", str(expected)])
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="sglang server YAML config")
    ap.add_argument("--workload", required=True, help="workload YAML")
    ap.add_argument("--mode", default="quick", choices=["quick", "medium", "full"])
    ap.add_argument("--out-dir", default=None,
                    help="If set, write metrics+logs into this dir; "
                         "else into experiments/tmp/<ts>.")
    ap.add_argument("--server-start-timeout", type=int, default=300)
    ap.add_argument("--benchmark-timeout", type=int, default=900)
    ap.add_argument("--warmup", action="store_true",
                    help="Run a tiny warmup benchmark before the real run.")
    ap.add_argument("--conda-env", default=SGLANG_CONDA_ENV)
    args = ap.parse_args()

    server_cfg_path = Path(args.config).resolve()
    workload_path = Path(args.workload).resolve()
    server_cfg = load_yaml(server_cfg_path)
    workload = load_yaml(workload_path)

    host = server_cfg.get("host", "127.0.0.1")
    port = int(server_cfg.get("port", 30000))
    free_port_or_die(host, port)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = PROJECT_ROOT / "experiments" / "tmp" / now_compact()
    out_dir.mkdir(parents=True, exist_ok=True)

    server_log = out_dir / "server.log"
    bench_log = out_dir / f"{args.mode}_benchmark.log"
    raw_out = out_dir / f"{args.mode}_raw.jsonl"
    metrics_out = out_dir / f"{args.mode}_metrics.json"

    # Copy snapshots for reproducibility
    (out_dir / "config_snapshot.yaml").write_bytes(server_cfg_path.read_bytes())
    (out_dir / "workload_snapshot.yaml").write_bytes(workload_path.read_bytes())

    expected_requests = workload.get("traffic", {}).get("num_prompts")

    t0 = time.monotonic()
    proc = launch_server(server_cfg_path, server_log, args.conda_env)
    print(f"[run_experiment] server pid={proc.pid}", flush=True)

    try:
        if not wait_ready(host, port, args.server_start_timeout, server_log):
            print("[run_experiment] server did not become ready", file=sys.stderr)
            # Compose a failure metrics record so caller still has structured output
            from utils import save_json
            save_json({
                "schema_version": 1, "mode": args.mode, "passed": False,
                "parse_error": "server_not_ready",
                "server_crash": True, "oom": False, "timeout": True,
                "raw_files": {"server_log": str(server_log)},
            }, metrics_out)
            return 2

        server_ready_s = time.monotonic() - t0
        print(f"[run_experiment] server ready in {server_ready_s:.1f}s", flush=True)

        if args.warmup:
            warmup_raw = out_dir / "warmup_raw.jsonl"
            warmup_log = out_dir / "warmup.log"
            warmup_workload = workload_path  # use same workload but with tiny num_prompts
            # Build a temporary workload with reduced num_prompts for warmup
            warm_path = out_dir / "warmup_workload.yaml"
            warm = dict(workload)
            warm["traffic"] = dict(workload["traffic"])
            warm["traffic"]["num_prompts"] = min(8, int(warm["traffic"]["num_prompts"]))
            from utils import save_yaml
            save_yaml(warm, warm_path)
            print("[run_experiment] warmup ...", flush=True)
            run_bench(warm_path, server_cfg_path, warmup_raw, warmup_log,
                      timeout=300, conda_env=args.conda_env)

        print(f"[run_experiment] benchmark mode={args.mode} ...", flush=True)
        rc = run_bench(workload_path, server_cfg_path, raw_out, bench_log,
                       timeout=args.benchmark_timeout, conda_env=args.conda_env)
        if rc == 124:
            from utils import save_json
            save_json({
                "schema_version": 1, "mode": args.mode, "passed": False,
                "parse_error": "benchmark_timeout",
                "server_crash": False, "oom": False, "timeout": True,
                "raw_files": {
                    "raw": str(raw_out), "benchmark_log": str(bench_log),
                    "server_log": str(server_log),
                },
            }, metrics_out)
            return 3

        prc = parse(raw_out, bench_log, server_log, metrics_out,
                    args.mode, expected_requests)
        print(f"[run_experiment] parse rc={prc} metrics={metrics_out}", flush=True)
        return prc
    finally:
        print(f"[run_experiment] killing server pid={proc.pid}", flush=True)
        kill_process_group(proc.pid)


if __name__ == "__main__":
    sys.exit(main())
