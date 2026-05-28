#!/usr/bin/env python3
"""Skill impl: noise-aware-scoring — compute CV of metrics over N repeats.

Spins one server and runs the same bench_serving N times back-to-back. Writes
mean/std/CV per metric.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from utils import (  # noqa: E402
    PROJECT_ROOT, SGLANG_CONDA_ENV, conda_run_argv, env_for_config,
    free_port_or_die, kill_process_group, load_yaml, now_compact, save_json,
    yaml_config_to_argv,
)


METRICS_TO_TRACK = [
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p99_ms",
    "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
    "output_throughput", "request_throughput",
    "e2e_p50_ms", "e2e_p90_ms", "e2e_p99_ms",
]


def stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "cv_pct": None, "min": None, "max": None}
    mean = statistics.mean(xs)
    std = statistics.pstdev(xs) if n > 1 else 0.0
    cv_pct = (std / mean * 100.0) if mean else 0.0
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cv_pct": round(cv_pct, 2),
        "min": min(xs),
        "max": max(xs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conda-env", default=SGLANG_CONDA_ENV)
    ap.add_argument("--server-start-timeout", type=int, default=240)
    ap.add_argument("--benchmark-timeout", type=int, default=300)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    wl_path = Path(args.workload).resolve()
    cfg = load_yaml(cfg_path)
    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 30000))

    free_port_or_die(host, port)

    tmp = PROJECT_ROOT / "experiments" / "tmp" / f"noise_calibration_{now_compact()}"
    tmp.mkdir(parents=True, exist_ok=True)
    server_log = tmp / "server.log"

    # Launch server ONCE
    argv = conda_run_argv(["-m", "sglang.launch_server", *yaml_config_to_argv(cfg)],
                          conda_env=args.conda_env)
    env = env_for_config(cfg, conda_env=args.conda_env)
    print(f"[noise] launching server: {' '.join(argv)}")
    logf = open(server_log, "ab")
    proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT,
                            env=env, cwd=str(PROJECT_ROOT), start_new_session=True)
    try:
        # wait_ready inline
        from urllib.request import urlopen
        from urllib.error import URLError
        deadline = time.monotonic() + args.server_start_timeout
        ready = False
        while time.monotonic() < deadline:
            try:
                with urlopen(f"http://{host}:{port}/health", timeout=1.5) as r:
                    if 200 <= r.status < 500:
                        ready = True
                        break
            except (URLError, OSError, TimeoutError):
                pass
            time.sleep(2.0)
        if not ready:
            print("[noise] server not ready", file=sys.stderr)
            save_json({"schema_version": 1, "ok": False, "error": "server_not_ready"}, args.out)
            return 2

        # Warmup
        from utils import save_yaml
        from run_benchmark import build_argv as build_bench_argv
        warm_path = tmp / "warmup.yaml"
        wl = load_yaml(wl_path)
        warm = dict(wl); warm["traffic"] = dict(wl["traffic"])
        warm["traffic"]["num_prompts"] = max(4, min(8, warm["traffic"]["num_prompts"]))
        save_yaml(warm, warm_path)
        wraw = tmp / "warmup_raw.jsonl"
        wlog = tmp / "warmup.log"
        wargv = build_bench_argv(warm, cfg, wraw, args.conda_env)
        print("[noise] warmup ...")
        with open(wlog, "ab") as wf:
            subprocess.call(wargv, stdout=wf, stderr=subprocess.STDOUT, env=env, timeout=300)

        # Repeats
        per_run_metrics: list[dict] = []
        for i in range(1, args.repeats + 1):
            raw = tmp / f"rep_{i:02d}_raw.jsonl"
            blog = tmp / f"rep_{i:02d}_bench.log"
            mout = tmp / f"rep_{i:02d}_metrics.json"
            if raw.exists():
                raw.unlink()
            bargv = build_bench_argv(wl, cfg, raw, args.conda_env)
            print(f"[noise] repeat {i}/{args.repeats}")
            with open(blog, "ab") as bf:
                rc = subprocess.call(bargv, stdout=bf, stderr=subprocess.STDOUT,
                                     env=env, timeout=args.benchmark_timeout)
            if rc != 0:
                print(f"[noise] repeat {i} bench rc={rc}; skipping")
                continue
            # parse with parse_metrics.py
            pmrc = subprocess.call([
                sys.executable, str(SCRIPTS_DIR / "parse_metrics.py"),
                "--raw", str(raw), "--log", str(blog),
                "--server-log", str(server_log),
                "--out", str(mout), "--mode", "repeat",
            ])
            if pmrc == 0 or pmrc == 1:
                with open(mout) as fh:
                    per_run_metrics.append(json.load(fh))

        if not per_run_metrics:
            save_json({"schema_version": 1, "ok": False, "error": "all_repeats_failed"}, args.out)
            return 3

        agg = {}
        for k in METRICS_TO_TRACK:
            agg[k] = stats([m.get(k) for m in per_run_metrics])

        out_doc = {
            "schema_version": 1,
            "ok": True,
            "config": str(cfg_path),
            "workload": str(wl_path),
            "repeats": args.repeats,
            "successful_repeats": len(per_run_metrics),
            "metrics": agg,
            "tmp_dir": str(tmp),
        }
        save_json(out_doc, args.out)
        print(json.dumps({k: {"mean": v["mean"], "cv_pct": v["cv_pct"]} for k, v in agg.items()},
                         indent=2))
        return 0
    finally:
        kill_process_group(proc.pid)


if __name__ == "__main__":
    sys.exit(main())
