#!/usr/bin/env python3
"""Capture a sglang Torch profile against a candidate config + workload.

End-to-end:
  1. Launch sglang server with SGLANG_TORCH_PROFILER_DIR set.
  2. Wait for /health.
  3. (Optional) warmup volley to bypass cold-start bias.
  4. Run bench_serving with --profile --profile-num-steps N.
  5. Locate produced *.pt.trace.json under raw_trace/.
  6. Call parse_trace.py to reduce → profile_summary.json.
  7. Kill server.

This is a stub: it produces a real trace and a real profile_summary.json
following the contract in ../SKILL.md, but parse_trace.py only fills a
subset of fields. See SKILL.md ROADMAP for what's still missing.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Reuse harness primitives.
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils import (  # noqa: E402
    SGLANG_CONDA_ENV,
    conda_run_argv,
    env_for_config,
    free_port_or_die,
    load_yaml,
    yaml_config_to_argv,
)


def wait_for_health(host: str, port: int, timeout: int = 600) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="candidate_config.yaml")
    ap.add_argument("--workload", required=True, help="workload yaml")
    ap.add_argument("--out-dir", required=True, help="output dir (will be created)")
    ap.add_argument("--profile-num-steps", type=int, default=10)
    ap.add_argument("--warmup-requests", type=int, default=16,
                    help="warmup requests to bypass cold-start (Finding-B lesson)")
    ap.add_argument("--server-start-timeout", type=int, default=600)
    ap.add_argument("--conda-env", default=SGLANG_CONDA_ENV)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_trace = out_dir / "raw_trace"
    raw_trace.mkdir(exist_ok=True)
    server_log = out_dir / "server.log"

    config = load_yaml(args.config)
    workload = load_yaml(args.workload)
    host = config.get("host", "127.0.0.1")
    port = int(config.get("port", 30000))
    model = config["model-path"]

    # 1. Launch server with profiler dir baked into env.
    env = env_for_config(config, conda_env=args.conda_env)
    env["SGLANG_TORCH_PROFILER_DIR"] = str(raw_trace)
    free_port_or_die(host, port)
    server_argv = conda_run_argv(
        ["-m", "sglang.launch_server", *yaml_config_to_argv(config)],
        conda_env=args.conda_env,
    )
    print(f"[run_profile] launching server (CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')})")
    with open(server_log, "ab") as logf:
        server = subprocess.Popen(
            server_argv, stdout=logf, stderr=subprocess.STDOUT,
            env=env, cwd=str(PROJECT_ROOT), start_new_session=True,
        )

    try:
        print(f"[run_profile] waiting for /health on {host}:{port} ...")
        if not wait_for_health(host, port, timeout=args.server_start_timeout):
            err = {"schema_version": 0, "ok": False,
                   "error": "server did not become healthy within timeout"}
            (out_dir / "profile_summary.json").write_text(json.dumps(err, indent=2))
            return 2        # 2. Build bench_serving argv (mirrors scripts/run_benchmark.build_argv,
        #    but adds --profile flags and uses warmup).
        ds = workload["dataset"]
        traffic = workload["traffic"]
        bench_argv = [
            "-m", "sglang.bench_serving",
            "--backend", "sglang",
            "--host", host, "--port", str(port),
            "--model", model,
            "--dataset-name", ds["name"],
            "--num-prompts", str(traffic["num_prompts"]),
            "--seed", str(workload.get("seed", 1234)),
            "--disable-tqdm",
            "--output-file", str(out_dir / "bench.jsonl"),
            "--profile",
            "--profile-num-steps", str(args.profile_num_steps),
            "--profile-output-dir", str(raw_trace),
            "--profile-prefix", "p_",
            "--profile-activities", "CPU", "GPU",
        ]
        if args.warmup_requests > 0:
            bench_argv.extend(["--warmup-requests", str(args.warmup_requests)])
        mc = traffic.get("max_concurrency")
        if mc is not None:
            bench_argv.extend(["--max-concurrency", str(mc)])
        if ds["name"] in ("random", "random-ids"):
            bench_argv.extend([
                "--random-input-len", str(ds["random_input_len"]),
                "--random-output-len", str(ds["random_output_len"]),
                "--random-range-ratio", str(ds.get("random_range_ratio", 0.0)),
            ])
        elif ds["name"] == "generated-shared-prefix":
            bench_argv.extend([
                "--gsp-num-groups", str(ds["gsp_num_groups"]),
                "--gsp-prompts-per-group", str(ds["gsp_prompts_per_group"]),
                "--gsp-system-prompt-len", str(ds["gsp_system_prompt_len"]),
                "--gsp-question-len", str(ds["gsp_question_len"]),
                "--gsp-output-len", str(ds["gsp_output_len"]),
            ])
        bench_full = conda_run_argv(bench_argv, conda_env=args.conda_env)
        print(f"[run_profile] running bench with --profile (num-steps={args.profile_num_steps})")
        rc = subprocess.call(bench_full, env=env, cwd=str(PROJECT_ROOT))
        if rc != 0:
            err = {"schema_version": 0, "ok": False,
                   "error": f"bench_serving exited rc={rc}"}
            (out_dir / "profile_summary.json").write_text(json.dumps(err, indent=2))
            return 3

        # Snapshot sglang's /get_server_info before killing the server.
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://{host}:{port}/get_server_info", timeout=10) as r:
                info = json.loads(r.read().decode("utf-8", errors="replace"))
            (out_dir / "server_info.json").write_text(json.dumps(info, indent=2))
            print(f"[run_profile] wrote server_info.json")
        except Exception as e:  # noqa: BLE001
            print(f"[run_profile] warn: /get_server_info failed: {e}")

    finally:
        # 3. Kill server group cleanly.
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except Exception as e:
            print(f"[run_profile] warn: could not kill server cleanly: {e}")

    # 4. Locate trace file(s). sglang writes either *.trace.json or *.trace.json.gz
    # (older versions used *.pt.trace.json). Accept all variants.
    traces = sorted(raw_trace.rglob("*.trace.json")) \
        + sorted(raw_trace.rglob("*.trace.json.gz")) \
        + sorted(raw_trace.rglob("*.pt.trace.json")) \
        + sorted(raw_trace.rglob("*.pt.trace.json.gz"))
    if not traces:
        err = {"schema_version": 0, "ok": False,
               "error": "no trace files produced under raw_trace/"}
        (out_dir / "profile_summary.json").write_text(json.dumps(err, indent=2))
        return 4
    trace_path = traces[0]
    print(f"[run_profile] trace captured: {trace_path}")

    # 5. Reduce.
    parser_script = HERE.parent / "parse_trace.py"
    rc = subprocess.call([
        sys.executable, str(parser_script),
        "--trace", str(trace_path),
        "--out", str(out_dir / "profile_summary.json"),
        "--workload-yaml", str(args.workload),
        "--config-yaml", str(args.config),
    ])
    return rc


if __name__ == "__main__":
    sys.exit(main())
