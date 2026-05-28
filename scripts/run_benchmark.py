#!/usr/bin/env python3
"""Run one sglang.bench_serving invocation against an already-running server.

Reads workload YAML, builds the bench_serving argv, and forwards stdout/stderr
to a log file. Always writes the raw jsonl summary to --raw-out.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils import SGLANG_CONDA_ENV, conda_run_argv, load_yaml


def build_argv(workload: dict, server_cfg: dict, raw_out: Path, conda_env: str) -> list[str]:
    ds = workload["dataset"]
    name = ds["name"]
    traffic = workload["traffic"]
    model = server_cfg["model-path"]
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 30000)

    argv = [
        "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--host", host,
        "--port", str(port),
        "--model", model,
        "--dataset-name", name,
        "--num-prompts", str(traffic["num_prompts"]),
        "--output-file", str(raw_out),
        "--output-details",
        "--disable-tqdm",
        "--seed", str(workload.get("seed", 1234)),
    ]

    mc = traffic.get("max_concurrency")
    if mc is not None:
        argv.extend(["--max-concurrency", str(mc)])
    rr = traffic.get("request_rate")
    if rr is not None:
        argv.extend(["--request-rate", str(rr)])

    # dataset-specific
    if name in ("random", "random-ids"):
        argv.extend([
            "--random-input-len", str(ds["random_input_len"]),
            "--random-output-len", str(ds["random_output_len"]),
            "--random-range-ratio", str(ds.get("random_range_ratio", 0.0)),
        ])
    elif name == "generated-shared-prefix":
        argv.extend([
            "--gsp-num-groups", str(ds["gsp_num_groups"]),
            "--gsp-prompts-per-group", str(ds["gsp_prompts_per_group"]),
            "--gsp-system-prompt-len", str(ds["gsp_system_prompt_len"]),
            "--gsp-question-len", str(ds["gsp_question_len"]),
            "--gsp-output-len", str(ds["gsp_output_len"]),
        ])
    elif name == "sharegpt":
        pass  # uses defaults / shipped dataset
    else:
        raise ValueError(f"unsupported dataset name: {name}")

    cache = workload.get("cache", {})
    if cache.get("flush_cache", True):
        argv.append("--flush-cache")

    return conda_run_argv(argv, conda_env=conda_env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--server-config", required=True)
    ap.add_argument("--raw-out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--conda-env", default=SGLANG_CONDA_ENV)
    args = ap.parse_args()

    workload = load_yaml(args.workload)
    server_cfg = load_yaml(args.server_config)

    raw_out = Path(args.raw_out)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    # bench_serving APPENDS to output-file; nuke any pre-existing file to keep parsing clean.
    if raw_out.exists():
        raw_out.unlink()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    argv = build_argv(workload, server_cfg, raw_out, args.conda_env)
    # Inherit env_for_config so bench_serving sees the same writable HF cache
    # and CUDA_HOME as the server.
    from utils import env_for_config
    env = env_for_config(server_cfg, conda_env=args.conda_env)
    print(f"[run_benchmark] argv: {' '.join(argv)}", flush=True)
    print(f"[run_benchmark] HF_HUB_CACHE={env.get('HF_HUB_CACHE','(unset)')}", flush=True)
    print(f"[run_benchmark] log: {log_path}", flush=True)

    with open(log_path, "ab") as logf:
        try:
            rc = subprocess.call(
                argv,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"[run_benchmark] TIMEOUT after {args.timeout}s", file=sys.stderr)
            return 124

    print(f"[run_benchmark] exit={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
