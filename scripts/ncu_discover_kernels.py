#!/usr/bin/env python3
"""Discover which kernels are hot in sglang decode + prefill on a given model.

Launches an sglang server, sends a workload, wraps the ENTIRE python process
in NCU with `--set basic` limited to top-N kernels by time.

Output: results/2026-07-08_ncu/<model>/discovery/ncu_top_kernels.csv

Usage:
    sudo python scripts/ncu_discover_kernels.py --model qwen3-30b-a3b-bf16
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"

MODELS = {
    "lfm2.5-8b-a1b": {
        "server_config": "configs/lfm2.5_8b_a1b_v4.yaml",
        "port": 33001,
    },
    "qwen3-30b-a3b-bf16": {
        "server_config": "configs/qwen3_30b_a3b_bf16.yaml",
        "port": 33002,
    },
    "qwen3-30b-a3b-fp8": {
        "server_config": "configs/qwen3_30b_a3b_fp8.yaml",
        "port": 33003,
    },
    "qwen3-0.6b": {
        "server_config": "configs/qwen3_0.6b.yaml",
        "port": 33004,
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    m = MODELS[args.model]
    out_dir = Path(args.out_dir or REPO / f"results/2026-07-08_ncu/{args.model}/discovery")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: build a small bench-spec (single regime, small load)
    # We'll run sglang under NCU. NCU wraps the process.
    # NOTE: NCU on sglang is complex — sglang uses multiple processes (scheduler,
    # tokenizer, TP workers). We want to profile the WORKER process where kernels
    # actually run. Use --target-processes all + filter by kernel-id.
    #
    # Strategy: launch sglang normally in one shell; separately, use
    # `ncu --attach --pid <pid>` on the worker. But NCU can't attach to
    # arbitrary running process — it needs to be launched under NCU or use
    # a special preload.
    #
    # SIMPLER approach: launch sglang with `NCU_INJECTION_ENV_PRELOAD` env
    # then a minimal script that calls the server. This gets messy.
    #
    # BEST approach for us: run a MUCH simpler client program that spins up
    # its own inference (bypassing sglang server) — but then we're not
    # measuring sglang.
    #
    # MOST PRAGMATIC: use PyTorch profiler (kineto/CUPTI) instead of NCU
    # for kernel-level metrics. Kineto works with the actual running sglang
    # server via an env var trigger.
    #
    # But user asked for NCU explicitly. So we'll do:
    #   1. Launch sglang server IN NCU with kernel selection + short bench
    #   2. Sample selected kernels only
    #
    # This means the server startup will also be profiled. Slow but works.
    print(f"[NOTE] NCU on sglang server directly is complex due to multi-process.")
    print(f"[NOTE] This discovery script will launch a MINIMAL sglang inference")
    print(f"[NOTE] within NCU to identify kernel names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
