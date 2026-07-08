#!/usr/bin/env python3
"""v6 NCU on REAL sglang kernels via sglang.bench_one_batch.

Fixed approach based on old scripts/bench_ncu_one_regime.sh (2026-06-09):
  - Use sglang.bench_one_batch (single-process CLI, not the HTTP server)
  - Pass --profile-activities CUDA_PROFILER so sglang calls
    cudaProfilerStart/Stop around the bench section
  - Wrap in ncu --profile-from-start off so NCU only profiles between the
    cudaProfilerStart/Stop markers
  - This means NCU sees ONLY the bench-phase kernels, not startup

Advantages over v5b:
  * Kernel names are REAL sglang kernels (fused_moe_kernel etc.)
  * No transformers involvement - measures what sglang actually runs
  * cudaProfilerStart/Stop markers built-in to sglang

Combos:
  3 models × 2 configs × 3 regimes = 18 combos.
  Est ~5-15 min per NCU combo. Total ~3-4h.

Output tree:
  results/2026-07-08_v6_ncu/<model>/<config>/<regime>/
    ncu_raw.csv       (NCU CSV export)
    bench.log         (bench_one_batch stdout+stderr)
    kernel_params.json (record of the config used)
    inference.sh      (the wrapper script used)
    ncu.ncu-rep       (binary NCU report, keep for later inspection)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
CONDA_ENV = "/home/t-jialianggu/.conda/envs/sglang-dev"
GPU_ID = 6

MODELS = [
    {
        "name": "lfm2.5-8b-a1b",
        "model_path": "/data/hf/LFM2.5-8B-A1B",
    },
    {
        "name": "qwen3-30b-a3b-bf16",
        "model_path": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
    },
    {
        "name": "qwen3-30b-a3b-fp8",
        "model_path": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507-FP8",
    },
]

CONFIGS = [
    {
        "name": "cookbook_baseline",
        "sglang_args": [
            "--mem-fraction-static", "0.85",
            "--max-running-requests", "32",
            "--chunked-prefill-size", "-1",
            "--schedule-policy", "lpm",
            "--moe-runner-backend", "auto",
        ],
    },
    {
        "name": "big_batch_cap128",
        "sglang_args": [
            "--mem-fraction-static", "0.90",
            "--max-running-requests", "128",
            "--chunked-prefill-size", "2048",
            "--schedule-policy", "fcfs",
            "--moe-runner-backend", "auto",
        ],
    },
]

# Regimes → (batch_size, input_len, output_len, profile_stage)
REGIMES = {
    "R_decode_c1_out2k":    (1,   130,  128,  "decode"),   # batch=1 pure decode
    "R_conc_ref":           (32,  260,  256,  "decode"),   # batch=32
    "R_decode_c128_out256": (128, 260,  256,  "decode"),   # batch=128
}


def build_wrapper_script(out_dir: Path, model_path: str, config_args: list[str],
                          batch: int, input_len: int, output_len: int,
                          stage: str) -> Path:
    """Write a shell wrapper that sudo'd ncu will invoke.
    Sets env then runs sglang.bench_one_batch with --profile-activities CUDA_PROFILER."""
    wrapper = out_dir / "inference.sh"
    log_prefix = out_dir / "sglang_bench"
    result_file = out_dir / "bench_one_batch_result.jsonl"

    sglang_args = " ".join(config_args)
    content = f'''#!/bin/bash
export HOME=/home/t-jialianggu
export CUDA_VISIBLE_DEVICES={GPU_ID}
export CUDA_HOME={CONDA_ENV}
export TRITON_CACHE_DIR=/tmp/sglang_triton_cache_ncu
mkdir -p $TRITON_CACHE_DIR
export CPATH={CONDA_ENV}/targets/x86_64-linux/include:{CONDA_ENV}/lib/python3.11/site-packages/nvidia/cublas/include:{CONDA_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/include
export LIBRARY_PATH={CONDA_ENV}/lib:{CONDA_ENV}/targets/x86_64-linux/lib
export LD_LIBRARY_PATH={CONDA_ENV}/lib:{CONDA_ENV}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH={CONDA_ENV}/bin:/usr/local/bin:/usr/bin:/bin

exec {CONDA_ENV}/bin/python -m sglang.bench_one_batch \\
  --model-path {model_path} \\
  --tokenizer-path {model_path} \\
  --trust-remote-code \\
  --batch-size {batch} \\
  --input-len {input_len} \\
  --output-len {output_len} \\
  --profile \\
  --profile-activities CUDA_PROFILER \\
  --profile-stage {stage} \\
  --profile-filename-prefix {log_prefix} \\
  --result-filename {result_file} \\
  --run-name ncu_v6 \\
  {sglang_args}
'''
    wrapper.write_text(content)
    wrapper.chmod(0o755)
    return wrapper


def run_ncu_on_combo(model: dict, config: dict, regime_id: str,
                      batch: int, input_len: int, output_len: int, stage: str,
                      out_dir: Path) -> None:
    """Run NCU wrapping sglang.bench_one_batch for one combo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = build_wrapper_script(
        out_dir, model["model_path"], config["sglang_args"],
        batch, input_len, output_len, stage,
    )
    # Record combo params
    (out_dir / "combo_params.json").write_text(json.dumps({
        "model": model["name"],
        "config": config["name"],
        "regime": regime_id,
        "batch": batch, "input_len": input_len,
        "output_len": output_len, "stage": stage,
        "config_args": config["sglang_args"],
    }, indent=2))

    ncu_rep = out_dir / "ncu"
    ncu_log = out_dir / "ncu.log"
    ncu_csv = out_dir / "ncu_raw.csv"
    bench_log = out_dir / "bench.log"

    # NCU command:
    # - profile-from-start off → wait for cudaProfilerStart inside sglang
    # - launch-count 20 → capture only 20 kernel launches (was 30, saves time)
    # - kernel-name regex filter → only hot kernels
    # - --section SpeedOfLight → just get SM% / DRAM% / L2% / Cache metrics
    #   (much lighter than --set basic which has 213 metrics)
    ncu_cmd = [
        "sudo", "-n", NCU,
        "--target-processes", "all",
        "--profile-from-start", "off",
        "--launch-count", "20",
        "--kernel-name-base", "demangled",
        "--kernel-name", "regex:fused_moe|nvjet|flash_fwd|cutlass|RMSNorm|act_and_mul|topk|conv1d",
        "--section", "SpeedOfLight",
        "--section", "Occupancy",
        "--section", "LaunchStats",
        "--force-overwrite",
        "--export", str(ncu_rep),
        "--", str(wrapper),
    ]
    print(f"  Launching NCU on {model['name']} × {config['name']} × {regime_id}...", flush=True)
    t0 = time.time()
    with bench_log.open("w") as f:
        proc = subprocess.run(ncu_cmd, stdout=f, stderr=subprocess.STDOUT,
                              timeout=1800)  # 30 min per combo
    wall = time.time() - t0
    print(f"    NCU exit={proc.returncode}, wall={wall:.0f}s", flush=True)

    # Convert .ncu-rep → csv
    if (out_dir / "ncu.ncu-rep").exists() or Path(str(ncu_rep) + ".ncu-rep").exists():
        rep_path = out_dir / "ncu.ncu-rep"
        if not rep_path.exists():
            rep_path = Path(str(ncu_rep) + ".ncu-rep")
        csv_cmd = ["sudo", "-n", NCU, "--import", str(rep_path), "--csv"]
        with ncu_csv.open("w") as f:
            subprocess.run(csv_cmd, stdout=f, stderr=subprocess.DEVNULL,
                           timeout=600)
        csv_size = ncu_csv.stat().st_size if ncu_csv.exists() else 0
        print(f"    CSV exported: {csv_size} bytes", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root",
                    default="results/2026-07-08_v6_ncu")
    ap.add_argument("--only-model", default=None,
                    help="Run only this model (for smoke testing)")
    ap.add_argument("--only-config", default=None)
    ap.add_argument("--only-regime", default=None)
    args = ap.parse_args()

    models = MODELS if not args.only_model else [m for m in MODELS if m["name"] == args.only_model]
    configs = CONFIGS if not args.only_config else [c for c in CONFIGS if c["name"] == args.only_config]
    regimes = REGIMES if not args.only_regime else {args.only_regime: REGIMES[args.only_regime]}

    out_root = REPO / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(models) * len(configs) * len(regimes)
    idx = 0
    t_start = time.time()
    for m in models:
        for c in configs:
            for reg_id, (batch, inlen, outlen, stage) in regimes.items():
                idx += 1
                elapsed = time.time() - t_start
                print(f"\n>>> [{idx}/{total}] {m['name']} × {c['name']} × {reg_id} "
                      f"(elapsed={elapsed:.0f}s) <<<", flush=True)
                combo_dir = out_root / m["name"] / c["name"] / reg_id
                try:
                    run_ncu_on_combo(m, c, reg_id, batch, inlen, outlen, stage,
                                      combo_dir)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    import traceback; traceback.print_exc()
    print(f"\n=== v6 NCU sweep done in {(time.time()-t_start)/60:.1f} min ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
