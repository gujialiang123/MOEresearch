#!/usr/bin/env python3
"""v9: NCU kernel profiling on REAL-workload representative points, with the
v8-tuned BEST config, to show that tuning alone cannot reach the hardware limit.

Methodology (same single-process trick as v6):
  - sglang.bench_one_batch (single process) + --profile-activities CUDA_PROFILER
  - sudo ncu --profile-from-start off  (profiles only the benched stage)

What's new vs v6:
  - Representative points come from the REAL toolagent workload characterization
    (v7): input ~2700 tokens, output ~207.  We profile BOTH prefill and decode.
  - Config = v8 tuned winner per model (chunked-prefill + triton MoE), so the
    kernels are running under the BEST config we found -> any remaining gap to
    the hardware ceiling is NOT closable by config tuning.
  - Richer NCU sections incl. MemoryWorkloadAnalysis to record detailed memory
    usage (DRAM bytes, DRAM %, L2 hit, achieved bandwidth) + SM% + TC% + occ.

Run one model per GPU in parallel:
  V9_MODEL=lfm  V9_GPU=1 python scripts/run_v9_ncu_realworkload.py
  V9_MODEL=qwen V9_GPU=2 python scripts/run_v9_ncu_realworkload.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
CONDA_ENV = "/home/t-jialianggu/.conda/envs/sglang-dev"
MODEL_KEY = os.environ.get("V9_MODEL", "lfm")
GPU_ID = os.environ.get("V9_GPU", "1")
OUT_ROOT = REPO / "results" / os.environ.get("V9_OUT", "2026-07-10_v9_ncu_realworkload")

# Model + its v8-tuned best config (chunked-prefill-size).
MODELS = {
    "lfm": {
        "name": "lfm2.5-8b-a1b", "path": "/data/hf/LFM2.5-8B-A1B",
        "chunked": 4096,
    },
    "qwen": {
        "name": "qwen3-30b-a3b-bf16", "path": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
        "chunked": 16384,
    },
}
MODEL = MODELS[MODEL_KEY]

# Representative points from the real toolagent workload (v7): in~2700, out~207.
# We profile prefill (agent's dominant stage) and decode (under concurrency).
# (batch, input_len, output_len, profile_stage)
REGIMES = {
    "agent_prefill_b1":   (1,  2700, 8,  "prefill"),   # agent prefill, single req
    "agent_decode_b32":   (32, 2700, 32, "decode"),    # agent decode @ concurrency 32
    "agent_decode_b64":   (64, 2700, 32, "decode"),    # agent decode @ concurrency 64
}

_only = os.environ.get("V9_REGIMES")
if _only:
    REGIMES = {k: v for k, v in REGIMES.items() if k in _only.split(",")}


def build_wrapper(out_dir: Path, batch: int, inlen: int, outlen: int, stage: str) -> Path:
    wrapper = out_dir / "inference.sh"
    log_prefix = out_dir / "sglang_bench"
    result_file = out_dir / "bench_one_batch_result.jsonl"
    content = f'''#!/bin/bash
export HOME=/home/t-jialianggu
export CUDA_VISIBLE_DEVICES={GPU_ID}
export CUDA_HOME={CONDA_ENV}
export TRITON_CACHE_DIR=/tmp/sglang_triton_cache_v9_{MODEL_KEY}
mkdir -p $TRITON_CACHE_DIR
export CPATH={CONDA_ENV}/targets/x86_64-linux/include:{CONDA_ENV}/lib/python3.11/site-packages/nvidia/cublas/include:{CONDA_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/include
export LIBRARY_PATH={CONDA_ENV}/lib:{CONDA_ENV}/targets/x86_64-linux/lib
export LD_LIBRARY_PATH={CONDA_ENV}/lib:{CONDA_ENV}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH={CONDA_ENV}/bin:/usr/local/bin:/usr/bin:/bin

exec {CONDA_ENV}/bin/python -m sglang.bench_one_batch \\
  --model-path {MODEL["path"]} \\
  --tokenizer-path {MODEL["path"]} \\
  --trust-remote-code \\
  --batch-size {batch} \\
  --input-len {inlen} \\
  --output-len {outlen} \\
  --profile \\
  --profile-activities CUDA_PROFILER \\
  --profile-stage {stage} \\
  --profile-filename-prefix {log_prefix} \\
  --result-filename {result_file} \\
  --run-name ncu_v9 \\
  --mem-fraction-static 0.85 \\
  --chunked-prefill-size {MODEL["chunked"]} \\
  --schedule-policy lpm \\
  --moe-runner-backend triton
'''
    wrapper.write_text(content)
    wrapper.chmod(0o755)
    return wrapper


def run_combo(reg_id: str, batch: int, inlen: int, outlen: int, stage: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = build_wrapper(out_dir, batch, inlen, outlen, stage)
    (out_dir / "combo_params.json").write_text(json.dumps({
        "model": MODEL["name"], "chunked": MODEL["chunked"], "regime": reg_id,
        "batch": batch, "input_len": inlen, "output_len": outlen, "stage": stage,
    }, indent=2))

    ncu_rep = out_dir / "ncu"
    ncu_csv = out_dir / "ncu_raw.csv"
    bench_log = out_dir / "bench.log"

    # Sections: default = memory-focused; override via V9_SECTIONS env (comma list).
    default_sections = ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "LaunchStats"]
    sections = os.environ.get("V9_SECTIONS")
    sections = sections.split(",") if sections else default_sections
    section_args = []
    for s in sections:
        section_args += ["--section", s]
    ncu_cmd = [
        "sudo", "-n", NCU,
        "--target-processes", "all",
        "--profile-from-start", "off",
        "--launch-count", "24",
        "--kernel-name-base", "demangled",
        "--kernel-name", "regex:fused_moe|nvjet|flash|cutlass|RMSNorm|act_and_mul|topk|conv1d|moe_sum|gemm",
        *section_args,
        "--force-overwrite",
        "--export", str(ncu_rep),
        "--", str(wrapper),
    ]
    print(f"  NCU {MODEL['name']} × {reg_id} (b={batch} in={inlen} {stage})...", flush=True)
    t0 = time.time()
    with bench_log.open("w") as f:
        proc = subprocess.run(ncu_cmd, stdout=f, stderr=subprocess.STDOUT, timeout=2400)
    print(f"    exit={proc.returncode} wall={time.time()-t0:.0f}s", flush=True)

    rep_path = out_dir / "ncu.ncu-rep"
    if not rep_path.exists():
        alt = Path(str(ncu_rep) + ".ncu-rep")
        if alt.exists():
            rep_path = alt
    if rep_path.exists():
        with ncu_csv.open("w") as f:
            subprocess.run(["sudo", "-n", NCU, "--import", str(rep_path), "--csv"],
                           stdout=f, stderr=subprocess.DEVNULL, timeout=600)
        print(f"    CSV {ncu_csv.stat().st_size} bytes", flush=True)


def main() -> int:
    t0 = time.time()
    for reg_id, (b, inl, outl, stage) in REGIMES.items():
        out_dir = OUT_ROOT / MODEL["name"] / reg_id
        print(f"\n=== {MODEL['name']} / {reg_id} ({(time.time()-t0)/60:.1f}m) ===", flush=True)
        try:
            run_combo(reg_id, b, inl, outl, stage, out_dir)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
    print(f"\n=== v9 NCU [{MODEL['name']}] done in {(time.time()-t0)/60:.1f} min ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
