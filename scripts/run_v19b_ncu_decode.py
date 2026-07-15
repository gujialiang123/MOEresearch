#!/usr/bin/env python3
"""v19 Part B: NCU decode-kernel profiling with Chendi's EXACT metric list.

Reuses the proven v9 single-process trick (sglang.bench_one_batch +
--profile-activities CUDA_PROFILER + ncu --profile-from-start off), but:
  - Collects EXACTLY the 11 metrics Chendi requested (raw NCU metric names).
  - Sweeps decode batch {32,64,128} to push toward the decode upper bound
    (larger effective batch amortizes expert weight movement -> best-case
    arithmetic intensity), plus one prefill point for reference.
  - Saves ALL artifacts: ncu.ncu-rep + ncu_raw.csv + bench.log + params.

Run: MODEL=qwen GPU=6 python scripts/run_v19b_ncu_decode.py
"""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
CONDA_ENV = "/home/t-jialianggu/.conda/envs/sglang-dev"
MODEL_KEY = os.environ.get("MODEL", "qwen")
GPU_ID = os.environ.get("GPU", "6")
OUT_ROOT = REPO / "results" / os.environ.get("OUT", "2026-07-15_v19b_ncu_decode")

# The 11 metrics Chendi asked for (raw NCU names).
METRICS = ",".join([
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
])

MODELS = {
    "lfm":  {"name": "lfm2.5-8b-a1b",       "path": "/data/hf/LFM2.5-8B-A1B",                         "chunked": 4096},
    "qwen": {"name": "qwen3-30b-a3b-bf16",  "path": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507",    "chunked": 16384},
}
MODEL = MODELS[MODEL_KEY]

# in~2700 agent prefill; decode at increasing batch toward upper bound.
# (batch, input_len, output_len, stage)
REGIMES = {
    "agent_decode_b32":  (32,  2700, 32, "decode"),
    "agent_decode_b64":  (64,  2700, 32, "decode"),
    "agent_decode_b128": (128, 2700, 32, "decode"),
    "agent_prefill_b1":  (1,   2700, 8,  "prefill"),
}
_only = os.environ.get("REGIMES")
if _only:
    REGIMES = {k: v for k, v in REGIMES.items() if k in _only.split(",")}

LAUNCH_COUNT = os.environ.get("LAUNCH_COUNT", "40")  # enough decode-step kernels for stable stats


def build_wrapper(out_dir: Path, batch, inlen, outlen, stage) -> Path:
    wrapper = out_dir / "inference.sh"
    log_prefix = out_dir / "sglang_bench"
    result_file = out_dir / "bench_one_batch_result.jsonl"
    content = f'''#!/bin/bash
export HOME=/home/t-jialianggu
export CUDA_VISIBLE_DEVICES={GPU_ID}
export CUDA_HOME={CONDA_ENV}
export TRITON_CACHE_DIR=/tmp/sglang_triton_cache_v19b_{MODEL_KEY}
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
  --run-name ncu_v19b \\
  --mem-fraction-static 0.85 \\
  --chunked-prefill-size {MODEL["chunked"]} \\
  --schedule-policy lpm \\
  --attention-backend fa3 \\
  --moe-runner-backend triton
'''
    wrapper.write_text(content)
    wrapper.chmod(0o755)
    return wrapper


def run_combo(reg_id, batch, inlen, outlen, stage, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = build_wrapper(out_dir, batch, inlen, outlen, stage)
    (out_dir / "combo_params.json").write_text(json.dumps({
        "model": MODEL["name"], "regime": reg_id, "batch": batch,
        "input_len": inlen, "output_len": outlen, "stage": stage,
        "metrics": METRICS.split(","),
    }, indent=2))

    ncu_rep = out_dir / "ncu"
    ncu_csv = out_dir / "ncu_raw.csv"
    bench_log = out_dir / "bench.log"
    ncu_cmd = [
        "sudo", "-n", NCU,
        "--target-processes", "all",
        "--profile-from-start", "off",
        "--launch-count", LAUNCH_COUNT,
        "--kernel-name-base", "demangled",
        "--kernel-name", "regex:fused_moe|nvjet|flash|cutlass|RMSNorm|act_and_mul|topk|conv1d|moe_sum|gemm",
        "--metrics", METRICS,
        "--force-overwrite",
        "--export", str(ncu_rep),
        "--", str(wrapper),
    ]
    print(f"  NCU {MODEL['name']} × {reg_id} (b={batch} {stage})...", flush=True)
    t0 = time.time()
    with bench_log.open("w") as f:
        proc = subprocess.run(ncu_cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3000)
    print(f"    exit={proc.returncode} wall={time.time()-t0:.0f}s", flush=True)

    rep_path = out_dir / "ncu.ncu-rep"
    if not rep_path.exists():
        alt = Path(str(ncu_rep) + ".ncu-rep")
        if alt.exists():
            rep_path = alt
    if rep_path.exists():
        with ncu_csv.open("w") as f:
            subprocess.run(["sudo", "-n", NCU, "--import", str(rep_path), "--csv", "--page", "raw"],
                           stdout=f, stderr=subprocess.DEVNULL, timeout=600)
        print(f"    CSV {ncu_csv.stat().st_size} bytes", flush=True)
    else:
        print("    !! no ncu-rep produced", flush=True)


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
    print(f"\n=== v19b NCU [{MODEL['name']}] done in {(time.time()-t0)/60:.1f} min ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
