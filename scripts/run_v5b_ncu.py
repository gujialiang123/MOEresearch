#!/usr/bin/env python3
"""v5b: NCU profiling for top hot kernels only.

Strategy:
  - Skip sglang server (too complex to NCU-wrap).
  - Use transformers to load model, run a controlled inference workload.
  - Wrap in NCU with -k <kernel_name_regex> to capture ONLY the hot kernels.
  - Extract: sm_pipe_tensor_active%, dram_throughput%, sm_active%, l2_hit_rate

For each (model, batch_size) profile with NCU:
  - Load model in bf16
  - Warmup 3 forward passes
  - cudaProfilerStart
  - 5 forward passes at target batch (produces N kernel launches to sample)
  - cudaProfilerStop
  - NCU catches specific kernels

Runs on GPU 6. Uses sudo for NCU.

Output:
  results/2026-07-08_v5b_ncu/<model>/batch_<B>/<kernel_short>/ncu.csv
"""
from __future__ import annotations
import subprocess, os, sys, argparse, json, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NCU = "/opt/nvidia/nsight-compute/2026.2.1/ncu"
PYTHON = "/home/t-jialianggu/.conda/envs/sglang-dev/bin/python"

MODELS = {
    "qwen3-0.6b":         "/data/hf/models/Qwen3-0.6B",
    "lfm2.5-8b-a1b":      "/data/hf/LFM2.5-8B-A1B",
    "qwen3-30b-a3b-bf16": "/data/hf/models/Qwen3-30B-A3B-Instruct-2507",
}


def make_inference_script(model_path: str, batch_size: int, out_path: Path) -> Path:
    """Write a Python script that runs inference at given batch size."""
    script_content = f'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = {model_path!r}
BATCH_SIZE = {batch_size}

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
)
model = model.cuda().eval()

# Build batch: same prompt repeated BATCH_SIZE times
prompt = "The quick brown fox jumps over the lazy dog. " * 20
inputs = tok([prompt] * BATCH_SIZE, return_tensors="pt", padding=True).input_ids.cuda()
print(f"input shape: {{inputs.shape}}", flush=True)

# Warmup 3 forward passes with kv cache
with torch.no_grad():
    for _ in range(3):
        out = model(inputs, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1:].argmax(-1)
torch.cuda.synchronize()
print("Warmup done", flush=True)

# Profiled: 5 decode steps
torch.cuda.cudart().cudaProfilerStart()
with torch.no_grad():
    current = next_tok
    for _ in range(5):
        out = model(current, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        current = out.logits[:, -1:].argmax(-1)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("Profiled section done", flush=True)
'''
    out_path.write_text(script_content)
    return out_path


def run_ncu_on_kernel(model_name: str, model_path: str, batch_size: int,
                     kernel_name: str, kernel_short: str, out_dir: Path,
                     gpu_id: int = 6) -> None:
    """Run NCU targeting a specific kernel pattern for one model+batch."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write inference script
    script = make_inference_script(model_path, batch_size,
                                    out_dir / "inference.py")

    # NCU command
    csv_out = out_dir / "ncu.csv"
    log_out = out_dir / "ncu.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TRANSFORMERS_OFFLINE"] = "1"

    # Filter by kernel name only; don't use profile-from-start to avoid
    # timing issues with cudaProfilerStart. NCU will consider all kernels
    # but only profile those matching -k, limited to -c count.
    cmd = [
        "sudo", "-E", NCU,
        "--target-processes", "all",
        "--set", "full",
        "-k", kernel_name,
        "-c", "5",
        "--csv",
        PYTHON, str(script),
    ]
    print(f"  NCU on {model_name} batch={batch_size} kernel={kernel_short}...", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                             timeout=1800)
    wall = time.time() - t0
    # NCU with --csv writes ALL its report to stdout (kernel data at the end).
    # Program stdout is intermixed. We save both stdout and stderr.
    csv_out.write_text(result.stdout)
    log_out.write_text(result.stderr)
    print(f"    Done in {wall:.0f}s, exit={result.returncode}, "
          f"csv_size={csv_out.stat().st_size}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root",
                    default="results/2026-07-08_v5b_ncu")
    ap.add_argument("--models", nargs="+",
                    default=["qwen3-0.6b", "lfm2.5-8b-a1b", "qwen3-30b-a3b-bf16"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 32, 128])
    ap.add_argument("--gpu", type=int, default=6)
    args = ap.parse_args()

    out_root = REPO / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Top kernels to profile. NCU -k accepts either exact name or
    # 'regex:<expr>'. We use regex to catch all variants.
    kernel_targets = {
        "fused_moe":  "regex:fused_moe_kernel",
        "flash_attn": "regex:flash_fwd",       # matches flash_fwd_splitkv_*
        "gemm":       "regex:nvjet_tst",        # matches all nvjet_tst_* variants
    }

    total = len(args.models) * len(args.batches) * len(kernel_targets)
    idx = 0
    t_start = time.time()

    for m_name in args.models:
        m_path = MODELS[m_name]
        for B in args.batches:
            for k_short, k_regex in kernel_targets.items():
                idx += 1
                elapsed = time.time() - t_start
                print(f"\n>>> [{idx}/{total}] {m_name} × B={B} × {k_short} "
                      f"(elapsed={elapsed:.0f}s) <<<", flush=True)
                out_dir = out_root / m_name / f"batch_{B}" / k_short
                try:
                    run_ncu_on_kernel(m_name, m_path, B, k_regex, k_short,
                                      out_dir, gpu_id=args.gpu)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    import traceback; traceback.print_exc()

    print(f"\n=== v5b NCU sweep done in {(time.time()-t_start)/60:.1f} min ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
