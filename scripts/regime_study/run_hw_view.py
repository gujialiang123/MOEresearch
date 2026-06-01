#!/usr/bin/env python3
"""Hardware-view run: one regime + nvidia-smi sampler + torch profile + /get_server_info.

For each (model, regime) triple this script:
  1. spawns scripts/regime_study/gpu_sampler.py (background)
  2. calls .github/skills/pytorch-profiling/impl/run_profile.py which itself:
     - launches sglang server with SGLANG_TORCH_PROFILER_DIR
     - bench_serving --profile --warmup-requests
     - dumps server_info.json (added)
     - parses trace → profile_summary.json
  3. stops the sampler, computes peak/mean GPU stats during the bench window
  4. writes hardware_view.json combining everything

The output of one call is one self-contained dir:
  out_dir/
    server.log
    bench.jsonl
    raw_trace/<TS>/p_*.trace.json.gz
    profile_summary.json   (top kernels, phase, MoE detection)
    server_info.json        (sglang /get_server_info snapshot)
    gpu_samples.csv         (nvidia-smi every 0.5s)
    hardware_view.json      ← the rolled-up answer to "what does this regime look like at the hardware level?"
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils import load_yaml  # noqa: E402

SKILL_RUN = PROJECT_ROOT / ".github" / "skills" / "pytorch-profiling" / "impl" / "run_profile.py"
SAMPLER = PROJECT_ROOT / "scripts" / "regime_study" / "gpu_sampler.py"


def parse_int_safe(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def parse_float_safe(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def reduce_gpu_samples(csv_path: Path, bench_window: tuple[float, float] | None) -> dict:
    """Compute peak/mean over the bench window."""
    import csv as _csv
    rows = []
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            t = parse_float_safe(r.get("t_unix"))
            if t is None:
                continue
            if bench_window and not (bench_window[0] <= t <= bench_window[1]):
                continue
            rows.append(r)
    if not rows:
        return {"sample_count": 0}

    def col_floats(key):
        return [v for v in (parse_float_safe(r.get(key)) for r in rows) if v is not None]
    mem_used = col_floats("memory.used")
    util_gpu = col_floats("utilization.gpu")
    util_mem = col_floats("utilization.memory")
    power = col_floats("power.draw")
    temp = col_floats("temperature.gpu")
    sm_clk = col_floats("clocks.current.sm")
    return {
        "sample_count": len(rows),
        "memory_used_mib_peak": max(mem_used) if mem_used else None,
        "memory_used_mib_mean": round(sum(mem_used) / len(mem_used), 1) if mem_used else None,
        "utilization_gpu_pct_mean": round(sum(util_gpu) / len(util_gpu), 1) if util_gpu else None,
        "utilization_gpu_pct_p95": sorted(util_gpu)[int(0.95 * (len(util_gpu) - 1))] if util_gpu else None,
        "utilization_mem_pct_mean": round(sum(util_mem) / len(util_mem), 1) if util_mem else None,
        "power_w_mean": round(sum(power) / len(power), 1) if power else None,
        "power_w_peak": max(power) if power else None,
        "temperature_c_peak": max(temp) if temp else None,
        "sm_clock_mhz_mean": round(sum(sm_clk) / len(sm_clk), 0) if sm_clk else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--profile-num-steps", type=int, default=10)
    ap.add_argument("--warmup-requests", type=int, default=8)
    ap.add_argument("--server-start-timeout", type=int, default=300)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler_csv = out_dir / "gpu_samples.csv"

    # 1. start sampler
    sampler = subprocess.Popen(
        [sys.executable, str(SAMPLER), "--gpu", str(args.gpu),
         "--interval", "0.5", "--out", str(sampler_csv)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    print(f"[hw-view] sampler pid={sampler.pid} → {sampler_csv}")

    # 2. capture bench-window-start (after sampler is up) and run profile
    t_start = time.time()
    rc = subprocess.call(
        [sys.executable, str(SKILL_RUN),
         "--config", args.config,
         "--workload", args.workload,
         "--out-dir", str(out_dir),
         "--profile-num-steps", str(args.profile_num_steps),
         "--warmup-requests", str(args.warmup_requests),
         "--server-start-timeout", str(args.server_start_timeout)],
        cwd=str(PROJECT_ROOT),
    )
    t_end = time.time()
    print(f"[hw-view] run_profile.py rc={rc} duration={t_end - t_start:.1f}s")

    # 3. stop sampler
    try:
        sampler.send_signal(signal.SIGTERM)
        sampler.wait(timeout=10)
    except Exception:  # noqa: BLE001
        sampler.kill()

    # 4. roll up
    profile_summary = {}
    p_json = out_dir / "profile_summary.json"
    if p_json.exists():
        try:
            profile_summary = json.loads(p_json.read_text())
        except Exception as e:  # noqa: BLE001
            profile_summary = {"_load_error": str(e)}

    server_info = {}
    s_json = out_dir / "server_info.json"
    if s_json.exists():
        try:
            server_info = json.loads(s_json.read_text())
        except Exception as e:  # noqa: BLE001
            server_info = {"_load_error": str(e)}

    gpu_stats = reduce_gpu_samples(sampler_csv, bench_window=(t_start, t_end))

    # Top kernels — keep top 10 trimmed
    top_kernels = (profile_summary.get("top_kernels") or [])[:10]
    # Heuristic kernel classification
    def classify(name: str) -> str:
        n = name.lower()
        if "flashattn" in n or "flash_attn" in n:
            return "attention (FlashAttention)"
        if "fmha" in n or "attention" in n:
            return "attention (other)"
        if "gemm" in n or "cublas" in n or "nvjet" in n or "cutlass::gemm" in n:
            return "GEMM"
        if "moe" in n or "grouped_gemm" in n:
            return "MoE"
        if "all_to_all" in n or "all2all" in n or "nccl" in n:
            return "collective"
        if "rmsnorm" in n or "layernorm" in n:
            return "norm"
        if "rotary" in n or "rope" in n:
            return "RoPE"
        if "topk" in n or "softmax" in n:
            return "topk/softmax"
        if "cudalaunch" in n or "cudagraphlaunch" in n or "cudamemcpyasync" in n or "cudadevicesynchronize" in n:
            return "cuda runtime/overhead"
        if "vectorized_elementwise" in n or "elementwise" in n:
            return "elementwise"
        return "other"
    kernel_class = {}
    for k in top_kernels:
        c = classify(k.get("name", ""))
        kernel_class.setdefault(c, 0.0)
        kernel_class[c] += parse_float_safe(k.get("self_time_pct")) or 0.0
    kernel_class = {k: round(v, 1) for k, v in sorted(
        kernel_class.items(), key=lambda kv: -kv[1])}

    # Pull useful internal_states out of server_info if present
    si = {}
    internal = server_info.get("internal_states") or [{}]
    if internal:
        s0 = internal[0]
        for k in [
            "load", "max_total_num_tokens", "context_len",
            "pp_size", "tp_size", "schedule_policy",
            "weight_load_method", "kv_cache_dtype",
            "max_running_requests", "max_queued_requests",
            "decode_throughput", "prefill_throughput",
            "cache_hit_rate",
        ]:
            if k in s0:
                si[k] = s0[k]
    # Top-level ServerArgs fields useful for backend
    for k in ["model_path", "attention_backend", "sampling_backend",
              "tokenizer_path", "torch_compile_max_bs"]:
        if k in server_info:
            si[k] = server_info[k]

    hw_view = {
        "schema_version": 1,
        "config": str(args.config),
        "workload": str(args.workload),
        "wall_window_s": round(t_end - t_start, 1),
        "gpu_id": args.gpu,
        "gpu_sampling": gpu_stats,
        "server_info_excerpt": si,
        "profile_totals": profile_summary.get("totals"),
        "profile_phase_breakdown_pct": profile_summary.get("phase_breakdown_pct"),
        "profile_moe_overhead": profile_summary.get("moe_overhead"),
        "profile_cuda_graph": profile_summary.get("cuda_graph"),
        "top_kernels": [
            {
                "rank": k.get("rank"),
                "self_time_pct": k.get("self_time_pct"),
                "calls": k.get("calls"),
                "avg_us": k.get("avg_us"),
                "phase": k.get("phase"),
                "name_short": (k.get("name", "")[:120] +
                               ("..." if len(k.get("name", "")) > 120 else "")),
                "category": classify(k.get("name", "")),
            }
            for k in top_kernels
        ],
        "kernel_category_pct": kernel_class,
        "profile_warnings": profile_summary.get("warnings"),
        "profile_ok": profile_summary.get("ok"),
        "rc": rc,
    }
    (out_dir / "hardware_view.json").write_text(json.dumps(hw_view, indent=2))
    print(f"[hw-view] wrote {out_dir / 'hardware_view.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
