#!/usr/bin/env python3
"""Parse NCU long-format ncu_raw.csv (Metric Name/Value) -> accurate achieved GFLOP/s.

Tensor-core FLOP = sum of hgmma/hmma op counts (already FLOP, .sum is cumulative
across the kernel). Non-tensor FP FLOP = fadd + fmul + 2*ffma (fp32) + hadd + hmul
+ 2*hfma (fp16), summed over threads. achieved FLOP/s = total_FLOP / duration.
"""
import csv, sys, json
from collections import defaultdict
from pathlib import Path

PEAK = 989.5e12
TENSOR = [
    "sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum",
    "sm__ops_path_tensor_op_hgmma_src_fp16_sparsity_off.sum",
    "sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32_sparsity_off.sum",
]


def fval(s):
    s = (s or "").strip().replace(",", "")
    if s in ("", "n/a", "N/A", "inf"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def short(kn):
    l = kn.lower()
    if "fused_moe" in l:
        return "fused_moe"
    if "flash" in l:
        return "flash_attn"
    if "nvjet" in l:
        return "gemm(dense)"
    if "rmsnorm" in l or "norm" in l:
        return "rmsnorm"
    if "rotary" in l:
        return "rotary"
    if "topkgating" in l or "topk" in l:
        return "router_topk"
    if "act_and_mul" in l or "activation" in l:
        return "activation"
    return kn.split("(")[0].split("<")[0][:18]


def parse(path):
    # each row: ID,PID,PName,Host,Kernel Name,Ctx,Stream,Block,Grid,Device,CC,Section,Metric Name,Metric Unit,Metric Value
    per = defaultdict(lambda: defaultdict(float))  # (id,kernel) -> metric -> value
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if len(row) < 15:
                continue
            kid = (row[0], row[4])
            per[kid][row[12]] += fval(row[14])
    return per


def analyze(path, label):
    per = parse(path)
    agg = defaultdict(lambda: {"dur": 0.0, "flop": 0.0, "tensor": 0.0, "sm_w": 0.0})
    for (kid, kn), m in per.items():
        dur_us = m.get("gpu__time_duration.sum", 0.0)  # NCU exports duration in microseconds
        if dur_us <= 0:
            continue
        dur_ns = dur_us * 1000.0
        tensor = sum(m.get(c, 0.0) for c in TENSOR)
        nontensor = (m.get("sm__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0)
                     + m.get("sm__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0)
                     + 2 * m.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0.0)
                     + m.get("sm__sass_thread_inst_executed_op_hadd_pred_on.sum", 0.0)
                     + m.get("sm__sass_thread_inst_executed_op_hmul_pred_on.sum", 0.0)
                     + 2 * m.get("sm__sass_thread_inst_executed_op_hfma_pred_on.sum", 0.0))
        k = short(kn)
        a = agg[k]
        a["dur"] += dur_ns
        a["flop"] += tensor + nontensor
        a["tensor"] += tensor
        a["sm_w"] += dur_ns * m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)

    print(f"\n===== {label} (ACCURATE FLOP counters) =====")
    print(f"{'kernel':>13}{'dur_us':>10}{'SM%':>7}{'tensor_FLOP%':>13}{'TFLOP/s':>10}{'%peak':>8}")
    td = sum(a["dur"] for a in agg.values())
    tfl = sum(a["flop"] for a in agg.values())
    ks = []
    for k, a in sorted(agg.items(), key=lambda x: -x[1]["dur"]):
        dur_s = a["dur"] / 1e9
        tflops = (a["flop"] / dur_s) / 1e12 if dur_s else 0.0
        smp = a["sm_w"] / a["dur"] if a["dur"] else 0.0
        tenpct = a["tensor"] / a["flop"] * 100 if a["flop"] else 0.0
        print(f"{k:>13}{a['dur']/1000:>10.1f}{smp:>7.1f}{tenpct:>12.1f}%{tflops:>10.1f}{tflops/(PEAK/1e12)*100:>7.1f}%")
        ks.append({"kernel": k, "dur_us": round(a["dur"]/1000, 1), "sm_pct": round(smp, 1),
                   "achieved_tflops": round(tflops, 1), "pct_peak": round(tflops/(PEAK/1e12)*100, 1)})
    overall = (tfl / (td/1e9)) / 1e12 if td else 0.0
    print(f"  >> time-weighted overall: {overall:.1f} TFLOP/s ({overall/(PEAK/1e12)*100:.1f}% of bf16 peak)")
    return {"label": label, "overall_tflops": round(overall, 1),
            "pct_peak": round(overall/(PEAK/1e12)*100, 1), "kernels": ks}


if __name__ == "__main__":
    res = [analyze(p, p) for p in sys.argv[1:]]
    Path("results/2026-07-15_v18_gflops/gflops_accurate.json").write_text(
        json.dumps({"peak_tflops": 989.5, "results": res}, indent=2))
    print("\nwrote results/2026-07-15_v18_gflops/gflops_accurate.json")
