#!/usr/bin/env python3
"""v18: compute achieved GFLOP/s per kernel from NCU FLOP counters.

Reads the exported ncu_raw.csv (targeted --metrics run) for each model x decode
point, sums tensor-core ops (hgmma+hmma, bf16/fp16 sources) and non-tensor FP
(fadd+fmul+2*ffma, hadd+hmul+2*hfma), divides by kernel duration to get
achieved FLOP/s. Aggregates per hot-kernel (time-weighted) and overall.

H200 bf16 tensor-core peak = 989.5 TFLOP/s (dense).
"""
import csv, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "results" / "2026-07-15_v18_gflops"
PEAK_BF16_TC = 989.5e12  # H200 SXM bf16 dense tensor peak FLOP/s

TENSOR_COLS = [
    "sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum",
    "sm__ops_path_tensor_op_hgmma_src_fp16_sparsity_off.sum",
    "sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32_sparsity_off.sum",
]
DUR = "gpu__time_duration.sum"        # nanoseconds
SMPCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"


def fval(s):
    if s is None:
        return 0.0
    s = s.strip().replace(",", "")
    if s in ("", "n/a", "N/A"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def kernel_short(kn):
    l = kn.lower()
    if "fused_moe" in l or ("moe" in l and "gemm" in l):
        return "fused_moe"
    if "flashattn" in l or "flash::" in l or "flashinfer" in l and "attn" in l:
        return "flash_attn"
    if "nvjet" in l or ("gemm" in l and "cutlass" in l):
        return "gemm(dense)"
    if "rmsnorm" in l or "norm" in l:
        return "rmsnorm"
    if "rotary" in l:
        return "rotary"
    if "topkgating" in l or "topk" in l:
        return "router_topk"
    if "act_and_mul" in l or "activation" in l:
        return "activation"
    if "conv1d" in l:
        return "conv1d"
    return kn.split("(")[0].split("<")[0][:22]


def analyze(csv_path, label):
    rows = [r for r in csv.DictReader(open(csv_path)) if r.get("Kernel Name", "").strip()]
    agg = {}  # short -> dict(dur, flop, sm_w)
    for r in rows:
        kn = r["Kernel Name"]
        dur_ns = fval(r.get(DUR))
        if dur_ns <= 0:
            continue
        tensor = sum(fval(r.get(c)) for c in TENSOR_COLS)
        fadd = fval(r.get("sm__sass_thread_inst_executed_op_fadd_pred_on.sum"))
        fmul = fval(r.get("sm__sass_thread_inst_executed_op_fmul_pred_on.sum"))
        ffma = fval(r.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum"))
        hadd = fval(r.get("sm__sass_thread_inst_executed_op_hadd_pred_on.sum"))
        hmul = fval(r.get("sm__sass_thread_inst_executed_op_hmul_pred_on.sum"))
        hfma = fval(r.get("sm__sass_thread_inst_executed_op_hfma_pred_on.sum"))
        nontensor = fadd + fmul + 2 * ffma + hadd + hmul + 2 * hfma
        flop = tensor + nontensor
        sm = fval(r.get(SMPCT))
        k = kernel_short(kn)
        a = agg.setdefault(k, {"dur": 0.0, "flop": 0.0, "sm_w": 0.0, "tensor": 0.0})
        a["dur"] += dur_ns
        a["flop"] += flop
        a["tensor"] += tensor
        a["sm_w"] += dur_ns * sm

    print(f"\n===== {label} =====")
    print(f"{'kernel':>14}{'dur_us':>11}{'SM%':>7}{'achieved_TFLOPs':>16}{'%peak_bf16':>12}")
    tot_dur = sum(a["dur"] for a in agg.values())
    tot_flop = sum(a["flop"] for a in agg.values())
    out_kernels = []
    for k, a in sorted(agg.items(), key=lambda x: -x[1]["dur"]):
        dur_s = a["dur"] / 1e9
        tflops = (a["flop"] / dur_s) / 1e12 if dur_s else 0.0
        smpct = a["sm_w"] / a["dur"] if a["dur"] else 0.0
        pk = tflops / (PEAK_BF16_TC / 1e12) * 100
        print(f"{k:>14}{a['dur']/1000:>11.1f}{smpct:>7.1f}{tflops:>16.1f}{pk:>11.1f}%")
        out_kernels.append({"kernel": k, "dur_us": round(a["dur"]/1000, 1),
                            "sm_pct": round(smpct, 1), "achieved_tflops": round(tflops, 1),
                            "pct_of_bf16_peak": round(pk, 1)})
    overall = (tot_flop / (tot_dur / 1e9)) / 1e12 if tot_dur else 0.0
    pk = overall / (PEAK_BF16_TC / 1e12) * 100
    print(f"  >> time-weighted overall achieved: {overall:.1f} TFLOP/s  ({pk:.1f}% of bf16 TC peak)")
    return {"label": label, "overall_tflops": round(overall, 1),
            "pct_of_bf16_peak": round(pk, 1), "kernels": out_kernels}


def main():
    results = []
    for model_dir in sorted(ROOT.glob("*")):
        if not model_dir.is_dir():
            continue
        for pt in sorted(model_dir.glob("agent_decode_*")):
            csvf = pt / "ncu_raw.csv"
            if csvf.exists() and csvf.stat().st_size > 0:
                results.append(analyze(csvf, f"{model_dir.name} / {pt.name}"))
    out = ROOT / "gflops_summary.json"
    json.dump({"peak_bf16_tc_tflops": PEAK_BF16_TC/1e12, "results": results},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
