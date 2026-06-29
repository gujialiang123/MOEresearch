#!/usr/bin/env python3
"""Analyze nsys profile of universal-config sglang run.

For the meeting-targeted analysis, we want:
1. Top kernels by total time (categorized: moe_gemm / dense_gemm / attention / norm / routing / other)
2. Kernel time vs HBM-bandwidth roofline estimate
3. Compare to 6/9 historical baseline kernel breakdown
4. Identify the % of time spent in each category

Inputs:
  --sqlite path to nsys profile sqlite
  --regime regime name (R_concurrent_decode etc.)
  --out-dir where to write analysis JSON + markdown

Outputs:
  <out_dir>/kernel_breakdown.json
  <out_dir>/analysis_report.md

Note: nsys gives us kernel time + grid/block/regs, NOT TC% or DRAM% (those need NCU).
So we ESTIMATE memory-boundness from kernel name + time + grid pattern.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

# Kernel categorization heuristics (based on 6/9 known kernel names)
def categorize(short_name: str, demangled: str) -> str:
    s = short_name.lower() if short_name else ""
    d = demangled.lower() if demangled else ""

    # MoE GEMM
    if "fused_moe_kernel" in d or "fused_moe" in s:
        return "moe_gemm_triton"
    if "moe_gemm" in d or "moe_kernel" in d:
        return "moe_gemm_other"
    if ("cutlass" in d or "cutlass" in s) and ("moe" in d or "fused" in d):
        return "moe_gemm_cutlass"
    if "cutlass" in d and ("group" in d or "grouped" in d):
        return "moe_gemm_cutlass"

    # Generic CUTLASS / cuBLAS GEMM (dense)
    if "nvjet" in d or "nvjet" in s:
        return "dense_gemm_cublas"
    if "device_kernel" in d and "cutlass" in d and "gemm" in d:
        return "dense_gemm_cutlass"
    if "device_kernel" in s and ("gemm" in d):
        return "dense_gemm_cutlass"
    if "gemm" in s and "moe" not in s:
        return "dense_gemm_other"

    # Attention
    if "flash" in d or "flashattn" in d or "fmha" in d:
        return "attention_flash"
    if "attention" in s or "attn" in s:
        return "attention_other"
    if "splitkv" in d.lower() or "splitkreduce" in d.lower():
        return "attention_other"

    # Norm / elementwise
    if "norm" in s or "rmsnorm" in s or "layer_norm" in d.lower():
        return "norm"
    if "elementwise" in s or "activation" in s or "silu" in d.lower() or "swiglu" in d.lower():
        return "elementwise"

    # MoE routing
    if "topk" in s or "top_k" in s:
        return "moe_routing"
    if "count_and_sort" in d.lower() or "sort_expert" in d.lower() or "permute" in d.lower():
        return "moe_routing"
    if "scatter" in s or "gather" in s:
        return "moe_routing"

    # Sampling / decode helpers
    if "sample" in s or "logits" in s or "softmax" in s:
        return "sampling"
    if "clamp" in s.lower() or "argmax" in s.lower():
        return "sampling"

    # Memory ops
    if "memcpy" in s or "memset" in s:
        return "memory_op"

    # Reduce / combine
    if "reduce" in s or "combine" in s:
        return "reduce"

    # Catch-all
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True, help="Path to nsys sqlite")
    ap.add_argument("--regime", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--window-trim-pct", type=float, default=20.0,
                    help="Trim first X%% of time (warmup) from analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Get window: trim first X% (warmup) and last 5% (shutdown noise)
    row = conn.execute(
        "SELECT MIN(start) AS t0, MAX(end) AS tN FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    t0, tN = row["t0"], row["tN"]
    duration_ns = tN - t0
    win_start = t0 + int(duration_ns * args.window_trim_pct / 100)
    win_end = tN - int(duration_ns * 0.05)

    print(f"trace window: full = {duration_ns/1e9:.1f}s")
    print(f"analysis window: {(win_start-t0)/1e9:.1f}s to {(win_end-t0)/1e9:.1f}s "
          f"({(win_end-win_start)/1e9:.1f}s)")

    # Top kernels in analysis window
    sql = """
    SELECT
        s.value AS short_name,
        MAX(ds.value) AS demangled_sample,
        SUM(k.end - k.start) AS self_ns,
        COUNT(*)             AS calls,
        AVG(k.end - k.start) AS avg_ns,
        MAX(k.registersPerThread) AS max_reg,
        MAX(k.gridX) AS max_grid_x,
        MAX(k.blockX) AS max_block_x
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    LEFT JOIN StringIds ds ON k.demangledName = ds.id
    WHERE k.start >= :w0 AND k.end <= :w1
    GROUP BY s.value
    ORDER BY self_ns DESC
    LIMIT :n;
    """
    rows = list(conn.execute(sql, {"w0": win_start, "w1": win_end, "n": args.top_n}))
    total_kernel_ns = sum(r["self_ns"] for r in rows)
    print(f"\ntop {args.top_n} kernels: total {total_kernel_ns/1e9:.3f}s")

    kernels = []
    for r in rows:
        category = categorize(r["short_name"] or "", r["demangled_sample"] or "")
        kernels.append({
            "short_name": r["short_name"],
            "demangled_sample": r["demangled_sample"],
            "category": category,
            "self_ns": r["self_ns"],
            "self_pct": r["self_ns"] / total_kernel_ns * 100,
            "calls": r["calls"],
            "avg_us": r["avg_ns"] / 1000,
            "max_reg": r["max_reg"],
            "max_grid_x": r["max_grid_x"],
            "max_block_x": r["max_block_x"],
        })

    # Aggregate by category
    cat_totals = {}
    for k in kernels:
        c = k["category"]
        cat_totals.setdefault(c, {"self_ns": 0, "calls": 0, "kernel_names": []})
        cat_totals[c]["self_ns"] += k["self_ns"]
        cat_totals[c]["calls"] += k["calls"]
        cat_totals[c]["kernel_names"].append(k["short_name"])
    for c in cat_totals:
        cat_totals[c]["self_pct"] = cat_totals[c]["self_ns"] / total_kernel_ns * 100

    # Get all kernels in window (for totals)
    total_window_kernel_ns = conn.execute(
        "SELECT SUM(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "WHERE k.start >= :w0 AND k.end <= :w1",
        {"w0": win_start, "w1": win_end}
    ).fetchone()[0]

    # Stream breakdown
    stream_rows = list(conn.execute(
        "SELECT streamId, SUM(end - start) AS self_ns, COUNT(*) AS calls "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start >= :w0 AND end <= :w1 GROUP BY streamId",
        {"w0": win_start, "w1": win_end}
    ))

    # Wall time vs kernel-busy time (rough GPU utilization signal)
    window_wall_ns = win_end - win_start
    kernel_busy_pct = total_window_kernel_ns / window_wall_ns * 100 if window_wall_ns > 0 else 0

    out = {
        "regime": args.regime,
        "sqlite": str(args.sqlite),
        "config": "universal (Optuna best for R_concurrent_decode)",
        "config_flags": {
            "moe-runner-backend": "flashinfer_cutlass",
            "disable-cuda-graph": False,
            "max-running-requests": 32,
            "chunked-prefill-size": -1,
            "schedule-policy": "fcfs",
        },
        "trace": {
            "duration_total_s": duration_ns / 1e9,
            "analysis_window_s": (win_end - win_start) / 1e9,
            "trim_pct_start": args.window_trim_pct,
            "trim_pct_end": 5.0,
        },
        "gpu_utilization_signal": {
            "kernel_busy_ns_in_window": total_window_kernel_ns,
            "wall_ns_in_window": window_wall_ns,
            "kernel_busy_pct": kernel_busy_pct,
            "note": "kernel_busy_pct counts cross-stream activity, so it can exceed 100% on multi-stream workloads. Use as relative signal across streams.",
        },
        "streams": [
            {"stream_id": r["streamId"], "self_ns": r["self_ns"],
             "calls": r["calls"], "self_pct": r["self_ns"]/total_window_kernel_ns*100}
            for r in stream_rows
        ],
        "top_kernels": kernels,
        "by_category": cat_totals,
    }

    json_path = out_dir / "kernel_breakdown.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {json_path}")

    # Print human summary
    print(f"\n=== Category breakdown for {args.regime} ===")
    sorted_cats = sorted(cat_totals.items(), key=lambda x: -x[1]["self_ns"])
    for cat, info in sorted_cats:
        print(f"  {cat:<22} {info['self_pct']:>6.2f}%  ({info['self_ns']/1e9:.3f}s, {info['calls']} calls)")

    print(f"\n=== Top 15 kernels ===")
    for k in kernels[:15]:
        name = (k["short_name"] or "")[:50]
        cat = k["category"]
        print(f"  {k['self_pct']:>6.2f}% [{cat:<22}] {name:<50} ({k['calls']} calls, avg {k['avg_us']:.1f}us)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
