#!/usr/bin/env python3
"""Generate human-readable markdown NCU report per regime from ncu_summary.json."""
import json
from pathlib import Path

BASE = Path("/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-06-09_sglang_triton_sweep")
REGIMES = ["R_short_decode", "R_medium_balanced", "R_long_prefill", "R_concurrent_decode"]

REGIME_META = {
    "R_short_decode":      "batch=1, in=100w, out=256 — decode (worst expert utilization)",
    "R_medium_balanced":   "batch=8, in=800w, out=256 — decode (typical workload)",
    "R_long_prefill":      "batch=4, in=4000w, out=32 — prefill-dominated",
    "R_concurrent_decode": "batch=32, in=200w, out=256 — high-concurrency decode",
}


def short(name, w):
    if len(name) <= w:
        return name
    return name[:w-3] + "..."


for rid in REGIMES:
    p = BASE / "ncu" / rid / "ncu_summary.json"
    d = json.loads(p.read_text())
    out_md = BASE / "ncu" / rid / "ncu_report.md"

    lines = [
        f"# NCU report — {rid}",
        "",
        f"**Workload**: {REGIME_META[rid]}",
        f"**Kernels profiled**: {d['kernel_count_profiled']} (NCU `--set full`, no kernel filter)",
        f"**Source**: `ncu_summary.json` (parsed from `ncu_raw_full.csv` via `scripts/ncu_csv_wide_to_summary.py`)",
        f"**Raw .ncu-rep**: `{rid}_ncu.ncu-rep` (open with `ncu-ui` for Nsight Compute GUI)",
        "",
        "## How to read",
        "",
        "- **SM %**: SM throughput as % of peak sustained — high = compute pipeline busy",
        "- **DRAM %**: HBM bandwidth as % of peak — high (≥70%) = memory-bound",
        "- **Occupancy %** (warps active): how full the warp slots are — low (<30%) = grid/block too small",
        "- **TC %** (tensor pipe active): how busy the Tensor Cores are — high on GEMM/MM is good",
        "- **L1/L2 hit %**: cache hit rates",
        "- **Long SB stall**: warps waiting on memory loads — high (>2) = memory-bound signal",
        "- **Math throttle**: warps waiting on math pipe — high (>1.5) = TC saturated",
        "- **Headroom %**: 100 - max(SM%, DRAM%); rough upper bound on improvement",
        "- **Verdict**: derived from rules in `.github/skills/ncu-microarch/SKILL.md`",
        "",
        "## Kernels ranked by headroom (highest first = most potential for optimization)",
        "",
        "| # | Kernel | Verdict | SM% | DRAM% | Occ% | TC% | L1% | L2% | Long-SB | Math-Th | Headroom% |",
        "|---|--------|---------|-----|-------|------|-----|-----|-----|---------|---------|-----------|",
    ]
    for i, k in enumerate(d["kernels"], 1):
        m = k["metrics"]
        lines.append(
            f"| {i} | `{short(k['kernel'], 60)}` | {k['verdict']} "
            f"| {m.get('sm_throughput_pct', 0):.1f} "
            f"| {m.get('dram_throughput_pct', 0):.1f} "
            f"| {m.get('warps_active_pct', 0):.1f} "
            f"| {m.get('tensor_pipe_active_pct', 0):.1f} "
            f"| {m.get('l1_hit_pct', 0):.1f} "
            f"| {m.get('l2_hit_pct', 0):.1f} "
            f"| {m.get('stall_long_scoreboard_avg', 0):.2f} "
            f"| {m.get('stall_math_pipe_throttle_avg', 0):.2f} "
            f"| {k['headroom_estimate_pct']:.1f} |"
        )

    # Notes section
    lines += ["", "## Per-kernel notes (auto-derived)", ""]
    for i, k in enumerate(d["kernels"], 1):
        if k.get("notes"):
            lines.append(f"**{i}. `{short(k['kernel'], 80)}`** ({k['verdict']})")
            for n in k["notes"]:
                lines.append(f"  - {n}")
            lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")
