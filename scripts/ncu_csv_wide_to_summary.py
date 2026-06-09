#!/usr/bin/env python3
"""Convert NCU 'wide' CSV (one row per kernel, many metric columns) to
ncu_summary.json shape. Used when CSV is from `ncu --import ... --csv --page raw`
with --set full (typical of long-form profiling)."""
import csv
import io
import json
import sys
from pathlib import Path

# Map ncu's full-set metric names (wide CSV) → short keys
METRIC_KEY = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":              "sm_throughput_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":        "dram_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active":             "warps_active_pct",
    "l1tex__t_sector_hit_rate.pct":                                  "l1_hit_pct",
    "lts__t_sector_hit_rate.pct":                                    "l2_hit_pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active":"tensor_pipe_active_pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio":    "stall_long_scoreboard_avg",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": "stall_math_pipe_throttle_avg",
}


def parse_wide(csv_path: Path):
    text = csv_path.read_text()
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID","Process ID"')), None)
    if start is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    # Build mapping: full metric name → short key (case-sensitive exact match on column name)
    fieldnames = reader.fieldnames or []
    col_map = {}
    for col in fieldnames:
        # col may have a prefix like "FE_B.TriageCompute."; strip to bare metric name
        # The bare name is the last dot-segment that matches our keys.
        # Try exact match first
        if col in METRIC_KEY:
            col_map[col] = METRIC_KEY[col]

    if not col_map:
        # debug
        print(f"WARN: no metric columns matched; available: {[c for c in fieldnames if c.startswith('sm__') or c.startswith('gpu__') or c.startswith('smsp__')][:10]}", file=sys.stderr)
        return []

    kernels = []
    for row in reader:
        kname = (row.get("Kernel Name") or "").strip()
        if not kname: continue
        metrics = {}
        for col, short in col_map.items():
            raw = (row.get(col) or "").strip().replace(",", "")
            if raw in ("", "-"): continue
            try:
                metrics[short] = float(raw)
            except ValueError:
                pass
        kernels.append({"kernel": kname, "metrics": metrics})
    return kernels


def verdict_for(m):
    sm = m.get("sm_throughput_pct") or 0
    dram = m.get("dram_throughput_pct") or 0
    occ = m.get("warps_active_pct") or 0
    tc = m.get("tensor_pipe_active_pct") or 0
    notes = []
    if occ < 30:
        v = "low_occupancy"
    elif tc < 10 and sm > 30:
        v = "tensor_core_idle"
        notes.append("Tensor Cores firing <10% — likely wrong dtype or non-TC kernel")
    elif dram >= 70:
        v = "memory_bound"
        notes.append("DRAM ≥70% — algorithmic reuse beats tile tuning")
    elif sm >= 70 and dram < 50:
        v = "compute_bound"
        notes.append(f"Tensor Cores firing at {tc:.1f}%")
    elif sm < 30 and dram < 30:
        v = "latency_bound"
        notes.append("Both SM and DRAM under 30% — launch overhead or warp stalls dominate")
    else:
        v = "balanced"
    sb = m.get("stall_long_scoreboard_avg") or 0
    mp = m.get("stall_math_pipe_throttle_avg") or 0
    if sb > 2.0:
        notes.append(f"long_scoreboard stall = {sb:.2f} warps/issue — severe memory-wait")
    if mp > 1.5:
        notes.append(f"math_pipe_throttle stall = {mp:.2f} — tensor cores saturated")
    return v, notes


def main():
    if len(sys.argv) != 3:
        print("usage: ncu_csv_wide_to_summary.py <ncu_raw.csv> <out_summary.json>", file=sys.stderr)
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if not csv_path.exists() or csv_path.stat().st_size < 1000:
        out.write_text(json.dumps({"schema_version": 0, "ok": False,
                                    "error": f"missing or empty: {csv_path}"}, indent=2))
        sys.exit(1)

    parsed = parse_wide(csv_path)
    if not parsed:
        out.write_text(json.dumps({"schema_version": 0, "ok": False,
                                    "error": "no kernels parsed from wide ncu csv"}, indent=2))
        sys.exit(1)

    kernels = []
    for entry in parsed:
        m = entry["metrics"]
        v, notes = verdict_for(m)
        headroom = round(100 - max(m.get("sm_throughput_pct") or 0,
                                   m.get("dram_throughput_pct") or 0), 1)
        kernels.append({
            "kernel": entry["kernel"],
            "samples": 1,
            "metrics": {k: round(val, 3) for k, val in m.items()},
            "verdict": v,
            "headroom_estimate_pct": headroom,
            "notes": notes,
        })
    kernels.sort(key=lambda k: -k["headroom_estimate_pct"])

    out.write_text(json.dumps({
        "schema_version": 0,
        "ok": True,
        "ncu_raw_csv": str(csv_path),
        "kernel_count_profiled": len(kernels),
        "metrics_in_csv": list(METRIC_KEY.values()),
        "kernels": kernels,
        "warnings": [],
    }, indent=2))
    print(f"wrote {out} — {len(kernels)} kernels")


if __name__ == "__main__":
    main()
