#!/usr/bin/env python3
"""Convert one regime's ncu_raw.csv (from bench_ncu_one_regime.sh) into the
ncu_summary.json shape expected by profile-summary-unified.

This is the bridge between our sglang.bench_one_batch + sudo ncu pipeline and
the existing ncu-microarch skill output contract.
"""
import csv
import io
import json
import sys
from pathlib import Path
from collections import defaultdict

# Same metric → short-name mapping used in ncu-microarch
METRIC_KEY = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":            "sm_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":          "dram_throughput_pct",
    "smsp__warps_active.avg.pct_of_peak_sustained_active":         "warps_active_pct",
    "l1tex__t_sector_hit_rate.pct":                                "l1_hit_pct",
    "lts__t_sector_hit_rate.pct":                                  "l2_hit_pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor_pipe_active_pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio":    "stall_long_scoreboard_avg",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": "stall_math_pipe_throttle_avg",
}


def parse_csv(path: Path):
    text = path.read_text()
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID","Process ID"')), None)
    if start is None:
        return {}
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    out = defaultdict(lambda: defaultdict(list))
    for row in reader:
        kname = (row.get("Kernel Name") or "").strip()
        m = (row.get("Metric Name") or "").strip()
        v = (row.get("Metric Value") or "").strip().replace(",", "")
        if not kname or not m: continue
        try: val = float(v)
        except ValueError: continue
        short = METRIC_KEY.get(m)
        if short is None: continue
        out[kname][short].append(val)
    return out


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
        print("usage: ncu_csv_to_summary.py <regime_dir> <out_summary.json>", file=sys.stderr)
        sys.exit(1)
    regime_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    csv_path = regime_dir / "ncu_raw.csv"
    if not csv_path.exists() or csv_path.stat().st_size < 1000:
        out.write_text(json.dumps({"schema_version": 0, "ok": False,
                                    "error": f"ncu_raw.csv missing or empty at {csv_path}"}, indent=2))
        sys.exit(1)

    parsed = parse_csv(csv_path)
    if not parsed:
        out.write_text(json.dumps({"schema_version": 0, "ok": False,
                                    "error": "no kernels parsed from ncu csv"}, indent=2))
        sys.exit(1)

    kernels = []
    for kname, mvals in parsed.items():
        samples = max(len(v) for v in mvals.values()) if mvals else 0
        agg = {k: (sum(v) / len(v)) for k, v in mvals.items()}
        v, notes = verdict_for(agg)
        headroom = round(100 - max(agg.get("sm_throughput_pct") or 0,
                                   agg.get("dram_throughput_pct") or 0), 1)
        kernels.append({
            "kernel": kname,
            "samples": samples,
            "metrics": {k: round(val, 3) for k, val in agg.items()},
            "verdict": v,
            "headroom_estimate_pct": headroom,
            "notes": notes,
        })
    kernels.sort(key=lambda k: -k["headroom_estimate_pct"])

    payload = {
        "schema_version": 0,
        "ok": True,
        "ncu_raw_csv": str(csv_path),
        "regime_dir": str(regime_dir),
        "kernel_count_profiled": len(kernels),
        "kernels": kernels,
        "warnings": [],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} — {len(kernels)} kernels")


if __name__ == "__main__":
    main()
