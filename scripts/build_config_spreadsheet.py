#!/usr/bin/env python3
"""Build consolidated config spreadsheet across v2 and v3 experiments.

One CSV where each row = one server config (baseline OR trial).
Columns:
  - Meta: source, experiment, run_id, notes
  - 7 knobs: moe_runner_backend, attention_backend, disable_cuda_graph,
             max_running_requests, chunked_prefill_size, schedule_policy,
             mem_fraction_static
  - Per-regime tokens/s (v3 has 8, v2 has 4; empty cells for missing)
  - Per-regime req/s
  - Per-regime MFU_simple %
  - Per-regime speedup vs v3 cookbook baseline (as ratio, e.g. 1.36×)
  - Aggregate metrics: primary_speedup_R_conc, geomean_speedup_all
  - Ordering: descending by primary_speedup_R_conc (or geomean; both cols included)

Baseline used for speedup normalization: v3 cookbook-default single run
  (this has all 8 regimes; v2 3-run avg used where v3 not available).

Usage:
    python scripts/build_config_spreadsheet.py \\
        --out results/consolidated_config_spreadsheet.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Standard regime ordering (short to long)
REGIMES = [
    "R_short_decode",
    "R_medium_balanced",
    "R_concurrent_decode",
    "R_long_prefill",
    "R_prompt_8k_c4_out128",
    "R_prompt_16k_c2_out128",
    "R_prompt_32k_c1_out128",
    "R_prompt_50k_c1_out64",
]

KNOB_COLS = [
    "moe_runner_backend",
    "attention_backend",
    "disable_cuda_graph",
    "max_running_requests",
    "chunked_prefill_size",
    "schedule_policy",
    "mem_fraction_static",
]


def load_summary_knobs(summary_path: Path) -> dict:
    """Extract the 7 knobs from summary.spec_resolved.server_config."""
    s = json.loads(summary_path.read_text())
    if not s.get("ok"):
        return None
    sc = s.get("spec_resolved", {}).get("server_config", {})
    knobs = {
        "moe_runner_backend": sc.get("moe-runner-backend", "auto"),
        "attention_backend": sc.get("attention-backend", "auto"),
        "disable_cuda_graph": bool(sc.get("disable-cuda-graph", False)),
        "max_running_requests": sc.get("max-running-requests"),
        "chunked_prefill_size": sc.get("chunked-prefill-size"),
        "schedule_policy": sc.get("schedule-policy"),
        "mem_fraction_static": sc.get("mem-fraction-static"),
    }
    return {"knobs": knobs, "regimes": s.get("regimes", {}),
            "spec_hash": s.get("spec_hash", "")[:16]}


def collect_configs() -> list[dict]:
    """Walk the results/ tree; return a list of config records."""
    records = []

    # ---- v3 baselines ----
    for name, path in [
        ("v3-baseline-true-default",
         "results/2026-07-02_lfm2.5_v3/baseline-true-default/summary.json"),
        ("v3-baseline-cookbook",
         "results/2026-07-02_lfm2.5_v3/baseline-cookbook/summary.json"),
    ]:
        data = load_summary_knobs(REPO / path)
        if data:
            records.append({
                "source": "v3",
                "experiment": "baseline",
                "run_id": name,
                "notes": "v3 baseline; 1 lifetime x 3 runs",
                **data,
            })

    # ---- v3 Optuna trials (30) ----
    v3_dir = REPO / "results/2026-07-02_lfm2.5_v3/optuna-v3-R_concurrent_decode"
    for tp in sorted(v3_dir.glob("trial_*/summary.json")):
        trial_num = int(tp.parent.name.split("_")[1])
        phase = "warm-start" if trial_num < 4 else "tpe"
        data = load_summary_knobs(tp)
        if data:
            records.append({
                "source": "v3",
                "experiment": "optuna",
                "run_id": f"v3-trial-{trial_num:04d}",
                "notes": f"v3 Optuna {phase} trial",
                **data,
            })

    # ---- v2 baselines ----
    for name, path in [
        ("v2-baseline-true-default",
         "results/2026-06-30_lfm2.5/true-default/summary.json"),
        ("v2-baseline-cookbook",
         "results/2026-06-30_lfm2.5/cookbook-default/summary.json"),
    ]:
        data = load_summary_knobs(REPO / path)
        if data:
            records.append({
                "source": "v2",
                "experiment": "baseline",
                "run_id": name,
                "notes": "v2 baseline (only 4 regimes)",
                **data,
            })

    # v2 explicit 3-lifetime baseline reruns
    for i in [1, 2, 3]:
        path = REPO / f"results/2026-06-30_lfm2.5/baseline-revalidation/cookbook_run{i}/summary.json"
        data = load_summary_knobs(path)
        if data:
            records.append({
                "source": "v2",
                "experiment": "baseline-revalidation",
                "run_id": f"v2-cookbook-run{i}",
                "notes": f"v2 explicit 3-lifetime baseline; run {i}/3",
                **data,
            })

    # v2 triton MoE manual validation (the "key" trial we found manually)
    tv = REPO / "results/2026-06-30_lfm2.5/baseline-revalidation/triton_moe_explicit/summary.json"
    data = load_summary_knobs(tv)
    if data:
        records.append({
            "source": "v2",
            "experiment": "manual-validation",
            "run_id": "v2-triton-moe-manual",
            "notes": "v2 manual validation: triton MoE + baseline batching",
            **data,
        })

    # ---- v2 Optuna trials (main + archives) ----
    v2_dirs = [
        (REPO / "results/2026-06-30_lfm2.5/optuna-v2-R_concurrent_decode", "main"),
        (REPO / "results/2026-06-30_lfm2.5/optuna-v2-R_concurrent_decode/_phase1_archive", "phase1"),
        (REPO / "results/2026-06-30_lfm2.5/optuna-v2-R_concurrent_decode/_phase2a_archive", "phase2a"),
    ]
    for d, phase in v2_dirs:
        for tp in sorted(d.glob("trial_*/summary.json")):
            trial_num = int(tp.parent.name.split("_")[1])
            data = load_summary_knobs(tp)
            if data:
                records.append({
                    "source": "v2",
                    "experiment": f"optuna-{phase}",
                    "run_id": f"v2-{phase}-trial-{trial_num:04d}",
                    "notes": f"v2 Optuna {phase} trial",
                    **data,
                })

    return records


def find_speedup_baseline(records: list[dict]) -> dict:
    """Use v3-baseline-cookbook (single run, has all 8 regimes) as normalization."""
    for r in records:
        if r["run_id"] == "v3-baseline-cookbook":
            return r["regimes"]
    raise RuntimeError("v3-baseline-cookbook not found; can't compute speedups")


def build_row(r: dict, baseline_regimes: dict) -> dict:
    """Build one CSV row from a config record."""
    row = {
        "source": r["source"],
        "experiment": r["experiment"],
        "run_id": r["run_id"],
        "notes": r["notes"],
        "spec_hash": r["spec_hash"],
    }
    for k in KNOB_COLS:
        v = r["knobs"].get(k)
        if k == "disable_cuda_graph":
            row["cuda_graph_on"] = "on" if not v else "off"
        else:
            row[k] = "" if v is None else v

    # Per-regime numbers + speedup
    speedups = []
    for reg in REGIMES:
        entry = r["regimes"].get(reg, {})
        if entry:
            rps = entry.get("req_per_s", {}).get("mean")
            tps = entry.get("tokens_per_s", {}).get("mean")
            mfu_dict = entry.get("mfu", {})
            row[f"{reg}__req_per_s"] = round(rps, 4) if rps is not None else ""
            row[f"{reg}__tokens_per_s"] = round(tps, 1) if tps is not None else ""
            # ALL FOUR utilization metrics — different ones are meaningful
            # in different regimes:
            #   MFU_simple: decode matmul only (decode-heavy regimes)
            #   MFU_amortized: includes prefill FLOPs (long-prefill regimes)
            #   MBU: HBM bandwidth utilization (memory-bound decode)
            row[f"{reg}__MFU_simple_pct"] = round(mfu_dict.get("mfu_pct_simple", 0), 3)
            row[f"{reg}__MFU_amortized_pct"] = round(mfu_dict.get("mfu_pct_amortized", 0), 3)
            row[f"{reg}__MBU_pct"] = round(mfu_dict.get("mbu_pct", 0), 3)
            # Speedup vs baseline
            b_tps = baseline_regimes.get(reg, {}).get("tokens_per_s", {}).get("mean")
            if tps and b_tps and b_tps > 0:
                sp = tps / b_tps
                row[f"{reg}__speedup"] = round(sp, 3)
                speedups.append(sp)
            else:
                row[f"{reg}__speedup"] = ""
        else:
            for suffix in ("__req_per_s", "__tokens_per_s",
                           "__MFU_simple_pct", "__MFU_amortized_pct",
                           "__MBU_pct", "__speedup"):
                row[reg + suffix] = ""

    # Aggregate metrics
    b_conc_tps = baseline_regimes.get("R_concurrent_decode", {}).get("tokens_per_s", {}).get("mean")
    conc_tps = r["regimes"].get("R_concurrent_decode", {}).get("tokens_per_s", {}).get("mean")
    if conc_tps and b_conc_tps:
        row["speedup_R_conc"] = round(conc_tps / b_conc_tps, 3)
    else:
        row["speedup_R_conc"] = ""

    if speedups:
        # Geometric mean of per-regime speedups
        log_mean = sum(math.log(s) for s in speedups) / len(speedups)
        row["speedup_geomean_all_regimes"] = round(math.exp(log_mean), 3)
        row["n_regimes_measured"] = len(speedups)
    else:
        row["speedup_geomean_all_regimes"] = ""
        row["n_regimes_measured"] = 0

    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/consolidated_config_spreadsheet.csv")
    ap.add_argument("--sort-by", default="speedup_R_conc",
                    choices=["speedup_R_conc", "speedup_geomean_all_regimes"])
    args = ap.parse_args()

    records = collect_configs()
    baseline_regimes = find_speedup_baseline(records)
    print(f"Loaded {len(records)} config records. "
          f"Using v3-baseline-cookbook as speedup denominator.", flush=True)

    rows = [build_row(r, baseline_regimes) for r in records]

    # Determine column order
    fixed_cols = ["source", "experiment", "run_id", "notes"] + \
                 [c for c in KNOB_COLS if c != "disable_cuda_graph"] + \
                 ["cuda_graph_on"] + \
                 ["speedup_R_conc", "speedup_geomean_all_regimes", "n_regimes_measured"]
    per_regime_cols = []
    for reg in REGIMES:
        per_regime_cols.extend([
            f"{reg}__speedup",
            f"{reg}__req_per_s",
            f"{reg}__tokens_per_s",
            f"{reg}__MFU_simple_pct",
            f"{reg}__MFU_amortized_pct",
            f"{reg}__MBU_pct",
        ])
    cols = fixed_cols + per_regime_cols + ["spec_hash"]

    # Sort: descending by chosen speedup metric; empties go to bottom
    def sort_key(r):
        v = r.get(args.sort_by)
        return (0 if v == "" else 1, v if v != "" else 0)
    rows.sort(key=sort_key, reverse=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows × {len(cols)} cols → {out_path}", flush=True)
    print(f"Top 5 by {args.sort_by}:")
    for r in rows[:5]:
        print(f"  {r['run_id']:<32} {args.sort_by}={r.get(args.sort_by)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
