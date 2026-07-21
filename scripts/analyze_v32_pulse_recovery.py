#!/usr/bin/env python3
"""Analyze v32 pulse-recovery from raw.jsonl (works on partial/in-progress runs).
Aggregates Part A (prefill recovery) and Part B (decode pulse) into summary_from_raw.json.

Run: python scripts/analyze_v32_pulse_recovery.py <result_dir>
"""
import os, sys, json, argparse, statistics
ap = argparse.ArgumentParser()
ap.add_argument("result_dir")
args = ap.parse_args()

A, B = [], []
for l in open(os.path.join(args.result_dir, "raw.jsonl")):
    r = json.loads(l)
    (A if r["part"] == "A" else B).append(r)


def _m(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 5) if xs else None


partA = {
    "n": len(A),
    "open_kl_first16_mean": _m([a["kl_first16"] for a in A]),
    "open_kl_last16_mean": _m([a["kl_last16"] for a in A]),
    "half_life_median": statistics.median([a["half_life"] for a in A if a.get("half_life") is not None]) if any(a.get("half_life") is not None for a in A) else None,
    "mean_base_token_dlogp": _m([a.get("mean_base_token_dlogp") for a in A]),
    "closed_first_div_median": statistics.median([a["closed_first_div"] for a in A]) if A else None,
    "closed_no_divergence_frac": round(sum(1 for a in A if a["closed_first_div"] < 0)/len(A), 3) if A else None,
    "closed_len_delta_mean": _m([a.get("len_delta") for a in A]),
}

durs = sorted({b["dur"] for b in B})
starts = sorted({b["start"] for b in B})
def bagg(rows):
    if not rows:
        return None
    flips = [b for b in rows if b["closed_first_flip"] is not None and b["closed_first_flip"] >= 0]
    return {"n": len(rows),
            "open_kl_in_mean": _m([b["open_kl_in"] for b in rows]),
            "open_kl_recovery_first8": _m([b["open_kl_recovery_first8"] for b in rows]),
            "open_kl_recovery_last8": _m([b["open_kl_recovery_last8"] for b in rows]),
            "closed_flip_frac": round(len(flips)/len(rows), 3),
            "closed_len_delta_mean": _m([b["closed_len_delta"] for b in rows]),
            "closed_reconverged_frac": round(sum(1 for b in rows if b["closed_reconverged"])/len(rows), 3)}
partB = {"by_duration": {d: bagg([b for b in B if b["dur"] == d]) for d in durs},
         "by_start": {s: bagg([b for b in B if b["start"] == s]) for s in starts}}

out = {"partA": partA, "partB": partB, "n_prompts_partA": len(A), "n_partB_records": len(B)}
json.dump(out, open(os.path.join(args.result_dir, "summary_from_raw.json"), "w"), indent=2)

print(f"== v32 (partial ok) partA n={partA['n']} ==")
print(f"open KL first16={partA['open_kl_first16_mean']} -> last16={partA['open_kl_last16_mean']} "
      f"half_life_med={partA['half_life_median']}")
print(f"closed first-div median={partA['closed_first_div_median']} len_delta={partA['closed_len_delta_mean']}")
print("== partB by duration (open recovery vs closed flip) ==")
for d in durs:
    b = partB["by_duration"][d]
    if b:
        print(f"dur={d:2d}: open_kl_in={b['open_kl_in_mean']} recov8={b['open_kl_recovery_first8']} "
              f"flip={b['closed_flip_frac']} len_delta={b['closed_len_delta_mean']} reconv={b['closed_reconverged_frac']}")
print(f"wrote {args.result_dir}/summary_from_raw.json")
