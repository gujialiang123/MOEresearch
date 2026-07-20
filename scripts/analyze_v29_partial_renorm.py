#!/usr/bin/env python3
"""Analyze v29 partial-renorm dose: per-config summary + paired bootstrap CI vs
k8_native + beta monotonic trend test (Spearman-like sign + isotonic check)."""
import os, sys, json, glob, argparse, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import stats as ST

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/home/t-jialianggu/work/MOEresearch/results/2026-07-20_v29_partial_renorm")
args = ap.parse_args()

configs = {}
for f in sorted(glob.glob(os.path.join(args.dir, "*_raw.jsonl"))):
    nm = os.path.basename(f).replace("_raw.jsonl", "")
    configs[nm] = [json.loads(l) for l in open(f)]

base = {r["id"]: r for r in configs.get("k8_native", [])}
print(f"dir: {args.dir}\nbaseline: k8_native (n={len(base)})\n")
print(f"{'config':12s}{'dk':>4}{'beta':>6}{'len':>7}{'Δvs_k8(95%CI)':>22}{'acc':>7}{'noMark':>8}{'hitmax':>8}")

def kof(nm): 
    import re
    m = re.search(r'k(\d+)', nm); return int(m.group(1)) if m else 8
def bof(nm):
    import re
    m = re.search(r'b([\d.]+)', nm); return float(m.group(1)) if m else 1.0

rows_out = {}
for nm, rows in sorted(configs.items(), key=lambda kv: (kof(kv[0]), bof(kv[0]))):
    n = len(rows); lens = [r["gen_len"] for r in rows]
    cur = {r["id"]: r for r in rows}
    ids = [i for i in base if i in cur]
    deltas = [cur[i]["gen_len"] - base[i]["gen_len"] for i in ids]
    dm, lo, hi = ST.paired_bootstrap_ci(deltas) if deltas else (0, 0, 0)
    acc = sum(r["correct_strict"] for r in rows)/n
    nomark = sum(1 for r in rows if r["hash_tok_pos"] < 0)/n
    hitmax = sum(1 for r in rows if r["hit_max"])/n
    rows_out[nm] = {"decode_k": kof(nm), "beta": bof(nm), "n": n, "len_mean": round(statistics.mean(lens),1),
                    "delta_len": dm, "ci": [lo, hi], "acc_strict": round(acc,4),
                    "no_marker": round(nomark,4), "hit_max": round(hitmax,4)}
    print(f"{nm:12s}{kof(nm):>4}{bof(nm):>6.2f}{statistics.mean(lens):>7.0f}{f'{dm} ({lo},{hi})':>22}{acc*100:>6.1f}%{nomark*100:>7.1f}%{hitmax*100:>7.1f}%")

# monotonic trend test per decode_k: is len_mean monotone increasing in beta?
print("\n== beta monotonic trend (per decode_k) ==")
trend = {}
for dk in sorted(set(r["decode_k"] for r in rows_out.values())):
    if dk == 8: continue
    pts = sorted([(v["beta"], v["len_mean"]) for k, v in rows_out.items() if v["decode_k"] == dk])
    lens = [p[1] for p in pts]
    inc = all(lens[i] <= lens[i+1] + 1e-9 for i in range(len(lens)-1))
    # count concordant adjacent pairs
    conc = sum(1 for i in range(len(lens)-1) if lens[i] <= lens[i+1])
    trend[dk] = {"betas": [p[0] for p in pts], "lens": lens, "strictly_monotone_inc": inc,
                 "concordant_pairs": f"{conc}/{len(lens)-1}"}
    print(f"decode_k={dk}: betas={[p[0] for p in pts]} lens={lens} monotone_inc={inc} ({conc}/{len(lens)-1} concordant)")

json.dump({"dir": args.dir, "configs": rows_out, "beta_trend": trend},
          open(os.path.join(args.dir, "analysis.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(args.dir, 'analysis.json')}")
