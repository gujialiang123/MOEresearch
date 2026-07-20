#!/usr/bin/env python3
"""Analyze config-based experiments (v23 factorial / v24 ablation / v28 dose).

Pure log analysis (no model). Computes per-config length/accuracy summaries,
paired bootstrap CIs vs baseline, McNemar on accuracy flips, phase-factor effects
(v23), and adjacent-K dose deltas (v28). Works from *_raw.jsonl in a results dir.

Run: python scripts/analyze_v23_v28.py --dir results/2026-07-20_v23_phase_factorial [--mode factorial|dose|ablation]
"""
import os, sys, json, glob, argparse, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import stats as ST

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--mode", default="auto", choices=["auto", "factorial", "dose", "ablation"])
ap.add_argument("--baseline", default="8x8")
args = ap.parse_args()


def load(path):
    return [json.loads(l) for l in open(path)]

configs = {}
for f in sorted(glob.glob(os.path.join(args.dir, "*_raw.jsonl"))):
    name = os.path.basename(f).replace("_raw.jsonl", "")
    configs[name] = load(f)

if not configs:
    print("no *_raw.jsonl found in", args.dir); sys.exit(1)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 2) if xs else None

def csummary(rows):
    n = len(rows)
    lens = [r["gen_len"] for r in rows]
    wh = [r for r in rows if r.get("hash_tok_pos", -1) >= 0]
    return {
        "n": n,
        "acc_strict": round(sum(r["correct_strict"] for r in rows)/n, 4),
        "acc_tol": round(sum(r["correct_tol"] for r in rows)/n, 4),
        "no_marker": round(sum(1 for r in rows if r.get("hash_tok_pos", -1) < 0)/n, 4),
        "no_eos": round(sum(1 for r in rows if r["eos_pos"] < 0)/n, 4),
        "hit_max": round(sum(1 for r in rows if r["hit_max_new"])/n, 4),
        "len_mean": round(statistics.mean(lens), 1),
        "len_p90": sorted(lens)[min(n-1, int(0.9*n))],
        "L_to_marker": mean([r["L_to_marker"] for r in wh]),
        "L_post_marker": mean([r["L_post_marker"] for r in wh]),
        "rep4": mean([r["rep_4gram"] for r in rows]),
        "avg_k_decode": rows[0].get("avg_k_decode"),
        "avg_k_prefill": rows[0].get("avg_k_prefill"),
    }

# common id set
base = args.baseline if args.baseline in configs else list(configs)[0]
base_by_id = {r["id"]: r for r in configs[base]}

print(f"dir: {args.dir}\nbaseline: {base}\nconfigs: {list(configs)}\n")
print(f"{'config':22s}{'n':>5}{'accS':>7}{'noMark':>8}{'len':>7}{'Lmark':>7}{'Lpost':>7}{'ΔlenvsBase(95%CI)':>24}{'McNemar b/c(p)':>18}")
rows_out = {}
for nm, rows in configs.items():
    s = csummary(rows)
    cur = {r["id"]: r for r in rows}
    ids = [i for i in base_by_id if i in cur]
    deltas = [cur[i]["gen_len"] - base_by_id[i]["gen_len"] for i in ids]
    dm, lo, hi = ST.paired_bootstrap_ci(deltas)
    mc = ST.mcnemar_exact([base_by_id[i]["correct_strict"] for i in ids],
                           [cur[i]["correct_strict"] for i in ids])
    s["delta_len_mean"] = dm; s["delta_len_ci"] = [lo, hi]
    s["mcnemar"] = mc
    rows_out[nm] = s
    ci = f"{dm} ({lo},{hi})" if dm is not None else "-"
    mct = f"{mc['b']}/{mc['c']}(p={mc['p_value']})"
    print(f"{nm:22s}{s['n']:>5}{s['acc_strict']*100:>6.1f}%{s['no_marker']*100:>7.1f}%"
          f"{s['len_mean']:>7.0f}{str(s['L_to_marker']):>7}{str(s['L_post_marker']):>7}{ci:>24}{mct:>18}")

# ---- factorial phase effects (v23) ----
def get(nm):
    return {r["id"]: r for r in configs[nm]} if nm in configs else None

mode = args.mode
if mode == "auto":
    names = set(configs)
    if {"8x8", "8x6", "6x8", "6x6"} <= names or {"8x8", "8x4", "4x8", "4x4"} <= names:
        mode = "factorial"
    elif {"8x8", "8x7", "8x6", "8x5", "8x4"} <= names:
        mode = "dose"
    else:
        mode = "ablation"

analysis = {"dir": args.dir, "baseline": base, "mode": mode, "configs": rows_out}

if mode == "factorial":
    print("\n== Phase-factor effects (paired, length) ==")
    b = get("8x8")
    for K in (6, 4):
        po, do, bo = get(f"{K}x8"), get(f"8x{K}"), get(f"{K}x{K}")
        if not (po and do and bo and b):
            continue
        ids = [i for i in b if i in po and i in do and i in bo]
        pe = ST.paired_bootstrap_ci([po[i]["gen_len"] - b[i]["gen_len"] for i in ids])
        de = ST.paired_bootstrap_ci([do[i]["gen_len"] - b[i]["gen_len"] for i in ids])
        inter = ST.paired_bootstrap_ci([bo[i]["gen_len"] - po[i]["gen_len"] - do[i]["gen_len"] + b[i]["gen_len"] for i in ids])
        print(f"K={K}: prefill_effect={pe[0]} {pe[1:]} | decode_effect={de[0]} {de[1:]} | interaction={inter[0]} {inter[1:]}")
        analysis[f"phase_effect_K{K}"] = {"prefill": pe, "decode": de, "interaction": inter, "n": len(ids)}

if mode == "dose":
    print("\n== Adjacent-K decode dose deltas (paired, length) ==")
    order = ["8x8", "8x7", "8x6", "8x5", "8x4"]
    order = [o for o in order if o in configs]
    for a, bb in zip(order[:-1], order[1:]):
        ga, gb = get(a), get(bb)
        ids = [i for i in ga if i in gb]
        d = ST.paired_bootstrap_ci([gb[i]["gen_len"] - ga[i]["gen_len"] for i in ids])
        print(f"{a} -> {bb}: Δlen={d[0]} {d[1:]}")
        analysis[f"dose_{a}_to_{bb}"] = d

json.dump(analysis, open(os.path.join(args.dir, "analysis.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(args.dir, 'analysis.json')}")
