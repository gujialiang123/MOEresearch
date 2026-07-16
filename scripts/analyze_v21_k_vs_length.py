#!/usr/bin/env python3
"""Analyze v21 K-vs-length raw logs -> dose curve + L_to_answer/L_post_answer
decomposition. Pure log analysis, NO model needed (re-runnable offline).

Answers:
  - Does mean generated length change monotonically with K?
  - Is the extra length in L_to_answer (reasoning) or L_post_answer (termination)?
  - Does the "no ####" (termination/format failure) rate rise as K drops?
  - Repetition / max-new-hit trends.
  - Paired per-sample length delta vs K=8 (with bootstrap CI).

Run: python scripts/analyze_v21_k_vs_length.py [--dir results/2026-07-16_v21_k_vs_length]
"""
import os, sys, json, glob, argparse, statistics, random

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-16_v21_k_vs_length")
args = ap.parse_args()

def load(name):
    p = os.path.join(args.dir, f"{name}_raw.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p)]

# discover configs present
configs = sorted({os.path.basename(f).replace("_raw.jsonl", "")
                  for f in glob.glob(os.path.join(args.dir, "*_raw.jsonl"))},
                 key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0))
data = {c: load(c) for c in configs}
data = {c: v for c, v in data.items() if v}

def kof(name):
    return int(''.join(c for c in name if c.isdigit()) or 0)

def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")

def bootstrap_ci(deltas, iters=2000, alpha=0.05):
    if not deltas:
        return (None, None)
    n = len(deltas)
    means = []
    for _ in range(iters):
        s = [deltas[random.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return (round(means[int(alpha/2*iters)], 2), round(means[int((1-alpha/2)*iters)], 2))

random.seed(0)
base_name = "fixed_k8" if "fixed_k8" in data else min(data, key=lambda c: abs(kof(c)-8))
base = {r["id"]: r for r in data[base_name]}

print(f"dir: {args.dir}")
print(f"configs: {configs}  baseline: {base_name}\n")
print(f"{'config':11s}{'K':>4}{'n':>6}{'acc%':>7}{'noHash%':>9}{'len':>7}{'L_to_ans':>9}{'L_post':>8}"
      f"{'rep4':>7}{'maxhit%':>8}{'ΔlenvsK8':>10}{'95%CI':>16}")
summary_rows = []
for c in sorted(data, key=kof):
    rows = data[c]
    n = len(rows)
    k = kof(c)
    acc = mean(r["correct"] for r in rows) * 100
    noh = mean(1 if r["hash_tok_pos"] < 0 else 0 for r in rows) * 100
    length = mean(r["gen_len"] for r in rows)
    lto = mean(r["L_to_answer"] for r in rows)
    lpo = mean(r["L_post_answer"] for r in rows)
    rep = mean(r.get("rep_4gram_frac", 0) for r in rows)
    maxhit = mean(1 if r["hit_max_new"] else 0 for r in rows) * 100
    # paired delta vs baseline
    deltas = [r["gen_len"] - base[r["id"]]["gen_len"] for r in rows if r["id"] in base]
    dmean = mean(deltas)
    ci = bootstrap_ci(deltas)
    print(f"{c:11s}{k:>4}{n:>6}{acc:>7.1f}{noh:>9.1f}{length:>7.0f}{lto:>9.0f}{lpo:>8.0f}"
          f"{rep:>7.3f}{maxhit:>8.1f}{dmean:>10.1f}{str(ci):>16}")
    summary_rows.append({
        "config": c, "K": k, "n": n, "accuracy_pct": round(acc, 2),
        "no_hash_pct": round(noh, 2), "gen_len_mean": round(length, 1),
        "L_to_answer_mean": round(lto, 1), "L_post_answer_mean": round(lpo, 1),
        "rep_4gram_mean": round(rep, 4), "hit_max_new_pct": round(maxhit, 1),
        "delta_len_vs_k8_mean": round(dmean, 1), "delta_len_95ci": ci,
    })

# decomposition of the length change: how much of Δlen is Δ(L_to_answer) vs Δ(L_post_answer)?
print("\n== Length-change decomposition vs K=8 (paired, samples with #### in BOTH) ==")
print(f"{'config':11s}{'K':>4}{'Δlen':>8}{'Δ L_to_answer':>15}{'Δ L_post_answer':>17}")
for c in sorted(data, key=kof):
    if c == base_name:
        continue
    cur = {r["id"]: r for r in data[c]}
    ids = [i for i in base if i in cur and base[i]["hash_tok_pos"] >= 0 and cur[i]["hash_tok_pos"] >= 0]
    dlen = mean(cur[i]["gen_len"] - base[i]["gen_len"] for i in ids)
    dto = mean(cur[i]["L_to_answer"] - base[i]["L_to_answer"] for i in ids)
    dpo = mean(cur[i]["L_post_answer"] - base[i]["L_post_answer"] for i in ids)
    print(f"{c:11s}{kof(c):>4}{dlen:>8.1f}{dto:>15.1f}{dpo:>17.1f}")

out = {"dir": args.dir, "baseline": base_name, "rows": summary_rows}
json.dump(out, open(os.path.join(args.dir, "analysis.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(args.dir, 'analysis.json')}")
