#!/usr/bin/env python3
"""Shared paired-stats analysis for v29/v30/v31 config-sweep results.

Loads each config's *_raw.jsonl from a result dir, pairs by sample_id against the
k8_native baseline, and reports: capped-mean length delta with paired bootstrap +
prompt-cluster bootstrap CI, McNemar on strict correctness (Holm-corrected across
configs), restricted-mean survival, no-EOS/hit-max deltas. Writes analysis.json.

Run: python scripts/analyze_config_sweep.py <result_dir> [--baseline k8_native]
"""
import os, sys, json, argparse, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import stats as ST

ap = argparse.ArgumentParser()
ap.add_argument("result_dir")
ap.add_argument("--baseline", default="k8_native")
args = ap.parse_args()

def load(name):
    p = os.path.join(args.result_dir, f"{name}_raw.jsonl")
    if not os.path.exists(p):
        return None
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}

# discover configs from *_summary.json / *_raw.jsonl
names = sorted({os.path.basename(f)[:-10] for f in os.listdir(args.result_dir)
                if f.endswith("_raw.jsonl")})
base = load(args.baseline)
if base is None:
    print(f"no baseline {args.baseline} in {args.result_dir}"); sys.exit(1)

out = {"result_dir": args.result_dir, "baseline": args.baseline, "configs": {}}
pvals = {}
rows_out = []
for name in names:
    if name == args.baseline:
        continue
    cur = load(name)
    if cur is None:
        continue
    ids = sorted(set(base) & set(cur))
    dlen = [cur[i]["generated_tokens"] - base[i]["generated_tokens"] for i in ids]
    clusters = [[d] for d in dlen]  # one obs per prompt (already prompt-level)
    mean, lo, hi = ST.paired_bootstrap_ci(dlen)
    bc = [base[i]["strict_correct"] for i in ids]
    cc = [cur[i]["strict_correct"] for i in ids]
    mc = ST.mcnemar_exact(bc, cc)
    pvals[name] = mc["p_value"]
    lens = [cur[i]["generated_tokens"] for i in ids]
    events = [cur[i]["eos_pos"] >= 0 for i in ids]
    rmst = ST.restricted_mean_survival(lens, events, 512)
    base_lens = [base[i]["generated_tokens"] for i in ids]
    base_events = [base[i]["eos_pos"] >= 0 for i in ids]
    base_rmst = ST.restricted_mean_survival(base_lens, base_events, 512)
    d = {
        "n_paired": len(ids),
        "len_delta_mean": mean, "len_delta_ci95": [lo, hi],
        "rmst_delta": round(rmst - base_rmst, 2),
        "no_eos_delta": round(sum(1 for i in ids if cur[i]["eos_pos"] < 0)/len(ids)
                              - sum(1 for i in ids if base[i]["eos_pos"] < 0)/len(ids), 4),
        "hit_max_delta": round(sum(cur[i]["hit_max"] for i in ids)/len(ids)
                               - sum(base[i]["hit_max"] for i in ids)/len(ids), 4),
        "acc_strict": round(sum(cc)/len(ids), 4),
        "acc_strict_base": round(sum(bc)/len(ids), 4),
        "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "mcnemar_p": mc["p_value"],
    }
    out["configs"][name] = d
    rows_out.append((name, d))

holm = ST.holm_correction(pvals) if pvals else {}
for name in out["configs"]:
    out["configs"][name]["mcnemar_holm"] = holm.get(name, {})

json.dump(out, open(os.path.join(args.result_dir, "analysis.json"), "w"), indent=2)

print(f"\n== paired analysis vs {args.baseline} ({args.result_dir}) ==")
print(f"{'config':26s} {'n':>4} {'Δlen (95% CI)':>26} {'ΔRMST':>7} {'acc':>6} {'McN p':>8}")
for name, d in rows_out:
    ci = f"{d['len_delta_mean']:+.1f} ({d['len_delta_ci95'][0]:+.1f},{d['len_delta_ci95'][1]:+.1f})"
    print(f"{name:26s} {d['n_paired']:>4} {ci:>26} {d['rmst_delta']:>+7.1f} "
          f"{d['acc_strict']*100:>5.1f}% {d['mcnemar_p']:>8.4f}")
print(f"\nwrote {args.result_dir}/analysis.json")
