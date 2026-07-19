#!/usr/bin/env python3
"""h2: analyze hot-expert traces -> viability of emergent shared-expert (Idea 1).

Answers three make-or-break questions using the per-workload [L,E] selection-count
matrices from h1 (DECODE positions are the primary signal; all-positions as a check):

Q1 CROSS-LAYER STABILITY: are the SAME experts hot across the 48 layers, or does each
   layer have its own hot set? If each layer differs, a single global residency set
   cannot cover all layers -> residency must be per-layer.
   Metric: for top-N per layer, pairwise Jaccard across layers; also global-topN coverage.

Q2 CROSS-WORKLOAD STABILITY: is hotness a global prior (same experts hot on gsm8k /
   coding / general / synthetic) or input-dependent? If input-dependent, you cannot
   precompute a static hot set.
   Metric: per-layer Jaccard of top-N sets between each workload pair.

Q3 L2 / RESIDENCY FIT: how big is one expert in bytes; how much VRAM to keep the top-N
   hot experts PER LAYER resident; does any meaningful hot set fit in H200 L2 (~50MB)?
   Also: traffic coverage vs residency budget curve (Pareto).

Outputs: results/<date>_hot_expert/analysis.json + printed tables. Plots via h3.

Run: python scripts/hot_expert/analyze_h2_hot_expert.py --dir results/2026-07-19_hot_expert
"""
import os, sys, json, argparse, itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/home/t-jialianggu/work/MOEresearch/results/2026-07-19_hot_expert")
ap.add_argument("--topn", type=int, default=16, help="hot set size per layer for Jaccard")
ap.add_argument("--pos", default="decode", choices=["decode", "all"], help="which positions")
args = ap.parse_args()
D = args.dir

meta = json.load(open(f"{D}/collect_meta.json"))
E, TOPK, L = meta["E"], meta["TOPK"], meta["L"]
moe_inter = meta["moe_intermediate_size"]
hidden = meta["hidden_size"]
workloads = [w["workload"] for w in meta["workloads"]]

def load_counts(wl):
    d = np.load(f"{D}/traces/{wl}.npz", allow_pickle=True)
    return d["counts_decode" if args.pos == "decode" else "counts_all"].astype(np.float64)  # [L,E]

counts = {wl: load_counts(wl) for wl in workloads}
# pooled across workloads (global hotness)
pooled = np.sum([counts[wl] for wl in workloads], axis=0)  # [L,E]

def topn_set(row, n):
    return set(np.argsort(row)[::-1][:n].tolist())

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0

N = args.topn

# ---------- Q1: cross-layer stability (within pooled) ----------
layer_hot = [topn_set(pooled[l], N) for l in range(L)]
cross_layer_jac = []
for i, j in itertools.combinations(range(L), 2):
    cross_layer_jac.append(jaccard(layer_hot[i], layer_hot[j]))
cross_layer_jac = np.array(cross_layer_jac)
# adjacent-layer jaccard (are neighbors more similar?)
adj_jac = np.array([jaccard(layer_hot[l], layer_hot[l+1]) for l in range(L-1)])
# global top-N (ignore layer): how much per-layer traffic does a single global set cover?
global_hot = topn_set(pooled.sum(axis=0), N)
global_cov = []  # fraction of each layer's decode traffic captured by the GLOBAL hot set
for l in range(L):
    tot = pooled[l].sum()
    cov = pooled[l][list(global_hot)].sum() / tot if tot else 0
    global_cov.append(cov)
global_cov = np.array(global_cov)
# per-layer self coverage (top-N of that layer)
perlayer_cov = np.array([pooled[l][list(layer_hot[l])].sum() / pooled[l].sum() for l in range(L)])

# ---------- Q2: cross-workload stability (per layer, pairwise) ----------
q2 = {}
wl_pairs = list(itertools.combinations(workloads, 2))
for a, b in wl_pairs:
    jacs = []
    for l in range(L):
        jacs.append(jaccard(topn_set(counts[a][l], N), topn_set(counts[b][l], N)))
    q2[f"{a}__vs__{b}"] = {
        "mean_jaccard": round(float(np.mean(jacs)), 4),
        "min_jaccard": round(float(np.min(jacs)), 4),
        "max_jaccard": round(float(np.max(jacs)), 4),
    }
# cross-workload coverage: hot set from gsm8k, how much of coding traffic does it cover?
xwl_cov = {}
if "gsm8k" in workloads:
    ref = "gsm8k"
    for wl in workloads:
        if wl == ref:
            continue
        covs = []
        for l in range(L):
            hot_ref = list(topn_set(counts[ref][l], N))
            tot = counts[wl][l].sum()
            covs.append(counts[wl][l][hot_ref].sum() / tot if tot else 0)
        xwl_cov[f"{ref}_hotset_covers_{wl}"] = round(float(np.mean(covs)), 4)

# ---------- Q3: L2 / residency fit ----------
# Qwen3 MoE expert: gate_proj [moe_inter, hidden], up_proj [moe_inter, hidden], down_proj [hidden, moe_inter]
# params per expert = 3 * moe_inter * hidden ; bf16 = 2 bytes
params_per_expert = 3 * moe_inter * hidden
bytes_per_expert_bf16 = params_per_expert * 2
H200_L2_MB = 50.0
# residency budget curve: keep top-n per layer resident across all L layers
res_rows = []
for n in [1, 2, 4, 8, 16, 32, 64, 128]:
    total_bytes = n * L * bytes_per_expert_bf16
    # traffic coverage if we keep top-n per layer (using pooled)
    cov = np.mean([pooled[l][list(topn_set(pooled[l], n))].sum() / pooled[l].sum() for l in range(L)])
    # how many hot experts fit in L2 (single layer)
    res_rows.append({
        "topn_per_layer": n,
        "resident_all_layers_GB": round(total_bytes / 1e9, 3),
        "one_layer_topn_MB": round(n * bytes_per_expert_bf16 / 1e6, 2),
        "fits_in_L2_one_layer": (n * bytes_per_expert_bf16 / 1e6) <= H200_L2_MB,
        "decode_traffic_coverage": round(float(cov), 4),
    })

report = {
    "meta": {"E": E, "TOPK": TOPK, "L": L, "positions": args.pos, "topn": N,
             "workloads": workloads,
             "n_tokens_decode": {w["workload"]: w["n_tokens_decode"] for w in meta["workloads"]}},
    "expert_size": {
        "params_per_expert": params_per_expert,
        "bytes_per_expert_bf16": bytes_per_expert_bf16,
        "MB_per_expert_bf16": round(bytes_per_expert_bf16 / 1e6, 3),
        "all_experts_one_layer_MB": round(E * bytes_per_expert_bf16 / 1e6, 1),
        "H200_L2_MB": H200_L2_MB,
        "experts_fitting_in_L2": int(H200_L2_MB * 1e6 // bytes_per_expert_bf16),
    },
    "Q1_cross_layer": {
        "topn": N,
        "mean_pairwise_jaccard": round(float(cross_layer_jac.mean()), 4),
        "median_pairwise_jaccard": round(float(np.median(cross_layer_jac)), 4),
        "p90_pairwise_jaccard": round(float(np.percentile(cross_layer_jac, 90)), 4),
        "adjacent_layer_mean_jaccard": round(float(adj_jac.mean()), 4),
        "global_hotset_mean_layer_coverage": round(float(global_cov.mean()), 4),
        "perlayer_hotset_mean_coverage": round(float(perlayer_cov.mean()), 4),
        "interp": "high jaccard => same experts hot across layers (global residency ok); "
                  "low => per-layer hot sets differ (need per-layer residency)",
    },
    "Q2_cross_workload": {
        "topn": N,
        "pairwise": q2,
        "mean_over_pairs": round(float(np.mean([v["mean_jaccard"] for v in q2.values()])), 4),
        "gsm8k_hotset_cross_coverage": xwl_cov,
        "interp": "high jaccard => hotness is a global prior (static hot set works); "
                  "low => input-dependent (cannot precompute)",
    },
    "Q3_l2_fit": {
        "residency_budget_curve": res_rows,
        "interp": "coverage vs VRAM/L2 budget. If small resident set already covers most "
                  "decode traffic AND fits, caching is viable.",
    },
}
json.dump(report, open(f"{D}/analysis.json", "w"), indent=2)

# ---------- print ----------
print(f"=== HOT-EXPERT ANALYSIS (pos={args.pos}, topN={N}) ===")
print(f"workloads: {workloads}  |  decode tokens: {report['meta']['n_tokens_decode']}\n")
es = report["expert_size"]
print(f"[expert size] {es['MB_per_expert_bf16']} MB/expert bf16 | all {E} experts/layer = "
      f"{es['all_experts_one_layer_MB']} MB | H200 L2={es['H200_L2_MB']}MB fits "
      f"{es['experts_fitting_in_L2']} experts\n")
q1 = report["Q1_cross_layer"]
print(f"[Q1 cross-layer] mean pairwise Jaccard(top{N}) = {q1['mean_pairwise_jaccard']} "
      f"(adjacent={q1['adjacent_layer_mean_jaccard']})")
print(f"   global hot set covers {q1['global_hotset_mean_layer_coverage']*100:.1f}% of per-layer decode traffic")
print(f"   per-layer hot set covers {q1['perlayer_hotset_mean_coverage']*100:.1f}%\n")
q2r = report["Q2_cross_workload"]
print(f"[Q2 cross-workload] mean pairwise Jaccard(top{N}) over pairs = {q2r['mean_over_pairs']}")
for k, v in q2r["pairwise"].items():
    print(f"   {k}: mean={v['mean_jaccard']} (min={v['min_jaccard']}, max={v['max_jaccard']})")
print(f"   cross-coverage: {q2r['gsm8k_hotset_cross_coverage']}\n")
print(f"[Q3 residency budget curve]")
print(f"   {'topN/layer':>10}{'1-layer MB':>12}{'fitL2':>7}{'all-L GB':>10}{'decode cov':>12}")
for r in report["Q3_l2_fit"]["residency_budget_curve"]:
    print(f"   {r['topn_per_layer']:>10}{r['one_layer_topn_MB']:>12}{str(r['fits_in_L2_one_layer']):>7}"
          f"{r['resident_all_layers_GB']:>10}{r['decode_traffic_coverage']*100:>11.1f}%")
print(f"\nwrote {D}/analysis.json")
