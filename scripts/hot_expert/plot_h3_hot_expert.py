#!/usr/bin/env python3
"""h3: plots for the hot-expert study (run after h2 analysis).

Generates:
  1. per-layer expert load heatmap (pooled decode) -> visualize hotness & cross-layer drift
  2. cross-layer Jaccard matrix (top-N hot set similarity between layers)
  3. cross-workload per-layer Jaccard curves
  4. residency budget Pareto: decode traffic coverage vs VRAM budget, with L2 line

Run: python scripts/hot_expert/plot_h3_hot_expert.py --dir results/2026-07-19_hot_expert
"""
import os, json, argparse, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/home/t-jialianggu/work/MOEresearch/results/2026-07-19_hot_expert")
ap.add_argument("--topn", type=int, default=16)
ap.add_argument("--pos", default="decode", choices=["decode", "all"])
args = ap.parse_args()
D = args.dir
os.makedirs(f"{D}/plots", exist_ok=True)

meta = json.load(open(f"{D}/collect_meta.json"))
E, TOPK, L = meta["E"], meta["TOPK"], meta["L"]
workloads = [w["workload"] for w in meta["workloads"]]
key = "counts_decode" if args.pos == "decode" else "counts_all"
counts = {wl: np.load(f"{D}/traces/{wl}.npz", allow_pickle=True)[key].astype(float) for wl in workloads}
pooled = np.sum([counts[wl] for wl in workloads], axis=0)  # [L,E]
N = args.topn

def topn_set(row, n):
    return set(np.argsort(row)[::-1][:n].tolist())
def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0

# 1. heatmap: normalize each layer to fraction, sort experts by GLOBAL pooled hotness
global_order = np.argsort(pooled.sum(0))[::-1]
norm = pooled[:, global_order] / pooled.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(norm, aspect="auto", cmap="hot", interpolation="nearest")
ax.set_xlabel("expert (sorted by GLOBAL hotness)"); ax.set_ylabel("layer")
ax.set_title(f"Per-layer decode expert load (fraction), experts globally sorted\n"
             f"vertical streaks on the left = same experts hot across layers")
fig.colorbar(im, ax=ax, label="fraction of layer decode traffic")
fig.tight_layout(); fig.savefig(f"{D}/plots/1_load_heatmap.png", dpi=120); plt.close(fig)

# 2. cross-layer Jaccard matrix
layer_hot = [topn_set(pooled[l], N) for l in range(L)]
M = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        M[i, j] = jaccard(layer_hot[i], layer_hot[j])
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
ax.set_title(f"Cross-layer top-{N} hot-set Jaccard\n(bright=same experts hot; dark=different)")
ax.set_xlabel("layer"); ax.set_ylabel("layer")
fig.colorbar(im, ax=ax, label="Jaccard")
fig.tight_layout(); fig.savefig(f"{D}/plots/2_cross_layer_jaccard.png", dpi=120); plt.close(fig)

# 3. cross-workload per-layer jaccard curves
fig, ax = plt.subplots(figsize=(11, 5))
for a, b in itertools.combinations(workloads, 2):
    js = [jaccard(topn_set(counts[a][l], N), topn_set(counts[b][l], N)) for l in range(L)]
    ax.plot(range(L), js, marker=".", label=f"{a} vs {b}")
ax.set_xlabel("layer"); ax.set_ylabel(f"top-{N} Jaccard")
ax.set_title("Cross-workload hot-set similarity per layer\n(high=global prior; low=input-dependent)")
ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{D}/plots/3_cross_workload_jaccard.png", dpi=120); plt.close(fig)

# 4. residency budget Pareto
ana = json.load(open(f"{D}/analysis.json"))
rows = ana["Q3_l2_fit"]["residency_budget_curve"]
mb_per_expert = ana["expert_size"]["MB_per_expert_bf16"]
xs = [r["resident_all_layers_GB"] for r in rows]
ys = [r["decode_traffic_coverage"] * 100 for r in rows]
ns = [r["topn_per_layer"] for r in rows]
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(xs, ys, marker="o")
for x, y, n in zip(xs, ys, ns):
    ax.annotate(f"top{n}", (x, y), fontsize=8, xytext=(3, -8), textcoords="offset points")
ax.set_xlabel("VRAM to keep top-N experts/layer resident, all 48 layers (GB)")
ax.set_ylabel("% of decode traffic covered")
ax.set_title("Residency budget vs decode-traffic coverage\n(keeping hot experts resident per layer)")
ax.axhline(90, ls="--", c="gray", alpha=0.6); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{D}/plots/4_residency_pareto.png", dpi=120); plt.close(fig)

print(f"wrote 4 plots to {D}/plots/")
