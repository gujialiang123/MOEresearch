#!/usr/bin/env python3
"""Plot v16 router distributions from raw.npz (CPU only, no GPU)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/t-jialianggu/work/MOEresearch/results/2026-07-15_v16_router_dist"
d = np.load(f"{OUT}/raw.npz")
E = int(d["E"]); TOPK = int(d["TOPK"])
ec20 = d["expert_counts_layer20"]; cum_share = d["cum_share"]
all_top1 = d["all_top1"]; rank_mean = d["rank_mean"]
rank_p25 = d["rank_p25"]; rank_p75 = d["rank_p75"]; r8 = d["r8"]
gini = float(d["gini_mean"])

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

ax[0,0].bar(range(E), np.sort(ec20)[::-1], color="steelblue")
ax[0,0].set_title(f"Q1: Expert load (layer 20, sorted by load)  Gini={gini:.2f}")
ax[0,0].set_xlabel("expert (sorted, hottest first)"); ax[0,0].set_ylabel("# tokens selected")

xs = np.arange(1, E+1)/E
ax[0,1].plot(xs, cum_share, color="crimson", lw=2, label="actual")
ax[0,1].plot([0,1],[0,1],"k--", alpha=0.5, label="perfectly balanced")
ax[0,1].axvline(0.25, color="gray", ls=":")
ax[0,1].set_title(f"Q1: Traffic concentration (Lorenz)\ntop25% experts = {cum_share[E//4-1]*100:.0f}% traffic")
ax[0,1].set_xlabel("fraction of experts (hottest first)"); ax[0,1].set_ylabel("cumulative traffic share"); ax[0,1].legend()

ax[1,0].hist(all_top1, bins=50, color="seagreen", alpha=0.8)
ax[1,0].axvline(all_top1.mean(), color="red", ls="--", label=f"mean={all_top1.mean():.3f}")
ax[1,0].axvline(1.0/E, color="gray", ls=":", label=f"uniform=1/{E}={1/E:.3f}")
ax[1,0].set_title("Q2: Router top-1 confidence distribution")
ax[1,0].set_xlabel("top-1 softmax probability"); ax[1,0].set_ylabel("# tokens"); ax[1,0].legend()

ranks = np.arange(1, TOPK+1)
ax[1,1].plot(ranks, rank_mean, "o-", color="darkorange", label="mean")
ax[1,1].fill_between(ranks, rank_p25, rank_p75, alpha=0.25, color="orange", label="IQR 25-75%")
ax[1,1].set_title(f"Q3: Normalized weight by rank (rank8={rank_mean[-1]:.3f} vs rank1={rank_mean[0]:.3f})")
ax[1,1].set_xlabel("expert rank (1=strongest)"); ax[1,1].set_ylabel("normalized weight"); ax[1,1].legend()

plt.tight_layout()
plt.savefig(f"{OUT}/router_distributions.png", dpi=110)
print("saved", f"{OUT}/router_distributions.png")
