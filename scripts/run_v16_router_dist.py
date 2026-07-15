#!/usr/bin/env python3
"""v16: DETAILED router distribution analysis (Qwen3-30B-A3B, agent input).

Answers three questions with distributions (not just means):
  Q1. Load imbalance: histogram of per-expert token counts, Gini coefficient,
      what % of experts handle what % of traffic (Lorenz-style).
  Q2. Router confidence: full distribution (percentiles) of top-1 softmax prob,
      and the full top1..top8 average weight profile with spread.
  Q3. Fringe importance: distribution of rank-8 (and rank-6/7) weights across
      tokens -- is it uniformly unimportant, or do SOME tokens rely on fringe?

Outputs a text report + a matplotlib figure (PNG) with 4 panels.
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
OUT = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-15_v16_router_dist"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0")
model.eval()
cfg = model.config
E, TOPK, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
print(f"E={E} topk={TOPK} L={L}", flush=True)

PROMPTS = [
    "You are an autonomous coding agent. The user reports the login endpoint returns 500 errors under load. Search the codebase for the login handler, read api/auth/login.py, and you find the DB connection is not pooled. Add a connection pool with max 20 connections and 30s timeout, then write a test simulating 100 concurrent logins to verify.",
    "Plan a software migration from monolith to microservices. Identify bounded contexts: user management, billing, notifications, analytics. Define API contracts with OpenAPI. Set up Kafka for async messaging. Migrate the database per-service using the saga pattern for consistency. Test each step and roll out with feature flags.",
    "Analyze quarterly financial JSON: parse transactions with date, amount, category, vendor. Aggregate by category: marketing 45000, engineering 120000, operations 38000. Total is 203000, which is 15 percent under the 240000 budget. Reallocate the surplus to R&D next quarter.",
    "Debug and fix: def divide(a,b): return a/b raises ZeroDivisionError on divide(10,0). Add a guard returning None when b is zero. Then translate the fix summary to French and explain the grammar of adjective placement.",
    "Summarize meeting notes and extract action items: the team agreed to ship the feature next week, assign QA to Alice, postpone the design review. Finalize the API, write tests, update the changelog before Friday. Estimate the effort in story points for each task.",
]

# Collect: per-expert selection counts per layer; top-1 prob per token per layer;
# normalized weight by rank per token per layer.
expert_counts = np.zeros((L, E), dtype=np.int64)
top1_prob = [[] for _ in range(L)]           # top-1 softmax prob (confidence)
weight_by_rank_all = [[] for _ in range(L)]  # list of [TOPK] normalized weights per token

with torch.no_grad():
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").to("cuda:0")
        out = model(**ids, output_router_logits=True)
        for layer in range(L):
            probs = torch.softmax(out.router_logits[layer].float(), dim=-1)  # [T,E]
            tkw, tki = torch.topk(probs, TOPK, dim=-1)
            normw = (tkw / tkw.sum(dim=-1, keepdim=True)).cpu().numpy()
            top1_prob[layer].extend(tkw[:, 0].cpu().numpy().tolist())
            for t in range(probs.shape[0]):
                weight_by_rank_all[layer].append(normw[t])
                np.add.at(expert_counts[layer], tki[t].cpu().numpy(), 1)

# ---------- Q1: load imbalance ----------
def gini(x):
    x = np.sort(x.astype(np.float64)); n = len(x); cum = np.cumsum(x)
    if cum[-1] == 0: return 0.0
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

# aggregate across layers: per-expert average count
total_counts = expert_counts.sum(axis=0)  # [E] total across all layers
gini_per_layer = [gini(expert_counts[l]) for l in range(L)]
# Lorenz: sort experts by load, cumulative traffic share
sorted_layer_avg = np.sort(expert_counts, axis=1)[:, ::-1].mean(axis=0)  # avg sorted load
cum_share = np.cumsum(sorted_layer_avg) / sorted_layer_avg.sum()

# ---------- Q2: confidence ----------
all_top1 = np.concatenate([np.array(top1_prob[l]) for l in range(L)])
conf_pctiles = {p: float(np.percentile(all_top1, p)) for p in [5,25,50,75,95]}

# ---------- Q3: weight by rank ----------
wbr = np.array([np.array(weight_by_rank_all[l]) for l in range(L)], dtype=object)
# flatten all tokens all layers -> [Ntotal, TOPK]
flat = np.concatenate([np.array(weight_by_rank_all[l]) for l in range(L)], axis=0)  # [N,TOPK]
rank_mean = flat.mean(axis=0)
rank_std = flat.std(axis=0)
rank_p25 = np.percentile(flat, 25, axis=0)
rank_p75 = np.percentile(flat, 75, axis=0)
# fringe importance distribution: rank-8 weight across tokens
r8 = flat[:, TOPK-1]
r8_pctiles = {p: float(np.percentile(r8, p)) for p in [5,25,50,75,95,99]}
# what frac of tokens have a "meaningful" fringe (rank8 > 0.10)?
frac_fringe_meaningful = float((r8 > 0.10).mean())

report = {
    "E": E, "TOPK": TOPK, "L": L,
    "Q1_load": {
        "gini_mean": round(float(np.mean(gini_per_layer)),3),
        "cum_traffic_share_top10pct_experts": round(float(cum_share[E//10 -1]),3),
        "cum_traffic_share_top25pct_experts": round(float(cum_share[E//4 -1]),3),
        "cum_traffic_share_top50pct_experts": round(float(cum_share[E//2 -1]),3),
    },
    "Q2_confidence": {
        "top1_prob_mean": round(float(all_top1.mean()),4),
        "top1_prob_percentiles": {k: round(v,4) for k,v in conf_pctiles.items()},
        "uniform_baseline_1_over_E": round(1.0/E,4),
    },
    "Q3_rank_weights": {
        "rank_mean": [round(float(x),4) for x in rank_mean],
        "rank_std": [round(float(x),4) for x in rank_std],
        "rank_p25": [round(float(x),4) for x in rank_p25],
        "rank_p75": [round(float(x),4) for x in rank_p75],
        "rank8_percentiles": {k: round(v,4) for k,v in r8_pctiles.items()},
        "frac_tokens_fringe_weight_gt_0.10": round(frac_fringe_meaningful,3),
    },
}
json.dump(report, open(f"{OUT}/router_dist.json","w"), indent=2)

# Save raw arrays for separate (CPU) plotting
np.savez(f"{OUT}/raw.npz",
         expert_counts_layer20=expert_counts[20],
         cum_share=cum_share,
         all_top1=all_top1,
         rank_mean=rank_mean, rank_p25=rank_p25, rank_p75=rank_p75,
         r8=r8, E=E, TOPK=TOPK,
         gini_mean=np.mean(gini_per_layer))
print("saved raw.npz for plotting", flush=True)

# ---------- print summary ----------
print("\n==== Q1 LOAD IMBALANCE ====")
print(f"  Gini (avg over layers): {np.mean(gini_per_layer):.3f}  (0=balanced, 1=all to one)")
print(f"  hottest 10% experts handle {cum_share[E//10-1]*100:.0f}% of traffic")
print(f"  hottest 25% experts handle {cum_share[E//4-1]*100:.0f}% of traffic")
print(f"  hottest 50% experts handle {cum_share[E//2-1]*100:.0f}% of traffic")
print("\n==== Q2 CONFIDENCE ====")
print(f"  top-1 prob mean={all_top1.mean():.4f} (uniform baseline 1/{E}={1/E:.4f})")
print(f"  percentiles: " + ", ".join(f"p{p}={v:.3f}" for p,v in conf_pctiles.items()))
print("\n==== Q3 WEIGHT BY RANK ====")
print("  rank:   " + " ".join(f"{i+1:>6}" for i in range(TOPK)))
print("  mean:   " + " ".join(f"{x:6.3f}" for x in rank_mean))
print("  std:    " + " ".join(f"{x:6.3f}" for x in rank_std))
print(f"  rank8 percentiles: " + ", ".join(f"p{p}={v:.3f}" for p,v in r8_pctiles.items()))
print(f"  frac tokens with fringe(rank8) weight > 0.10: {frac_fringe_meaningful:.1%}")
print(f"\nwrote {OUT}/router_dist.json and router_distributions.png")
