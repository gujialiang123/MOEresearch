#!/usr/bin/env python3
"""v13: analyze MoE router behavior on agentic input (Qwen3-30B-A3B).

Runs real agent-style prompts through HF transformers with output_router_logits,
and measures per-layer:
  1. Expert selection distribution (which experts are hot/cold; how many experts
     get almost no tokens = "fringe" load imbalance).
  2. Router confidence (softmax prob of the chosen experts; how decisive routing is).
  3. Top-k weight distribution (weight of the #1 vs #8 selected expert -> how much
     the marginal/fringe expert contributes).

Goal: quantify how much room a "batch-level expert consolidation" idea has ---
i.e. how weak the fringe (8th) expert selections are, and how imbalanced/
consolidatable the per-batch expert set is.
"""
import json, os
import numpy as np
import torch

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
OUT = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-15_v13_router"
os.makedirs(OUT, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.bfloat16
)
model = model.to("cuda:0")
model.eval()
cfg = model.config
E, TOPK, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
print(f"E={E} topk={TOPK} L={L}", flush=True)

# Agent-style prompts (tool-calling / reasoning flavored, like toolagent workload)
PROMPTS = [
    "You are an agent with tools: search(query), read_file(path), run_code(code). "
    "Task: find the bug in the authentication module and fix it. First, let me search "
    "for the auth module.\nAction: search(\"authentication module login\")\nObservation: "
    "Found auth/login.py with a token validation function. Let me read it.\nAction: "
    "read_file(\"auth/login.py\")\nObservation: def validate_token(t): return t in cache. "
    "The bug is it doesn't check expiry. Let me fix it.\nAction: run_code(",
    "You are a helpful assistant that plans multi-step tasks. The user wants to book a "
    "flight, reserve a hotel, and rent a car for a trip to Tokyo next month. Let me break "
    "this down step by step. Step 1: search for flights to Tokyo. Step 2: compare hotel "
    "prices near the city center. Step 3: check car rental availability. Starting with",
    "Analyze the following JSON API response and extract the total revenue: "
    "{\"quarters\": [{\"q\": 1, \"rev\": 1200}, {\"q\": 2, \"rev\": 1500}]}. "
    "To compute total revenue, I sum the rev field across all quarters. Let me calculate:",
]

# Accumulators across all tokens
expert_counts = np.zeros((L, E), dtype=np.int64)      # how many token-selections per expert per layer
conf_top1 = [[] for _ in range(L)]                    # softmax prob of top-1 expert
conf_topk_sum = [[] for _ in range(L)]                # sum of softmax prob over chosen topk
weight_by_rank = np.zeros((L, TOPK))                  # avg normalized weight by rank (0=top1..7=top8)
weight_by_rank_n = np.zeros((L, TOPK))
total_tokens = 0

with torch.no_grad():
    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").to("cuda:0")
        out = model(**ids, output_router_logits=True)
        # out.router_logits: tuple of L tensors, each [num_tokens, E]
        rl = out.router_logits
        n_tok = ids["input_ids"].shape[1]
        total_tokens += n_tok
        for layer in range(L):
            logits = rl[layer].float()                # [T, E]
            probs = torch.softmax(logits, dim=-1)     # routing probs
            topk_probs, topk_idx = torch.topk(probs, TOPK, dim=-1)  # [T, TOPK]
            # normalized weights within topk (what actually weights the expert outputs)
            norm_w = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            # accumulate
            for r in range(TOPK):
                weight_by_rank[layer, r] += norm_w[:, r].sum().item()
                weight_by_rank_n[layer, r] += norm_w.shape[0]
            conf_top1[layer].extend(topk_probs[:, 0].cpu().numpy().tolist())
            conf_topk_sum[layer].extend(topk_probs.sum(dim=-1).cpu().numpy().tolist())
            idx_np = topk_idx.cpu().numpy().reshape(-1)
            np.add.at(expert_counts[layer], idx_np, 1)
        print(f"  prompt {pi}: {n_tok} tokens done", flush=True)

# ---- Summaries ----
report = {"E": E, "TOPK": TOPK, "L": L, "total_tokens": total_tokens, "layers": []}
for layer in range(L):
    counts = expert_counts[layer]
    total_sel = counts.sum()
    # fraction of experts that got 0 selections
    zero_experts = int((counts == 0).sum())
    # concentration: what fraction of selections go to the top 25% hottest experts
    sorted_c = np.sort(counts)[::-1]
    top25 = sorted_c[: E // 4].sum() / max(total_sel, 1)
    wbr = (weight_by_rank[layer] / np.maximum(weight_by_rank_n[layer], 1)).tolist()
    report["layers"].append({
        "layer": layer,
        "experts_never_selected": zero_experts,
        "frac_selections_to_top25pct_experts": round(float(top25), 3),
        "router_conf_top1_mean": round(float(np.mean(conf_top1[layer])), 3),
        "router_conf_topk_sum_mean": round(float(np.mean(conf_topk_sum[layer])), 3),
        "weight_rank1": round(wbr[0], 3),
        "weight_rank_last": round(wbr[-1], 3),
        "weight_by_rank": [round(x, 3) for x in wbr],
    })

json.dump(report, open(f"{OUT}/router_analysis.json", "w"), indent=2)

# print aggregate across layers
avg_zero = np.mean([r["experts_never_selected"] for r in report["layers"]])
avg_top25 = np.mean([r["frac_selections_to_top25pct_experts"] for r in report["layers"]])
avg_c1 = np.mean([r["router_conf_top1_mean"] for r in report["layers"]])
avg_w1 = np.mean([r["weight_rank1"] for r in report["layers"]])
avg_wlast = np.mean([r["weight_rank_last"] for r in report["layers"]])
print("\n==== AGGREGATE (across 48 layers) ====")
print(f"total tokens analyzed: {total_tokens}")
print(f"avg experts never selected / layer: {avg_zero:.1f} / {E}")
print(f"avg frac of selections to hottest 25% experts: {avg_top25:.1%}")
print(f"avg router top-1 confidence: {avg_c1:.3f}")
print(f"avg normalized weight of rank-1 expert: {avg_w1:.3f}")
print(f"avg normalized weight of rank-{TOPK} (fringe) expert: {avg_wlast:.3f}")
print(f"\nwrote {OUT}/router_analysis.json")
