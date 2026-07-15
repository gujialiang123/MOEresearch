#!/usr/bin/env python3
"""v14: simulate batch-level expert consolidation -> transfer saving vs cost.

For a real agent batch, capture per-token per-layer top-k expert choices + router
probs. Then simulate: for each token, redirect its FRINGE experts (rank >= r) to
an ALREADY-ACTIVATED expert in that (batch, layer) — specifically the
already-activated expert with the highest router score among all experts for that
token. Measure:
  - distinct experts activated per layer BEFORE vs AFTER (= transfer count saved)
  - fraction of routing weight that was redirected (= proxy for accuracy cost)
Sweep the fringe threshold r = 8 (none), 7, 6, 5, 4 to get a tradeoff curve.

This quantifies the user's idea: how much HBM transfer can consolidation save,
at what routing-weight cost.
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
OUT = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-15_v14_consolidation"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0")
model.eval()
cfg = model.config
E, TOPK, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
print(f"E={E} topk={TOPK} L={L}", flush=True)

# One agent "batch": treat a set of agent prompts' tokens as concurrent decode batch.
PROMPTS = [
    "You are an agent with tools: search(query), read_file(path), run_code(code). Task: find and fix the auth bug. Action: search(\"auth login token\") Observation: found auth/login.py. Action: read_file(\"auth/login.py\")",
    "Plan a trip to Tokyo: book flight, reserve hotel, rent car. Step 1: search flights. Step 2: compare hotels near center. Step 3: check car rental. Starting with flights now.",
    "Extract total revenue from JSON: {\"quarters\":[{\"q\":1,\"rev\":1200},{\"q\":2,\"rev\":1500}]}. Sum rev across quarters. Calculating: 1200 + 1500 =",
    "Debug this Python: def f(x): return x/0. The error is ZeroDivisionError. To fix, add a check: if x denominator is zero, return None. Let me rewrite the function.",
]

# Collect per-layer, per-token: topk expert indices + normalized weights + full router probs (for finding alt experts)
# We simulate a "batch" = all tokens from all prompts at their last position (decode-like), plus we also do full-seq.
per_layer_topk_idx = [[] for _ in range(L)]     # list of [TOPK] arrays
per_layer_topk_w = [[] for _ in range(L)]       # normalized weights
per_layer_probs = [[] for _ in range(L)]        # full [E] router prob per token

with torch.no_grad():
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").to("cuda:0")
        out = model(**ids, output_router_logits=True)
        for layer in range(L):
            probs = torch.softmax(out.router_logits[layer].float(), dim=-1)  # [T,E]
            tkw, tki = torch.topk(probs, TOPK, dim=-1)
            normw = (tkw / tkw.sum(dim=-1, keepdim=True)).cpu().numpy()
            for t in range(probs.shape[0]):
                per_layer_topk_idx[layer].append(tki[t].cpu().numpy())
                per_layer_topk_w[layer].append(normw[t])
                per_layer_probs[layer].append(probs[t].cpu().numpy())

# Simulate consolidation at fringe threshold r: redirect ranks [r..TOPK-1] to an
# already-activated expert (highest router prob among currently-active set for that token).
def simulate(r):
    saved_frac_list = []   # per layer: (distinct_before - distinct_after)/distinct_before
    weight_moved_list = [] # per layer: fraction of total routing weight redirected
    for layer in range(L):
        idxs = per_layer_topk_idx[layer]      # list of [TOPK]
        ws = per_layer_topk_w[layer]
        probs = per_layer_probs[layer]
        T = len(idxs)
        # BEFORE: distinct experts across the batch
        before = set()
        for a in idxs: before.update(a.tolist())
        # Build the active set = experts kept by rank<r across all tokens (the "core")
        core = set()
        for a in idxs:
            core.update(a[:r].tolist())
        # AFTER: each token keeps rank<r; fringe ranks redirected to best-scoring core expert
        after = set(core)
        weight_moved = 0.0
        total_w = 0.0
        for a, w, p in zip(idxs, ws, probs):
            total_w += w.sum()
            # keep core
            # fringe ranks r..TOPK-1: redirect
            for rank in range(r, TOPK):
                weight_moved += w[rank]
                # (redirect target is in core; core already in 'after', no new expert added)
            # note: we do NOT add fringe experts to 'after' (they're redirected into core)
        distinct_before = len(before)
        distinct_after = len(after)
        saved_frac_list.append((distinct_before - distinct_after) / max(distinct_before,1))
        weight_moved_list.append(weight_moved / max(total_w,1e-9))
    return float(np.mean(saved_frac_list)), float(np.mean(weight_moved_list))

print("\n==== Batch-level expert consolidation tradeoff ====")
print(f"batch tokens (concat): {sum(len(x) for x in per_layer_topk_idx[0:1])}")
print(f"{'fringe_from_rank':>16}{'transfer_saved':>16}{'weight_redirected':>18}")
results = []
for r in [TOPK, 7, 6, 5, 4, 3]:   # r=TOPK means no consolidation (baseline)
    saved, moved = simulate(r)
    label = "none(baseline)" if r == TOPK else f"rank>={r}"
    print(f"{label:>16}{saved:>15.1%}{moved:>17.1%}")
    results.append({"consolidate_fringe_from_rank": r, "avg_transfer_saved": round(saved,4), "avg_weight_redirected": round(moved,4)})

json.dump({"E":E,"TOPK":TOPK,"L":L,"results":results}, open(f"{OUT}/consolidation_tradeoff.json","w"), indent=2)
print(f"\nwrote {OUT}/consolidation_tradeoff.json")
