#!/usr/bin/env python3
"""v14b: consolidation tradeoff AS A FUNCTION OF BATCH SIZE.

Key insight from v14: at large token counts (250) experts are ~fully covered
(102/128), so consolidation saves little. But real decode concurrency is 6-20,
where expert coverage is partial and consolidation should save more.

This script: capture per-token per-layer topk from agent prompts, then form
random decode "batches" of size B in {4,8,16,32,64}, and for each B measure
consolidation (redirect fringe rank>=r to already-active core expert):
  - transfer_saved = (distinct_before - distinct_after)/distinct_before
  - weight_redirected (accuracy-cost proxy)
Averaged over many random batches. Shows saving is larger at small B.
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
OUT = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-15_v14b_consolidation_batch"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0")
model.eval()
cfg = model.config
E, TOPK, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
print(f"E={E} topk={TOPK} L={L}", flush=True)

PROMPTS = [
    "You are an agent with tools: search(query), read_file(path), run_code(code). Task: find and fix the auth bug. Action: search(\"auth login token\") Observation: found auth/login.py. Action: read_file(\"auth/login.py\") Observation: validate_token doesn't check expiry.",
    "Plan a trip to Tokyo: book flight, reserve hotel, rent car. Step 1: search flights to Tokyo next month. Step 2: compare hotels near city center by price and rating. Step 3: check car rental availability at the airport.",
    "Extract total revenue from JSON: {\"quarters\":[{\"q\":1,\"rev\":1200},{\"q\":2,\"rev\":1500},{\"q\":3,\"rev\":1800}]}. Sum rev across all quarters. Calculating step by step: 1200 + 1500 + 1800 = 4500 total.",
    "Debug this Python code: def divide(a,b): return a/b. Calling divide(10,0) raises ZeroDivisionError. To fix, add a guard: if b equals zero, return None or raise a clear error message. Rewriting the function now.",
    "Summarize the meeting notes: the team agreed to ship the feature next week, assign QA to Alice, and postpone the design review. Key action items: finalize the API, write tests, update the changelog before Friday.",
    "Translate to French and explain grammar: 'The quick brown fox jumps over the lazy dog.' In French this is 'Le renard brun rapide saute par-dessus le chien paresseux.' Note the adjective placement differs.",
]

# Collect per-token per-layer topk indices (each token = one row we can sample into batches)
# Store as: tokens[i] = list over layers of topk_idx[TOPK] and topk_w[TOPK]
all_tok_idx = []   # [Ntok][L][TOPK]
all_tok_w = []
with torch.no_grad():
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").to("cuda:0")
        out = model(**ids, output_router_logits=True)
        T = ids["input_ids"].shape[1]
        layer_idx = []; layer_w = []
        for layer in range(L):
            probs = torch.softmax(out.router_logits[layer].float(), dim=-1)
            tkw, tki = torch.topk(probs, TOPK, dim=-1)
            normw = (tkw / tkw.sum(dim=-1, keepdim=True))
            layer_idx.append(tki.cpu().numpy())   # [T,TOPK]
            layer_w.append(normw.cpu().numpy())
        for t in range(T):
            all_tok_idx.append(np.stack([layer_idx[l][t] for l in range(L)]))  # [L,TOPK]
            all_tok_w.append(np.stack([layer_w[l][t] for l in range(L)]))
Ntok = len(all_tok_idx)
all_tok_idx = np.array(all_tok_idx)  # [Ntok, L, TOPK]
all_tok_w = np.array(all_tok_w)
print(f"collected {Ntok} tokens", flush=True)

rng = np.random.default_rng(0)
def consolidation_for_batch(batch_tok_ids, r):
    """batch_tok_ids: indices into all_tok_idx. Return (transfer_saved, weight_moved) avg over layers."""
    saved=[]; moved=[]
    idx = all_tok_idx[batch_tok_ids]  # [B,L,TOPK]
    w = all_tok_w[batch_tok_ids]
    B = len(batch_tok_ids)
    for layer in range(L):
        before=set(); core=set(); wm=0.0; tw=0.0
        for b in range(B):
            a=idx[b,layer]; ww=w[b,layer]
            before.update(a.tolist())
            core.update(a[:r].tolist())
            tw+=ww.sum()
            for rank in range(r,TOPK): wm+=ww[rank]
        db=len(before); da=len(core)
        saved.append((db-da)/max(db,1)); moved.append(wm/max(tw,1e-9))
    return np.mean(saved), np.mean(moved)

BATCH_SIZES=[4,8,16,32,64]
FRINGE_R=6  # redirect rank>=6 (i.e. the 3 weakest of 8): moderate
NREP=40
print(f"\n== Consolidation (redirect rank>={FRINGE_R}) vs batch size ==")
print(f"{'batch':>6}{'avg_active_experts/layer':>26}{'transfer_saved':>16}{'weight_redirected':>18}")
results=[]
for B in BATCH_SIZES:
    if B>Ntok: B=Ntok
    svs=[]; mvs=[]; acts=[]
    for _ in range(NREP):
        sel=rng.choice(Ntok, size=B, replace=False)
        s,m=consolidation_for_batch(sel, FRINGE_R)
        svs.append(s); mvs.append(m)
        # active experts before (avg over layers)
        idx=all_tok_idx[sel]
        a_layers=[len(set(idx[:,l,:].reshape(-1).tolist())) for l in range(L)]
        acts.append(np.mean(a_layers))
    print(f"{B:>6}{np.mean(acts):>26.0f}{np.mean(svs):>15.1%}{np.mean(mvs):>17.1%}")
    results.append({"batch":B,"avg_active_experts":round(float(np.mean(acts)),1),
                    "transfer_saved":round(float(np.mean(svs)),4),"weight_redirected":round(float(np.mean(mvs)),4)})

json.dump({"E":E,"TOPK":TOPK,"L":L,"fringe_from_rank":FRINGE_R,"results":results},
          open(f"{OUT}/consolidation_vs_batch.json","w"), indent=2)
print(f"\nwrote {OUT}/consolidation_vs_batch.json")
