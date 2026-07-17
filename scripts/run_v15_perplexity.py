#!/usr/bin/env python3
"""v15: REAL perplexity cost of reducing active experts (top-k reduction).

The v14 'weight_redirected' was only a proxy. This measures ACTUAL perplexity
when we use only the top-r experts per token instead of all 8. Top-k reduction
is a clean, well-defined intervention that upper-bounds the cost of the
consolidation idea (dropping the weakest experts is at least as harmful as
redirecting them to a similar active expert).

We patch every Qwen3MoeSparseMoeBlock.top_k and measure perplexity on held-out
agent-style text for r in {8(baseline),7,6,5,4,3,2}.
"""
import json, os, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
OUT = "/home/t-jialianggu/work/MOEresearch/results/2026-07-15_v15_ppl"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0")
model.eval()
cfg = model.config
E, TOPK, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
print(f"E={E} topk={TOPK} L={L}", flush=True)

# Collect all MoE blocks to patch their top_k
moe_blocks = [m for m in model.modules() if isinstance(m, M.Qwen3MoeSparseMoeBlock)]
print(f"found {len(moe_blocks)} MoE blocks", flush=True)

def set_topk(r):
    for b in moe_blocks:
        b.top_k = r

# Held-out agent-style evaluation texts (longer, to get stable perplexity)
EVAL_TEXTS = [
    "You are an autonomous coding agent. The user reports that the login endpoint returns 500 errors intermittently. Your task is to diagnose and fix the issue. First, you search the codebase for the login handler. You find it in api/auth/login.py. Reading the file, you notice the database connection is not properly pooled, causing connection exhaustion under load. You decide to add a connection pool with a maximum of 20 connections and a timeout of 30 seconds. After implementing the fix, you write a unit test that simulates 100 concurrent login requests to verify the pool works correctly. The test passes, confirming the fix resolves the intermittent 500 errors.",
    "The assistant is helping plan a complex software migration. The current system runs on a monolithic architecture and needs to be split into microservices. Step one is to identify the bounded contexts: user management, billing, notifications, and analytics. Step two is to define the API contracts between services using OpenAPI specifications. Step three involves setting up a message queue with Kafka for asynchronous communication. Step four is to migrate the database from a single Postgres instance to per-service databases, ensuring data consistency through the saga pattern. Each step requires careful testing and gradual rollout using feature flags.",
    "To analyze the quarterly financial data, we parse the JSON response from the accounting API. The structure contains an array of transactions, each with fields for date, amount, category, and vendor. We aggregate the amounts by category to compute total spending per department. Marketing spent 45000, engineering spent 120000, and operations spent 38000. The total quarterly expenditure is 203000 dollars, which is 15 percent under the allocated budget of 240000. This surplus can be reallocated to the research and development initiatives planned for next quarter.",
]

def perplexity(text):
    ids = tok(text, return_tensors="pt").to("cuda:0")
    input_ids = ids["input_ids"]
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return math.exp(out.loss.item())

results = []
for r in [TOPK, 7, 6, 5, 4, 3, 2]:
    set_topk(r)
    ppls = [perplexity(t) for t in EVAL_TEXTS]
    avg = sum(ppls) / len(ppls)
    label = "baseline(8)" if r == TOPK else f"top{r}"
    results.append({"topk": r, "avg_perplexity": round(avg, 4), "per_text": [round(p,3) for p in ppls]})
    print(f"{label:>12}: avg PPL = {avg:.4f}", flush=True)

# compute % increase vs baseline
base = results[0]["avg_perplexity"]
print("\n== PPL cost vs experts used ==")
print(f"{'topk':>6}{'avg_PPL':>12}{'PPL_increase':>14}")
for x in results:
    inc = (x["avg_perplexity"]/base - 1) * 100
    x["ppl_increase_pct"] = round(inc, 2)
    print(f"{x['topk']:>6}{x['avg_perplexity']:>12.3f}{inc:>13.2f}%")

json.dump({"E":E,"TOPK":TOPK,"baseline_ppl":base,"results":results},
          open(f"{OUT}/ppl_vs_topk.json","w"), indent=2)
print(f"\nwrote {OUT}/ppl_vs_topk.json")
