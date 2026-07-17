#!/usr/bin/env python3
"""v17: GSM8K accuracy + timing vs active experts (top-k reduction).

For each r in {8(baseline),7,6,5,4} we patch every Qwen3MoeSparseMoeBlock.top_k
and run GSM8K (greedy, batched CoT generation), recording:
  - accuracy (flexible last-number extraction vs gold after '####')
  - wall time, generated tokens, throughput (tok/s), per-question latency dist.

This turns v15's perplexity proxy into a real downstream-task cost curve, and
also captures where the *time* goes so we can see the accuracy/speed tradeoff.
"""
import json, os, re, time, argparse, statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
OUT = "/home/t-jialianggu/work/MOEresearch/results/2026-07-15_v17_gsm8k_topk"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--topks", type=str, default="8,7,6,5,4")
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
def last_number(s):
    nums = NUM_RE.findall(s)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")

def gold_answer(a):
    return a.split("####")[-1].strip().replace(",", "")

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0")
model.eval()
cfg = model.config
E, TOPK, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
print(f"E={E} topk={TOPK} L={L}", flush=True)

moe_blocks = [m for m in model.modules() if isinstance(m, M.Qwen3MoeSparseMoeBlock)]
def set_topk(r):
    for b in moe_blocks:
        b.top_k = r

ds = load_dataset("gsm8k", "main", split="test").select(range(args.limit))
questions = [ex["question"] for ex in ds]
golds = [gold_answer(ex["answer"]) for ex in ds]

# Build prompts once (chat template, non-thinking instruct)
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
prompts = []
for q in questions:
    msgs = [{"role": "user", "content": q + SUFFIX}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

def run_topk(r):
    set_topk(r)
    preds, per_q_time, gen_tokens = [], [], []
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i+args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        torch.cuda.synchronize(); tb = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize(); dt = time.time() - tb
        in_len = enc["input_ids"].shape[1]
        new = out[:, in_len:]
        for j in range(new.shape[0]):
            gen = new[j]
            n_tok = int((gen != tok.pad_token_id).sum())
            text = tok.decode(gen, skip_special_tokens=True)
            preds.append(last_number(text))
            gen_tokens.append(n_tok)
            per_q_time.append(dt / new.shape[0])  # amortized within batch
        print(f"  topk={r} batch {i//args.batch+1}: {dt:.1f}s ({new.shape[0]} q)", flush=True)
    torch.cuda.synchronize()
    wall = time.time() - t0
    correct = sum(1 for p, g in zip(preds, golds) if p is not None and p == g)
    acc = correct / len(golds)
    tot_tok = sum(gen_tokens)
    return {
        "topk": r,
        "accuracy": round(acc, 4),
        "correct": correct,
        "n": len(golds),
        "wall_s": round(wall, 1),
        "total_gen_tokens": tot_tok,
        "tok_per_s": round(tot_tok / wall, 1),
        "avg_gen_tokens": round(statistics.mean(gen_tokens), 1),
        "median_gen_tokens": int(statistics.median(gen_tokens)),
        "p90_gen_tokens": int(sorted(gen_tokens)[int(0.9*len(gen_tokens))-1]),
        "s_per_q": round(wall / len(golds), 3),
    }

topks = [int(x) for x in args.topks.split(",")]
results = []
for r in topks:
    print(f"\n=== topk={r} ===", flush=True)
    res = run_topk(r)
    results.append(res)
    print(json.dumps(res), flush=True)

base = next((x for x in results if x["topk"] == TOPK), results[0])
for x in results:
    x["acc_drop_pct_pts"] = round((base["accuracy"] - x["accuracy"]) * 100, 2)
    x["speedup_vs_base"] = round(base["wall_s"] / x["wall_s"], 3)

summary = {"model": MODEL, "E": E, "TOPK": TOPK, "L": L, "limit": args.limit,
           "batch": args.batch, "max_new": args.max_new, "results": results}
json.dump(summary, open(f"{OUT}/gsm8k_vs_topk.json", "w"), indent=2)

print("\n== GSM8K accuracy + time vs topk ==")
print(f"{'topk':>5}{'acc':>8}{'drop_pp':>9}{'wall_s':>9}{'tok/s':>9}{'avg_tok':>9}{'speedup':>9}")
for x in results:
    print(f"{x['topk']:>5}{x['accuracy']*100:>7.1f}%{x['acc_drop_pct_pts']:>8.1f}"
          f"{x['wall_s']:>9.1f}{x['tok_per_s']:>9.1f}{x['avg_gen_tokens']:>9.1f}{x['speedup_vs_base']:>8.2f}x")
print(f"\nwrote {OUT}/gsm8k_vs_topk.json")
