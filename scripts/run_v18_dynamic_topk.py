#!/usr/bin/env python3
"""v18: DYNAMIC top-k (confidence-adaptive) vs fixed top-k on GSM8K.

Idea: instead of a fixed number of experts per token, keep experts until the
cumulative (normalized) router probability reaches a threshold tau (top-p style).
Easy tokens naturally use few experts; hard tokens keep more. We sweep tau and
record BOTH accuracy and the realized average-k, so we can plot accuracy vs
average-active-experts and overlay it on v17's fixed-top-k curve. The claim we
test: dynamic reaches the same accuracy at a LOWER average k than fixed.

Implementation: monkeypatch Qwen3MoeSparseMoeBlock.forward to build a per-token
keep-mask from cumulative probability, zero the dropped experts' weights
(mathematically equivalent to not selecting them), and count realized k.
"""
import json, os, re, time, argparse, statistics, types
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
OUT = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-15_v18_dynamic_topk"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max_new", type=int, default=400)
ap.add_argument("--taus", type=str, default="0.70,0.80,0.85,0.90,0.95")
ap.add_argument("--kmin", type=int, default=1)
ap.add_argument("--kmax", type=int, default=8)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
def last_number(s):
    nums = NUM_RE.findall(s)
    return nums[-1].replace(",", "").rstrip(".") if nums else None
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

# ---- dynamic forward (monkeypatch) ----
def dynamic_forward(self, hidden_states):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # (T, E)

    kmax = self._kmax
    rw, selected_experts = torch.topk(routing_weights, kmax, dim=-1)      # (T, kmax) desc
    rw_norm = rw / rw.sum(dim=-1, keepdim=True)                            # normalize over the pool
    cum_before = torch.cumsum(rw_norm, dim=-1) - rw_norm                   # cum prob BEFORE each expert
    keep = cum_before < self._tau                                         # keep until tau reached
    ar = torch.arange(kmax, device=keep.device)
    keep = keep | (ar.unsqueeze(0) < self._kmin)                          # enforce k_min floor
    # record realized k
    self._k_sum += int(keep.sum().item())
    self._tok_count += keep.shape[0]

    rw = rw * keep                                                        # zero dropped experts
    if self.norm_topk_prob:
        rw = rw / rw.sum(dim=-1, keepdim=True)
    routing_weights = rw.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

_orig_forward = M.Qwen3MoeSparseMoeBlock.forward
def enable_dynamic(tau, kmin, kmax):
    for b in moe_blocks:
        b._tau, b._kmin, b._kmax = tau, kmin, kmax
        b._k_sum, b._tok_count = 0, 0
        b.forward = types.MethodType(dynamic_forward, b)
def reset_counters():
    for b in moe_blocks:
        b._k_sum, b._tok_count = 0, 0
def realized_avg_k():
    ks = sum(b._k_sum for b in moe_blocks)
    tc = sum(b._tok_count for b in moe_blocks)
    return ks / tc if tc else 0.0

ds = load_dataset("gsm8k", "main", split="test").select(range(args.limit))
questions = [ex["question"] for ex in ds]
golds = [gold_answer(ex["answer"]) for ex in ds]
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
prompts = [tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}],
                                   tokenize=False, add_generation_prompt=True) for q in questions]

def run_eval():
    preds, gen_tokens = [], []
    reset_counters()
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i+args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        in_len = enc["input_ids"].shape[1]
        new = out[:, in_len:]
        for j in range(new.shape[0]):
            gen = new[j]
            gen_tokens.append(int((gen != tok.pad_token_id).sum()))
            preds.append(last_number(tok.decode(gen, skip_special_tokens=True)))
        print(f"  batch {i//args.batch+1}: {new.shape[0]} q, avg_k so far={realized_avg_k():.3f}", flush=True)
    torch.cuda.synchronize(); wall = time.time() - t0
    correct = sum(1 for p, g in zip(preds, golds) if p is not None and p == g)
    return {"accuracy": round(correct/len(golds), 4), "correct": correct, "n": len(golds),
            "wall_s": round(wall, 1), "avg_k": round(realized_avg_k(), 3),
            "total_gen_tokens": sum(gen_tokens), "tok_per_s": round(sum(gen_tokens)/wall, 1),
            "avg_gen_tokens": round(statistics.mean(gen_tokens), 1)}

taus = [float(x) for x in args.taus.split(",")]
results = []
for tau in taus:
    print(f"\n=== dynamic tau={tau} (kmin={args.kmin}, kmax={args.kmax}) ===", flush=True)
    enable_dynamic(tau, args.kmin, args.kmax)
    res = run_eval(); res["tau"] = tau; res["kmin"] = args.kmin; res["kmax"] = args.kmax
    results.append(res)
    print(json.dumps(res), flush=True)

summary = {"model": MODEL, "E": E, "TOPK": TOPK, "L": L, "limit": args.limit,
           "batch": args.batch, "max_new": args.max_new, "results": results}
json.dump(summary, open(f"{OUT}/dynamic_vs_tau.json", "w"), indent=2)

print("\n== DYNAMIC top-k: accuracy vs realized avg-k ==")
print(f"{'tau':>6}{'avg_k':>8}{'acc':>8}{'wall_s':>9}{'avg_gen':>9}")
for x in results:
    print(f"{x['tau']:>6}{x['avg_k']:>8.3f}{x['accuracy']*100:>7.1f}%{x['wall_s']:>9.1f}{x['avg_gen_tokens']:>9.1f}")
print(f"\nwrote {OUT}/dynamic_vs_tau.json")
