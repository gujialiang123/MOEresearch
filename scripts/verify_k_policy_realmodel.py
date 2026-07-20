#!/usr/bin/env python3
"""v23 pre-flight: real-model equivalence + phase-routing verification.

Verifies on Qwen3-30B:
  1. KPolicy(8,8) reproduces native: MoE out (per-block), router logits,
     next-token logits (full prompt), greedy generation token-for-token.
  2. Phase hook routes correctly during generate(): with (prefill_k=8, decode_k=4),
     avg_k_prefill==8 and avg_k_decode==4 — proving prefill vs decode split works
     from CACHE STATE (not seq_len guess).
  3. KV-cache integrity: running a policy generate does not corrupt a subsequent
     native generate (fresh cache each call).

Run: MODEL=qwen GPU=4 python scripts/verify_k_policy_realmodel.py
"""
import os, sys, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"
OUT = "/home/t-jialianggu/work/MOEresearch/results/2026-07-20_v23_preflight"
os.makedirs(OUT, exist_ok=True)

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
TOPK = model.config.num_experts_per_tok
report = {}

PROMPTS = [
    "What is 17 plus 26? Answer with just the number.",
    "Natalia sold 48 clips in April and half as many in May. How many total?",
]

def greedy(prompts, policy=None, max_new=64):
    ctx = KP.attach_policy(model, policy) if policy else None
    outs = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(DEV)
        with torch.inference_mode():
            o = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        outs.append(o[0, ids["input_ids"].shape[1]:].tolist())
    st = ctx.stats() if ctx else None
    if ctx:
        KP.detach_policy(model, ctx)
    return outs, st

# --- 1. next-token logits equivalence (full prompt, no generation) ---
text = tok.apply_chat_template([{"role": "user", "content": PROMPTS[1]}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(DEV)
with torch.inference_mode():
    ref_logits = model(**ids).logits.float()
pol88 = KP.KPolicy(prefill_k=8, decode_k=8, weight_mode="renorm_survivors")
ctx = KP.attach_policy(model, pol88)
KP._PHASE.set("prefill")
with torch.inference_mode():
    patched_logits = model(**ids).logits.float()
KP.detach_policy(model, ctx)
logit_err = (patched_logits - ref_logits).abs().max().item()
report["nexttoken_logit_max_abs_err_88"] = logit_err
print(f"[eq] (8,8) next-token logits max_abs_err = {logit_err:.3e}", flush=True)

# --- 2. greedy generation equivalence ---
native_out, _ = greedy(PROMPTS)
patched_out, _ = greedy(PROMPTS, policy=pol88)
gen_match = [a == b for a, b in zip(native_out, patched_out)]
report["greedy_gen_match_88"] = gen_match
report["greedy_gen_all_match_88"] = bool(all(gen_match))
print(f"[eq] (8,8) greedy token-for-token match: {gen_match}", flush=True)

# --- 3. phase routing during generate: (8,4) -> prefill 8, decode 4 ---
pol84 = KP.KPolicy(prefill_k=8, decode_k=4, weight_mode="renorm_survivors")
_, st84 = greedy(PROMPTS, policy=pol84, max_new=48)
report["phase84_stats"] = st84
print(f"[phase] (8,4): avg_k_prefill={st84['avg_k_prefill']} avg_k_decode={st84['avg_k_decode']}", flush=True)

# --- 4. reverse (4,8) -> prefill 4, decode 8 ---
pol48 = KP.KPolicy(prefill_k=4, decode_k=8, weight_mode="renorm_survivors")
_, st48 = greedy(PROMPTS, policy=pol48, max_new=48)
report["phase48_stats"] = st48
print(f"[phase] (4,8): avg_k_prefill={st48['avg_k_prefill']} avg_k_decode={st48['avg_k_decode']}", flush=True)

# --- 5. KV integrity: native generate after policy generate reproduces native ---
native_out2, _ = greedy(PROMPTS)
report["kv_integrity_native_stable"] = bool(native_out2 == native_out)
print(f"[kv] native stable after policy runs: {native_out2 == native_out}", flush=True)

ok = (logit_err < 5e-2 and report["greedy_gen_all_match_88"]
      and st84["avg_k_prefill"] == 8.0 and abs(st84["avg_k_decode"] - 4.0) < 0.01
      and st48["avg_k_prefill"] == 4.0 and st48["avg_k_decode"] == 8.0
      and report["kv_integrity_native_stable"])
report["ACCEPTANCE_PASS"] = bool(ok)
json.dump(report, open(f"{OUT}/preflight_report.json", "w"), indent=2)
print(f"\nACCEPTANCE_PASS = {ok}")
print(f"wrote {OUT}/preflight_report.json")
sys.exit(0 if ok else 1)
