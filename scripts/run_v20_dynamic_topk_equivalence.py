#!/usr/bin/env python3
"""v20 equivalence check on the REAL Qwen3-30B MoE (P0.3 acceptance #1).

Proves the dynamic monkeypatch reproduces the native model when nothing is
pruned (tau=1.0, kmin=kmax=8), on:
  (a) router logits + MoE-output numerics on fixed hidden states (per-block), and
  (b) greedy generation token-for-token on a few prompts.

Also runs a tiny dynamic config (tau=0.7) to confirm the physical-skip path
produces a realized avg_k_decode < 8 without crashing. This is a SMALL check,
not a sweep.

Run: MODEL=qwen GPU=6 python scripts/run_v20_dynamic_topk_equivalence.py
"""
import os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
import dynamic_topk_utils as U
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"
OUT = "/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-16_v20_equivalence"
os.makedirs(OUT, exist_ok=True)

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
blocks = [m for m in model.modules() if isinstance(m, M.Qwen3MoeSparseMoeBlock)]
E, TOPK, L = model.config.num_experts, model.config.num_experts_per_tok, model.config.num_hidden_layers
print(f"E={E} topk={TOPK} L={L}, {len(blocks)} MoE blocks", flush=True)

report = {"model": MODEL, "E": E, "topk": TOPK, "L": L}

# ---- (a) per-block numeric equivalence on fixed hidden states ----
torch.manual_seed(0)
hidden = torch.randn(1, 6, model.config.hidden_size, dtype=torch.bfloat16, device=DEV)
orig_forward = M.Qwen3MoeSparseMoeBlock.forward

def block_out(block, patched=None):
    if patched is not None:
        import types
        block._k_sum_prefill = torch.zeros((), device=DEV, dtype=torch.long)
        block._k_sum_decode = torch.zeros((), device=DEV, dtype=torch.long)
        block._tok_prefill = 0; block._tok_decode = 0
        block.forward = types.MethodType(patched, block)
    with torch.inference_mode():
        out, logits = block.forward(hidden)
    if patched is not None:
        block.forward = types.MethodType(orig_forward, block)
    return out, logits

b = blocks[0]
ref_out, ref_logits = block_out(b, patched=None)
for renorm in U.RENORM_MODES:
    fwd = U.make_dynamic_forward("top_p_within_topk", 1.0, TOPK, TOPK, phase="all",
                                 renorm=renorm, benchmark_mode=True)
    out, logits = block_out(b, patched=fwd)
    max_abs = (out.float() - ref_out.float()).abs().max().item()
    l2 = ((out.float() - ref_out.float()).norm() / ref_out.float().norm().clamp_min(1e-9)).item()
    logit_err = (logits.float() - ref_logits.float()).abs().max().item()
    report[f"block0_keepall_{renorm}"] = {
        "moe_out_max_abs_err": max_abs, "moe_out_rel_l2": l2, "router_logit_max_abs_err": logit_err}
    print(f"[keep-all {renorm}] MoE max_abs={max_abs:.2e} rel_L2={l2:.2e} logit_err={logit_err:.2e}", flush=True)

# ---- (b) greedy generation equivalence (native vs keep-all patch) ----
PROMPTS = [
    "What is 17 plus 26? Answer with just the number.",
    "Natalia sold 48 clips in April and half as many in May. How many total?",
]
def gen(prompts, patched=None):
    if patched is not None:
        import types
        for blk in blocks:
            blk._k_sum_prefill = torch.zeros((), device=DEV, dtype=torch.long)
            blk._k_sum_decode = torch.zeros((), device=DEV, dtype=torch.long)
            blk._tok_prefill = 0; blk._tok_decode = 0
            blk.forward = types.MethodType(patched, blk)
    outs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(DEV)
        with torch.inference_mode():
            o = model.generate(**ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
        outs.append(o[0, ids["input_ids"].shape[1]:].tolist())
    if patched is not None:
        import types
        for blk in blocks:
            blk.forward = types.MethodType(orig_forward, blk)
    return outs

native = gen(PROMPTS, patched=None)
keepall_fwd = U.make_dynamic_forward("top_p_within_topk", 1.0, TOPK, TOPK, phase="all",
                                     renorm="renorm_survivors", benchmark_mode=True)
patched = gen(PROMPTS, patched=keepall_fwd)
gen_match = [a == b for a, b in zip(native, patched)]
report["greedy_generation_token_match"] = gen_match
report["greedy_generation_all_match"] = bool(all(gen_match))
print(f"[greedy keep-all] token-for-token match: {gen_match}", flush=True)

# ---- (c) tiny dynamic run confirms physical skip yields avg_k_decode < 8 ----
ctrl = U.DynamicKController(blocks, M.Qwen3MoeSparseMoeBlock)
ctrl.enable("top_p_within_topk", 0.7, 1, TOPK, phase="decode_only",
            renorm="renorm_survivors", benchmark_mode=False)
ids = tok(tok.apply_chat_template([{"role": "user", "content": PROMPTS[1]}],
          tokenize=False, add_generation_prompt=True), return_tensors="pt").to(DEV)
with torch.inference_mode():
    model.generate(**ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
st = ctrl.stats()
ctrl.disable()
report["tiny_dynamic_tau0.7"] = st
print(f"[dynamic tau=0.7 decode_only] stats: {st}", flush=True)

json.dump(report, open(f"{OUT}/equivalence_report.json", "w"), indent=2)
ok = (report["greedy_generation_all_match"]
      and all(report[f"block0_keepall_{r}"]["moe_out_max_abs_err"] < 5e-2 for r in U.RENORM_MODES)
      and (st["avg_k_decode"] is not None and st["avg_k_decode"] < TOPK))
report["ACCEPTANCE_PASS"] = bool(ok)
json.dump(report, open(f"{OUT}/equivalence_report.json", "w"), indent=2)
print(f"\nACCEPTANCE_PASS = {ok}")
print(f"wrote {OUT}/equivalence_report.json")
sys.exit(0 if ok else 1)
