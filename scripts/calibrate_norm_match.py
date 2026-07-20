#!/usr/bin/env python3
"""Calibrate per-layer norm-match scalars s_{l,K} = E||y8|| / E||yK|| for
calibrated_norm_match (v24 mode D), on a calibration set (GSM8K train).

For each MoE layer l and target K, we measure the mean L2 norm of the MoE branch
output y under native K=8 and under fixed-K=K (no_renorm), then s_{l,K} scales
the no_renorm output to match the K8 norm. Frozen scalars are saved to JSON and
loaded at eval; NO per-token K8 reference is used at eval time.

Run: GPU=6 python scripts/calibrate_norm_match.py --kset 6,4 --n 64
"""
import os, sys, json, argparse
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"
ap = argparse.ArgumentParser()
ap.add_argument("--kset", default="6,4")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--out", default="/home/t-jialianggu/work/MOEresearch/results/2026-07-20_v24_weight_ablation/norm_match_scalars.json")
args = ap.parse_args()
KSET = [int(x) for x in args.kset.split(",")]

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
blocks = [m for m in model.modules() if isinstance(m, M.Qwen3MoeSparseMoeBlock)]
for i, b in enumerate(blocks):
    b._layer_idx = i
TOPK = model.config.num_experts_per_tok

ds = load_dataset("gsm8k", "main", split="train").select(range(args.n))
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."

# capture MoE branch output norm per layer via forward hooks on the native module
norms = {}  # (layer, K) -> [sum_norm, count]

def make_hook(idx):
    def hook(module, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        n = y.float().norm(dim=-1).mean().item()
        key = (idx, CUR_K[0])
        s = norms.setdefault(key, [0.0, 0])
        s[0] += n; s[1] += 1
    return hook

CUR_K = [TOPK]
handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]

# We run fixed-K (no_renorm) so the branch reflects the pruned output; K8 is native.
@torch.inference_mode()
def run_k(k, ids):
    CUR_K[0] = k
    if k == TOPK:
        # native: detach policy
        pol = KP.KPolicy(prefill_k=TOPK, decode_k=TOPK, weight_mode="no_renorm")
    else:
        pol = KP.KPolicy(prefill_k=k, decode_k=k, weight_mode="no_renorm")
    ctx = KP.attach_policy(model, pol)
    KP._PHASE.set("prefill")
    model(ids)
    KP.detach_policy(model, ctx)

for qi, ex in enumerate(ds):
    text = tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(DEV)["input_ids"]
    for k in sorted(set(KSET + [TOPK])):
        run_k(k, ids)
    if (qi + 1) % 16 == 0:
        print(f"  calib {qi+1}/{args.n}", flush=True)

for h in handles:
    h.remove()

# scalars s_{l,K} = meannorm(8) / meannorm(K)
scalars = {}
for (l, k), (s, c) in norms.items():
    if k == TOPK:
        continue
    m8 = norms[(l, TOPK)]
    mean8 = m8[0] / m8[1]
    meanK = s / c
    scalars[f"{l},{k}"] = round(mean8 / meanK, 5) if meanK > 0 else 1.0

json.dump(scalars, open(args.out, "w"), indent=2)
print(f"wrote {len(scalars)} scalars to {args.out}")
# print summary per K
for k in KSET:
    vals = [v for kk, v in scalars.items() if kk.endswith(f",{k}")]
    if vals:
        print(f"K={k}: mean s={sum(vals)/len(vals):.3f} range [{min(vals):.3f},{max(vals):.3f}]")
