#!/usr/bin/env python3
"""v29 calibration: DECODE-specific norm-match scalars via same-hidden dual-branch.

Runs native K8 greedy on GSM8K train[0:128], teacher-forces the full sequences, and
estimates per-layer s(l,K) = E_decode[||y8||]/E_decode[||yK_no_renorm||] separately for
PREFILL and DECODE positions (+ decode-position bins). Also reports the realized ratio
on the calibration tokens (== 1.0 by construction) and, if --dev is set, a HELD-OUT dev
realized ratio using the FROZEN scalars.

Run:
  HF_HOME=/home/t-jialianggu/work/EndtoEnd-auto-optimization/.hf_cache \
  CUDA_VISIBLE_DEVICES=3 python scripts/calibrate_v29_decode_norm.py --ks 6,4 --n 128
"""
import os, sys, json, argparse
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _harness as H
from moe_research import k_policy as KP
from moe_research.decode_norm_calib import DecodeNormCalibrator
from moe_research import trace_schema as TS

ap = argparse.ArgumentParser()
ap.add_argument("--ks", default="6,4")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--out", default="/home/t-jialianggu/work/MOEresearch/results/2026-07-20_v29_decode_norm/calibration")
args = ap.parse_args()
KS = [int(x) for x in args.ks.split(",")]
os.makedirs(args.out, exist_ok=True)

print("loading model...", flush=True)
model, tok = H.load_model()
H.set_decoder(tok)
blocks, _ = KP._find_moe_blocks(model)
for i, b in enumerate(blocks):
    b._kp_layer_idx = i
TOPK = model.config.num_experts_per_tok

lo, hi = H.CALIB_RANGE[0], H.CALIB_RANGE[0] + args.n
ds = H.gsm8k("train", lo, hi)
prompts = [ex["question"] for ex in ds]
print(f"calibrating on train[{lo}:{hi}] ({len(prompts)} prompts), K={KS}", flush=True)

cal = DecodeNormCalibrator(blocks, top_k=TOPK, k_targets=KS)
cal.record(model, tok, prompts, H.DEV, max_new=args.max_new)

decode_scalars = cal.scalars("decode")
prefill_scalars = cal.scalars("prefill")
realized = cal.realized_ratio(decode_scalars, "decode")
print(f"decode realized mean ratio (should be ~1.0): {realized}", flush=True)

d = cal.save(os.path.join(args.out, "norm_match_scalars.json"),
             extra={"calib_split": f"train[{lo}:{hi}]", "realized_ratio_calib": realized})
json.dump({"decode_scalars": decode_scalars}, open(os.path.join(args.out, "norm_match_decode_scalars.json"), "w"), indent=2)
json.dump({"prefill_scalars": prefill_scalars}, open(os.path.join(args.out, "norm_match_prefill_scalars.json"), "w"), indent=2)

TS.write_manifest(args.out, tag="v29_decode_norm_calib", model=H.MODEL, env=H.env_dict(),
                  config={"ks": KS, "n": args.n, "max_new": args.max_new,
                          "calib_range": [lo, hi]},
                  extra={"realized_ratio_calib": realized})
# quick per-layer summary of the decode scalar magnitude
import statistics as _s
for k in KS:
    vals = [v for key, v in decode_scalars.items() if key.endswith(f",{k}")]
    pv = [v for key, v in prefill_scalars.items() if key.endswith(f",{k}")]
    if vals:
        print(f"K={k}: decode s mean={_s.mean(vals):.4f} [min {min(vals):.3f}, max {max(vals):.3f}]; "
              f"prefill s mean={_s.mean(pv):.4f}" if pv else f"K={k}: decode s mean={_s.mean(vals):.4f}", flush=True)
print(f"\nwrote {args.out}/norm_match_scalars.json")
