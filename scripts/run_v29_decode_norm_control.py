#!/usr/bin/env python3
"""v29: Decode-specific norm control. Does matching the AVERAGE decode MoE branch
scale (calibrated on real decode tokens) remove the length effect?

9 configs (fixed prefill K8): native K8; decode K6/K4 in {no_renorm, full_renorm,
prefill-calibrated norm-match, decode-calibrated norm-match}. The two norm-match arms
share the SAME retained subset and differ only in the frozen scalar (prefill- vs
decode-calibrated), isolating the calibration-domain confound.

Also runs a HELD-OUT realized-ratio diagnostic (dual-branch on dev prompts with frozen
scalars) so a control is only called "norm matched" if its realized mean ratio is in
[0.95, 1.05].

Run (smoke/pilot/confirmatory via --split/--range):
  HF_HOME=... CUDA_VISIBLE_DEVICES=3 python scripts/run_v29_decode_norm_control.py \
     --scalars results/2026-07-20_v29_decode_norm/calibration/norm_match_scalars.json \
     --stage pilot
"""
import os, sys, json, time, argparse
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _harness as H
from moe_research import k_policy as KP
from moe_research import trace_schema as TS
from moe_research.decode_norm_calib import DecodeNormCalibrator

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v29_decode_norm")
ap.add_argument("--scalars", required=True, help="norm_match_scalars.json from calibrate_v29")
ap.add_argument("--stage", choices=["smoke", "pilot", "confirmatory"], default="pilot")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--decode_ks", default="6,4")
ap.add_argument("--out_root", default="/home/t-jialianggu/work/MOEresearch/results")
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()

STAGE = args.stage
if STAGE == "smoke":
    SPLIT, LO, HI = "train", *H.SMOKE_RANGE
elif STAGE == "pilot":
    SPLIT, LO, HI = "train", *H.DEV_RANGE
else:
    SPLIT, LO, HI = "test", *H.TEST_RANGE
DKS = [int(x) for x in args.decode_ks.split(",")]

SC = json.load(open(args.scalars))
DECODE_SCALARS = SC["decode_scalars"]
PREFILL_SCALARS = SC["prefill_scalars"]

OUT = os.path.join(args.out_root, f"2026-07-20_{args.tag}_{STAGE}")
os.makedirs(OUT, exist_ok=True)

# config: (name, decode_k, weight_mode, scalars)
CONFIGS = [("k8_native", 8, "renorm_survivors", None)]
for dk in DKS:
    CONFIGS += [
        (f"k{dk}_no_renorm", dk, "no_renorm", None),
        (f"k{dk}_full_renorm", dk, "renorm_survivors", None),
        (f"k{dk}_prefill_normmatch", dk, "calibrated_norm_match", PREFILL_SCALARS),
        (f"k{dk}_decode_normmatch", dk, "decode_norm_match", DECODE_SCALARS),
    ]

print(f"[{STAGE}] {SPLIT}[{LO}:{HI}], {len(CONFIGS)} configs", flush=True)
model, tok = H.load_model()
H.set_decoder(tok)
blocks, _ = KP._find_moe_blocks(model)
for i, b in enumerate(blocks):
    b._kp_layer_idx = i
TOPK = model.config.num_experts_per_tok
HASH_IDS = tok.encode("####", add_special_tokens=False)
EOS = H.eos_set_of(tok)

ds = H.gsm8k(SPLIT, LO, HI)
prompts, golds, questions = H.make_prompts(tok, ds)

TS.write_manifest(OUT, tag=args.tag, model=H.MODEL, env=H.env_dict(),
                  config={"stage": STAGE, "split": SPLIT, "range": [LO, HI],
                          "decode_ks": DKS, "max_new": args.max_new,
                          "scalars_src": args.scalars},
                  extra={"configs": [c[0] for c in CONFIGS],
                         "sample_ids": list(range(LO, HI))})


def run(name, dk, wm, scalars, baseline_seqs):
    path = os.path.join(OUT, f"{name}_raw.jsonl")
    done = set()
    if args.resume and os.path.exists(path):
        for l in open(path):
            try: done.add(json.loads(l)["id"])
            except: pass
    policy = KP.KPolicy(prefill_k=8, decode_k=dk, weight_mode=wm,
                        calib_scalars=scalars or {})
    ctx = KP.attach_policy(model, policy)
    fout = open(path, "a")
    todo = [i for i in range(len(prompts)) if (LO + i) not in done]
    for bs in range(0, len(todo), args.batch):
        idxs = todo[bs:bs+args.batch]
        enc = tok([prompts[i] for i in idxs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(H.DEV)
        ctx.reset_counters()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        dt = (time.time() - t0) * 1000
        in_len = enc["input_ids"].shape[1]
        new = out[:, in_len:].tolist()
        st = ctx.stats()
        for j, seq in enumerate(new):
            gi = LO + idxs[j]
            base = baseline_seqs.get(gi) if baseline_seqs else None
            row, gen = H.build_row(gi, seq, golds[idxs[j]], EOS, HASH_IDS, args.max_new,
                                   baseline_seq=base,
                                   extra={"config": name, "decode_k": dk, "weight_mode": wm,
                                          "avg_k_decode": st["avg_k_decode"],
                                          "wall_time_ms": round(dt/len(idxs), 1),
                                          "gen_token_ids": gen})
            fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(f"  [{name}] {min(bs+args.batch,len(todo))}/{len(todo)} k_dec={st['avg_k_decode']}", flush=True)
    fout.close(); KP.detach_policy(model, ctx)
    return [json.loads(l) for l in open(path)]


baseline_seqs = None
summaries = {}
for name, dk, wm, scalars in CONFIGS:
    print(f"\n=== {name} (dk={dk} {wm}) ===", flush=True)
    rows = run(name, dk, wm, scalars, baseline_seqs)
    if name == "k8_native" and baseline_seqs is None:
        baseline_seqs = {r["id"]: r["gen_token_ids"] for r in rows}
    s = H.summarize(name, rows, args.max_new)
    summaries[name] = s
    json.dump(s, open(os.path.join(OUT, f"{name}_summary.json"), "w"), indent=2)
    print(json.dumps(s), flush=True)

# ---- realized-ratio diagnostic on held-out dev tokens (frozen scalars) ----
realized = {}
try:
    dvds = H.gsm8k("train", *H.SMOKE_RANGE)  # 32 dev prompts, cheap
    dvp = [ex["question"] for ex in dvds]
    for label, scal in (("decode_calibrated", DECODE_SCALARS), ("prefill_calibrated", PREFILL_SCALARS)):
        cal = DecodeNormCalibrator(blocks, top_k=TOPK, k_targets=DKS)
        cal.record(model, tok, dvp, H.DEV, max_new=256)
        realized[label] = cal.realized_ratio(scal, "decode")
    print(f"\nrealized ratio on held-out dev (target 0.95-1.05): {realized}", flush=True)
except Exception as e:
    realized = {"error": str(e)}
    print(f"realized-ratio diagnostic failed: {e}", flush=True)

json.dump({"tag": args.tag, "stage": STAGE, "summaries": summaries,
           "realized_ratio_dev": realized},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("\n== v29 decode-norm control ==")
for nm, s in summaries.items():
    print(f"{nm:24s} len={s['len_mean']:6.1f} rmst={s['rmst_512']:6.1f} "
          f"acc={s['acc_strict']*100:5.1f}% noEOS={s['no_eos']*100:.1f}% hitmax={s['hit_max']*100:.1f}%")
print(f"\nwrote {OUT}/summary.json")
