#!/usr/bin/env python3
"""v31: Gain controls. Decompose the full-renorm effect into (A) average scale,
(B) gain variance/tail, (C) gain-to-router-state correspondence. Fixed prefill K8,
decode K4, same retained top-4 subset; compare 8 aggregation modes.

Gains g=1/r are calibrated on DECODE tokens of train[0:128] (routing-only). shuffled
gain draws from the calibration pool matched by (layer, decode-position bin) — never
from the eval request. clipped gains use layer-wise calibration P90/P95.

Run:
  HF_HOME=... CUDA_VISIBLE_DEVICES=4 python scripts/run_v31_gain_controls.py --stage pilot
"""
import os, sys, json, time, argparse, random
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _harness as H
from moe_research import k_policy as KP
from moe_research import trace_schema as TS
from moe_research.gain_calibration import GainCalibrator
from moe_research.decode_norm_calib import load_scalars

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v31_gain_controls")
ap.add_argument("--stage", choices=["smoke", "pilot", "confirmatory"], default="pilot")
ap.add_argument("--decode_k", type=int, default=4)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--calib_n", type=int, default=128)
ap.add_argument("--decode_scalars", default="/home/t-jialianggu/work/MOEresearch/results/2026-07-20_v29_decode_norm/calibration/norm_match_scalars.json")
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
DK = args.decode_k
OUT = os.path.join(args.out_root, f"2026-07-20_{args.tag}_{STAGE}")
os.makedirs(OUT, exist_ok=True)

print(f"[{STAGE}] {SPLIT}[{LO}:{HI}] decode_k={DK}", flush=True)
model, tok = H.load_model()
H.set_decoder(tok)
blocks, _ = KP._find_moe_blocks(model)
for i, b in enumerate(blocks):
    b._kp_layer_idx = i
TOPK = model.config.num_experts_per_tok
HASH_IDS = tok.encode("####", add_special_tokens=False)
EOS = H.eos_set_of(tok)

# ---- gain calibration on decode tokens of train[0:calib_n] ----
print(f"gain calibration on train[0:{args.calib_n}] decode tokens...", flush=True)
cds = H.gsm8k("train", 0, args.calib_n)
cprompts = [ex["question"] for ex in cds]
gcal = GainCalibrator(blocks, low_k=DK)
gcal.record_decode(model, tok, cprompts, H.DEV, max_new=args.max_new)
gcal.save(os.path.join(OUT, "gain_calibration.json"))
LAYER_MEAN = gcal.scalars()                       # fixed_gain
BIN_MEAN = gcal.position_bin_means()              # position_bin_gain
QUANT = gcal.quantiles((0.90, 0.95))
shuffled_provider = gcal.make_shuffled_provider(seed=0)
# decode norm-match scalars (from v29 calibration if available)
try:
    DECODE_NM = load_scalars(args.decode_scalars, "decode_scalars")
except Exception:
    DECODE_NM = {}
print(f"gain q90={QUANT.get('0.9')} q95={QUANT.get('0.95')}; layer_mean keys={len(LAYER_MEAN)}", flush=True)

ds = H.gsm8k(SPLIT, LO, HI)
prompts, golds, questions = H.make_prompts(tok, ds)

# config: (name, weight_mode, kwargs)
CONFIGS = [
    ("k8_native", "renorm_survivors", {}),
    ("no_renorm", "no_renorm", {}),
    ("full_renorm", "renorm_survivors", {}),
    ("decode_normmatch", "decode_norm_match", {"calib_scalars": DECODE_NM}),
    ("fixed_layer_gain", "fixed_gain", {"calib_scalars": LAYER_MEAN}),
    ("position_bin_gain", "position_bin_gain", {"calib_scalars": BIN_MEAN}),
    ("shuffled_gain", "shuffled_gain", {"gain_provider": shuffled_provider}),
    ("clipped_q90", "clipped_gain", {"gain_clip": QUANT.get("0.9", 10.0)}),
    ("clipped_q95", "clipped_gain", {"gain_clip": QUANT.get("0.95", 10.0)}),
]

TS.write_manifest(OUT, tag=args.tag, model=H.MODEL, env=H.env_dict(),
                  config={"stage": STAGE, "split": SPLIT, "range": [LO, HI],
                          "decode_k": DK, "calib_n": args.calib_n, "max_new": args.max_new,
                          "quantiles": QUANT},
                  extra={"configs": [c[0] for c in CONFIGS], "sample_ids": list(range(LO, HI))})


def run(name, wm, kwargs, baseline_seqs):
    path = os.path.join(OUT, f"{name}_raw.jsonl")
    done = set()
    if args.resume and os.path.exists(path):
        for l in open(path):
            try: done.add(json.loads(l)["id"])
            except: pass
    dk = 8 if name == "k8_native" else DK
    policy = KP.KPolicy(prefill_k=8, decode_k=dk, weight_mode=wm, **kwargs)
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
                                          "wall_time_ms": round(dt/len(idxs), 1),})
            fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(f"  [{name}] {min(bs+args.batch,len(todo))}/{len(todo)} k_dec={st['avg_k_decode']}", flush=True)
    fout.close(); KP.detach_policy(model, ctx)
    return [json.loads(l) for l in open(path)]


baseline_seqs = None
summaries = {}
for name, wm, kwargs in CONFIGS:
    print(f"\n=== {name} ({wm}) ===", flush=True)
    rows = run(name, wm, kwargs, baseline_seqs)
    if name == "k8_native" and baseline_seqs is None:
        baseline_seqs = {r["id"]: r["gen_token_ids"] for r in rows}
    s = H.summarize(name, rows, args.max_new)
    summaries[name] = s
    json.dump(s, open(os.path.join(OUT, f"{name}_summary.json"), "w"), indent=2)
    print(json.dumps(s), flush=True)

json.dump({"tag": args.tag, "stage": STAGE, "decode_k": DK, "summaries": summaries,
           "quantiles": QUANT},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("\n== v31 gain controls (decode K%d) ==" % DK)
for nm, s in summaries.items():
    print(f"{nm:20s} len={s['len_mean']:6.1f} rmst={s['rmst_512']:6.1f} "
          f"acc={s['acc_strict']*100:5.1f}% hitmax={s['hit_max']*100:.1f}%")
print(f"\nwrote {OUT}/summary.json")
