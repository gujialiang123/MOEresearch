#!/usr/bin/env python3
"""v30: Partial-renormalization dose response. Same retained subset, vary only the
renormalization strength beta in {0,.25,.5,.75,1}; does length/termination show a
monotone dose relationship? (prefill K8, decode K in {6,4}, + native K8 baseline).

Run:
  HF_HOME=... CUDA_VISIBLE_DEVICES=4 python scripts/run_v30_partial_renorm.py --stage pilot
"""
import os, sys, json, time, argparse
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _harness as H
from moe_research import k_policy as KP
from moe_research import trace_schema as TS

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v30_partial_renorm")
ap.add_argument("--stage", choices=["smoke", "pilot", "confirmatory"], default="pilot")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--decode_ks", default="6,4")
ap.add_argument("--betas", default="0,0.25,0.5,0.75,1.0")
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
BETAS = [float(x) for x in args.betas.split(",")]

OUT = os.path.join(args.out_root, f"2026-07-20_{args.tag}_{STAGE}")
os.makedirs(OUT, exist_ok=True)

CONFIGS = [("k8_native", 8, 1.0)]
for dk in DKS:
    for b in BETAS:
        CONFIGS.append((f"k{dk}_b{b}", dk, b))

print(f"[{STAGE}] {SPLIT}[{LO}:{HI}], {len(CONFIGS)} configs", flush=True)
model, tok = H.load_model()
H.set_decoder(tok)
blocks, _ = KP._find_moe_blocks(model)
for i, b in enumerate(blocks):
    b._kp_layer_idx = i
HASH_IDS = tok.encode("####", add_special_tokens=False)
EOS = H.eos_set_of(tok)
ds = H.gsm8k(SPLIT, LO, HI)
prompts, golds, questions = H.make_prompts(tok, ds)

TS.write_manifest(OUT, tag=args.tag, model=H.MODEL, env=H.env_dict(),
                  config={"stage": STAGE, "split": SPLIT, "range": [LO, HI],
                          "decode_ks": DKS, "betas": BETAS, "max_new": args.max_new},
                  extra={"sample_ids": list(range(LO, HI))})


def run(name, dk, beta, baseline_seqs):
    path = os.path.join(OUT, f"{name}_raw.jsonl")
    done = set()
    if args.resume and os.path.exists(path):
        for l in open(path):
            try: done.add(json.loads(l)["id"])
            except: pass
    wm = "renorm_survivors" if dk == 8 else "partial_renorm"
    policy = KP.KPolicy(prefill_k=8, decode_k=dk, weight_mode=wm, renorm_beta=beta)
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
                                   extra={"config": name, "decode_k": dk, "beta": beta,
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
for name, dk, beta in CONFIGS:
    print(f"\n=== {name} (dk={dk} beta={beta}) ===", flush=True)
    rows = run(name, dk, beta, baseline_seqs)
    if name == "k8_native" and baseline_seqs is None:
        baseline_seqs = {r["id"]: r["gen_token_ids"] for r in rows}
    s = H.summarize(name, rows, args.max_new)
    s["decode_k"] = dk; s["beta"] = beta
    summaries[name] = s
    json.dump(s, open(os.path.join(OUT, f"{name}_summary.json"), "w"), indent=2)
    print(json.dumps(s), flush=True)

json.dump({"tag": args.tag, "stage": STAGE, "summaries": summaries},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("\n== v30 partial-renorm dose ==")
for nm, s in summaries.items():
    print(f"{nm:12s} dk={s['decode_k']} b={s['beta']:.2f} len={s['len_mean']:6.1f} "
          f"rmst={s['rmst_512']:6.1f} acc={s['acc_strict']*100:5.1f}% hitmax={s['hit_max']*100:.1f}%")
print(f"\nwrote {OUT}/summary.json")
