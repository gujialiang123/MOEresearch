#!/usr/bin/env python3
"""v30: Gain controls — is the length effect from AVERAGE branch scale or from
TOKEN-CONDITIONED gain (1/r tied to the current router state)?

Fixed decode_k=4, prefill=8, greedy. Compare gain modes:
  no_renorm          g=1
  true_token_gain    g=1/r_{t,l}          (== full renorm_survivors)
  layer_mean_gain    frozen E[g_l]        (avg scale, from calibration)
  shuffled_gain      g sampled from matching (layer,pos-bin) pool (breaks token corr.)
  clipped_gain_q90   g=min(1/r, q90)
  clipped_gain_q95   g=min(1/r, q95)
  norm_match         existing calibrated_norm_match

Calibration (layer-mean, quantiles, shuffle pool) is computed on GSM8K TRAIN only.
Saves raw.jsonl per mode + summary + manifest.

Run: GPU=6 python scripts/run_v30_gain_controls.py --limit 200 --calib_n 64
"""
import os, sys, json, time, argparse, statistics
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP
from moe_research import answer_parsing as AP
from moe_research import gain_calibration as GC
from moe_research import trace_schema as TS
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v30_gain_controls")
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--calib_n", type=int, default=64)
ap.add_argument("--decode_k", type=int, default=4)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--out_root", default="/home/t-jialianggu/work/MOEresearch/results")
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()
OUT = os.path.join(args.out_root, f"2026-07-20_{args.tag}")
os.makedirs(OUT, exist_ok=True)
DK = args.decode_k

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
blocks = [m for m in model.modules() if isinstance(m, M.Qwen3MoeSparseMoeBlock)]
for i, b in enumerate(blocks):
    b._layer_idx = i
TOPK = model.config.num_experts_per_tok

# ---- calibration on TRAIN split only ----
print(f"calibrating gains on {args.calib_n} TRAIN prompts (low_k={DK})...", flush=True)
train = load_dataset("gsm8k", "main", split="train").select(range(args.calib_n))
cal = GC.GainCalibrator(blocks, low_k=DK, norm_topk_prob=model.config.norm_topk_prob)
cal.record(model, tok, [ex["question"] for ex in train], DEV)
cal.save(os.path.join(OUT, "gain_calibration.json"))
layer_mean = cal.scalars()            # "l,k" -> mean gain
quants = cal.quantiles()              # q90/q95/q99
shuffle_provider = cal.make_shuffled_provider(seed=0)
print(f"  calibrated: {len(layer_mean)} layers, q90={quants['0.9']} q95={quants['0.95']}", flush=True)

# ---- eval configs on TEST ----
ds = load_dataset("gsm8k", "main", split="test").select(range(args.limit))
golds = [AP.parse_gold(ex["answer"]) for ex in ds]
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
prompts = [tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
           tokenize=False, add_generation_prompt=True) for ex in ds]
eos_ids = tok.eos_token_id
eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
HASH_IDS = tok.encode("####", add_special_tokens=False)

def find_subseq(seq, sub):
    for i in range(len(seq)-len(sub)+1):
        if seq[i:i+len(sub)] == sub: return i
    return -1
def rep_frac(seq, n):
    if len(seq) < n: return 0.0
    g=[tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]; return round(1-len(set(g))/len(g),4)

def make_policy(mode):
    if mode == "no_renorm":
        return KP.KPolicy(prefill_k=8, decode_k=DK, weight_mode="no_renorm")
    if mode == "true_token_gain":
        return KP.KPolicy(prefill_k=8, decode_k=DK, weight_mode="renorm_survivors")
    if mode == "layer_mean_gain":
        return KP.KPolicy(prefill_k=8, decode_k=DK, weight_mode="fixed_gain", calib_scalars=layer_mean)
    if mode == "shuffled_gain":
        return KP.KPolicy(prefill_k=8, decode_k=DK, weight_mode="shuffled_gain", gain_provider=shuffle_provider)
    if mode == "clipped_gain_q90":
        return KP.KPolicy(prefill_k=8, decode_k=DK, weight_mode="clipped_gain", gain_clip=quants["0.9"])
    if mode == "clipped_gain_q95":
        return KP.KPolicy(prefill_k=8, decode_k=DK, weight_mode="clipped_gain", gain_clip=quants["0.95"])
    if mode == "k8_native":
        return KP.KPolicy(prefill_k=8, decode_k=8, weight_mode="renorm_survivors")
    raise ValueError(mode)

MODES = ["k8_native", "no_renorm", "true_token_gain", "layer_mean_gain",
         "shuffled_gain", "clipped_gain_q90", "clipped_gain_q95"]

def run(mode, baseline_seqs):
    path = os.path.join(OUT, f"{mode}_raw.jsonl")
    done = set()
    if args.resume and os.path.exists(path):
        for l in open(path):
            try: done.add(json.loads(l)["id"])
            except: pass
    policy = make_policy(mode)
    ctx = KP.attach_policy(model, policy)
    fout = open(path, "a"); todo=[i for i in range(len(prompts)) if i not in done]
    for bs in range(0, len(todo), args.batch):
        idxs = todo[bs:bs+args.batch]
        enc = tok([prompts[i] for i in idxs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        ctx.reset_counters()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        in_len = enc["input_ids"].shape[1]; new = out[:, in_len:].tolist(); st = ctx.stats()
        for j, seq in enumerate(new):
            gi = idxs[j]; glen=len(seq); eos_pos=-1
            for pos,t in enumerate(seq):
                if t in eos_set: glen=pos+1; eos_pos=pos; break
            gen=seq[:glen]; text=tok.decode(gen, skip_special_tokens=True)
            ps,ss=AP.parse_strict(text); pt,ts=AP.parse_tolerant(text); hp=find_subseq(gen,HASH_IDS)
            fdiv=None
            if baseline_seqs and gi in baseline_seqs:
                b=baseline_seqs[gi]; fdiv=-1
                for p in range(min(len(b),len(gen))):
                    if b[p]!=gen[p]: fdiv=p; break
                else: fdiv=min(len(b),len(gen)) if len(b)!=len(gen) else -1
            row={"id":gi,"gold":golds[gi],"mode":mode,
                 "pred_strict":ps,"correct_strict":(ps==golds[gi] and ps is not None),
                 "pred_tol":pt,"correct_tol":(pt==golds[gi] and pt is not None),
                 "gen_len":glen,"eos_pos":eos_pos,"hit_max":glen>=args.max_new,"hash_tok_pos":hp,
                 "L_to_marker":hp if hp>=0 else None,"rep_4gram":rep_frac(gen,4),"first_div":fdiv,
                 "avg_k_decode":st["avg_k_decode"],"gen_token_ids":gen,"text":text}
            fout.write(json.dumps(row)+"\n")
        fout.flush(); print(f"  [{mode}] {min(bs+args.batch,len(todo))}/{len(todo)} k={st['avg_k_decode']}", flush=True)
    fout.close(); KP.detach_policy(model, ctx)
    return [json.loads(l) for l in open(path)]

def summarize(mode, rows):
    n=len(rows); lens=[r["gen_len"] for r in rows]; wh=[r for r in rows if r["hash_tok_pos"]>=0]
    return {"mode":mode,"n":n,"acc_strict":round(sum(r["correct_strict"] for r in rows)/n,4),
            "acc_tol":round(sum(r["correct_tol"] for r in rows)/n,4),
            "no_marker":round(sum(1 for r in rows if r["hash_tok_pos"]<0)/n,4),
            "hit_max":round(sum(1 for r in rows if r["hit_max"])/n,4),
            "len_mean":round(statistics.mean(lens),1),
            "L_to_marker_mean":round(statistics.mean([r["L_to_marker"] for r in wh]),1) if wh else None,
            "rep_4gram":round(statistics.mean([r["rep_4gram"] for r in rows]),4)}

env={"torch":torch.__version__,"transformers":__import__("transformers").__version__,
     "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}
TS.write_manifest(OUT, tag=args.tag, model=MODEL, env=env,
                  config={"decode_k":DK,"calib_n":args.calib_n,"limit":args.limit,"modes":MODES})

baseline_seqs=None; summaries={}
for mode in MODES:
    print(f"\n=== {mode} ===", flush=True)
    rows=run(mode, baseline_seqs)
    if mode=="k8_native" and baseline_seqs is None:
        baseline_seqs={r["id"]:r["gen_token_ids"] for r in rows}
    s=summarize(mode,rows); summaries[mode]=s
    json.dump(s, open(os.path.join(OUT,f"{mode}_summary.json"),"w"), indent=2)
    print(json.dumps(s), flush=True)

json.dump({"tag":args.tag,"summaries":summaries,"quantiles":quants},
          open(os.path.join(OUT,"summary.json"),"w"), indent=2)
print("\n== v30 gain controls (decode K4) ==")
print(f"{'mode':20s}{'len':>7}{'Δvs_k8':>8}{'acc':>7}{'noMark':>8}")
base=summaries.get("k8_native",{}).get("len_mean",0)
for m,s in summaries.items():
    print(f"{m:20s}{s['len_mean']:>7.0f}{s['len_mean']-base:>8.1f}{s['acc_strict']*100:>6.1f}%{s['no_marker']*100:>7.1f}%")
print(f"\nwrote {OUT}/summary.json")
