#!/usr/bin/env python3
"""v26: TRUE current-step direct-effect probe (improves v22).

v22 fixed token IDs but low-K accumulated from sequence start, so it could not
isolate changing K at ONLY the current step. v26: build the K=8 baseline KV cache
incrementally; at each probed position fork a COPY of that identical-K8 cache and
run the single current token under K=8/6/4 for the CURRENT forward only.

Efficiency: one incremental K8 pass per problem (O(G)); at each probed position a
cache deepcopy + one single-token forward per k. policy.decode_k is mutated in
place (no per-step re-patch). k=8 reference is a K8 fork at the same position.

Run: GPU=7 python scripts/run_v26_direct_effect_probe.py --limit 60
"""
import os, sys, json, argparse, statistics, random, copy
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=60)
ap.add_argument("--kset", type=str, default="4,6,8")
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--window", type=int, default=48)
ap.add_argument("--n_random", type=int, default=12)
ap.add_argument("--max_pos", type=int, default=48)
ap.add_argument("--out", default="/home/t-jialianggu/work/MOEresearch/results/2026-07-20_v26_direct_effect")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
KSET = sorted(int(x) for x in args.kset.split(","))
NONREF = [k for k in KSET if k != 8]
random.seed(0)

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
TOPK = model.config.num_experts_per_tok
eos_ids = tok.eos_token_id
EOS = eos_ids[0] if isinstance(eos_ids, list) else eos_ids
eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
HASH_IDS = tok.encode("####", add_special_tokens=False)
MARK0 = HASH_IDS[0]

ds = load_dataset("gsm8k", "main", split="test").select(range(args.limit))
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."

policy = KP.KPolicy(prefill_k=TOPK, decode_k=TOPK, weight_mode="renorm_survivors")
ctx = KP.attach_policy(model, policy)


@torch.inference_mode()
def gen_baseline(prompt_ids):
    policy.prefill_k = TOPK; policy.decode_k = TOPK
    out = model.generate(prompt_ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=EOS)
    return out[0, prompt_ids.shape[1]:].tolist()


def margin(l):
    z = l.clone(); ze = z[EOS].clone(); z[EOS] = float("-inf")
    return float(ze - z.max())

def kl(ref, cur):
    lp = F.log_softmax(ref, -1); lq = F.log_softmax(cur, -1)
    return float((lp.exp() * (lp - lq)).sum())


@torch.inference_mode()
def probe_problem(prompt_ids, base_seq, positions):
    G = len(base_seq)
    recs = []
    posset = set(positions)
    policy.prefill_k = TOPK; policy.decode_k = TOPK
    KP._PHASE.set("prefill")
    cache = model(prompt_ids, use_cache=True).past_key_values  # history = prompt (state for t=0)
    KP._PHASE.set("decode")
    for t in range(G):
        cur_tok = torch.tensor([[base_seq[t]]], device=DEV)
        gold_tok = base_seq[t + 1] if t + 1 < G else None
        if t in posset:
            per_k = {}
            for k in KSET:
                policy.decode_k = k
                o = model(cur_tok, past_key_values=copy.deepcopy(cache), use_cache=True)
                per_k[k] = o.logits[0, -1].float()
            ref = per_k[TOPK]
            lref = F.log_softmax(ref, -1)
            for k in KSET:
                cur = per_k[k]; lcur = F.log_softmax(cur, -1)
                recs.append({
                    "t": t, "k": k, "G": G,
                    "kl_p8_pk": kl(ref, cur),
                    "eos_dlogp": float(lcur[EOS] - lref[EOS]),
                    "eos_margin_delta": margin(cur) - margin(ref),
                    "gold_dlogp": float(lcur[gold_tok] - lref[gold_tok]) if gold_tok is not None else None,
                    "marker_dlogp": float(lcur[MARK0] - lref[MARK0]),
                    "top1_agree": bool(ref.argmax().item() == cur.argmax().item()),
                    "logit_l2": float((cur - ref).norm()),
                })
        policy.decode_k = TOPK
        cache = model(cur_tok, past_key_values=cache, use_cache=True).past_key_values
    return recs


results = []
for qi, ex in enumerate(ds):
    text = tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
                                   tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(text, return_tensors="pt").to(DEV)["input_ids"]
    base_seq = gen_baseline(prompt_ids)
    G = len(base_seq)
    if G < 4:
        continue
    base_eos = next((i for i, t in enumerate(base_seq) if t in eos_set), G - 1)
    hash_pos = next((i for i in range(len(base_seq)-len(HASH_IDS)+1) if base_seq[i:i+len(HASH_IDS)] == HASH_IDS), -1)
    pos = set(range(max(1, base_eos - args.window), base_eos))
    if hash_pos > 0:
        pos |= set(range(max(1, hash_pos - min(16, args.window)), hash_pos))
    ctrl = [p for p in range(1, G - 1) if p not in pos]
    if ctrl:
        pos |= set(random.sample(ctrl, min(args.n_random, len(ctrl))))
    pos = sorted(p for p in pos if 1 <= p < G - 1)
    if len(pos) > args.max_pos:
        pos = sorted(random.sample(pos, args.max_pos))
    recs = probe_problem(prompt_ids, base_seq, pos)
    for r in recs:
        r.update({"qi": qi, "base_eos": base_eos, "hash_pos": hash_pos,
                  "dist_to_eos": base_eos - r["t"], "dist_to_marker": (hash_pos - r["t"]) if hash_pos > 0 else None})
    results.extend(recs)
    if (qi + 1) % 5 == 0:
        print(f"  done {qi+1}/{args.limit} (positions this q={len(pos)}, total recs={len(results)})", flush=True)
        with open(f"{args.out}/per_position_raw.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

KP.detach_policy(model, ctx)
with open(f"{args.out}/per_position_raw.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")


def agg(rows, k):
    rk = [r for r in rows if r["k"] == k]
    if not rk:
        return None
    return {
        "k": k, "n": len(rk),
        "kl_p8_pk_mean": round(statistics.mean(r["kl_p8_pk"] for r in rk), 5),
        "eos_dlogp_mean": round(statistics.mean(r["eos_dlogp"] for r in rk), 4),
        "eos_margin_delta_mean": round(statistics.mean(r["eos_margin_delta"] for r in rk), 4),
        "gold_dlogp_mean": round(statistics.mean(r["gold_dlogp"] for r in rk if r["gold_dlogp"] is not None), 4),
        "top1_agree_frac": round(statistics.mean(1.0 if r["top1_agree"] else 0.0 for r in rk), 4),
    }

near = [r for r in results if r["dist_to_eos"] <= 8]
summary = {"kset": KSET, "n_problems": args.limit, "window": args.window,
           "overall_by_k": {k: agg(results, k) for k in KSET},
           "near_eos_by_k": {k: agg(near, k) for k in KSET}}
json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=2)
print("\n== v26 current-step direct effect (identical K8 history) ==")
print(f"{'K':>3}{'KL(p8||pk)':>13}{'EOS dlogp':>12}{'margin d':>11}{'gold dlogp':>12}{'top1agree':>11}")
for scope, rows in (("overall", results), ("near-EOS(<=8)", near)):
    print(f"-- {scope} --")
    for k in KSET:
        a = agg(rows, k)
        if a:
            print(f"{k:>3}{a['kl_p8_pk_mean']:>13}{a['eos_dlogp_mean']:>12}{a['eos_margin_delta_mean']:>11}{a['gold_dlogp_mean']:>12}{a['top1_agree_frac']:>11}")
print(f"\nwrote {args.out}/summary.json")
