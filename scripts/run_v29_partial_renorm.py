#!/usr/bin/env python3
"""v29: Partial-renorm dose response. Does generation length change monotonically
with renormalization strength beta (at fixed retained top-K)?

Config: prefill_k=8, decode only, greedy. Sweep decode_k in {6,4}, beta in
{0,.25,.5,.75,1}, plus native K8 baseline. beta=0 == no_renorm, beta=1 == full
renorm_survivors.

Saves per-request raw.jsonl (full token ids), summary.json, manifest.json.
Analysis (paired bootstrap CI, McNemar, Holm, beta monotonic trend) is in
analyze_v29_partial_renorm.py.

Run: GPU=4 python scripts/run_v29_partial_renorm.py --limit 500 --configs auto
"""
import os, sys, json, time, argparse, statistics
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP
from moe_research import answer_parsing as AP
from moe_research import trace_schema as TS
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v29_partial_renorm")
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=500)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--decode_ks", default="6,4")
ap.add_argument("--betas", default="0,0.25,0.5,0.75,1.0")
ap.add_argument("--out_root", default="/home/t-jialianggu/work/MOEresearch/results")
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()

OUT = os.path.join(args.out_root, f"2026-07-20_{args.tag}")
os.makedirs(OUT, exist_ok=True)
DKS = [int(x) for x in args.decode_ks.split(",")]
BETAS = [float(x) for x in args.betas.split(",")]

# config list: baseline 8x8 (native), then (dk, beta) grid
CONFIGS = [{"name": "k8_native", "decode_k": 8, "beta": 1.0}]
for dk in DKS:
    for b in BETAS:
        CONFIGS.append({"name": f"k{dk}_b{b}", "decode_k": dk, "beta": b})

print(f"loading model... {len(CONFIGS)} configs", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
TOPK = model.config.num_experts_per_tok

ds = load_dataset("gsm8k", "main", split=args.split).select(range(args.limit))
golds = [AP.parse_gold(ex["answer"]) for ex in ds]
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
prompts = [tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
           tokenize=False, add_generation_prompt=True) for ex in ds]
eos_ids = tok.eos_token_id
eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
HASH_IDS = tok.encode("####", add_special_tokens=False)

def find_subseq(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i+len(sub)] == sub:
            return i
    return -1

def rep_frac(seq, n):
    if len(seq) < n: return 0.0
    g = [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    return round(1 - len(set(g))/len(g), 4)

def run(cfg, baseline_seqs):
    path = os.path.join(OUT, f"{cfg['name']}_raw.jsonl")
    done = set()
    if args.resume and os.path.exists(path):
        for l in open(path):
            try: done.add(json.loads(l)["id"])
            except: pass
    wm = "renorm_survivors" if (cfg["decode_k"] == 8) else "partial_renorm"
    policy = KP.KPolicy(prefill_k=8, decode_k=cfg["decode_k"], weight_mode=wm, renorm_beta=cfg["beta"])
    ctx = KP.attach_policy(model, policy)
    fout = open(path, "a")
    todo = [i for i in range(len(prompts)) if i not in done]
    rows = []
    for bs in range(0, len(todo), args.batch):
        idxs = todo[bs:bs+args.batch]
        enc = tok([prompts[i] for i in idxs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        ctx.reset_counters()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        in_len = enc["input_ids"].shape[1]
        new = out[:, in_len:].tolist()
        st = ctx.stats()
        for j, seq in enumerate(new):
            gi = idxs[j]
            glen = len(seq); eos_pos = -1
            for pos, t in enumerate(seq):
                if t in eos_set: glen = pos+1; eos_pos = pos; break
            gen = seq[:glen]
            text = tok.decode(gen, skip_special_tokens=True)
            ps, ss = AP.parse_strict(text); pt, ts = AP.parse_tolerant(text)
            hp = find_subseq(gen, HASH_IDS)
            fdiv = None
            if baseline_seqs and gi in baseline_seqs:
                b = baseline_seqs[gi]; fdiv = -1
                for p in range(min(len(b), len(gen))):
                    if b[p] != gen[p]: fdiv = p; break
                else: fdiv = min(len(b), len(gen)) if len(b) != len(gen) else -1
            row = {"id": gi, "gold": golds[gi], "decode_k": cfg["decode_k"], "beta": cfg["beta"],
                   "pred_strict": ps, "correct_strict": (ps == golds[gi] and ps is not None),
                   "pred_tol": pt, "correct_tol": (pt == golds[gi] and pt is not None),
                   "parse_strict": ss, "gen_len": glen, "eos_pos": eos_pos,
                   "hit_max": glen >= args.max_new, "hash_tok_pos": hp,
                   "L_to_marker": hp if hp >= 0 else None,
                   "L_post_marker": (glen-hp) if hp >= 0 else None,
                   "rep_4gram": rep_frac(gen, 4), "first_div": fdiv,
                   "avg_k_decode": st["avg_k_decode"], "gen_token_ids": gen, "text": text}
            rows.append(row); fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(f"  [{cfg['name']}] {min(bs+args.batch,len(todo))}/{len(todo)} k_dec={st['avg_k_decode']}", flush=True)
    fout.close(); KP.detach_policy(model, ctx)
    allrows = [json.loads(l) for l in open(path)]
    return allrows

def summarize(cfg, rows):
    n = len(rows); lens = [r["gen_len"] for r in rows]
    wh = [r for r in rows if r["hash_tok_pos"] >= 0]
    srt = sorted(lens)
    return {"config": cfg["name"], "decode_k": cfg["decode_k"], "beta": cfg["beta"], "n": n,
            "acc_strict": round(sum(r["correct_strict"] for r in rows)/n, 4),
            "acc_tol": round(sum(r["correct_tol"] for r in rows)/n, 4),
            "no_marker": round(sum(1 for r in rows if r["hash_tok_pos"] < 0)/n, 4),
            "no_eos": round(sum(1 for r in rows if r["eos_pos"] < 0)/n, 4),
            "hit_max": round(sum(1 for r in rows if r["hit_max"])/n, 4),
            "len_mean": round(statistics.mean(lens), 1), "len_median": statistics.median(lens),
            "len_p90": srt[min(n-1, int(0.9*n))], "len_p95": srt[min(n-1, int(0.95*n))],
            "L_to_marker_mean": round(statistics.mean([r["L_to_marker"] for r in wh]), 1) if wh else None,
            "rep_4gram": round(statistics.mean([r["rep_4gram"] for r in rows]), 4),
            "avg_k_decode": rows[0]["avg_k_decode"]}

env = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
TS.write_manifest(OUT, tag=args.tag, model=MODEL, env=env,
                  config={"decode_ks": DKS, "betas": BETAS, "limit": args.limit,
                          "max_new": args.max_new, "split": args.split})

baseline_seqs = None
summaries = {}
for cfg in CONFIGS:
    print(f"\n=== {cfg['name']} (dk={cfg['decode_k']} beta={cfg['beta']}) ===", flush=True)
    rows = run(cfg, baseline_seqs)
    if cfg["name"] == "k8_native" and baseline_seqs is None:
        baseline_seqs = {r["id"]: r["gen_token_ids"] for r in rows}
    s = summarize(cfg, rows); summaries[cfg["name"]] = s
    json.dump(s, open(os.path.join(OUT, f"{cfg['name']}_summary.json"), "w"), indent=2)
    print(json.dumps(s), flush=True)

json.dump({"tag": args.tag, "summaries": summaries}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("\n== v29 partial-renorm dose: config | dk | beta | len | acc | noMark ==")
for nm, s in summaries.items():
    print(f"{nm:12s} dk={s['decode_k']} b={s['beta']:.2f} len={s['len_mean']:.0f} acc={s['acc_strict']*100:.1f}% noMark={s['no_marker']*100:.1f}%")
print(f"\nwrote {OUT}/summary.json")
