#!/usr/bin/env python3
"""Shared GSM8K config-driven harness for v23/v24/v28 (prefill_k x decode_k x weight_mode).

Runs greedy generation over GSM8K with a list of KPolicy configs, saving rich
per-sample logs (full token ids + text -> any metric recomputable offline).
Baseline config (8x8) runs first so first-divergence vs baseline is available.
Incremental JSONL write + --resume.

Config token format:  "<prefill_k>x<decode_k>[:<weight_mode>]"
  e.g.  8x8  8x6  8x4  6x8  6x6  4x8  4x4        (v23, default weight_mode)
        8x6:no_renorm  8x4:fold_mass_to_top1     (v24)

Run:
  MODEL=qwen GPU=4 python scripts/run_gsm8k_configs.py \
     --tag v23_phase_factorial --split test --limit 500 \
     --configs 8x8,8x6,8x4,6x8,6x6,4x8,4x4
"""
import os, sys, json, time, argparse, statistics
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP
from moe_research import answer_parsing as AP
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--split", default="test", choices=["test", "train"])
ap.add_argument("--limit", type=int, default=500)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--configs", required=True, help="comma list of PxD[:weight_mode]")
ap.add_argument("--default_weight_mode", default="renorm_survivors")
ap.add_argument("--out_root", default="/home/t-jialianggu/work/MOEresearch/results")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--calib_scalars", default=None, help="JSON of {'layer,K': scalar} for calibrated_norm_match")
args = ap.parse_args()

CALIB = {}
if args.calib_scalars and os.path.exists(args.calib_scalars):
    CALIB = json.load(open(args.calib_scalars))

OUT = os.path.join(args.out_root, f"2026-07-20_{args.tag}")
os.makedirs(OUT, exist_ok=True)


def parse_cfg(tok_str):
    name = tok_str
    wm = args.default_weight_mode
    body = tok_str
    if ":" in tok_str:
        body, wm = tok_str.split(":", 1)
    p, d = body.split("x")
    return {"name": name.replace(":", "_"), "prefill_k": int(p), "decode_k": int(d), "weight_mode": wm}

CONFIGS = [parse_cfg(c) for c in args.configs.split(",")]
# ensure 8x8 baseline first if present
CONFIGS.sort(key=lambda c: 0 if (c["prefill_k"] == 8 and c["decode_k"] == 8 and c["weight_mode"] == "renorm_survivors") else 1)

print(f"loading model... configs={[c['name'] for c in CONFIGS]}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
TOPK = model.config.num_experts_per_tok

ds = load_dataset("gsm8k", "main", split=args.split).select(range(args.offset, args.offset + args.limit))
golds = [AP.parse_gold(ex["answer"]) for ex in ds]
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
prompts = [tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
           tokenize=False, add_generation_prompt=True) for ex in ds]

eos_ids = tok.eos_token_id
eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
HASH_IDS = tok.encode("####", add_special_tokens=False)


def find_subseq(seq, sub):
    if not sub:
        return -1
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i+len(sub)] == sub:
            return i
    return -1

def rep_frac(seq, n):
    if len(seq) < n:
        return 0.0
    g = [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    return round(1 - len(set(g))/len(g), 4)


def run_config(cfg, baseline_seqs):
    out_path = os.path.join(OUT, f"{cfg['name']}_raw.jsonl")
    done_ids = set()
    if args.resume and os.path.exists(out_path):
        for l in open(out_path):
            try:
                done_ids.add(json.loads(l)["id"])
            except Exception:
                pass
    policy = KP.KPolicy(prefill_k=cfg["prefill_k"], decode_k=cfg["decode_k"], weight_mode=cfg["weight_mode"],
                        calib_scalars=CALIB)
    ctx = KP.attach_policy(model, policy)
    rows = []
    fout = open(out_path, "a")
    gen_ms = 0.0
    torch.cuda.synchronize()
    todo = [i for i in range(len(prompts)) if i not in done_ids]
    for bstart in range(0, len(todo), args.batch):
        idxs = todo[bstart:bstart+args.batch]
        chunk = [prompts[i] for i in idxs]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        ctx.reset_counters()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize(); gen_ms += (time.perf_counter()-t0)*1e3
        in_len = enc["input_ids"].shape[1]
        new = out[:, in_len:].tolist()
        st = ctx.stats()
        for j, seq in enumerate(new):
            gi = idxs[j]
            glen = len(seq); eos_pos = -1
            for pos, t in enumerate(seq):
                if t in eos_set:
                    glen = pos+1; eos_pos = pos; break
            gen_ids = seq[:glen]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            ps, ss = AP.parse_strict(text)
            pt, ts = AP.parse_tolerant(text)
            hash_tok = find_subseq(gen_ids, HASH_IDS)
            L_to_marker = hash_tok if hash_tok >= 0 else None
            L_post_marker = (glen - hash_tok) if hash_tok >= 0 else None
            first_div = None
            if baseline_seqs is not None and gi in baseline_seqs:
                b = baseline_seqs[gi]
                first_div = -1
                for pos in range(min(len(b), len(gen_ids))):
                    if b[pos] != gen_ids[pos]:
                        first_div = pos; break
                else:
                    first_div = min(len(b), len(gen_ids)) if len(b) != len(gen_ids) else -1
            row = {
                "id": gi, "gold": golds[gi],
                "pred_strict": ps, "correct_strict": (ps is not None and ps == golds[gi]),
                "pred_tol": pt, "correct_tol": (pt is not None and pt == golds[gi]),
                "parse_status_strict": ss, "parse_status_tol": ts,
                "gen_len": glen, "eos_pos": eos_pos, "hit_max_new": glen >= args.max_new,
                "hash_tok_pos": hash_tok, "L_to_marker": L_to_marker, "L_post_marker": L_post_marker,
                "L_to_eos": (eos_pos+1) if eos_pos >= 0 else None,
                "rep_3gram": rep_frac(gen_ids, 3), "rep_4gram": rep_frac(gen_ids, 4),
                "first_div": first_div,
                "avg_k_prefill": st["avg_k_prefill"], "avg_k_decode": st["avg_k_decode"],
                "gen_token_ids": gen_ids, "text": text,
            }
            rows.append(row)
            fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(f"  [{cfg['name']}] {min(bstart+args.batch,len(todo))}/{len(todo)} "
              f"k_dec={st['avg_k_decode']} k_pre={st['avg_k_prefill']}", flush=True)
    fout.close()
    KP.detach_policy(model, ctx)
    # reload all rows (incl resumed) for summary
    allrows = [json.loads(l) for l in open(out_path)]
    return cfg, allrows, gen_ms


def summarize(cfg, rows, gen_ms):
    n = len(rows)
    lens = [r["gen_len"] for r in rows]
    wh = [r for r in rows if r["hash_tok_pos"] >= 0]
    def m(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 2) if xs else None
    srt = sorted(lens)
    def pct(p):
        return srt[min(len(srt)-1, int(p*len(srt)))]
    return {
        "config": cfg["name"], "prefill_k": cfg["prefill_k"], "decode_k": cfg["decode_k"],
        "weight_mode": cfg["weight_mode"], "n": n,
        "acc_strict": round(sum(r["correct_strict"] for r in rows)/n, 4),
        "acc_tol": round(sum(r["correct_tol"] for r in rows)/n, 4),
        "no_marker_frac": round(sum(1 for r in rows if r["hash_tok_pos"] < 0)/n, 4),
        "no_eos_frac": round(sum(1 for r in rows if r["eos_pos"] < 0)/n, 4),
        "hit_max_frac": round(sum(1 for r in rows if r["hit_max_new"])/n, 4),
        "len_mean": round(statistics.mean(lens), 1), "len_median": statistics.median(lens),
        "len_p90": pct(0.90), "len_p95": pct(0.95), "len_p99": pct(0.99),
        "L_to_marker_mean": m([r["L_to_marker"] for r in wh]),
        "L_post_marker_mean": m([r["L_post_marker"] for r in wh]),
        "rep_4gram_mean": m([r["rep_4gram"] for r in rows]),
        "avg_k_decode": rows[0]["avg_k_decode"], "avg_k_prefill": rows[0]["avg_k_prefill"],
        "gen_wall_ms": round(gen_ms, 1),
    }


baseline_seqs = None
summaries = {}
raw_by = {}
for cfg in CONFIGS:
    print(f"\n=== {cfg['name']} (pk={cfg['prefill_k']} dk={cfg['decode_k']} wm={cfg['weight_mode']}) ===", flush=True)
    cfg, rows, gen_ms = run_config(cfg, baseline_seqs)
    s = summarize(cfg, rows, gen_ms)
    summaries[cfg["name"]] = s
    raw_by[cfg["name"]] = rows
    if cfg["prefill_k"] == 8 and cfg["decode_k"] == 8 and baseline_seqs is None:
        baseline_seqs = {r["id"]: r["gen_token_ids"] for r in rows}
    json.dump(s, open(os.path.join(OUT, f"{cfg['name']}_summary.json"), "w"), indent=2)
    print(json.dumps(s), flush=True)

env = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
       "git_commit": os.popen("cd /home/t-jialianggu/work/MOEresearch && git rev-parse HEAD").read().strip()}
json.dump({"tag": args.tag, "args": vars(args), "env": env, "summaries": summaries},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(f"\n== {args.tag} ==")
print(f"{'config':16s}{'pk':>3}{'dk':>3}{'acc_s':>7}{'noMark':>8}{'len':>7}{'Lmark':>7}{'Lpost':>7}{'rep4':>7}")
for nm, s in summaries.items():
    print(f"{nm:16s}{s['prefill_k']:>3}{s['decode_k']:>3}{s['acc_strict']*100:>6.1f}%"
          f"{s['no_marker_frac']*100:>7.1f}%{s['len_mean']:>7.0f}{str(s['L_to_marker_mean']):>7}"
          f"{str(s['L_post_marker_mean']):>7}{str(s['rep_4gram_mean']):>7}")
print(f"\nwrote {OUT}/summary.json")
