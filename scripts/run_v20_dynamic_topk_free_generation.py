#!/usr/bin/env python3
"""v20 SMALL validation: fixed vs dynamic top-k on a mini GSM8K, with the FIXED
implementation (physical skip + strict parser + sync-free counters) and rich
per-sample logging for the "K vs output length" question.

This is intentionally small (default limit=16) — a sanity/validation run, not a
sweep. Records per-sample: gold, strict prediction, parse status, correct,
generated length, first-#### position, tokens-after-####, hit-max-new.

Run: MODEL=qwen GPU=6 python scripts/run_v20_dynamic_topk_free_generation.py \
        --limit 16 --configs baseline,dyn_tau0.7 --max_new 400
"""
import os, sys, json, time, argparse, statistics
import torch
sys.path.insert(0, os.path.dirname(__file__))
import dynamic_topk_utils as U
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=16)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--configs", type=str, default="fixed_k4,fixed_k6,fixed_k8,fixed_k10,fixed_k12")
ap.add_argument("--phase", type=str, default="all")
ap.add_argument("--renorm", type=str, default="renorm_survivors")
ap.add_argument("--out", type=str,
                default="/home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-16_v21_k_vs_length")
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)

# config spec: name -> (policy, threshold, kmin, kmax) or None for native baseline.
# Fixed-K forced via kmin==kmax. K>8 is OUT-OF-DISTRIBUTION (native top-8): the
# topk pool is widened to K, activating experts ranked 9..K that the model was
# never trained to mix this way. Flagged super_native in metadata.
CONFIGS = {
    "fixed_k4":  ("min_weight_cutoff", -1.0, 4, 4, False),
    "fixed_k6":  ("min_weight_cutoff", -1.0, 6, 6, False),
    "fixed_k8":  ("min_weight_cutoff", -1.0, 8, 8, False),   # == native baseline (verified by v20 equivalence)
    "fixed_k10": ("min_weight_cutoff", -1.0, 10, 10, True),  # OOD super-native
    "fixed_k12": ("min_weight_cutoff", -1.0, 12, 12, True),  # OOD super-native
    # dynamic policies kept available but not in the default fixed-K dose sweep
    "dyn_tau0.7": ("top_p_within_topk", 0.7, 1, 8, False),
    "dyn_tau0.8": ("top_p_within_topk", 0.8, 1, 8, False),
}

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
blocks = [m for m in model.modules() if isinstance(m, M.Qwen3MoeSparseMoeBlock)]
TOPK = model.config.num_experts_per_tok
ctrl = U.DynamicKController(blocks, M.Qwen3MoeSparseMoeBlock)

ds = load_dataset("gsm8k", "main", split="test").select(range(args.limit))
golds = [U.parse_gold(ex["answer"]) for ex in ds]
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
prompts = [tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
           tokenize=False, add_generation_prompt=True) for ex in ds]

eos_ids = tok.eos_token_id
eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}

# token ids for the "####" marker (to locate answer position at TOKEN granularity)
HASH_IDS = tok.encode("####", add_special_tokens=False)

def find_subseq(seq, sub):
    """Return start index of first occurrence of list `sub` in `seq`, else -1."""
    if not sub:
        return -1
    for i in range(0, len(seq) - len(sub) + 1):
        if seq[i:i+len(sub)] == sub:
            return i
    return -1

def rep_ngram_frac(seq, n=4):
    """Fraction of repeated n-grams (degeneration signal)."""
    if len(seq) < n:
        return 0.0
    grams = [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    total = len(grams)
    uniq = len(set(grams))
    return round(1 - uniq/total, 4) if total else 0.0

def run_config(name, baseline_seqs=None):
    raw_path = f"{OUT}/{name}_raw.jsonl"
    sum_path = f"{OUT}/{name}_summary.json"
    # config-level resume: a written per-config summary means this config fully
    # completed. avg_k counters need a full pass, so we resume at config
    # granularity (skip whole finished configs, re-run partial ones).
    if os.path.exists(sum_path):
        rows = [json.loads(l) for l in open(raw_path)]
        s = json.load(open(sum_path))
        print(f"[resume] {name} already complete ({len(rows)} rows) -> skip", flush=True)
        return s, rows
    spec = CONFIGS[name]
    if spec is None:
        ctrl.disable()
        ood = False
    else:
        policy, thr, kmin, kmax, ood = spec
        ctrl.enable(policy, thr, kmin, kmax, phase=args.phase, renorm=args.renorm,
                    benchmark_mode=False)
    rows = []
    gen_ms_total = 0.0
    open(raw_path, "w").close()  # truncate: fresh pass (config-level resume only)
    written = 0
    torch.cuda.synchronize()
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i+args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize(); gen_ms_total += (time.perf_counter() - t0) * 1e3
        in_len = enc["input_ids"].shape[1]
        new = out[:, in_len:]
        new_cpu = new.tolist()  # single batched D2H
        for j, seq in enumerate(new_cpu):
            gi = i + j
            # true generated length = up to & including first EOS (else full)
            glen = len(seq)
            eos_pos = -1
            for pos, t in enumerate(seq):
                if t in eos_set:
                    glen = pos + 1
                    eos_pos = pos
                    break
            gen_ids = seq[:glen]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            pred, status = U.parse_strict(text)
            # TOKEN-level answer position: first "####" marker
            hash_tok = find_subseq(gen_ids, HASH_IDS)
            # L decomposition (in TOKENS): to-answer vs post-answer
            if hash_tok >= 0:
                L_to_answer = hash_tok                    # tokens before "####"
                L_post_answer = glen - hash_tok           # tokens from "####" to end
            else:
                L_to_answer = glen                        # never emitted "####"
                L_post_answer = 0
            first_div = -1
            if baseline_seqs is not None:
                b = baseline_seqs[gi]
                for pos in range(min(len(b), len(gen_ids))):
                    if b[pos] != gen_ids[pos]:
                        first_div = pos
                        break
                else:
                    first_div = min(len(b), len(gen_ids)) if len(b) != len(gen_ids) else -1
            rows.append({
                "id": gi, "gold": golds[gi], "pred": pred, "parse_status": status,
                "correct": (pred is not None and pred == golds[gi]),
                "gen_len": glen, "hit_max_new": glen >= args.max_new,
                "eos_pos": eos_pos, "hash_tok_pos": hash_tok,
                "L_to_answer": L_to_answer, "L_post_answer": L_post_answer,
                "rep_4gram_frac": rep_ngram_frac(gen_ids, 4),
                "first_div_from_baseline": first_div,
                "first_hash_char": text.find("####"),
                "gen_token_ids": gen_ids,   # FULL raw ids -> recompute any metric later
                "text": text,               # FULL generated text
            })
        # incremental write: flush rows produced in this batch (crash-safe,
        # "write as much as computed" -> partial configs are inspectable / not lost)
        with open(raw_path, "a") as f:
            for r in rows[written:]:
                f.write(json.dumps(r) + "\n")
            f.flush(); os.fsync(f.fileno())
        written = len(rows)
        print(f"  [{name}] {written}/{len(prompts)} samples written", flush=True)
    stats = ctrl.stats() if spec is not None else {"avg_k_decode": float(TOPK), "avg_k_prefill": float(TOPK)}
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    pf = sum(r["parse_status"] == "parse_failure" for r in rows)
    lens = [r["gen_len"] for r in rows]
    with_hash = [r for r in rows if r["hash_tok_pos"] >= 0]
    summary = {
        "config": name, "ood_super_native": bool(ood), "n": n,
        "strict_accuracy": round(correct/n, 4), "correct": correct,
        "parse_failures": pf, "no_hash_frac": round(sum(1 for r in rows if r["hash_tok_pos"] < 0)/n, 3),
        "avg_k_decode": stats.get("avg_k_decode"), "avg_k_prefill": stats.get("avg_k_prefill"),
        "gen_len_mean": round(statistics.mean(lens), 1),
        "gen_len_median": int(statistics.median(lens)),
        "gen_len_p90": int(sorted(lens)[max(0, int(0.9*n)-1)]),
        "gen_len_max": max(lens),
        "hit_max_new_frac": round(sum(r["hit_max_new"] for r in rows)/n, 3),
        "L_to_answer_mean": round(statistics.mean([r["L_to_answer"] for r in rows]), 1),
        "L_post_answer_mean": round(statistics.mean([r["L_post_answer"] for r in rows]), 1),
        "L_to_answer_mean_withhash": round(statistics.mean([r["L_to_answer"] for r in with_hash]), 1) if with_hash else None,
        "L_post_answer_mean_withhash": round(statistics.mean([r["L_post_answer"] for r in with_hash]), 1) if with_hash else None,
        "rep_4gram_mean": round(statistics.mean([r["rep_4gram_frac"] for r in rows]), 4),
        "eos_emitted_frac": round(sum(1 for r in rows if r["eos_pos"] >= 0)/n, 3),
        "generation_wall_ms": round(gen_ms_total, 1),
    }
    # per-config summary doubles as the "config complete" marker for resume
    # (raw JSONL was already written incrementally, batch by batch, above)
    json.dump(summary, open(sum_path, "w"), indent=2)
    return summary, rows

names = args.configs.split(",")
summaries = {}
raw_by = {}
baseline_seqs = None
# ensure baseline-equivalent config (fixed_k8) runs first so divergence can be measured
if "fixed_k8" in names:
    names = ["fixed_k8"] + [n for n in names if n != "fixed_k8"]
for nm in names:
    print(f"\n=== {nm} ===", flush=True)
    s, rows = run_config(nm, baseline_seqs=baseline_seqs)
    summaries[nm] = s; raw_by[nm] = rows
    if nm == "fixed_k8" and baseline_seqs is None:
        baseline_seqs = {r["id"]: r["gen_token_ids"] for r in rows}
    print(json.dumps({k: v for k, v in s.items() if k != "flip_vs_baseline"}), flush=True)

# paired flip table vs baseline (fixed_k8 == native)
BASE = "fixed_k8" if "fixed_k8" in summaries else names[0]
base = {r["id"]: r["correct"] for r in raw_by[BASE]}
for nm in names:
    if nm == BASE:
        continue
    cur = {r["id"]: r["correct"] for r in raw_by[nm]}
    bc_pw = sum(1 for i in base if base[i] and not cur[i])
    bw_pc = sum(1 for i in base if not base[i] and cur[i])
    both_c = sum(1 for i in base if base[i] and cur[i])
    both_w = sum(1 for i in base if not base[i] and not cur[i])
    summaries[nm]["flip_vs_baseline"] = {
        "base_correct_policy_wrong": bc_pw, "base_wrong_policy_correct": bw_pc,
        "both_correct": both_c, "both_wrong": both_w}

env = {"torch": torch.__version__,
       "transformers": __import__("transformers").__version__,
       "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0),
       "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip()}
out = {"model": MODEL, "args": vars(args), "env": env, "baseline_config": BASE, "summaries": summaries}
json.dump(out, open(f"{OUT}/summary.json", "w"), indent=2)

print("\n== config | K | acc | pf | no_hash | len | L_to_ans | L_post | rep4 | wall ==")
for nm in names:
    s = summaries[nm]
    kd = s["avg_k_decode"]
    print(f"{nm:11s} k={kd} acc={s['strict_accuracy']*100:5.1f}% pf={s['parse_failures']:2d} "
          f"noH={s['no_hash_frac']:.2f} len={s['gen_len_mean']:.0f} "
          f"Lto={s['L_to_answer_mean']:.0f} Lpost={s['L_post_answer_mean']:.0f} "
          f"rep={s['rep_4gram_mean']:.3f} wall={s['generation_wall_ms']/1000:.0f}s")
print(f"\nwrote {OUT}/summary.json")
