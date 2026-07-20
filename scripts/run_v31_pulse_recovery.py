#!/usr/bin/env python3
"""v31: Pulse-and-recovery — is prefill perturbation recovered, and is decode
perturbation amplified by autoregressive feedback?

Part A (prefill recovery):
  baseline = prefill K8 / decode K8 ; intervention = prefill K4 / decode K8.
  - open-loop: teacher-force the baseline K8 tokens; per decode step measure
    KL(p_K8prefill || p_K4prefill), baseline-token Δlogprob. Does drift DECAY?
  - closed-loop: free greedy generation; first divergence, final length, EOS.

Part B (decode pulse):
  baseline K8 everywhere; only steps [start, start+dur) use K4 (full-renorm), then
  restore K8. duration in {1,4,16,64}; start in {early,middle,late}.
  - open-loop: teacher-force baseline tokens; KL during & after pulse -> recovery.
  - closed-loop: free generation; token-flip in pulse, trajectory divergence, length.

Key comparison: teacher-forced drift (should stay small) vs free-running drift
(should amplify) => proves autoregressive feedback drives the big effect.

Run: GPU=5 python scripts/run_v31_pulse_recovery.py --limit 60
"""
import os, sys, json, argparse, statistics, copy
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP
from moe_research import trace_schema as TS
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v31_pulse_recovery")
ap.add_argument("--limit", type=int, default=60)
ap.add_argument("--max_new", type=int, default=400)
ap.add_argument("--open_steps", type=int, default=128)
ap.add_argument("--out_root", default="/home/t-jialianggu/work/MOEresearch/results")
args = ap.parse_args()
OUT = os.path.join(args.out_root, f"2026-07-20_{args.tag}")
os.makedirs(OUT, exist_ok=True)

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
TOPK = model.config.num_experts_per_tok
eos_ids = tok.eos_token_id
EOS = eos_ids[0] if isinstance(eos_ids, list) else eos_ids
eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
ds = load_dataset("gsm8k", "main", split="test").select(range(args.limit))
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."

policy = KP.KPolicy(prefill_k=TOPK, decode_k=TOPK, weight_mode="renorm_survivors")
ctx = KP.attach_policy(model, policy)

def set_k(pk, dk, beta=1.0):
    policy.prefill_k = pk; policy.decode_k = dk
    policy.weight_mode = "renorm_survivors" if beta == 1.0 else "partial_renorm"
    policy.renorm_beta = beta

def kl(p, q):
    lp = F.log_softmax(p, -1); lq = F.log_softmax(q, -1)
    return float((lp.exp() * (lp - lq)).sum())

@torch.inference_mode()
def gen_baseline(prompt_ids):
    set_k(TOPK, TOPK)
    out = model.generate(prompt_ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=EOS)
    return out[0, prompt_ids.shape[1]:].tolist()

@torch.inference_mode()
def teacher_force_logits(prompt_ids, gen_tokens, prefill_k):
    """Incremental teacher-forced pass: encode PROMPT at prefill_k, then step through
    the generated tokens at decode K8. Returns per-generated-step logits [G, V].
    This correctly isolates a prefill perturbation followed by K8 decode recovery."""
    set_k(prefill_k, TOPK)
    KP._PHASE.set("prefill")
    cache = model(prompt_ids, use_cache=True).past_key_values
    KP._PHASE.set("decode")
    set_k(prefill_k, TOPK)  # decode_k=8; prefill_k irrelevant now (cache present)
    G = len(gen_tokens)
    outs = []
    cur = prompt_ids[0, -1].item()  # last prompt token predicts gen[0]... but we already
    # have cache after prompt; feed gen tokens one by one, logit predicts NEXT.
    cur = gen_tokens[0]
    # logit BEFORE consuming gen[0]: we need dist that predicts gen[0]. The prompt cache's
    # last position already gives it, but simplest: step gen tokens and collect the logit
    # produced when feeding gen[t] (predicts gen[t+1]); prepend the prompt->gen[0] logit.
    # Prompt->gen[0]:
    # (recompute: feed nothing; use a fresh forward of prompt last token already in cache)
    # We approximate by stepping: feed gen[t], logit predicts gen[t+1].
    for t in range(G):
        o = model(torch.tensor([[gen_tokens[t]]], device=DEV), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        outs.append(o.logits[0, -1].float())  # predicts gen[t+1]
    return torch.stack(outs)  # [G, V]; row t predicts gen[t+1]

# ---------------- Part A: prefill recovery ----------------
@torch.inference_mode()
def partA(prompt_ids, base_seq):
    G = min(len(base_seq), args.open_steps)
    # open-loop: teacher-force baseline tokens under prefill K8 vs K4 (decode K8 both)
    lg8 = teacher_force_logits(prompt_ids, base_seq[:G], prefill_k=TOPK)
    lg4 = teacher_force_logits(prompt_ids, base_seq[:G], prefill_k=4)
    kl_series = [kl(lg8[t], lg4[t]) for t in range(G)]
    # closed-loop: free gen with prefill K4 / decode K8
    set_k(4, TOPK)
    out = model.generate(prompt_ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=EOS)
    seq4 = out[0, prompt_ids.shape[1]:].tolist()
    fdiv = -1
    for p in range(min(len(base_seq), len(seq4))):
        if base_seq[p] != seq4[p]: fdiv = p; break
    else:
        fdiv = min(len(base_seq), len(seq4)) if len(base_seq) != len(seq4) else -1
    return {"kl_series": [round(x, 5) for x in kl_series],
            "kl_first16": round(statistics.mean(kl_series[:16]), 5) if kl_series else None,
            "kl_last16": round(statistics.mean(kl_series[-16:]), 5) if len(kl_series) >= 16 else None,
            "closed_first_div": fdiv, "base_len": len(base_seq), "k4prefill_len": len(seq4)}

# ---------------- Part B: decode pulse (efficient) ----------------
@torch.inference_mode()
def k8_reference_logits(prompt_ids, base_seq):
    """Pure-K8 incremental teacher-forced logits over base_seq (one pass). row t predicts t+1."""
    set_k(TOPK, TOPK); KP._PHASE.set("prefill")
    cache = model(prompt_ids, use_cache=True).past_key_values
    KP._PHASE.set("decode"); set_k(TOPK, TOPK)
    outs = []
    for t in range(len(base_seq)):
        o = model(torch.tensor([[base_seq[t]]], device=DEV), past_key_values=cache, use_cache=True)
        cache = o.past_key_values; outs.append(o.logits[0, -1].float())
    return outs  # list [V] per step

@torch.inference_mode()
def pulsed_open_loop(prompt_ids, base_seq, ref_logits, start, dur, window=16):
    """One teacher-forced pass applying K4 in [start,start+dur); compare KL to ref during
    pulse and in the `window` steps after. Only steps up to start+dur+window are run."""
    set_k(TOPK, TOPK); KP._PHASE.set("prefill")
    cache = model(prompt_ids, use_cache=True).past_key_values
    KP._PHASE.set("decode")
    end = min(len(base_seq), start + dur + window)
    kl_in, kl_after = [], []
    for t in range(end):
        in_pulse = (start <= t < start + dur)
        set_k(TOPK, 4 if in_pulse else TOPK)
        o = model(torch.tensor([[base_seq[t]]], device=DEV), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        d = kl(ref_logits[t], o.logits[0, -1].float())
        if in_pulse: kl_in.append(d)
        elif t >= start + dur: kl_after.append(d)
    return (round(statistics.mean(kl_in), 5) if kl_in else None,
            round(statistics.mean(kl_after[:8]), 5) if len(kl_after) >= 4 else None)

@torch.inference_mode()
def pulsed_closed_loop(prompt_ids, base_seq, start, dur):
    """Free greedy generation with K4 only during decode steps [start,start+dur) via
    decode_step_selector; one generate() call. Returns first-flip vs baseline + length."""
    policy.prefill_k = TOPK; policy.decode_k = 4
    policy.weight_mode = "renorm_survivors"; policy.renorm_beta = 1.0
    policy.decode_step_selector = (lambda s, a=start, b=start + dur: a <= s < b)
    out = model.generate(prompt_ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=EOS)
    policy.decode_step_selector = None
    seq = out[0, prompt_ids.shape[1]:].tolist()
    fdiv = -1
    for p in range(min(len(base_seq), len(seq))):
        if base_seq[p] != seq[p]: fdiv = p; break
    else:
        fdiv = min(len(base_seq), len(seq)) if len(base_seq) != len(seq) else -1
    return fdiv, len(seq)

results = {"partA": [], "partB": []}
PULSE_DURS = [1, 16]
PULSE_STARTS = ("early", "late")
for qi, ex in enumerate(ds):
    text = tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
                                   tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(text, return_tensors="pt").to(DEV)["input_ids"]
    base = gen_baseline(prompt_ids)
    G = len(base)
    if G < 8: continue
    base_eos = next((i for i, t in enumerate(base) if t in eos_set), G-1)
    # Part A
    a = partA(prompt_ids, base); a["qi"] = qi; results["partA"].append(a)
    # Part B: precompute K8 reference once, then pulse combos
    ref_logits = k8_reference_logits(prompt_ids, base)
    all_starts = {"early": 1, "middle": int(0.4*base_eos), "late": max(1, base_eos-64)}
    starts = {k: all_starts[k] for k in PULSE_STARTS}
    for sname, s in starts.items():
        for dur in PULSE_DURS:
            if s + dur > G: continue
            kin, kaft = pulsed_open_loop(prompt_ids, base, ref_logits, s, dur)
            flip, clen = pulsed_closed_loop(prompt_ids, base, s, dur)
            results["partB"].append({"qi": qi, "start": sname, "dur": dur,
                                     "open_kl_in": kin, "open_kl_after_first8": kaft,
                                     "closed_first_flip": flip, "closed_len": clen,
                                     "base_eos": base_eos})
    if (qi + 1) % 5 == 0:
        print(f"  done {qi+1}/{args.limit}", flush=True)
        json.dump(results, open(f"{OUT}/raw.json", "w"))

KP.detach_policy(model, ctx)
json.dump(results, open(f"{OUT}/raw.json", "w"), indent=2)

# ---- aggregate ----
A = results["partA"]
partA_summary = {
    "n": len(A),
    "open_kl_first16_mean": round(statistics.mean(a["kl_first16"] for a in A if a["kl_first16"] is not None), 5),
    "open_kl_last16_mean": round(statistics.mean(a["kl_last16"] for a in A if a["kl_last16"] is not None), 5),
    "closed_first_div_median": statistics.median([a["closed_first_div"] for a in A]),
    "closed_no_divergence_frac": round(sum(1 for a in A if a["closed_first_div"] < 0)/len(A), 3),
}
# Part B: teacher-forced (open) drift vs free (closed) flip, by duration
B = results["partB"]
def bagg(dur):
    rows = [b for b in B if b["dur"] == dur]
    if not rows: return None
    flips = [b["closed_first_flip"] for b in rows if b["closed_first_flip"] is not None and b["closed_first_flip"] >= 0]
    return {"dur": dur, "n": len(rows),
            "open_kl_in_mean": round(statistics.mean(b["open_kl_in"] for b in rows if b["open_kl_in"] is not None), 5),
            "open_kl_recovery_first8": round(statistics.mean(b["open_kl_after_first8"] for b in rows if b["open_kl_after_first8"] is not None), 5) if any(b["open_kl_after_first8"] is not None for b in rows) else None,
            "closed_flip_frac": round(len(flips)/len(rows), 3)}
partB_summary = {"by_duration": {d: bagg(d) for d in PULSE_DURS}}

env = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
TS.write_manifest(OUT, tag=args.tag, model=MODEL, env=env,
                  config={"limit": args.limit, "max_new": args.max_new, "pulse_durs": PULSE_DURS})
json.dump({"partA": partA_summary, "partB": partB_summary},
          open(f"{OUT}/summary.json", "w"), indent=2)

print("\n== v31 Part A: prefill recovery ==")
print(f"open-loop KL first16={partA_summary['open_kl_first16_mean']} -> last16={partA_summary['open_kl_last16_mean']} (decay?)")
print(f"closed-loop first-div median={partA_summary['closed_first_div_median']} no-div frac={partA_summary['closed_no_divergence_frac']}")
print("\n== v31 Part B: decode pulse (open KL vs closed flip) ==")
for d in PULSE_DURS:
    b = partB_summary["by_duration"][d]
    if b: print(f"dur={d}: open_kl_in={b['open_kl_in_mean']} recovery8={b['open_kl_recovery_first8']} closed_flip_frac={b['closed_flip_frac']}")
print(f"\nwrote {OUT}/summary.json")
