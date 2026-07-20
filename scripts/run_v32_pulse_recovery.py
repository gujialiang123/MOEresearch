#!/usr/bin/env python3
"""v32: Prefill/decode pulse-and-recovery. Is a prefill perturbation recovered once
K8 decode resumes? Does a transient decode low-K pulse leave a lasting trajectory
change? Is teacher-forced drift small while free-running drift is amplified?

Part A (prefill recovery): baseline prefill K8/decode K8 vs intervention prefill K4/
decode K8. Open-loop teacher-forces baseline tokens (KL decay + recovery half-life);
closed-loop free greedy (first divergence, final length, correctness).

Part B (decode pulse): K8 everywhere except a K4 full-renorm pulse on steps
[start,start+dur); dur in {1,4,16,64}, start in {early,middle,late} (fixed rule).
Open-loop: KL during pulse vs recovery after. Closed-loop: first flip, re-convergence,
final length delta, marker/EOS, correctness.

Decisive comparison: teacher-forced (open) drift recovers, free-running (closed) drift
amplifies => autoregressive feedback drives the length effect.

Run:
  HF_HOME=... CUDA_VISIBLE_DEVICES=4 python scripts/run_v32_pulse_recovery.py --stage smoke
"""
import os, sys, json, argparse, statistics
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _harness as H
from moe_research import k_policy as KP
from moe_research import trace_schema as TS
from moe_research import answer_parsing as AP
from moe_research.interventions import pulse_start

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v32_pulse_recovery")
ap.add_argument("--stage", choices=["smoke", "pilot"], default="smoke")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--open_steps", type=int, default=128)
ap.add_argument("--recov_window", type=int, default=32)
ap.add_argument("--out_root", default="/home/t-jialianggu/work/MOEresearch/results")
args = ap.parse_args()

# v32 uses development prompts (train, disjoint from calibration)
LO = H.DEV_RANGE[0]
HI = LO + args.n
OUT = os.path.join(args.out_root, f"2026-07-20_{args.tag}_{args.stage}")
os.makedirs(OUT, exist_ok=True)
PULSE_DURS = [1, 4, 16, 64]
PULSE_STARTS = ("early", "middle", "late")

print(f"[{args.stage}] v32 on train[{LO}:{HI}]", flush=True)
model, tok = H.load_model()
H.set_decoder(tok)
blocks, _ = KP._find_moe_blocks(model)
for i, b in enumerate(blocks):
    b._kp_layer_idx = i
TOPK = model.config.num_experts_per_tok
eos_ids = tok.eos_token_id
EOS = eos_ids[0] if isinstance(eos_ids, list) else eos_ids
eos_set = H.eos_set_of(tok)
HASH_IDS = tok.encode("####", add_special_tokens=False)

ds = H.gsm8k("train", LO, HI)
prompts, golds, questions = H.make_prompts(tok, ds)

policy = KP.KPolicy(prefill_k=TOPK, decode_k=TOPK, weight_mode="renorm_survivors")
ctx = KP.attach_policy(model, policy)


def set_k(pk, dk):
    policy.prefill_k = pk; policy.decode_k = dk
    policy.weight_mode = "renorm_survivors"; policy.renorm_beta = 1.0


def kl(p, q):
    lp = F.log_softmax(p, -1); lq = F.log_softmax(q, -1)
    return float((lp.exp() * (lp - lq)).sum())


def correctness(gen, gold):
    glen = len(gen); 
    for pos, t in enumerate(gen):
        if t in eos_set:
            glen = pos + 1; break
    text = tok.decode(gen[:glen], skip_special_tokens=True)
    ps, _ = AP.parse_strict(text)
    hp = H.find_subseq(gen[:glen], HASH_IDS)
    has_eos = any(t in eos_set for t in gen)
    return {"len": glen, "correct": (ps == gold and ps is not None),
            "has_marker": hp >= 0, "has_eos": has_eos}


@torch.inference_mode()
def gen_baseline(prompt_ids):
    set_k(TOPK, TOPK)
    out = model.generate(prompt_ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=EOS)
    return out[0, prompt_ids.shape[1]:].tolist()


@torch.inference_mode()
def teacher_force_logits(prompt_ids, gen_tokens, prefill_k):
    set_k(prefill_k, TOPK); KP._PHASE.set("prefill")
    cache = model(prompt_ids, use_cache=True).past_key_values
    KP._PHASE.set("decode"); set_k(prefill_k, TOPK)
    outs = []
    for t in range(len(gen_tokens)):
        o = model(torch.tensor([[gen_tokens[t]]], device=H.DEV), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        outs.append(o.logits[0, -1].float())
    return torch.stack(outs)


def half_life(series):
    """steps until KL falls below half of its initial (first-4-mean) value."""
    if len(series) < 4:
        return None
    init = statistics.mean(series[:4])
    if init <= 0:
        return 0
    for t, v in enumerate(series):
        if v <= init / 2:
            return t
    return None


@torch.inference_mode()
def partA(prompt_ids, base_seq, gold):
    G = min(len(base_seq), args.open_steps)
    lg8 = teacher_force_logits(prompt_ids, base_seq[:G], prefill_k=TOPK)
    lg4 = teacher_force_logits(prompt_ids, base_seq[:G], prefill_k=4)
    kl_series = [kl(lg8[t], lg4[t]) for t in range(G)]
    lbl8 = F.log_softmax(lg8, -1); lbl4 = F.log_softmax(lg4, -1)
    # baseline-token delta logprob per step (token that baseline actually emitted)
    dlp = []
    for t in range(G - 1):
        bt = base_seq[t + 1]
        dlp.append(float(lbl4[t, bt] - lbl8[t, bt]))
    set_k(4, TOPK)
    out = model.generate(prompt_ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=EOS)
    seq4 = out[0, prompt_ids.shape[1]:].tolist()
    fdiv = -1
    for p in range(min(len(base_seq), len(seq4))):
        if base_seq[p] != seq4[p]:
            fdiv = p; break
    else:
        fdiv = min(len(base_seq), len(seq4)) if len(base_seq) != len(seq4) else -1
    c = correctness(seq4, gold)
    return {"kl_series": [round(x, 5) for x in kl_series],
            "kl_first16": round(statistics.mean(kl_series[:16]), 5) if kl_series else None,
            "kl_last16": round(statistics.mean(kl_series[-16:]), 5) if len(kl_series) >= 16 else None,
            "half_life": half_life(kl_series),
            "mean_base_token_dlogp": round(statistics.mean(dlp), 5) if dlp else None,
            "closed_first_div": fdiv, "base_len": len(base_seq), "k4prefill_len": c["len"],
            "k4prefill_correct": c["correct"], "len_delta": c["len"] - len(base_seq)}


@torch.inference_mode()
def k8_reference_logits(prompt_ids, base_seq, upto):
    set_k(TOPK, TOPK); KP._PHASE.set("prefill")
    cache = model(prompt_ids, use_cache=True).past_key_values
    KP._PHASE.set("decode"); set_k(TOPK, TOPK)
    outs = []
    for t in range(upto):
        o = model(torch.tensor([[base_seq[t]]], device=H.DEV), past_key_values=cache, use_cache=True)
        cache = o.past_key_values; outs.append(o.logits[0, -1].float())
    return outs


@torch.inference_mode()
def pulsed_open_loop(prompt_ids, base_seq, ref_logits, start, dur, window):
    set_k(TOPK, TOPK); KP._PHASE.set("prefill")
    cache = model(prompt_ids, use_cache=True).past_key_values
    KP._PHASE.set("decode")
    end = min(len(base_seq), start + dur + window, len(ref_logits))
    kl_in, kl_after = [], []
    for t in range(end):
        in_pulse = (start <= t < start + dur)
        set_k(TOPK, 4 if in_pulse else TOPK)
        o = model(torch.tensor([[base_seq[t]]], device=H.DEV), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        d = kl(ref_logits[t], o.logits[0, -1].float())
        if in_pulse:
            kl_in.append(d)
        elif t >= start + dur:
            kl_after.append(d)
    return (round(statistics.mean(kl_in), 5) if kl_in else None,
            round(statistics.mean(kl_after[:8]), 5) if len(kl_after) >= 4 else None,
            round(statistics.mean(kl_after[-8:]), 5) if len(kl_after) >= 8 else None)


@torch.inference_mode()
def pulsed_closed_loop(prompt_ids, base_seq, start, dur, gold):
    policy.prefill_k = TOPK; policy.decode_k = 4
    policy.weight_mode = "renorm_survivors"; policy.renorm_beta = 1.0
    policy.decode_step_selector = (lambda s, a=start, b=start + dur: a <= s < b)
    out = model.generate(prompt_ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=EOS)
    policy.decode_step_selector = None
    seq = out[0, prompt_ids.shape[1]:].tolist()
    fdiv = -1
    for p in range(min(len(base_seq), len(seq))):
        if base_seq[p] != seq[p]:
            fdiv = p; break
    else:
        fdiv = min(len(base_seq), len(seq)) if len(base_seq) != len(seq) else -1
    # re-convergence: does the tail of seq re-join baseline suffix?
    reconv = (fdiv >= 0 and len(seq) == len(base_seq) and seq[-8:] == base_seq[-8:])
    c = correctness(seq, gold)
    return {"first_flip": fdiv, "len": c["len"], "len_delta": c["len"] - len(base_seq),
            "correct": c["correct"], "has_marker": c["has_marker"], "has_eos": c["has_eos"],
            "reconverged": bool(reconv)}


results = {"partA": [], "partB": []}
raw_path = os.path.join(OUT, "raw.jsonl")
fraw = open(raw_path, "w")
for qi, ex in enumerate(ds):
    gold = golds[qi]
    prompt_ids = tok(prompts[qi], return_tensors="pt", add_special_tokens=False).to(H.DEV)["input_ids"]
    base = gen_baseline(prompt_ids)
    G = len(base)
    if G < 8:
        continue
    base_eos = next((i for i, t in enumerate(base) if t in eos_set), G - 1)
    a = partA(prompt_ids, base, gold); a["qi"] = LO + qi
    a_slim = {k: v for k, v in a.items() if k != "kl_series"}
    results["partA"].append(a_slim)
    fraw.write(json.dumps({"part": "A", **a}) + "\n")
    upto = min(G, max(PULSE_DURS) + args.recov_window +
               max(pulse_start(s, G, max(PULSE_DURS)) or 0 for s in PULSE_STARTS) + 1)
    ref_logits = k8_reference_logits(prompt_ids, base, min(G, upto))
    for sname in PULSE_STARTS:
        for dur in PULSE_DURS:
            s = pulse_start(sname, G, dur)
            if s is None or s + dur >= G:
                continue
            kin, kaft, klate = pulsed_open_loop(prompt_ids, base, ref_logits, s, dur, args.recov_window)
            cl = pulsed_closed_loop(prompt_ids, base, s, dur, gold)
            rec = {"qi": LO + qi, "start": sname, "dur": dur, "start_pos": s,
                   "open_kl_in": kin, "open_kl_recovery_first8": kaft, "open_kl_recovery_last8": klate,
                   "base_eos": base_eos, **{f"closed_{k}": v for k, v in cl.items()}}
            results["partB"].append(rec)
            fraw.write(json.dumps({"part": "B", **rec}) + "\n")
    fraw.flush()
    if (qi + 1) % 4 == 0:
        print(f"  done {qi+1}/{len(ds)}", flush=True)
fraw.close()
KP.detach_policy(model, ctx)
json.dump(results, open(os.path.join(OUT, "raw_grouped.json"), "w"))

# ---- aggregate ----
A = results["partA"]
def _m(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 5) if xs else None
partA_summary = {
    "n": len(A),
    "open_kl_first16_mean": _m([a["kl_first16"] for a in A]),
    "open_kl_last16_mean": _m([a["kl_last16"] for a in A]),
    "half_life_median": statistics.median([a["half_life"] for a in A if a["half_life"] is not None]) if any(a["half_life"] is not None for a in A) else None,
    "mean_base_token_dlogp": _m([a["mean_base_token_dlogp"] for a in A]),
    "closed_first_div_median": statistics.median([a["closed_first_div"] for a in A]) if A else None,
    "closed_no_divergence_frac": round(sum(1 for a in A if a["closed_first_div"] < 0)/len(A), 3) if A else None,
    "closed_len_delta_mean": _m([a["len_delta"] for a in A]),
}
B = results["partB"]
def bagg(dur):
    rows = [b for b in B if b["dur"] == dur]
    if not rows:
        return None
    flips = [b for b in rows if b["closed_first_flip"] is not None and b["closed_first_flip"] >= 0]
    return {"dur": dur, "n": len(rows),
            "open_kl_in_mean": _m([b["open_kl_in"] for b in rows]),
            "open_kl_recovery_first8": _m([b["open_kl_recovery_first8"] for b in rows]),
            "open_kl_recovery_last8": _m([b["open_kl_recovery_last8"] for b in rows]),
            "closed_flip_frac": round(len(flips)/len(rows), 3),
            "closed_len_delta_mean": _m([b["closed_len_delta"] for b in rows]),
            "closed_reconverged_frac": round(sum(1 for b in rows if b["closed_reconverged"])/len(rows), 3)}
partB_summary = {"by_duration": {d: bagg(d) for d in PULSE_DURS},
                 "by_start": {}}
for sname in PULSE_STARTS:
    rows = [b for b in B if b["start"] == sname]
    if rows:
        flips = [b for b in rows if b["closed_first_flip"] is not None and b["closed_first_flip"] >= 0]
        partB_summary["by_start"][sname] = {
            "n": len(rows), "closed_flip_frac": round(len(flips)/len(rows), 3),
            "closed_len_delta_mean": _m([b["closed_len_delta"] for b in rows])}

TS.write_manifest(OUT, tag=args.tag, model=H.MODEL, env=H.env_dict(),
                  config={"stage": args.stage, "n": args.n, "max_new": args.max_new,
                          "pulse_durs": PULSE_DURS, "pulse_starts": list(PULSE_STARTS),
                          "range": [LO, HI]}, extra={"sample_ids": list(range(LO, HI))})
json.dump({"partA": partA_summary, "partB": partB_summary},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)

print("\n== v32 Part A: prefill recovery ==")
print(f"open KL first16={partA_summary['open_kl_first16_mean']} -> last16={partA_summary['open_kl_last16_mean']} "
      f"half_life_med={partA_summary['half_life_median']}")
print(f"closed first-div median={partA_summary['closed_first_div_median']} "
      f"len_delta_mean={partA_summary['closed_len_delta_mean']}")
print("\n== v32 Part B: decode pulse (open recovery vs closed flip) ==")
for d in PULSE_DURS:
    b = partB_summary["by_duration"][d]
    if b:
        print(f"dur={d:2d}: open_kl_in={b['open_kl_in_mean']} recov8={b['open_kl_recovery_first8']} "
              f"flip={b['closed_flip_frac']} len_delta={b['closed_len_delta_mean']} reconv={b['closed_reconverged_frac']}")
print(f"\nwrote {OUT}/summary.json")
