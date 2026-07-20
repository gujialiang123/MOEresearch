#!/usr/bin/env python3
"""v25: Answer-readiness probe. Reuses v23 trajectories (no re-generation).

Question: when K drops and L_to_marker grows, is it because the model forms the
answer LATER (t_ready delayed) — or it knows early but emits the marker later?

For each problem and each decode-K trajectory (from v23: 8x8, 8x6, 8x4), at
checkpoints t along the SAVED token sequence:
  - Probe 1 (gold-answer conditional logprob): append a fixed cue
    "\nTherefore, the final answer is ####" and teacher-force the gold number under
    NATIVE K=8; record mean token logprob. t_ready_logprob = first t whose mean
    logprob exceeds a threshold calibrated on K8-correct samples.
  - Probe 2 (short greedy answer): from prefix under K8, greedily emit <=24 tokens,
    parse; t_ready_greedy = first t that yields the correct answer (and stays).

Report t_ready, t_marker, t_eos and the decompositions (t_marker - t_ready),
(t_eos - t_marker) per K.

Run: GPU=4 python scripts/run_v25_answer_readiness.py --v23_dir results/2026-07-20_v23_phase_factorial --limit 80
"""
import os, sys, json, argparse, statistics
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP
from moe_research import answer_parsing as AP
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--v23_dir", required=True)
ap.add_argument("--configs", default="8x8,8x6,8x4")
ap.add_argument("--limit", type=int, default=80)
ap.add_argument("--step", type=int, default=32)
ap.add_argument("--out", default="/home/t-jialianggu/work/MOEresearch/results/2026-07-20_v25_answer_readiness")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
CFGS = args.configs.split(",")

print("loading model + v23 trajectories...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
TOPK = model.config.num_experts_per_tok
eos_ids = tok.eos_token_id
EOS = eos_ids[0] if isinstance(eos_ids, list) else eos_ids
eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
HASH_IDS = tok.encode("####", add_special_tokens=False)

# native K8 policy (probes always use K8, per plan)
policy = KP.KPolicy(prefill_k=TOPK, decode_k=TOPK, weight_mode="renorm_survivors")
ctx = KP.attach_policy(model, policy)

ds = load_dataset("gsm8k", "main", split="test")
golds = {i: AP.parse_gold(ds[i]["answer"]) for i in range(len(ds))}
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
def prompt_ids_for(i):
    text = tok.apply_chat_template([{"role": "user", "content": ds[i]["question"] + SUFFIX}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").to(DEV)["input_ids"]

CUE = "\nTherefore, the final answer is #### "
CUE_IDS = tok(CUE, add_special_tokens=False)["input_ids"]

def load_traj(cfg):
    p = os.path.join(args.v23_dir, f"{cfg}_raw.jsonl")
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}

traj = {c: load_traj(c) for c in CFGS}
common = set.intersection(*[set(traj[c]) for c in CFGS]) if all(traj.values()) else set()
ids = sorted(common)[:args.limit]
print(f"probing {len(ids)} problems x {len(CFGS)} configs", flush=True)


@torch.inference_mode()
def gold_logprob(prompt_ids, prefix_tokens, gold_str):
    """Mean token logprob of gold answer after prefix + cue, under native K8."""
    gold_ids = tok(gold_str, add_special_tokens=False)["input_ids"]
    if not gold_ids:
        return None
    seq = torch.tensor([prompt_ids[0].tolist() + list(prefix_tokens) + CUE_IDS + gold_ids], device=DEV)
    policy.prefill_k = TOPK; policy.decode_k = TOPK
    KP._PHASE.set("prefill")
    logits = model(seq).logits[0].float()
    # positions predicting the gold tokens: last len(gold_ids) targets
    start = seq.shape[1] - len(gold_ids)
    lp = F.log_softmax(logits[start-1:seq.shape[1]-1], dim=-1)
    tgt = torch.tensor(gold_ids, device=DEV)
    toklp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(toklp.mean())


def checkpoints(glen, marker_pos):
    cps = list(range(0, glen, args.step))
    if marker_pos and marker_pos > 0:
        cps += [marker_pos - 32, marker_pos - 16, marker_pos - 8, marker_pos]
    return sorted({c for c in cps if 0 <= c <= glen})


# --- calibration: threshold from K8-correct samples' logprob at their marker ---
records = []
calib_vals = []
for i in ids:
    r8 = traj["8x8"][i]
    if not r8["correct_strict"]:
        continue
    seq = r8["gen_token_ids"]
    hp = r8.get("hash_tok_pos", -1)
    if hp <= 0:
        continue
    pid = prompt_ids_for(i)
    lp = gold_logprob(pid, seq[:hp], golds[i])
    if lp is not None:
        calib_vals.append(lp)
THRESH = statistics.median(calib_vals) - 0.5 if calib_vals else -3.0  # lenient offset
print(f"calibrated readiness threshold (mean gold logprob) = {THRESH:.3f} (from {len(calib_vals)} K8-correct)", flush=True)

for i in ids:
    pid = prompt_ids_for(i)
    for c in CFGS:
        r = traj[c][i]
        seq = r["gen_token_ids"]; glen = len(seq)
        marker = r.get("hash_tok_pos", -1)
        eos_pos = r["eos_pos"]
        t_ready = None
        for t in checkpoints(glen, marker):
            lp = gold_logprob(pid, seq[:t], golds[i])
            if lp is not None and lp >= THRESH:
                t_ready = t
                break
        records.append({
            "id": i, "config": c, "K": r.get("avg_k_decode"),
            "glen": glen, "t_ready": t_ready, "t_marker": marker if marker >= 0 else None,
            "t_eos": (eos_pos + 1) if eos_pos >= 0 else None,
            "correct": r["correct_strict"],
            "marker_minus_ready": (marker - t_ready) if (t_ready is not None and marker >= 0) else None,
            "eos_minus_marker": ((eos_pos + 1) - marker) if (marker >= 0 and eos_pos >= 0) else None,
        })
    if (ids.index(i) + 1) % 10 == 0:
        print(f"  probed {ids.index(i)+1}/{len(ids)}", flush=True)
        with open(f"{args.out}/per_problem_raw.jsonl", "w") as f:
            for rr in records:
                f.write(json.dumps(rr) + "\n")

KP.detach_policy(model, ctx)
with open(f"{args.out}/per_problem_raw.jsonl", "w") as f:
    for rr in records:
        f.write(json.dumps(rr) + "\n")

def agg(c):
    rc = [r for r in records if r["config"] == c]
    def m(key):
        xs = [r[key] for r in rc if r[key] is not None]
        return round(statistics.mean(xs), 1) if xs else None
    return {"config": c, "n": len(rc), "t_ready_mean": m("t_ready"), "t_marker_mean": m("t_marker"),
            "t_eos_mean": m("t_eos"), "marker_minus_ready_mean": m("marker_minus_ready"),
            "eos_minus_marker_mean": m("eos_minus_marker"),
            "ready_found_frac": round(sum(1 for r in rc if r["t_ready"] is not None)/len(rc), 3) if rc else None}

summary = {"threshold": THRESH, "n_calib": len(calib_vals), "by_config": {c: agg(c) for c in CFGS}}
json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=2)
print("\n== v25 answer-readiness ==")
print(f"{'config':8s}{'t_ready':>9}{'t_marker':>10}{'t_eos':>8}{'mark-ready':>12}{'eos-mark':>10}")
for c in CFGS:
    a = summary["by_config"][c]
    print(f"{c:8s}{str(a['t_ready_mean']):>9}{str(a['t_marker_mean']):>10}{str(a['t_eos_mean']):>8}"
          f"{str(a['marker_minus_ready_mean']):>12}{str(a['eos_minus_marker_mean']):>10}")
print(f"\nInterpretation: t_ready delayed at low K -> real answer-formation delay;")
print(f"t_ready flat but t_marker/t_eos later -> verbosity/termination, not reasoning.")
print(f"wrote {args.out}/summary.json")
