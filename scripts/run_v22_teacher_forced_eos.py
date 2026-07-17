#!/usr/bin/env python3
"""v22 (Experiment B): teacher-forced EOS / margin / KL analysis.

Separates the DIRECT termination effect from the TRAJECTORY-mediated effect of
reducing K, using teacher forcing on a FIXED baseline (K=8) sequence.

For each problem:
  1. Generate the baseline (K=8) token sequence greedily; record it.
  2. Teacher-force that SAME sequence through the model under K in {kset}, in a
     single forward pass (labels = the sequence), and read per-position:
       - log p(EOS) at each step
       - EOS margin: z_EOS - max_{v != EOS} z_v
       - entropy of next-token distribution
       - KL( p_{K=8} || p_{K=k} )  at each position
       - delta-NLL of the baseline next token
  3. Focus on the LAST W tokens before the baseline EOS (termination zone).

Because the prefix is IDENTICAL across K (teacher forced), any change in EOS
logit/margin is the DIRECT effect of reducing K on the termination distribution
— not mediated by a diverging trajectory. (The trajectory-mediated arm is the
free-generation run v21.)

Run: MODEL=qwen GPU=6 python scripts/run_v22_teacher_forced_eos.py --limit 100 --kset 4,6,8,10,12
"""
import os, sys, json, argparse, statistics
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import dynamic_topk_utils as U
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as M
from datasets import load_dataset

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--kset", type=str, default="4,6,8,10,12")
ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--window", type=int, default=64, help="termination-zone window before baseline EOS")
ap.add_argument("--out", type=str,
                default="/home/t-jialianggu/work/MOEresearch/results/2026-07-16_v22_teacher_forced")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
KSET = [int(x) for x in args.kset.split(",")]

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
blocks = [m for m in model.modules() if isinstance(m, M.Qwen3MoeSparseMoeBlock)]
TOPK = model.config.num_experts_per_tok
ctrl = U.DynamicKController(blocks, M.Qwen3MoeSparseMoeBlock)

eos_ids = tok.eos_token_id
EOS = eos_ids[0] if isinstance(eos_ids, list) else eos_ids
eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}

ds = load_dataset("gsm8k", "main", split="test").select(range(args.limit))
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."


def set_fixed_k(k):
    """Apply fixed-K everywhere (phase=all) via kmin=kmax=k; k==TOPK => native."""
    if k == TOPK:
        ctrl.disable()
    else:
        ctrl.enable("min_weight_cutoff", -1.0, k, k, phase="all",
                    renorm="renorm_survivors", benchmark_mode=True)


@torch.inference_mode()
def gen_baseline(prompt_ids):
    set_fixed_k(TOPK)
    out = model.generate(prompt_ids, max_new_tokens=args.max_new, do_sample=False,
                         pad_token_id=EOS)
    return out[0, prompt_ids.shape[1]:].tolist()


@torch.inference_mode()
def logits_for(full_ids, k):
    """Teacher-force full_ids under fixed-K=k; return logits [T, V] (float32, cpu-free)."""
    set_fixed_k(k)
    out = model(full_ids)
    return out.logits[0].float()  # [T, V]


results = []
per_pos_dump = []
raw_path = f"{args.out}/per_problem_raw.jsonl"
open(raw_path, "w").close()  # truncate; incremental append per problem (crash-safe)
written = 0
for qi, ex in enumerate(ds):
    text = tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
                                   tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(text, return_tensors="pt").to(DEV)["input_ids"]
    plen = prompt_ids.shape[1]
    base_seq = gen_baseline(prompt_ids)
    # locate baseline EOS
    base_eos = next((i for i, t in enumerate(base_seq) if t in eos_set), len(base_seq) - 1)
    full = torch.cat([prompt_ids, torch.tensor([base_seq], device=DEV)], dim=1)  # [1, plen+G]

    # positions in the *generated* region: logits at index (plen-1 + t) predict gen token t
    G = len(base_seq)
    def extract(k):
        lg = logits_for(full, k)                      # [plen+G, V]
        pred_slice = lg[plen - 1: plen - 1 + G]       # [G, V]
        logp = F.log_softmax(pred_slice, dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)                     # [G]
        eos_logp = logp[:, EOS]                       # [G]
        z = pred_slice.clone()
        z_eos = z[:, EOS].clone()
        z[:, EOS] = float("-inf")
        margin = z_eos - z.max(-1).values             # [G]
        tgt = torch.tensor(base_seq, device=DEV)
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [G]
        return {"logp": logp, "eos_logp": eos_logp, "margin": margin, "ent": ent, "nll": nll}

    # compute K=8 reference FIRST (so KL is available for ALL k, incl. k<8)
    ref = extract(TOPK)
    lo = max(0, base_eos - args.window)
    zone = slice(lo, base_eos + 1)

    for k in KSET:
        row = ref if k == TOPK else extract(k)
        kl = (ref["logp"].exp() * (ref["logp"] - row["logp"])).sum(-1)  # KL(p8 || pk) [G]
        rec = {
            "qi": qi, "k": k, "G": G, "base_eos": base_eos,
            "eos_logp_at_eos": float(row["eos_logp"][base_eos]),
            "margin_at_eos": float(row["margin"][base_eos]),
            "eos_logp_zone_mean": float(row["eos_logp"][zone].mean()),
            "margin_zone_mean": float(row["margin"][zone].mean()),
            "entropy_zone_mean": float(row["ent"][zone].mean()),
            "nll_zone_mean": float(row["nll"][zone].mean()),
            "kl_from_k8_zone_mean": float(kl[zone].mean()),
            "kl_from_k8_at_eos": float(kl[base_eos]),
        }
        results.append(rec)
    # incremental: flush this problem's rows immediately (crash-safe, resumable-ish)
    with open(raw_path, "a") as f:
        for r in results[written:]:
            f.write(json.dumps(r) + "\n")
        f.flush(); os.fsync(f.fileno())
    written = len(results)
    if qi < 20:  # dump per-position for a few problems for figures
        per_pos_dump.append({"qi": qi, "base_eos": base_eos, "G": G})
    if (qi + 1) % 10 == 0:
        print(f"  done {qi+1}/{len(ds)}", flush=True)

ctrl.disable()

# aggregate by k
agg = {}
for k in KSET:
    rows = [r for r in results if r["k"] == k]
    agg[k] = {
        "k": k, "ood": k > TOPK, "n": len(rows),
        "eos_logp_at_eos_mean": round(statistics.mean(r["eos_logp_at_eos"] for r in rows), 4),
        "margin_at_eos_mean": round(statistics.mean(r["margin_at_eos"] for r in rows), 4),
        "eos_logp_zone_mean": round(statistics.mean(r["eos_logp_zone_mean"] for r in rows), 4),
        "margin_zone_mean": round(statistics.mean(r["margin_zone_mean"] for r in rows), 4),
        "entropy_zone_mean": round(statistics.mean(r["entropy_zone_mean"] for r in rows), 4),
        "nll_zone_mean": round(statistics.mean(r["nll_zone_mean"] for r in rows), 4),
        "kl_from_k8_zone_mean": round(statistics.mean(r.get("kl_from_k8_zone_mean", 0.0) for r in rows), 5),
    }

env = {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
       "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip()}
out = {"model": MODEL, "args": vars(args), "env": env, "window": args.window, "by_k": agg}
json.dump(out, open(f"{args.out}/summary.json", "w"), indent=2)
# per_problem_raw.jsonl was already written incrementally, per problem, above

print("\n== Teacher-forced termination zone (last {} tok before baseline EOS) ==".format(args.window))
print(f"{'K':>4}{'ood':>5}{'logp(EOS)@eos':>15}{'margin@eos':>12}{'logp(EOS)_zone':>16}{'margin_zone':>13}{'KL(p8||pk)':>12}")
for k in KSET:
    a = agg[k]
    print(f"{k:>4}{str(a['ood']):>5}{a['eos_logp_at_eos_mean']:>15}{a['margin_at_eos_mean']:>12}"
          f"{a['eos_logp_zone_mean']:>16}{a['margin_zone_mean']:>13}{a['kl_from_k8_zone_mean']:>12}")
print(f"\nInterpretation: if lower K systematically LOWERS logp(EOS)/margin in the")
print(f"termination zone (identical prefix), that is DIRECT evidence K reduces the")
print(f"termination probability (not mediated by trajectory divergence).")
print(f"\nwrote {args.out}/summary.json")
