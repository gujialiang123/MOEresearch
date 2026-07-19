#!/usr/bin/env python3
"""h1: collect per-layer expert-selection traces across workloads for the
"hot-expert" study (emergent shared-expert viability).

Unlike v16 (which only kept aggregate stats for layer 20 + cross-layer means),
this saves the FULL per-layer x per-expert selection-count matrix [L, E] SEPARATELY
FOR EACH WORKLOAD, plus per-(layer) selection counts restricted to DECODE positions.

Why decode-only matters: the systems bottleneck is decode expert weight movement,
so hotness must be measured on decode-position routing, not just prompt/prefill.

For each workload we run greedy generation, then a single teacher-forced forward
over prompt+generation with output_router_logits=True, and tally topk expert ids
at (a) all positions and (b) decode positions only (index >= prompt_len).

Outputs per workload to results/<date>_hot_expert/traces/<workload>.npz:
  counts_all[L,E], counts_decode[L,E], n_tokens_all, n_tokens_decode,
  and a compact per-(layer,decode-step) top1 expert id array for stability-over-time.

Run: GPU=4 python scripts/hot_expert/run_h1_collect_traces.py --workloads gsm8k,coding,general,synthetic
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--workloads", type=str, default="gsm8k,coding,general,synthetic")
ap.add_argument("--n_per_workload", type=int, default=40, help="prompts per workload")
ap.add_argument("--max_new", type=int, default=160)
ap.add_argument("--out", type=str,
                default="/home/t-jialianggu/work/MOEresearch/results/2026-07-19_hot_expert")
args = ap.parse_args()
OUT = args.out
os.makedirs(f"{OUT}/traces", exist_ok=True)

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
model.eval()
cfg = model.config
E, TOPK, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
print(f"E={E} topk={TOPK} L={L}", flush=True)


def build_prompts(name, n):
    """Return a list of chat-templated prompt strings for a workload."""
    from datasets import load_dataset
    prompts = []
    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test").select(range(n))
        for ex in ds:
            prompts.append(ex["question"] + "\nPlease reason step by step, and put your final answer after '#### '.")
    elif name == "coding":
        templates = [
            "Write a Python function to {t}. Include edge-case handling and a short test.",
        ]
        tasks = ["reverse a linked list", "find the longest palindromic substring",
                 "implement binary search", "detect a cycle in a directed graph",
                 "compute edit distance between two strings", "serialize and deserialize a binary tree",
                 "implement an LRU cache", "merge k sorted lists", "find median of two sorted arrays",
                 "implement quicksort with median-of-three pivot"]
        for i in range(n):
            prompts.append(templates[0].format(t=tasks[i % len(tasks)]))
    elif name == "general":
        qs = ["Explain how photosynthesis works.", "What caused the fall of the Roman Empire?",
              "Describe the water cycle.", "How does a vaccine train the immune system?",
              "Explain the theory of plate tectonics.", "What is the difference between weather and climate?",
              "How do black holes form?", "Explain supply and demand in economics.",
              "What is CRISPR and how is it used?", "Describe how neurons transmit signals."]
        for i in range(n):
            prompts.append(qs[i % len(qs)])
    elif name == "synthetic":
        # fixed structural prompts -> stable decode length, format-heavy
        for i in range(n):
            prompts.append(f"Count from {i+1} to {i+20} and for each number say whether it is prime. "
                           f"Format each line as 'N: prime' or 'N: not prime'.")
    else:
        raise ValueError(name)
    # chat template
    out = []
    for p in prompts:
        try:
            out.append(tok.apply_chat_template([{"role": "user", "content": p}],
                                               tokenize=False, add_generation_prompt=True))
        except Exception:
            out.append(p)
    return out


@torch.inference_mode()
def collect_workload(name, prompts):
    path = f"{OUT}/traces/{name}.npz"
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        print(f"  [resume] {name} trace exists -> skip", flush=True)
        return {"workload": name, "n_prompts": len(prompts),
                "n_tokens_all": int(d["n_tokens_all"]), "n_tokens_decode": int(d["n_tokens_decode"]),
                "trace": path}
    counts_all = np.zeros((L, E), dtype=np.int64)
    counts_decode = np.zeros((L, E), dtype=np.int64)
    n_all = 0
    n_dec = 0
    # per-(layer, decode-step) top1 expert id, for a few sequences (stability-over-time)
    top1_over_time = []  # list of [L, T_dec] arrays (first few seqs)
    t0 = time.time()
    for qi, text in enumerate(prompts):
        enc = tok(text, return_tensors="pt").to(DEV)
        plen = enc["input_ids"].shape[1]
        gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        full = gen  # [1, plen+G]
        G = full.shape[1] - plen
        if G <= 0:
            continue
        # teacher-forced forward to get router logits at every position
        out = model(full, output_router_logits=True)
        seq_top1 = np.zeros((L, max(G, 1)), dtype=np.int16)
        for layer in range(L):
            logits = out.router_logits[layer].float()  # [plen+G, E]
            tki = torch.topk(logits, TOPK, dim=-1).indices  # [T, TOPK]
            tki_np = tki.cpu().numpy()
            # all positions
            for t in range(tki_np.shape[0]):
                np.add.at(counts_all[layer], tki_np[t], 1)
            # decode positions: router logit row at index j predicts token j+1;
            # decode tokens are those generated, i.e. rows [plen-1 .. plen+G-2]
            dec_rows = tki_np[plen - 1: plen - 1 + G]  # [G, TOPK]
            for t in range(dec_rows.shape[0]):
                np.add.at(counts_decode[layer], dec_rows[t], 1)
            seq_top1[layer, :dec_rows.shape[0]] = dec_rows[:, 0].astype(np.int16)
        n_all += tki_np.shape[0] if False else out.router_logits[0].shape[0]
        n_dec += G
        if qi < 5:
            top1_over_time.append(seq_top1)
        if (qi + 1) % 10 == 0:
            print(f"  [{name}] {qi+1}/{len(prompts)}  ({time.time()-t0:.0f}s)", flush=True)
    # save (top1_over_time is ragged per-seq [L, G_i]; store as object list safely)
    path = f"{OUT}/traces/{name}.npz"
    save_kw = dict(counts_all=counts_all, counts_decode=counts_decode,
                   n_tokens_all=n_all, n_tokens_decode=n_dec,
                   E=E, TOPK=TOPK, L=L, n_top1_seqs=len(top1_over_time))
    for si, arr in enumerate(top1_over_time):
        save_kw[f"top1_seq{si}"] = arr  # each [L, G_i]
    np.savez_compressed(path, **save_kw)
    print(f"  saved {path}  (n_all={n_all}, n_decode={n_dec})", flush=True)
    return {"workload": name, "n_prompts": len(prompts), "n_tokens_all": int(n_all),
            "n_tokens_decode": int(n_dec), "trace": path}


summary = []
for wl in args.workloads.split(","):
    print(f"\n=== workload: {wl} ===", flush=True)
    prompts = build_prompts(wl, args.n_per_workload)
    summary.append(collect_workload(wl, prompts))

meta = {
    "model": MODEL, "E": E, "TOPK": TOPK, "L": L,
    "moe_intermediate_size": getattr(cfg, "moe_intermediate_size", None),
    "hidden_size": cfg.hidden_size,
    "args": vars(args),
    "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
    "torch": torch.__version__,
    "workloads": summary,
}
json.dump(meta, open(f"{OUT}/collect_meta.json", "w"), indent=2)
print(f"\nwrote {OUT}/collect_meta.json", flush=True)
