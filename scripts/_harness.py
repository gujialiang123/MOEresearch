"""Shared harness for v29-v32: model/tokenizer load, fixed GSM8K splits, greedy
generation, per-request row construction, and summary. Keeps all runners consistent
with the frozen protocol (see docs/2026-07-20/v29_v32_preflight_audit.md §7)."""
import os, sys, json, statistics
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import answer_parsing as AP
from moe_research import trace_schema as TS
from moe_research import stats as ST

MODEL = "/data/hf/models/Qwen3-30B-A3B-Instruct-2507"
DEV = "cuda:0"
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."

# Frozen sample-ID protocol
CALIB_RANGE = (0, 128)        # GSM8K train[0:128]
DEV_RANGE = (128, 328)        # GSM8K train[128:328] (disjoint)
SMOKE_RANGE = (128, 160)      # first 32 development prompts
TEST_RANGE = (0, 500)         # GSM8K test[0:500] (matches v23-v28 .select(range(500)))


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                                 dtype=torch.bfloat16).to(DEV)
    model.eval()
    return model, tok


def gsm8k(split, lo, hi):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split).select(range(lo, hi))
    return ds


def make_prompts(tok, ds):
    prompts = [tok.apply_chat_template([{"role": "user", "content": ex["question"] + SUFFIX}],
               tokenize=False, add_generation_prompt=True) for ex in ds]
    golds = [AP.parse_gold(ex["answer"]) for ex in ds]
    questions = [ex["question"] for ex in ds]
    return prompts, golds, questions


def eos_set_of(tok):
    e = tok.eos_token_id
    return set(e) if isinstance(e, list) else {e}


def find_subseq(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i+len(sub)] == sub:
            return i
    return -1


def rep_frac(seq, n):
    if len(seq) < n:
        return 0.0
    g = [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    return round(1 - len(set(g))/len(g), 4)


def build_row(gi, seq, gold, eos_set, hash_ids, max_new, baseline_seq=None, extra=None):
    glen = len(seq); eos_pos = -1
    for pos, t in enumerate(seq):
        if t in eos_set:
            glen = pos + 1; eos_pos = pos; break
    gen = seq[:glen]
    ps, ss = AP.parse_strict(_decode(gen))
    pt, _ = AP.parse_tolerant(_decode(gen))
    hp = find_subseq(gen, hash_ids)
    fdiv = None
    if baseline_seq is not None:
        fdiv = -1
        for p in range(min(len(baseline_seq), len(gen))):
            if baseline_seq[p] != gen[p]:
                fdiv = p; break
        else:
            fdiv = min(len(baseline_seq), len(gen)) if len(baseline_seq) != len(gen) else -1
    row = {"id": gi, "gold": gold,
           "pred_strict": ps, "strict_correct": (ps == gold and ps is not None),
           "pred_tol": pt, "tolerant_correct": (pt == gold and pt is not None),
           "generated_tokens": glen, "eos_pos": eos_pos, "hit_max": glen >= max_new,
           "first_marker_pos": hp if hp >= 0 else None,
           "parsed_answer_pos": hp if hp >= 0 else None,
           "L_post_marker": (glen - hp) if hp >= 0 else None,
           "rep_4gram": rep_frac(gen, 4), "first_divergence_pos": fdiv}
    if extra:
        row.update(extra)
    row["gen_token_ids"] = gen
    return row, gen


_TOK = {"tok": None}
def set_decoder(tok):
    _TOK["tok"] = tok
def _decode(ids):
    return _TOK["tok"].decode(ids, skip_special_tokens=True)


def summarize(name, rows, max_new):
    n = len(rows)
    lens = [r["generated_tokens"] for r in rows]
    events = [r["eos_pos"] >= 0 for r in rows]   # True = terminated (EOS); False = censored
    srt = sorted(lens)
    wh = [r for r in rows if r["first_marker_pos"] is not None]
    return {
        "config": name, "n": n,
        "acc_strict": round(sum(r["strict_correct"] for r in rows)/n, 4),
        "acc_tol": round(sum(r["tolerant_correct"] for r in rows)/n, 4),
        "no_marker": round(sum(1 for r in rows if r["first_marker_pos"] is None)/n, 4),
        "no_eos": round(sum(1 for r in rows if r["eos_pos"] < 0)/n, 4),
        "hit_max": round(sum(1 for r in rows if r["hit_max"])/n, 4),
        "len_mean": round(statistics.mean(lens), 1),
        "len_median": statistics.median(lens),
        "len_p90": srt[min(n-1, int(0.9*n))], "len_p95": srt[min(n-1, int(0.95*n))],
        "rmst_512": ST.restricted_mean_survival(lens, events, max_new),
        "marker_pos_mean": round(statistics.mean([r["first_marker_pos"] for r in wh]), 1) if wh else None,
        "rep_4gram": round(statistics.mean([r["rep_4gram"] for r in rows]), 4),
    }


def env_dict():
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
