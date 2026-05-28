#!/usr/bin/env python3
"""Stage 1: score suspiciousness of each workload run.

Reads `raw_results.jsonl` (one row per workload), then for each PASSED run computes:

  score = w1*local_nonlinearity
        + w2*tail_latency_ratio_score
        + w3*diagnostic_sensitivity     (set to 0 for v0.2 MVP — no toggle pairs yet)
        + w4*stack_gap                   (set to 0 for v0.2 MVP)
        + w5*failure_nearness
        + w6*variance_score              (set to 0 for v0.2 MVP — single repeat)

Failed runs still get a record with `score: null` and `failure_kind` populated,
so they show up in the regime map.

OUTPUT: regime_scout/outputs/suspicious_cases.jsonl
        one row per workload, sorted by score descending.

Score components are documented inline. Each row records the raw component
values so a later pass (or a human) can audit where the score came from.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from utils import append_jsonl, load_yaml, read_jsonl


# -------- helper math --------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    return num / den


# -------- workload feature signature for neighbor lookup --------

def workload_signature(rec: dict) -> dict:
    """Pull workload axis values into a flat dict for neighbor comparison."""
    wl_path = Path(rec["workload_file"])
    wl = load_yaml(wl_path) if wl_path.exists() else {}
    ds = wl.get("dataset", {})
    traffic = wl.get("traffic", {})
    return {
        "dataset_name": ds.get("name"),
        "input_len": ds.get("random_input_len") or ds.get("gsp_system_prompt_len"),
        "output_len": ds.get("random_output_len") or ds.get("gsp_output_len"),
        "max_concurrency": traffic.get("max_concurrency"),
        "num_prompts": traffic.get("num_prompts"),
        "regime_hint": rec.get("regime_hint"),
    }


# -------- score components --------

def local_nonlinearity(rec: dict, all_recs: list[dict],
                       primary_metric: str, direction: str,
                       large_jump_threshold: float) -> tuple[float, dict]:
    """How sharply does the primary metric change against a NEAREST NEIGHBOR
    that shares dataset type and regime hint but differs along one axis?

    For MVP we use a coarse same-regime-hint neighbor: among PASSED runs sharing
    `regime_hint`, find the one whose (input_len, output_len, max_concurrency)
    is closest in log-space; compare metric ratio.
    """
    if rec.get("status") != "pass":
        return 0.0, {"reason": "not_passed"}
    sig = workload_signature(rec)
    metric = rec["metrics"].get(primary_metric)
    if metric is None or metric <= 0:
        return 0.0, {"reason": "no_primary_metric"}

    same = []
    for other in all_recs:
        if other is rec or other.get("status") != "pass":
            continue
        if other.get("regime_hint") != rec.get("regime_hint"):
            continue
        sig_o = workload_signature(other)
        if sig_o.get("dataset_name") != sig.get("dataset_name"):
            continue
        other_metric = other["metrics"].get(primary_metric)
        if other_metric is None or other_metric <= 0:
            continue
        # distance in log-axis: sum of |log ratio| for the 3 axes
        dist = 0.0
        for axis in ("input_len", "output_len", "max_concurrency"):
            a, b = sig.get(axis), sig_o.get(axis)
            if not a or not b:
                continue
            dist += abs(math.log2(b / a))
        same.append((dist, other_metric, other.get("workload_name")))

    if not same:
        return 0.0, {"reason": "no_same_regime_neighbor"}

    same.sort(key=lambda t: t[0])
    nb_dist, nb_metric, nb_name = same[0]

    if direction == "lower":
        ratio = metric / nb_metric
    else:
        ratio = nb_metric / max(metric, 1e-9)
    # ratio > 1 ⇒ this workload is worse than neighbor
    excess = max(0.0, ratio - 1.0) / max(large_jump_threshold - 1.0, 1e-9)
    score = clamp01(excess)
    return score, {
        "neighbor_workload": nb_name,
        "neighbor_dist_log2": round(nb_dist, 3),
        "neighbor_metric": nb_metric,
        "this_metric": metric,
        "ratio_worse_than_nb": round(ratio, 3),
        "direction": direction,
    }


def tail_latency_ratio_score(rec: dict, high_tail_threshold: float) -> tuple[float, dict]:
    if rec.get("status") != "pass":
        return 0.0, {}
    m = rec["metrics"]
    ratios = []
    for p50_k, pX_k in [
        ("ttft_p50_ms", "ttft_p99_ms"),
        ("tpot_p50_ms", "tpot_p99_ms"),
        ("itl_p50_ms",  "itl_p99_ms"),
    ]:
        r = safe_div(m.get(pX_k), m.get(p50_k))
        if r is not None:
            ratios.append((pX_k.split("_")[0], r))

    if not ratios:
        return 0.0, {"reason": "no_ratios"}

    max_r = max(r for _, r in ratios)
    # Map (1, high_tail_threshold) → (0, 1)
    score = clamp01((max_r - 1.0) / max(high_tail_threshold - 1.0, 1e-9))
    return score, {
        "ratios": {k: round(v, 2) for k, v in ratios},
        "max_ratio": round(max_r, 2),
        "threshold": high_tail_threshold,
    }


def failure_nearness(rec: dict) -> tuple[float, dict]:
    if rec.get("status") == "fail":
        m = rec.get("metrics") or {}
        if m.get("oom") or m.get("server_crash") or m.get("timeout"):
            return 1.0, {"hard_failure": True, "err": rec.get("error")}
        return 0.7, {"soft_failure": True, "err": rec.get("error")}
    m = rec.get("metrics") or {}
    sr = m.get("success_rate")
    if sr is not None and sr < 1.0:
        return clamp01(1.0 - sr), {"success_rate": sr}
    # If max_concurrency or input_len is at the high end of the suite, we
    # can't tell without more data. v0.2 leaves this at 0.
    return 0.0, {}


# -------- per-run scorer --------

def pick_primary(rec: dict) -> tuple[str, str]:
    """Default primary metric per regime_hint. Lower-is-better unless noted."""
    hint = (rec.get("regime_hint") or "").lower()
    if "decode" in hint:
        return "output_throughput", "higher"
    if "scheduler" in hint or "cache_churn" in hint or "prefill" in hint or "prefix" in hint:
        return "ttft_p95_ms", "lower"
    return "ttft_p95_ms", "lower"


def score_one(rec: dict, all_recs: list[dict], weights: dict, thresh: dict) -> dict:
    primary, direction = pick_primary(rec)
    out: dict = {
        "run_id": rec["run_id"],
        "workload_file": rec["workload_file"],
        "workload_name": rec["workload_name"],
        "regime_hint": rec["regime_hint"],
        "status": rec["status"],
        "error": rec.get("error"),
        "primary_metric": primary,
        "primary_direction": direction,
    }
    if rec.get("metrics"):
        out["primary_value"] = rec["metrics"].get(primary)
    else:
        out["primary_value"] = None

    if rec.get("status") == "fail":
        f_score, f_evi = failure_nearness(rec)
        out["score"] = round(weights["failure_nearness"] * f_score, 4)
        out["components"] = {
            "failure_nearness": {"score": f_score, "evidence": f_evi},
        }
        out["failure_kind"] = rec.get("error")
        return out

    ln_score, ln_evi = local_nonlinearity(
        rec, all_recs, primary, direction,
        thresh.get("large_metric_jump", 2.0),
    )
    tl_score, tl_evi = tail_latency_ratio_score(
        rec, thresh.get("high_tail_ratio", 3.0),
    )
    fn_score, fn_evi = failure_nearness(rec)

    components = {
        "local_nonlinearity":     {"weight": weights["local_nonlinearity"],
                                   "score": round(ln_score, 4), "evidence": ln_evi},
        "tail_latency_ratio":     {"weight": weights["tail_latency_ratio"],
                                   "score": round(tl_score, 4), "evidence": tl_evi},
        "diagnostic_sensitivity": {"weight": weights["diagnostic_sensitivity"],
                                   "score": 0.0, "evidence": {"reason": "not_implemented_in_v0.2"}},
        "stack_gap":              {"weight": weights["stack_gap"],
                                   "score": 0.0, "evidence": {"reason": "not_implemented_in_v0.2"}},
        "failure_nearness":       {"weight": weights["failure_nearness"],
                                   "score": round(fn_score, 4), "evidence": fn_evi},
        "variance":               {"weight": weights["variance"],
                                   "score": 0.0, "evidence": {"reason": "single_repeat_in_v0.2"}},
    }
    total = sum(c["weight"] * c["score"] for c in components.values())
    out["score"] = round(total, 4)
    out["components"] = components
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",         default="regime_scout/outputs/raw_results.jsonl")
    ap.add_argument("--search-space",default="regime_scout/search_space.yaml")
    ap.add_argument("--out",         default="regime_scout/outputs/suspicious_cases.jsonl")
    args = ap.parse_args()

    space = load_yaml(args.search_space)
    weights = space.get("scoring", {}).get("suspicion_weights", {})
    thresh  = space.get("scoring", {}).get("thresholds", {})
    # default weights if missing
    default_w = {
        "local_nonlinearity":     0.30,
        "tail_latency_ratio":     0.25,
        "diagnostic_sensitivity": 0.15,
        "stack_gap":              0.05,
        "failure_nearness":       0.15,
        "variance":               0.10,
    }
    for k, v in default_w.items():
        weights.setdefault(k, v)

    rows = read_jsonl(args.raw)
    if not rows:
        print(f"[score_suspicion] no rows in {args.raw}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    scored = [score_one(r, rows, weights, thresh) for r in rows]
    scored.sort(key=lambda x: (x.get("score") or 0.0), reverse=True)

    for s in scored:
        append_jsonl(s, out_path)

    print(f"[score_suspicion] wrote {len(scored)} scored row(s) → {out_path}")
    print(f"[score_suspicion] top 5:")
    for s in scored[:5]:
        print(f"  {s['workload_name']:40s} score={s['score']} status={s['status']} "
              f"primary={s['primary_metric']}={s.get('primary_value')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
