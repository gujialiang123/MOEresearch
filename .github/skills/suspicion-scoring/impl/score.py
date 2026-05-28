#!/usr/bin/env python3
"""Skill impl: suspicion-scoring (v2). Composes upstream skills.

Pipeline per run:
  1. mine server.log → server_features.json
  2. classify          → classification.json
  3. score             = w*local_nonlinearity_v2
                        + w*tail_latency_ratio_v2 (noise-aware)
                        + w*server_log_signal
                        + w*failure_class_score
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR  = Path(__file__).resolve().parents[4] / "scripts"
SKILLS_DIR   = Path(__file__).resolve().parents[3] / "skills"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SKILLS_DIR / "noise-aware-scoring" / "impl"))
sys.path.insert(0, str(SKILLS_DIR / "failure-classification" / "impl"))
sys.path.insert(0, str(SKILLS_DIR / "server-log-mining" / "impl"))
from utils import append_jsonl, load_yaml, read_jsonl   # noqa: E402
from threshold import adjusted_threshold, load_baseline  # noqa: E402
from classify import classify                            # noqa: E402
from parse_server_log import mine                        # noqa: E402


DEFAULT_WEIGHTS = {
    "local_nonlinearity_primary":   0.20,
    "tail_latency_ratio":            0.15,
    "server_log_signal":             0.30,
    "failure_class":                 0.25,
    "local_nonlinearity_secondary":  0.10,
}

FAILURE_CLASS_SCORE = {
    "clean_pass":            0.0,
    "load_shed_concurrency": 0.7,
    "partial_success":       0.8,
    "near_failure_retract":  0.85,
    "near_failure_kv":       0.9,
    "oom":                   1.0,
    "server_crash":          1.0,
    "benchmark_timeout":     0.9,
    "parse_error":           0.3,
    "unknown_failure":       0.3,
}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# -------- score components --------

def server_log_signal(features: dict) -> tuple[float, dict]:
    if not features:
        return 0.0, {"reason": "no_features"}
    score = 0.0
    contributions = {}
    if features.get("cuda_graph_too_small"):
        score += 1.0; contributions["cuda_graph_too_small"] = 1.0
    if features.get("at_capacity"):
        score += 0.9; contributions["at_capacity"] = 0.9
    elif features.get("near_capacity"):
        score += 0.4; contributions["near_capacity"] = 0.4
    if features.get("concurrency_capped"):
        score += 0.7; contributions["concurrency_capped"] = 0.7
    if (features.get("retract_events") or 0) > 0:
        score += 0.8; contributions["retract_events"] = 0.8
    if (features.get("kv_pool_full_events") or 0) > 0:
        score += 1.0; contributions["kv_pool_full_events"] = 1.0
    if features.get("max_running_above_cuda_graph"):
        score += 0.5; contributions["max_running_above_cuda_graph"] = 0.5
    return clamp01(score), {
        "contributions": contributions,
        "peak_running_reqs": features.get("peak_running_reqs"),
        "peak_queue_reqs":   features.get("peak_queue_reqs"),
        "max_running_requests": features.get("max_running_requests"),
        "peak_token_usage":  features.get("peak_token_usage"),
    }


def tail_latency_ratio_v2(metrics: dict, baseline: dict, base_threshold: float = 3.0) -> tuple[float, dict]:
    if not metrics or not metrics.get("passed"):
        return 0.0, {"reason": "not_passed"}
    ratios = []
    for p50_k, pX_k in [
        ("ttft_p50_ms", "ttft_p99_ms"),
        ("tpot_p50_ms", "tpot_p99_ms"),
        ("itl_p50_ms",  "itl_p99_ms"),
    ]:
        a, b = metrics.get(p50_k), metrics.get(pX_k)
        if a and b and a > 0:
            ratios.append((p50_k.split("_")[0], b / a))
    if not ratios:
        return 0.0, {"reason": "no_ratios"}

    max_kind, max_r = max(ratios, key=lambda t: t[1])
    metric_for_adjust = f"{max_kind}_p99_ms"
    threshold = adjusted_threshold(metric_for_adjust, base_threshold, baseline, k=2.0)
    score = clamp01((max_r - 1.0) / max(threshold - 1.0, 1e-9))
    return score, {
        "ratios": {k: round(v, 2) for k, v in ratios},
        "max_ratio": round(max_r, 2),
        "max_kind": max_kind,
        "base_threshold": base_threshold,
        "noise_adjusted_threshold": round(threshold, 3),
    }


def failure_class_component(classification_doc: dict) -> tuple[float, dict]:
    cls = (classification_doc or {}).get("classification") or "unknown_failure"
    return FAILURE_CLASS_SCORE.get(cls, 0.3), {
        "classification": cls,
        "evidence": (classification_doc or {}).get("evidence", {}),
    }


# Local nonlinearity reuses the v1 logic but is now optional and clearly
# attributed.

def local_nonlinearity_v2(rec: dict, all_recs: list[dict],
                          primary_metric: str, direction: str,
                          large_jump: float) -> tuple[float, dict]:
    if rec.get("status") != "pass":
        return 0.0, {"reason": "not_passed"}
    me = rec["metrics"].get(primary_metric)
    if me is None or me <= 0:
        return 0.0, {"reason": "no_primary_metric"}
    same = []
    for o in all_recs:
        if o is rec or o.get("status") != "pass":
            continue
        if o.get("regime_hint") != rec.get("regime_hint"):
            continue
        ov = o["metrics"].get(primary_metric)
        if ov is None or ov <= 0:
            continue
        # distance: include axis we know about
        wl_me = load_yaml(Path(rec["workload_file"]))
        wl_o  = load_yaml(Path(o["workload_file"]))
        dist = 0.0
        for axis, getter in [
            ("input_len", lambda w: (w.get("dataset") or {}).get("random_input_len")
                                   or (w.get("dataset") or {}).get("gsp_system_prompt_len")),
            ("output_len", lambda w: (w.get("dataset") or {}).get("random_output_len")
                                    or (w.get("dataset") or {}).get("gsp_output_len")),
            ("max_concurrency", lambda w: (w.get("traffic") or {}).get("max_concurrency")),
        ]:
            a, b = getter(wl_me), getter(wl_o)
            if a and b:
                dist += abs(math.log2(b / a))
        same.append((dist, ov, o.get("workload_name")))
    if not same:
        return 0.0, {"reason": "no_same_regime_neighbor"}
    same.sort(key=lambda t: t[0])
    nb_dist, nb_v, nb_name = same[0]
    if direction == "lower":
        ratio = me / nb_v
    else:
        ratio = nb_v / max(me, 1e-9)
    excess = max(0.0, ratio - 1.0) / max(large_jump - 1.0, 1e-9)
    return clamp01(excess), {
        "neighbor_workload": nb_name,
        "neighbor_dist_log2": round(nb_dist, 3),
        "neighbor_metric": nb_v,
        "this_metric": me,
        "ratio_worse_than_nb": round(ratio, 3),
    }


def pick_primary(rec: dict) -> tuple[str, str]:
    h = (rec.get("regime_hint") or "").lower()
    if "decode" in h:
        return "output_throughput", "higher"
    return "ttft_p95_ms", "lower"


def ensure_features_and_class(rec: dict, force: bool) -> tuple[dict, dict]:
    """For one rec, make sure server_features.json + classification.json exist
    in its run_dir. If missing, compute them via the upstream skills."""
    run_dir = Path(rec.get("run_dir") or "")
    features = {}
    classification = {}
    if not run_dir.exists():
        return features, classification

    feat_path = run_dir / "server_features.json"
    if not feat_path.exists() or force:
        srv_log = run_dir / "server.log"
        feat_doc = mine(srv_log)
        feat_path.write_text(json.dumps(feat_doc, indent=2))
    feat_doc = json.loads(feat_path.read_text())
    features = feat_doc.get("fields", {})

    cls_path = run_dir / "classification.json"
    if not cls_path.exists() or force:
        cls_doc = classify(rec.get("metrics"), features)
        cls_path.write_text(json.dumps(cls_doc, indent=2))
    classification = json.loads(cls_path.read_text())

    return features, classification


def score_one(rec: dict, all_recs: list[dict], baseline: dict,
              weights: dict, force_mine: bool) -> dict:
    features, classification = ensure_features_and_class(rec, force_mine)

    primary, direction = pick_primary(rec)

    out: dict = {
        "run_id": rec["run_id"],
        "workload_file": rec["workload_file"],
        "workload_name": rec["workload_name"],
        "regime_hint":   rec["regime_hint"],
        "status":        rec["status"],
        "error":         rec.get("error"),
        "primary_metric":   primary,
        "primary_direction":direction,
        "primary_value":    (rec.get("metrics") or {}).get(primary),
        "classification":   classification.get("classification"),
    }

    sls_score, sls_evi = server_log_signal(features)
    fc_score,  fc_evi  = failure_class_component(classification)
    tlr_score, tlr_evi = tail_latency_ratio_v2(rec.get("metrics") or {}, baseline)
    ln_score,  ln_evi  = local_nonlinearity_v2(rec, all_recs, primary, direction, 2.0)

    components = {
        "server_log_signal": {
            "weight": weights["server_log_signal"],
            "score":  round(sls_score, 4),
            "evidence": sls_evi,
        },
        "failure_class": {
            "weight": weights["failure_class"],
            "score":  round(fc_score, 4),
            "evidence": fc_evi,
        },
        "tail_latency_ratio": {
            "weight": weights["tail_latency_ratio"],
            "score":  round(tlr_score, 4),
            "evidence": tlr_evi,
        },
        "local_nonlinearity_primary": {
            "weight": weights["local_nonlinearity_primary"],
            "score":  round(ln_score, 4),
            "evidence": ln_evi,
        },
        "local_nonlinearity_secondary": {
            "weight": weights["local_nonlinearity_secondary"],
            "score":  0.0,
            "evidence": {"reason": "v2 stub"},
        },
    }
    total = sum(c["weight"] * c["score"] for c in components.values())
    out["score"] = round(total, 4)
    out["components"] = components
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",         default="regime_scout/outputs/raw_results.jsonl")
    ap.add_argument("--noise-baseline", default="experiments/noise_baseline.json")
    ap.add_argument("--out",         default="regime_scout/outputs/suspicious_cases.jsonl")
    ap.add_argument("--force-mine", action="store_true",
                    help="Regenerate server_features.json even if present")
    args = ap.parse_args()

    rows = read_jsonl(args.raw)
    if not rows:
        print(f"[score v2] no rows in {args.raw}", file=sys.stderr)
        return 1

    baseline = load_baseline(args.noise_baseline)
    if not baseline:
        print(f"[score v2] WARNING: no noise baseline at {args.noise_baseline}; "
              f"using v1 hardcoded thresholds.")

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    scored = [score_one(r, rows, baseline, DEFAULT_WEIGHTS, args.force_mine) for r in rows]
    scored.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
    for s in scored:
        append_jsonl(s, out_path)

    print(f"[score v2] wrote {len(scored)} rows → {out_path}")
    print(f"[score v2] top 10:")
    for s in scored[:10]:
        cls = s.get("classification") or "—"
        print(f"  {s['workload_name']:42s} score={s['score']:.3f}  class={cls:25s}  "
              f"primary={s.get('primary_metric')}={s.get('primary_value')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
