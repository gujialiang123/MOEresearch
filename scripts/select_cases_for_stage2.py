#!/usr/bin/env python3
"""Stage 1: pick top suspicious cases for Stage 2/3 handoff.

Rules (deterministic; agent has no creative freedom here):
  1. Score >= selected_case_score threshold.
  2. At most one case per regime cluster (diversity).
  3. At most `--max-cases` total.
  4. Hard failures (oom/crash/timeout) always selected (no threshold).
  5. Each selected case gets its own `experiments/regimes/cases/SNNN/` dir with:
       case.json            (frozen contract per supplement §9)
       workload.yaml        (frozen copy of the workload file)
       metrics.json         (copy of the stage1 metrics for traceability)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import append_jsonl, load_json, load_yaml, now_str, read_jsonl, save_json, save_yaml


def select(scored: list[dict], cluster_by_hint: dict[str, dict],
           threshold: float, max_cases: int) -> list[dict]:
    chosen: list[dict] = []
    seen_clusters: set[str] = set()

    # Hard failures first
    for s in scored:
        if s.get("status") == "fail" and s.get("failure_kind") in {
            "oom", "server_crash", "timeout"
        }:
            chosen.append(s)
            seen_clusters.add(s.get("regime_hint"))
            if len(chosen) >= max_cases:
                return chosen

    # Then top-scored passed runs, one per cluster
    for s in scored:
        if s.get("status") != "pass":
            continue
        score = s.get("score") or 0.0
        if score < threshold:
            continue
        hint = s.get("regime_hint")
        if hint in seen_clusters:
            continue
        chosen.append(s)
        seen_clusters.add(hint)
        if len(chosen) >= max_cases:
            break
    return chosen


def build_case(idx: int, scored: dict, raw_row: dict, server_cfg_path: Path,
               cases_root: Path) -> tuple[Path, dict]:
    case_id = f"S{idx:03d}"
    case_dir = cases_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    workload_src = Path(raw_row["workload_file"])
    workload_doc = load_yaml(workload_src) if workload_src.exists() else {}

    metrics = raw_row.get("metrics") or {}

    # Frozen workload copy (rename to `workload.yaml` inside case dir)
    save_yaml(workload_doc, case_dir / "workload.yaml")

    if metrics:
        save_json(metrics, case_dir / "metrics.json")

    components = scored.get("components", {})
    suspected = []
    if components.get("tail_latency_ratio", {}).get("score", 0) > 0.5:
        suspected.append("tail_latency")
    if components.get("local_nonlinearity", {}).get("score", 0) > 0.5:
        suspected.append("metric_jump_vs_neighbor")
    if components.get("failure_nearness", {}).get("score", 0) > 0.5:
        suspected.append("near_failure_or_failed")
    if not suspected:
        suspected = ["unknown"]

    suggested_first_knobs = []
    hint = (raw_row.get("regime_hint") or "").lower()
    if "prefill" in hint:
        suggested_first_knobs = ["chunked-prefill-size", "max-prefill-tokens",
                                 "schedule-conservativeness"]
    elif "decode" in hint:
        suggested_first_knobs = ["num-continuous-decode-steps", "max-running-requests",
                                 "cuda-graph-max-bs"]
    elif "prefix" in hint or "cache" in hint:
        suggested_first_knobs = ["schedule-policy", "disable-radix-cache",
                                 "schedule-conservativeness"]
    elif "scheduler" in hint or "short" in hint:
        suggested_first_knobs = ["cuda-graph-max-bs", "max-running-requests",
                                 "num-continuous-decode-steps"]
    else:
        suggested_first_knobs = ["chunked-prefill-size", "max-running-requests"]

    server_cfg = load_yaml(server_cfg_path)
    case = {
        "case_id": case_id,
        "regime_id": f"R_{raw_row.get('regime_hint')}",
        "created_at": now_str(),
        "model_path": server_cfg.get("model-path"),
        "hardware": "H200",
        "sglang_version": None,
        "baseline_config": str(server_cfg_path),
        "workload_file": str((case_dir / "workload.yaml").resolve()),
        "stage1_run_dir": raw_row.get("run_dir"),

        "workload_summary": {
            "dataset_name": (workload_doc.get("dataset") or {}).get("name"),
            "regime_hint": raw_row.get("regime_hint"),
            "max_concurrency": (workload_doc.get("traffic") or {}).get("max_concurrency"),
            "num_prompts": (workload_doc.get("traffic") or {}).get("num_prompts"),
            "cache_mode": (workload_doc.get("cache") or {}).get("mode"),
        },

        "symptom": {
            "type": "score_threshold_or_failure",
            "metric": scored.get("primary_metric"),
            "direction": scored.get("primary_direction"),
            "observed_value": scored.get("primary_value"),
            "description": (
                f"score {scored.get('score'):.3f} on {scored.get('primary_metric')}"
                if scored.get("score") is not None else "hard failure"
            ),
        },

        "evidence": {
            "status": raw_row.get("status"),
            "error": raw_row.get("error"),
            "components": components,
        },

        "diagnostics": {
            "suspected_categories": suspected,
        },

        "suspicion_score": scored.get("score"),

        "recommended_stage2": {
            "action": "diagnose_then_fix",
            "primary_metric": scored.get("primary_metric"),
            "primary_direction": scored.get("primary_direction"),
            "minimum_improvement_pct": 10.0,
            "suggested_first_knobs": suggested_first_knobs,
        },

        "frozen": True,
    }
    save_json(case, case_dir / "case.json")
    return case_dir, case


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",          default="regime_scout/outputs/raw_results.jsonl")
    ap.add_argument("--suspicious",   default="regime_scout/outputs/suspicious_cases.jsonl")
    ap.add_argument("--regime-map",   default="regime_scout/outputs/regime_map.json")
    ap.add_argument("--search-space", default="regime_scout/search_space.yaml")
    ap.add_argument("--server-config",default="configs/base.yaml")
    ap.add_argument("--out",          default="regime_scout/outputs/selected_cases.jsonl")
    ap.add_argument("--cases-root",   default="experiments/regimes/cases")
    ap.add_argument("--max-cases",    type=int, default=5)
    ap.add_argument("--threshold",    type=float, default=None,
                    help="Override selected_case_score threshold from search_space.")
    args = ap.parse_args()

    scored = read_jsonl(args.suspicious)
    raw = {r["run_id"]: r for r in read_jsonl(args.raw)}
    if not scored:
        print(f"[select_cases] no scored rows in {args.suspicious}", file=sys.stderr)
        return 1

    space = load_yaml(args.search_space) if Path(args.search_space).exists() else {}
    thresh = (space.get("scoring", {}).get("thresholds", {}).get("selected_case_score", 0.55))
    if args.threshold is not None:
        thresh = args.threshold

    cluster_json = load_json(args.regime_map) if Path(args.regime_map).exists() else {}
    cluster_by_hint = {c["regime_hint"]: c for c in cluster_json.get("clusters", [])}

    chosen = select(scored, cluster_by_hint, thresh, args.max_cases)

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    cases_root = Path(args.cases_root)
    cases_root.mkdir(parents=True, exist_ok=True)

    if not chosen:
        print(f"[select_cases] nothing met threshold {thresh}; "
              f"top scores: " + ", ".join(
                  f"{s['workload_name']}={s.get('score')}" for s in scored[:3]),
              file=sys.stderr)
        # Still emit empty file so downstream knows we ran
        out_path.touch()
        return 0

    for idx, s in enumerate(chosen, start=1):
        raw_row = raw.get(s["run_id"], {})
        case_dir, case = build_case(idx, s, raw_row, Path(args.server_config), cases_root)
        rec = {
            "case_id": case["case_id"],
            "regime_id": case["regime_id"],
            "case_dir": str(case_dir.resolve()),
            "case_json": str((case_dir / "case.json").resolve()),
            "workload_yaml": str((case_dir / "workload.yaml").resolve()),
            "suspicion_score": case["suspicion_score"],
            "primary_metric": case["recommended_stage2"]["primary_metric"],
        }
        append_jsonl(rec, out_path)
        print(f"[select_cases] {case['case_id']}: {raw_row.get('workload_name')} "
              f"(score={s.get('score')}) → {case_dir}")

    print(f"[select_cases] {len(chosen)} case(s) → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
