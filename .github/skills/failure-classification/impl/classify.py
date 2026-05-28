#!/usr/bin/env python3
"""Skill impl: failure-classification. Pure-function classifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCHEMA_VERSION = 1

ENUM = [
    "clean_pass",
    "near_failure_kv", "near_failure_retract", "partial_success",
    "load_shed_concurrency",
    "oom", "server_crash", "benchmark_timeout", "parse_error",
    "unknown_failure",
]


def classify(metrics: dict | None, features: dict, bench_log: str = "") -> dict:
    ev: dict = {}
    cls = "unknown_failure"

    if metrics is None or not metrics.get("passed", False):
        if features.get("oom_events", 0) > 0:
            cls = "oom"
        elif features.get("crash_events", 0) > 0:
            cls = "server_crash"
        elif metrics and metrics.get("timeout"):
            cls = "benchmark_timeout"
        elif metrics and metrics.get("parse_error"):
            cls = "parse_error"
            ev["parse_error"] = metrics["parse_error"]
        elif metrics is None and not features:
            ev["reason"] = "no_data"
        ev["passed"] = bool(metrics.get("passed")) if metrics else None
    else:
        ev["passed"] = True
        ev["success_rate"] = metrics.get("success_rate")
        ev["peak_token_usage"] = features.get("peak_token_usage")
        ev["retract_events"] = features.get("retract_events", 0)
        ev["concurrency_capped"] = features.get("concurrency_capped")
        ev["max_running_requests"] = features.get("max_running_requests")
        ev["peak_running_reqs"] = features.get("peak_running_reqs")
        ev["peak_queue_reqs"] = features.get("peak_queue_reqs")
        if features.get("at_capacity"):
            cls = "near_failure_kv"
        elif features.get("retract_events", 0) > 0:
            cls = "near_failure_retract"
        elif (metrics.get("success_rate") or 1.0) < 1.0:
            cls = "partial_success"
        elif features.get("concurrency_capped"):
            cls = "load_shed_concurrency"
        else:
            cls = "clean_pass"

    return {
        "schema_version": SCHEMA_VERSION,
        "classification": cls,
        "_classification_enum": ENUM,
        "evidence": ev,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None, help="parse_metrics.py output (or omit)")
    ap.add_argument("--features", required=True, help="server-log-mining output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    metrics = None
    if args.metrics and Path(args.metrics).exists():
        with open(args.metrics) as fh:
            metrics = json.load(fh)
    with open(args.features) as fh:
        feat_doc = json.load(fh)
    features = feat_doc.get("fields", {})

    result = classify(metrics, features)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
