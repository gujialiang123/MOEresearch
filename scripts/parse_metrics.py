#!/usr/bin/env python3
"""Normalize one sglang.bench_serving --output-file jsonl into a stable metrics schema.

bench_serving writes a single-line JSON object per run with `--output-details`.
That object contains aggregate metrics + per-request `ttfts[]`, `itls[]`, `errors[]`.

We:
  1. compute p50/p95/p99 ourselves from `ttfts[]` so we always have p95
     (bench_serving only reports p99 for TTFT/TPOT);
  2. derive a single canonical metrics schema (see SCHEMA below);
  3. detect oom/crash from the server log;
  4. fail loudly if parsing fails — never silently return zeros.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from utils import (
    detect_oom,
    detect_server_crash,
    read_jsonl,
    read_text_safe,
    save_json,
)


# -------- schema --------
# Keep these keys stable; downstream stage1 scoring depends on them.
SCHEMA_VERSION = 1
NUMERIC_KEYS = [
    "duration_s",
    "request_throughput", "input_throughput", "output_throughput",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "ttft_std_ms",
    "tpot_mean_ms", "tpot_p50_ms", "tpot_p99_ms",
    "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
    "e2e_p50_ms", "e2e_p90_ms", "e2e_p99_ms", "e2e_mean_ms",
]


def percentile(xs_ms_or_s: list[float], p: float, scale_ms: float = 1000.0) -> float | None:
    """Compute percentile (0..100). bench_serving stores ttfts/itls in seconds;
    we multiply by scale_ms=1000 to get milliseconds. Pass scale_ms=1.0 if already ms."""
    xs = [x for x in xs_ms_or_s if x is not None and not math.isnan(x)]
    if not xs:
        return None
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        v = xs_sorted[int(k)]
    else:
        v = xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (k - lo)
    return v * scale_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="bench_serving --output-file jsonl")
    ap.add_argument("--log", required=True, help="bench_serving stdout/stderr log")
    ap.add_argument("--server-log", required=True, help="sglang server stdout/stderr log")
    ap.add_argument("--out", required=True, help="normalized metrics.json")
    ap.add_argument("--mode", default="quick")
    ap.add_argument("--expected-requests", type=int, default=None,
                    help="If set, compare to completed count to detect partial runs.")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    server_log_text = read_text_safe(args.server_log)
    bench_log_text = read_text_safe(args.log)

    oom = detect_oom(server_log_text) or detect_oom(bench_log_text)
    crash = detect_server_crash(server_log_text)

    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "passed": False,
        "completed": 0,
        "expected": args.expected_requests,
        "failed_requests": None,
        "success_rate": None,
        "server_crash": crash,
        "oom": oom,
        "timeout": False,
        "parse_error": None,
        "raw_files": {
            "raw": str(raw_path),
            "benchmark_log": str(args.log),
            "server_log": str(args.server_log),
        },
    }
    for k in NUMERIC_KEYS:
        result[k] = None

    rows = read_jsonl(raw_path)
    if not rows:
        result["parse_error"] = f"empty raw jsonl: {raw_path}"
        save_json(result, args.out)
        print(json.dumps(result, indent=2))
        return 1
    if len(rows) > 1:
        # bench_serving appends one row per invocation; we cleared the file before invoking,
        # so >1 row means something weird happened.
        result["parse_error"] = f"multiple rows in raw jsonl ({len(rows)}); using last"

    row = rows[-1]

    try:
        completed = int(row.get("completed", 0) or 0)
        duration = float(row.get("duration", 0.0) or 0.0)

        result["completed"] = completed
        result["duration_s"] = duration
        result["request_throughput"] = row.get("request_throughput")
        result["input_throughput"] = row.get("input_throughput")
        result["output_throughput"] = row.get("output_throughput")

        # bench_serving aggregates (in ms already)
        result["ttft_mean_ms"]   = row.get("mean_ttft_ms")
        result["ttft_p50_ms"]    = row.get("median_ttft_ms")
        result["ttft_p99_ms"]    = row.get("p99_ttft_ms")
        result["ttft_std_ms"]    = row.get("std_ttft_ms")
        result["tpot_mean_ms"]   = row.get("mean_tpot_ms")
        result["tpot_p50_ms"]    = row.get("median_tpot_ms")
        result["tpot_p99_ms"]    = row.get("p99_tpot_ms")
        result["itl_p50_ms"]     = row.get("median_itl_ms")
        result["itl_p95_ms"]     = row.get("p95_itl_ms")
        result["itl_p99_ms"]     = row.get("p99_itl_ms")
        result["e2e_p50_ms"]     = row.get("median_e2e_latency_ms")
        result["e2e_p90_ms"]     = row.get("p90_e2e_latency_ms")
        result["e2e_p99_ms"]     = row.get("p99_e2e_latency_ms")
        result["e2e_mean_ms"]    = row.get("mean_e2e_latency_ms")

        # Compute ttft p95 ourselves from per-request ttfts[] (seconds → ms)
        ttfts_s = row.get("ttfts") or []
        if ttfts_s:
            result["ttft_p95_ms"] = percentile(ttfts_s, 95, scale_ms=1000.0)
            # sanity: if bench_serving's median disagrees with ours by >1ms, prefer ours
            our_p50 = percentile(ttfts_s, 50, scale_ms=1000.0)
            if our_p50 is not None and result["ttft_p50_ms"] is not None:
                if abs(our_p50 - result["ttft_p50_ms"]) > 1.0:
                    result["_ttft_p50_disagreement"] = {
                        "raw": result["ttft_p50_ms"],
                        "ours": our_p50,
                    }

        errors = row.get("errors") or []
        n_fail = sum(1 for e in errors if e)
        result["failed_requests"] = n_fail
        if args.expected_requests is not None and args.expected_requests > 0:
            result["success_rate"] = (completed - n_fail) / args.expected_requests
        elif completed > 0:
            result["success_rate"] = (completed - n_fail) / completed
        else:
            result["success_rate"] = 0.0

        # passed = bench actually produced metrics + no fatal signals
        result["passed"] = (
            not crash
            and not oom
            and completed > 0
            and result["ttft_p50_ms"] is not None
            and result["output_throughput"] is not None
        )
    except (KeyError, TypeError, ValueError) as e:
        result["parse_error"] = f"{type(e).__name__}: {e}"
        result["passed"] = False

    save_json(result, args.out)
    # Print compact one-line summary to stdout for the orchestrator log
    compact = {
        "mode": args.mode,
        "passed": result["passed"],
        "completed": result["completed"],
        "ttft_p50_ms": result["ttft_p50_ms"],
        "ttft_p95_ms": result["ttft_p95_ms"],
        "ttft_p99_ms": result["ttft_p99_ms"],
        "tpot_p50_ms": result["tpot_p50_ms"],
        "output_throughput": result["output_throughput"],
        "request_throughput": result["request_throughput"],
        "oom": result["oom"],
        "crash": result["server_crash"],
        "parse_error": result["parse_error"],
    }
    print(json.dumps(compact, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
