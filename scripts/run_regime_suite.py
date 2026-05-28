#!/usr/bin/env python3
"""Run a list of workload YAMLs under one fixed server config and collect metrics.

This is the Stage 1 workhorse. It:
  1. iterates through workload files in --workload-dir (or --workload-list);
  2. for each, calls scripts/run_experiment.py once;
  3. appends a normalized record to --out (jsonl);
  4. continues across failures (records them, doesn't abort the suite);
  5. enforces a wall-time budget.

Each output row is JSONL with this shape (kept stable):
  {
    "run_id":        "run_0001",
    "workload_file": "regime_scout/candidates/seed_00_smoke.yaml",
    "workload_name": "smoke",
    "regime_hint":   "sanity",
    "config_file":   "configs/base.yaml",
    "mode":          "quick",
    "started_at":    "...",
    "finished_at":   "...",
    "duration_s":    47.3,
    "status":        "pass" | "fail" | "skip",
    "error":         null | "server_not_ready" | "benchmark_timeout" | ...,
    "metrics":       { ... full metrics.json content ... },
    "run_dir":       "experiments/tmp/regime_scout/run_0001"
  }
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from utils import (
    PROJECT_ROOT,
    append_jsonl,
    load_json,
    load_yaml,
    now_compact,
    now_str,
)
from logging_setup import setup_logger


SCRIPTS_DIR = Path(__file__).resolve().parent


def discover_workloads(workload_dir: Path, pattern: str = "*.yaml") -> list[Path]:
    return sorted(workload_dir.glob(pattern))


def run_one(server_cfg: Path, workload: Path, run_dir: Path,
            mode: str, server_timeout: int, bench_timeout: int) -> tuple[str, str | None, dict | None]:
    """Returns (status, error_or_None, metrics_or_None)."""
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_experiment.py"),
        "--config", str(server_cfg),
        "--workload", str(workload),
        "--mode", mode,
        "--out-dir", str(run_dir),
        "--server-start-timeout", str(server_timeout),
        "--benchmark-timeout", str(bench_timeout),
    ]
    # Tee orchestrator's child stdout/stderr into run_dir/orchestrator.log
    orch_log = run_dir / "orchestrator.log"
    with open(orch_log, "ab") as logf:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
    metrics_path = run_dir / f"{mode}_metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else None
    if metrics is None:
        return "fail", "no_metrics_file", None
    if metrics.get("passed"):
        return "pass", None, metrics
    err = metrics.get("parse_error") or (
        "oom" if metrics.get("oom") else
        "server_crash" if metrics.get("server_crash") else
        "timeout" if metrics.get("timeout") else
        "unknown_failure"
    )
    return "fail", err, metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="sglang server YAML config")
    ap.add_argument("--workload-dir", default="regime_scout/candidates")
    ap.add_argument("--out", default="regime_scout/outputs/raw_results.jsonl")
    ap.add_argument("--run-root", default="experiments/tmp/regime_scout",
                    help="Base dir for per-workload run directories.")
    ap.add_argument("--mode", default="quick", choices=["quick", "medium", "full"])
    ap.add_argument("--server-start-timeout", type=int, default=300)
    ap.add_argument("--benchmark-timeout", type=int, default=600)
    ap.add_argument("--wall-budget-s", type=int, default=5400,
                    help="Stop after this many wall-clock seconds.")
    ap.add_argument("--max-workloads", type=int, default=999)
    ap.add_argument("--reset", action="store_true",
                    help="Delete existing --out file before running.")
    ap.add_argument("--log", default=None,
                    help="Suite-level log file. Default: logs/regime_suite_<ts>.log")
    args = ap.parse_args()

    server_cfg = Path(args.config).resolve()
    workload_dir = Path(args.workload_dir).resolve()
    out_path = Path(args.out).resolve()
    ts = now_compact()
    run_root = Path(args.run_root).resolve() / ts
    run_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log) if args.log else PROJECT_ROOT / "logs" / f"regime_suite_{ts}.log"
    log = setup_logger("regime_suite", log_path)
    log.info(f"=== Stage 1 RegimeScout run started at {now_str()} ===")
    log.info(f"server_config={server_cfg}")
    log.info(f"workload_dir={workload_dir}")
    log.info(f"mode={args.mode}")
    log.info(f"out={out_path}")
    log.info(f"run_root={run_root}")
    log.info(f"wall_budget_s={args.wall_budget_s} server_start_timeout={args.server_start_timeout} "
             f"bench_timeout={args.benchmark_timeout}")

    if args.reset and out_path.exists():
        log.info(f"resetting {out_path}")
        out_path.unlink()

    workloads = discover_workloads(workload_dir)
    if not workloads:
        log.error(f"no workloads found in {workload_dir}")
        return 1
    workloads = workloads[:args.max_workloads]

    t0 = time.monotonic()
    log.info(f"{len(workloads)} workload(s) to run")

    summary = {"pass": 0, "fail": 0, "skip": 0}

    for i, wl in enumerate(workloads, start=1):
        elapsed = time.monotonic() - t0
        if elapsed > args.wall_budget_s:
            log.warning(f"BUDGET EXCEEDED ({elapsed:.0f}s); skipping remaining")
            for remaining in workloads[i-1:]:
                wl_doc = load_yaml(remaining)
                append_jsonl({
                    "run_id": f"run_{i:04d}",
                    "workload_file": str(remaining),
                    "workload_name": wl_doc.get("name"),
                    "regime_hint": wl_doc.get("regime_hint"),
                    "config_file": str(server_cfg),
                    "mode": args.mode,
                    "started_at": now_str(),
                    "finished_at": now_str(),
                    "duration_s": 0,
                    "status": "skip",
                    "error": "budget_exhausted",
                    "metrics": None,
                    "run_dir": None,
                }, out_path)
                summary["skip"] += 1
            break

        wl_doc = load_yaml(wl)
        name = wl_doc.get("name", wl.stem)
        run_dir = run_root / f"run_{i:04d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Persist a copy of the workload doc for traceability
        (run_dir / "workload_input.yaml").write_bytes(wl.read_bytes())

        log.info("-" * 72)
        log.info(f"[{i}/{len(workloads)}] START workload={name} "
                 f"regime_hint={wl_doc.get('regime_hint')} run_dir={run_dir}")

        ts_start = now_str()
        t_run = time.monotonic()
        try:
            status, err, metrics = run_one(
                server_cfg, wl, run_dir, args.mode,
                args.server_start_timeout, args.benchmark_timeout,
            )
        except KeyboardInterrupt:
            log.error("interrupted by user")
            return 130
        except Exception as e:
            status, err, metrics = "fail", f"orchestrator_error: {type(e).__name__}: {e}", None
            log.exception("orchestrator crash")

        dur = time.monotonic() - t_run

        record = {
            "run_id": f"run_{i:04d}",
            "workload_file": str(wl),
            "workload_name": name,
            "regime_hint": wl_doc.get("regime_hint"),
            "config_file": str(server_cfg),
            "mode": args.mode,
            "started_at": ts_start,
            "finished_at": now_str(),
            "duration_s": round(dur, 2),
            "status": status,
            "error": err,
            "metrics": metrics,
            "run_dir": str(run_dir),
        }
        append_jsonl(record, out_path)
        summary[status] = summary.get(status, 0) + 1

        # Compact one-line summary in the log
        if metrics:
            log.info(f"[{i}/{len(workloads)}] {status.upper()} ({dur:.1f}s) "
                     f"ttft_p50={metrics.get('ttft_p50_ms')} "
                     f"ttft_p95={metrics.get('ttft_p95_ms')} "
                     f"tpot_p50={metrics.get('tpot_p50_ms')} "
                     f"out_tps={metrics.get('output_throughput')} "
                     f"req_tps={metrics.get('request_throughput')} "
                     f"err={err}")
        else:
            log.warning(f"[{i}/{len(workloads)}] {status.upper()} ({dur:.1f}s) err={err}")

    elapsed = time.monotonic() - t0
    log.info("=" * 72)
    log.info(f"done in {elapsed:.0f}s. pass={summary.get('pass',0)} "
             f"fail={summary.get('fail',0)} skip={summary.get('skip',0)}")
    log.info(f"results: {out_path}")
    log.info(f"per-run dirs: {run_root}")
    log.info(f"=== Stage 1 RegimeScout run finished at {now_str()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
