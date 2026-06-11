#!/usr/bin/env python3
"""harness/run_bench.py — main CLI entry.

Orchestrates: load spec → snapshot env → launch server → wait /health → run
e2e-bench-runner → quality gate → write schema-v1 summary.json → cleanup.

Exit codes:
  0 — ok + quality gate passed
  1 — hard failure (spec invalid, server didn't start, executor crashed)
  2 — bench succeeded but quality gate failed
  3 — bench succeeded, quality OK, but stddev_pct unreliable on any regime
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the harness package importable when run as a script (not via -m).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.spec import BenchSpec, SpecValidationError
from harness.env_snapshot import snapshot
from harness.lifecycle import ServerLifecycle, LifecycleError
from harness.executor import (
    run_bench as run_executor,
    regimes_section_from_bench_summary,
    ExecutorError,
)
from harness.quality import run_quality_gate
from harness.output import SummaryWriter, SUMMARY_SCHEMA_VERSION


EXIT_OK = 0
EXIT_HARD_FAIL = 1
EXIT_QUALITY_FAIL = 2
EXIT_UNRELIABLE = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _summary_skeleton(spec: BenchSpec, environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "ok": False,
        "submission_id": spec.submission_id,
        "spec_hash": spec.spec_hash,
        "captured_at": _now_iso(),
        "spec_resolved": {
            "server_config": spec.resolved_server_config(),
            "regimes": spec.resolved_regimes(),
            "bench": {
                "num_runs": spec.bench.num_runs,
                "reliable_stddev_pct": spec.bench.reliable_stddev_pct,
                "per_request_timeout_s": spec.bench.per_request_timeout_s,
                "backend": spec.bench.backend,
            },
            "quality_gate": {"type": spec.quality_gate.type},
        },
        "environment": environment,
        "server": {
            "startup_wall_s": 0.0,
            "first_health_at_s": 0.0,
            "log_path": "",
        },
        "regimes": {},
        "quality_gate": {"type": spec.quality_gate.type, "passed": False, "checks": {}},
        "warnings": [],
    }


def _write_failure(
    out_dir: Path,
    spec: BenchSpec | None,
    environment: dict[str, Any],
    phase: str,
    message: str,
) -> int:
    """Best-effort write of an ok=false summary; never crash."""
    try:
        if spec is None:
            # Minimal failure record when spec didn't even load.
            summary = {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "ok": False,
                "submission_id": "<spec-failed-to-load>",
                "spec_hash": "sha256:" + "0" * 64,
                "captured_at": _now_iso(),
                "spec_resolved": {
                    "server_config": {}, "regimes": {}, "bench": {},
                    "quality_gate": {},
                },
                "environment": environment,
                "server": {"startup_wall_s": 0.0, "first_health_at_s": 0.0, "log_path": ""},
                "regimes": {},
                "quality_gate": {"type": "sanity", "passed": False, "checks": {}},
                "warnings": [],
                "error": {"phase": phase, "message": message},
            }
        else:
            summary = _summary_skeleton(spec, environment)
            summary["error"] = {"phase": phase, "message": message}
        SummaryWriter(out_dir).write(summary)
        print(f"[run_bench] failure recorded → {out_dir / 'summary.json'}", flush=True)
    except Exception as e:
        print(f"[run_bench] CRITICAL: failed to write failure summary: {e}", flush=True)
    return EXIT_HARD_FAIL


def main() -> int:
    ap = argparse.ArgumentParser(description="regime-bench-harness v1")
    ap.add_argument("--spec", required=True, help="Path to bench-spec.yaml")
    ap.add_argument("--out-dir", required=True, help="Output directory for summary.json + per_run/")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Resolve spec + print spec_hash + exit (no server launch).",
    )
    ap.add_argument(
        "--no-server-start", action="store_true",
        help="Assume server already running at spec.server.base_url (skip launch + cleanup).",
    )
    ap.add_argument(
        "--keep-server", action="store_true",
        help="Don't kill the launched server on exit (debug; user must clean up).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load spec -----
    try:
        spec = BenchSpec.load(args.spec)
    except SpecValidationError as e:
        print(f"[run_bench] spec invalid: {e}", flush=True)
        # Use minimal environment since spec didn't even load.
        env_dummy = {
            "hostname": "", "gpu": {"name": "", "uuid": "", "id": 0, "sm": ""},
            "driver": "", "cuda": "", "engine_version": {}, "git": {"commit": "", "dirty": False},
        }
        return _write_failure(out_dir, None, env_dummy, "spec_validation", str(e))

    print(f"[run_bench] spec_hash = {spec.spec_hash}", flush=True)
    print(f"[run_bench] submission_id = {spec.submission_id}", flush=True)

    if args.dry_run:
        print("[run_bench] --dry-run: spec resolved OK, exiting.")
        # Still write a "dry run" summary so the output dir isn't empty.
        # Skip env snapshot to keep dry-run fast.
        return EXIT_OK

    # ----- Environment snapshot -----
    gpu_id = int(spec.resolved_server_config().get("_gpu_id", 0))
    environment = snapshot(gpu_id=gpu_id, conda_env=spec.server.conda_env)

    # ----- Run pipeline -----
    summary = _summary_skeleton(spec, environment)

    lifecycle: ServerLifecycle | None = None
    if not args.no_server_start:
        lifecycle = ServerLifecycle(
            resolved_server_config=spec.resolved_server_config(),
            conda_env=spec.server.conda_env,
            base_url=spec.server.base_url,
            health_url=spec.server.health_url,
            startup_timeout_s=spec.server.startup_timeout_s,
            out_dir=out_dir,
            keep_server=args.keep_server,
        )

    try:
        # Server lifecycle
        if lifecycle:
            try:
                lifecycle.start()
                lifecycle.wait_healthy()
            except LifecycleError as e:
                print(f"[run_bench] server startup failed: {e}", flush=True)
                return _write_failure(
                    out_dir, spec, environment, "server_startup", str(e)
                )
            summary["server"] = {
                "startup_wall_s": round(lifecycle.startup_wall_s(), 2),
                "first_health_at_s": round(lifecycle.startup_wall_s(), 2),
                "log_path": str(lifecycle.server_log_path.relative_to(out_dir.parent.parent)
                                if lifecycle.server_log_path.is_relative_to(out_dir.parent.parent)
                                else lifecycle.server_log_path),
            }

        # Executor
        try:
            bench_summary = run_executor(
                base_url=spec.server.base_url,
                backend=spec.bench.backend,
                submission_id=spec.submission_id,
                num_runs=spec.bench.num_runs,
                resolved_regimes=spec.resolved_regimes(),
                out_dir=out_dir,
            )
            summary["regimes"] = regimes_section_from_bench_summary(bench_summary)
        except ExecutorError as e:
            print(f"[run_bench] executor failed: {e}", flush=True)
            return _write_failure(out_dir, spec, environment, "bench_execution", str(e))

        # Quality gate
        per_run_dir = out_dir / "per_run"
        summary["quality_gate"] = run_quality_gate(spec.quality_gate.type, per_run_dir)

        # Reliability check (independent of quality gate)
        unreliable_regimes = [
            r_id for r_id, r in summary["regimes"].items()
            if isinstance(r, dict) and r.get("reliable") is False
        ]
        if unreliable_regimes:
            summary["warnings"].append(
                f"unreliable stddev_pct on regimes: {unreliable_regimes} "
                f"(threshold={spec.bench.reliable_stddev_pct}%)"
            )

        summary["ok"] = True
        SummaryWriter(out_dir).write(summary)
        print(f"[run_bench] OK → {out_dir / 'summary.json'}", flush=True)

        if not summary["quality_gate"]["passed"]:
            return EXIT_QUALITY_FAIL
        if unreliable_regimes:
            return EXIT_UNRELIABLE
        return EXIT_OK

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[run_bench] unexpected error: {e}\n{tb}", flush=True)
        return _write_failure(out_dir, spec, environment, "unexpected", f"{e}\n{tb}")
    finally:
        if lifecycle:
            # __exit__ would handle this via context manager, but we're using
            # manual control flow. Call stop() unless keep_server.
            if not args.keep_server:
                lifecycle.stop()


if __name__ == "__main__":
    sys.exit(main())
