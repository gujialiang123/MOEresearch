#!/usr/bin/env python3
"""Stage 1 end-to-end orchestrator.

Pipeline:
    1. generate_seed_suite.py        seed_suite.yaml → candidates/*.yaml
    2. run_regime_suite.py            candidates/*.yaml → raw_results.jsonl
    3. score_suspicion.py             raw_results.jsonl → suspicious_cases.jsonl
    4. cluster_regimes.py             → regime_map.{md,json}
    5. select_cases_for_stage2.py     → selected_cases.jsonl + cases/SNNN/

Each step is invoked as a subprocess so failures localize to that step.
Returns nonzero if any step fails.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from utils import PROJECT_ROOT, now_compact, now_str
from logging_setup import setup_logger


SCRIPTS_DIR = Path(__file__).resolve().parent


def step(log, name: str, cmd: list[str]) -> int:
    log.info(f"STEP {name}: {' '.join(cmd)}")
    t0 = time.monotonic()
    rc = subprocess.call(cmd)
    dur = time.monotonic() - t0
    if rc != 0:
        log.error(f"STEP {name} FAILED rc={rc} ({dur:.1f}s)")
    else:
        log.info(f"STEP {name} OK ({dur:.1f}s)")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     default="configs/base.yaml")
    ap.add_argument("--seed",       default="regime_scout/seed_suite.yaml")
    ap.add_argument("--candidates", default="regime_scout/candidates")
    ap.add_argument("--raw",        default="regime_scout/outputs/raw_results.jsonl")
    ap.add_argument("--suspicious", default="regime_scout/outputs/suspicious_cases.jsonl")
    ap.add_argument("--regime-md",  default="regime_scout/outputs/regime_map.md")
    ap.add_argument("--regime-json",default="regime_scout/outputs/regime_map.json")
    ap.add_argument("--selected",   default="regime_scout/outputs/selected_cases.jsonl")
    ap.add_argument("--max-cases",  type=int, default=5)
    ap.add_argument("--mode",       default="quick", choices=["quick", "medium", "full"])
    ap.add_argument("--wall-budget-s", type=int, default=5400)
    ap.add_argument("--server-start-timeout", type=int, default=300)
    ap.add_argument("--benchmark-timeout", type=int, default=600)
    ap.add_argument("--skip-generate", action="store_true",
                    help="Reuse existing candidates/*.yaml")
    ap.add_argument("--skip-suite", action="store_true",
                    help="Reuse existing raw_results.jsonl")
    args = ap.parse_args()

    ts = now_compact()
    log_path = PROJECT_ROOT / "logs" / f"stage1_{ts}.log"
    log = setup_logger("stage1", log_path)
    log.info(f"=== Stage 1 orchestrator started at {now_str()} ===")
    log.info(f"args={vars(args)}")

    if not args.skip_generate:
        rc = step(log, "1/5 generate_seed_suite", [
            sys.executable, str(SCRIPTS_DIR / "generate_seed_suite.py"),
            "--seed", args.seed,
            "--out-dir", args.candidates,
            "--prune",
        ])
        if rc:
            return rc

    if not args.skip_suite:
        rc = step(log, "2/5 run_regime_suite", [
            sys.executable, str(SCRIPTS_DIR / "run_regime_suite.py"),
            "--config", args.config,
            "--workload-dir", args.candidates,
            "--out", args.raw,
            "--mode", args.mode,
            "--server-start-timeout", str(args.server_start_timeout),
            "--benchmark-timeout", str(args.benchmark_timeout),
            "--wall-budget-s", str(args.wall_budget_s),
            "--reset",
        ])
        if rc:
            log.warning("suite returned nonzero; continuing to analyze whatever ran")

    rc = step(log, "3/5 score_suspicion", [
        sys.executable, str(SCRIPTS_DIR / "score_suspicion.py"),
        "--raw", args.raw,
        "--out", args.suspicious,
    ])
    if rc:
        return rc

    rc = step(log, "4/5 cluster_regimes", [
        sys.executable, str(SCRIPTS_DIR / "cluster_regimes.py"),
        "--raw", args.raw,
        "--suspicious", args.suspicious,
        "--server-config", args.config,
        "--out-md", args.regime_md,
        "--out-json", args.regime_json,
    ])
    if rc:
        return rc

    rc = step(log, "5/5 select_cases", [
        sys.executable, str(SCRIPTS_DIR / "select_cases_for_stage2.py"),
        "--raw", args.raw,
        "--suspicious", args.suspicious,
        "--regime-map", args.regime_json,
        "--server-config", args.config,
        "--out", args.selected,
        "--max-cases", str(args.max_cases),
    ])
    log.info(f"=== Stage 1 orchestrator finished at {now_str()} ===")
    log.info(f"Summary outputs:")
    log.info(f"  raw_results:       {args.raw}")
    log.info(f"  suspicious_cases:  {args.suspicious}")
    log.info(f"  regime_map.md:     {args.regime_md}")
    log.info(f"  regime_map.json:   {args.regime_json}")
    log.info(f"  selected_cases:    {args.selected}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
