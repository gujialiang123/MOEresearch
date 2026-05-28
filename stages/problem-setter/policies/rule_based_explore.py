#!/usr/bin/env python3
"""Stage 1 explore loop — the v0.3 successor to run_stage1.py.

Combines all skills into an adaptive explore:

  WAVE 0 — seed suite (10 hand-picked workloads)
  → mine + classify + score (v2)
  → write regime map after wave 0
  → triage: for each cluster, decide whether to EXPAND
       if (concurrency_capped OR cuda_graph_too_small) → expand max_concurrency around the suspect
       if (at_capacity)                                → expand input_len + max_concurrency upward
       if (lonely member of cluster, score ≥ 0.4)      → expand the hint's natural axis
       else                                             → no expansion
  → WAVE 1 — run all expanded workloads
  → re-mine + re-classify + re-score (now local_nonlinearity has neighbors)
  → emit final regime_map + selected_cases

Wave-by-wave to keep the budget under control: stop after a max wave count
or wall-time budget, whichever first.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


FILE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
SKILLS_DIR   = PROJECT_ROOT / ".github" / "skills"

sys.path.insert(0, str(SCRIPTS_DIR))
from utils import load_yaml, now_compact, now_str, read_jsonl  # noqa: E402
from logging_setup import setup_logger                          # noqa: E402


def step(log, name: str, cmd: list[str]) -> int:
    log.info(f"STEP {name}: {' '.join(str(c) for c in cmd)}")
    t0 = time.monotonic()
    rc = subprocess.call(cmd)
    log.info(f"STEP {name} {'OK' if rc == 0 else 'FAIL'} ({time.monotonic()-t0:.1f}s) rc={rc}")
    return rc


def triage(scored_rows: list[dict], log) -> list[dict]:
    """Decide which (workload, axis, strategy) tuples to expand in next wave.

    Returns a list of dicts: {workload, axis, strategy, reason}.
    """
    plans: list[dict] = []
    by_hint: dict[str, list[dict]] = {}
    for s in scored_rows:
        by_hint.setdefault(s.get("regime_hint") or "unknown", []).append(s)

    for s in scored_rows:
        cls = (s.get("classification") or "")
        sls_evi = (s.get("components", {}).get("server_log_signal", {}).get("evidence", {}))
        contributions = sls_evi.get("contributions", {})

        # Rule 1: concurrency-capped → bracket max_concurrency
        if "concurrency_capped" in contributions:
            plans.append({
                "workload": s["workload_file"],
                "axis": "max_concurrency",
                "strategy": "bracket",
                "reason": "concurrency_capped per server-log-mining",
                "source_workload_name": s["workload_name"],
            })
            continue

        # Rule 2: cuda_graph_too_small → bracket max_concurrency too
        if "cuda_graph_too_small" in contributions:
            plans.append({
                "workload": s["workload_file"],
                "axis": "max_concurrency",
                "strategy": "bracket",
                "reason": "cuda_graph_too_small",
                "source_workload_name": s["workload_name"],
            })
            continue

        # Rule 3: at/near capacity → expand input_len upward
        if "at_capacity" in contributions or "near_capacity" in contributions:
            plans.append({
                "workload": s["workload_file"],
                "axis": "input_len",
                "strategy": "upward",
                "reason": "approaching KV capacity",
                "source_workload_name": s["workload_name"],
            })
            continue

        # Rule 4: lonely + reasonable score → expand the hint's natural axis
        if len(by_hint.get(s.get("regime_hint") or "", [])) == 1:
            hint = (s.get("regime_hint") or "").lower()
            if "prefill" in hint:
                axis = "input_len"
            elif "decode" in hint:
                axis = "output_len"
            elif "scheduler" in hint:
                axis = "max_concurrency"
            elif "prefix" in hint or "cache" in hint:
                axis = "max_concurrency"
            else:
                axis = "max_concurrency"
            if (s.get("score") or 0) >= 0.10:
                plans.append({
                    "workload": s["workload_file"],
                    "axis": axis,
                    "strategy": "bracket",
                    "reason": f"lonely cluster ({s.get('regime_hint')}); probe natural axis",
                    "source_workload_name": s["workload_name"],
                })

    # Dedupe: at most one plan per (workload, axis)
    seen = set()
    out = []
    for p in plans:
        key = (p["workload"], p["axis"])
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def expand_one(plan: dict, search_space: Path, out_dir: Path, log) -> list[str]:
    summary_path = out_dir / f"_summary_{Path(plan['workload']).stem}_{plan['axis']}.json"
    rc = step(log, f"expand({Path(plan['workload']).stem}, {plan['axis']})", [
        sys.executable, str(SKILLS_DIR / "boundary-expansion" / "impl" / "expand.py"),
        "--parent", plan["workload"],
        "--axis", plan["axis"],
        "--strategy", plan["strategy"],
        "--search-space", str(search_space),
        "--neighbors-out", str(out_dir),
        "--summary-json", str(summary_path),
    ])
    if rc != 0:
        return []
    try:
        summary = json.loads(summary_path.read_text())
        return [g["path"] for g in summary.get("generated", [])]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def run_wave(config: Path, workload_dir: Path, raw_out: Path,
             mode: str, server_to: int, bench_to: int, budget_s: int,
             log_path: Path) -> int:
    return subprocess.call([
        sys.executable, str(SCRIPTS_DIR / "run_regime_suite.py"),
        "--config", str(config),
        "--workload-dir", str(workload_dir),
        "--out", str(raw_out),
        "--mode", mode,
        "--server-start-timeout", str(server_to),
        "--benchmark-timeout", str(bench_to),
        "--wall-budget-s", str(budget_s),
        "--log", str(log_path),
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     default="configs/base.yaml")
    ap.add_argument("--seed",       default="regime_scout/seed_suite.yaml")
    ap.add_argument("--candidates", default="regime_scout/candidates")
    ap.add_argument("--expanded-dir", default="regime_scout/candidates/expanded")
    ap.add_argument("--raw",        default="regime_scout/outputs/raw_results.jsonl")
    ap.add_argument("--suspicious", default="regime_scout/outputs/suspicious_cases.jsonl")
    ap.add_argument("--regime-md",  default="regime_scout/outputs/regime_map.md")
    ap.add_argument("--regime-json",default="regime_scout/outputs/regime_map.json")
    ap.add_argument("--selected",   default="regime_scout/outputs/selected_cases.jsonl")
    ap.add_argument("--search-space",default="regime_scout/search_space.yaml")
    ap.add_argument("--noise-baseline", default="experiments/noise_baseline.json")
    ap.add_argument("--max-cases",  type=int, default=5)
    ap.add_argument("--wave-budget-s", type=int, default=2400)
    ap.add_argument("--max-waves",  type=int, default=2,
                    help="Total waves including seed wave (wave 0). 2 = seed + one expansion.")
    ap.add_argument("--max-neighbors-per-plan", type=int, default=3)
    ap.add_argument("--reuse-seed-run", action="store_true",
                    help="Skip generating + running seed wave; reuse existing raw_results.jsonl")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="selected_case_score threshold for final selection")
    args = ap.parse_args()

    ts = now_compact()
    log_path = PROJECT_ROOT / "logs" / f"explore_{ts}.log"
    log = setup_logger("explore", log_path)
    log.info(f"=== Stage 1 EXPLORE started at {now_str()} ===")
    log.info(f"args={vars(args)}")

    # ---------------- WAVE 0: seeds ----------------
    if not args.reuse_seed_run:
        rc = step(log, "generate_seed_suite", [
            sys.executable, str(SCRIPTS_DIR / "generate_seed_suite.py"),
            "--seed", args.seed, "--out-dir", args.candidates, "--prune",
        ])
        if rc:
            return rc
        rc = run_wave(Path(args.config), Path(args.candidates), Path(args.raw),
                      "quick", 240, 600, args.wave_budget_s,
                      PROJECT_ROOT / "logs" / f"explore_wave0_{ts}.log")
        if rc and rc != 0:
            log.warning(f"wave 0 returned rc={rc}; continuing with whatever ran")
    else:
        log.info("--reuse-seed-run: skipping wave 0 generation and suite run")

    # ---------------- score wave 0 ----------------
    rc = step(log, "score v2 (wave 0)", [
        sys.executable, str(SKILLS_DIR / "suspicion-scoring" / "impl" / "score.py"),
        "--raw", args.raw,
        "--noise-baseline", args.noise_baseline,
        "--out", args.suspicious,
        "--force-mine",
    ])
    if rc:
        return rc

    if args.max_waves <= 1:
        log.info("max_waves=1: skipping expansion")
    else:
        # ---------------- TRIAGE ----------------
        scored = read_jsonl(args.suspicious)
        plans = triage(scored, log)
        log.info(f"triage produced {len(plans)} expansion plan(s)")
        for p in plans:
            log.info(f"  PLAN: expand {p['source_workload_name']} along {p['axis']} "
                     f"({p['strategy']}) — {p['reason']}")

        # ---------------- WAVE 1: expansion ----------------
        Path(args.expanded_dir).mkdir(parents=True, exist_ok=True)
        all_generated: list[str] = []
        for plan in plans:
            gens = expand_one(plan, Path(args.search_space), Path(args.expanded_dir), log)
            log.info(f"  expanded {Path(plan['workload']).stem} → {len(gens)} neighbor(s)")
            all_generated.extend(gens[:args.max_neighbors_per_plan])

        if not all_generated:
            log.info("no neighbors generated; skipping wave 1")
        else:
            log.info(f"WAVE 1: {len(all_generated)} expanded workloads")
            rc = run_wave(Path(args.config), Path(args.expanded_dir),
                          Path(args.raw),  # appends
                          "quick", 240, 600, args.wave_budget_s,
                          PROJECT_ROOT / "logs" / f"explore_wave1_{ts}.log")
            if rc:
                log.warning(f"wave 1 returned rc={rc}; continuing with whatever ran")

            # ---------------- re-score with neighbors present ----------------
            rc = step(log, "score v2 (wave 1)", [
                sys.executable, str(SKILLS_DIR / "suspicion-scoring" / "impl" / "score.py"),
                "--raw", args.raw,
                "--noise-baseline", args.noise_baseline,
                "--out", args.suspicious,
                "--force-mine",
            ])

    # ---------------- final reporting ----------------
    rc = step(log, "cluster_regimes", [
        sys.executable, str(SCRIPTS_DIR / "cluster_regimes.py"),
        "--raw", args.raw,
        "--suspicious", args.suspicious,
        "--server-config", args.config,
        "--out-md", args.regime_md,
        "--out-json", args.regime_json,
    ])
    if rc:
        return rc

    rc = step(log, "select_cases", [
        sys.executable, str(SCRIPTS_DIR / "select_cases_for_stage2.py"),
        "--raw", args.raw,
        "--suspicious", args.suspicious,
        "--regime-map", args.regime_json,
        "--server-config", args.config,
        "--out", args.selected,
        "--max-cases", str(args.max_cases),
        "--threshold", str(args.threshold),
    ])
    log.info(f"=== Stage 1 EXPLORE finished at {now_str()} ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())
