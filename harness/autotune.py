"""harness/autotune.py — Optuna integration for framework-level autotuning.

Treats sglang command-line flags as hyperparameters. Each Optuna trial:
  1. Samples a flag combination from the search space
  2. Synthesizes a bench-spec YAML (via template + overrides)
  3. Runs the spec through harness/run_bench.py
  4. Returns the target regime's req_per_s as the objective

Usage:
    python -m harness.autotune \
        --template-spec bench-specs/sglang-triton-bf16-baseline.yaml \
        --target-regime R_medium_balanced \
        --gpu-id 4 \
        --port 31200 \
        --n-trials 30 \
        --out-dir results/2026-06-25_autotuning/R_medium_balanced/

Per-trial outputs live under <out_dir>/trial_NNNN/. Optuna study persisted
to <out_dir>/study.db (SQLite). Resumes automatically if interrupted.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import yaml

# Make harness importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.spec import BenchSpec  # noqa: E402

logger = logging.getLogger("autotune")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Search space — kept in code (not config) for clarity. See
# docs/2026-06-25/sglang_autotuning_search_space.md for rationale.
# ---------------------------------------------------------------------------

def suggest_flags(trial: optuna.Trial) -> dict[str, Any]:
    """Return one flag combination sampled from the v1 search space."""
    return {
        "moe-runner-backend":   trial.suggest_categorical(
            "moe_runner_backend",
            ["triton", "flashinfer_cutlass"],
        ),
        "disable-cuda-graph":   trial.suggest_categorical(
            "disable_cuda_graph",
            [True, False],
        ),
        "max-running-requests": trial.suggest_categorical(
            "max_running_requests",
            [8, 16, 32, 64],
        ),
        "chunked-prefill-size": trial.suggest_categorical(
            "chunked_prefill_size",
            [-1, 2048, 8192],
        ),
        "schedule-policy":      trial.suggest_categorical(
            "schedule_policy",
            ["lpm", "fcfs"],
        ),
    }


# ---------------------------------------------------------------------------
# Spec synthesis
# ---------------------------------------------------------------------------

def build_trial_spec(
    template_spec: BenchSpec,
    flag_overrides: dict[str, Any],
    *,
    submission_id: str,
    gpu_id: int,
    port: int,
    health_url: str,
    base_url: str,
) -> dict[str, Any]:
    """Generate a bench-spec dict for one trial by merging template + overrides."""
    server_overrides = dict(template_spec.server.overrides)
    server_overrides.update(flag_overrides)
    server_overrides["_gpu_id"] = gpu_id
    server_overrides["port"] = port

    return {
        "submission_id": submission_id,
        "description": f"Optuna trial: {flag_overrides}",
        "tags": ["autotune", "trial"],
        "server": {
            "config": template_spec.server.config,
            "overrides": server_overrides,
            "conda_env": template_spec.server.conda_env,
            "health_url": health_url,
            "base_url": base_url,
            "startup_timeout_s": template_spec.server.startup_timeout_s,
        },
        "regimes": {
            "file": template_spec.regimes.file,
        },
        "bench": {
            "num_runs": template_spec.bench.num_runs,
            "reliable_stddev_pct": template_spec.bench.reliable_stddev_pct,
            "per_request_timeout_s": template_spec.bench.per_request_timeout_s,
            "backend": template_spec.bench.backend,
        },
        "quality_gate": {"type": template_spec.quality_gate.type},
    }


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

class TrialFailure(Exception):
    """Hard failure during trial; objective returns penalty."""


def run_trial(
    trial: optuna.Trial,
    *,
    template_spec: BenchSpec,
    target_regime: str,
    out_dir: Path,
    gpu_id: int,
    port: int,
    python_exe: str,
) -> float:
    """Run one Optuna trial. Returns req_per_s on target_regime."""
    trial_dir = out_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    flag_overrides = suggest_flags(trial)
    logger.info(f"trial {trial.number}: flags={flag_overrides}")

    # Persist sampled flags BEFORE running, so post-mortem of crashed trials works.
    (trial_dir / "flags.json").write_text(json.dumps(flag_overrides, indent=2))

    submission_id = f"autotune-{target_regime}-trial-{trial.number:04d}"
    spec_dict = build_trial_spec(
        template_spec,
        flag_overrides,
        submission_id=submission_id,
        gpu_id=gpu_id,
        port=port,
        health_url=f"http://127.0.0.1:{port}/health",
        base_url=f"http://127.0.0.1:{port}",
    )

    spec_path = trial_dir / "trial_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))

    # Invoke harness/run_bench.py via subprocess so any crash is sandboxed.
    cmd = [
        python_exe,
        str(_REPO_ROOT / "harness" / "run_bench.py"),
        "--spec", str(spec_path),
        "--out-dir", str(trial_dir),
    ]
    logger.info(f"trial {trial.number}: launching harness")
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - start
    logger.info(f"trial {trial.number}: harness returned in {wall:.1f}s, exit={proc.returncode}")

    # Persist harness stdout for debugging.
    (trial_dir / "harness_stdout.log").write_text(proc.stdout)
    if proc.stderr:
        (trial_dir / "harness_stderr.log").write_text(proc.stderr)

    summary_path = trial_dir / "summary.json"
    if not summary_path.exists():
        trial.set_user_attr("error", "summary.json missing")
        raise TrialFailure(f"trial {trial.number}: no summary.json produced")

    summary = json.loads(summary_path.read_text())

    # Persist full summary as user_attr for cross-regime post-hoc analysis
    # (so we can later read every regime's req/s for any trial, not just the target).
    if summary.get("ok") and summary.get("regimes"):
        per_regime_rps = {
            r_id: r.get("req_per_s", {}).get("mean", 0.0)
            for r_id, r in summary["regimes"].items()
        }
        trial.set_user_attr("per_regime_req_per_s", per_regime_rps)
        trial.set_user_attr("spec_hash", summary["spec_hash"])
        trial.set_user_attr("server_startup_s", summary.get("server", {}).get("startup_wall_s", 0))

    if not summary.get("ok"):
        err = summary.get("error", {})
        trial.set_user_attr("error", f"{err.get('phase')}: {err.get('message','')[:200]}")
        logger.warning(f"trial {trial.number}: ok=false, phase={err.get('phase')}")
        # Penalty value (not -inf — keep TPE's prior usable). Lowest reasonable req/s.
        return 0.0

    regimes = summary.get("regimes", {})
    target = regimes.get(target_regime)
    if not target:
        trial.set_user_attr("error", f"target_regime {target_regime} missing")
        raise TrialFailure(f"trial {trial.number}: target regime {target_regime} not in summary")

    if not target.get("reliable", True):
        trial.set_user_attr(
            "warning",
            f"unreliable stddev_pct={target.get('req_per_s', {}).get('stddev_pct')}",
        )

    rps = target.get("req_per_s", {}).get("mean", 0.0)
    if rps is None:
        rps = 0.0
    logger.info(f"trial {trial.number}: {target_regime} req/s = {rps:.3f}")
    return rps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Optuna-driven framework autotuning for sglang")
    ap.add_argument("--template-spec", required=True,
                    help="Path to base bench-spec YAML (used for non-search fields)")
    ap.add_argument("--target-regime", required=True,
                    help="Which regime's req/s to maximize")
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--port", type=int, required=True,
                    help="Base port for sglang server (each trial reuses this port)")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--out-dir", required=True,
                    help="Output dir; will contain study.db + trial_NNNN/ subdirs")
    ap.add_argument("--python-exe", default=sys.executable,
                    help="Python interpreter for harness subprocess")
    ap.add_argument("--seed", type=int, default=2026,
                    help="TPE sampler seed for reproducibility")
    ap.add_argument("--study-name", default=None,
                    help="Optuna study name (default: derived from regime)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_spec = BenchSpec.load(args.template_spec)
    logger.info(f"template spec hash: {template_spec.spec_hash}")

    study_name = args.study_name or f"autotune-{args.target-regime if False else args.target_regime}"
    storage = f"sqlite:///{out_dir / 'study.db'}"
    logger.info(f"study storage: {storage}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    # Wrap run_trial to expose extra args.
    def objective(trial: optuna.Trial) -> float:
        try:
            return run_trial(
                trial,
                template_spec=template_spec,
                target_regime=args.target_regime,
                out_dir=out_dir,
                gpu_id=args.gpu_id,
                port=args.port,
                python_exe=args.python_exe,
            )
        except TrialFailure as e:
            logger.warning(str(e))
            return 0.0

    n_existing = len(study.trials)
    n_to_run = max(0, args.n_trials - n_existing)
    logger.info(f"existing trials: {n_existing}; will run {n_to_run} more")

    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)

    # Persist headline result.
    best_trial = study.best_trial
    logger.info(f"BEST trial #{best_trial.number}: value={best_trial.value:.3f}, params={best_trial.params}")

    best_path = out_dir / "best.json"
    best_path.write_text(json.dumps({
        "study_name": study_name,
        "target_regime": args.target_regime,
        "best_trial_number": best_trial.number,
        "best_value_req_per_s": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": dict(best_trial.user_attrs),
        "total_trials": len(study.trials),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    logger.info(f"wrote {best_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
