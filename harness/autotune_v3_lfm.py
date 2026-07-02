"""harness/autotune_v3_lfm.py — Optuna v3 with warm-start + stratified fix.

Fixes the v2 (2026-06-30) TPE failure mode where TPE converged on
flashinfer_cutlass MoE and missed the actual optimum (triton MoE + good
batching = 23.5 req/s vs Optuna best 22.3).

Root cause (see docs/2026-06-30/lfm2.5_conditional_autotuning.md §6.4):
early random trials paired triton MoE with cap=8 / cg-off / bad configs;
all 5 triton trials returned <10 req/s; TPE marginalized over batching
and learned "triton = bad"; never re-sampled triton with good batching.

Fix: BEFORE letting TPE take over, enqueue N warm-start trials that
guarantee every categorical value gets ≥1 fair evaluation (paired with
known-good priors on the other dimensions). Then TPE has an unbiased
starting distribution.

Warm-start trials for LFM2.5-8B-A1B on 1× H200:
  Prior "good batching" values (from v2 experiment):
    max-running-requests=32, chunked-prefill-size=-1, schedule-policy=lpm,
    mem-fraction-static=0.9, disable-cuda-graph=False.
  Categorical dimensions to stratify:
    moe-runner-backend ∈ {triton, flashinfer_cutlass, auto}   # 3 values
    attention-backend ∈ {fa3}                                 # 1 value
  Total stratified warm trials = 3 × 1 = 3.
  Additionally enqueue the cookbook-default explicit config (moe=auto)
  as a "known baseline" reference trial → 4 warm trials total.

After the 4 warm trials, TPE takes over with the full 7-dim search space.

Design notes
------------
- We use `study.enqueue_trial()` before `study.optimize()` to force the
  warm-start configurations. TPE will still evaluate them and update its
  model based on their outcomes.
- Per-trial detail logging same as v2 (CSV + trial_NNNN/ dirs + user_attrs).
- MFU annotation is delegated to run_bench.py (--mfu-hardware, --mfu-model
  CLI flags).

Usage:
    python -m harness.autotune_v3_lfm \
        --template-spec bench-specs/lfm2.5-v3-true-default-longctx.yaml \
        --target-regime R_concurrent_decode \
        --gpu-id 6 --port 31700 \
        --n-trials 30 \
        --n-warm-trials 4 \
        --hardware configs/hardware/h200.yaml \
        --model configs/models/lfm2.5-8b-a1b.yaml \
        --out-dir results/2026-07-02_lfm2.5_v3/optuna-v3-R_concurrent_decode/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.spec import BenchSpec  # noqa: E402

logger = logging.getLogger("autotune_v3")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Conditional search space (v3 = v2 with added moe=auto)
# ---------------------------------------------------------------------------

# NOTE: attention-backend is fixed to ["fa3"] because on LFM2.5-8B-A1B in
# our env, "triton" errors on hybrid arch and "flashinfer" JIT-fails.
# See docs/2026-06-30/lfm2.5_conditional_autotuning.md §3 for evidence.

MOE_BACKEND_CANDIDATES = ["triton", "flashinfer_cutlass", "auto"]
ATTENTION_BACKEND_CANDIDATES = ["fa3"]

SEARCH_SPACE_DOC_V3 = """
V3 SEARCH SPACE (2026-07-02, fixed TPE failure via warm-start):

  ACTIVE (7 knobs, all sweeping):
    --mem-fraction-static ∈ {0.75, 0.85, 0.90}
    --max-running-requests ∈ {8, 16, 32, 64}
    --chunked-prefill-size ∈ {-1, 2048, 8192}
    --schedule-policy ∈ {lpm, fcfs}
    --attention-backend ∈ {fa3}     [others rejected by model/env]
    --disable-cuda-graph ∈ {True, False}
    --moe-runner-backend ∈ {triton, flashinfer_cutlass, auto}  [+auto vs v2]

  Total combos: 3 × 4 × 3 × 2 × 1 × 2 × 3 = 432

  WARM-START (enqueued before TPE):
    Trial 0: moe=auto,   good batching prior (cookbook-equivalent)
    Trial 1: moe=triton, good batching prior [key: v2 missed this!]
    Trial 2: moe=flashinfer_cutlass, good batching prior [v2 winner]
    Trial 3: moe=auto,   fcfs schedule policy (control)

  Good batching prior: cap=32, chunk=-1, sched=lpm, mem=0.9, cg-on

  TPE runs from trial 4 onwards with the full space.

INACTIVE / OUT-OF-SCOPE (unchanged from v2):
  - Parallelism (tp/dp/ep/pp), Speculative, PD disagg, Quantization,
    KV dtype, HiCache, LoRA, Multimodal.
"""


def suggest_flags_v3(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one configuration from the v3 conditional search space."""
    flags: dict[str, Any] = {}
    flags["mem-fraction-static"] = trial.suggest_categorical(
        "mem_fraction_static", [0.75, 0.85, 0.90]
    )
    flags["max-running-requests"] = trial.suggest_categorical(
        "max_running_requests", [8, 16, 32, 64]
    )
    flags["chunked-prefill-size"] = trial.suggest_categorical(
        "chunked_prefill_size", [-1, 2048, 8192]
    )
    flags["schedule-policy"] = trial.suggest_categorical(
        "schedule_policy", ["lpm", "fcfs"]
    )
    flags["attention-backend"] = trial.suggest_categorical(
        "attention_backend", ATTENTION_BACKEND_CANDIDATES
    )
    flags["disable-cuda-graph"] = trial.suggest_categorical(
        "disable_cuda_graph", [True, False]
    )
    flags["moe-runner-backend"] = trial.suggest_categorical(
        "moe_runner_backend", MOE_BACKEND_CANDIDATES
    )
    return flags


def make_warm_start_trials() -> list[dict[str, Any]]:
    """4 stratified warm-start trials — one per MoE backend + control."""
    good_prior = {
        "mem_fraction_static": 0.90,
        "max_running_requests": 32,
        "chunked_prefill_size": -1,
        "schedule_policy": "lpm",
        "attention_backend": "fa3",
        "disable_cuda_graph": False,
    }
    return [
        # Trial 0: reference cookbook-equivalent (moe=auto let sglang decide)
        {**good_prior, "moe_runner_backend": "auto"},
        # Trial 1: THE trial v2 missed — triton MoE + good batching
        {**good_prior, "moe_runner_backend": "triton"},
        # Trial 2: v2 winner reproduced — flashinfer_cutlass + good batching
        {**good_prior, "moe_runner_backend": "flashinfer_cutlass"},
        # Trial 3: same as trial 0 but fcfs → sanity that sched-policy barely matters
        {**good_prior, "moe_runner_backend": "auto", "schedule_policy": "fcfs"},
    ]


# ---------------------------------------------------------------------------
# Spec + port + trial exec (same as v2, with MFU flag pass-through)
# ---------------------------------------------------------------------------

def _wait_port_free(port: int, *, host: str = "127.0.0.1",
                    timeout_s: float = 60.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                rc = s.connect_ex((host, port))
            except OSError:
                rc = 1
        if rc != 0:
            return
        time.sleep(2.0)
    logger.warning(f"port {host}:{port} still in use after {timeout_s}s")


def build_trial_spec(
    template_spec: BenchSpec,
    flag_overrides: dict[str, Any],
    *,
    submission_id: str,
    gpu_id: int,
    port: int,
) -> dict[str, Any]:
    server_overrides = dict(template_spec.server.overrides)
    server_overrides.update(flag_overrides)
    server_overrides["_gpu_id"] = gpu_id
    server_overrides["port"] = port

    return {
        "submission_id": submission_id,
        "description": f"Optuna v3 trial: {flag_overrides}",
        "tags": ["autotune-v3", "trial", "lfm2.5", "longctx"],
        "server": {
            "config": template_spec.server.config,
            "overrides": server_overrides,
            "conda_env": template_spec.server.conda_env,
            "health_url": f"http://127.0.0.1:{port}/health",
            "base_url":   f"http://127.0.0.1:{port}",
            "startup_timeout_s": template_spec.server.startup_timeout_s,
        },
        "regimes": {"file": template_spec.regimes.file},
        "bench": {
            "num_runs": template_spec.bench.num_runs,
            "reliable_stddev_pct": template_spec.bench.reliable_stddev_pct,
            "per_request_timeout_s": template_spec.bench.per_request_timeout_s,
            "backend": template_spec.bench.backend,
        },
        "quality_gate": {"type": template_spec.quality_gate.type},
    }


PER_TRIAL_CSV_COLUMNS = [
    "trial", "phase", "wall_s", "ok", "target_req_per_s",
    "reliable", "warnings",
    # Flags
    "moe_runner_backend", "attention_backend", "disable_cuda_graph",
    "max_running_requests", "chunked_prefill_size", "schedule_policy",
    "mem_fraction_static",
    # Per-regime req/s
    "R_short_decode_rps", "R_medium_balanced_rps", "R_long_prefill_rps",
    "R_concurrent_decode_rps", "R_prompt_8k_c4_out128_rps",
    "R_prompt_16k_c2_out128_rps", "R_prompt_32k_c1_out128_rps",
    "R_prompt_50k_c1_out64_rps",
    # Per-regime MFU (simple)
    "R_short_decode_mfu", "R_medium_balanced_mfu", "R_long_prefill_mfu",
    "R_concurrent_decode_mfu", "R_prompt_8k_c4_out128_mfu",
    "R_prompt_16k_c2_out128_mfu", "R_prompt_32k_c1_out128_mfu",
    "R_prompt_50k_c1_out64_mfu",
    # Meta
    "spec_hash", "server_startup_s",
]


def append_csv_row(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_TRIAL_CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PER_TRIAL_CSV_COLUMNS})


class TrialFailure(Exception):
    pass


def run_trial(
    trial: optuna.Trial,
    *,
    template_spec: BenchSpec,
    target_regime: str,
    out_dir: Path,
    gpu_id: int,
    port: int,
    python_exe: str,
    per_trial_csv: Path,
    mfu_hardware: str | None,
    mfu_model: str | None,
    n_warm_trials: int,
) -> float:
    trial_dir = out_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    flag_overrides = suggest_flags_v3(trial)
    phase = "warm-start" if trial.number < n_warm_trials else "tpe"
    logger.info(f"trial {trial.number} [{phase}]: flags={flag_overrides}")

    (trial_dir / "flags.json").write_text(json.dumps(flag_overrides, indent=2))

    submission_id = f"autotune-v3-{target_regime}-trial-{trial.number:04d}"
    spec_dict = build_trial_spec(
        template_spec, flag_overrides,
        submission_id=submission_id, gpu_id=gpu_id, port=port,
    )
    spec_path = trial_dir / "trial_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))

    cmd = [
        python_exe,
        str(_REPO_ROOT / "harness" / "run_bench.py"),
        "--spec", str(spec_path),
        "--out-dir", str(trial_dir),
    ]
    if mfu_hardware and mfu_model:
        cmd += ["--mfu-hardware", mfu_hardware, "--mfu-model", mfu_model]

    _wait_port_free(port)
    logger.info(f"trial {trial.number}: launching harness")
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    wall = time.perf_counter() - start
    logger.info(f"trial {trial.number}: harness returned in {wall:.1f}s, exit={proc.returncode}")

    (trial_dir / "harness_stdout.log").write_text(proc.stdout)
    if proc.stderr:
        (trial_dir / "harness_stderr.log").write_text(proc.stderr)

    summary_path = trial_dir / "summary.json"
    csv_row: dict[str, Any] = {
        "trial": trial.number,
        "phase": phase,
        "wall_s": round(wall, 1),
        **{k.replace("-", "_"): v for k, v in flag_overrides.items()},
    }

    if not summary_path.exists():
        csv_row.update({"ok": False, "warnings": "summary missing"})
        append_csv_row(per_trial_csv, csv_row)
        trial.set_user_attr("error", "summary.json missing")
        raise TrialFailure(f"trial {trial.number}: no summary.json produced")

    summary = json.loads(summary_path.read_text())
    csv_row["ok"] = bool(summary.get("ok", False))
    csv_row["spec_hash"] = summary.get("spec_hash", "")
    csv_row["server_startup_s"] = round(
        summary.get("server", {}).get("startup_wall_s", 0), 1
    )
    csv_row["warnings"] = ";".join(summary.get("warnings", []))

    if summary.get("ok") and summary.get("regimes"):
        per_regime_rps = {}
        per_regime_mfu = {}
        for r_id, r in summary["regimes"].items():
            per_regime_rps[r_id] = r.get("req_per_s", {}).get("mean", 0.0)
            per_regime_mfu[r_id] = r.get("mfu", {}).get("mfu_pct_simple", 0.0)
        trial.set_user_attr("per_regime_req_per_s", per_regime_rps)
        trial.set_user_attr("per_regime_mfu_simple", per_regime_mfu)
        trial.set_user_attr("spec_hash", summary["spec_hash"])
        trial.set_user_attr(
            "server_startup_s",
            summary.get("server", {}).get("startup_wall_s", 0),
        )
        for r_id, rps in per_regime_rps.items():
            csv_row[f"{r_id}_rps"] = round(rps, 4)
        for r_id, mfu in per_regime_mfu.items():
            csv_row[f"{r_id}_mfu"] = round(mfu, 3)

    if not summary.get("ok"):
        err = summary.get("error", {})
        trial.set_user_attr(
            "error", f"{err.get('phase')}: {err.get('message','')[:200]}"
        )
        logger.warning(
            f"trial {trial.number}: ok=false, phase={err.get('phase')}"
        )
        csv_row["target_req_per_s"] = 0.0
        csv_row["reliable"] = False
        append_csv_row(per_trial_csv, csv_row)
        return 0.0

    regimes = summary.get("regimes", {})
    target = regimes.get(target_regime)
    if not target:
        trial.set_user_attr("error", f"target_regime {target_regime} missing")
        csv_row["target_req_per_s"] = 0.0
        csv_row["reliable"] = False
        append_csv_row(per_trial_csv, csv_row)
        raise TrialFailure(
            f"trial {trial.number}: target regime {target_regime} not in summary"
        )

    rps = target.get("req_per_s", {}).get("mean", 0.0) or 0.0
    csv_row["target_req_per_s"] = round(rps, 4)
    csv_row["reliable"] = target.get("reliable", True)
    append_csv_row(per_trial_csv, csv_row)
    logger.info(f"trial {trial.number}: {target_regime} req/s = {rps:.3f}")
    return rps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Optuna v3 with warm-start")
    ap.add_argument("--template-spec", required=True)
    ap.add_argument("--target-regime", required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--n-warm-trials", type=int, default=4,
                    help="First N trials are stratified warm-start (default 4).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--hardware", default=None,
                    help="MFU hardware yaml (passed to run_bench.py)")
    ap.add_argument("--model", default=None,
                    help="MFU model yaml (passed to run_bench.py)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "search_space_v3.md").write_text(SEARCH_SPACE_DOC_V3.strip())
    logger.info(f"search space documented at {out_dir / 'search_space_v3.md'}")

    template_spec = BenchSpec.load(args.template_spec)
    logger.info(f"template spec hash: {template_spec.spec_hash}")

    study_name = args.study_name or f"autotune-v3-{args.target_regime}"
    storage = f"sqlite:///{out_dir / 'study.db'}"
    logger.info(f"study storage: {storage}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing == 0:
        warm_trials = make_warm_start_trials()[: args.n_warm_trials]
        for i, params in enumerate(warm_trials):
            study.enqueue_trial(params)
            logger.info(f"enqueued warm-start trial {i}: {params}")
    else:
        logger.info(f"resuming from {n_existing} existing trials; skipping warm-start")

    per_trial_csv = out_dir / "per_trial_log.csv"

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
                per_trial_csv=per_trial_csv,
                mfu_hardware=args.hardware,
                mfu_model=args.model,
                n_warm_trials=args.n_warm_trials,
            )
        except TrialFailure as e:
            logger.warning(str(e))
            return 0.0

    n_to_run = max(0, args.n_trials - n_existing)
    logger.info(f"existing trials: {n_existing}; will run {n_to_run} more")
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run)

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
        "n_warm_trials": args.n_warm_trials,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    logger.info(f"wrote {best_path}")
    logger.info(f"per-trial CSV: {per_trial_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
