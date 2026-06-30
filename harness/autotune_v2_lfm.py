"""harness/autotune_v2_lfm.py — Conditional search space autotune (Chendi v2 design).

Differences from harness/autotune.py (v1, used 6/25):
  1. Search space is CONDITIONAL — based on Chendi's 6/30 framework. We first
     decide which subspaces are active for (model, hw, workload), then sweep
     only valid candidates inside active subspaces.
  2. Expanded search dimensions beyond the original 5 flags. v1 had:
     moe-runner-backend, disable-cuda-graph, max-running-requests,
     chunked-prefill-size, schedule-policy.
     v2 adds: attention-backend, mem-fraction-static, kv-cache-dtype,
     cuda-graph-max-bs, schedule-conservativeness.
  3. Per-trial detail is preserved (already from v1 via user_attrs; this
     module also writes a per-trial summary CSV for easy reading).

Conditional logic for LFM2.5-8B-A1B + 1× H200 + bf16:
  - Parallelism subspace: INACTIVE (tp=1 fixed, single GPU)
  - Speculative subspace: INACTIVE (no draft model setup)
  - PD disagg subspace: INACTIVE (single GPU)
  - HiCache/MM subspace: INACTIVE (not relevant)
  - Quantization subspace: PARTIALLY ACTIVE (weight=bf16 fixed, KV cache can vary)
  - MoE subspace: ACTIVE
  - Memory/KV subspace: ACTIVE
  - Batching/scheduling subspace: ACTIVE
  - Attention backend subspace: ACTIVE
  - CUDA graph subspace: ACTIVE

Usage:
    python -m harness.autotune_v2_lfm \
        --template-spec bench-specs/lfm2.5-8b-a1b-true-default.yaml \
        --target-regime R_concurrent_decode \
        --gpu-id 4 \
        --port 31500 \
        --n-trials 25 \
        --out-dir results/2026-06-30_lfm2.5/optuna/R_concurrent_decode/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
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

logger = logging.getLogger("autotune_v2")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Conditional search space (v2, per Chendi 6/30 guidance)
# ---------------------------------------------------------------------------

# Each entry documents: (subspace, flag, candidates, why this is in/out of v2 scope)
SEARCH_SPACE_DOC = """
ACTIVE SUBSPACES (sweeping):

  [Memory / KV cache]
    --mem-fraction-static : 0.75 / 0.85 / 0.90
        Larger = more KV cache, more parallel requests; too large risks OOM.

  [Batching / scheduling]
    --max-running-requests : 8 / 16 / 32 / 64
        Scheduler concurrency cap. Higher → more in-flight requests for
        concurrent-decode regimes; lower → less KV pressure.
    --chunked-prefill-size : -1 / 2048 / 8192
        -1 = no chunking. Affects long-prefill regimes only.
    --schedule-policy : lpm / fcfs
        lpm = longest-prefix-match (cache-friendly), fcfs = first-come.

  [Attention backend]
    --attention-backend : fa3 / flashinfer / triton
        fa3 = FlashAttention v3 (Hopper-optimized default).
        flashinfer = flashinfer attention kernel.
        triton = pure-Triton (slowest but most portable).

  [CUDA graph]
    --disable-cuda-graph : true / false
        false (default) = capture. true = eager mode (slower for decode).

  [MoE]
    --moe-runner-backend : triton / flashinfer_cutlass
        Which MoE GEMM kernel to use. Note: LFM2.5 only has 32 experts top-4
        (vs Qwen3 128/8), so the GEMM shapes are quite different.

INACTIVE SUBSPACES (held fixed):
  - tp_size / dp_size / ep_size / pp_size : all = 1 (single GPU)
  - speculative_algorithm : None (no spec decode in this study)
  - disaggregation_mode : null (single-server mode)
  - quantization : None (bf16; no fp8/awq versions of LFM2.5-8B-A1B available)
  - kv_cache_dtype : auto (we don't risk fp8 KV; needs separate validation)
  - lora_* / hicache_* / multimodal_* : not applicable

OUT-OF-SCOPE FOR v2 (would be v3 expansion):
  - cuda_graph_max_bs / cuda_graph_bs (explicit lists; integer-valued)
  - schedule_conservativeness (float; effect typically subtle)
  - max_prefill_tokens (interacts with chunked_prefill_size)
  - radix_eviction_policy (lru/lfu/etc; only matters under heavy contention)
"""


def suggest_flags_v2(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one configuration from the v2 conditional search space."""
    flags: dict[str, Any] = {}

    # Memory / KV
    flags["mem-fraction-static"] = trial.suggest_categorical(
        "mem_fraction_static", [0.75, 0.85, 0.90]
    )

    # Batching / scheduling
    flags["max-running-requests"] = trial.suggest_categorical(
        "max_running_requests", [8, 16, 32, 64]
    )
    flags["chunked-prefill-size"] = trial.suggest_categorical(
        "chunked_prefill_size", [-1, 2048, 8192]
    )
    flags["schedule-policy"] = trial.suggest_categorical(
        "schedule_policy", ["lpm", "fcfs"]
    )

    # Attention backend.
    # NOTE: On LFM2.5-8B-A1B (hybrid conv + full_attention layers) on this env:
    #   - "triton" raises ValueError (layer_id=0 not in full attention layers)
    #   - "flashinfer" fails to JIT-build (ld can't find -lcuda in conda env)
    #   - "fa3" works.
    # We document the search-space *candidates* in search_space_v2.md but force
    # the live search to "fa3" only to avoid burning trials on configs that
    # cannot start. Re-enable when env / model compatibility is fixed.
    flags["attention-backend"] = trial.suggest_categorical(
        "attention_backend", ["fa3"]
    )

    # CUDA graph
    flags["disable-cuda-graph"] = trial.suggest_categorical(
        "disable_cuda_graph", [True, False]
    )

    # MoE
    flags["moe-runner-backend"] = trial.suggest_categorical(
        "moe_runner_backend", ["triton", "flashinfer_cutlass"]
    )

    return flags


# Total combinations: 3 × 4 × 3 × 2 × 1 (fa3 forced) × 2 × 2 = 288
# Optuna TPE with ~25 trials should find a strong config.
TOTAL_COMBOS_V2 = 3 * 4 * 3 * 2 * 1 * 2 * 2


# ---------------------------------------------------------------------------
# Spec synthesis (same as v1)
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
        "description": f"Optuna v2 trial: {flag_overrides}",
        "tags": ["autotune-v2", "trial", "lfm2.5"],
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
# Per-trial detail logging
# ---------------------------------------------------------------------------

PER_TRIAL_CSV_COLUMNS = [
    "trial",
    "wall_s",
    "ok",
    "target_req_per_s",
    "reliable",
    "warnings",
    # Flag values
    "moe_runner_backend",
    "attention_backend",
    "disable_cuda_graph",
    "max_running_requests",
    "chunked_prefill_size",
    "schedule_policy",
    "mem_fraction_static",
    # All-regime breakdowns
    "R_short_decode_rps",
    "R_medium_balanced_rps",
    "R_long_prefill_rps",
    "R_concurrent_decode_rps",
    "spec_hash",
    "server_startup_s",
]


def append_csv_row(csv_path: Path, row: dict) -> None:
    """Append one row to per_trial_log.csv. Writes header if file is new."""
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_TRIAL_CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PER_TRIAL_CSV_COLUMNS})


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

class TrialFailure(Exception):
    """Hard failure during trial; objective returns penalty."""


def _wait_port_free(port: int, *, host: str = "127.0.0.1", timeout_s: float = 60.0) -> None:
    """Block until TCP port is unbound on host (or until timeout). Used between trials."""
    import socket
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
    logger.warning(f"port {host}:{port} still in use after {timeout_s}s; proceeding anyway")


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
) -> float:
    """Run one Optuna trial. Returns req_per_s on target_regime."""
    trial_dir = out_dir / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    flag_overrides = suggest_flags_v2(trial)
    logger.info(f"trial {trial.number}: flags={flag_overrides}")

    (trial_dir / "flags.json").write_text(json.dumps(flag_overrides, indent=2))

    submission_id = f"autotune-v2-{target_regime}-trial-{trial.number:04d}"
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

    cmd = [
        python_exe,
        str(_REPO_ROOT / "harness" / "run_bench.py"),
        "--spec", str(spec_path),
        "--out-dir", str(trial_dir),
    ]
    _wait_port_free(port)
    logger.info(f"trial {trial.number}: launching harness")
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - start
    logger.info(f"trial {trial.number}: harness returned in {wall:.1f}s, exit={proc.returncode}")

    (trial_dir / "harness_stdout.log").write_text(proc.stdout)
    if proc.stderr:
        (trial_dir / "harness_stderr.log").write_text(proc.stderr)

    summary_path = trial_dir / "summary.json"
    csv_row: dict[str, Any] = {
        "trial": trial.number,
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
    csv_row["server_startup_s"] = round(summary.get("server", {}).get("startup_wall_s", 0), 1)
    csv_row["warnings"] = ";".join(summary.get("warnings", []))

    if summary.get("ok") and summary.get("regimes"):
        per_regime_rps = {
            r_id: r.get("req_per_s", {}).get("mean", 0.0)
            for r_id, r in summary["regimes"].items()
        }
        trial.set_user_attr("per_regime_req_per_s", per_regime_rps)
        trial.set_user_attr("spec_hash", summary["spec_hash"])
        trial.set_user_attr("server_startup_s",
                             summary.get("server", {}).get("startup_wall_s", 0))
        for r_id, rps in per_regime_rps.items():
            csv_row[f"{r_id}_rps"] = round(rps, 4)

    if not summary.get("ok"):
        err = summary.get("error", {})
        trial.set_user_attr("error", f"{err.get('phase')}: {err.get('message','')[:200]}")
        logger.warning(f"trial {trial.number}: ok=false, phase={err.get('phase')}")
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
        raise TrialFailure(f"trial {trial.number}: target regime {target_regime} not in summary")

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
    ap = argparse.ArgumentParser(description="Optuna v2 conditional autotuning")
    ap.add_argument("--template-spec", required=True)
    ap.add_argument("--target-regime", required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--study-name", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write search space doc once
    (out_dir / "search_space_v2.md").write_text(SEARCH_SPACE_DOC.strip())
    logger.info(f"search space documented at {out_dir / 'search_space_v2.md'}")

    template_spec = BenchSpec.load(args.template_spec)
    logger.info(f"template spec hash: {template_spec.spec_hash}")

    study_name = args.study_name or f"autotune-v2-{args.target_regime}"
    storage = f"sqlite:///{out_dir / 'study.db'}"
    logger.info(f"study storage: {storage}")
    logger.info(f"v2 search space size = {TOTAL_COMBOS_V2} combinations (vs v1=96)")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

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
            )
        except TrialFailure as e:
            logger.warning(str(e))
            return 0.0

    n_existing = len(study.trials)
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
        "search_space_combinations": TOTAL_COMBOS_V2,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    logger.info(f"wrote {best_path}")
    logger.info(f"per-trial CSV: {per_trial_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
