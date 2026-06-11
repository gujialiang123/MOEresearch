"""harness/executor.py — drive the e2e-bench-runner skill, return its
bench_summary in the shape `summary.json["regimes"]` expects.

We shell out to the existing skill impl rather than importing it because:
  1. The skill is the deterministic Layer-1 surface; we don't reach into its
     internals.
  2. Its impl runs prompt generation + concurrency with deterministic seed and
     we don't want to re-implement.

The executor also dumps a fresh regime YAML on disk (filtered by spec.regimes.only)
so the underlying skill sees exactly what the spec resolved.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_IMPL = REPO_ROOT / ".github" / "skills" / "e2e-bench-runner" / "impl" / "run_bench.py"


class ExecutorError(RuntimeError):
    pass


def run_bench(
    *,
    base_url: str,
    backend: str,
    submission_id: str,
    num_runs: int,
    resolved_regimes: Mapping[str, Any],
    out_dir: Path,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Run the e2e-bench-runner skill against an already-running server.

    Returns the bench_summary.json dict the skill emitted.

    Raises ExecutorError on hard failure (skill exited nonzero or summary missing).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the resolved regimes to a temp YAML so the skill picks it up via
    # its --regimes-file flag. We MUST wrap under top-level "regimes:" because
    # that's the schema the skill expects (see run_bench.py:~189).
    regimes_dump = out_dir / "regimes_resolved.yaml"
    regimes_dump.write_text(yaml.safe_dump({"regimes": dict(resolved_regimes)}))

    if not SKILL_IMPL.exists():
        raise ExecutorError(f"e2e-bench-runner impl not found at {SKILL_IMPL}")

    py = python_executable or sys.executable
    cmd = [
        py, str(SKILL_IMPL),
        "--url", base_url,
        "--backend", backend,
        "--tag", submission_id,
        "--num-runs", str(num_runs),
        "--regimes-file", str(regimes_dump),
        "--out-dir", str(out_dir),
    ]

    # Print the command for debuggability.
    print(f"[executor] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise ExecutorError(
            f"e2e-bench-runner exited with code {proc.returncode}. "
            f"stderr tail: {proc.stderr[-400:]!r}"
        )

    summary_path = out_dir / "bench_summary.json"
    if not summary_path.exists():
        raise ExecutorError(
            f"e2e-bench-runner exited 0 but {summary_path} missing"
        )
    try:
        return json.loads(summary_path.read_text())
    except json.JSONDecodeError as e:
        raise ExecutorError(f"bench_summary.json is not valid JSON: {e}") from e


def regimes_section_from_bench_summary(
    bench_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the skill's `regimes` block into summary.json schema v1.

    The bench-runner already produces close-to-target shape; we just
    pass-through (additionalProperties=True in schema accepts extras like
    `errors_count`)."""
    if not bench_summary.get("ok", False):
        raise ExecutorError(
            f"bench_summary reports ok=false: {bench_summary.get('error')}"
        )
    out = bench_summary.get("regimes", {})
    if not isinstance(out, dict) or not out:
        raise ExecutorError("bench_summary.regimes is empty or malformed")
    return dict(out)
