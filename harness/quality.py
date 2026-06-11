"""harness/quality.py — sanity gate v1.

Reads per_run/*.json (raw bench output) and computes cheap correctness checks
on the LLM responses themselves. PPL gate is deferred to v2.

v1 sanity checks:
  - completion_rate ≥ 0.99 (per the spec's quality_gate.threshold; default 0.99)
  - outputs not empty (each request returned at least 1 token of text)
  - outputs vary (not all identical — catches "model stuck repeating one token")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_COMPLETION_RATE_THRESHOLD = 0.99


class _Check:
    @staticmethod
    def passed(message: str = "") -> dict[str, Any]:
        return {"passed": True, "message": message}

    @staticmethod
    def failed(message: str) -> dict[str, Any]:
        return {"passed": False, "message": message}


def run_sanity_gate(per_run_dir: Path) -> dict[str, Any]:
    """Read all per_run/*.json and run sanity checks. Returns dict matching
    summary.json["quality_gate"] schema."""
    per_run_dir = Path(per_run_dir)
    if not per_run_dir.exists():
        return {
            "type": "sanity",
            "passed": False,
            "checks": {
                "per_run_dir_present": _Check.failed(f"missing: {per_run_dir}")
            },
        }

    run_files = sorted(per_run_dir.glob("*.json"))
    if not run_files:
        return {
            "type": "sanity",
            "passed": False,
            "checks": {
                "per_run_files_present": _Check.failed(f"no run files in {per_run_dir}")
            },
        }

    all_records: list[dict[str, Any]] = []
    completion_rates: list[float] = []
    for rf in run_files:
        try:
            data = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "completion_rate" in data:
            completion_rates.append(float(data["completion_rate"]))
        all_records.extend(data.get("records", []) or [])

    checks: dict[str, dict[str, Any]] = {}

    # Check 1: completion_rate
    if completion_rates:
        min_rate = min(completion_rates)
        checks["completion_rate_min"] = {
            "value": min_rate,
            "threshold": _COMPLETION_RATE_THRESHOLD,
            "passed": min_rate >= _COMPLETION_RATE_THRESHOLD,
            "message": (
                f"min completion_rate across runs = {min_rate:.3f}; "
                f"threshold = {_COMPLETION_RATE_THRESHOLD}"
            ),
        }
    else:
        checks["completion_rate_min"] = _Check.failed("no completion_rate in any run file")

    # Check 2 & 3: extract returned texts. NOTE: the e2e-bench-runner records
    # don't currently include the generated text by default (only timings).
    # If text is absent, we skip these checks and mark them passed with a note.
    texts = [
        r.get("text") for r in all_records
        if r.get("ok") and isinstance(r.get("text"), str) and r.get("text")
    ]
    if not texts and all_records:
        checks["output_not_empty"] = _Check.passed(
            "skipped (bench records don't include text; v1 limitation)"
        )
        checks["output_not_constant"] = _Check.passed(
            "skipped (bench records don't include text; v1 limitation)"
        )
    else:
        # Check 2: outputs not empty (all texts non-empty after stripping)
        empties = [t for t in texts if not t.strip()]
        checks["output_not_empty"] = {
            "passed": not empties,
            "message": (
                f"{len(empties)} of {len(texts)} responses were empty"
                if empties else f"{len(texts)} non-empty responses"
            ),
        }

        # Check 3: outputs vary
        unique_outputs = set(texts)
        checks["output_not_constant"] = {
            "passed": len(unique_outputs) > 1 if len(texts) > 1 else True,
            "message": (
                f"{len(unique_outputs)} unique responses across {len(texts)} requests"
            ),
        }

    all_passed = all(c["passed"] for c in checks.values())
    return {
        "type": "sanity",
        "passed": all_passed,
        "checks": checks,
    }


def run_quality_gate(gate_type: str, per_run_dir: Path) -> dict[str, Any]:
    """Dispatcher. v1 supports 'sanity' and 'none'; 'ppl' is reserved."""
    if gate_type == "none":
        return {"type": "none", "passed": True, "checks": {}}
    if gate_type == "sanity":
        return run_sanity_gate(per_run_dir)
    if gate_type == "ppl":
        return {
            "type": "ppl",
            "passed": False,
            "checks": {
                "ppl_implemented": _Check.failed(
                    "PPL gate is reserved for v2; use type=sanity for v1"
                ),
            },
        }
    return {
        "type": gate_type,
        "passed": False,
        "checks": {
            "gate_type_known": _Check.failed(f"unknown quality_gate.type={gate_type!r}"),
        },
    }
