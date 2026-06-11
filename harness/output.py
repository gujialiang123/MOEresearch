"""harness/output.py — summary.json writer + jsonschema validator.

Schema v1. All keys are required unless marked optional below.

Bumping `SUMMARY_SCHEMA_VERSION` MUST be paired with a schema migration: the
field shape is consumed by downstream tooling (cross-regime-anomaly, agent
decision loop) and silent shape drift is a known footgun.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema

SUMMARY_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Schema definition (jsonschema draft-2020-12 compatible)
# ---------------------------------------------------------------------------

_NUMBER = {"type": "number"}
_INT = {"type": "integer"}
_STRING = {"type": "string"}
_BOOL = {"type": "boolean"}

_PER_RUN_STATS = {
    "type": "object",
    "required": ["mean", "stddev", "stddev_pct", "runs"],
    "properties": {
        "mean": {"type": ["number", "null"]},
        "stddev": {"type": ["number", "null"]},
        "stddev_pct": {"type": ["number", "null"]},
        "runs": {"type": "array", "items": _NUMBER},
    },
}

_PERCENTILES = {
    "type": "object",
    "required": ["p50", "p99"],
    "properties": {"p50": _NUMBER, "p99": _NUMBER},
}  # reserved for future ttft/itl when streaming bench lands

_E2E_LATENCY = {
    "type": "object",
    "required": ["p50", "p99", "count"],
    "properties": {
        "p50": _NUMBER,
        "p99": _NUMBER,
        "count": _INT,
    },
}

_PERCENTILES = {
    "type": "object",
    "required": ["p50", "p99"],
    "properties": {"p50": _NUMBER, "p99": _NUMBER},
}

_REGIME_ENTRY = {
    "type": "object",
    "required": [
        "num_prompts", "prompt_words", "max_new", "concurrency",
        "req_per_s", "tokens_per_s", "e2e_ms",
        "completion_rate", "wall_s", "reliable",
    ],
    "properties": {
        "num_prompts": _INT,
        "prompt_words": _INT,
        "max_new": _INT,
        "concurrency": _INT,
        "req_per_s": _PER_RUN_STATS,
        "tokens_per_s": _PER_RUN_STATS,
        "e2e_ms": _E2E_LATENCY,
        "completion_rate": _NUMBER,
        "wall_s": _PER_RUN_STATS,
        "reliable": _BOOL,
    },
    "additionalProperties": True,
}

_QUALITY_CHECK_RESULT = {
    "type": "object",
    "required": ["passed"],
    "properties": {
        "passed": _BOOL,
        "value": _NUMBER,
        "threshold": _NUMBER,
        "message": _STRING,
    },
    "additionalProperties": True,
}

SUMMARY_SCHEMA: Mapping[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version", "ok", "submission_id", "spec_hash", "captured_at",
        "spec_resolved", "environment", "server", "regimes", "quality_gate",
        "warnings",
    ],
    "properties": {
        "schema_version": {"const": SUMMARY_SCHEMA_VERSION},
        "ok": _BOOL,
        "submission_id": _STRING,
        "spec_hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        "captured_at": _STRING,  # ISO 8601
        "spec_resolved": {
            "type": "object",
            "required": ["server_config", "regimes", "bench", "quality_gate"],
            "properties": {
                "server_config": {"type": "object"},
                "regimes": {"type": "object"},
                "bench": {"type": "object"},
                "quality_gate": {"type": "object"},
            },
            "additionalProperties": True,
        },
        "environment": {
            "type": "object",
            "required": ["hostname", "gpu", "driver", "cuda", "engine_version", "git"],
            "properties": {
                "hostname": _STRING,
                "gpu": {
                    "type": "object",
                    "required": ["name", "uuid", "id", "sm"],
                    "properties": {
                        "name": _STRING, "uuid": _STRING,
                        "id": _INT, "sm": _STRING,
                    },
                    "additionalProperties": True,
                },
                "driver": _STRING,
                "cuda": _STRING,
                "engine_version": {"type": "object"},
                "git": {
                    "type": "object",
                    "required": ["commit", "dirty"],
                    "properties": {"commit": _STRING, "dirty": _BOOL},
                },
            },
            "additionalProperties": True,
        },
        "server": {
            "type": "object",
            "required": ["startup_wall_s", "first_health_at_s", "log_path"],
            "properties": {
                "startup_wall_s": _NUMBER,
                "first_health_at_s": _NUMBER,
                "log_path": _STRING,
            },
        },
        "regimes": {
            "type": "object",
            "additionalProperties": _REGIME_ENTRY,
        },
        "quality_gate": {
            "type": "object",
            "required": ["type", "passed", "checks"],
            "properties": {
                "type": {"enum": ["sanity", "ppl", "none"]},
                "passed": _BOOL,
                "checks": {
                    "type": "object",
                    "additionalProperties": _QUALITY_CHECK_RESULT,
                },
            },
            "additionalProperties": True,
        },
        "warnings": {"type": "array", "items": _STRING},
        # Optional, only present when ok=false
        "error": {
            "type": "object",
            "required": ["phase", "message"],
            "properties": {"phase": _STRING, "message": _STRING},
            "additionalProperties": True,
        },
    },
    "additionalProperties": False,
}


def validate_summary(summary: Mapping[str, Any]) -> None:
    """Raise jsonschema.ValidationError if `summary` doesn't match schema v1."""
    jsonschema.validate(instance=summary, schema=SUMMARY_SCHEMA)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class SummaryWriter:
    """Writes summary.json to out_dir. Validates against schema before writing
    when input is well-formed; writes anyway with `ok=false` if schema-invalid,
    so the harness never silently loses a failure record."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, summary: Mapping[str, Any]) -> Path:
        """Validate (best-effort) and write summary.json. Returns the written path."""
        try:
            validate_summary(summary)
        except jsonschema.ValidationError:
            # Re-raise — caller (run_bench.py) MUST handle. We don't silently
            # downgrade because schema-invalid output corrupts downstream tools.
            raise
        target = self.out_dir / "summary.json"
        target.write_text(json.dumps(summary, indent=2, sort_keys=True))
        return target
