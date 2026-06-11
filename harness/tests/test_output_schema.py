"""Tests for harness/output.py — summary.json schema v1 contract.

Pin the OutputWriter interface and schema BEFORE writing the implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.output import (
    SummaryWriter,
    SUMMARY_SCHEMA_VERSION,
    validate_summary,
)


def _minimal_valid_summary() -> dict:
    """Hand-rolled summary that should validate against schema v1."""
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "ok": True,
        "submission_id": "minimal-test-spec",
        "spec_hash": "sha256:" + "0" * 64,
        "captured_at": "2026-06-11T18:00:00Z",
        "spec_resolved": {
            "server_config": {"model-path": "/x", "port": 31000, "_gpu_id": 4},
            "regimes": {"R_medium": {"num_prompts": 16}},
            "bench": {"num_runs": 3, "backend": "sglang"},
            "quality_gate": {"type": "sanity"},
        },
        "environment": {
            "hostname": "h200-node",
            "gpu": {"name": "NVIDIA H200", "uuid": "GPU-xxx", "id": 4, "sm": "90"},
            "driver": "550.x",
            "cuda": "12.4",
            "engine_version": {"sglang": "0.5.12.post1", "flashinfer": "0.6.11"},
            "git": {"commit": "abc1234", "dirty": False},
        },
        "server": {
            "startup_wall_s": 87.3,
            "first_health_at_s": 87.3,
            "log_path": "results/x/server.log",
        },
        "regimes": {
            "R_medium": {
                "num_prompts": 16,
                "prompt_words": 800,
                "max_new": 256,
                "concurrency": 8,
                "req_per_s":    {"mean": 4.49, "stddev": 0.10, "stddev_pct": 2.2, "runs": [4.40, 4.58]},
                "tokens_per_s": {"mean": 1135, "stddev": 22, "stddev_pct": 1.9, "runs": [1120, 1150]},
                "e2e_ms":       {"p50": 1820, "p99": 2950, "count": 32},
                "completion_rate": 1.0,
                "wall_s":       {"mean": 3.56, "stddev": 0.08, "stddev_pct": 2.2, "runs": [3.50, 3.62]},
                "reliable": True,
            }
        },
        "quality_gate": {
            "type": "sanity",
            "passed": True,
            "checks": {
                "completion_rate_min": {"value": 1.0, "threshold": 0.99, "passed": True},
                "output_not_empty": {"passed": True},
                "output_not_constant": {"passed": True},
            },
        },
        "warnings": [],
    }


def test_valid_summary_passes(tmp_path):
    s = _minimal_valid_summary()
    validate_summary(s)  # must not raise


def test_writer_emits_valid_json(tmp_path):
    s = _minimal_valid_summary()
    out_dir = tmp_path / "out"
    SummaryWriter(out_dir).write(s)
    written = json.loads((out_dir / "summary.json").read_text())
    assert written == s
    validate_summary(written)


def test_writer_writes_even_on_failure(tmp_path):
    """User constraint: harness must produce schema-valid summary.json with
    ok=false rather than crashing. Downstream tools should never need try/except."""
    s = _minimal_valid_summary()
    s["ok"] = False
    s["error"] = {"phase": "server_startup", "message": "port 31000 already in use"}
    s["regimes"] = {}
    s["quality_gate"] = {"type": "sanity", "passed": False, "checks": {}}
    out_dir = tmp_path / "out"
    SummaryWriter(out_dir).write(s)
    written = json.loads((out_dir / "summary.json").read_text())
    assert written["ok"] is False
    assert written["error"]["phase"] == "server_startup"
    validate_summary(written)


def test_missing_required_field_rejected():
    s = _minimal_valid_summary()
    del s["spec_hash"]
    with pytest.raises(Exception):
        validate_summary(s)


def test_schema_version_is_one():
    assert SUMMARY_SCHEMA_VERSION == 1
