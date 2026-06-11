"""Tests for harness/spec.py — bench-spec loading, override merging, spec_hash.

Pin the BenchSpec interface contract before writing the implementation. These
tests describe the expected behavior; harness/spec.py must satisfy them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

# Import the module under test. These tests fail with ImportError until
# harness/spec.py exists — that's intentional (tests-first).
from harness.spec import BenchSpec, SpecValidationError

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MINIMAL_SPEC = FIXTURE_DIR / "minimal_spec.yaml"


def test_load_minimal_spec():
    spec = BenchSpec.load(MINIMAL_SPEC)
    assert spec.submission_id == "minimal-test-spec"
    assert spec.server.config == "configs/moe_qwen3_30b.yaml"
    assert spec.server.overrides.get("moe-runner-backend") == "triton"
    assert spec.server.overrides.get("_gpu_id") == 4
    assert spec.bench.num_runs == 3
    assert spec.bench.backend == "sglang"
    assert spec.quality_gate.type == "sanity"


def test_resolved_server_config_merges_overrides(tmp_path):
    """overrides must shadow base config keys, base must contribute the rest."""
    spec = BenchSpec.load(MINIMAL_SPEC)
    resolved = spec.resolved_server_config()
    # From overrides:
    assert resolved["moe-runner-backend"] == "triton"
    assert resolved["_gpu_id"] == 4
    assert resolved["port"] == 31000
    # From base config (configs/moe_qwen3_30b.yaml):
    assert "model-path" in resolved
    assert "served-model-name" in resolved


def test_spec_hash_is_deterministic():
    """Same spec content -> same hash, every time, every process."""
    spec1 = BenchSpec.load(MINIMAL_SPEC)
    spec2 = BenchSpec.load(MINIMAL_SPEC)
    assert spec1.spec_hash == spec2.spec_hash
    # Must be a sha256 hex digest (64 chars) prefixed with the algo name.
    assert spec1.spec_hash.startswith("sha256:")
    assert len(spec1.spec_hash) == len("sha256:") + 64


def test_spec_hash_includes_base_config(tmp_path):
    """If the base config file changes, spec_hash must change too — otherwise
    'same spec_hash = same result' guarantee is violated."""
    # Copy the fixture, then mutate the referenced base config locally and
    # verify hash changes.
    spec = BenchSpec.load(MINIMAL_SPEC)
    h_before = spec.spec_hash

    # Synthesize an alternate spec that points to a hand-rolled base config
    # with different content, but everything else identical.
    alt_base = tmp_path / "alt_base.yaml"
    alt_base.write_text(
        "model-path: /dummy\n"
        "served-model-name: dummy\n"
        "host: 127.0.0.1\n"
        "port: 31000\n"
    )
    alt_spec_yaml = tmp_path / "alt_spec.yaml"
    alt_spec_yaml.write_text(
        f"submission_id: minimal-test-spec\n"
        f"server:\n"
        f"  config: {alt_base}\n"
        f"  overrides:\n"
        f"    moe-runner-backend: triton\n"
        f"    _gpu_id: 4\n"
        f"    port: 31000\n"
        f"  conda_env: sglang-dev\n"
        f"  health_url: http://127.0.0.1:31000/health\n"
        f"  base_url: http://127.0.0.1:31000\n"
        f"regimes:\n"
        f"  file: regimes/qwen3_30b_moe_sglang_perf_sweep.yaml\n"
        f"bench:\n"
        f"  num_runs: 3\n"
        f"  reliable_stddev_pct: 8\n"
        f"  per_request_timeout_s: 600\n"
        f"  backend: sglang\n"
        f"quality_gate:\n"
        f"  type: sanity\n"
    )
    alt_spec = BenchSpec.load(alt_spec_yaml)
    h_after = alt_spec.spec_hash
    assert h_before != h_after, "spec_hash must reflect base config content"


def test_spec_hash_excludes_volatile_fields(tmp_path):
    """description and tags are metadata; they must NOT influence spec_hash
    (otherwise editing prose breaks reproducibility)."""
    spec1 = BenchSpec.load(MINIMAL_SPEC)

    edited = tmp_path / "edited_spec.yaml"
    edited.write_text(
        MINIMAL_SPEC.read_text().replace(
            'description: "Minimal fixture used by harness/tests; never actually launches a server."',
            'description: "EDITED DESCRIPTION"',
        )
    )
    spec2 = BenchSpec.load(edited)
    assert spec1.spec_hash == spec2.spec_hash, (
        "Editing description (metadata) must not change spec_hash"
    )


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("submission_id: oops\n")  # no server/regimes/bench
    with pytest.raises(SpecValidationError):
        BenchSpec.load(bad)


def test_unknown_top_level_key_rejected(tmp_path):
    """Strict schema: unknown keys are a typo and must fail loudly."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(MINIMAL_SPEC.read_text() + "\nfoo_typo_field: 123\n")
    with pytest.raises(SpecValidationError):
        BenchSpec.load(bad)


def test_quality_gate_optional(tmp_path):
    """quality_gate is optional; default = sanity."""
    text = MINIMAL_SPEC.read_text().split("quality_gate:")[0]
    spec_path = tmp_path / "no_qg.yaml"
    spec_path.write_text(text)
    spec = BenchSpec.load(spec_path)
    assert spec.quality_gate.type == "sanity"


def test_gpu_id_included_in_spec():
    """User decision (2026-06-11): GPU id IS part of bench-spec."""
    spec = BenchSpec.load(MINIMAL_SPEC)
    resolved = spec.resolved_server_config()
    assert resolved["_gpu_id"] == 4
