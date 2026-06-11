"""harness/spec.py — BenchSpec dataclass + validation + spec_hash.

A BenchSpec is the deterministic input to `harness/run_bench.py`. Same spec
content (including referenced base config + regimes YAML) → same `spec_hash` →
same `summary.json` (modulo run-to-run noise reported as stddev).

Hashing rule (CRITICAL for reproducibility):
    spec_hash = sha256(
        canonical_json(stable_spec_fields)
        + canonical_json(resolved_server_config)
        + canonical_json(resolved_regimes)
    )
where `stable_spec_fields` excludes metadata (description, tags) that should be
free to edit without invalidating prior runs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


# Repo root is the parent of `harness/`.
REPO_ROOT = Path(__file__).resolve().parent.parent


class SpecValidationError(ValueError):
    """Raised when a bench-spec is malformed or references a missing file."""


_ALLOWED_TOP_LEVEL_KEYS = {
    "submission_id", "description", "tags",
    "server", "regimes", "bench", "quality_gate",
}
_ALLOWED_SERVER_KEYS = {
    "config", "overrides", "conda_env",
    "health_url", "base_url", "startup_timeout_s",
}
_ALLOWED_REGIMES_KEYS = {"file", "only", "inline"}
_ALLOWED_BENCH_KEYS = {
    "num_runs", "reliable_stddev_pct", "per_request_timeout_s", "backend",
}
_ALLOWED_QUALITY_GATE_KEYS = {"type"}

_VALID_QG_TYPES = {"sanity", "ppl", "none"}
_VALID_BACKENDS = {"sglang", "vllm"}


@dataclass(frozen=True)
class _Server:
    config: str
    overrides: Mapping[str, Any]
    conda_env: str
    health_url: str
    base_url: str
    startup_timeout_s: int


@dataclass(frozen=True)
class _Regimes:
    file: str
    only: Sequence[str] | None = None
    inline: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _Bench:
    num_runs: int
    reliable_stddev_pct: float
    per_request_timeout_s: int
    backend: str


@dataclass(frozen=True)
class _QualityGate:
    type: str = "sanity"


@dataclass(frozen=True)
class BenchSpec:
    submission_id: str
    server: _Server
    regimes: _Regimes
    bench: _Bench
    quality_gate: _QualityGate
    # Free-form metadata; excluded from spec_hash.
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    # The directory of the spec file, used to resolve relative paths.
    _spec_path: Path = field(default=Path("."))

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    @classmethod
    def load(cls, spec_path: Path | str) -> "BenchSpec":
        spec_path = Path(spec_path).resolve()
        if not spec_path.exists():
            raise SpecValidationError(f"Spec file not found: {spec_path}")
        try:
            raw = yaml.safe_load(spec_path.read_text())
        except yaml.YAMLError as e:
            raise SpecValidationError(f"YAML parse error in {spec_path}: {e}") from e
        if not isinstance(raw, dict):
            raise SpecValidationError(f"Spec must be a YAML mapping at top level: {spec_path}")

        cls._validate_keys(raw, _ALLOWED_TOP_LEVEL_KEYS, "(top-level)", spec_path)
        for required in ("submission_id", "server", "regimes", "bench"):
            if required not in raw:
                raise SpecValidationError(f"Missing required key '{required}' in {spec_path}")

        # Server
        server_raw = raw["server"]
        if not isinstance(server_raw, dict):
            raise SpecValidationError("server: must be a mapping")
        cls._validate_keys(server_raw, _ALLOWED_SERVER_KEYS, "server", spec_path)
        for required in ("config", "conda_env", "health_url", "base_url"):
            if required not in server_raw:
                raise SpecValidationError(f"Missing server.{required} in {spec_path}")
        server = _Server(
            config=server_raw["config"],
            overrides=dict(server_raw.get("overrides") or {}),
            conda_env=server_raw["conda_env"],
            health_url=server_raw["health_url"],
            base_url=server_raw["base_url"],
            startup_timeout_s=int(server_raw.get("startup_timeout_s", 600)),
        )

        # Regimes
        regimes_raw = raw["regimes"]
        if not isinstance(regimes_raw, dict):
            raise SpecValidationError("regimes: must be a mapping")
        cls._validate_keys(regimes_raw, _ALLOWED_REGIMES_KEYS, "regimes", spec_path)
        if "file" not in regimes_raw and "inline" not in regimes_raw:
            raise SpecValidationError("regimes: must specify either 'file' or 'inline'")
        regimes = _Regimes(
            file=regimes_raw.get("file", ""),
            only=tuple(regimes_raw["only"]) if regimes_raw.get("only") else None,
            inline=regimes_raw.get("inline"),
        )

        # Bench
        bench_raw = raw["bench"]
        if not isinstance(bench_raw, dict):
            raise SpecValidationError("bench: must be a mapping")
        cls._validate_keys(bench_raw, _ALLOWED_BENCH_KEYS, "bench", spec_path)
        for required in ("num_runs", "backend"):
            if required not in bench_raw:
                raise SpecValidationError(f"Missing bench.{required}")
        backend = str(bench_raw["backend"])
        if backend not in _VALID_BACKENDS:
            raise SpecValidationError(
                f"bench.backend must be one of {sorted(_VALID_BACKENDS)}, got {backend!r}"
            )
        bench = _Bench(
            num_runs=int(bench_raw["num_runs"]),
            reliable_stddev_pct=float(bench_raw.get("reliable_stddev_pct", 8.0)),
            per_request_timeout_s=int(bench_raw.get("per_request_timeout_s", 600)),
            backend=backend,
        )

        # Quality gate
        qg_raw = raw.get("quality_gate") or {}
        if not isinstance(qg_raw, dict):
            raise SpecValidationError("quality_gate: must be a mapping")
        cls._validate_keys(qg_raw, _ALLOWED_QUALITY_GATE_KEYS, "quality_gate", spec_path)
        qg_type = qg_raw.get("type", "sanity")
        if qg_type not in _VALID_QG_TYPES:
            raise SpecValidationError(
                f"quality_gate.type must be one of {sorted(_VALID_QG_TYPES)}, got {qg_type!r}"
            )
        quality_gate = _QualityGate(type=qg_type)

        return cls(
            submission_id=str(raw["submission_id"]),
            description=str(raw.get("description", "")),
            tags=tuple(raw.get("tags") or []),
            server=server,
            regimes=regimes,
            bench=bench,
            quality_gate=quality_gate,
            _spec_path=spec_path,
        )

    @staticmethod
    def _validate_keys(d: Mapping[str, Any], allowed: set[str], where: str, path: Path) -> None:
        unknown = set(d.keys()) - allowed
        if unknown:
            raise SpecValidationError(
                f"Unknown key(s) {sorted(unknown)} under {where} in {path}. "
                f"Allowed: {sorted(allowed)}. (Typo?)"
            )

    # -----------------------------------------------------------------------
    # Resolution
    # -----------------------------------------------------------------------

    def _resolve_path(self, p: str | Path) -> Path:
        """Resolve a path relative to repo root first, then relative to the
        spec file's directory."""
        p = Path(p)
        if p.is_absolute():
            return p
        # Prefer repo-root-relative (matches how configs/ and regimes/ are referenced).
        candidate = (REPO_ROOT / p).resolve()
        if candidate.exists():
            return candidate
        # Fall back to spec-dir-relative (handy for ad-hoc specs in /tmp).
        return (self._spec_path.parent / p).resolve()

    def resolved_server_config(self) -> dict[str, Any]:
        """Load the base server config YAML, merge overrides on top, return dict."""
        base_path = self._resolve_path(self.server.config)
        if not base_path.exists():
            raise SpecValidationError(
                f"server.config references missing file: {self.server.config} "
                f"(resolved to {base_path})"
            )
        try:
            base = yaml.safe_load(base_path.read_text()) or {}
        except yaml.YAMLError as e:
            raise SpecValidationError(f"YAML parse error in {base_path}: {e}") from e
        if not isinstance(base, dict):
            raise SpecValidationError(f"Base server config must be a mapping: {base_path}")
        merged = dict(base)
        merged.update(self.server.overrides)
        return merged

    def resolved_regimes(self) -> dict[str, Any]:
        """Load the regimes YAML (or return inline), optionally filtered by 'only'."""
        if self.regimes.inline is not None:
            regimes = dict(self.regimes.inline)
        else:
            r_path = self._resolve_path(self.regimes.file)
            if not r_path.exists():
                raise SpecValidationError(
                    f"regimes.file references missing file: {self.regimes.file} "
                    f"(resolved to {r_path})"
                )
            try:
                doc = yaml.safe_load(r_path.read_text()) or {}
            except yaml.YAMLError as e:
                raise SpecValidationError(f"YAML parse error in {r_path}: {e}") from e
            # The regime files in regimes/ wrap entries under top-level "regimes:".
            if isinstance(doc, dict) and "regimes" in doc:
                regimes = dict(doc["regimes"])
            elif isinstance(doc, dict):
                regimes = doc
            else:
                raise SpecValidationError(f"Regimes YAML must be a mapping: {r_path}")
        if self.regimes.only:
            missing = [r for r in self.regimes.only if r not in regimes]
            if missing:
                raise SpecValidationError(
                    f"regimes.only references unknown regime(s): {missing}. "
                    f"Available: {sorted(regimes.keys())}"
                )
            regimes = {k: regimes[k] for k in self.regimes.only}
        return regimes

    # -----------------------------------------------------------------------
    # Hashing
    # -----------------------------------------------------------------------

    def _stable_payload(self) -> dict[str, Any]:
        """Subset of spec used for hashing — excludes metadata (description,
        tags) that should be free to edit without breaking reproducibility."""
        return {
            "submission_id": self.submission_id,
            "server": {
                "config_basename": Path(self.server.config).name,  # path-stable
                "overrides": dict(self.server.overrides),
                "conda_env": self.server.conda_env,
                "health_url": self.server.health_url,
                "base_url": self.server.base_url,
                "startup_timeout_s": self.server.startup_timeout_s,
            },
            "regimes": {
                "file_basename": Path(self.regimes.file).name if self.regimes.file else None,
                "only": list(self.regimes.only) if self.regimes.only else None,
                "inline": dict(self.regimes.inline) if self.regimes.inline else None,
            },
            "bench": {
                "num_runs": self.bench.num_runs,
                "reliable_stddev_pct": self.bench.reliable_stddev_pct,
                "per_request_timeout_s": self.bench.per_request_timeout_s,
                "backend": self.bench.backend,
            },
            "quality_gate": {"type": self.quality_gate.type},
        }

    @property
    def spec_hash(self) -> str:
        """Deterministic hash of (stable spec fields + resolved base config +
        resolved regimes). Same hash ⇒ same expected output (modulo noise)."""
        payload = {
            "stable_spec": self._stable_payload(),
            "resolved_server_config": self.resolved_server_config(),
            "resolved_regimes": self.resolved_regimes(),
        }
        # Canonical JSON: sort_keys + no whitespace.
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
