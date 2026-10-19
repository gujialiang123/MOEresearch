"""Read-only same-state diagnostics and paired prompt-level inference.

The observer receives the *actual residual-addition input*, separately from the
normalized MoE input. Driver output, expert selection and cache are never touched.
Only per-token scalar diagnostics and per-prompt FP64 sufficient moments survive;
whole hidden states are neither retained nor written to disk.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np
import torch

from .state_calibration import NumericalProtection, StreamingMoments, canonical_hash


DEFAULT_POSITION_BINS = ((1, 32), (33, 128), (129, 256), (257, 512),
                         (513, 1024), (1025, 2047))
SUM_FIELDS = ("native_norm", "u4_norm", "z4_norm", "residual_norm", "error_norm",
              "residual_relative_error", "projection_error_signed", "orthogonal_error_norm",
              "cos_u4_y8", "cos_defined")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive(temporary, destination):
    # Same-directory hard-link publication is atomic and cannot overwrite a
    # committed/raw filename, unlike os.replace. The temporary is disposable.
    os.link(temporary, destination)
    temporary.unlink()


def _atomic_json_exclusive(destination, value):
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.pending"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _publish_exclusive(temporary, destination)
    _fsync_directory(destination.parent)


def position_bin(index: int, bins=DEFAULT_POSITION_BINS) -> str:
    if index < 1:
        raise ValueError("decode_forward_index is one-based; prefill is excluded")
    for lo, hi in bins:
        if lo <= index <= hi:
            return f"{lo}-{hi}"
    raise ValueError(f"decode index {index} outside declared position bins")


class StateSufficientStats:
    """One prompt/path/layer/position group, with stable centered moments."""
    def __init__(self):
        self.native = StreamingMoments()
        self.calibrated = StreamingMoments()
        self.sums: torch.Tensor | None = None
        self.positions: list[tuple[int, int]] = []
        self.driver_hash: str | None = None
        self.calibration_hash: str | None = None

    @property
    def count(self) -> int:
        return self.native.count

    @property
    def mask_hash(self) -> str:
        return canonical_hash(self.positions)

    def update(self, y8, z4, scalar_values, index, token_id, driver_hash, calibration_hash):
        if self.positions and index <= self.positions[-1][0]:
            raise ValueError("duplicate or out-of-order decode position")
        if self.count and (self.driver_hash != driver_hash or self.calibration_hash != calibration_hash):
            raise ValueError("driver or calibration changed inside a prompt statistics group")
        self.driver_hash, self.calibration_hash = driver_hash, calibration_hash
        self.native.update(y8)
        self.calibrated.update(z4)
        values = scalar_values.detach().to(dtype=torch.float64)
        self.sums = values.clone() if self.sums is None else self.sums + values
        self.positions.append((int(index), int(token_id)))

    def cpu(self):
        self.native = self.native.cpu()
        self.calibrated = self.calibrated.cpu()
        if self.sums is not None:
            self.sums = self.sums.detach().cpu()
        return self


class StateCollector:
    """Collect full, predeclared-bin and optional shared-prefix diagnostics.

    Call ``finish_prompt`` after each replayed prompt to release its GPU moments.
    With ``output_dir`` this also persists a compressed NPZ and scalar JSONL for
    that prompt; the whole run need not finish before raw data becomes durable.
    ``common_decode_limit`` must be fixed from trace lengths before observation,
    and creates additional ``common`` / ``common:<bin>`` groups. It is never
    inferred from mismatch. A/B and C/D comparisons require identical masks.
    """
    SCHEMA_VERSION = "state_collection.v2"

    def __init__(self, output_dir: str | Path | None = None, *,
                 protection: NumericalProtection | None = None,
                 bins=DEFAULT_POSITION_BINS):
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.protection = protection or NumericalProtection()
        self.bins = tuple((int(lo), int(hi)) for lo, hi in bins)
        previous = 0
        for lo, hi in self.bins:
            if lo != previous + 1 or hi < lo:
                raise ValueError("position bins must form a contiguous one-based partition")
            previous = hi
        self.records: dict[tuple[str, str, int, str], StateSufficientStats] = {}
        self.token_rows: dict[tuple[str, str], list[dict]] = {}
        self.finished: set[tuple[str, str]] = set()
        self.empty_records: dict[tuple[str, str], str] = {}
        self.empty_metadata: dict[tuple[str, str], dict] = {}
        self.incomplete_archives: list[str] = []
        self._common_limits: dict[tuple[str, str], int | None] = {}

    def observe(self, *, prompt_id: Any, path: str, layer: int,
                decode_forward_index: int, token_id: int, y8: torch.Tensor,
                u4: torch.Tensor, z4: torch.Tensor, residual: torch.Tensor,
                driver_hash: str, calibration_hash: str,
                common_decode_limit: int | None = None) -> dict:
        pid, path = str(prompt_id), str(path)
        prompt_key = (pid, path)
        if prompt_key in self.finished:
            raise ValueError("cannot append positions to a finished prompt/path")
        if not driver_hash or not calibration_hash:
            raise ValueError("driver_hash and observed calibration_hash must be explicit")
        if int(layer) < 0:
            raise ValueError("layer must be nonnegative")
        index = int(decode_forward_index)
        group = position_bin(index, self.bins)
        if common_decode_limit is not None and common_decode_limit < 0:
            raise ValueError("common_decode_limit must be nonnegative")
        if prompt_key in self._common_limits and self._common_limits[prompt_key] != common_decode_limit:
            raise ValueError("common position mask changed during a replay")
        self._common_limits[prompt_key] = common_decode_limit
        branches = (y8, u4, z4, residual)
        if any(not isinstance(x, torch.Tensor) or not x.is_floating_point() for x in branches):
            raise ValueError("branches and actual residual must be floating tensors")
        if any(x.shape != y8.shape or x.device != y8.device for x in branches):
            raise ValueError("same-state branches and residual must share shape and device")
        if y8.ndim < 1 or y8.shape[-1] == 0 or y8.numel() != y8.shape[-1]:
            raise ValueError("observe accepts exactly one token at a time (batch=1)")
        native, raw, calibrated, resid = [x.detach().reshape(-1).double() for x in branches]
        n8, nu, nz, nr = [x.norm() for x in (native, raw, calibrated, resid)]
        error = calibrated - native
        ne = error.norm()
        eps = self.protection.eps_diagnostic_norm
        direction = native / (n8 + eps)
        projection = torch.dot(error, direction)
        orthogonal = (error - projection * direction).norm()
        cos_defined = ((n8 > 0) & (nu > 0)).double()
        cosine = torch.dot(raw, native) / (n8 * nu + eps ** 2)
        scalar_values = torch.stack((n8, nu, nz, nr, ne, ne / (nr + eps),
                                     projection, orthogonal, cosine, cos_defined))
        # One tiny scalar transfer, not hidden-dimension activations. It also
        # validates non-finite values before a misleading row can be written.
        values = scalar_values.cpu().tolist()
        if any(not math.isfinite(v) for v in values):
            raise ValueError(f"non-finite diagnostic at {pid}/{path}/{layer}/{index}")
        groups = ["all", group]
        if common_decode_limit is not None and index <= common_decode_limit:
            groups.extend(("common", f"common:{group}"))
        # Validate all groups before updating any of them.
        for name in groups:
            existing = self.records.get((pid, path, int(layer), name))
            if existing and existing.positions and index <= existing.positions[-1][0]:
                raise ValueError("duplicate or out-of-order decode position")
            if existing and (existing.driver_hash != driver_hash or
                             existing.calibration_hash != calibration_hash):
                raise ValueError("driver or calibration changed during a replay")
        for name in groups:
            stats = self.records.setdefault((pid, path, int(layer), name), StateSufficientStats())
            stats.update(y8, z4, scalar_values, index, token_id, driver_hash, calibration_hash)
        row = {"prompt_id": pid, "path": path, "layer": int(layer),
               "decode_forward_index": index, "token_id": int(token_id),
               "position_bin": group, "driver_hash": driver_hash,
               "calibration_hash": calibration_hash,
               **dict(zip(SUM_FIELDS, values))}
        row["cos_defined"] = bool(row["cos_defined"])
        if not row["cos_defined"]:
            row["cos_u4_y8"] = None
        self.token_rows.setdefault(prompt_key, []).append(row)
        return row

    def finish_prompt(self, prompt_id: Any, path: str, *, allow_empty: bool = False,
                      empty_reason: str = "trace_has_no_valid_decode_positions",
                      empty_metadata: Mapping[str, Any] | None = None) -> Path | None:
        key = (str(prompt_id), str(path))
        if key in self.finished:
            raise ValueError("prompt/path already finished")
        matches = [(k, v) for k, v in self.records.items() if k[:2] == key]
        if not matches:
            if not allow_empty:
                raise ValueError("cannot finish a prompt with no valid decode observations")
            if not isinstance(empty_reason, str) or not empty_reason:
                raise ValueError("empty observation reason must be explicit")
            metadata = json.loads(json.dumps(dict(empty_metadata or {}), sort_keys=True, allow_nan=False))
            destination = None
            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                destination = self._write_empty_marker(self.output_dir, key, empty_reason, metadata)
            self.empty_records[key] = empty_reason
            self.empty_metadata[key] = metadata
            self.finished.add(key)
            return destination
        for _, stats in matches:
            stats.cpu()
        destination = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Filename contains no untrusted prompt/path characters.
            stem = canonical_hash(list(key))
            destination = self.output_dir / f"{stem}.npz"
            self._publish_archive(self.output_dir, stem, matches, self.token_rows.get(key, []))
            self.token_rows.pop(key, None)
        self.finished.add(key)
        return destination

    def _write_empty_marker(self, directory, key, reason, metadata=None):
        destination = directory / f"{canonical_hash(list(key))}.empty.json"
        marker = {"schema_version": "state_collection.empty.v1",
                  "collection_schema_version": self.SCHEMA_VERSION,
                  "prompt_id": key[0], "path": key[1], "count": 0, "reason": reason,
                  "dtype": "float64", "ddof": 0, "position_bins": self.bins,
                  "numerical_protection": asdict(self.protection),
                  "metadata": dict(metadata or {}),
                  "state_evidence": "no_valid_decode_observations; no moments or activations imputed"}
        _atomic_json_exclusive(destination, marker)
        return destination

    def _publish_archive(self, directory, stem, entries, rows):
        """Commit a statistics/token pair last, with crash-safe visibility.

        Final files without a commit are never considered finished. Retrying
        preserves such remnants in an uncommitted directory. A committed pair
        is immutable and cannot be replaced by a retry under the same run ID.
        """
        entries = list(entries)
        expected_rows = sum(stats.count for key, stats in entries if key[3] == "all")
        if expected_rows != len(rows):
            raise ValueError("token rows do not cover every archived layer/position; use original per-prompt archives")
        archive = directory / f"{stem}.npz"
        tokens = directory / f"{stem}.tokens.jsonl"
        committed = directory / f"{stem}.complete.json"
        if committed.exists():
            raise FileExistsError(f"committed diagnostic archive exists: {committed}")
        remnants = [path for path in (archive, tokens) if path.exists()]
        if remnants:
            quarantine = directory / "uncommitted" / f"{stem}.{uuid.uuid4().hex}"
            quarantine.mkdir(parents=True)
            for path in remnants:
                path.rename(quarantine / path.name)
        transaction = uuid.uuid4().hex
        temporary_archive = directory / f".{stem}.{transaction}.npz.pending"
        temporary_tokens = directory / f".{stem}.{transaction}.tokens.pending"
        self._write_archive(temporary_archive, entries)
        with temporary_tokens.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        commit = {"schema_version": "state_collection.commit.v1",
                  "collection_schema_version": self.SCHEMA_VERSION,
                  "archive": archive.name, "tokens": tokens.name,
                  "archive_sha256": _file_hash(temporary_archive),
                  "tokens_sha256": _file_hash(temporary_tokens), "token_rows": expected_rows}
        _publish_exclusive(temporary_archive, archive)
        _publish_exclusive(temporary_tokens, tokens)
        _atomic_json_exclusive(committed, commit)
        _fsync_directory(directory)

    def _write_archive(self, destination: Path, entries):
        arrays, metadata = {}, []
        for i, (key, stats) in enumerate(sorted(entries)):
            prefix = f"r{i}"
            stats.native.to_dict(); stats.calibrated.to_dict()
            arrays[f"{prefix}_native_mean"] = stats.native.mean.cpu().numpy()
            arrays[f"{prefix}_native_m2"] = stats.native.m2.cpu().numpy()
            arrays[f"{prefix}_calibrated_mean"] = stats.calibrated.mean.cpu().numpy()
            arrays[f"{prefix}_calibrated_m2"] = stats.calibrated.m2.cpu().numpy()
            arrays[f"{prefix}_sums"] = stats.sums.cpu().numpy()
            arrays[f"{prefix}_positions"] = np.asarray(stats.positions, dtype=np.int64)
            metadata.append({"key": list(key), "prefix": prefix, "count": stats.count,
                             "mask_hash": stats.mask_hash, "driver_hash": stats.driver_hash,
                             "calibration_hash": stats.calibration_hash})
        contract = {"schema_version": self.SCHEMA_VERSION, "dtype": "float64", "ddof": 0,
                    "moment_representation": "count_mean_centered_m2",
                    "sum_fields": list(SUM_FIELDS), "position_bins": self.bins,
                    "numerical_protection": asdict(self.protection), "records": metadata}
        arrays["metadata_json"] = np.array(json.dumps(contract, sort_keys=True, allow_nan=False))
        # The descriptor uses exclusive creation; existing raw data is never
        # overwritten, including an accidental second attempt under a run ID.
        with destination.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())

    def save(self, directory: str | Path) -> Path:
        """Save one complete snapshot; prefer per-prompt writes during long runs."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for stats in self.records.values():
            stats.cpu()
        destination = directory / "sufficient_stats.npz"
        rows = [row for key in sorted(self.token_rows) for row in self.token_rows[key]]
        self._publish_archive(directory, "sufficient_stats", self.records.items(), rows)
        for key, reason in self.empty_records.items():
            self._write_empty_marker(directory, key, reason, self.empty_metadata.get(key))
        return destination

    @classmethod
    def load(cls, source: str | Path) -> "StateCollector":
        """Load one NPZ or merge a directory of per-prompt NPZ archives."""
        source = Path(source)
        is_directory = source.is_dir()
        markers = sorted(source.glob("*.empty.json")) if is_directory else (
            [source] if source.name.endswith(".empty.json") else [])
        paths = sorted(source.glob("*.npz")) if is_directory else ([] if markers else [source])
        if not paths and not markers:
            raise ValueError("no sufficient-statistic archives found")
        result = None
        incomplete = []
        for path in paths:
            commit_path = path.with_suffix(".complete.json")
            if not commit_path.exists():
                incomplete.append(str(path))
                continue
            with commit_path.open(encoding="utf-8") as handle:
                commit = json.load(handle)
            token_name = commit.get("tokens", "")
            token_path = path.parent / token_name
            if (commit.get("schema_version") != "state_collection.commit.v1" or
                commit.get("collection_schema_version") != cls.SCHEMA_VERSION or
                commit.get("archive") != path.name or Path(token_name).name != token_name or
                not token_path.is_file() or _file_hash(path) != commit.get("archive_sha256") or
                _file_hash(token_path) != commit.get("tokens_sha256")):
                raise ValueError(f"inconsistent committed diagnostic pair: {path}")
            with token_path.open("rb") as handle:
                token_rows = sum(block.count(b"\n") for block in iter(lambda: handle.read(1024 * 1024), b""))
            if token_rows != commit.get("token_rows"):
                raise ValueError("token row count does not match diagnostic commit")
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"]))
                if (metadata["schema_version"] != cls.SCHEMA_VERSION or
                    metadata["dtype"] != "float64" or metadata["ddof"] != 0 or
                    metadata["sum_fields"] != list(SUM_FIELDS)):
                    raise ValueError("unsupported collection numerical contract")
                if sum(int(entry["count"]) for entry in metadata["records"]
                       if entry["key"][3] == "all") != token_rows:
                    raise ValueError("token metrics do not cover archived all-position moments")
                protection = NumericalProtection(**metadata["numerical_protection"])
                if result is None:
                    result = cls(protection=protection, bins=metadata["position_bins"])
                elif result.protection != protection or result.bins != tuple(
                        tuple(x) for x in metadata["position_bins"]):
                    raise ValueError("archives use different numerical contracts")
                for entry in metadata["records"]:
                    pid, path_id, layer, group = entry["key"]
                    key = (pid, path_id, int(layer), group)
                    if key in result.records:
                        raise ValueError(f"duplicate sufficient-statistic record {key}")
                    stats = StateSufficientStats()
                    prefix = entry["prefix"]
                    for name, attr in (("native", stats.native), ("calibrated", stats.calibrated)):
                        attr.count = int(entry["count"])
                        attr.mean = torch.from_numpy(archive[f"{prefix}_{name}_mean"].copy())
                        attr.m2 = torch.from_numpy(archive[f"{prefix}_{name}_m2"].copy())
                        if attr.mean.dtype != torch.float64 or attr.m2.dtype != torch.float64:
                            raise ValueError("moment array dtype is not float64")
                        attr.to_dict()
                    stats.sums = torch.from_numpy(archive[f"{prefix}_sums"].copy())
                    stats.positions = [tuple(map(int, row)) for row in archive[f"{prefix}_positions"]]
                    if (stats.count != len(stats.positions) or stats.mask_hash != entry["mask_hash"] or
                        stats.sums.shape != (len(SUM_FIELDS),) or
                        not bool(torch.isfinite(stats.sums).all())):
                        raise ValueError("corrupted sufficient statistics or position mask")
                    stats.driver_hash, stats.calibration_hash = entry["driver_hash"], entry["calibration_hash"]
                    result.records[key] = stats
                    result.finished.add(key[:2])
        for path in markers:
            with path.open(encoding="utf-8") as handle:
                marker = json.load(handle)
            if (marker.get("schema_version") != "state_collection.empty.v1" or
                marker.get("collection_schema_version") != cls.SCHEMA_VERSION or
                marker.get("count") != 0 or marker.get("dtype") != "float64" or
                marker.get("ddof") != 0 or not marker.get("reason")):
                raise ValueError("invalid empty-observation marker")
            protection = NumericalProtection(**marker["numerical_protection"])
            bins = tuple(tuple(x) for x in marker["position_bins"])
            if result is None:
                result = cls(protection=protection, bins=bins)
            elif result.protection != protection or result.bins != bins:
                raise ValueError("empty marker uses a different numerical contract")
            key = (str(marker["prompt_id"]), str(marker["path"]))
            if key in result.finished:
                raise ValueError("duplicate or contradictory empty-observation marker")
            result.empty_records[key] = marker["reason"]
            result.empty_metadata[key] = json.loads(json.dumps(marker.get("metadata", {}), allow_nan=False))
            result.finished.add(key)
        if result is None:
            result = cls()
        result.incomplete_archives = incomplete
        return result

    def summarize(self, path: str, *, position_bin: str = "all", weighting: str = "token",
                  prompt_ids: list[str] | None = None) -> dict:
        pack = _prepare(self, path, position_bin, prompt_ids=prompt_ids)
        weights = np.ones((1, len(pack["prompts"])), dtype=np.float64)
        values, layers = _evaluate(pack, weights, weighting, self.protection, details=True)
        return {"path": path, "position_bin": position_bin, "weighting": weighting,
                "prompt_count": len(pack["prompts"]), "prompt_ids": pack["prompts"],
                "observed_prompt_count": len(pack["active_prompts"]),
                "observed_prompt_ids": pack["active_prompts"],
                "zero_contribution_prompt_ids": [p for p in pack["prompts"] if p not in pack["active_prompts"]],
                "layer_token_count": int(pack["counts"].sum()),
                "effective_position_count": int(pack["counts"][:, 0].sum()),
                "calibration_hashes": pack["calibration_hashes"],
                "driver_hashes": pack["driver_hashes"], "mask_hash": pack["mask_hash"],
                "numerical_protection": asdict(self.protection),
                "E_norm": _finite_or_none(values["E_norm"][0]),
                "E_moment": _finite_or_none(values["E_moment"][0]),
                "layers": layers,
                "status": "ok" if all(np.isfinite(v[0]) for v in values.values()) else
                    "undefined_zero_norm_ratio"}


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def _prepare(collector: StateCollector, path: str, group: str, prompt_ids=None):
    rows = {k: v for k, v in collector.records.items() if k[1] == path and k[3] == group}
    available = {k[0] for k in rows}
    universe = ({k[0] for k in collector.records if k[1] == path} |
                {pid for pid, marker_path in collector.empty_records if marker_path == path})
    prompts = sorted(universe) if prompt_ids is None else [str(x) for x in prompt_ids]
    if not prompts or len(set(prompts)) != len(prompts):
        raise ValueError("summary needs nonempty, unique prompt IDs")
    if not set(prompts) <= universe:
        raise ValueError("requested prompt is absent from path/group")
    active = [pid for pid in prompts if pid in available]
    if not active:
        raise ValueError("no valid decode observations in requested position group")
    layers = sorted({k[2] for k in rows if k[0] in prompts})
    # Never silently compare different prompt coverage across layers.
    if any((pid, path, layer, group) not in rows for pid in active for layer in layers):
        raise ValueError("incomplete prompt/layer diagnostic rectangle")
    for pid in active:
        positions = rows[(pid, path, layers[0], group)].positions
        if any(rows[(pid, path, layer, group)].positions != positions for layer in layers):
            raise ValueError("incomplete or inconsistent positions across observed layers")
    counts = np.zeros((len(prompts), len(layers)), dtype=np.float64)
    sums = np.zeros((len(prompts), len(layers), len(SUM_FIELDS)), dtype=np.float64)
    selected = [rows[(p, path, l, group)] for p in active for l in layers]
    if any(s.count <= 0 for s in selected):
        raise ValueError("zero-token diagnostic record")
    if any(s.native.mean.device.type != "cpu" for s in selected):
        raise ValueError("finish_prompt must move moments to CPU before analysis")
    for i, pid in enumerate(prompts):
        if pid not in available:
            continue
        for j, layer in enumerate(layers):
            stats = rows[(pid, path, layer, group)]
            counts[i, j] = stats.count
            sums[i, j] = stats.sums.numpy()
    empty_mask_hash = canonical_hash([])
    mask_entries = [[p, l, rows[(p, path, l, group)].mask_hash if p in available else empty_mask_hash]
                    for p in prompts for l in layers]
    return {"prompts": prompts, "layers": layers, "counts": counts, "rows": rows,
            "active_prompts": active,
            "active_indices": np.asarray([prompts.index(pid) for pid in active], dtype=np.int64),
            "path": path, "group": group, "sums": sums,
            "mask_hash": canonical_hash(mask_entries),
            "calibration_hashes": sorted({s.calibration_hash for s in selected}),
            "driver_hashes": sorted({s.driver_hash for s in selected})}


def _evaluate(pack, weights: np.ndarray, weighting: str, protection, *, details=False,
              include_layers=False):
    """Recompute nonlinear ratios and pooled moments for each prompt draw.

    Fixed centers keep the bootstrap moment calculation stable even if means
    are large and variance is tiny. Centered M2 plus between-prompt mean offsets
    supplies population variance; raw sumsq subtraction is never used.
    """
    if weighting not in ("token", "prompt"):
        raise ValueError("weighting must be 'token' or 'prompt'")
    batch = weights.shape[0]
    results = {"E_norm": np.zeros(batch), "E_moment": np.zeros(batch)}
    detail_rows = []
    for j, layer in enumerate(pack["layers"]):
        counts = pack["counts"][:, j]
        multiplier = weights if weighting == "token" else np.divide(
            weights, counts[None, :], out=np.zeros_like(weights), where=counts[None, :] > 0)
        total = multiplier @ counts
        scalar_sums = multiplier @ pack["sums"][:, j, :]
        nsum, zsum = scalar_sums[:, 0], scalar_sums[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = zsum / nsum
            signed_log = np.log(ratio)
        results["E_norm"] += np.abs(signed_log) / len(pack["layers"])
        moments = []
        cache = pack.setdefault("_moment_cache", {})
        if layer not in cache:
            entries = [pack["rows"][(p, pack["path"], layer, pack["group"])] for p in pack["active_prompts"]]
            active_counts = counts[pack["active_indices"]]
            cache[layer] = {}
            for name in ("native", "calibrated"):
                means = np.stack([getattr(s, name).mean.numpy() for s in entries])
                m2s = np.stack([getattr(s, name).m2.numpy() for s in entries])
                center = means[0].copy()
                offsets = means - center
                cache[layer][name] = (center, active_counts[:, None] * offsets,
                                      m2s + active_counts[:, None] * offsets ** 2)
        for name in ("native", "calibrated"):
            center, weighted_offsets, centered_second = cache[layer][name]
            active_multiplier = multiplier[:, pack["active_indices"]]
            with np.errstate(divide="ignore", invalid="ignore"):
                delta = (active_multiplier @ weighted_offsets) / total[:, None]
                second = active_multiplier @ centered_second / total[:, None]
            variance = np.maximum(second - delta ** 2, protection.eps_variance)
            moments.append((center[None, :] + delta, np.sqrt(variance)))
        (mu_y, sigma_y), (mu_z, sigma_z) = moments
        mean_error = np.sum((mu_z - mu_y) ** 2, axis=1)
        std_error = np.sum((sigma_z - sigma_y) ** 2, axis=1)
        denominator = np.sum(mu_y ** 2 + sigma_y ** 2, axis=1) + protection.eps_diagnostic_moment
        moment_error = np.sqrt((mean_error + std_error) / denominator)
        results["E_moment"] += moment_error / len(pack["layers"])
        if include_layers:
            results[f"layer:{layer}:signed_log_ratio"] = signed_log
            results[f"layer:{layer}:E_moment"] = moment_error
            results[f"layer:{layer}:moment_mean_error_component"] = np.sqrt(mean_error / denominator)
            results[f"layer:{layer}:moment_std_error_component"] = np.sqrt(std_error / denominator)
        if details:
            if batch != 1:
                raise ValueError("layer details require one point estimate")
            summary = {"layer": layer, "valid_positions": int(counts.sum()),
                       "prompt_count": len(pack["active_prompts"]),
                       "bootstrap_prompt_count": len(pack["prompts"]), "ratio": _finite_or_none(ratio[0]),
                       "signed_log_ratio": _finite_or_none(signed_log[0]),
                       "E_moment": float(moment_error[0]),
                       "moment_mean_error_component": float(np.sqrt(mean_error[0] / denominator[0])),
                       "moment_std_error_component": float(np.sqrt(std_error[0] / denominator[0]))}
            for i, field in enumerate(SUM_FIELDS):
                if field == "cos_defined":
                    summary["cos_defined_positions"] = int(pack["sums"][:, j, i].sum())
                elif field == "cos_u4_y8":
                    cos_den = scalar_sums[0, SUM_FIELDS.index("cos_defined")]
                    summary["mean_cos_u4_y8"] = float(scalar_sums[0, i] / cos_den) if cos_den > 0 else None
                else:
                    summary[f"mean_{field}"] = float(scalar_sums[0, i] / total[0])
            detail_rows.append(summary)
    return results, detail_rows


def assert_matched_masks(collector: StateCollector, left: str, right: str, *,
                         position_bin: str = "all") -> dict:
    """Reject missing prompts, layers, positions, or token IDs before pairing."""
    lp = _prepare(collector, left, position_bin)
    rp = _prepare(collector, right, position_bin)
    if lp["prompts"] != rp["prompts"] or lp["layers"] != rp["layers"] or lp["mask_hash"] != rp["mask_hash"]:
        raise ValueError("paired diagnostic paths do not have identical prompt/layer/token masks")
    if lp["calibration_hashes"] != rp["calibration_hashes"]:
        raise ValueError("state-source comparison must observe the same frozen calibration")
    return {"mask_hash": lp["mask_hash"], "prompt_ids": lp["prompts"], "layers": lp["layers"]}


def paired_state_bootstrap(collector: StateCollector, left: str, right: str, *,
                           position_bin: str = "all", weighting: str = "token",
                           iters: int = 10000, seed: int = 20261019,
                           alpha: float = 0.05, chunk_size: int = 128,
                           require_matched_masks: bool = True,
                           require_same_calibration: bool = True) -> dict:
    """Bootstrap E(left)-E(right), resampling independent prompts together.

    Each draw carries all layers and positions, then rebuilds layer ratios and
    moment distributions. It does not bootstrap precomputed token/layer errors.
    Frozen calibration parameters are never refitted; the resulting CI excludes
    calibration-data selection uncertainty. For updated own-trajectory models,
    differing masks/calibrations are allowed only via explicit flags; prompt and
    layer IDs must still match. That comparison is not a fixed-text state test.
    """
    if iters < 1 or chunk_size < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap configuration")
    lp, rp = _prepare(collector, left, position_bin), _prepare(collector, right, position_bin)
    if lp["prompts"] != rp["prompts"] or lp["layers"] != rp["layers"]:
        raise ValueError("paired bootstrap requires complete matching prompt and layer IDs")
    if require_matched_masks and lp["mask_hash"] != rp["mask_hash"]:
        raise ValueError("paired diagnostic paths have different position/token masks")
    if require_same_calibration and lp["calibration_hashes"] != rp["calibration_hashes"]:
        raise ValueError("comparison observes different calibrations; make update comparison explicit")
    n = len(lp["prompts"])
    point_l, _ = _evaluate(lp, np.ones((1, n)), weighting, collector.protection)
    point_r, _ = _evaluate(rp, np.ones((1, n)), weighting, collector.protection)
    rng = np.random.default_rng(seed)
    differences = {key: [] for key in point_l}
    for offset in range(0, iters, chunk_size):
        count = min(chunk_size, iters - offset)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=count).astype(np.float64)
        vl, _ = _evaluate(lp, weights, weighting, collector.protection)
        vr, _ = _evaluate(rp, weights, weighting, collector.protection)
        for key in differences:
            differences[key].append(vl[key] - vr[key])
    output = {}
    for key, chunks in differences.items():
        values = np.concatenate(chunks)
        point = float(point_l[key][0] - point_r[key][0])
        if not np.isfinite(values).all() or not math.isfinite(point):
            output[key] = {"difference": _finite_or_none(point), "ci_low": None, "ci_high": None,
                           "status": "undefined_bootstrap_draw", "finite_draws": int(np.isfinite(values).sum())}
        else:
            lo, hi = np.quantile(values, (alpha / 2, 1 - alpha / 2))
            output[key] = {"difference": point, "ci_low": float(lo), "ci_high": float(hi), "status": "ok"}
    return {"left": left, "right": right, "contrast": "left_minus_right",
            "position_bin": position_bin, "weighting": weighting,
            "bootstrap_unit": "prompt", "resamples": iters, "seed": seed,
            "confidence_level": 1 - alpha, "prompt_count": n,
            "matched_masks": lp["mask_hash"] == rp["mask_hash"],
            "left_mask_hash": lp["mask_hash"], "right_mask_hash": rp["mask_hash"],
            "left_calibration_hashes": lp["calibration_hashes"],
            "right_calibration_hashes": rp["calibration_hashes"],
            "calibration_parameters_refitted": False, "metrics": output}


def _bootstrap_intervals(point: Mapping[str, np.ndarray], draws, alpha: float) -> dict:
    output = {}
    for key, chunks in draws.items():
        values = np.concatenate(chunks)
        estimate = float(point[key][0])
        if not np.isfinite(values).all() or not math.isfinite(estimate):
            output[key] = {"estimate": _finite_or_none(estimate), "ci_low": None, "ci_high": None,
                           "status": "undefined_bootstrap_draw", "finite_draws": int(np.isfinite(values).sum())}
        else:
            lo, hi = np.quantile(values, (alpha / 2, 1 - alpha / 2))
            output[key] = {"estimate": estimate, "ci_low": float(lo), "ci_high": float(hi), "status": "ok"}
    return output


def state_bootstrap(collector: StateCollector, path: str, *, position_bin: str = "all",
                    weighting: str = "token", iters: int = 10000, seed: int = 20261019,
                    alpha: float = 0.05, chunk_size: int = 128) -> dict:
    """Prompt CIs for a path, including each layer's signed log norm ratio.

    Call for each predeclared position bin to report long-suffix coverage and
    uncertainty. A bin with no observations raises rather than implying that
    its unobserved suffix is well calibrated.
    """
    if iters < 1 or chunk_size < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap configuration")
    pack = _prepare(collector, path, position_bin)
    n = len(pack["prompts"])
    point, _ = _evaluate(pack, np.ones((1, n)), weighting, collector.protection, include_layers=True)
    draws = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for offset in range(0, iters, chunk_size):
        count = min(chunk_size, iters - offset)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=count).astype(np.float64)
        values, _ = _evaluate(pack, weights, weighting, collector.protection, include_layers=True)
        for key in draws:
            draws[key].append(values[key])
    return {"path": path, "position_bin": position_bin, "weighting": weighting,
            "bootstrap_unit": "prompt", "resamples": iters, "seed": seed,
            "confidence_level": 1 - alpha, "prompt_count": n,
            "observed_prompt_count": len(pack["active_prompts"]),
            "effective_position_count": int(pack["counts"][:, 0].sum()),
            "layer_token_count": int(pack["counts"].sum()), "mask_hash": pack["mask_hash"],
            "calibration_hashes": pack["calibration_hashes"],
            "calibration_parameters_refitted": False,
            "metrics": _bootstrap_intervals(point, draws, alpha)}


def common_mask_interaction_bootstrap(collector: StateCollector, *,
                                      a: str, b: str, c: str, d: str,
                                      position_bin: str = "common", weighting: str = "token",
                                      iters: int = 10000, seed: int = 20261019,
                                      alpha: float = 0.05, chunk_size: int = 128) -> dict:
    """Estimate (E(C)-E(D))-(E(B)-E(A)) on four-way common positions.

    A/B must share text and C/D must share text. Across those two text sources
    token IDs can differ, while prompt/layer/position coverage and weights must
    agree. The same prompt draw carries all four paths. This is a conditional
    shared-prefix analysis, not a substitute for each student's complete trace.
    """
    if iters < 1 or chunk_size < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap configuration")
    assert_matched_masks(collector, a, b, position_bin=position_bin)
    assert_matched_masks(collector, c, d, position_bin=position_bin)
    packs = [_prepare(collector, path, position_bin) for path in (a, b, c, d)]
    reference = packs[0]
    for pack in packs[1:]:
        if pack["prompts"] != reference["prompts"] or pack["layers"] != reference["layers"]:
            raise ValueError("interaction requires four-way matching prompt and layer IDs")
        if pack["calibration_hashes"] != reference["calibration_hashes"]:
            raise ValueError("interaction requires one fixed observed calibration")
        for pid in pack["prompts"]:
            for layer in pack["layers"]:
                left_row = reference["rows"].get((pid, reference["path"], layer, position_bin))
                right_row = pack["rows"].get((pid, pack["path"], layer, position_bin))
                left = left_row.positions if left_row is not None else []
                right = right_row.positions if right_row is not None else []
                if [i for i, _ in left] != [i for i, _ in right]:
                    raise ValueError("interaction requires four-way common decode positions")
    n = len(reference["prompts"])

    def contrast(weights):
        values = [_evaluate(pack, weights, weighting, collector.protection)[0] for pack in packs]
        return {key: values[2][key] - values[3][key] - values[1][key] + values[0][key]
                for key in values[0]}

    point = contrast(np.ones((1, n)))
    draws = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for offset in range(0, iters, chunk_size):
        count = min(chunk_size, iters - offset)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=count).astype(np.float64)
        values = contrast(weights)
        for key in draws:
            draws[key].append(values[key])
    return {"paths": {"A": a, "B": b, "C": c, "D": d},
            "contrast": "(C-D)-(B-A)", "position_bin": position_bin, "weighting": weighting,
            "bootstrap_unit": "prompt", "resamples": iters, "seed": seed,
            "confidence_level": 1 - alpha, "prompt_count": n,
            "four_way_common_positions": True,
            "common_position_count": int(reference["counts"][:, 0].sum()),
            "calibration_parameters_refitted": False,
            "metrics": _bootstrap_intervals(point, draws, alpha)}
