"""Decode-only state calibration with an explicit floating-point contract.

Branches must already have been aggregated in the model dtype by the same
primitive used for deployment. Scalar fits use the native-reference reduced
branch ``u4``; LDA fits use the full-renormalized reduced branch ``y4_full``.
The caller decides which driver propagates to the next layer. This module never
runs a model or substitutes the native branch for a student driver's output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def canonical_hash(value: Any) -> str:
    """Hash finite, JSON-serializable scientific parameters without rounding."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class NumericalProtection:
    """Different units and roles have independently recorded constants.

    ``eps_variance`` is a floor on population variance (squared activation
    units), not the epsilon added to the LDA standard deviation. Its default of
    zero only clamps negative roundoff. The diagnostic epsilons have norm and
    squared-norm units respectively. No fit/validation-dependent tuning occurs.
    """
    eps_lda_std: float = 1e-6
    eps_variance: float = 0.0
    eps_diagnostic_norm: float = 1e-12
    eps_diagnostic_moment: float = 1e-24

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not (0 <= value < float("inf")):
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.eps_lda_std <= 0:
            raise ValueError("eps_lda_std must be positive for zero-variance dimensions")
        if self.eps_diagnostic_norm <= 0 or self.eps_diagnostic_moment <= 0:
            raise ValueError("diagnostic epsilons must be positive")


class StreamingMoments:
    """Mergeable FP64 Chan/Welford population moments on the input device.

    Count is an integer; mean and centered sum of squares stay FP64. Even when
    called once per decode token this estimates one distribution over *all*
    fitted tokens. It never averages per-prompt standard deviations. CPU copies
    are explicit, normally made only when a prompt has finished.
    """
    def __init__(self):
        self.count = 0
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> "StreamingMoments":
        if not values.is_floating_point() or values.ndim < 1:
            raise ValueError("moments require floating [..., hidden] values")
        x = values.detach().reshape(-1, values.shape[-1]).to(dtype=torch.float64)
        if not x.shape[0]:
            return self
        # Non-finite checks are deliberately deferred to serialization/finalize
        # to avoid synchronizing CUDA on every observer callback.
        other = StreamingMoments()
        other.count = x.shape[0]
        other.mean = x.mean(dim=0)
        other.m2 = ((x - other.mean) ** 2).sum(dim=0)
        return self.merge(other)

    def merge(self, other: "StreamingMoments", weight: float = 1.0) -> "StreamingMoments":
        if weight != int(weight) or weight < 0:
            raise ValueError("streaming merge weight must be a nonnegative integer")
        if not other.count or not weight:
            return self
        n = other.count * int(weight)
        if not self.count:
            self.count = n
            self.mean = other.mean.detach().clone()
            self.m2 = other.m2.detach().clone() * int(weight)
            return self
        if self.mean.shape != other.mean.shape:
            raise ValueError("hidden dimension changed between moment updates")
        mean = other.mean.to(device=self.mean.device, dtype=torch.float64)
        m2 = other.m2.to(device=self.mean.device, dtype=torch.float64)
        total = self.count + n
        delta = mean - self.mean
        self.m2 = self.m2 + m2 * int(weight) + delta.square() * (self.count * n / total)
        self.mean = self.mean + delta * (n / total)
        self.count = total
        return self

    def variance(self, eps_variance: float = 0.0) -> torch.Tensor:
        if not self.count:
            raise ValueError("cannot estimate variance from zero tokens")
        return (self.m2 / self.count).clamp_min(eps_variance)

    def std(self, eps_variance: float = 0.0) -> torch.Tensor:
        return self.variance(eps_variance).sqrt()

    def cpu(self) -> "StreamingMoments":
        result = StreamingMoments()
        result.count = self.count
        if self.count:
            result.mean = self.mean.detach().cpu().clone()
            result.m2 = self.m2.detach().cpu().clone()
        return result

    def to_dict(self) -> dict:
        if not self.count:
            return {"count": 0, "mean": None, "m2": None, "dtype": "float64", "ddof": 0}
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.m2).all()):
            raise ValueError("non-finite streaming moments")
        return {"count": self.count, "mean": self.mean.cpu().tolist(),
                "m2": self.m2.cpu().tolist(), "dtype": "float64", "ddof": 0}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StreamingMoments":
        if data.get("ddof", 0) != 0:
            raise ValueError("only population moments (ddof=0) are supported")
        result = cls()
        result.count = int(data["count"])
        if result.count < 0:
            raise ValueError("negative count")
        if result.count:
            result.mean = torch.tensor(data["mean"], dtype=torch.float64)
            result.m2 = torch.tensor(data["m2"], dtype=torch.float64)
            if result.mean.ndim != 1 or result.mean.shape != result.m2.shape:
                raise ValueError("invalid moment dimensions")
            result.to_dict()  # reject corrupted/non-finite archives
        return result


class FrozenStateCalibration:
    """Frozen, hash-addressed scalar or per-dimension LDA parameters.

    ``apply`` is decode-only and raises on unknown decode layers. LDA applies
    ``(x.float()-mu4.float()) * (sigma8.float()/(sigma4.float()+eps)) + mu8.float()``
    and then casts back to x.dtype. Scalar similarly multiplies in FP32 after
    aggregation. Using this very method for both driver and observer avoids an
    accidental BF16 pre-aggregation gain or a different affine rounding order.
    """
    SCHEMA_VERSION = "state_calibration.v1"

    def __init__(self, family: str, parameters: Mapping[str, Mapping], *,
                 protection: NumericalProtection | None = None,
                 metadata: Mapping[str, Any] | None = None):
        if family not in ("scalar", "lda"):
            raise ValueError("family must be 'scalar' or 'lda'")
        self.family = family
        # Copy through canonical JSON so mutable caller dictionaries cannot
        # silently change the meaning of a frozen calibration hash.
        self._parameters_json = json.dumps(parameters, sort_keys=True, allow_nan=False)
        self._parameters = json.loads(self._parameters_json)
        self.protection = protection or NumericalProtection()
        self._metadata_json = json.dumps(dict(metadata or {}), sort_keys=True, allow_nan=False)
        self._cache: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        self._hash: str | None = None
        if not self._parameters:
            raise ValueError("calibration requires at least one fitted layer")
        for layer, params in self._parameters.items():
            if str(int(layer)) != str(layer) or int(layer) < 0:
                raise ValueError(f"invalid layer identifier {layer!r}")
            if family == "scalar":
                value = params["scale"]
                if not isinstance(value, (int, float)) or not (0 <= value < float("inf")):
                    raise ValueError("scalar gain must be finite and nonnegative")
            else:
                arrays = [torch.tensor(params[key], dtype=torch.float64) for key in
                          ("mean_reduced", "mean_native", "std_reduced", "std_native")]
                if any(x.ndim != 1 or x.numel() == 0 for x in arrays):
                    raise ValueError("LDA parameters must be nonempty vectors")
                if any(x.shape != arrays[0].shape for x in arrays):
                    raise ValueError("LDA parameter dimensions disagree")
                if any(not bool(torch.isfinite(x).all()) for x in arrays):
                    raise ValueError("non-finite LDA parameters")
                if bool((arrays[2] < 0).any() or (arrays[3] < 0).any()):
                    raise ValueError("standard deviations cannot be negative")

    @property
    def parameters(self) -> dict:
        return json.loads(self._parameters_json)

    @property
    def metadata(self) -> dict:
        return json.loads(self._metadata_json)

    @property
    def hash(self) -> str:
        if self._hash is None:
            self._hash = canonical_hash(self.to_dict(include_hash=False))
        return self._hash

    @property
    def calibration_hash(self) -> str:
        return self.hash

    def apply(self, layer: int, reduced: torch.Tensor, phase: str = "decode") -> torch.Tensor:
        if phase not in ("prefill", "decode"):
            raise ValueError(f"unknown execution phase {phase!r}")
        if phase == "prefill":
            return reduced
        if not reduced.is_floating_point():
            raise ValueError("calibration requires a floating model branch")
        key = (int(layer), str(reduced.device))
        params = self._cache.get(key)
        if params is None:
            raw = self._parameters.get(str(int(layer)))
            if raw is None:
                raise KeyError(f"missing calibration for decode layer {layer}")
            names = ("scale",) if self.family == "scalar" else (
                "mean_reduced", "mean_native", "std_reduced", "std_native")
            params = {name: torch.tensor(raw[name], device=reduced.device, dtype=torch.float32)
                      for name in names}
            self._cache[key] = params
        x = reduced.float()
        if self.family == "scalar":
            out = x * params["scale"]
        else:
            if x.shape[-1] != params["mean_reduced"].numel():
                raise ValueError("branch hidden dimension does not match LDA parameters")
            gain = params["std_native"] / (params["std_reduced"] + self.protection.eps_lda_std)
            out = (x - params["mean_reduced"]) * gain + params["mean_native"]
        return out.to(dtype=reduced.dtype)

    def output_transform(self, layer: int, phase: str, step: int,
                         branch: torch.Tensor) -> torch.Tensor:
        """Adapter for KPolicy.output_transform; phase gates nontrivial bias too."""
        return self.apply(layer, branch, phase)

    def to_dict(self, *, include_hash: bool = True) -> dict:
        result = {
            "schema_version": self.SCHEMA_VERSION, "family": self.family,
            "reduced_branch": "u4_native_reference" if self.family == "scalar" else "y4_full_renorm",
            "application_phase": "decode", "statistics_dtype": "float64", "ddof": 0,
            "application_dtype": "float32_then_model_dtype",
            "operation_order": "x_fp32*scale_fp32" if self.family == "scalar" else
                "(x_fp32-mu4_fp32)*(std8_fp32/(std4_fp32+eps_lda_std))+mu8_fp32",
            "numerical_protection": asdict(self.protection),
            "parameters": self.parameters, "metadata": self.metadata,
        }
        if include_hash:
            result["calibration_hash"] = self.hash
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrozenStateCalibration":
        if data.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported state calibration schema")
        result = cls(data["family"], data["parameters"],
                     protection=NumericalProtection(**data["numerical_protection"]),
                     metadata=data.get("metadata"))
        # Validate all contract fields, not only coefficients.
        if dict(data) != result.to_dict(include_hash="calibration_hash" in data):
            raise ValueError("calibration hash or numerical contract does not match payload")
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False, allow_nan=False)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenStateCalibration":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class CalibrationFitter:
    """Streaming same-state fits. The caller supplies driver and source metadata.

    ``update`` must only receive valid decode positions from C_fit. Fitting does
    not inspect labels, output length or diagnostic-set mismatch. An empty
    calibration and a zero reduced scalar denominator fail explicitly.
    """
    def __init__(self, family: str, *, protection: NumericalProtection | None = None):
        if family not in ("scalar", "lda"):
            raise ValueError("family must be 'scalar' or 'lda'")
        self.family = family
        self.protection = protection or NumericalProtection()
        self.layers: dict[int, dict] = {}
        self.prompt_counts: dict[str, dict[str, int]] = {}

    def update(self, layer: int, y8: torch.Tensor, u4: torch.Tensor | None = None,
               y4_full: torch.Tensor | None = None, *, prompt_id: Any = None,
               phase: str = "decode") -> None:
        if phase != "decode":
            raise ValueError("prefill positions must not enter decode calibration")
        if int(layer) < 0:
            raise ValueError("layer must be nonnegative")
        reduced = u4 if self.family == "scalar" else y4_full
        if reduced is None or reduced.shape != y8.shape or reduced.device != y8.device:
            raise ValueError("fit requires matching same-device native and reduced branches")
        if y8.ndim < 1 or y8.shape[-1] == 0:
            raise ValueError("invalid branch shape")
        entry = self.layers.setdefault(int(layer), {
            "native": StreamingMoments(), "reduced": StreamingMoments(),
            "sum_native_norm": torch.zeros((), dtype=torch.float64, device=y8.device),
            "sum_reduced_norm": torch.zeros((), dtype=torch.float64, device=y8.device),
        })
        entry["native"].update(y8)
        entry["reduced"].update(reduced)
        entry["sum_native_norm"] += y8.detach().double().reshape(-1, y8.shape[-1]).norm(dim=-1).sum()
        entry["sum_reduced_norm"] += reduced.detach().double().reshape(-1, reduced.shape[-1]).norm(dim=-1).sum()
        count = y8.numel() // y8.shape[-1]
        if prompt_id is not None:
            counts = self.prompt_counts.setdefault(str(prompt_id), {})
            counts[str(int(layer))] = counts.get(str(int(layer)), 0) + count

    def finalize(self, source_metadata: Mapping[str, Any] | None = None) -> FrozenStateCalibration:
        if not self.layers:
            raise ValueError("no valid calibration tokens were observed")
        parameters, counts, diagnostics = {}, {}, {}
        for layer, entry in sorted(self.layers.items()):
            native, reduced = entry["native"], entry["reduced"]
            native.to_dict(); reduced.to_dict()  # finite validation at the fit boundary
            if native.count != reduced.count or not native.count:
                raise ValueError("native/reduced calibration counts do not match")
            nsum = float(entry["sum_native_norm"].item())
            rsum = float(entry["sum_reduced_norm"].item())
            if not (0 <= nsum < float("inf") and 0 <= rsum < float("inf")):
                raise ValueError("non-finite calibration norm sums")
            if self.family == "scalar":
                if rsum <= 0:
                    raise ValueError(f"zero reduced norm denominator at layer {layer}")
                parameters[str(layer)] = {"scale": nsum / rsum}
            else:
                native_std = native.std(self.protection.eps_variance)
                reduced_std = reduced.std(self.protection.eps_variance)
                parameters[str(layer)] = {
                    "mean_reduced": reduced.mean.cpu().tolist(),
                    "mean_native": native.mean.cpu().tolist(),
                    "std_reduced": reduced_std.cpu().tolist(),
                    "std_native": native_std.cpu().tolist(),
                }
                diagnostics[str(layer)] = {
                    "reduced_dimensions_at_or_below_std_epsilon": int(
                        (reduced_std <= self.protection.eps_lda_std).sum().item()),
                    "native_zero_variance_dimensions": int((native_std == 0).sum().item()),
                    "reduced_zero_variance_dimensions": int((reduced_std == 0).sum().item()),
                }
            counts[str(layer)] = native.count
            diagnostics.setdefault(str(layer), {}).update(
                sum_native_norm=nsum, sum_reduced_norm=rsum)
        metadata = dict(source_metadata or {})
        for reserved in ("counts_per_layer", "counts_per_prompt_layer", "fit_diagnostics"):
            if reserved in metadata:
                raise ValueError(f"source metadata shadows reserved field {reserved}")
        metadata.update(counts_per_layer=counts, counts_per_prompt_layer=self.prompt_counts,
                        fit_diagnostics=diagnostics)
        return FrozenStateCalibration(self.family, parameters, protection=self.protection,
                                      metadata=metadata)
