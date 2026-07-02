"""harness/mfu.py — Model FLOPs Utilization computation.

Computes per-regime MFU (compute utilization) and MBU (memory bandwidth
utilization) from bench summary regime data + hardware config + model config.

Formulas
--------
For decode phase:
    FLOPs_per_token       = 2 × active_params_per_token
                            + 4 × num_attn_layers × avg_kv_seq_len × hidden_size
    Bytes_HBM_per_token   = active_params × dtype_bytes
                            (single-request KV read; grows with batch)
    tokens_per_s          = from bench summary (mean over runs)

    MFU_pct = 100 × (FLOPs_per_token × tokens_per_s) / peak_flops
    MBU_pct = 100 × (Bytes_HBM_per_token × tokens_per_s) / peak_hbm_bw
              (approximate; for batch B in flight, actual is / B factor)

Prefill phase FLOPs include quadratic attention:
    FLOPs_per_prompt_token = 2 × active_params_per_token
                             + 2 × num_attn_layers × prompt_len × hidden_size

For a mixed prefill+decode regime, we approximate MFU per phase.

Design notes
------------
- MFU is a *cross-config comparability* metric. Absolute value depends on
  which params we count. Consistency across trials is what matters.
- We use a *simple* MFU (weight-matmul-only) as primary, and expose
  attention-included MFU as a secondary field for long-prefill regimes.
- For MoE top-k of E, only k experts execute; active_params_per_token
  already accounts for this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    """Peak FLOPS and HBM bandwidth for one accelerator."""
    name: str
    peak_flops_by_dtype: dict[str, float]
    peak_hbm_bandwidth_bytes_per_s: float

    @classmethod
    def load(cls, path: str | Path) -> "HardwareConfig":
        d = yaml.safe_load(Path(path).read_text())
        return cls(
            name=d["name"],
            peak_flops_by_dtype=d["peak_flops"],
            peak_hbm_bandwidth_bytes_per_s=float(d["peak_hbm_bandwidth_bytes_per_s"]),
        )

    def peak_flops(self, dtype: str) -> float:
        # Common aliases from HF configs → our canonical keys.
        alias = {
            "bfloat16": "bf16",
            "float16": "fp16", "half": "fp16",
            "float32": "fp32", "float": "fp32",
            "float64": "fp64", "double": "fp64",
            "float8_e4m3": "fp8", "float8_e5m2": "fp8",
        }
        canonical = alias.get(dtype, dtype)
        if canonical not in self.peak_flops_by_dtype:
            raise ValueError(
                f"unknown dtype {dtype!r} (canonical {canonical!r}) "
                f"for hardware {self.name}. "
                f"available: {list(self.peak_flops_by_dtype)}"
            )
        return float(self.peak_flops_by_dtype[canonical])


@dataclass
class ModelConfig:
    """Model shape + derived FLOPs/bytes constants."""
    name: str
    hidden_size: int
    num_attention_layers: int
    active_params_per_token: float
    total_params: float
    dtype: str
    dtype_bytes: int
    max_position_embeddings: int
    # For prefill attention overhead:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        d = yaml.safe_load(Path(path).read_text())
        return cls(
            name=d["name"],
            hidden_size=int(d["hidden_size"]),
            num_attention_layers=int(d["num_attention_layers"]),
            active_params_per_token=float(d["active_params_per_token"]),
            total_params=float(d["total_params"]),
            dtype=d["dtype"],
            dtype_bytes=int(d["dtype_bytes"]),
            max_position_embeddings=int(d["max_position_embeddings"]),
            num_heads=int(d["num_heads"]),
            num_kv_heads=int(d["num_kv_heads"]),
            head_dim=int(d["head_dim"]),
            vocab_size=int(d["vocab_size"]),
        )


# ---------------------------------------------------------------------------
# FLOPs / bytes formulas
# ---------------------------------------------------------------------------

def matmul_flops_per_token(model: ModelConfig) -> float:
    """FLOPs from all weight-matmuls (attention proj, conv, FFN, MoE, LM head)
    for a single token, ignoring the score/context attention term."""
    return 2.0 * model.active_params_per_token


def attention_score_context_flops(
    model: ModelConfig,
    *,
    query_tokens: int,
    kv_len: int,
) -> float:
    """FLOPs for attention Q·K^T + softmax(·)·V, summed over all layers.

    query_tokens: how many query tokens are processed together (prefill=N,
        decode=1 per step).
    kv_len: length of the KV sequence being attended to.
    """
    per_layer = 2.0 * query_tokens * kv_len * model.hidden_size * 2  # score + context
    return per_layer * model.num_attention_layers


def decode_step_flops(model: ModelConfig, *, kv_len: int) -> float:
    """Total FLOPs for producing one decoded token when KV cache is kv_len long."""
    return matmul_flops_per_token(model) + attention_score_context_flops(
        model, query_tokens=1, kv_len=kv_len
    )


def prefill_flops(model: ModelConfig, *, prompt_len: int) -> float:
    """Total FLOPs to prefill a prompt of prompt_len tokens.
    Attention score/context grows as O(prompt_len^2)."""
    matmul = matmul_flops_per_token(model) * prompt_len
    attn = attention_score_context_flops(
        model, query_tokens=prompt_len, kv_len=prompt_len
    ) / 2.0  # causal mask halves the work
    return matmul + attn


def bytes_per_token_decode(model: ModelConfig) -> float:
    """Weight bytes read from HBM per decoded token (single request)."""
    return model.active_params_per_token * model.dtype_bytes


# ---------------------------------------------------------------------------
# Regime → MFU
# ---------------------------------------------------------------------------

def _words_to_tokens(words: int) -> int:
    """~1.3 tokens/word for English (rough)."""
    return int(round(words * 1.3))


def compute_regime_mfu(
    regime_entry: dict[str, Any],
    *,
    model: ModelConfig,
    hardware: HardwareConfig,
    kv_dtype_bytes: int | None = None,
) -> dict[str, Any]:
    """Given one regime's summary dict (must have tokens_per_s.mean,
    prompt_words, max_new, concurrency), return a dict of MFU/MBU metrics."""
    tokens_per_s = float(regime_entry.get("tokens_per_s", {}).get("mean", 0.0))
    if tokens_per_s <= 0:
        return {
            "mfu_pct": 0.0,
            "mbu_pct": 0.0,
            "note": "tokens_per_s <= 0; cannot compute MFU",
        }

    prompt_len_tokens = _words_to_tokens(int(regime_entry.get("prompt_words", 0)))
    max_new = int(regime_entry.get("max_new", 0))
    concurrency = max(1, int(regime_entry.get("concurrency", 1)))

    # Simple MFU (weight matmul only, no attention score/context).
    # Good approximation for decode with modest seq length.
    simple_flops_per_token = matmul_flops_per_token(model)
    peak = hardware.peak_flops(model.dtype)
    mfu_simple_pct = 100.0 * simple_flops_per_token * tokens_per_s / peak

    # Full MFU: include attention score/context using average KV length
    # during decode (=prompt_len + max_new/2 as average).
    avg_kv = prompt_len_tokens + max_new // 2
    full_flops_per_token = decode_step_flops(model, kv_len=avg_kv)
    mfu_full_pct = 100.0 * full_flops_per_token * tokens_per_s / peak

    # Amortized-per-request MFU counting prefill work too (since request
    # generates max_new decode tokens after processing prompt_len prefill).
    # This "effective" MFU tells us how well the full request lifecycle
    # utilizes the GPU.
    prefill_j = prefill_flops(model, prompt_len=prompt_len_tokens)
    decode_j = full_flops_per_token * max_new
    total_j_per_request = prefill_j + decode_j
    req_per_s = float(regime_entry.get("req_per_s", {}).get("mean", 0.0))
    mfu_amortized_pct = (
        100.0 * total_j_per_request * req_per_s / peak if req_per_s > 0 else 0.0
    )

    # MBU (memory bandwidth utilization).
    # For decode: weights fetched from HBM once per forward pass, shared
    # across the batch. So effective bytes/sec = weight_bytes * fwd_passes/s.
    # forward_passes/s ≈ tokens_per_s / concurrency (each fwd pass produces
    # one token per in-flight request).
    fwd_passes_per_s = tokens_per_s / concurrency
    bytes_per_s = bytes_per_token_decode(model) * fwd_passes_per_s
    mbu_pct = 100.0 * bytes_per_s / hardware.peak_hbm_bandwidth_bytes_per_s

    return {
        "mfu_pct_simple": round(mfu_simple_pct, 3),
        "mfu_pct_full_decode": round(mfu_full_pct, 3),
        "mfu_pct_amortized": round(mfu_amortized_pct, 3),
        "mbu_pct": round(mbu_pct, 3),
        "assumed_kv_len": avg_kv,
        "peak_flops_used": peak,
        "peak_hbm_bw_used": hardware.peak_hbm_bandwidth_bytes_per_s,
        "model_dtype": model.dtype,
    }


def annotate_summary_with_mfu(
    summary: dict[str, Any],
    *,
    model: ModelConfig,
    hardware: HardwareConfig,
) -> dict[str, Any]:
    """Mutates `summary` in-place: adds `mfu` dict to each regime entry.
    Returns the summary for chaining."""
    for regime_id, regime_entry in summary.get("regimes", {}).items():
        try:
            mfu = compute_regime_mfu(
                regime_entry, model=model, hardware=hardware,
            )
            regime_entry["mfu"] = mfu
        except Exception as e:
            regime_entry["mfu"] = {"error": str(e)}
    # Also record the assumptions at the top level for reproducibility.
    summary["mfu_assumptions"] = {
        "model_config": model.name,
        "active_params_per_token": model.active_params_per_token,
        "dtype": model.dtype,
        "hardware": hardware.name,
        "peak_flops": hardware.peak_flops(model.dtype),
        "peak_hbm_bw_bytes_per_s": hardware.peak_hbm_bandwidth_bytes_per_s,
    }
    return summary


# ---------------------------------------------------------------------------
# Standalone CLI: retro-compute MFU on existing summary.json
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Add MFU fields to bench summary.json")
    ap.add_argument("--summary", required=True, help="path to summary.json")
    ap.add_argument("--hardware", required=True, help="path to hardware yaml")
    ap.add_argument("--model", required=True, help="path to model yaml")
    ap.add_argument("--in-place", action="store_true",
                    help="rewrite summary in place (default: print to stdout)")
    args = ap.parse_args()

    hw = HardwareConfig.load(args.hardware)
    mdl = ModelConfig.load(args.model)
    summary = json.loads(Path(args.summary).read_text())
    annotate_summary_with_mfu(summary, model=mdl, hardware=hw)

    if args.in_place:
        Path(args.summary).write_text(json.dumps(summary, indent=2))
        print(f"[mfu] annotated in place: {args.summary}")
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
