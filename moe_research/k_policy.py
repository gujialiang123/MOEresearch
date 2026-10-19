"""Cache-aware physical top-k policy with one shared MoE execution primitive.

Reference-k is independent of computed-k. New scalar/LDA calibration operates
AFTER native-dtype expert aggregation in FP32, then casts back. Historical weight
modes remain available, but prefill/native identity and unselected windows never
apply a non-unit calibration or invoke a shuffled provider.
"""
from __future__ import annotations

import contextvars
import types
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch

from .state_adapter import apply_post_calibration, route_and_aggregate, validate_block_layout

WEIGHT_MODES = (
    "renorm_survivors", "no_renorm", "fold_mass_to_top1", "calibrated_norm_match",
    "partial_renorm", "clipped_gain", "fixed_gain", "shuffled_gain",
    "decode_norm_match", "position_bin_gain", "reference_k",
)
_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar("moe_phase", default="prefill")
_MISSING = object()


def _position_bin(step: int) -> int:
    return 0 if step <= 32 else (1 if step <= 96 else 2)


@dataclass
class KPolicy:
    prefill_k: int = 8
    decode_k: int = 8
    weight_mode: str = "renorm_survivors"
    renorm_beta: float = 1.0
    gain_clip: Optional[float] = None
    layer_selector: Optional[Callable[[int], bool]] = None
    decode_step_selector: Optional[Callable[[int], bool]] = None
    calib_scalars: dict = field(default_factory=dict)
    gain_provider: Optional[Callable] = None
    # Explicit decode-only overrides. Reference mass never requires extra FFNs.
    k_compute: Optional[int] = None
    k_reference: Optional[int] = None
    # Native identity normally uses the checkpoint's original implementation.
    # P0 MUST also test this flag to actually exercise the shared executor at K8.
    force_native_execution: bool = False
    # fn(layer_idx, phase, decode_step, native_dtype_aggregated_output) -> Tensor
    output_transform: Optional[Callable] = None
    # fn(block, h, transformed_output, phase, layer_idx, decode_step, route_info)
    observer: Optional[Callable] = None

    def __post_init__(self):
        if self.weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode {self.weight_mode!r} not in {WEIGHT_MODES}")
        for name in ("prefill_k", "decode_k", "k_compute", "k_reference"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name}={value} must be a positive integer")
        if self.k_compute is not None and self.k_reference is not None \
                and self.k_reference < self.k_compute:
            raise ValueError("k_reference must be >= k_compute")
        if self.weight_mode == "clipped_gain" and (self.gain_clip is None or self.gain_clip < 1):
            raise ValueError("clipped_gain requires gain_clip >= 1")

    def k_for_phase(self, phase: str) -> int:
        if phase == "prefill":
            return self.prefill_k
        return self.decode_k if self.k_compute is None else self.k_compute


class PolicyContext:
    def __init__(self, policy: KPolicy, blocks, block_cls, orig_forward):
        self.policy = policy
        self.blocks = list(blocks)
        self.block_cls = block_cls
        self.orig_forward = orig_forward
        self.original_forwards = {id(b): b.forward for b in self.blocks}
        self.original_instance_forwards = {id(b): b.__dict__.get("forward", _MISSING) for b in self.blocks}
        self.hook_handle = None
        self.decode_step = 0

    def reset_counters(self):
        for i, block in enumerate(self.blocks):
            device = next(block.parameters()).device
            block._kp_ksum_prefill = torch.zeros((), device=device, dtype=torch.long)
            block._kp_ksum_decode = torch.zeros((), device=device, dtype=torch.long)
            block._kp_tok_prefill = 0
            block._kp_tok_decode = 0
            block._kp_layer_idx = i
        self.decode_step = 0

    def stats(self):
        kp = sum(b._kp_ksum_prefill.item() for b in self.blocks)
        kd = sum(b._kp_ksum_decode.item() for b in self.blocks)
        tp = sum(b._kp_tok_prefill for b in self.blocks)
        td = sum(b._kp_tok_decode for b in self.blocks)
        return {
            "avg_k_prefill": round(kp / tp, 4) if tp else None,
            "avg_k_decode": round(kd / td, 4) if td else None,
            "tok_prefill": tp, "tok_decode": td,
        }


def _phase_from_cache(kwargs) -> str:
    """Infer phase from actual cache state, never from sequence length."""
    past = kwargs.get("past_key_values")
    if past is None:
        position = kwargs.get("cache_position")
        if position is not None and position.numel():
            return "prefill" if int(position[0]) == 0 else "decode"
        return "prefill"
    if hasattr(past, "get_seq_length"):
        return "prefill" if past.get_seq_length() == 0 else "decode"
    if isinstance(past, (list, tuple)):
        if not past or past[0] is None:
            return "prefill"
        first_layer = past[0]
        if isinstance(first_layer, (list, tuple)) and first_layer:
            key = first_layer[0]
            if isinstance(key, torch.Tensor) and key.shape[-2] == 0:
                return "prefill"
        return "decode"
    raise TypeError(f"Unsupported cache type {type(past).__name__}")


def _make_moe_forward(ctx: PolicyContext):
    policy = ctx.policy

    def forward(self, hidden_states):
        phase = _PHASE.get()
        layer_idx = getattr(self, "_kp_layer_idx", -1)
        apply = policy.layer_selector is None or policy.layer_selector(layer_idx)
        if apply and phase == "decode" and policy.decode_step_selector is not None:
            apply = policy.decode_step_selector(ctx.decode_step)
        native_k = self.top_k
        requested = policy.k_for_phase(phase) if apply else native_k
        # Historical decode_k allowed k>E (capped). Explicit compute/reference
        # arguments are a reproducibility contract and must fail on invalid K.
        if phase == "decode" and apply and policy.k_compute is not None and requested > self.num_experts:
            raise ValueError("k_compute exceeds num_experts")
        kc = min(requested, self.num_experts)
        wm = policy.weight_mode
        explicit_reference = apply and phase == "decode" and policy.k_reference is not None
        if not apply:
            kr = native_k
        elif explicit_reference:
            kr = policy.k_reference
        elif wm == "renorm_survivors" or (wm == "partial_renorm" and policy.renorm_beta == 1.0):
            kr = kc
        else:
            kr = max(native_k, kc)
        identity = kc == native_k and kr == native_k
        # All frozen calibration and shuffled/random providers are decode-only
        # and only affect selected reduced-compute branches.
        calibrate = apply and phase == "decode" and kc < native_k
        legacy_transform = None
        if apply and not identity and not explicit_reference:
            def legacy_transform(weights, route_info):
                top = route_info["top_probabilities"]
                mass_native = top[:, :max(native_k, kc)].sum(-1, keepdim=True)
                inv_r = mass_native / top[:, :kc].sum(-1, keepdim=True)
                # norm_topk_prob=False historically retained the pool's raw mass
                # during survivor renormalization; preserve that legacy meaning.
                if wm == "renorm_survivors" and not self.norm_topk_prob:
                    return weights * inv_r
                if wm == "partial_renorm":
                    if policy.renorm_beta == 1.0 and self.norm_topk_prob:
                        return weights
                    return weights * inv_r.pow(float(policy.renorm_beta))
                if wm == "clipped_gain":
                    return weights * inv_r.clamp(max=float(policy.gain_clip))
                if wm == "shuffled_gain" and calibrate:
                    gain = inv_r if policy.gain_provider is None else policy.gain_provider(
                        layer_idx, phase, ctx.decode_step, weights.shape[0], weights.device
                    )
                    return weights * gain
                if wm == "fold_mass_to_top1":
                    dropped = top[:, kc:max(native_k, kc)].sum(-1, keepdim=True)
                    if self.norm_topk_prob:
                        dropped = dropped / mass_native
                    weights = weights.clone()
                    weights[:, :1] = weights[:, :1] + dropped
                return weights
        native = ctx.original_forwards.get(id(self))
        # CPU toys without a defined forward still exercise the shared path.
        can_call_native = getattr(ctx.orig_forward, "__name__", "") != "_forward_unimplemented"
        if identity and not policy.force_native_execution and can_call_native:
            result = native(hidden_states)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("Unsupported native MoE return format: expected (output, router_logits)")
            final, router_logits = result
            route_info = {"adapter": "original_forward", "k_compute": kc, "k_reference": kr}
        else:
            final, router_logits, route_info = route_and_aggregate(
                self, hidden_states, kc, kr, legacy_transform
            )
        if calibrate:
            if wm in ("calibrated_norm_match", "decode_norm_match", "fixed_gain", "position_bin_gain"):
                scalar = policy.calib_scalars.get(f"{layer_idx},{kc}", 1.0)
                if wm == "position_bin_gain":
                    scalar = policy.calib_scalars.get(f"{layer_idx},{_position_bin(ctx.decode_step)},{kc}", scalar)
                final = apply_post_calibration(final, scale=scalar)
            if policy.output_transform is not None:
                transformed = policy.output_transform(layer_idx, phase, ctx.decode_step, final)
                if transformed.shape != final.shape or transformed.dtype != final.dtype \
                        or transformed.device != final.device:
                    raise ValueError("output_transform must preserve MoE output shape, dtype and device")
                final = transformed
        token_count = hidden_states.shape[0] * hidden_states.shape[1]
        if phase == "prefill":
            self._kp_ksum_prefill = self._kp_ksum_prefill + token_count * kc
            self._kp_tok_prefill += token_count
        else:
            self._kp_ksum_decode = self._kp_ksum_decode + token_count * kc
            self._kp_tok_decode += token_count
        if policy.observer is not None:
            # Passing independent copies prevents observer-side in-place changes
            # from altering tensors actually propagated through the network.
            policy.observer(self, hidden_states.detach().clone(), final.detach().clone(),
                            phase, layer_idx, ctx.decode_step, route_info)
        return final, router_logits

    return forward


def _find_moe_blocks(model):
    from transformers.models.qwen3_moe import modeling_qwen3_moe as M
    blocks = [m for m in model.modules() if isinstance(m, M.Qwen3MoeSparseMoeBlock)]
    if not blocks:
        raise TypeError("No supported Qwen3MoeSparseMoeBlock modules found")
    return blocks, M.Qwen3MoeSparseMoeBlock


def attach_policy(model, policy: KPolicy) -> PolicyContext:
    blocks, block_cls = _find_moe_blocks(model)
    # Fail before changing any forwards; packed experts require another adapter.
    for block in blocks:
        validate_block_layout(block)
        if getattr(block, "_kp_attached", False):
            raise RuntimeError("A policy is already attached; detach it before changing arms")
    ctx = PolicyContext(policy, blocks, block_cls, block_cls.forward)
    ctx.reset_counters()
    fwd = _make_moe_forward(ctx)
    for block in blocks:
        block.forward = types.MethodType(fwd, block)
        block._kp_attached = True

    def pre_hook(module, args, kwargs):
        phase = _phase_from_cache(kwargs)
        _PHASE.set(phase)
        ctx.decode_step = ctx.decode_step + 1 if phase == "decode" else 0

    ctx.hook_handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    return ctx


def detach_policy(model, ctx: PolicyContext):
    for block in ctx.blocks:
        original = ctx.original_instance_forwards[id(block)]
        if original is _MISSING:
            block.__dict__.pop("forward", None)
        else:
            block.forward = original
        block.__dict__.pop("_kp_attached", None)
    if ctx.hook_handle is not None:
        ctx.hook_handle.remove()
        ctx.hook_handle = None
    _PHASE.set("prefill")
