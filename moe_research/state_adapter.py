"""Verified ModuleList MoE execution shared by deployed and shadow branches.

Numerical protocol: router softmax and normalization in FP32. Expert selection
uses the native top-k pool (or a larger explicitly computed k), independently of
reference-k. Larger reference masses use a separate router-only top-k. Cast FINAL
weights to the hidden dtype; expert output multiplication and ascending-expert
``index_add_`` accumulation in that dtype. Scalar/affine output calibration is
separate: aggregate first, convert to FP32, apply calibration, cast back.

Only the Qwen ModuleList layout is supported. Packed/3-D experts must have their
own verified adapter; they are never interpreted as a ModuleList here.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_NAME = "qwen-modulelist-fixed-native-selection-fp32-reference-native-dtype-aggregate-v2"


def validate_block_layout(block):
    if not isinstance(getattr(block, "experts", None), nn.ModuleList):
        raise TypeError(
            f"Unsupported MoE expert layout {type(getattr(block, 'experts', None)).__name__}; "
            "the verified adapter requires torch.nn.ModuleList. Packed/3-D experts "
            "require a separately verified adapter (use the locked Transformers version)."
        )
    for attr in ("num_experts", "top_k", "norm_topk_prob", "gate"):
        if not hasattr(block, attr):
            raise TypeError(f"Unsupported MoE block: missing {attr}")
    if len(block.experts) != block.num_experts:
        raise ValueError("MoE num_experts does not match ModuleList length")


def apply_post_calibration(output, scale=1.0, bias=0.0):
    """Affine transform AFTER native-dtype aggregation, with FP32 arithmetic."""
    scale = torch.as_tensor(scale, device=output.device, dtype=torch.float32)
    bias = torch.as_tensor(bias, device=output.device, dtype=torch.float32)
    return (output.float() * scale + bias).to(output.dtype)


def execute_branches(block, hidden_states, branches: Mapping, weight_transforms=None):
    """Evaluate ``name: (k_compute, k_reference)`` branches on identical h.

    Reference mass only evaluates the router. FFNs are called for the union of
    computed experts, never for reference-only experts. In diagnostic shadows the
    union includes native top-8; this is explicitly additional observer overhead.
    ``weight_transforms`` is an internal compatibility hook for historical modes.
    """
    validate_block_layout(block)
    if hidden_states.ndim != 3:
        raise ValueError("MoE hidden_states must have shape [batch, sequence, hidden]")
    n_experts = block.num_experts
    for name, (kc, kr) in branches.items():
        if not 1 <= kc <= n_experts or not 1 <= kr <= n_experts:
            raise ValueError(f"{name}: k_compute={kc}, k_reference={kr}; experts={n_experts}")
        if kr < kc:
            raise ValueError(f"{name}: k_reference must be >= k_compute")
    hdim = hidden_states.shape[-1]
    hs = hidden_states.reshape(-1, hdim)
    logits = block.gate(hs)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise TypeError("Unsupported gate return format: expected [tokens, experts] Tensor")
    probabilities = F.softmax(logits, dim=1, dtype=torch.float)
    if probabilities.shape[-1] != n_experts:
        raise ValueError("Gate expert dimension does not match num_experts")
    compute_max = max(spec[0] for spec in branches.values())
    # Changing the reference must NEVER change which experts execute. In BF16,
    # tied logits can cause topk(8) and topk(16) to return different rank orders.
    # Keep the native top-k operation for every reduced branch; any larger
    # reference gets an independent router-only probability-mass calculation.
    selection_pool_k = max(block.top_k, compute_max)
    top_prob, selected = torch.topk(probabilities, selection_pool_k, dim=-1)
    reference_prob = {
        kr: (top_prob[:, :kr] if kr <= selection_pool_k else
             torch.topk(probabilities, kr, dim=-1).values)
        for kr in {spec[1] for spec in branches.values()}
    }
    reference_mass = {kr: values.sum(dim=-1, keepdim=True)
                      for kr, values in reference_prob.items()}
    selected_compute = selected[:, :compute_max]
    weights, info = {}, {}
    for name, (kc, kr) in branches.items():
        if block.norm_topk_prob:
            w = top_prob[:, :kc] / reference_mass[kr]
        else:
            # Preserve the checkpoint's native unnormalized semantics. The new
            # reference-k scientific protocol requires norm_topk_prob=True.
            w = top_prob[:, :kc]
        ri = {
            "adapter": ADAPTER_NAME, "k_compute": kc, "k_reference": kr,
            "selected": selected[:, :kc], "top_probabilities": top_prob,
            "probabilities": probabilities, "selection_pool_k": selection_pool_k,
            "reference_mass": reference_mass[kr],
            "reference_top_probabilities": reference_prob[kr],
        }
        if weight_transforms and name in weight_transforms:
            w = weight_transforms[name](w, ri)
        weights[name] = w.to(hs.dtype)
        info[name] = dict(ri, weights=weights[name])
    outputs = {name: torch.zeros_like(hs) for name in branches}
    mask = F.one_hot(selected_compute, num_classes=n_experts).permute(2, 1, 0)
    # This is the native ascending-expert order. Only ranks below compute_max are
    # in this mask, so reference-only experts are physically skipped.
    expert_hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        rank, token = torch.where(mask[expert_idx].squeeze(0))
        expert_output = block.experts[expert_idx](hs[None, token].reshape(-1, hdim))
        for name, (kc, _) in branches.items():
            kept = rank < kc
            ix, tx = rank[kept], token[kept]
            contribution = expert_output[kept] * weights[name][tx, ix, None]
            outputs[name].index_add_(0, tx, contribution.to(hs.dtype))
    return {name: value.reshape_as(hidden_states) for name, value in outputs.items()}, logits, info


def route_and_aggregate(block, hidden_states, k_compute, k_reference, weight_transform=None):
    transforms = {"driver": weight_transform} if weight_transform else None
    outputs, logits, info = execute_branches(
        block, hidden_states, {"driver": (k_compute, k_reference)}, transforms
    )
    return outputs["driver"], logits, info["driver"]


def shadow_branches(block, hidden_states, k_compute=4, k_native=None):
    """Same-hidden native, reduced native-reference and full-renorm outputs."""
    k_native = block.top_k if k_native is None else k_native
    if k_compute > k_native:
        raise ValueError("Shadow reduced k_compute must not exceed native top-k")
    outputs, _, _ = execute_branches(block, hidden_states, {
        "y8": (k_native, k_native),
        "u4": (k_compute, k_native),
        "y4_full": (k_compute, k_compute),
    })
    return outputs
