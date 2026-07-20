"""Unified K-policy for MoE dynamic-K mechanism experiments (v23+).

Design goals (from the v23-v28 plan, section 三):
  - One KPolicy applied to all MoE blocks (no per-experiment monkeypatch copies).
  - Separate prefill_k and decode_k.
  - PHASE decided by CACHE STATE at the top-level forward (NOT seq_len==1 guess):
    empty/None past_key_values -> prefill; non-empty -> decode. Robust to chunked
    prefill and multi-token decode.
  - PHYSICAL skip of dropped experts (they never run their FFN).
  - Sync-free hot path (no .item()/.cpu()/.tolist() inside MoE forward).
  - weight_mode in {renorm_survivors, no_renorm, fold_mass_to_top1, calibrated_norm_match}.
  - Optional layer_selector / decode_step_selector (for v27 localization).

Usage:
    policy = KPolicy(prefill_k=8, decode_k=6, weight_mode="renorm_survivors")
    ctx = attach_policy(model, policy)          # patches MoE blocks + top-level hook
    ...                                          # run model.generate(...)
    stats = ctx.stats()                          # single sync read of counters
    detach_policy(model, ctx)                    # restore native forward
"""
from __future__ import annotations
import contextvars
import types
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
import torch.nn.functional as F

WEIGHT_MODES = ("renorm_survivors", "no_renorm", "fold_mass_to_top1", "calibrated_norm_match")

# thread/async-safe current phase ("prefill" | "decode"); default prefill for a
# bare forward with no cache.
_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar("moe_phase", default="prefill")


@dataclass
class KPolicy:
    prefill_k: int = 8
    decode_k: int = 8
    weight_mode: str = "renorm_survivors"
    # optional selectors for localization (v27). Return True => APPLY the policy K
    # at this (layer_idx)/(decode_step). When None => apply everywhere.
    layer_selector: Optional[Callable[[int], bool]] = None
    decode_step_selector: Optional[Callable[[int], bool]] = None
    # calibrated_norm_match scalars: dict[(layer_idx, k)] -> float, frozen offline.
    calib_scalars: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode {self.weight_mode!r} not in {WEIGHT_MODES}")
        for name, k in (("prefill_k", self.prefill_k), ("decode_k", self.decode_k)):
            if not (1 <= k):
                raise ValueError(f"{name}={k} must be >= 1")

    def k_for_phase(self, phase: str) -> int:
        return self.prefill_k if phase == "prefill" else self.decode_k


class PolicyContext:
    """Holds runtime state: the policy, per-block sync-free counters, the top-level
    pre-hook handle, and the decode-step counter."""

    def __init__(self, policy: KPolicy, blocks, block_cls, orig_forward):
        self.policy = policy
        self.blocks = list(blocks)
        self.block_cls = block_cls
        self.orig_forward = orig_forward
        self.hook_handle = None
        self.decode_step = 0  # advanced by the top-level hook on each decode call

    def reset_counters(self):
        for i, b in enumerate(self.blocks):
            dev = next(b.parameters()).device
            b._kp_ksum_prefill = torch.zeros((), device=dev, dtype=torch.long)
            b._kp_ksum_decode = torch.zeros((), device=dev, dtype=torch.long)
            b._kp_tok_prefill = 0
            b._kp_tok_decode = 0
            b._kp_layer_idx = i
        self.decode_step = 0

    def stats(self):
        """Single-sync read of realized average K per phase."""
        kp = torch.stack([b._kp_ksum_prefill for b in self.blocks]).sum().item()
        kd = torch.stack([b._kp_ksum_decode for b in self.blocks]).sum().item()
        tp = sum(b._kp_tok_prefill for b in self.blocks)
        td = sum(b._kp_tok_decode for b in self.blocks)
        return {
            "avg_k_prefill": round(kp / tp, 4) if tp else None,
            "avg_k_decode": round(kd / td, 4) if td else None,
            "tok_prefill": tp, "tok_decode": td,
        }


def _phase_from_cache(kwargs) -> str:
    """Decide phase from CACHE STATE, not sequence length.
    prefill := no usable past_key_values (None or length 0).
    decode  := past cache already present.
    Also honors cache_position[0] == 0 as prefill for the first step.
    """
    pkv = kwargs.get("past_key_values", None)
    if pkv is None:
        # could still be a cached call passing cache via positional; fall back to
        # cache_position if available.
        cp = kwargs.get("cache_position", None)
        if cp is not None and cp.numel() > 0:
            return "prefill" if int(cp[0]) == 0 else "decode"
        return "prefill"
    # HF Cache object: get_seq_length()==0 means empty
    try:
        seqlen = pkv.get_seq_length()
        return "prefill" if seqlen == 0 else "decode"
    except Exception:
        # legacy tuple cache
        try:
            if len(pkv) == 0 or pkv[0] is None:
                return "prefill"
            return "decode"
        except Exception:
            return "prefill"


def _make_moe_forward(ctx: PolicyContext):
    policy = ctx.policy

    def forward(self, hidden_states):
        bsz, seqlen, hdim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hdim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # [T, E]

        phase = _PHASE.get()
        E = routing_weights.shape[-1]
        k = min(policy.k_for_phase(phase), E)

        # decide whether to APPLY policy here (layer/step selectors); else native top-8
        apply = True
        if policy.layer_selector is not None and not policy.layer_selector(getattr(self, "_kp_layer_idx", -1)):
            apply = False
        if apply and phase == "decode" and policy.decode_step_selector is not None \
                and not policy.decode_step_selector(ctx.decode_step):
            apply = False
        eff_k = k if apply else min(self.top_k, E)

        rw, selected = torch.topk(routing_weights, self.top_k if not apply else max(self.top_k, eff_k), dim=-1)
        # We always take a pool of at least native top_k so that keep-all == native.
        pool = rw.shape[-1]
        # keep top eff_k of the pool
        ar = torch.arange(pool, device=rw.device).unsqueeze(0)
        keep = ar < eff_k  # [1, pool] broadcast -> [T, pool]
        keep = keep.expand(rw.shape[0], pool)

        base = rw / rw.sum(dim=-1, keepdim=True) if self.norm_topk_prob else rw
        keep_f = keep.to(base.dtype)

        wm = policy.weight_mode
        if wm == "renorm_survivors":
            surv = (base * keep_f).sum(dim=-1, keepdim=True).clamp_min(1e-20)
            weights = base * keep_f * (base.sum(dim=-1, keepdim=True) / surv)
        elif wm == "calibrated_norm_match":
            # no_renorm relative mix, scaled by a frozen per-(layer,K) scalar to
            # match the K8 branch norm (isolates norm from redistribution).
            s = policy.calib_scalars.get(f"{getattr(self, '_kp_layer_idx', -1)},{int(eff_k)}", 1.0)
            weights = base * keep_f * float(s)
        elif wm == "no_renorm":
            weights = base * keep_f
        else:  # fold_mass_to_top1
            dropped = (base * (~keep).to(base.dtype)).sum(dim=-1, keepdim=True)
            weights = base * keep_f
            weights[:, 0:1] = weights[:, 0:1] + dropped
        weights = weights.to(hidden_states.dtype)

        # PHYSICAL skip: dropped (token, rank) removed from the mask
        expert_mask = F.one_hot(selected, num_classes=self.num_experts).permute(2, 1, 0).bool()  # [E, pool, T]
        expert_mask &= keep.transpose(0, 1).unsqueeze(0)  # [1, pool, T]

        final = torch.zeros((bsz * seqlen, hdim), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            cur = hidden_states[None, top_x].reshape(-1, hdim)
            out = expert_layer(cur) * weights[top_x, idx, None]
            final.index_add_(0, top_x, out.to(hidden_states.dtype))

        # sync-free counters
        ksum = keep.sum()
        if phase == "prefill":
            self._kp_ksum_prefill = self._kp_ksum_prefill + ksum
            self._kp_tok_prefill += keep.shape[0]
        else:
            self._kp_ksum_decode = self._kp_ksum_decode + ksum
            self._kp_tok_decode += keep.shape[0]

        return final.reshape(bsz, seqlen, hdim), router_logits

    return forward


def _find_moe_blocks(model):
    from transformers.models.qwen3_moe import modeling_qwen3_moe as M
    blocks = [m for m in model.modules() if isinstance(m, M.Qwen3MoeSparseMoeBlock)]
    return blocks, M.Qwen3MoeSparseMoeBlock


def attach_policy(model, policy: KPolicy) -> PolicyContext:
    blocks, block_cls = _find_moe_blocks(model)
    ctx = PolicyContext(policy, blocks, block_cls, block_cls.forward)
    ctx.reset_counters()
    fwd = _make_moe_forward(ctx)
    for b in blocks:
        b.forward = types.MethodType(fwd, b)

    # top-level pre-hook: set phase from cache state, advance decode step counter
    def pre_hook(module, args, kwargs):
        phase = _phase_from_cache(kwargs)
        _PHASE.set(phase)
        if phase == "decode":
            ctx.decode_step += 1
        else:
            ctx.decode_step = 0
        return None

    ctx.hook_handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    return ctx


def detach_policy(model, ctx: PolicyContext):
    for b in ctx.blocks:
        try:
            del b.forward
        except AttributeError:
            pass
    ctx.block_cls.forward = ctx.orig_forward
    if ctx.hook_handle is not None:
        ctx.hook_handle.remove()
        ctx.hook_handle = None
