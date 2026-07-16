"""Shared utilities for dynamic top-k MoE experiments (v20+).

Fixes the correctness/measurement problems found in v18:
  - PHYSICAL expert skip: dropped (token, rank, expert) assignments never enter
    the expert FFN (not just zeroed after compute).
  - NO CPU-GPU sync inside the timed MoE forward (no .item()/.cpu()/.tolist();
    counters accumulate in GPU tensors and are read ONCE at the end).
  - Named, monotonic-documented threshold policies.
  - Strict GSM8K `####` parsing with explicit parse-failure tracking.
  - prefill vs decode K accounted separately.

This module is model-agnostic: `make_dynamic_forward` returns a forward that can
replace `Qwen3MoeSparseMoeBlock.forward` (or any block exposing `.gate`,
`.experts`, `.num_experts`, `.norm_topk_prob`).
"""
from __future__ import annotations
import re
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Threshold policies. Each returns a boolean keep-mask over the top-`k` pool.
# Monotonic direction is documented and enforced by tests.
# ---------------------------------------------------------------------------
POLICIES = ("top_p_within_topk", "min_weight_cutoff", "max_dropped_mass")
RENORM_MODES = ("renorm_survivors", "no_renorm", "fold_mass_to_top1")


def compute_keep_mask(rw_norm: torch.Tensor, policy: str, threshold: float,
                      kmin: int, kmax: int) -> torch.Tensor:
    """rw_norm: [T, k] descending, normalized so each row sums to 1 over the pool.
    Returns bool keep-mask [T, k]. All policies respect kmin (always keep the
    top kmin) and kmax (pool size).

    Direction:
      top_p_within_topk : keep until cumulative retained mass reaches `threshold`
                          (tau). tau UP  => K UP.
      min_weight_cutoff : keep assignments with normalized weight > `threshold`.
                          cutoff UP => K DOWN.
      max_dropped_mass  : drop a tail suffix while total dropped mass <= `threshold`
                          (beta). beta UP => K DOWN.
    """
    device = rw_norm.device
    k = rw_norm.shape[-1]
    ar = torch.arange(k, device=device).unsqueeze(0)  # [1, k]

    if policy == "top_p_within_topk":
        cum_before = torch.cumsum(rw_norm, dim=-1) - rw_norm  # mass strictly before j
        keep = cum_before < threshold
    elif policy == "min_weight_cutoff":
        keep = rw_norm > threshold
    elif policy == "max_dropped_mass":
        # cum_tail[j] = sum_{i>=j} w_i  (mass of suffix starting at j)
        cum_tail = torch.cumsum(rw_norm.flip(-1), dim=-1).flip(-1)
        # keep j while dropping from j would remove more than beta (i.e. must keep)
        keep = cum_tail > threshold
    else:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")

    keep = keep | (ar < kmin)      # floor: always keep top-kmin
    keep = keep & (ar < kmax)      # ceiling
    return keep


# ---------------------------------------------------------------------------
# Dynamic MoE forward with PHYSICAL skip + sync-free counters.
# ---------------------------------------------------------------------------
def make_dynamic_forward(policy: str, threshold: float, kmin: int, kmax: int,
                         phase: str = "decode_only",
                         renorm: str = "renorm_survivors",
                         benchmark_mode: bool = False):
    """Return a forward(self, hidden_states) implementing dynamic-K with a real
    physical skip of dropped experts.

    phase: 'decode_only' | 'prefill_only' | 'all'. Decode is approximated by
      sequence_length == 1 (valid only for the current non-chunked HF generate
      reference path). When the current call is not in the active phase, the
      block runs native top-kmax (no pruning).

    benchmark_mode: when True, NO statistics are accumulated (zero extra kernels
      / no D2H). When False, counters accumulate in GPU tensors read once later.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy {policy!r} not in {POLICIES}")
    if renorm not in RENORM_MODES:
        raise ValueError(f"renorm {renorm!r} not in {RENORM_MODES}")
    if phase not in ("decode_only", "prefill_only", "all"):
        raise ValueError(phase)

    def forward(self, hidden_states):
        bsz, seqlen, hdim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hdim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # [T, E]

        k = min(kmax, routing_weights.shape[-1])
        rw, selected = torch.topk(routing_weights, k, dim=-1)                  # [T, k]
        # baseline per-assignment weight (matches native forward exactly when keep-all)
        base = rw / rw.sum(dim=-1, keepdim=True) if self.norm_topk_prob else rw
        rw_norm = rw / rw.sum(dim=-1, keepdim=True)                            # for policies

        is_prefill = seqlen > 1  # python int from shape -> no sync
        active = (phase == "all") or (phase == "decode_only" and not is_prefill) \
            or (phase == "prefill_only" and is_prefill)

        if active:
            keep = compute_keep_mask(rw_norm, policy, threshold, kmin, k)      # [T, k] bool
        else:
            keep = torch.ones_like(rw, dtype=torch.bool)

        # ---- weight aggregation over survivors ----
        keep_f = keep.to(base.dtype)
        if renorm == "renorm_survivors":
            # rescale survivors to carry the FULL original mass (== native when keep-all,
            # regardless of norm_topk_prob)
            surv = (base * keep_f).sum(dim=-1, keepdim=True).clamp_min(1e-20)
            weights = base * keep_f * (base.sum(dim=-1, keepdim=True) / surv)
        elif renorm == "no_renorm":
            weights = base * keep_f
        else:  # fold_mass_to_top1
            dropped = (base * (~keep).to(base.dtype)).sum(dim=-1, keepdim=True)
            weights = base * keep_f
            weights[:, 0:1] = weights[:, 0:1] + dropped
        weights = weights.to(hidden_states.dtype)                             # [T, k]

        # ---- PHYSICAL skip: dropped (token, rank) removed from the mask ----
        expert_mask = F.one_hot(selected, num_classes=self.num_experts)       # [T, k, E]
        expert_mask = expert_mask.permute(2, 1, 0).bool()                     # [E, k, T]
        expert_mask &= keep.transpose(0, 1).unsqueeze(0)                      # [1, k, T]

        final = torch.zeros((bsz * seqlen, hdim), dtype=hidden_states.dtype,
                            device=hidden_states.device)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            cur = hidden_states[None, top_x].reshape(-1, hdim)
            out = expert_layer(cur) * weights[top_x, idx, None]
            final.index_add_(0, top_x, out.to(hidden_states.dtype))

        # ---- sync-free counters (GPU tensors; read once at the end) ----
        if not benchmark_mode:
            ksum = keep.sum()  # stays a GPU scalar tensor; never read here
            if is_prefill:
                self._k_sum_prefill = self._k_sum_prefill + ksum
                self._tok_prefill += keep.shape[0]
            else:
                self._k_sum_decode = self._k_sum_decode + ksum
                self._tok_decode += keep.shape[0]

        return final.reshape(bsz, seqlen, hdim), router_logits

    return forward


class DynamicKController:
    """Attach a dynamic forward to every MoE block and collect sync-free stats."""

    def __init__(self, moe_blocks, block_cls):
        self.blocks = list(moe_blocks)
        self.block_cls = block_cls
        self._orig_forward = block_cls.forward

    def enable(self, policy, threshold, kmin, kmax, phase="decode_only",
               renorm="renorm_survivors", benchmark_mode=False):
        import types
        fwd = make_dynamic_forward(policy, threshold, kmin, kmax, phase, renorm, benchmark_mode)
        for b in self.blocks:
            dev = next(b.parameters()).device
            b._k_sum_prefill = torch.zeros((), device=dev, dtype=torch.long)
            b._k_sum_decode = torch.zeros((), device=dev, dtype=torch.long)
            b._tok_prefill = 0
            b._tok_decode = 0
            b.forward = types.MethodType(fwd, b)

    def reset(self):
        for b in self.blocks:
            dev = next(b.parameters()).device
            b._k_sum_prefill = torch.zeros((), device=dev, dtype=torch.long)
            b._k_sum_decode = torch.zeros((), device=dev, dtype=torch.long)
            b._tok_prefill = 0
            b._tok_decode = 0

    def disable(self):
        self.block_cls.forward = self._orig_forward
        for b in self.blocks:
            if type(b).forward is not self._orig_forward:
                try:
                    del b.forward
                except AttributeError:
                    pass

    def stats(self):
        """Read GPU counters ONCE (single sync)."""
        kp = torch.stack([b._k_sum_prefill for b in self.blocks]).sum().item()
        kd = torch.stack([b._k_sum_decode for b in self.blocks]).sum().item()
        tp = sum(b._tok_prefill for b in self.blocks)
        td = sum(b._tok_decode for b in self.blocks)
        return {
            "avg_k_prefill": round(kp / tp, 4) if tp else None,
            "avg_k_decode": round(kd / td, 4) if td else None,
            "tok_prefill": tp, "tok_decode": td,
            "k_sum_prefill": kp, "k_sum_decode": kd,
        }


# ---------------------------------------------------------------------------
# Strict GSM8K answer parsing.
# ---------------------------------------------------------------------------
_HASH_RE = re.compile(r"####\s*([-+]?[0-9][0-9,]*\.?[0-9]*)")
_NUM_RE = re.compile(r"[-+]?[0-9][0-9,]*\.?[0-9]*")


def _normalize_num(s: str):
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s.startswith("+"):
        s = s[1:]
    if "." in s:
        s = s.rstrip("0").rstrip(".")  # 72.0 -> 72
    return s if s not in ("", "-", "+") else None


def parse_strict(text: str):
    """Return (prediction, status). Strict: last number after a `####` marker.
    status in {'hash', 'parse_failure'}."""
    matches = _HASH_RE.findall(text)
    if matches:
        v = _normalize_num(matches[-1])
        return (v, "hash") if v is not None else (None, "parse_failure")
    return (None, "parse_failure")


def parse_fallback(text: str):
    """Fallback: last number anywhere. status 'fallback_last_number' or 'parse_failure'."""
    nums = _NUM_RE.findall(text)
    if nums:
        v = _normalize_num(nums[-1])
        return (v, "fallback_last_number") if v is not None else (None, "parse_failure")
    return (None, "parse_failure")


def parse_gold(answer: str):
    """GSM8K gold is the number after the final `####`."""
    v, status = parse_strict(answer)
    if v is not None:
        return v
    return _normalize_num(answer.split("####")[-1]) if "####" in answer else None


def first_hash_position(text: str):
    """Character index of the first `####`, or -1."""
    i = text.find("####")
    return i
