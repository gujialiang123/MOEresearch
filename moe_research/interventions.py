"""Intervention helpers for v32 pulse-and-recovery (localized K in time).

A "pulse" applies the low-K policy only on a contiguous window of DECODE steps
[start, start+dur) and restores native K8 elsewhere. The KPolicy machinery already
supports this via `decode_step_selector`; these helpers build the selectors and the
per-sample pulse-start positions under a FIXED, pre-registered rule (never chosen
from results).

Decode-step convention: the top-level pre-hook increments `ctx.decode_step` to 1 on
the first generated token, 2 on the second, etc. So a window selector returns True
for `start <= step < start+dur` with `start` 1-indexed.
"""
from __future__ import annotations


def pulse_selector(start: int, dur: int):
    """decode_step_selector: apply low-K on steps [start, start+dur)."""
    def sel(step: int) -> bool:
        return start <= step < start + dur
    return sel


def pulse_start(kind: str, baseline_len: int, dur: int, late_offset: int = 64):
    """Fixed rule for pulse start given a sample's baseline generation length.
    - early:  step 1
    - middle: ~40% of baseline length
    - late:   ~late_offset tokens before baseline EOS/marker
    Fallback for short samples: clamp so the pulse fits inside [1, baseline_len-1];
    returns None if the sample is too short to host the pulse (caller skips it)."""
    if baseline_len < dur + 2:
        return None
    if kind == "early":
        s = 1
    elif kind == "middle":
        s = max(1, int(round(0.40 * baseline_len)))
    elif kind == "late":
        s = max(1, baseline_len - late_offset)
    else:
        raise ValueError(kind)
    # clamp so [s, s+dur) stays strictly inside the generation
    s = min(s, max(1, baseline_len - dur - 1))
    return s


def layer_window_selector(layers):
    """layer_selector: apply policy K only on the given layer indices."""
    lset = set(layers)
    def sel(layer_idx: int) -> bool:
        return layer_idx in lset
    return sel


def clone_cache(cache):
    """Deep-copy an HF Cache so a branch/probe forward that appends new K/V does NOT
    mutate the incoming baseline cache (required for v32 fixed-history probes)."""
    import copy
    if cache is None:
        return None
    return copy.deepcopy(cache)
