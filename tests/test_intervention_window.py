"""test_intervention_window: pulse/layer selectors apply low-K only inside the window
and leave everything outside exactly native."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from _toy import ToyMoE, native_forward, attach, HDIM
from moe_research import k_policy as KP
from moe_research.interventions import pulse_selector, pulse_start, layer_window_selector


def test_pulse_selector_window():
    sel = pulse_selector(start=5, dur=3)
    assert [s for s in range(1, 12) if sel(s)] == [5, 6, 7]
    print("PASS test_pulse_selector_window")


def test_pulse_start_rules_and_fallback():
    assert pulse_start("early", 200, 16) == 1
    assert pulse_start("middle", 200, 16) == 80
    assert pulse_start("late", 200, 16, late_offset=64) == 136
    # too short -> None
    assert pulse_start("late", 5, 16) is None
    # clamp so pulse fits
    s = pulse_start("late", 30, 16, late_offset=64)
    assert s is not None and s + 16 < 30
    print("PASS test_pulse_start_rules_and_fallback")


def test_decode_step_selector_isolates():
    """Outside the pulse window decode output == native; inside it differs (K4)."""
    block = ToyMoE()
    pol = KP.KPolicy(prefill_k=8, decode_k=4, weight_mode="renorm_survivors",
                     decode_step_selector=pulse_selector(2, 1))
    ctx = attach(block, pol)
    h = torch.randn(1, 1, HDIM)
    ref = native_forward(block, h)
    KP._PHASE.set("decode")
    ctx.decode_step = 1  # outside window -> native
    o1, _ = block(h)
    ctx.decode_step = 2  # inside window -> K4
    o2, _ = block(h)
    assert (o1 - ref).abs().max().item() < 1e-5, "outside window must be native"
    assert (o2 - ref).abs().max().item() > 1e-4, "inside window must differ"
    print("PASS test_decode_step_selector_isolates")


def test_layer_selector():
    sel = layer_window_selector([1, 3])
    assert sel(1) and sel(3) and not sel(0) and not sel(2)
    print("PASS test_layer_selector")


if __name__ == "__main__":
    test_pulse_selector_window()
    test_pulse_start_rules_and_fallback()
    test_decode_step_selector_isolates()
    test_layer_selector()
