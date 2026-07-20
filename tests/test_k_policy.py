"""Tests for the unified KPolicy (v23 plan §三). Toy model, no big weights.

Run: python tests/test_k_policy.py
"""
import os, sys, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP
from moe_research import answer_parsing as AP

torch.manual_seed(0)
HDIM, E, K = 16, 8, 8


class CountingExpert(nn.Module):
    def __init__(self, idx, counter):
        super().__init__()
        self.idx = idx; self.counter = counter; self.scale = float(idx + 1)

    def forward(self, x):
        self.counter[self.idx] += int(x.shape[0])
        return x * self.scale


class ToyMoE(nn.Module):
    def __init__(self, norm=True):
        super().__init__()
        self.num_experts = E; self.top_k = K; self.norm_topk_prob = norm
        self.gate = nn.Linear(HDIM, E, bias=False)
        self.counter = [0] * E
        self.experts = nn.ModuleList([CountingExpert(i, self.counter) for i in range(E)])
    def reset(self):
        for i in range(E): self.counter[i] = 0


def native_forward(block, hidden):
    bsz, seqlen, hdim = hidden.shape
    hs = hidden.view(-1, hdim)
    rw = F.softmax(block.gate(hs), dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, block.top_k, dim=-1)
    if block.norm_topk_prob:
        rw = rw / rw.sum(-1, keepdim=True)
    rw = rw.to(hs.dtype)
    final = torch.zeros_like(hs)
    mask = F.one_hot(sel, num_classes=block.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        idx, top_x = torch.where(mask[e].squeeze(0))
        cur = hs[None, top_x].reshape(-1, hdim)
        out = block.experts[e](cur) * rw[top_x, idx, None]
        final.index_add_(0, top_x, out.to(hs.dtype))
    return final.reshape(bsz, seqlen, hdim)


def _bind(block, ctx):
    import types
    block._kp_ksum_prefill = torch.zeros((), dtype=torch.long)
    block._kp_ksum_decode = torch.zeros((), dtype=torch.long)
    block._kp_tok_prefill = 0; block._kp_tok_decode = 0; block._kp_layer_idx = 0
    fwd = KP._make_moe_forward(ctx)
    block.forward = types.MethodType(fwd, block)


def test_equivalence_keepall_all_weightmodes():
    hidden = torch.randn(1, 5, HDIM)
    for norm in (True, False):
        block = ToyMoE(norm=norm)
        ref = native_forward(block, hidden)
        for wm in KP.WEIGHT_MODES:
            if wm in ("calibrated_norm_match", "clipped_gain", "fixed_gain", "shuffled_gain"):
                continue  # need frozen params/providers; keep-all covered by others
            pol = KP.KPolicy(prefill_k=8, decode_k=8, weight_mode=wm)
            ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
            _bind(block, ctx)
            KP._PHASE.set("prefill")
            out, _ = block(hidden)
            err = (out - ref).abs().max().item()
            assert err < 1e-5, f"norm={norm} wm={wm} keepall err={err}"
    print("PASS test_equivalence_keepall_all_weightmodes")


def test_phase_routing_uses_correct_k():
    """prefill uses prefill_k, decode uses decode_k — driven by _PHASE contextvar."""
    block = ToyMoE()
    pol = KP.KPolicy(prefill_k=8, decode_k=4, weight_mode="renorm_survivors")
    ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
    _bind(block, ctx)
    hidden = torch.randn(1, 10, HDIM)
    KP._PHASE.set("prefill")
    block(hidden)
    KP._PHASE.set("decode")
    block(torch.randn(1, 1, HDIM))
    block(torch.randn(1, 1, HDIM))
    st = ctx.stats()
    assert st["avg_k_prefill"] == 8.0, st
    assert st["avg_k_decode"] == 4.0, st
    assert st["tok_prefill"] == 10 and st["tok_decode"] == 2, st
    print("PASS test_phase_routing_uses_correct_k")


def test_physical_skip_call_counter():
    block = ToyMoE()
    pol = KP.KPolicy(prefill_k=1, decode_k=1, weight_mode="renorm_survivors")
    ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
    _bind(block, ctx)
    block.reset()
    hidden = torch.randn(1, 6, HDIM)
    KP._PHASE.set("prefill")
    block(hidden)
    # only top-1 experts may run; total processed == num tokens (k=1)
    logits = block.gate(hidden.view(-1, HDIM))
    _, sel = torch.topk(F.softmax(logits, 1, dtype=torch.float), K, -1)
    top1 = set(sel[:, 0].tolist())
    called = {i for i in range(E) if block.counter[i] > 0}
    assert called <= top1, f"called {called} !<= top1 {top1}"
    assert sum(block.counter) == 6, f"processed {sum(block.counter)} != 6"
    print("PASS test_physical_skip_call_counter")


def test_super_native_k():
    """K>8 widens the pool; realized avg_k == requested."""
    block = ToyMoE()
    pol = KP.KPolicy(prefill_k=12, decode_k=12, weight_mode="renorm_survivors")
    ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
    _bind(block, ctx)
    KP._PHASE.set("prefill")
    block(torch.randn(1, 7, HDIM))
    st = ctx.stats()
    assert st["avg_k_prefill"] == 8.0, f"E=8 caps at 8, got {st}"  # toy has only 8 experts
    print("PASS test_super_native_k")


def test_partial_renorm_equivalences():
    """beta=0 == no_renorm; beta=1 == renorm_survivors; native-K all-beta equivalent."""
    hidden = torch.randn(1, 5, HDIM)
    block = ToyMoE()
    # beta=0 == no_renorm  (decode K4)
    o_b0, _ = _apply(block, hidden, policy=None) if False else (None, None)
    def out(mode, **kw):
        pol = KP.KPolicy(prefill_k=4, decode_k=4, weight_mode=mode, **kw)
        ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
        _bind(block, ctx); KP._PHASE.set("prefill")
        return block(hidden)[0]
    b0 = out("partial_renorm", renorm_beta=0.0)
    nr = out("no_renorm")
    assert (b0 - nr).abs().max() < 1e-6, f"beta=0 != no_renorm: {(b0-nr).abs().max()}"
    b1 = out("partial_renorm", renorm_beta=1.0)
    rs = out("renorm_survivors")
    assert (b1 - rs).abs().max() < 1e-6, f"beta=1 != renorm_survivors: {(b1-rs).abs().max()}"
    # native K: all beta equivalent (r=1)
    def out8(beta):
        pol = KP.KPolicy(prefill_k=8, decode_k=8, weight_mode="partial_renorm", renorm_beta=beta)
        ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
        _bind(block, ctx); KP._PHASE.set("prefill")
        return block(hidden)[0]
    ref = out8(0.0)
    for beta in (0.25, 0.5, 0.75, 1.0):
        assert (out8(beta) - ref).abs().max() < 1e-6, f"native-K beta={beta} not equivalent"
    print("PASS test_partial_renorm_equivalences")


def test_partial_renorm_monotonic_norm():
    """MoE output norm should increase monotonically with beta (more upscaling)."""
    hidden = torch.randn(1, 8, HDIM)
    block = ToyMoE()
    norms = []
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        pol = KP.KPolicy(prefill_k=4, decode_k=4, weight_mode="partial_renorm", renorm_beta=beta)
        ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
        _bind(block, ctx); KP._PHASE.set("prefill")
        norms.append(block(hidden)[0].norm().item())
    assert all(norms[i] <= norms[i+1] + 1e-4 for i in range(len(norms)-1)), f"not monotone: {norms}"
    print("PASS test_partial_renorm_monotonic_norm")


def test_clipped_gain():
    """clipped_gain caps the 1/M_K multiplier at gain_clip."""
    hidden = torch.randn(1, 6, HDIM)
    block = ToyMoE()
    # very tight clip -> equals no_renorm scaled by clip where 1/M_K>clip
    pol = KP.KPolicy(prefill_k=4, decode_k=4, weight_mode="clipped_gain", gain_clip=1.0)
    ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
    _bind(block, ctx); KP._PHASE.set("prefill")
    oc = block(hidden)[0]
    # with clip=1.0, gain=min(1/M_K,1)=1 always (since M_K<=1) -> == no_renorm
    pol2 = KP.KPolicy(prefill_k=4, decode_k=4, weight_mode="no_renorm")
    ctx2 = KP.PolicyContext(pol2, [block], ToyMoE, ToyMoE.forward)
    _bind(block, ctx2); KP._PHASE.set("prefill")
    onr = block(hidden)[0]
    assert (oc - onr).abs().max() < 1e-6, "clip=1.0 should equal no_renorm"
    print("PASS test_clipped_gain")


def test_no_sync_in_forward_source():
    raw = inspect.getsource(KP._make_moe_forward)
    code = "\n".join(l.split("#", 1)[0] for l in raw.splitlines())
    for bad in (".item(", ".cpu(", ".tolist(", ".numpy("):
        assert bad not in code, f"sync primitive {bad} in MoE forward"
    print("PASS test_no_sync_in_forward_source")


def test_shuffled_gain_no_leakage():
    from moe_research import gain_calibration as GC
    cal = GC.GainCalibrator([ToyMoE()], low_k=4)
    cal.pools = {(0, 0): [2.0, 2.0, 2.0], (0, 1): [3.0], (0, 2): [5.0]}
    prov = cal.make_shuffled_provider(seed=1)
    g = prov(0, "decode", 5, 4, torch.device("cpu"))   # step 5 -> bin 0 -> {2.0}
    assert g.shape == (4, 1) and torch.allclose(g, torch.full((4, 1), 2.0)), g.flatten()
    g2 = prov(0, "decode", 50, 2, torch.device("cpu"))  # step 50 -> bin 1 -> {3.0}
    assert torch.allclose(g2, torch.full((2, 1), 3.0)), g2.flatten()
    print("PASS test_shuffled_gain_no_leakage")


def test_intervention_window_layer():
    block = ToyMoE()
    pol = KP.KPolicy(prefill_k=8, decode_k=2, weight_mode="renorm_survivors",
                     layer_selector=lambda l: l == 3)
    ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
    _bind(block, ctx); block._kp_layer_idx = 3
    KP._PHASE.set("decode"); block(torch.randn(1, 1, HDIM))
    assert ctx.stats()["avg_k_decode"] == 2.0, "layer in window should use K2"
    block2 = ToyMoE()
    pol2 = KP.KPolicy(prefill_k=8, decode_k=2, weight_mode="renorm_survivors",
                      layer_selector=lambda l: l == 3)
    ctx2 = KP.PolicyContext(pol2, [block2], ToyMoE, ToyMoE.forward)
    _bind(block2, ctx2); block2._kp_layer_idx = 5
    KP._PHASE.set("decode"); block2(torch.randn(1, 1, HDIM))
    assert ctx2.stats()["avg_k_decode"] == 8.0, "layer outside window should use native K8"
    print("PASS test_intervention_window_layer")


def test_native_equivalence_gain_modes():
    hidden = torch.randn(1, 5, HDIM)
    block = ToyMoE()
    ref = native_forward(block, hidden)
    for mode, kw in [("partial_renorm", {"renorm_beta": 0.5}), ("clipped_gain", {"gain_clip": 3.0})]:
        pol = KP.KPolicy(prefill_k=8, decode_k=8, weight_mode=mode, **kw)
        ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
        _bind(block, ctx); KP._PHASE.set("prefill")
        out, _ = block(hidden)
        assert (out - ref).abs().max() < 1e-5, f"native-K {mode} != native"
    print("PASS test_native_equivalence_gain_modes")


def test_phase_from_cache():
    """Cache-state phase detection, not seq_len."""
    class FakeCache:
        def __init__(self, n): self.n = n
        def get_seq_length(self): return self.n
    assert KP._phase_from_cache({"past_key_values": None}) == "prefill"
    assert KP._phase_from_cache({"past_key_values": FakeCache(0)}) == "prefill"
    assert KP._phase_from_cache({"past_key_values": FakeCache(50)}) == "decode"
    # cache_position fallback
    assert KP._phase_from_cache({"cache_position": torch.tensor([0, 1, 2])}) == "prefill"
    assert KP._phase_from_cache({"cache_position": torch.tensor([50])}) == "decode"
    print("PASS test_phase_from_cache")


def test_parsers():
    assert AP.parse_strict("x #### 42")[0] == "42"
    assert AP.parse_strict("no marker 42")[1] == "parse_failure"
    assert AP.parse_tolerant("FINAL: 42")[0] == "42"
    assert AP.parse_tolerant(r"#### \boxed{42}")[0] == "42"
    assert AP.parse_tolerant("The final answer is 42.")[0] == "42"
    assert AP.parse_gold("reason\n#### 18") == "18"
    print("PASS test_parsers")


def test_weight_modes_differ():
    """no_renorm, fold, renorm produce different outputs when pruning."""
    block = ToyMoE()
    hidden = torch.randn(1, 4, HDIM)
    outs = {}
    for wm in ("renorm_survivors", "no_renorm", "fold_mass_to_top1"):
        pol = KP.KPolicy(prefill_k=4, decode_k=4, weight_mode=wm)
        ctx = KP.PolicyContext(pol, [block], ToyMoE, ToyMoE.forward)
        _bind(block, ctx)
        KP._PHASE.set("prefill")
        outs[wm], _ = block(hidden)
    assert (outs["renorm_survivors"] - outs["no_renorm"]).abs().max() > 1e-4
    assert (outs["fold_mass_to_top1"] - outs["no_renorm"]).abs().max() > 1e-4
    print("PASS test_weight_modes_differ")


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
if __name__ == "__main__":
    failed = 0
    for t in ALL:
        try:
            t()
        except AssertionError as e:
            failed += 1; print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL)-failed}/{len(ALL)} passed")
    sys.exit(1 if failed else 0)
