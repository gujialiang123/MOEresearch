"""P0 execution checks: reference mass, BF16 order, gating and observer neutrality."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _toy import CountingExpert, HDIM, native_forward
from moe_research import k_policy as KP
from moe_research.gain_calibration import GainCalibrator
from moe_research.state_adapter import (
    apply_post_calibration, execute_branches, route_and_aggregate, shadow_branches,
)


class WideMoE(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        torch.manual_seed(714)
        self.num_experts, self.top_k, self.norm_topk_prob = 16, 8, True
        self.gate = nn.Linear(HDIM, self.num_experts, bias=False)
        self.counter = [0] * self.num_experts
        self.experts = nn.ModuleList([CountingExpert(i, self.counter) for i in range(self.num_experts)])
        self.native_calls = 0
        self.to(dtype=dtype)

    def forward(self, hidden):
        self.native_calls += 1
        return native_forward(self, hidden), self.gate(hidden.reshape(-1, HDIM))


class FakeCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class ToyModel(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, past_key_values=None):
        return self.block(hidden)


def attach(block, policy):
    model = ToyModel(block)
    with patch.object(KP, "_find_moe_blocks", return_value=([block], WideMoE)):
        ctx = KP.attach_policy(model, policy)
    return model, ctx


def test_reference16_computes_only_four_ffns_per_token():
    block = WideMoE()
    hidden = torch.randn(1, 5, HDIM)
    out, logits, info = route_and_aggregate(block, hidden, 4, 16)
    assert sum(block.counter) == 5 * 4
    expected_weights = logits.softmax(-1).topk(4, dim=-1).values
    torch.testing.assert_close(info["weights"], expected_weights)
    # Reference-16 is all experts in this toy: the denominator is exactly one.
    expected = torch.zeros_like(hidden.reshape(-1, HDIM))
    hs = hidden.reshape(-1, HDIM)
    for rank in range(4):
        for token, expert in enumerate(info["selected"][:, rank]):
            expected[token] += block.experts[expert](hs[token:token+1])[0] * expected_weights[token, rank]
    torch.testing.assert_close(out.reshape_as(expected), expected)


def test_bf16_tied_router_keeps_selection_fixed_across_reference_k():
    """R16 changes only denominator mass, including when all logits tie."""
    block = WideMoE(torch.bfloat16)
    with torch.no_grad():
        block.gate.weight.zero_()
    hidden = torch.ones(1, 3, HDIM, dtype=torch.bfloat16)
    probabilities = torch.full((3, 16), 1 / 16, dtype=torch.float32)
    # This is the exact native top-8 operation, including its native tie order.
    native_selection = probabilities.topk(8, -1).indices[:, :4]
    rows, outputs = {}, {}
    for arm, reference in (("F", 4), ("R8", 8), ("R16", 16)):
        block.counter[:] = [0] * block.num_experts
        output, _, info = route_and_aggregate(block, hidden, 4, reference)
        assert torch.equal(info["selected"], native_selection)
        assert info["selection_pool_k"] == 8
        assert torch.equal(info["reference_mass"], torch.full((3, 1), reference / 16))
        assert torch.equal(info["weights"], torch.full((3, 4), 1 / reference, dtype=torch.bfloat16))
        assert sum(block.counter) == 3 * 4
        assert {i for i, count in enumerate(block.counter) if count} == set(native_selection.flatten().tolist())
        rows[arm], outputs[arm] = info, output
    assert torch.equal(rows["F"]["selected"], rows["R8"]["selected"])
    assert torch.equal(rows["F"]["selected"], rows["R16"]["selected"])
    torch.testing.assert_close(outputs["R8"], outputs["F"] * 0.5, rtol=0, atol=0)
    torch.testing.assert_close(outputs["R16"], outputs["F"] * 0.25, rtol=0, atol=0)
    # Joint shadow execution must not let the largest reference change the
    # selection pool for any branch, nor execute reference-only FFNs.
    block.counter[:] = [0] * block.num_experts
    combined, _, info = execute_branches(block, hidden, {"F": (4, 4), "R8": (4, 8), "R16": (4, 16)})
    assert sum(block.counter) == 3 * 4
    for arm in outputs:
        assert torch.equal(info[arm]["selected"], native_selection)
        torch.testing.assert_close(combined[arm], outputs[arm], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("arm,reference", [("N", 8), ("F", 4), ("R8", 8), ("R16", 16), ("M_N", 8), ("L_N", 4)])
def test_every_arm_shadow_matches_real(dtype, arm, reference):
    block = WideMoE(dtype)
    # Decode batch=1 is the frozen deployment protocol. Each expert sees exactly
    # the same one-row FFN input in the full shadow and reduced real branch.
    hidden = torch.randn(1, 1, HDIM).to(dtype)
    shadow = shadow_branches(block, hidden)
    transform = None
    if arm == "M_N":
        transform = lambda layer, phase, step, output: apply_post_calibration(output, 1.731)
        expected = transform(0, "decode", 1, shadow["u4"])
    elif arm == "L_N":
        mu = torch.linspace(-0.7, 0.8, HDIM)
        ratio = torch.linspace(0.5, 1.9, HDIM)
        target_mu = torch.linspace(0.3, -0.4, HDIM)
        transform = lambda layer, phase, step, output: ((output.float() - mu) * ratio + target_mu).to(dtype)
        expected = transform(0, "decode", 1, shadow["y4_full"])
    elif arm == "N":
        expected = shadow["y8"]
    elif arm == "F":
        expected = shadow["y4_full"]
    elif arm == "R8":
        expected = shadow["u4"]
    else:
        expected = execute_branches(block, hidden, {"R16": (4, 16)})[0]["R16"]
    model, ctx = attach(block, KP.KPolicy(
        k_compute=8 if arm == "N" else 4, k_reference=reference,
        force_native_execution=True, output_transform=transform,
    ))
    actual, _ = model(hidden, past_key_values=FakeCache(3))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    KP.detach_policy(model, ctx)


def test_forced_native_executes_adapter_and_detach_restores_original():
    block = WideMoE()
    hidden = torch.randn(1, 4, HDIM)
    reference, _ = block(hidden)
    model, ctx = attach(block, KP.KPolicy(force_native_execution=True))
    calls_before = block.native_calls
    for cache in (None, FakeCache(4), FakeCache(5)):
        out, _ = model(hidden if cache is None else hidden[:, :1], past_key_values=cache)
        expected = reference if cache is None else native_forward(block, hidden[:, :1])
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert block.native_calls == calls_before, "force flag must bypass original forward"
    assert ctx.decode_step == 2
    KP.detach_policy(model, ctx)
    restored, _ = model(hidden)
    assert block.native_calls == calls_before + 1
    torch.testing.assert_close(restored, reference, rtol=0, atol=0)
    # A new arm starts a new cache-phase counter and no previous transform.
    model, ctx = attach(block, KP.KPolicy(k_compute=4, k_reference=8))
    assert ctx.decode_step == 0
    KP.detach_policy(model, ctx)


@pytest.mark.parametrize("mode", ["fixed_gain", "shuffled_gain"])
def test_nontrivial_calibration_is_gated_in_prefill_windows_and_identity(mode):
    block = WideMoE()
    hidden = torch.randn(1, 1, HDIM)
    reference, _ = block(hidden)
    cal = GainCalibrator([], 4)
    cal.pools = {(0, 0): [2.0, 2.5]}
    transforms = []
    def affine(layer, phase, step, output):
        transforms.append((layer, phase, step))
        return apply_post_calibration(output, scale=1.8, bias=0.75)
    policy = KP.KPolicy(
        decode_k=4, weight_mode=mode, calib_scalars={"0,4": 2.3, "0,8": 9.1},
        gain_provider=cal.make_shuffled_provider(14), output_transform=affine,
        layer_selector=lambda layer: layer == 0,
        decode_step_selector=lambda step: step == 2,
    )
    model, ctx = attach(block, policy)
    # Native prefill, first decode (outside window), then active second decode.
    torch.testing.assert_close(model(hidden)[0], reference, rtol=0, atol=0)
    torch.testing.assert_close(model(hidden, past_key_values=FakeCache(1))[0], reference, rtol=0, atol=0)
    active = model(hidden, past_key_values=FakeCache(2))[0]
    assert not torch.equal(active, reference)
    assert transforms == [(0, "decode", 2)]
    # Same non-unit scalar, nonzero bias and real shuffled provider, but identity K8.
    policy.decode_k = 8
    ctx.decode_step = 1
    torch.testing.assert_close(model(hidden, past_key_values=FakeCache(3))[0], reference, rtol=0, atol=0)
    policy.decode_k = 4
    policy.layer_selector = lambda layer: False
    ctx.decode_step = 1
    torch.testing.assert_close(model(hidden, past_key_values=FakeCache(3))[0], reference, rtol=0, atol=0)
    assert transforms == [(0, "decode", 2)]
    KP.detach_policy(model, ctx)


def test_observer_is_after_affine_and_cannot_mutate_driver_tensors():
    block = WideMoE(torch.bfloat16)
    hidden = torch.randn(1, 1, HDIM).bfloat16()
    original_hidden = hidden.clone()
    reduced, _, _ = route_and_aggregate(block, hidden, 4, 8)
    expected = apply_post_calibration(reduced, 2.17, 0.43)
    observed = []
    def observer(module, h, output, phase, layer, step, route_info):
        assert phase == "decode" and layer == 0 and step == 1
        observed.append(output.clone())
        h.zero_()
        output.zero_()
        return torch.zeros_like(output)
    model, ctx = attach(block, KP.KPolicy(
        k_compute=4, k_reference=8,
        output_transform=lambda layer, phase, step, output: apply_post_calibration(output, 2.17, 0.43),
        observer=observer,
    ))
    actual = model(hidden, past_key_values=FakeCache(10))[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    assert torch.equal(hidden, original_hidden)
    KP.detach_policy(model, ctx)


def test_postaggregate_scalar_is_bf16_then_fp32_then_bf16():
    block = WideMoE(torch.bfloat16)
    hidden = torch.randn(1, 1, HDIM).bfloat16()
    reduced = route_and_aggregate(block, hidden, 4, 8)[0]
    model, ctx = attach(block, KP.KPolicy(
        decode_k=4, weight_mode="decode_norm_match", calib_scalars={"0,4": 1.731}
    ))
    actual = model(hidden, past_key_values=FakeCache(10))[0]
    expected = (reduced.float() * 1.731).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    KP.detach_policy(model, ctx)


def test_unsupported_layout_and_invalid_reference_fail_explicitly():
    block = WideMoE()
    hidden = torch.randn(1, 1, HDIM)
    with pytest.raises(ValueError, match="k_reference must be >="):
        route_and_aggregate(block, hidden, 4, 2)
    block.experts = nn.Linear(HDIM, HDIM)
    with pytest.raises(TypeError, match="Unsupported MoE expert layout"):
        route_and_aggregate(block, hidden, 4, 8)
    assert sum(block.counter) == 0


def test_shuffled_provider_prefill_is_unit_and_does_not_consume_rng():
    cal = GainCalibrator([], 4)
    cal.pools = {(0, 0): [2.0, 3.0, 4.0, 5.0]}
    first, second = cal.make_shuffled_provider(10), cal.make_shuffled_provider(10)
    assert torch.equal(first(0, "prefill", 0, 9, "cpu"), torch.ones(9, 1))
    assert torch.equal(first(0, "decode", 1, 15, "cpu"), second(0, "decode", 1, 15, "cpu"))


def test_tiny_qwen_forced_native_cached_decode_and_local_outputs():
    """Exercise the actual locked HF interface without downloading weights."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(808)
    config = Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        moe_intermediate_size=12, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        num_experts=16, num_experts_per_tok=8, norm_topk_prob=True,
        max_position_embeddings=128, attention_dropout=0.0,
        eos_token_id=1, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).eval()
    ids = torch.tensor([[3, 5, 8, 13]])
    blocks, _ = KP._find_moe_blocks(model)
    captured = []
    handles = [block.register_forward_hook(
        lambda block, args, output: captured.append(tuple(x.detach().clone() for x in output))
    ) for block in blocks]
    def run():
        captured.clear()
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            logits, tokens = [out.logits.clone()], []
            cache = out.past_key_values
            for _ in range(3):
                token = out.logits[:, -1:].argmax(-1)
                tokens.append(token.clone())
                out = model(token, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits.append(out.logits.clone())
        return logits, tokens, list(captured)
    native, native_tokens, native_local = run()
    repeat, _, _ = run()
    assert all(torch.equal(a, b) for a, b in zip(native, repeat)), "CPU native noise baseline"
    ctx = KP.attach_policy(model, KP.KPolicy(force_native_execution=True))
    forced, forced_tokens, forced_local = run()
    assert ctx.decode_step == 3
    assert all(torch.equal(a, b) for a, b in zip(native, forced))
    assert all(torch.equal(a, b) for a, b in zip(native_tokens, forced_tokens))
    for native_pair, forced_pair in zip(native_local, forced_local):
        for native_tensor, forced_tensor in zip(native_pair, forced_pair):
            torch.testing.assert_close(native_tensor, forced_tensor, rtol=0, atol=0)
        # Both the router choices and final cast weights must coincide.
        p_native = native_pair[1].float().softmax(-1).topk(8, -1)
        p_forced = forced_pair[1].float().softmax(-1).topk(8, -1)
        assert torch.equal(p_native.indices, p_forced.indices)
        assert torch.equal(p_native.values / p_native.values.sum(-1, keepdim=True),
                           p_forced.values / p_forced.values.sum(-1, keepdim=True))
    KP.detach_policy(model, ctx)
    restored, restored_tokens, _ = run()
    assert all(torch.equal(a, b) for a, b in zip(native, restored))
    assert all(torch.equal(a, b) for a, b in zip(native_tokens, restored_tokens))
    for handle in handles:
        handle.remove()
