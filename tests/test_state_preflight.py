"""The P0 gate must reject real errors, missing coverage and failed setup."""
import json
from types import SimpleNamespace

import pytest
import torch

from moe_research import state_preflight as P


def test_fixed_logit_gate_accepts_identity_and_rejects_errors():
    values = [torch.tensor([0.5, -1.0, 2.0]), torch.tensor([1.0, 0.0, -0.5])]
    error = P.compare_logits(values, [x.clone() for x in values])
    P.assert_logit_gate(error)
    assert error["max_abs"] == 0 and error["relative_l2"] == 0
    assert error["fixed_prefix_kl_max"] == 0 and error["argmax_agreement"] == 1
    changed = [x.clone() for x in values]
    changed[0][0] += 0.01
    with pytest.raises(P.PreflightFailure, match="Fixed logit"):
        P.assert_logit_gate(P.compare_logits(values, changed))
    # The tolerance is an immutable experiment constant, not adjusted by noise.
    assert P.TOLERANCES["logits_max_abs"] == 1e-4


def test_nonfinite_and_shape_mismatches_do_not_become_passing_metrics():
    with pytest.raises(P.PreflightFailure, match="shape mismatch"):
        P.tensor_errors(torch.zeros(2), torch.zeros(3))
    with pytest.raises(P.PreflightFailure, match="Non-finite"):
        P.tensor_errors(torch.zeros(1), torch.tensor([float("nan")]))
    with pytest.raises(P.PreflightFailure, match="lengths"):
        P.compare_logits([torch.zeros(2)], [])


def test_local_gate_checks_states_residuals_router_and_coverage():
    fields = {name: torch.ones(1, 2) for name in
              ("h", "output", "router_logits", "native_weights", "residual")}
    fields["native_selected"] = torch.tensor([[1, 0]])
    left = SimpleNamespace(records={(0, 0): fields}, phases=["prefill"], validate=lambda: None)
    right_fields = {key: value.clone() for key, value in fields.items()}
    right = SimpleNamespace(records={(0, 0): right_fields}, phases=["prefill"], validate=lambda: None)
    assert P.compare_captures(left, right)["layer_forward_comparisons"] == 1
    right_fields["residual"][0, 1] += 0.01
    with pytest.raises(P.PreflightFailure, match="field=residual"):
        P.compare_captures(left, right)
    right.records = {}
    with pytest.raises(P.PreflightFailure, match="Missing local"):
        P.compare_captures(left, right)


def test_failed_implementation_is_persisted_and_never_overwritten(tmp_path, monkeypatch):
    driver = SimpleNamespace(blocks=[])
    monkeypatch.setattr(P, "_interface", lambda driver: {"test": True})
    with pytest.raises(P.PreflightFailure, match="original 48-layer"):
        P.run_preflight(driver, tmp_path)
    first = (tmp_path / "preflight.json").read_bytes()
    record = json.loads(first)
    assert record["status"] == "FAILED_IMPLEMENTATION"
    assert record["tolerances"] == P.TOLERANCES
    assert record["error"]["type"] == "PreflightFailure"
    with pytest.raises(RuntimeError, match="preserved"):
        P.run_preflight(driver, tmp_path)
    assert (tmp_path / "preflight.json").read_bytes() == first


def test_synthetic_calibrations_are_nontrivial_and_labeled():
    driver = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(hidden_size=8)), blocks=[None] * 48)
    calibrations = P.synthetic_calibrations(driver)
    x = torch.linspace(-0.2, 0.3, 8).reshape(1, 1, 8).bfloat16()
    for family in ("M_N", "L_N"):
        cal = calibrations[family]
        assert not torch.equal(cal.apply(0, x), x)
        assert torch.equal(cal.apply(0, x, "prefill"), x)
        assert "not fitted" in cal.metadata["purpose"]
    lda = calibrations["L_N"].parameters["0"]
    assert any(value != 0 for value in lda["mean_native"])
    assert len(set(lda["std_native"])) > 1


def test_capture_owns_residual_hooks_without_production_observer():
    class Block(torch.nn.Module):
        top_k, norm_topk_prob = 1, True
        def forward(self, x):
            return x * 2, torch.zeros(x.shape[0] * x.shape[1], 2)
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = torch.nn.Identity()
            self.mlp = Block()
        def forward(self, x):
            return self.mlp(self.post_attention_layernorm(x + 3))[0]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Decoder()])
        def forward(self, x, past_key_values=None):
            return self.model.layers[0](x)
    model = Model()
    driver = SimpleNamespace(model=model, blocks=[model.model.layers[0].mlp], residuals={})
    x = torch.ones(1, 2, 3)
    with P._Capture(driver) as capture:
        model(x)
    capture.validate()
    assert driver.residuals == {}, "P0 capture must not require production observer state"
    assert torch.equal(capture.records[(0, 0)]["residual"], x + 3)
    assert torch.equal(capture.records[(0, 0)]["output"], (x + 3) * 2)
    assert not model.model.layers[0].post_attention_layernorm._forward_pre_hooks
