"""test_partial_renorm: beta=0 == no_renorm, beta=1 == full renorm_survivors,
K=native is identity for any beta, and branch norm is monotonic in beta."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from _toy import ToyMoE, attach, HDIM
from moe_research import k_policy as KP


def _run(block, wm, dk, beta=1.0):
    pol = KP.KPolicy(prefill_k=8, decode_k=dk, weight_mode=wm, renorm_beta=beta)
    ctx = attach(block, pol)
    KP._PHASE.set("decode")
    out, _ = block(torch.randn(1, 4, HDIM, generator=torch.Generator().manual_seed(1)))
    return out


def test_beta0_equals_no_renorm():
    for dk in (4, 6):
        h = torch.randn(1, 4, HDIM)
        b1 = ToyMoE(); b2 = ToyMoE()
        b2.load_state_dict(b1.state_dict())
        p1 = KP.KPolicy(prefill_k=8, decode_k=dk, weight_mode="partial_renorm", renorm_beta=0.0)
        p2 = KP.KPolicy(prefill_k=8, decode_k=dk, weight_mode="no_renorm")
        attach(b1, p1); attach(b2, p2); KP._PHASE.set("decode")
        o1, _ = b1(h); o2, _ = b2(h)
        assert (o1 - o2).abs().max().item() < 1e-5, dk
    print("PASS test_beta0_equals_no_renorm")


def test_beta1_equals_full_renorm():
    for dk in (4, 6):
        h = torch.randn(1, 4, HDIM)
        b1 = ToyMoE(); b2 = ToyMoE(); b2.load_state_dict(b1.state_dict())
        p1 = KP.KPolicy(prefill_k=8, decode_k=dk, weight_mode="partial_renorm", renorm_beta=1.0)
        p2 = KP.KPolicy(prefill_k=8, decode_k=dk, weight_mode="renorm_survivors")
        attach(b1, p1); attach(b2, p2); KP._PHASE.set("decode")
        o1, _ = b1(h); o2, _ = b2(h)
        assert (o1 - o2).abs().max().item() < 1e-5, dk
    print("PASS test_beta1_equals_full_renorm")


def test_native_k_identity_any_beta():
    h = torch.randn(1, 4, HDIM)
    b0 = ToyMoE(); ref = None
    for beta in (0.0, 0.3, 0.7, 1.0):
        b = ToyMoE(); b.load_state_dict(b0.state_dict())
        p = KP.KPolicy(prefill_k=8, decode_k=8, weight_mode="partial_renorm", renorm_beta=beta)
        attach(b, p); KP._PHASE.set("decode")
        o, _ = b(h)
        if ref is None:
            ref = o
        assert (o - ref).abs().max().item() < 1e-6, beta
    print("PASS test_native_k_identity_any_beta")


def test_branch_norm_monotonic_in_beta():
    h = torch.randn(1, 8, HDIM)
    prev = None
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        b = ToyMoE()
        p = KP.KPolicy(prefill_k=8, decode_k=4, weight_mode="partial_renorm", renorm_beta=beta)
        attach(b, p); KP._PHASE.set("decode")
        o, _ = b(h)
        nrm = o.float().norm(dim=-1).mean().item()
        if prev is not None:
            assert nrm >= prev - 1e-4, f"beta {beta} norm {nrm} < prev {prev}"
        prev = nrm
    print("PASS test_branch_norm_monotonic_in_beta")


if __name__ == "__main__":
    test_beta0_equals_no_renorm()
    test_beta1_equals_full_renorm()
    test_native_k_identity_any_beta()
    test_branch_norm_monotonic_in_beta()
