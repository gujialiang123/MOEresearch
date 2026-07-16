"""Tests for dynamic_topk_utils: physical skip, equivalence, monotonicity, no-sync.

Run:  python tests/test_dynamic_topk.py   (or: pytest tests/test_dynamic_topk.py)

Uses a tiny synthetic MoE block with call-counting experts, so NO large model
is needed. Proves the P0 correctness fixes.
"""
import os, sys, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import dynamic_topk_utils as U  # noqa: E402

torch.manual_seed(0)
HDIM, E, K = 16, 8, 8


class CountingExpert(nn.Module):
    """Deterministic expert that records how many tokens it processed."""
    def __init__(self, idx, counter):
        super().__init__()
        self.idx = idx
        self.counter = counter
        self.scale = float(idx + 1)

    def forward(self, x):
        self.counter[self.idx] += int(x.shape[0])
        return x * self.scale


class ToyMoEBlock(nn.Module):
    def __init__(self, norm_topk_prob=True):
        super().__init__()
        self.num_experts = E
        self.top_k = K
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(HDIM, E, bias=False)
        self.counter = [0] * E
        self.experts = nn.ModuleList([CountingExpert(i, self.counter) for i in range(E)])

    def reset_counter(self):
        for i in range(E):
            self.counter[i] = 0


def _native_reference(block, hidden):
    """Faithful native top-K forward (compute all, weight by normalized top-k)."""
    bsz, seqlen, hdim = hidden.shape
    hs = hidden.view(-1, hdim)
    logits = block.gate(hs)
    rw = F.softmax(logits, dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, block.top_k, dim=-1)
    if block.norm_topk_prob:
        rw = rw / rw.sum(dim=-1, keepdim=True)
    rw = rw.to(hs.dtype)
    final = torch.zeros((bsz * seqlen, hdim), dtype=hs.dtype, device=hs.device)
    mask = F.one_hot(sel, num_classes=block.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        idx, top_x = torch.where(mask[e].squeeze(0))
        cur = hs[None, top_x].reshape(-1, hdim)
        out = block.experts[e](cur) * rw[top_x, idx, None]
        final.index_add_(0, top_x, out.to(hs.dtype))
    return final.reshape(bsz, seqlen, hdim)


def _apply(block, hidden, **kw):
    import types
    fwd = U.make_dynamic_forward(**kw)
    block._k_sum_prefill = torch.zeros((), dtype=torch.long)
    block._k_sum_decode = torch.zeros((), dtype=torch.long)
    block._tok_prefill = 0
    block._tok_decode = 0
    bound = types.MethodType(fwd, block)
    return bound(hidden)


def test_equivalence_keep_all():
    """tau/kmin=kmax=8 keep-all must reproduce native forward (both norm modes)."""
    for norm in (True, False):
        block = ToyMoEBlock(norm_topk_prob=norm)
        hidden = torch.randn(1, 5, HDIM)
        ref = _native_reference(block, hidden)
        block.reset_counter()
        for renorm in U.RENORM_MODES:
            out, _ = _apply(block, hidden, policy="top_p_within_topk", threshold=1.0,
                            kmin=K, kmax=K, phase="all", renorm=renorm)
            err = (out - ref).abs().max().item()
            assert err < 1e-5, f"norm={norm} renorm={renorm} keep-all err={err}"
    print("PASS test_equivalence_keep_all")


def test_physical_skip_drops_experts():
    """Dropped assignments must NOT call their expert (top-1 only)."""
    block = ToyMoEBlock()
    hidden = torch.randn(1, 4, HDIM)
    block.reset_counter()
    _apply(block, hidden, policy="top_p_within_topk", threshold=0.0,
           kmin=1, kmax=K, phase="all", renorm="renorm_survivors")
    logits = block.gate(hidden.view(-1, HDIM))
    rw = F.softmax(logits, dim=1, dtype=torch.float)
    _, sel = torch.topk(rw, K, dim=-1)
    top1 = set(sel[:, 0].tolist())
    called = {i for i in range(E) if block.counter[i] > 0}
    assert called <= top1, f"called experts {called} exceed top-1 set {top1}"
    assert sum(block.counter) == hidden.shape[1], f"processed {sum(block.counter)} != {hidden.shape[1]}"
    print("PASS test_physical_skip_drops_experts")


def test_fully_dropped_expert_not_called():
    block = ToyMoEBlock()
    hidden = torch.randn(1, 6, HDIM)
    block.reset_counter()
    _apply(block, hidden, policy="top_p_within_topk", threshold=0.0, kmin=1, kmax=K,
           phase="all", renorm="renorm_survivors")
    logits = block.gate(hidden.view(-1, HDIM))
    _, sel = torch.topk(F.softmax(logits, dim=1, dtype=torch.float), K, dim=-1)
    top1 = set(sel[:, 0].tolist())
    for e in range(E):
        if e not in top1:
            assert block.counter[e] == 0, f"expert {e} ran but no token routed to it as top-1"
    print("PASS test_fully_dropped_expert_not_called")


def test_kept_output_matches_zero_weight_reference():
    """Physical-skip output == 'compute all then zero dropped' (math equivalence)."""
    block = ToyMoEBlock()
    hidden = torch.randn(1, 5, HDIM)
    out, _ = _apply(block, hidden, policy="top_p_within_topk", threshold=0.5,
                    kmin=1, kmax=K, phase="all", renorm="renorm_survivors")
    hs = hidden.view(-1, HDIM)
    rw = F.softmax(block.gate(hs), dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, K, dim=-1)
    rw_norm = rw / rw.sum(dim=-1, keepdim=True)
    keep = U.compute_keep_mask(rw_norm, "top_p_within_topk", 0.5, 1, K)
    base = rw_norm
    surv = (base * keep).sum(-1, keepdim=True).clamp_min(1e-20)
    w = ((base * keep) * (base.sum(-1, keepdim=True) / surv)).to(hs.dtype)
    final = torch.zeros_like(hs)
    mask = F.one_hot(sel, num_classes=E).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        idx, top_x = torch.where(mask[e].squeeze(0))
        cur = hs[None, top_x].reshape(-1, HDIM)
        o = block.experts[e](cur) * w[top_x, idx, None]
        final.index_add_(0, top_x, o)
    ref = final.reshape(1, 5, HDIM)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"kept-output mismatch err={err}"
    print("PASS test_kept_output_matches_zero_weight_reference")


def test_kmin_respected():
    block = ToyMoEBlock()
    hidden = torch.randn(1, 20, HDIM)
    for kmin in (1, 2, 3):
        rw = F.softmax(block.gate(hidden.view(-1, HDIM)), dim=1, dtype=torch.float)
        rw, _ = torch.topk(rw, K, dim=-1)
        rw_norm = rw / rw.sum(-1, keepdim=True)
        for policy, thr in [("top_p_within_topk", 0.0), ("min_weight_cutoff", 1.0),
                            ("max_dropped_mass", 1.0)]:
            keep = U.compute_keep_mask(rw_norm, policy, thr, kmin, K)
            realized = keep.sum(-1)
            assert (realized >= kmin).all(), f"{policy} kmin={kmin} violated: min={realized.min()}"
    print("PASS test_kmin_respected")


def test_monotonicity():
    block = ToyMoEBlock()
    hidden = torch.randn(1, 200, HDIM)
    rw = F.softmax(block.gate(hidden.view(-1, HDIM)), dim=1, dtype=torch.float)
    rw, _ = torch.topk(rw, K, dim=-1)
    rw_norm = rw / rw.sum(-1, keepdim=True)

    def avg_k(policy, thr):
        return U.compute_keep_mask(rw_norm, policy, thr, 1, K).float().sum(-1).mean().item()

    taus = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    ks = [avg_k("top_p_within_topk", t) for t in taus]
    assert all(ks[i] <= ks[i + 1] + 1e-9 for i in range(len(ks) - 1)), f"top_p not monotone up: {ks}"

    cuts = [0.0, 0.05, 0.1, 0.2, 0.4]
    ks = [avg_k("min_weight_cutoff", c) for c in cuts]
    assert all(ks[i] >= ks[i + 1] - 1e-9 for i in range(len(ks) - 1)), f"cutoff not monotone down: {ks}"

    betas = [0.0, 0.1, 0.2, 0.4, 0.6]
    ks = [avg_k("max_dropped_mass", b) for b in betas]
    assert all(ks[i] >= ks[i + 1] - 1e-9 for i in range(len(ks) - 1)), f"beta not monotone down: {ks}"
    print("PASS test_monotonicity")


def test_no_sync_in_forward_source():
    # strip comments so we check actual code, not descriptive comments
    raw = inspect.getsource(U.make_dynamic_forward)
    code = "\n".join(line.split("#", 1)[0] for line in raw.splitlines())
    for bad in (".item(", ".cpu(", ".tolist(", ".numpy("):
        assert bad not in code, f"forward source contains sync primitive {bad}"
    print("PASS test_no_sync_in_forward_source")


def test_prefill_decode_split():
    import types
    block = ToyMoEBlock()
    block._k_sum_prefill = torch.zeros((), dtype=torch.long)
    block._k_sum_decode = torch.zeros((), dtype=torch.long)
    block._tok_prefill = 0
    block._tok_decode = 0
    fwd = U.make_dynamic_forward("top_p_within_topk", 0.5, 1, K, phase="all")
    block.forward = types.MethodType(fwd, block)
    block(torch.randn(1, 7, HDIM))   # prefill-like
    block(torch.randn(2, 1, HDIM))   # decode-like
    assert block._tok_prefill == 7, block._tok_prefill
    assert block._tok_decode == 2, block._tok_decode
    print("PASS test_prefill_decode_split")


def test_strict_parser():
    assert U.parse_strict("The answer is #### 72")[0] == "72"
    assert U.parse_strict("blah #### 1,234")[0] == "1234"
    assert U.parse_strict("x #### 72.0")[0] == "72"
    assert U.parse_strict("#### -5")[0] == "-5"
    assert U.parse_strict("the result is 42 dollars")[1] == "parse_failure"
    assert U.parse_strict("#### 1 then #### 2")[0] == "2"
    assert U.parse_gold("Reasoning...\n#### 18") == "18"
    print("PASS test_strict_parser")


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in ALL:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(ALL) - failed}/{len(ALL)} passed")
    sys.exit(1 if failed else 0)
