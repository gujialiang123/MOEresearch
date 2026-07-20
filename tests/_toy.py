"""Shared toy MoE for unit tests (no 30B weights)."""
import os, sys, types
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_research import k_policy as KP

HDIM, E, K = 16, 8, 8


class CountingExpert(nn.Module):
    def __init__(self, idx, counter):
        super().__init__()
        self.idx = idx; self.counter = counter
        # distinct nonlinear-ish transform per expert so branch outputs differ
        torch.manual_seed(100 + idx)
        self.w = nn.Parameter(torch.randn(HDIM, HDIM) * 0.2)

    def forward(self, x):
        self.counter[self.idx] += int(x.shape[0])
        return torch.tanh(x @ self.w) * float(self.idx + 1)


class ToyMoE(nn.Module):
    def __init__(self, norm=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_experts = E; self.top_k = K; self.norm_topk_prob = norm
        self.gate = nn.Linear(HDIM, E, bias=False)
        self.counter = [0] * E
        self.experts = nn.ModuleList([CountingExpert(i, self.counter) for i in range(E)])

    def reset(self):
        for i in range(E):
            self.counter[i] = 0


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


def bind(block, ctx, layer_idx=0):
    block._kp_ksum_prefill = torch.zeros((), dtype=torch.long)
    block._kp_ksum_decode = torch.zeros((), dtype=torch.long)
    block._kp_tok_prefill = 0; block._kp_tok_decode = 0; block._kp_layer_idx = layer_idx
    fwd = KP._make_moe_forward(ctx)
    block.forward = types.MethodType(fwd, block)


def attach(block, policy, layer_idx=0):
    ctx = KP.PolicyContext(policy, [block], ToyMoE, ToyMoE.forward)
    bind(block, ctx, layer_idx)
    return ctx
