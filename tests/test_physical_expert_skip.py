"""test_physical_expert_skip: dropped experts never execute their FFN (not run-then-zero).
We count expert calls and assert only the retained top-K experts are invoked, and the
total processed token-rank pairs equals num_tokens * K."""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from _toy import ToyMoE, attach, HDIM, E
from moe_research import k_policy as KP


def test_dropped_experts_not_executed():
    for k in (1, 2, 4):
        block = ToyMoE(); block.reset()
        pol = KP.KPolicy(prefill_k=k, decode_k=k, weight_mode="renorm_survivors")
        attach(block, pol)
        h = torch.randn(1, 6, HDIM)
        KP._PHASE.set("prefill")
        block(h)
        # which experts are in the top-k of at least one token
        rw = F.softmax(block.gate(h.view(-1, HDIM)), dim=1, dtype=torch.float)
        _, sel = torch.topk(rw, E, dim=-1)
        topk_experts = set(sel[:, :k].flatten().tolist())
        called = {i for i in range(E) if block.counter[i] > 0}
        assert called <= topk_experts, f"k={k} called {called} !subset {topk_experts}"
        assert sum(block.counter) == 6 * k, f"k={k} processed {sum(block.counter)} != {6*k}"
    print("PASS test_dropped_experts_not_executed")


if __name__ == "__main__":
    test_dropped_experts_not_executed()
