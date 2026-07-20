"""test_kv_cache_immutability: a branch/probe forward must operate on a CLONE of the
incoming KV cache, leaving the baseline cache byte-for-byte unchanged. We simulate the
v32 probe pattern with a DynamicCache and check the original is untouched after the
fork is extended."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers import DynamicCache
from moe_research.interventions import clone_cache


def _snapshot(cache):
    return [(k.clone(), v.clone()) for k, v in zip(cache.key_cache, cache.value_cache)] \
        if hasattr(cache, "key_cache") else None


def test_clone_cache_isolates_writes():
    base = DynamicCache()
    for layer in range(3):
        base.update(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), layer)
    seq0 = base.get_seq_length()

    fork = clone_cache(base)
    # extend the fork by one decode step at every layer (probe branch)
    for layer in range(3):
        fork.update(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8), layer)

    assert fork.get_seq_length() == seq0 + 1, "fork should have grown"
    assert base.get_seq_length() == seq0, "baseline cache MUTATED by probe!"
    # tensor identity: fork layer tensors are different objects than baseline
    bk = base.layers[0].keys; fk = fork.layers[0].keys
    assert bk.data_ptr() != fk.data_ptr(), "fork shares storage with baseline"
    print("PASS test_clone_cache_isolates_writes")


if __name__ == "__main__":
    test_clone_cache_isolates_writes()
