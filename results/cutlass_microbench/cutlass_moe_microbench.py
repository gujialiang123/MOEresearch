"""
Microbenchmark: directly call flashinfer.cutlass_fused_moe on Qwen3MoE shapes.

Goal: answer two questions:
(1) Does running with autotune() context give faster kernel than fallback (tactic=-1)?
    → If YES: hand-tuned tactic matters. Adding more candidates to search space is worth trying.
    → If NO: existing tactic ≈ fallback. Adding more candidates won't help.

(2) Across batch sizes, how much spread exists between best and worst tactic?
    → Wide spread: tuning matters, search space could matter
    → Narrow spread: all tactics roughly same, tuning doesn't matter much

Setup: Qwen3-30B-A3B-Instruct-2507 shape (E=128, N=768 intermediate, K=2048 hidden, top_k=8)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import time
import statistics

# Imports (must be after CUDA_VISIBLE_DEVICES)
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.autotuner import AutoTuner, autotune

# Qwen3-30B-A3B shape
E = 128       # num_experts
K = 2048      # hidden_size
N = 768       # intermediate_size (per expert)
TOP_K = 8

device = torch.device("cuda:0")
dtype = torch.bfloat16

def make_inputs(B):
    """Construct inputs for batch_size=B tokens."""
    hidden_states = torch.randn(B, K, device=device, dtype=dtype) * 0.02
    # w13 = [gate, up] concat — shape (E, 2*N, K)
    w13 = torch.randn(E, 2*N, K, device=device, dtype=dtype) * 0.02
    # w2 = down proj — shape (E, K, N)
    w2 = torch.randn(E, K, N, device=device, dtype=dtype) * 0.02
    # routing: for each token, pick top_k expert ids and weights
    routing_logits = torch.randn(B, E, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(routing_logits, dim=-1), TOP_K, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int)
    return hidden_states, w13, w2, topk_weights, topk_ids

def bench(B, n_iter=20, warmup=5, with_autotune=False):
    """Time cutlass_fused_moe for batch_size=B."""
    h, w13, w2, tw, tids = make_inputs(B)
    
    def call():
        return cutlass_fused_moe(
            input=h,
            token_selected_experts=tids,
            token_final_scales=tw,
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            output_dtype=h.dtype,
            quant_scales=None,
            tune_max_num_tokens=8192,
        )
    
    if with_autotune:
        with torch.inference_mode(), autotune():
            for _ in range(3):
                call()
    
    # Warmup outside autotune
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    
    # Measure
    t0 = time.perf_counter()
    for _ in range(n_iter):
        call()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter * 1000  # ms
    return elapsed

# Test sizes — cover both small (decode-like) and large (prefill-like)
sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

print(f"\n{'='*80}")
print(f"Qwen3-30B-A3B shape: E={E}, K={K}, N={N}, top_k={TOP_K}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*80}\n")

# First pass: fallback only (no autotune ctx → AutoTuner returns -1 = fallback tactic)
print("Phase 1: WITHOUT autotune (uses fallback tactic = -1)")
print(f"{'batch':>6} {'ms':>10} {'tok/ms':>10}")
fallback_results = {}
for B in sizes:
    try:
        ms = bench(B, with_autotune=False)
        tok_per_ms = B / ms
        print(f"{B:>6} {ms:>10.3f} {tok_per_ms:>10.2f}")
        fallback_results[B] = ms
    except Exception as e:
        print(f"{B:>6} FAILED: {e}")
        fallback_results[B] = None

# Reset AutoTuner cache
AutoTuner.get().profiling_cache.clear()

# Second pass: with autotune
print("\nPhase 2: WITH autotune(True) (sweeps all candidates, picks best)")
print(f"{'batch':>6} {'ms':>10} {'tok/ms':>10} {'speedup_vs_fallback':>20}")
tuned_results = {}
for B in sizes:
    try:
        ms = bench(B, with_autotune=True)
        tok_per_ms = B / ms
        sp = fallback_results.get(B, None)
        sp_str = f"{sp/ms:.2f}x" if sp else "?"
        print(f"{B:>6} {ms:>10.3f} {tok_per_ms:>10.2f} {sp_str:>20}")
        tuned_results[B] = ms
    except Exception as e:
        print(f"{B:>6} FAILED: {e}")
        tuned_results[B] = None

print(f"\n{'='*80}")
print("SUMMARY: tuned vs fallback speedup")
print(f"{'batch':>6} {'fallback ms':>14} {'tuned ms':>12} {'speedup':>10}")
for B in sizes:
    f = fallback_results.get(B)
    t = tuned_results.get(B)
    if f and t:
        print(f"{B:>6} {f:>14.3f} {t:>12.3f} {f/t:>9.2f}x")
