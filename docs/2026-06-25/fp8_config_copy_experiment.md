# FP8 Tuned Config Copy Experiment — Negative Result

> **2026-06-25** — Hypothesis tested + rejected. Copying vLLM's tuned config
> closed only ~10% of the fp8 vs bf16 gap; root cause is deeper than missing config.

## Hypothesis

In our 6/11 4-way bench, sglang triton fp8 was 0.6× of bf16 baseline (a
**regression**). We hypothesized this was because sglang's hand-tuned config
table is missing `E=128, N=768, NVIDIA_H200, dtype=fp8_w8a8, block_shape=[128, 128]`
for our model shape. vLLM ships this exact file in its own config dir.

**Predicted fix**: Copy vLLM's config into sglang's `triton_3_3_1/` dir → fp8
performance should jump from 0.6× to 1.2-1.5× of bf16 (since fp8 weights halve
HBM traffic for memory-bound decode).

## Experiment

1. Source: `vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`
2. Target: `sglang/.../fused_moe_triton/configs/triton_3_3_1/E=128,N=768,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`
   - Note naming difference: vLLM uses `[128,128]` (no space), sglang uses
     `[128, 128]` (with space). Renamed during copy.
3. Re-ran `bench-specs/sglang-triton-fp8-baseline.yaml` (same harness, same regimes).

## Result — negative

| regime | bf16 triton | fp8 triton (no config) | fp8 triton (copied config) | improvement | vs bf16 |
|---|---|---|---|---|---|
| R_short_decode | 0.10 | 0.06 | 0.06 | +1.08× | **0.65×** (still regression) |
| R_medium_balanced | 0.74 | 0.49 | 0.55 | +1.12× | **0.75×** |
| R_long_prefill | 2.52 | 1.63 | 1.65 | +1.02× | **0.66×** |
| R_concurrent_decode | 2.94 | 1.96 | 1.98 | +1.01× | **0.67×** |

Closing only 1-12% of the gap. **Hypothesis rejected**.

## Revised hypothesis

The Triton fp8 block path has a fundamental performance issue on H200 beyond
tuned config selection. Possible causes:

1. **Triton's fp8 WGMMA codegen on Hopper may be suboptimal**. Triton supports
   fp8 on Hopper but the inner loop schedule may not match what the model
   needs (e.g., per-block scale broadcasting in the inner loop has overhead).

2. **Block fp8 (128×128 grouped scales) adds inner-loop scale broadcasting**.
   For each tile compute, Triton must load + broadcast the per-block scale.
   Hopper's fp8 WGMMA doesn't natively handle per-block scaling — Triton
   emits explicit scale-apply instructions, hurting throughput.

3. **Tuned configs are batch-size-keyed, but Triton autotune still
   compiles new SASS variants for first-seen shapes**. The "tuned config" only
   sets BLOCK_SIZE/num_warps; the compilation step still happens.

4. **Activation type mismatch**: Qwen3-30B-A3B-FP8 uses dynamic activation
   scaling; vLLM's tuned config might have been profiled with a different
   activation scheme (per-tensor or static), making it suboptimal here.

5. **Different MoE pipeline overhead**. The Triton fp8 path in sglang has
   `Fp8MoEMethod.apply()` dispatch chain; vLLM has its own. Even with same
   kernel config, the framework wrapper overhead differs.

## Implication

This is **a useful finding for the autotuning ceiling discussion**:

- "Just provide tuned config" is **not** sufficient for fp8 on H200.
- Reaching parity with bf16 likely requires either (a) actual Triton kernel
  improvements (not just config selection) or (b) using `cutlass` MoE runner
  backend instead of `triton` for fp8 (as our 6/11 native-cutlass-fp8 result
  showed: 5-6× over triton-bf16 baseline).

**So for fp8 on H200**, the right framework-level decision is:
`--moe-runner-backend cutlass` (not triton), and the gap vs `triton-bf16`
flips from 0.67× to 5-6×.

## What this means for the broader autotuning study

When v1 Optuna runs with bf16 search space, fp8 is held constant (we're not
sweeping quantization). But the lesson generalizes:

> **Framework-level autotuning can recover huge gains when the right backend
> is selected**. It can NOT fix fundamental kernel-level deficiencies.

This is exactly the kind of "ceiling test" Debadeepta asked for: we've shown
that **for fp8 + triton, autotuning bottoms out**. Whether that bottom is at
hardware-roofline is a separate question (would need NCU on fp8 triton kernel).

## Artifacts

- Original config (kept): `patches/vllm_h200_fp8_config_for_sglang.json` (TODO: stash it)
- Re-run result: `results/2026-06-25_autotuning/fp8-config-fixed/summary.json`
- spec_hash: see file
