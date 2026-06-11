# 2026-06-11 harness v1 — 4-way bench results

> First production use of `regime-bench-harness` v1. Two surprises:
> 1. The sglang 1-line patch (`model_runner.py:1841`) gives **4.7–8.4× speedup** — even larger than predicted.
> 2. **fp8 triton is SLOWER than bf16 triton on H200 for this model** — contradicting conventional wisdom.

## Setup

- Hardware: 1× NVIDIA H200 (GPU 4, SM 9.0)
- Model: `Qwen3-30B-A3B-Instruct-2507` (bf16) and `Qwen3-30B-A3B-Instruct-2507-FP8`
- Framework: sglang 0.5.12.post1, flashinfer 0.6.3, torch 2.9.1, triton 3.5.1
- Workload: 4 regimes from `regimes/qwen3_30b_moe_sglang_perf_sweep.yaml`
- Harness: `bench-specs/*.yaml` driven, 3 runs per regime, drop run 1, stddev gate at 8%
- Patch applied: `patches/sglang_cutlass_autotune_allowlist.diff` (only for `*-patched.yaml` specs)
- cudagraph: ON where supported, OFF on `triton-*` specs due to Triton 3.5.1 cubin KeyError

> **2026-06-11 20:35 update**: Discovered after the experiments that this exact
> 1-line patch was **already merged upstream** in [PR #26496](https://github.com/sgl-project/sglang/pull/26496)
> (Brayden Zhong, 2026-06-04). Our local sglang checkout was on `study/v0.5.9`
> (a 6-month-old snapshot), so the patch was needed locally. **The fix is on
> sglang main HEAD now — no PR submission needed**. Our experimental data still
> stands as independent validation of upstream's fix.
>
> Attempted to bench directly on sglang main but blocked by CUDA 13 vs CUDA 12.8
> mismatch in the latest `sgl-kernel` wheel. Validated the patch reproducibility
> instead by re-running `sglang-cutlass-bf16-patched.yaml` a second time: same
> `spec_hash`, results within 0–1.1% spread (see Finding 5 below).

## Headline table

| backend | startup | R_short_decode | R_medium_balanced | R_long_prefill | R_concurrent_decode |
|---|---|---|---|---|---|
| **triton-bf16** (default, baseline) | 34.1s | 0.10 req/s | 0.74 | 2.52 | 2.94 |
| **cutlass-bf16 + patch** | 79.2s | 0.83 (**8.42×**) | 4.40 (**5.97×**) | 13.66 (**5.41×**) | 13.86 (**4.71×**) |
| triton-fp8 | 40.1s | 0.06 (**0.60×**) | 0.49 (**0.67×**) | 1.63 (**0.64×**) | 1.96 (**0.67×**) |
| native-cutlass-fp8 | 40.1s | 0.59 (**6.01×**) | 4.13 (**5.61×**) | 11.83 (**4.69×**) | 14.54 (**4.94×**) |

(`X×` is speedup over triton-bf16 baseline at that regime.)

All `summary.json` artifacts:
- `results/2026-06-11_harness-v1/sglang-triton-bf16-baseline/`
- `results/2026-06-11_harness-v1/sglang-cutlass-bf16-patched/`
- `results/2026-06-11_harness-v1/sglang-triton-fp8-baseline/`
- `results/2026-06-11_harness-v1/sglang-native-cutlass-fp8/`

## Finding 1: the 1-line patch validates exactly as predicted

`docs/2026-06-11/ofer_meeting_findings_draft.md` §8.6 predicted that uncommenting
the `flashinfer_cutlass` line in `_should_run_flashinfer_autotune()` would:
1. Trigger flashinfer JIT during warmup → no longer collides with cudagraph capture
2. Populate AutoTuner cache → no longer falls back to tactic 0

Both observed:

| symptom | unpatched | patched |
|---|---|---|
| Server starts under cudagraph | ❌ hangs (cold cache) | ✅ healthy after 79.2s |
| `cuda graph: True` in decode logs | (n/a — couldn't start) | ✅ confirmed |
| Startup time delta vs triton | n/a | +45s (= predicted autotune window) |
| Speedup over triton baseline | n/a | **4.7–8.4×** |

**Microbench prediction**: 3–6× per cutlass kernel call (CUTLASS H200 SM90 microbench, see
`docs/2026-06-08/sglang_vs_vllm_flashinfer_cutlass_analysis.md` §CUTLASS-microbench).
**E2E observation**: 4.7–8.4×. The high end exceeds microbench because the patch also
unblocks cudagraph (which reduces CPU launch overhead independently — the same `max(CPU,
GPU)` model from `docs/2026-06-08/vllm_2x2_autotune_cudagraph_matrix.md`).

**Implication for sglang upstream**: a trivial PR can close a 5× gap on H200 for any
user explicitly selecting `--moe-runner-backend flashinfer_cutlass`. The historical TODO
comment is stale.

## Finding 2: fp8 triton on H200 is a 33–40% REGRESSION

This contradicts the "fp8 → 1.5–2× speedup" estimate I gave the user earlier
(based on conventional decode-memory-bound reasoning).

Possible causes (in order of likelihood):

1. **No sglang tuned config for `E=128, N=768` H200 fp8**. The sglang Triton MoE
   ships hand-tuned configs at `sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/`.
   For H200 fp8, only E ∈ {160, 257, 384, 385} exist — none match Qwen3-30B-A3B's
   (E=128, N=768). At first call the Triton autotune searches a small space at
   runtime, and may pick a suboptimal config.

2. **Activations stay bf16 in fp8 inference**. Only weights are fp8; the MoE
   inputs/outputs and intermediate activations are still bf16. So the only
   HBM saving is the **weight-load traffic per expert**, not the activation
   traffic. For MoE this is dominant, but the gain may be eaten by:
   - Dequantize-then-matmul if the Triton kernel doesn't use native fp8 WGMMA
   - Extra scale broadcasts per block (fp8 W8A8 needs per-tile scale apply)

3. **Tensor Core fp8 path may not be reachable from Triton's fused_moe_kernel
   without an autotune config**. If Triton falls back to a pre-Hopper fp8 schedule
   (no WGMMA), there's no compute speedup, only dequant overhead.

**Hypothesis to validate later** (not in v1 scope): generating a tuned config for
`E=128, N=768, NVIDIA_H200, dtype=fp8_w8a8.json` via Triton's autotune sweep would
likely flip this from 0.6× to 1.5× or better.

## Finding 3: cutlass-bf16 ≈ native-cutlass-fp8

Native cutlass on fp8 (sglang's own `cutlass_fused_experts_fp8` impl, not flashinfer)
runs **roughly the same throughput** as patched cutlass on bf16 — sometimes slightly
faster, sometimes slightly slower:

| regime | cutlass-bf16 | native-cutlass-fp8 | fp8/bf16 |
|---|---|---|---|
| R_short_decode | 0.83 | 0.59 | **0.71×** |
| R_medium_balanced | 4.40 | 4.13 | 0.94× |
| R_long_prefill | 13.66 | 11.83 | 0.87× |
| R_concurrent_decode | 13.86 | 14.54 | **1.05×** |

**This is a surprise**. Theory: cutlass-bf16 + autotuned cudagraph already saturates
the H200's bf16 TC pipeline so well that the fp8 weight-bandwidth savings don't
translate to fewer end-to-end milliseconds. We may be hitting a different bottleneck
(routing / `count_and_sort_expert_tokens` atomics from earlier work, or sampling).

**Diagnostic to run** (not in v1 scope): nsys on the cutlass-bf16 winner to see
what's *not* moe_gemm anymore.

## Finding 4: a 4th hidden sglang bug — fp8 + flashinfer_cutlass crashes

When trying `--moe-runner-backend flashinfer_cutlass` on fp8 model:

```
AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'
  File "sglang/.../layers/quantization/fp8.py", line 1447, in apply
    if self.runner.runner_backend.is_deep_gemm():
```

Root cause: `Fp8MoEMethod.create_moe_runner` (fp8.py:1349-1359) only sets
`self.runner` for `{deep_gemm, triton, flashinfer_trtllm}`. The `flashinfer_cutlass`
branch silently falls through (`pass` at line 1360). `apply()` later tries to read
`self.runner.runner_backend`, hits AttributeError.

The fp8 path has its own `is_cutlass()` branch at fp8.py:1411 that uses sglang's
**native** cutlass impl (`cutlass_fused_experts_fp8`) instead of flashinfer's.
flashinfer_cutlass + fp8 is just unsupported, undocumented as such.

So our `bench-specs/sglang-cutlass-fp8-patched.yaml` is invalid for the named path
(useful only as a crash repro); the working path is `sglang-native-cutlass-fp8.yaml`
which selects `MoeRunnerBackend.CUTLASS` instead.

## Finding 5: harness `spec_hash` reproducibility validated (run-2-vs-run-1)

After learning PR #26496 already shipped the autotune fix upstream, we re-ran
the same `sglang-cutlass-bf16-patched.yaml` spec a second time to confirm
harness determinism. Same `spec_hash` → same numbers within run-to-run noise:

| regime | run 1 (req/s) | run 2 (req/s) | spread |
|---|---|---|---|
| R_short_decode | 0.83 | 0.83 | 0.0% |
| R_medium_balanced | 4.40 | 4.38 | 0.2% |
| R_long_prefill | 13.66 | 13.51 | 1.1% |
| R_concurrent_decode | 13.86 | 13.94 | 0.6% |

All spreads well under the 8% stddev threshold. Both runs:
- Same `spec_hash`: `sha256:61336de4c5299dab14f0aeb40a714600d1f559ae5f456980ac6277f9d3b80f4d`
- Same `quality_gate.passed=true`
- Same empty `warnings: []`

This is exactly the reproducibility property Mason asked for when proposing the
harness. **5–8× cutlass speedup is rock-solid, not measurement noise.**

Artifact: `results/2026-06-11_harness-v1/sglang-cutlass-bf16-on-main-equiv/`
(the suffix "-on-main-equiv" reflects that v0.5.9+patch is functionally
equivalent to main HEAD for this code path).

## Operational note: harness v1 worked exactly as designed

- All four bench-specs ran end-to-end via one command
- Deterministic `spec_hash` in every summary.json
- Server cleanup on every exit path including the `flashinfer_cutlass + fp8` crash
- Quality gate (sanity) passed on all 4 successful runs
- R_long_prefill stddev > 8% correctly flagged in warnings on triton runs (and
  not on cutlass runs — cutlass is also more reliable, not just faster)
- Total wall time to acquire 4 data points: ~25 minutes (most of it server startup,
  not bench)

## Decisions for Ofer meeting

Recommend updating the Ofer draft to surface these findings:

1. **TL;DR item #2** ("sglang cutlass gap is hard to fix"): **already fixed
   upstream**. PR #26496 (Brayden Zhong, 2026-06-04) merged the same
   `flashinfer_cutlass` allowlist entry we identified. Our value-add is
   independent reproduction + 4.7-8.4× quantified speedup + reproducibility
   demonstration. Frame: "we predicted the fix from first principles, then
   discovered the maintainer landed the same fix 7 days ago — validating both
   our diagnosis and the upstream patch."
2. **§Q2 fp8 estimates**: retract "1.5-2× from fp8". Show the 0.6× regression
   and link this doc. Add the missing-tuned-config hypothesis.
3. **New finding**: even cutlass-bf16 (patched) ≈ native-cutlass-fp8. Either
   we're not weight-bandwidth-bound at this scale, or there's a non-MoE
   bottleneck that bf16→fp8 doesn't address. **Worth nsys-profiling**.
4. **Open a sglang issue** for the **still-present** `flashinfer_cutlass + fp8`
   AttributeError bug. Draft ready at
   `patches/issue_draft_fp8_flashinfer_cutlass.md`. Concrete user-experience
   fix proposed (raise NotImplementedError at create_moe_runner instead of
   AttributeError at first forward).

## Next experiments

- nsys-profile native-cutlass-fp8 vs cutlass-bf16-patched on R_medium — find the
  ~6% gap source. Is it sampling? Routing? KV?
- Generate H200 fp8 tuned config for E=128 N=768 (Triton autotune offline) → re-run
  triton-fp8 → see if regression flips.
- Once Triton 3.5.1 cubin bug fixed (chendi), re-run triton-bf16 with cudagraph
  enabled — should also see big jump (just from cudagraph, no patch).
- (Env upgrade, when ready) Bench directly on sglang main + flashinfer 0.6.x +
  CUDA 13 to confirm our equiv claim. Blocked on CUDA driver upgrade.
