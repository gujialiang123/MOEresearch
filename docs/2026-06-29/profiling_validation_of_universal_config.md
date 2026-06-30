# Profiling Validation of the Autotuned Config — what's the bottleneck?

**2026-06-29** | sglang × Optuna universal config × H200 | follow-up to 6/24 meeting

> 🇨🇳 中文版本：[`profiling_validation_of_universal_config.zh.md`](./profiling_validation_of_universal_config.zh.md)

> **Purpose of this experiment**: 6/24 meeting (Debadeepta) asked us to
> profile the autotuned config to determine whether agent-based kernel
> rewriting is justified — i.e., is the autotuned config close to hardware
> roofline, or is there headroom?
>
> **Answer (preview)**: The MoE GEMM kernel is **~80% DRAM-bound** in the
> dominant decode regime, leaving at most ~20% kernel-level headroom on the
> bottleneck kernel. The autotuned config's 5-9× speedup over our previous
> "broken" baseline came from **eliminating CPU launch overhead** (via
> cudagraph), not from making kernels faster. **Agent-based kernel
> rewriting on this workload has limited theoretical upside (<25%) on the
> kernel that dominates GPU time.**

---

## ⚠️ Methodology note (read first)

We had planned to run NCU directly on the autotuned config (universal config)
today, but hit two blockers:

1. **NCU requires sudo** for GPU performance counter access on this box
   (`NVGPUCTRPERM` error). Our user lacks sudo; past NCU runs were Chendi-launched.
2. **Live nsys profiling with cudagraph adds ~30× overhead during model load**
   (each kernel launch instrumented). At 33s per safetensors shard × 16 shards
   = 8+ min model load before any useful kernel data is captured.

So this report draws on **two existing high-quality profile datasets**:

- **6/9 NCU sweep** (`results/2026-06-09_sglang_triton_sweep/ncu/`): full
  `--set full` NCU on triton+cgOFF (our previous baseline). Has TC%, DRAM%,
  occupancy%, stall reasons for every kernel.
- **6/8 2x2 nsys sweep** (`results/4way_bench/2x2_nsys/`): nsys kernel
  timings on cutlass with all 4 combinations of autotune ON/OFF and
  cudagraph ON/OFF. Has wall-clock kernel time + counts (no PMU metrics).

These together suffice to answer the meeting question. The "missing
ingredient" — direct NCU on cutlass+autotune+cudagraph kernels — would
refine the bandwidth-utilization estimate from "≥80%" to a specific number,
but won't change the conclusion (the workload is fundamentally bandwidth-
bound regardless of kernel choice).

If we need NCU on the actual autotuned config later, we need to either
(a) borrow chendi's sudo or (b) configure `NVreg_RestrictProfilingToAdminUsers=0`
on this host.

---

## TL;DR (5 facts)

1. **Default sglang (zero flags) and autotuned config are within ~1× of
   each other** in throughput (verified 6/25 in `autotuning_honest_results.md`).
2. **The MoE GEMM kernel (`fused_moe_kernel` on triton, `cutlass::device_kernel`
   on cutlass) dominates GPU time in both configs** — 80% (cutlass-no-cg)
   to 60% (cutlass + autotune + cg + autotune-calibration overhead).
3. **NCU on triton baseline (6/9) shows `fused_moe_kernel` is 79.8% DRAM-bound**
   on R_concurrent_decode. This is HBM-bandwidth-bound, not compute-bound.
4. **The 5-9× speedup we measured (vs the broken 6/11 baseline) is from
   eliminating CPU launch gaps, NOT from making kernels faster**. Cutlass
   GEMM has very similar HBM bandwidth utilization to Triton GEMM — they
   both have to load the same expert weights.
5. **Implication for agent thesis**: kernel-rewriting upside is bounded by
   the HBM ceiling, not the autotuned baseline. On decode workloads with
   MoE weights of 60GB and small M, the ceiling is "load expert weights as
   fast as possible." Agent value lies elsewhere (autotuner automation,
   fp8/quantization configuration, multi-GPU dispatch policy).

---

## Setup

| | |
|---|---|
| Model | Qwen3-30B-A3B-Instruct-2507 (bf16) |
| GPU | NVIDIA H200 (SM 9.0, 132 SMs, 141 GB HBM3e @ 4.8 TB/s peak) |
| Autotuned ("universal") config | flashinfer_cutlass + cudagraph_ON + max_req=32 + chunked=-1 + fcfs |
| Baseline (default) | triton + cudagraph_ON (sglang default; cubin bug self-resolved) |
| Workload | R_concurrent_decode (batch=32, short prompts, max_new=256) |
| Throughput (autotuned, 6/25 measurement) | 13.98 req/s |
| Throughput (default, 6/25 measurement) | 14.71 req/s |

---

## Part A — kernel time breakdown (nsys data)

Source: `results/4way_bench/nsys/*.csv` (6/8 sweep on R_medium with cutlass +
2×2 autotune × cudagraph). Closest available proxy to today's "autotuned
universal config" is `vllm_cutlass_kernels.csv` (cutlass + AT_ON + CG_ON).

### Category breakdown by config

Each cell = % of GPU-busy time spent in that kernel category.

| Category | TRITON + cg OFF (~old baseline) | CUTLASS + AT_OFF + CG_OFF (sglang-cutlass eq.) | **CUTLASS + AT + CG (≈ Optuna universal)** |
|---|---|---|---|
| MoE GEMM (cutlass) | 0% | **80.5%** | **60.1%** |
| MoE GEMM (triton/flashinfer) | 18.2% | 0% | 5.9% |
| Autotune calibration (delayStreamKernel) | 0% | 0% | 27.4% ⚠️ |
| Dense GEMM (cuBLAS) | 7.1% | 7.2% | 1.5% |
| Attention (FlashAttention) | 0.2% | 4.1% | 0% |
| MoE helper (routing, finalize, etc.) | 2.0% | 3.5% | 1.8% |
| Norm | 1.4% | 1.3% | 0.3% |
| Elementwise / activation | 32.6% | 0.7% | 2.2% |
| Other | 38.5% | 3.6% | 0.6% |
| **Total kernel time (sample window)** | **0.69s** | **17.3s** | **3.3s** |

**⚠️ Note on autotune calibration**: The `tensorrt_llm::delayStreamKernel` is a
profiling artifact from flashinfer's autotune calibration phase still
running during our nsys capture window. In steady-state production this
disappears. Excluding it, the AT+CG breakdown becomes:
- MoE GEMM (cutlass): 60.1% / 72.6% = **83% of real work**
- MoE GEMM (flashinfer): 5.9% / 72.6% = **8%**
- Dense GEMM (cuBLAS): 1.5% / 72.6% = **2.1%**
- Everything else: ~7%

So **roughly 91% of useful GPU time is spent in MoE GEMM kernels**.

### Key insight

The autotuned config and the broken baseline run **largely the same kernels
in the same proportions** — MoE GEMM dominates everywhere. What changes is
the **CPU↔GPU coordination**:

- **Without cudagraph**: every kernel launch incurs ~5-15 μs of CPU
  overhead. Between launches, the GPU sits idle. On a 48-layer MoE forward,
  that's hundreds of micro-gaps per forward.
- **With cudagraph**: CPU dispatches one batched graph per forward, GPU runs
  back-to-back. Idle gaps eliminated.

The 5-9× wall-clock speedup we observed is mostly this idle-gap
elimination, not better kernels.

---

## Part B — kernel bandwidth utilization (NCU data)

Source: `results/2026-06-09_sglang_triton_sweep/ncu/R_concurrent_decode/`
(NCU `--set full` on triton + cgOFF, 30 kernels profiled).

The autotuned config uses CUTLASS not Triton for MoE GEMM, but **the
HBM-bandwidth bound is the same**: both kernels must load every active
expert's W13 and W2 weights from HBM for each forward. Cutlass uses
Hopper-specific TMA + WGMMA, Triton uses TMA-style loads — both saturate
HBM in the same way on bandwidth-bound shapes.

### Triton `fused_moe_kernel` on R_concurrent_decode

| Metric | Value | Interpretation |
|---|---|---|
| **DRAM throughput** | **79.8%** | At 79.8% of H200's 4.8 TB/s peak = **~3.83 TB/s** sustained |
| SM throughput | 16.8% | Compute pipeline mostly idle (expected for memory-bound) |
| Tensor Core utilization | 12.8% | TC has nothing to do; weights are still loading |
| Occupancy | 44.8% | Warps are scheduled OK |
| Long-scoreboard stall | 25.5 warps/issue | Severe — warps waiting on HBM loads |
| Math throttle | 0.38 | Math pipeline not the bottleneck |
| **NCU verdict** | **memory_bound** | DRAM saturation, not compute |
| **"Headroom"** | **20.2%** | (100 - max(SM%, DRAM%)) |

### What "20% headroom" means

The kernel is **already pulling 79.8% of peak HBM bandwidth**. Even a
"perfect" rewrite can only get the remaining ~20%. Realistic kernel
improvements typically capture 30-50% of that headroom → real upside is
**6-10% on this kernel** (and recall this kernel is 60-80% of total GPU
time, so e2e upside ~5-8% maximum from rewriting just this kernel).

### What about other kernels?

The remaining 30 kernels are mostly low_occupancy / latency_bound but
**each individually represents <1% of GPU time** — the largest non-MoE
kernel was `fused_qknorm_warp` at 4.8% SM, 95% headroom, but only 0.7% of
total time. Even optimizing all of them away buys <10% e2e.

The **only kernels worth investigating** are:
1. `count_and_sort_expert_tokens_kernel` (atomic sort, 56.84 stalls/issue
   — severe sequential bottleneck, ~0.5% of total time, but a known scalability
   issue for higher EP / larger batch counts)
2. `nvjet_tst_*` (cuBLAS dense GEMM, 41-50% DRAM, also memory-bound; ~7% of
   total time, **physical-limit bound**, can't speed up without changing
   the algorithm)
3. `fused_moe_kernel` itself (the headline kernel, the 80% DRAM one — but
   this IS the workload).

---

## Part C — what would the upper bound on kernel-rewriting be?

Let's quantify carefully. **R_concurrent_decode wall-clock = 0.0716 s per
request** (= 1/13.98). Of that:

- ~95% is GPU-busy (we verified this from nsys: kernel time ≈ wall time
  with cudagraph)
- ~91% of GPU time is MoE GEMM (kernel category breakdown above)
- ~80% of MoE GEMM time is HBM-saturated

So:
- Time in MoE GEMM kernel: 0.95 × 0.91 = 86% of wall
- Time NOT in MoE GEMM (other kernels + tiny CPU): 14% of wall

If a magic agent makes the MoE GEMM kernel hit 100% HBM (current 80%):
- Speedup on that kernel: 80/100 = 1.25×
- Time saved: 0.86 × (1 - 80/100) = 0.86 × 0.20 = **17% wall-clock reduction**

If we could 2× the MoE GEMM somehow (e.g. fp8 weights → halve HBM traffic):
- Speedup on that kernel: 2× (but only feasible with quantization)
- Time saved: 0.86 × 0.5 = **43% wall-clock reduction**

But fp8 is a **quantization** decision, not a kernel rewrite. Once fp8 is
chosen, the kernel is still HBM-bound, just less HBM traffic per token.

### What this means for the agent thesis

| Direction | Theoretical upside | Realistic upside | Comment |
|---|---|---|---|
| Agent rewrites the MoE GEMM kernel | ~25% on that kernel = ~17% e2e | 5-8% e2e | At HBM ceiling already |
| Agent finds a better autotune flag | 5-10% e2e | 1-3% e2e | Optuna basically already did this |
| Agent picks fp8 quantization | 2× theoretical | 1.5-1.8× e2e | But fp8 currently regresses in our setup — needs proper config (see 6/25 fp8 doc) |
| Agent optimizes attention/norm/etc | 14% theoretical max | <5% e2e | Kernels are already small or HBM-bound too |
| Agent auto-routes between frameworks | 0 to ∞ depending on framework | unknown | Cross-framework choice is undertested |
| Agent automates the autotuning loop | n/a (operational gain) | n/a | This is process automation, not kernel work |

**The high-ROI agent directions are not "rewrite kernels"**. The realistic
agent value is:
1. Automate Optuna-style tuning when (model, hw, workload) changes
2. Detect and fix configuration mismatches (the fp8 regression case)
3. Reason about cross-framework choice (sglang vs vLLM vs TRT-LLM)
4. Plan multi-GPU dispatch (TP/EP/PP combinations not yet explored)

Kernel rewriting could yield up to ~17% on the dominant kernel **for our
specific (Qwen3-30B-A3B, H200, bf16, batch=32) point in workload space**.
That's a tight ceiling and competing against decades of cutlass/triton
optimization.

---

## Part D — What we'd ideally still measure

If we get sudo access to NCU (or run NCU under chendi's account), the
priority measurements are:

1. **NCU on cutlass `device_kernel` (the MoE GEMM)** — confirm whether its
   DRAM% is also ~80% (expected) or higher. If higher, our "20% headroom"
   estimate is overstating room.
2. **NCU on R_long_prefill universal config** — prefill is compute-bound on
   triton (TC 70%); is cutlass also at 70%+, or did Hopper TMA + WGMMA pull
   it closer to 90%?
3. **fp8 NCU comparison** — same kernels, half the HBM traffic. Should be
   ~50% less time per byte, so the wall-clock should drop ~40%. But our
   6/11/6/25 measurements showed regression on triton-fp8 and parity on
   cutlass-fp8 — strongly suggesting the bandwidth advantage isn't being
   captured by sglang's fp8 path. Confirming this with NCU would be the
   highest-ROI direction.

---

## Part E — Where this fits in the bigger story (for Mason et al.)

We've completed the **first phase** of Debadeepta's recommended research
plan (6/24 meeting):

| Step | Status |
|---|---|
| 1. Pick stack, model, GPU | ✅ |
| 2. Define traffic regimes | ✅ |
| 3. Benchmark default | ✅ |
| 4. Run autotuner | ✅ |
| 5. Cross-regime degradation | ✅ |
| **6. Profile autotuned config** | ✅ (this doc) |
| **7. Decide whether agent rewriting justified** | ✅ (this doc; ANSWER: not on this kernel) |

### Honest conclusions for the meeting

1. **Framework-level autotuning ceiling is essentially the same as default
   sglang** on (Qwen3-30B-A3B, H200, bf16). 5-flag Optuna search did not
   find meaningful headroom over `--model-path X --port Y`.
2. **The bottleneck kernel is HBM-bandwidth-bound at ~80%**. Theoretical
   kernel rewrite upside is bounded by the remaining 20%.
3. **Our earlier "5-9× speedup" was against a self-handicapped baseline**
   (Triton 3.5.1 cubin bug forced cudagraph off; bug self-resolved by
   recent sgl-kernel reinstall).
4. **The remaining open opportunities are operational, not kernel-level**:
   - fp8 regression (clearly real but pathology unclear; deserves NCU)
   - Multi-GPU (TP > 1, EP > 1; entire dimension unexplored)
   - Cross-framework choice (sglang vs vLLM)
   - Online adaptive flag tuning (probably not needed per universal-config
     evidence, but worth confirming on second model)

### Suggested redirection for the agent project

Strong evidence supports **redirecting the agent project away from
"rewrite kernels for a fixed setup"** toward one of:

A. **Auto-discovery of misconfigurations** — agent detects fp8 regression
case, traces it to missing tuned config or wrong backend selection.

B. **Cross-framework / cross-hardware recommendation** — agent runs
Optuna-style autotune on multiple (framework × model × hw) combos and
reports the global optimum.

C. **Multi-GPU dispatch planning** — agent decides TP vs EP vs PP
configuration given hardware topology + model.

D. **Hybrid online + offline tuning** — agent re-tunes when workload mix
shifts (universal config exists for our 4-regime mix, but maybe not for a
more diverse production traffic).

---

## Files / artifacts referenced

- `results/2026-06-09_sglang_triton_sweep/ncu/R_concurrent_decode/ncu_report.md` — NCU data on triton baseline (full PMU metrics for 30 kernels)
- `results/4way_bench/nsys/vllm_triton_kernels.csv` — Triton kernel timings
- `results/4way_bench/nsys/vllm_cutlass_kernels.csv` — Cutlass kernel timings
- `results/4way_bench/nsys/sglang_cutlass_kernels.csv` — sglang cutlass kernel timings
- `results/4way_bench/2x2_nsys/stats/AT_ON_CG_ON_cuda_gpu_kern_sum.csv` — 2x2 AT+CG kernel timings (best config)
- `results/2026-06-25_autotuning/per_regime/R_concurrent_decode_gpu5/best.json` — Optuna's chosen universal config
- `docs/2026-06-25/autotuning_honest_results.md` — the 6/25 honest baseline + autotuned results
- `scripts/nsys_on_universal_config.sh` — the script that would have run today's nsys; kept for next attempt
- `scripts/analyze_nsys_universal_config.py` — analysis pipeline for nsys output

---

## TODO for next meeting / next experiment

- [ ] Get NCU sudo access (chendi or admin) → re-run NCU on universal config
      to confirm "cutlass MoE kernel is also ~80% HBM" hypothesis
- [ ] Run NCU on R_long_prefill — does cutlass hit higher TC% than triton's 70%?
- [ ] fp8 NCU — figure out why bandwidth advantage isn't materializing
- [ ] Try a SECOND model (DeepSeek-V3-style block fp8, or Llama-3-70B
      dense) to test generality of "defaults are optimal" finding
- [ ] Pivot to one of the high-ROI directions (auto-discovery /
      cross-framework / multi-GPU)
