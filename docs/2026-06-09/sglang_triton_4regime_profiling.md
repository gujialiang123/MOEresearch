# sglang Triton MoE — 4-regime nsys + ncu profiling sweep
## 2026-06-09 evening run

> **Status**: in-progress. NCU batch running in background under `/tmp/ncu_batch/`.
> nsys + bench completed. NCU expected to finish in ~3 hours.

**Mission**: User asked for a per-regime nsys + ncu deep profile of sglang's
default Triton MoE backend on Qwen3-30B-A3B, with cudagraph DISABLED for clean
per-kernel visibility. ncu run with **full metric set** and **no kernel filter**
to "include all nsys kernels".

---

## 1. Setup

| Item | Value |
|---|---|
| Framework | sglang 0.5.9 |
| Model | Qwen3-30B-A3B-Instruct-2507 (bf16) |
| Hardware | NVIDIA H200 (GPU 1), SM 9.0, 132 SMs |
| MoE backend | `--moe-runner-backend triton` (default) |
| cudagraph | **DISABLED** (`--disable-cuda-graph`) so every kernel launch is visible |
| TP / mem / max-seq | 1 / 0.85 / 32 |

## 2. The 4 regimes (after dropping low-signal ones)

From `regimes/qwen3_30b_moe_sglang_perf_sweep.yaml` — picked to span the
performance-relevant axes (expert utilization × prefill-decode mix × concurrency):

| Regime | num_prompts | prompt_words | max_new | concurrency | What it tests |
|---|---|---|---|---|---|
| `R_short_decode`      |  8 |  100 | 256 |  1 | Very low expert utilization (batch=1 → 8 of 128 experts get 1 token) |
| `R_medium_balanced`   | 16 |  800 | 256 |  8 | Typical batch=8 — most experts active |
| `R_long_prefill`      |  4 | 4000 |  32 |  4 | Prefill-dominated, attention kernels visible |
| `R_concurrent_decode` | 32 |  200 | 256 | 32 | High concurrency decode — MoE batch behavior |

**Regimes intentionally dropped**: short-input + short-output (no extra signal vs R_short_decode); generic concurrency sweep middle points (do not differentiate kernels).

---

## 3. e2e-bench-runner (Phase 1)

Used `e2e-bench-runner` skill with `--regimes-file regimes/qwen3_30b_moe_sglang_perf_sweep.yaml`:

| Regime | req/s mean | tokens/s mean | stddev % | reliable? |
|---|---|---|---|---|
| R_short_decode      |  0.11 |   28 |  0.3% | ✅ |
| R_medium_balanced   |  0.80 |  205 |  0.9% | ✅ |
| R_long_prefill      |  2.74 |   88 | 10.3% | ❌ stddev > 8% |
| R_concurrent_decode |  3.20 |  820 |  1.5% | ✅ |

**Observation**: `R_long_prefill` is unreliable per the skill's stddev gate.
For prefill-dominated regimes, the kernel-level data is still meaningful (one
forward pass dominates), but throughput comparisons across runs should not be
trusted.

---

## 4. nsys per-regime (Phase 2)

Sglang launched **once** under nsys; all 4 regime workloads ran back-to-back;
single 200 MB .nsys-rep sliced into 4 windows by wall-time alignment (see
`regime_windows_aligned.json`). nsys-timeline-sql skill applied per window.

| Regime | gpu_active ms | gpu_util % | top_kernel | top % of active | launch ratio (graph/eager) |
|---|---|---|---|---|---|
| R_short_decode      | 7739 |  8.5% | `fused_moe_kernel` | 31.5% | 0.000 (all eager — cudagraph off) |
| R_medium_balanced   | 3091 | 11.6% | `fused_moe_kernel` | 47.7% | 0.000 |
| R_long_prefill      |  214 | 12.1% | `fused_moe_kernel` | 47.4% | 0.000 |
| R_concurrent_decode | 2101 | 15.4% | `fused_moe_kernel` | 54.4% | 0.000 |

**Observations**:
- `fused_moe_kernel` (Triton-generated MoE GEMM) is the top kernel everywhere — 31% to 54% of active time.
- GPU utilization is uniformly LOW (8.5% – 15.4%) because cudagraph is disabled, so launch overhead dominates.
- Higher-concurrency regimes have higher MoE share (more tokens × more experts active per layer).

Each per-regime `timeline_summary.json` has top-15 kernels, largest 10 idle gaps, CPU API counts, memcpy aggregate (see `results/.../nsys/<regime>/timeline_summary.json`).

### Top 5 kernels — R_concurrent_decode (highest gpu_util)

Will be filled from JSON after ncu completes for cross-validation.

---

## 5. ncu per-regime (Phase 3) — partial: R_long_prefill complete

For each regime, sglang launched under sudo ncu wrap of `sglang.bench_one_batch`
with `--profile --profile-activities CUDA_PROFILER --profile-stage {prefill|decode}`.
This way:
- ncu uses `--profile-from-start off` and waits for sglang's `cudaProfilerStart` trigger
- Model loading + warmup are NOT profiled (skipped automatically)
- Only the bench-of-interest window is captured

NCU flags (per user request):
- `--set full` (~7,000 metrics per kernel; ~1 min capture per kernel)
- `--kernel-name regex:.*` (no filter — every kernel that runs in the trigger window)
- `--launch-count 50` for R_long_prefill, `--launch-count 30` for the remaining three
  (we reduced after seeing how slow full-set replay is)

**Per-regime status (live)**:

| Regime | bench-one-batch stage | NCU runtime | unique kernels profiled | status |
|---|---|---|---|---|
| R_long_prefill      | prefill, B=4 in=8000 out=32   | ~60 min | 50 (43 unique by name) | ✅ done |
| R_concurrent_decode | decode,  B=32 in=400 out=256  | ~40 min | 7 / 30 | running |
| R_medium_balanced   | decode,  B=8  in=1600 out=256 | TBD | 0 | queued |
| R_short_decode      | decode,  B=1  in=200  out=256 | TBD | 0 | queued |

To monitor: `tail -f /tmp/ncu_batch/index.log`

### R_long_prefill — NCU verdict highlights

| Kernel (truncated) | Verdict | SM% | DRAM% | Occupancy% | TC% | Headroom% |
|---|---|---|---|---|---|---|
| `fused_moe_kernel` (Triton, 17ms total) | **low_occupancy** | 69.9 | 22.5 | **12.4** | 70.6 | 30.2 |
| `cutlass::device_kernel<flash::*>` (FlashAttn) | low_occupancy | 69.1 | 3.6 | 18.7 | 69.2 | 30.9 |
| `nvjet_tst_192x192_64x4_*_coopB_TNN` (cuBLAS) | low_occupancy | **94.7** | 24.7 | 14.8 | **96.0** | 5.3 |
| `nvjet_tst_320x128_64x3_*_coopB_TNT` (cuBLAS) | low_occupancy | 89.6 | 18.2 | 14.8 | 91.8 | 10.4 |
| `RMSNormKernel`  | tensor_core_idle | 74.4 | 67.1 | 91.6 | 0.0 | 25.6 |
| `FusedAddRMSNormKernel` | tensor_core_idle | 52.6 | **82.4** | 92.0 | 0.0 | 17.6 |
| `act_and_mul_kernel` (silu) | tensor_core_idle | 37.2 | 78.1 | 45.9 | 0.0 | 21.9 |
| `BatchQKApplyRotary*` | memory_bound | 24.9 | 71.3 | 38.6 | 0.0 | 28.7 |
| `topkGatingSoftmax` | tensor_core_idle | 71.6 | 6.4 | 77.2 | 0.1 | 28.4 |
| `moe_sum_reduce_warp_per_token_vec_kernel` | memory_bound | 25.2 | **91.6** | 42.8 | 0.4 | 8.4 |

### Key findings (R_long_prefill, the prefill-dominated regime)

1. **`fused_moe_kernel` (Triton MoE)**: SM throughput 69.9%, Tensor Core 70.6%
   — pretty good for a Triton-codegen kernel, but **occupancy is only 12.4%**
   (low warps active per SM). Verdict `low_occupancy` means kernel could run
   faster with a bigger block/grid; 30% headroom.
2. **cuBLAS Hopper kernels (`nvjet_*`)** are nearly maxed out (SM 89-94%, TC 91-96%).
   These are the QKV/MLP linears outside MoE. No optimization room.
3. **RMSNorm + activation + rotary kernels** are all memory-bound (DRAM 67-92%)
   and TC-idle — expected for elementwise ops, but a candidate for fusion
   (8.4-25.6% headroom).
4. **`moe_sum_reduce_warp_per_token_vec_kernel`**: DRAM 91.6% — the most
   memory-saturated kernel in the regime. Sums expert outputs back together.
   Tightly memory-bound, likely already close to roofline.
5. **`topkGatingSoftmax`**: SM 71.6% but DRAM only 6.4% — clearly compute-bound
   but TC-idle. Softmax is hard to put on TC; this is by design.

### Universal observation (will compare across regimes once others complete)

**No kernel in the prefill regime is on the Tensor Core peak (96-100%).** Best is
cuBLAS dense GEMM at 96%. Hot Triton MoE kernel uses TC at 70.6% — so it does
use TC, but not at peak. This is the headroom for a hand-written CUTLASS MoE
replacement (matching our earlier finding from 2026-06-09 morning that CUTLASS
in the standalone microbench also showed TC at only 7.7% — but that was a
*different code path*; here in sglang triton, TC IS being used for the MoE).

---

## 6. profile-summary-unified per regime (Phase 4)

After NCU finishes, run `scripts/unify_sweep.py` to produce
`results/.../unified/<regime>/profile_unified.json` per regime.
Each unified JSON merges:
- `e2e`: from bench_summary.json
- `gpu_macro` + `kernel_breakdown`: from timeline_summary.json (nsys)
- `kernel_micro`: from ncu_summary.json
- `evidence_chain`: skill attribution for each field

These 4 unified JSONs are the canonical artifacts for downstream consumers
(handoff drafts, comparison tables, etc.).

---

## 7. Per-regime detailed reports (TBD)

After unified JSON is built, this section will have one subsection per regime
with side-by-side comparison: bench numbers + nsys top kernels + ncu verdicts.

Looking forward to:
- Does fused_moe_kernel verdict change across regimes?
- Which regime exposes the worst SM occupancy / Tensor Core idle / etc.?
- Are there shared bottleneck kernels across all regimes (universal D-direction candidates)?

---

## 8. Skill attribution

| Phase | Skill(s) used | Output |
|---|---|---|
| Setup | (config + custom YAML in `regimes/`) | regime definition |
| Phase 1 | `e2e-bench-runner` v1 (--regimes-file) | bench_summary.json with stddev gate |
| Phase 2 | `nsys-capture`-style + `nsys-timeline-sql` × 4 windows | 4 timeline_summary.json |
| Phase 3 | `ncu-microarch`-style wrapping (custom adapter for sglang.bench_one_batch) | 4 ncu_summary.json |
| Phase 4 | `profile-summary-unified` × 4 | 4 profile_unified.json with evidence_chain |
| Reporting | this doc | sglang_triton_4regime_profiling.md |

---

## 9. Files

```
results/2026-06-09_sglang_triton_sweep/
├── bench/                          # e2e-bench-runner output
│   ├── bench_summary.json
│   └── per_run/<regime>_runN.json
├── nsys/
│   ├── sglang_all4regimes.nsys-rep  (200 MB — gitignored)
│   ├── sglang_all4regimes.sqlite    (470 MB — gitignored)
│   ├── regime_windows.json
│   ├── regime_windows_aligned.json
│   └── <regime>/timeline_summary.json
├── ncu/<regime>/
│   ├── bench.log                    # ncu progress
│   ├── <regime>_ncu.ncu-rep         (~MB — gitignored)
│   ├── ncu_raw.csv
│   └── ncu_summary.json             # produced by scripts/ncu_csv_to_summary.py
└── unified/<regime>/
    └── profile_unified.json         # the canonical artifact
```

Scripts:
- `scripts/bench_ncu_one_regime.sh` — single-regime NCU runner
- `scripts/bench_ncu_all_regimes.sh` — batch runner (serial)
- `scripts/ncu_csv_to_summary.py` — CSV → ncu_summary.json adapter
- `scripts/unify_sweep.py` — call profile-summary-unified per regime
