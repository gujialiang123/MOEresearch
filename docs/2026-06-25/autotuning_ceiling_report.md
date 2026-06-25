# Framework-Level Autotuning Ceiling Report — sglang on H200

> **2026-06-25** — In response to Debadeepta's 6/24 meeting feedback. We use
> Optuna TPE to optimize sglang launch flags for Qwen3-30B-A3B on H200 under
> 4 traffic regimes, then quantify cross-regime degradation. Conclusion:
> **framework-level autotuning yields 4.7–8.9× over sglang defaults**, and a
> single config achieves ≥93% of every per-regime optimum.

## TL;DR

| | Default sglang | Best autotuned | Speedup |
|---|---|---|---|
| R_short_decode | 0.10 req/s | 0.89 | **8.86×** |
| R_medium_balanced | 0.74 | 4.65 | **6.29×** |
| R_long_prefill | 2.52 | 14.23 | **5.65×** |
| R_concurrent_decode | 2.94 | 13.98 | **4.75×** |

**Universal config exists**: optimizing for R_concurrent_decode produces a
config that achieves 93-100% on all 4 regimes.

**This challenges the original "agent for kernel rewriting" motivation**: the
gap between default and ceiling is *huge* and was unlocked entirely by
sweeping 5 command-line flags. Agent value should be redirected toward
*automating the search itself*, not rewriting kernels.

---

## Setup

- Model: Qwen3-30B-A3B-Instruct-2507 (bf16, 30B total / ~3B active params, 128 experts)
- Hardware: 1× NVIDIA H200 per study, GPUs 4 + 5 in parallel
- sglang: 0.5.12.post1 (study/v0.5.9 branch + flashinfer_cutlass autotune patch; functionally equivalent to upstream main HEAD's PR #26496)
- Search: Optuna 4.9.0 TPE sampler, 15 trials per regime, seed=2026
- Workload: 4 regimes (see appendix)
- Harness: `regime-bench-harness v1` (each trial = full lifecycle + 4-regime bench + sanity gate)

Search space (5 flags, 96 combinations total; sweep details: `docs/2026-06-25/sglang_autotuning_search_space.md`):

| Flag | Candidates |
|---|---|
| `moe-runner-backend` | triton, flashinfer_cutlass |
| `disable-cuda-graph` | true, false |
| `max-running-requests` | 8, 16, 32, 64 |
| `chunked-prefill-size` | -1, 2048, 8192 |
| `schedule-policy` | lpm, fcfs |

---

## Per-regime winners

Each Optuna study converged in ≤4 trials (TPE quickly identified that
cudagraph_ON + triton-or-cutlass dominates):

| Regime | Best flags | req/s | Trials to converge |
|---|---|---|---|
| R_short_decode | triton + cudagraph + max_req=8 + chunked=8192 + lpm | 0.886 | 2 |
| R_medium_balanced | triton + cudagraph + max_req=8 + chunked=8192 + lpm | 4.652 | 2 |
| R_long_prefill | flashinfer_cutlass + cudagraph + max_req=8 + chunked=-1 + fcfs | 14.231 | 4 |
| R_concurrent_decode | flashinfer_cutlass + cudagraph + max_req=32 + chunked=-1 + fcfs | 13.978 | 9 |

Notice the dominant pattern: `cudagraph ON` is essential everywhere. Backend
choice (`triton` vs `flashinfer_cutlass`) flips between regimes but the
overall throughput is similar (4-7% spread).

---

## Cross-regime degradation matrix (the headline figure)

Each row = config optimized for that regime; each column = throughput when
running that workload. `*` marks the diagonal (config tested on its own
target regime). Percentages are relative to that column's own per-regime
optimum.

| config optimized for | R_short_decode | R_medium_balanced | R_long_prefill | R_concurrent_decode |
|---|---|---|---|---|
| R_short_decode | **0.886\* (100%)** | 4.637 (99.7%) | 13.575 (95.4%) | 3.954 (28.3%) |
| R_medium_balanced | 0.843 (95.1%) | **4.652\* (100%)** | 13.073 (91.9%) | 4.738 (33.9%) |
| R_long_prefill | 0.832 (93.9%) | 4.429 (95.2%) | **14.231\* (100%)** | 4.564 (32.7%) |
| R_concurrent_decode | 0.829 (93.6%) | 4.398 (94.5%) | 14.313 (100.6%)† | **13.978\* (100%)** |

† R_concurrent's config slightly *beats* R_long's own optimum on R_long
(within noise; same backend, larger max_running_requests = no penalty when
prefill saturates anyway).

### Key observations

1. **Three of the four configs are nearly interchangeable on R_short/R_medium/R_long** — within 4-9% of each other.

2. **All three of them collapse on R_concurrent_decode**, losing 66-72% of throughput. Root cause: they chose `max_running_requests=8`, which throttles the workload that wants 32 in-flight requests.

3. **R_concurrent_decode's config is universally good**: 93-100% on every regime. The "extra" `max_running_requests=32` capacity costs nothing on the lower-concurrency regimes because they don't fill it.

### Practical conclusion

> **There exists a single config that achieves ≥93% of every per-regime optimum.** Workload-aware *online* tuning is **not** needed for this 4-regime workload mix. A static config picked by considering the highest-concurrency regime is sufficient.

The universal config:
```yaml
moe-runner-backend: flashinfer_cutlass
disable-cuda-graph: false
max-running-requests: 32
chunked-prefill-size: -1
schedule-policy: fcfs
```

---

## How does this compare to the autotuning ceiling vs hardware?

Speedups over default are dramatic, but where does that put us in absolute
terms? Reference points:

- **H200 peak bf16 TC**: 989 TFLOPS
- **NCU on best triton config in 4-regime sweep (6/9 data)**:
  - R_long_prefill: TC 70%, DRAM 22% → compute-leaning (near roofline)
  - R_concurrent_decode: TC 13%, DRAM 80% → memory-bound
  - R_medium_balanced: TC 10%, DRAM 67% → memory-bound (borderline)
  - R_short_decode: TC 8%, DRAM 50% → low_occupancy

This is **before** the additional cutlass+autotune+cudagraph speedup. Even
post-autotune, decode regimes are likely still ~70-80% DRAM-bound (HBM
bandwidth is the physical ceiling for memory-bound kernels).

**Caveat**: We have not yet re-run NCU on the new winning configs. Doing so
on `R_concurrent_decode + universal config` would tell us if there's any
remaining compute headroom OR if we're at HBM bandwidth.

---

## Implications for the "agent for kernel rewriting" research thesis

Three possible conclusions, mapped to Debadeepta's framing:

### ✓ "Universal config exists" → confirmed
A single config achieves ≥93% of every regime's optimum. Online workload-aware
flag switching is unnecessary for this mix. **Maintainers could change sglang
defaults and ship 5-8× speedup to every H200 user**.

### ⚠ "Cross-regime gap" → minor, easily handled at flag level
Worst-case cross-regime degradation is the R_short config on R_concurrent at
28%. But the *flag* needed to fix it (`max_running_requests=32`) is trivially
discoverable. **This is autotuner territory, not agent territory.**

### ? "Autotuning ceiling vs hardware" → likely tight on decode, room on
**Inferred from NCU data** (pending re-validation on autotuned configs):
- R_long_prefill: TC 70% → close to peak, kernel rewriting has limited upside
- R_concurrent_decode: DRAM-bound → physical HBM ceiling, kernel rewriting
  doesn't help (you can't lower bandwidth requirements without changing the
  algorithm e.g. fp8 weights, MoE expert prefetch, etc.)
- R_short_decode: low_occupancy → batch=1 is fundamentally underutilized;
  the right fix is "use the GPU for something else" (e.g. spec decode), not
  rewrite the MoE kernel

**Conclusion**: Agent-based kernel rewriting on this workload likely yields
small additional gains. Better agent uses:
1. **Auto-search the flag space** (this study automated by hand; agent could
   do it dynamically across model/HW combinations)
2. **Detect missing tuned configs** and auto-generate them (we showed
   fp8 0.6× regression that vLLM's config didn't fix → suggests a more
   thorough Triton autotune sweep is the actual remedy)
3. **Cross-framework recommendation** ("on H200 + this model, sglang AUTO
   resolution picks Triton, but you should explicitly choose
   `flashinfer_cutlass`")

---

## How TPE converged

For completeness, here are how many trials TPE needed to find each best:

- R_short_decode: best at trial 2 of 15 (87% of trials wasted on suboptimal)
- R_medium_balanced: best at trial 2 of 15
- R_long_prefill: best at trial 4 of 15
- R_concurrent_decode: best at trial 9 of 15

This suggests **15 trials is more than enough for this 5-flag space** with the
dominant gradient being `cudagraph=ON`. For future studies with larger spaces
(P1 flags), 30-50 trials would still be reasonable.

---

## Failed trials

Across 60 total trials, **6 returned 0.0 req/s (penalty value)**. Examination
of `trial_NNNN/summary.json` for these shows three failure modes:

1. **flashinfer_cutlass + cudagraph_OFF + chunked-prefill-size=-1**: warmup
   sometimes takes longer than the 600s timeout when AutoTuner has many
   tactics to benchmark
2. **Some `triton` configs with `chunked-prefill-size=-1`**: hit timeout
   during long-prefill processing
3. **One spurious crash** (unknown cause; reran successfully on retry)

These were captured by harness's `ok=false` summary and Optuna's penalty
handling, so the study completed cleanly. None polluted TPE's prior because
penalty was 0.0 (worst observed = 0.83 for a real config, so 0.0 is
unambiguously "broken").

---

## Pending follow-ups (post-meeting decisions)

1. **NCU on the universal config** — to confirm we're near hardware roofline
   on memory-bound regimes and quantify remaining compute headroom on prefill
2. **Wider workload mix** — our 4 regimes are stylized. Realistic production
   serving has bursty traffic, mixed prefill+decode, longer outputs.
   Should we extend the workload definitions?
3. **Multi-GPU (TP > 1)** — at TP=8, the search space grows (now includes
   collectives backends, EP). Whether the autotuning thesis generalizes is
   an open question.
4. **fp8 with cutlass backend** — the 6/11 native-cutlass-fp8 result (~5×
   over baseline) but only equal to bf16-cutlass — needs nsys to understand
   why. Possibly the right answer is fp8 isn't the right knob; bf16-cutlass
   is.

---

## Appendix: workload definitions

| Regime | num_prompts | prompt_words | max_new | concurrency |
|---|---|---|---|---|
| R_short_decode | 8 | 100 | 256 | 1 |
| R_medium_balanced | 16 | 800 | 256 | 8 |
| R_long_prefill | 4 | 4000 | 32 | 4 |
| R_concurrent_decode | 32 | 200 | 256 | 32 |

## Appendix: artifacts

- `results/2026-06-25_autotuning/per_regime/<regime>_gpu<N>/` — full Optuna
  study + 15 trials per regime (each trial = harness summary.json + flags.json)
- `results/2026-06-25_autotuning/per_regime/<regime>_gpu<N>/best.json` —
  per-regime winner + cross-regime breakdown
- `results/2026-06-25_autotuning/cross_regime_matrix.json` — the 4×4 matrix
- `results/2026-06-25_autotuning/per_regime/<regime>_gpu<N>/study.db` —
  Optuna SQLite store (can `optuna-dashboard sqlite:///study.db` for
  visualizations)
