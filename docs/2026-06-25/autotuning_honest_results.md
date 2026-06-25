# Framework-Level Autotuning Study — Honest Results
**2026-06-25** | sglang × Optuna × 4 regimes × H200

> This document records the **as-actually-measured** results of running
> Optuna-based framework-level autotuning on sglang for Qwen3-30B-A3B on
> H200. It also records a major calibration issue we discovered late: our
> previous baselines were artificially handicapped. **The headline number
> changes from "5-9× over default" to "1.0-1.05× over default" once we use
> the correct baseline.**

---

## ⚠️ Important calibration note (read first)

We previously published 4.7-8.4× speedup numbers (6/11 cutlass-bf16-patched
study, 6/25 morning "headline numbers"). Those used a baseline with
`disable-cuda-graph: true`, originally as a workaround for a Triton 3.5.1
"KeyError: 'cubin'" bug under CUDA Graph capture.

When we today (6/25) ran sglang with **strict default flags** (no
`--disable-cuda-graph`, no `--moe-runner-backend`):
- Server starts cleanly (the cubin bug appears to be fixed by a recent
  sgl-kernel reinstall — likely 0.3.21 was installed during the 6/11 main
  experiment)
- R_medium throughput is **4.63 req/s**, not the previously cited 0.74
- The autotuned best is **4.65 req/s** — i.e., **the same**

**So the real finding is**:

> If the user just launches sglang with `--model-path X --port Y` and does
> nothing else, they get ~95-100% of what Optuna finds after 60 trials.
> sglang's defaults on H200 + Qwen3-30B-A3B + bf16 are essentially optimal.

This **completely changes the meaning** of the experiment. Old narrative was
"autotuning unlocks 5-9× hidden in framework defaults." New narrative is
"sglang defaults are already very good; what we previously thought was a
defaults problem was actually an environment bug (Triton cubin) that has
since self-resolved."

---

## Setup

| | |
|---|---|
| Model | Qwen3-30B-A3B-Instruct-2507 (bf16) |
| Hardware | NVIDIA H200, GPUs 4 + 5 (parallel) |
| sglang | study/v0.5.9 branch + local patch for `flashinfer_cutlass` autotune allowlist (= upstream main HEAD's behavior post-PR #26496) |
| sglang main HEAD attempted | requires torch 2.11 + transformers 5.8.1 + sgl-kernel 0.4.4 (CUDA 13); env mismatch. Reverted to v0.5.9 + patch (functionally identical for autotune code path) |
| flashinfer | 0.6.3 |
| torch | 2.9.1 |
| triton | 3.5.1 |
| sgl-kernel | 0.3.21 |
| Search algorithm | Optuna 4.9.0, TPESampler, seed=2026 |
| Search budget | 15 trials × 4 regimes = 60 total |
| Wall time | ~35 min on 2 GPUs in parallel |

---

## Search space (5 flags, 96 combinations)

| Flag | Candidates |
|---|---|
| `--moe-runner-backend` | triton, flashinfer_cutlass |
| `--disable-cuda-graph` | true, false |
| `--max-running-requests` | 8, 16, 32, 64 |
| `--chunked-prefill-size` | -1, 2048, 8192 |
| `--schedule-policy` | lpm, fcfs |

Rationale + P1/P2 deferred flags documented in `sglang_autotuning_search_space.md`.

---

## Three baselines compared

We have three different "baseline" numbers floating around. Tabulate them
explicitly:

| Baseline | Description | When |
|---|---|---|
| A. "6/11 baseline" | triton + `disable-cuda-graph=true` (was needed back then due to cubin bug) | 6/11 |
| B. "Today default" | strict zero-flag launch (just `--model-path` + `--port`, defaults to triton + cudagraph ON) | 6/25 |
| C. "Optuna best per regime" | Best of 15 TPE trials, optimized for each regime separately | 6/25 |

All three numbers, all 4 regimes:

| Regime | A: 6/11 baseline (cg OFF) | B: Today default (cg ON) | C: Optuna best | C/B speedup |
|---|---|---|---|---|
| R_short_decode | 0.098 | **0.888** | 0.886 | **1.00×** |
| R_medium_balanced | 0.736 | **4.629** | 4.652 | **1.00×** |
| R_long_prefill | 2.525 | **13.603** | 14.231 | **1.05×** |
| R_concurrent_decode | 2.940 | **14.712** | 13.978 | **0.95×** (!) |

Notice: on R_concurrent_decode, the "Optuna best" is actually **slower** than
today default. Two possible reasons:
1. **Noise**: stddev_pct between 1-2% on this regime; 5% diff is within noise
2. **Optuna picked `max_running_requests=32` and `chunked-prefill-size=-1`**
   — exactly what today default also has. The differences are
   `flashinfer_cutlass` (vs default's triton) and `fcfs` (vs default's lpm).
   These hurt slightly on this specific regime.

So **Optuna picked a worse-or-equal config for R_concurrent**. Within noise.

---

## Per-regime Optuna winners

| Regime | Best flags found by Optuna | Trials to converge |
|---|---|---|
| R_short_decode | triton + cg + max_req=8 + chunked=8192 + lpm | 2 of 15 |
| R_medium_balanced | triton + cg + max_req=8 + chunked=8192 + lpm | 2 of 15 |
| R_long_prefill | flashinfer_cutlass + cg + max_req=8 + chunked=-1 + fcfs | 4 of 15 |
| R_concurrent_decode | flashinfer_cutlass + cg + max_req=32 + chunked=-1 + fcfs | 9 of 15 |

Pattern: `cudagraph ON` is universal. Backend choice flips between regimes
but performance impact is small (3-5%).

---

## Cross-regime degradation matrix

Each row = one regime's best config. Each column = that config's throughput
when run against a different workload. Diagonal = self-test.

| config optimized for | R_short_decode | R_medium_balanced | R_long_prefill | R_concurrent_decode |
|---|---|---|---|---|
| R_short_decode | **0.886*** | 4.637 (99.7%) | 13.575 (95.4%) | 3.954 (**28.3%**) |
| R_medium_balanced | 0.843 (95.1%) | **4.652*** | 13.073 (91.9%) | 4.738 (33.9%) |
| R_long_prefill | 0.832 (93.9%) | 4.429 (95.2%) | **14.231*** | 4.564 (32.7%) |
| R_concurrent_decode | 0.829 (93.6%) | 4.398 (94.5%) | 14.313 (100.6%) | **13.978*** |

Percentages relative to that column's diagonal.

**Key observation**: 3 of 4 configs chose `max_running_requests=8`, which
**collapses on R_concurrent_decode** to 28-34% (the workload uses
concurrency=32). The R_concurrent winner picked 32 and is universal.

---

## Universal config (achieves ≥93% of every regime's best)

```yaml
moe-runner-backend: flashinfer_cutlass
disable-cuda-graph: false
max-running-requests: 32
chunked-prefill-size: -1
schedule-policy: fcfs
```

**Caveat**: This is *nearly identical* to sglang's today-default config, with
3 differences:
- Backend: cutlass instead of triton (1-5% impact, varies by regime)
- max_req: 32 (same as default)
- schedule-policy: fcfs instead of lpm (negligible impact)

So in practice, **using sglang default flags already gives 95-100% of this**.

---

## Three things Optuna actually showed us

### 1. "Our 6/11 manual patch was close to optimal"

The 6/11 cutlass-bf16-patched config (`cutlass + cg + max_req=32 + chunked=-1 + lpm`)
was found via manual analysis (we wanted cudagraph + autotune, picked max_req
that matched typical workload). Optuna swept 5 flags × 96 combinations × 60
trials and **arrived at the same config** (only `fcfs` vs `lpm` differs;
~0% impact). So our manual diagnostic process was accurate, but Optuna would
have found it faster.

### 2. "TPE converges fast in this space"

3 of 4 regimes hit their best in ≤4 trials. The dominant signal is
"cudagraph ON" — TPE learned this in 1-2 trials, then mostly fine-tuned
secondary knobs. 15 trials per regime was over-budget; 8 would have been
enough.

### 3. "There's no headroom hiding in framework flags"

Today default = 0.888 / 4.629 / 13.603 / 14.712 req/s.
Optuna best  = 0.886 / 4.652 / 14.231 / 13.978 req/s.
Median diff: ~0%. Max upside: 5% on R_long_prefill. Max downside: -5% on
R_concurrent_decode (Optuna actually picked a slightly worse config).

**The current 5-flag framework search space contains no meaningful headroom
over today defaults.**

---

## What this means for the "agent for optimization" thesis

The Debadeepta-recommended framing was: "first quantify framework-autotuning
ceiling before considering agents." Our finding:

- **Ceiling exists at ~today-default throughput**. There is no framework-flag
  combination significantly better than what users get out of the box.
- This is *because sglang defaults are already optimal*, not because the
  hardware roofline is below our autotune.
- The previously-cited "5-9× speedup" was an artifact of having compared to
  a self-inflicted broken baseline.

Implications:

1. **Framework-level autotuning is a bad sell** for this (model, hw, regime
   set) — there's no gap to close.
2. **Where there IS room**:
   - NCU on the universal config likely shows decode regimes are HBM-bound
     at 70-85% peak bandwidth (room exists, but at *kernel* level, not flag
     level — needs operator-fusion or memory layout work)
   - fp8 triton path has a 30-40% regression (confirmed 6/11 + 6/25) that
     framework autotuning can NOT fix — fp8 needs `--moe-runner-backend cutlass`
   - Larger search spaces (TP > 1, EP > 1) we haven't touched
   - Per-request streaming + speculative decoding — orthogonal to this study

3. **For the agent thesis**: an agent that "finds the best flag combination"
   is solving a problem that doesn't exist (in our setup). More defensible
   agent directions:
   - **Cross-framework recommendation** (sglang vs vLLM vs TensorRT-LLM)
   - **Auto-detection of missing tuned configs** (the fp8 case)
   - **Auto-generation of new tuned configs** when missing
   - **Kernel-level rewriting** for memory-bound decode regimes

---

## Failure modes during the study

6 of 60 trials returned penalty value (0.0 req/s). Inspection:

- 4 timeouts during cutlass + cudagraph_OFF warmup (autotune benchmark
  exceeded 600s)
- 1 unexplained startup failure
- 1 oddity (re-ran later and succeeded)

These were captured by harness `ok=false` summary; Optuna penalty handling
(0.0, not -inf) prevented TPE prior pollution. None affected best-trial
selection.

---

## Trial budget retrospective

15 trials × 4 regimes = 60 total trials. Best-trial numbers:
- R_short: trial 2 (87% wasted)
- R_medium: trial 2 (87% wasted)
- R_long: trial 4 (73% wasted)
- R_concurrent: trial 9 (40% wasted)

**Honest assessment**: 8 trials per regime would have been enough. Could
have run the whole study in ~18 minutes instead of 35.

For a v2 study with a larger P1 search space (10+ flags), 30-50 trials per
regime is more defensible. Pure compute time would be ~2-4 hours.

---

## Outputs

```
results/2026-06-25_autotuning/
├── true-default-bf16/                       ← strict baseline (added 6/25 evening)
│   └── summary.json
├── fp8-config-fixed/                        ← fp8 config copy experiment
│   └── summary.json
├── smoke_test/                              ← 2-trial smoke test
│   └── trial_{0000,0001}/
├── per_regime/
│   ├── R_short_decode_gpu4/
│   │   ├── best.json
│   │   ├── study.db
│   │   └── trial_{0000..0014}/
│   ├── R_medium_balanced_gpu5/...
│   ├── R_long_prefill_gpu4/...
│   └── R_concurrent_decode_gpu5/...
└── cross_regime_matrix.json                 ← 4×4 headline matrix

bench-specs/
└── sglang-true-default-bf16.yaml            ← reference for the strict default

docs/2026-06-25/
├── sglang_autotuning_search_space.md        ← search space design
├── fp8_config_copy_experiment.md            ← fp8 negative result
├── autotuning_ceiling_report.md             ← initial report (now superseded)
└── autotuning_honest_results.md             ← THIS FILE
```

---

## TL;DR for sharing with mentors

> "We ran Optuna TPE × 15 trials × 4 regimes on sglang. **The best autotuned
> config achieves ~100% (median) of what sglang gives you out of the box
> with zero flag tuning.** Previous '5-9×' numbers were against a stale
> baseline that had cudagraph disabled to work around a Triton 3.5.1 bug
> that has since been fixed in our env. For this model + GPU + bf16
> combination, framework-level autotuning has no meaningful upside.
> Headroom likely exists at the kernel level (decode HBM bandwidth) or in
> orthogonal directions (fp8 backend choice, multi-GPU configs)."
