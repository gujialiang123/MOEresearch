# sglang Framework-Level Autotuning — Search Space Design

> **Context**: Following Debadeepta's 6/24 meeting feedback, we are quantifying
> the ceiling of framework-level autotuning before considering agent-based
> kernel rewriting. This document enumerates every sglang knob we considered,
> classifies it as P0 (in v1 search space) / P1 (add later) / P2 (excluded),
> and explains why.

## TL;DR

| Tier | Knobs | Total combinations |
|---|---|---|
| **P0 (v1 search space)** | 5 flags | 96 combinations |
| **P1 (add in v2)** | 6 flags | adds ~144× more |
| **P2 (excluded, fixed)** | ~10 flags | n/a |

For v1, we run **Optuna TPE × 30 trials per regime × 4 regimes**, plus a
cross-regime degradation swap matrix. Total wall-clock: 14-25h on 1-2 GPUs.

---

## P0 — v1 Search Space (5 flags, 96 combinations)

Selected based on:
- Strong evidence from our 6/8–6/11 experiments that these affect performance
- Cover at least one bottleneck dimension each (kernel selection, runtime
  overhead, scheduling, prefill behavior, scheduling policy)
- Independent enough that TPE can model them efficiently

### 1. `--moe-runner-backend` ⭐ headline flag

**What**: Which MoE GEMM kernel implementation to use.

**Candidates**:
- `triton` — sglang's Triton MoE kernel (default for bf16 on H200)
- `flashinfer_cutlass` — flashinfer's CUTLASS implementation

**Evidence**:
- bf16 baseline (triton): 0.74 req/s on R_medium
- bf16 patched (flashinfer_cutlass + autotune + cudagraph): 4.40 req/s
- **5.97× delta on R_medium; 4.7-8.4× across regimes**
- Source: `results/2026-06-11_harness-v1/sglang-cutlass-bf16-patched/summary.json`

**Why included**: Single biggest known knob. Even if TPE quickly converges to
cutlass, keeping it shows "Optuna discovers the same conclusion as our manual
experiment."

**Caveat**: Requires sglang main (or local patch with
`patches/sglang_cutlass_autotune_allowlist.diff`) for flashinfer_cutlass to
trigger autotune in warmup. Without that, cutlass falls back to tactic 0 and
is 3-6× slower.

### 2. `--disable-cuda-graph`

**What**: Whether to disable CUDA Graph capture/replay.

**Candidates**:
- `false` (cudagraph ON, default) — capture decode kernel sequence as graph
- `true` (cudagraph OFF) — eager mode, every kernel launched individually

**Evidence**:
- 2×2 matrix on vLLM CUTLASS (R_medium):
  - cg_OFF + autotune_OFF: 0.93 req/s (baseline)
  - cg_ON  + autotune_OFF: 1.36 req/s (+1.46×)
  - cg_OFF + autotune_ON:  1.01 req/s (+1.09×)
  - cg_ON  + autotune_ON:  **4.66 req/s (+5.0×)**
- Source: `docs/2026-06-08/vllm_2x2_autotune_cudagraph_matrix.md`

**Why included**: Multiplicative interaction with `moe-runner-backend`
(`max(CPU_work, GPU_work)` model). TPE should learn this pair-dependence.

### 3. `--max-running-requests`

**What**: scheduler's cap on concurrent in-flight requests.

**Candidates**: `8, 16, 32, 64`

**Evidence**:
- R_concurrent_decode uses concurrency=32 in workload; cap < 32 would throttle
- R_short_decode uses concurrency=1; cap is irrelevant
- KV cache pool size is independent (determined by `mem-fraction-static`), so
  this is purely a scheduler cap

**Why included**: This is the **canonical regime-dependent flag**. Optuna
should pick different values for R_short vs R_concurrent — cross-regime
degradation should be visible here.

### 4. `--chunked-prefill-size`

**What**: Maximum prefill batch token count before chunking. `-1` = no chunking.

**Candidates**: `-1, 2048, 8192`

**Evidence**:
- R_long_prefill workload: 4 prompts × 8000 tokens
- Without chunking, a single forward processes all 32k prefill tokens at once
  → blocks decode of in-flight requests
- 6/1 historical data: `--chunked-prefill-size=2048` improved R_long throughput
  by ~33% in some configurations
- Source: `docs/2026-06-01/regime_benchmark_experiment.md`

**Why included**: Selective effect — should only matter for R_long_prefill.
Excellent stress test for whether TPE picks regime-specific values.

### 5. `--schedule-policy`

**What**: How scheduler picks next request from waiting queue.

**Candidates**:
- `lpm` — Longest Prefix Match (default; prefix-cache friendly)
- `fcfs` — First Come First Served

**Evidence**: weakest evidence in our experiments; not directly measured.
Included as a **canary**: if Optuna finds no preference, we can confirm via
ablation that it's safe to prune in v2.

**Why included**: Cheap to keep (only 2 values), and useful to validate that
TPE recognizes "indifferent" knobs.

### Full v1 search space

```python
search_space = {
    "moe-runner-backend":   ["triton", "flashinfer_cutlass"],     # 2
    "disable-cuda-graph":   [True, False],                         # 2
    "max-running-requests": [8, 16, 32, 64],                       # 4
    "chunked-prefill-size": [-1, 2048, 8192],                      # 3
    "schedule-policy":      ["lpm", "fcfs"],                       # 2
}
# Total: 2 × 2 × 4 × 3 × 2 = 96 combinations
# Full grid search: 96 × 10 min = 16h (skipped)
# Optuna TPE 30 trials per regime: ~5h
```

---

## P1 — Add in v2 if v1 ceiling is unclear (6 flags)

If v1 results show "framework autotuning doesn't close the gap," expand to:

### 1. `--mem-fraction-static`

- Candidates: `0.80, 0.85, 0.90`
- Affects KV cache pool size → indirectly affects max concurrency
- Excluded from v1 because the dominant effect is OOM risk, not performance

### 2. `--enable-piecewise-cuda-graph`

- Boolean
- Captures more granular cudagraph segments
- Excluded from v1 because of historical instability bugs in our env

### 3. `--schedule-conservativeness`

- Candidates: `0.5, 1.0, 1.5`
- Affects how aggressively scheduler dispatches batches before KV overflows
- Excluded from v1 because effect is subtle and noise-prone

### 4. `--cuda-graph-bs` (custom batch size list)

- Candidates: a few different lists, e.g.,
  `[1,2,4,8,16,32]` (sparse) vs `[1,...,256]` (dense)
- Affects which batch sizes get cudagraph coverage
- Excluded from v1 because it's a list-valued hyperparameter (harder for TPE)

### 5. `--triton-attention-num-kv-splits`

- Candidates: `1, 4, 8, 16`
- Splits KV across heads for parallel decode
- Excluded from v1 because we already use `fa3` not `triton` for attention

### 6. `--attention-backend`

- Candidates: `fa3, flashinfer, triton`
- We already know `fa3` is best on H200 + MHA; including it tests TPE's
  robustness to "obvious" choices
- Excluded from v1 to keep search space small

### Aggregate impact

Adding all 6 P1 flags expands search space from 96 → ~14,000 combinations.
TPE would need 60-100 trials per regime (vs 30 in v1) to converge.

---

## P2 — Excluded, fixed for entire study

These are held constant because:
- They are physical limits (TP/EP, single-GPU constraint)
- They are model-determined (context-length from HF config)
- They are universally optimal in our setup (attention-backend=fa3)
- They are tested elsewhere or have no signal

| Flag | Fixed value | Why fixed |
|---|---|---|
| `--tp-size` | `1` | Only 1 GPU per server |
| `--ep-size` | `1` | Only 1 GPU per server |
| `--pp-size` | `1` | Only 1 GPU per server |
| `--attention-backend` | `fa3` | Hopper + MHA optimal; already validated |
| `--context-length` | `32768` | Capped (workload max ~8k tokens) |
| `--kv-cache-dtype` | `auto` | Not exploring KV quantization in v1 |
| `--quantization` | `None` (bf16) | This study is bf16-only (fp8 is separate) |
| `--mem-fraction-static` | `0.85` | Safe value; moved to P1 |
| `--max-prefill-tokens` | `16384` | Decoupled from chunked-prefill-size |
| `--enable-mixed-chunk` | `false` | Off by default |
| `--enable-torch-compile` | `false` | Known unstable in our env |
| `--enable-dp-attention` | `false` | Single-GPU |

---

## Search algorithm — Optuna TPE

**Why TPE (Tree-structured Parzen Estimator)**:
- Optuna's default; battle-tested
- Models posterior `P(config | good_objective)` vs `P(config | bad_objective)`
- Suggests next trial by maximizing the ratio
- Works well for ≤ ~15 discrete/numeric hyperparameters
- Converges in 20-50 trials for spaces our size

**Alternatives considered**:
- Grid search: 96 combinations × 10 min × 4 regimes = 64h, wasteful
- Random search: would work but slower convergence than TPE
- Bayesian optimization (GP-based): overkill for discrete spaces
- Multi-armed bandits: better when many cheap trials; ours are expensive

**Trial budget**: 30 per regime is a defensible default. Can extend to 50 if
results show plateau hasn't been reached.

---

## Objective function

**Single-objective**: `req_per_s.mean` on the target regime, taken from
`summary.json["regimes"][<regime>]["req_per_s"]["mean"]` (run 1 already dropped
as cold).

**Why single-objective (not multi)**:
- Multi-objective (`req/s` + `p99 latency`) doubles complexity
- For this study, throughput is the headline metric (matches Debadeepta's §5)
- p99 latency can be reported separately, not optimized for

**Constraints** (handled outside Optuna's objective):
- `reliable == True` required (stddev_pct < 8%); if not, retry once
- `quality_gate.passed == True` (sanity outputs); else drop trial
- `ok == True`; if `false`, return a small penalty value (not -inf, to avoid
  poisoning TPE's prior)

---

## Per-regime workloads (unchanged from 4-regime sweep)

These regimes are held constant across all autotuning runs. We optimize **the
config** for each regime, not the regime parameters themselves.

| Regime | num_prompts | prompt_words | max_new | concurrency |
|---|---|---|---|---|
| R_short_decode | 8 | 100 | 256 | 1 |
| R_medium_balanced | 16 | 800 | 256 | 8 |
| R_long_prefill | 4 | 4000 | 32 | 4 |
| R_concurrent_decode | 32 | 200 | 256 | 32 |

Source: `regimes/qwen3_30b_moe_sglang_perf_sweep.yaml`

---

## What we measure per trial

Per trial, harness writes a full `summary.json` (schema v1). Optuna reads
`regimes.<target_regime>.req_per_s.mean` as the objective. We also persist
the full summary to the Optuna trial's `user_attrs` so that:
- Post-hoc analysis can plot the entire 4-regime profile of each trial
  (not just the one we optimized for)
- Cross-regime degradation matrix can be assembled without re-running

---

## Cross-regime degradation experiment (after autotuning)

After 4 per-regime studies complete, we have 4 "winning configs":
- `config_A` = best for R_short_decode
- `config_B` = best for R_medium_balanced
- `config_C` = best for R_long_prefill
- `config_D` = best for R_concurrent_decode

We run each on every regime → 4×4 matrix:

```
              R_short    R_medium    R_long    R_concurrent
config_A     [diag]     X           Y           Z
config_B      X        [diag]       Y           Z
config_C      X         Y          [diag]       Z
config_D      X         Y           Z          [diag]
```

Diagonal = per-regime optimum.
Off-diagonal = "configured for regime X, served regime Y" degradation.

**This is the headline figure for Debadeepta**: it directly quantifies how
much workload-aware tuning matters.

---

## Decision criteria — when does agent-based rewriting become justified?

After the cross-regime matrix, three possible conclusions:

### Outcome 1: "Universal config exists"

If one config achieves ≥90% of every per-regime optimum, framework-level
autotuning is sufficient. The agent's value would be reduced — sglang
maintainers could just change defaults. (Low motivation for agent research.)

### Outcome 2: "Workload-aware tuning needed, but autotuning is the answer"

If different regimes require different configs but each individual config can
be found via Optuna in <1h, the answer is "runtime adaptive switching of
flags" — still framework-level work, no agent needed. (Some motivation.)

### Outcome 3: "Autotuning ceiling far from hardware limit"

If post-autotuning NCU profiling shows the winning kernel is at 30% TC% or
70% DRAM% on memory-bound work, there's headroom for kernel-level changes.
This is where agent-based kernel rewriting becomes a defensible research
direction. (Strong motivation.)

We'll decide which outcome we're in once §4 and §5 complete (see milestones).

---

## Milestones

| ID | Description | Status | Estimated wall-clock |
|---|---|---|---|
| at-t1 | 5-min fp8 config copy experiment | in_progress | 10 min |
| at-t2 | Write this doc | in_progress | 1 hour |
| at-t3 | Wire Optuna into harness | pending | 1 day |
| at-t4 | 3-trial smoke test | pending | 1 hour |
| at-t5 | 4× per-regime Optuna studies | pending | 10-20h on 2 GPUs |
| at-t6 | Cross-regime degradation matrix | pending | 4h |
| at-t7 | Report writing | pending | half day |

Total to completion: **3-4 working days + overnight runs**.

---

## Files this will produce

```
docs/2026-06-25/
├── sglang_autotuning_search_space.md     ← this doc
├── fp8_config_copy_experiment.md          ← P3 result
└── autotuning_ceiling_report.md           ← final headline report

harness/
└── autotune.py                            ← new Optuna integration

bench-specs/
└── autotune/<regime>/                     ← generated per-trial specs

results/2026-06-25_autotuning/
├── fp8-config-fixed/summary.json
├── studies/<regime>.db                    ← Optuna SQLite storage
├── trials/<regime>/<trial_N>/summary.json
└── cross_regime_matrix.json               ← 4×4 final matrix
```
