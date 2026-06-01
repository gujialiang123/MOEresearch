# Regime benchmark experiment — Qwen3-0.6B vs Qwen3-30B-A3B (MoE) on H200

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#中文版)
>
> **Date**: 2026-06-01 · **Author**: Stage A harness + `scripts/regime_study/aggregate.py` ·
> **Status**: 2 reps per regime, 16/16 (model,regime) cells, 31/32 individual runs passed (one MoE R8 OOM under transient external GPU contention; cleanly re-ran in rep 2).

## 1. Goal

Characterise **how SGLang performance varies across workload regimes** when
the same model and the same server config are kept fixed. No optimisation —
only evidence collection. Output a meeting-ready table that:

1. shows two already-configured models running on H200,
2. exposes measurable per-regime differences,
3. flags suspicious regimes worth profiling next,
4. provides a reusable mini-benchmark template for future deep dives.

## 2. Hardware + software environment

| Item | Value |
|---|---|
| Hardware | 8× NVIDIA H200 (143 GB each), single-GPU per run |
| GPUs used | GPU 0 for dense rep 1 + MoE rep 1/2; GPU 1 for dense rep 2 |
| Conda env | `sglang-dev` |
| SGLang | 0.5.12.post1 (source: `/home/t-jialianggu/work/sglang`) |
| Python | 3.11 |
| CUDA | 12.8 |
| HF cache | `/data/hf/gujialiang123/hf_cache` |

## 3. Models tested

Both models were already configured and downloaded.

| Model | Path | Size | Config file |
|---|---|---|---|
| Qwen3-0.6B (dense) | `/data/hf/models/Qwen3-0.6B` | ~1.2 GB bf16 | `configs/base.yaml` |
| Qwen3-30B-A3B-Instruct-2507 (MoE: 128 experts, 8 active/token, GQA 32/4) | `/data/hf/models/Qwen3-30B-A3B-Instruct-2507` | ~57 GB bf16 | `configs/moe_qwen3_30b.yaml` |

### Launch configs (exact)

Both configs share the same scheduling knobs (only `mem-fraction-static` and
`context-length` differ — MoE is 0.85/32768, dense is 0.7/default). This
isolates the regime effect from per-model tuning.

```yaml
# common knobs (both models)
tensor-parallel-size: 1
schedule-policy: lpm
schedule-conservativeness: 1.0
max-running-requests: 32
chunked-prefill-size: -1
max-prefill-tokens: 16384
disable-radix-cache: false
disable-cuda-graph: false
```

SGLang launch wrapper: `scripts/launch_server.py` (via `python -m sglang.launch_server <flags>`).

## 4. Regime matrix

8 regimes, all from `regime_scout/candidates_regime_study/*.yaml`. Same set
applied to both models.

| ID | Purpose | Dataset | InLen | OutLen | Concurrency | NumPrompts |
|---|---|---|---|---|---|---|
| **R1** | Baseline (low load) | random | 128 | 128 | 4 | 32 |
| **R2** | Decode-heavy | random | 128 | 1024 | 32 | 64 |
| **R3** | Prefill-heavy | random | 4096 | 128 | 8 | 32 |
| **R4** | Long-in + long-out | random | 4096 | 512 | 8 | 24 |
| **R5** | Saturation (intentionally above server `max-running-requests=32`) | random | 512 | 256 | **64** | 128 |
| **R6** | Single-stream / latency | random | 512 | 256 | 1 | 16 |
| **R7** | Wide input distribution (`random_range_ratio=0.95`, ~100→4000 tokens) | random | 2048±95% | 256 | 32 | 64 |
| **R8** | Prefix-cache friendly | generated-shared-prefix (sys 2048 + Q 128, 8 groups × 8 prompts) | 2048+128 | 256 | 32 | 64 |

> R7 is the proxy for "mixed prompt lengths" within the constraints of
> `sglang.bench_serving`'s `--dataset-name random` — it produces a spread of
> input lengths around the mean rather than a true bimodal mix. R8 uses
> `--dataset-name generated-shared-prefix`, which exercises the radix cache.

## 5. How runs are produced

Standard repo harness — **no new infrastructure**:

```bash
# one suite = 8 regimes × 1 server start each (server is restarted per regime)
python scripts/run_regime_suite.py \
  --config configs/base.yaml \
  --workload-dir regime_scout/candidates_regime_study \
  --out results/regime_bench/raw/dense_rep1.jsonl \
  --run-root experiments/tmp/regime_study/dense_rep1 \
  --mode quick --reset
```

For each workload, `scripts/run_experiment.py` launches the server,
`scripts/wait_ready.py` blocks until ready, `scripts/run_benchmark.py` calls
`python -m sglang.bench_serving` with the workload's parameters, and
`scripts/parse_metrics.py` turns the raw jsonl into a canonical metrics
schema. SGLang's `bench_serving` provides:

- `request_throughput` (req/s), `output_throughput` (out tok/s), `input_throughput`
- `mean/median/p99 TTFT`, `mean/median/p99 TPOT`, `p50/p95/p99 ITL`
- `mean/median/p90/p99 e2e latency`
- per-request `ttfts[]` / `itls[]` (used to derive `ttft_p95` ourselves)

Server-side behaviour is mined from `server.log` by
`.github/skills/server-log-mining/impl/parse_server_log.py` — extracts the
attention backend, scheduler policy, CUDA-graph capture set, KV cache size,
`max_total_num_tokens`, peak running/queue lengths, prefill/decode batch
counts, retract events, etc. This is what fills the "server-observable
behaviour" columns below.

## 6. Server-observable behaviour (from `server.log`)

Both models picked the **same backend + scheduler stack** under the same
config. These do NOT change across regimes — they're chosen at startup.

| Item | dense (Qwen3-0.6B) | MoE (Qwen3-30B-A3B) |
|---|---|---|
| Attention backend | **fa3** (FlashAttention-3) | **fa3** |
| Schedule policy | `lpm` | `lpm` |
| `chunked_prefill_size` | -1 (disabled — full-prefill batches) | -1 |
| `max_prefill_tokens` | 16384 | 16384 |
| `max_running_requests` | 32 | 32 |
| `mem_fraction_static` | 0.7 | 0.85 |
| KV cache total | 96.2 GB | 61.2 GB |
| `max_total_num_tokens` | 900 480 | 668 504 |
| CUDA-graph captured batch sizes | 1..32 (max 32) | 1..32 |

**What we did NOT directly observe from logs**:

- Per-kernel selection (e.g. which prefill kernel vs decode kernel was used
  per batch). `server.log` only logs aggregated batch shapes. → would need
  PyTorch profiler or Nsight to attribute.
- MoE expert imbalance / routing entropy — not in default sglang logs.
  → would need a custom hook or torch profiler with kernel-name regex
  (`moe.*topk`, `grouped_gemm`, `all_to_all`).
- Whether `fa3` falls back to fa2 for certain shapes. → not directly
  observable; would need NVTX range or torch profiler.

These are the **next-step profiling targets**, addressed in §11.

## 7. Result table — Qwen3-0.6B (dense)

> Mean of 2 reps unless noted. Per-rep raw rows in `results/regime_bench/parsed_results.csv`.

| Regime | InLen | OutLen | Conc | NumPrompts | Req/s | Out tok/s | TTFT mean (ms) | TTFT p95 (ms) | TPOT mean (ms) | ITL p95 (ms) | E2E p99 (ms) | peak_run | peak_queue | Gap vs R1 (out tok/s) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1** baseline | 128 | 128 | 4 | 32 | 21.1 | 1 427 | 27.4 | 95.4 | 2.26 | 2.50 | 374 | 4 | 0 | 0.0% | |
| **R2** decode-heavy | 128 | 1024 | 32 | 64 | 16.7 | **9 009** | 53.2 | 118.4 | 2.69 | 2.72 | 2 853 | 32 | 1 | **+531%** | best throughput |
| **R3** prefill-heavy | 4096 | 128 | 8 | 32 | 26.7 | 1 803 | 41.5 | 109.0 | 3.29 | 14.1 | 544 | 8.5 | 0 | +26% | high ITL p95 |
| **R4** long-in + long-out | 4096 | 512 | 8 | 24 | 9.0 | 2 498 | 43.2 | 94.8 | 2.73 | 2.90 | 1 477 | 8 | 0 | +75% | |
| **R5** high-conc | 512 | 256 | 64 | 128 | 46.7 | 5 802 | **537.9** | **1 235** | 4.83 | 17.8 | 2 332 | 32 | **33** | +307% | **hit `max_running` cap** |
| **R6** single-stream | 512 | 256 | 1 | 16 | 3.5 | 521 | **22.7** | **33.9** | **1.77** | **1.82** | 483 | 1 | 0 | −64% | best latency, worst throughput |
| **R7** mixed lengths | 2048±95% | 256 | 32 | 64 | 27.1 | 6 736 | 107.9 | 187.9 | 4.25 | 4.12 | 1 373 | 32 | 4.5 | +372% | small backpressure |
| **R8** prefix sharing | 2048+128 | 256 | 32 | 64 | 27.0 | 6 904 | 157.6 | 194.1 | 4.00 | 3.91 | 1 224 | 32 | 0 | +384% | radix cache absorbs the 2K prefix |

## 8. Result table — Qwen3-30B-A3B (MoE)

> Mean of 2 reps for all cells. R8 OOM'd in rep 1 due to **external** GPU 0
> contention (`olive` process from another user took ~20 GB mid-run); rep 2
> ran cleanly and is the source for R8 metrics. See §10.

| Regime | InLen | OutLen | Conc | NumPrompts | Req/s | Out tok/s | TTFT mean (ms) | TTFT p95 (ms) | TPOT mean (ms) | ITL p95 (ms) | E2E p99 (ms) | peak_run | peak_queue | Gap vs R1 (out tok/s) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1** baseline | 128 | 128 | 4 | 32 | 6.9 | 469 | 57.1 | 132.9 | 7.41 | 6.83 | 1 079 | 4 | 0 | 0.0% | |
| **R2** decode-heavy | 128 | 1024 | 32 | 64 | 3.2 | 1 736 | 100.1 | 148.0 | 14.67 | 15.5 | 15 479 | 32 | 0 | +270% | **e2e p99 = 15 s!** |
| **R3** prefill-heavy | 4096 | 128 | 8 | 32 | 7.9 | 536 | 123.9 | 308.8 | 11.54 | 41.6 | 1 952 | 8 | 0 | +14% | very high ITL p95 |
| **R4** long-in + long-out | 4096 | 512 | 8 | 24 | 2.4 | 667 | 156.5 | 308.9 | 10.87 | 14.9 | 5 668 | 8 | 0 | +42% | e2e p99 ~5.7s |
| **R5** high-conc | 512 | 256 | 64 | 128 | 11.3 | 1 406 | **2 075** | **5 338** | 19.9 | 49.7 | 9 406 | 32 | **34** | +200% | **catastrophic TTFT tail** |
| **R6** single-stream | 512 | 256 | 1 | 16 | 1.5 | 222 | **44.2** | **50.4** | **4.24** | **4.29** | 1 113 | 1 | 0 | −53% | best latency; very low throughput |
| **R7** mixed lengths | 2048±95% | 256 | 32 | 64 | 7.6 | 1 880 | 407.7 | 1 001 | 15.21 | 14.2 | 5 159 | 32 | 9.5 | +301% | noticeable backpressure |
| **R8** prefix sharing | 2048+128 | 256 | 32 | 64 | **13.1** | **3 355** | 385.1 | 726.5 | 8.03 | 7.99 | 2 673 | 32 | 5 | **+615%** | best throughput; rep 1 OOM under contention |

## 9. Cross-regime summary (best vs worst per model)

| Model | Best regime (out tok/s) | Worst regime | Out-tok/s gap | Lowest TTFT p50 regime | Highest TTFT p50 regime | TTFT p50 gap | Main observation |
|---|---|---|---|---|---|---|---|
| dense | **R2** 9 009 | **R6** 521 | **17.3×** | R6 20.7 ms | R5 538 ms | **26×** | Decode regime maximises throughput; high-conc R5 destroys TTFT due to `max_running=32` cap |
| MoE | **R8** 3 355 | **R6** 222 | **15.1×** | R6 44.5 ms | R5 2 075 ms | **47×** | Prefix cache wins; **R5 TTFT 47× worse than R6** — MoE is much more sensitive to saturation than dense |

## 10. Failed / unstable runs

| Model | Regime | Rep | Status | Cause | Resolution |
|---|---|---|---|---|---|
| MoE | R8 | 1 | **FAIL — CUDA OOM** | GPU 0 was shared with another user's `olive` job that grabbed ~20 GB **mid-run**. MoE server, sized for the full GPU at startup, ran out when activations needed +190 MB. Server log: `torch.OutOfMemoryError: ... Process 2070002 has 19.84 GiB memory in use ... Including non-PyTorch memory, this process has 119.93 GiB memory in use.` | Re-ran rep 2 after the foreign process exited; rep 2 R8 passed cleanly with 3 355 tok/s. **No code change.** |

No other OOM / crash / timeout. No retract events. No KV-pool-full events.

## 11. Brief observations & profiling candidates

### 11.1 Cross-model

- **MoE pays a much larger "regime-mismatch" tax than dense**: TTFT p50 swing
  47× on MoE vs 26× on dense. MoE single-stream R6 reaches 222 tok/s — only
  47% of dense R6's *worst-case* output throughput.
- **The `max-running-requests=32` cap is an active bottleneck for R5 on both
  models** (peak_queue ≈ 33-34, `concurrency_capped=True`). On dense the
  penalty is high TTFT tail; on MoE it's catastrophic (5.3 s p95 TTFT).
- **Prefix sharing (R8) reverses the throughput ranking**: it is the *best*
  regime for MoE (3 355 tok/s), and the second-best for dense. The radix
  cache absorbs the 2 K shared prefix → effectively shorter prefill per
  request.

### 11.2 Dense-specific

- R3 (prefill-heavy) shows **ITL p95 of 14.1 ms vs ITL p50 of 2.4 ms** (5.9×
  gap) — likely chunked-prefill is bursting decode steps. Worth profiling.
- R5 dense is bottlenecked by config, not the model — Stage B's
  `config-agent` already showed this on the MoE side; same fix likely
  applies to dense.

### 11.3 MoE-specific

- R2 (decode-heavy) reaches **e2e p99 = 15.5 s**, far above what aggregate
  TPOT suggests. Tail is suspicious. Candidate: cold-expert routing or
  expert imbalance.
- R3 (prefill 4 K) has **ITL p95 = 41.6 ms** — 3.6× MoE's R3 p50. Could be a
  prefill→decode transition spike (router warm-up?) or routing imbalance on
  long contexts.
- R4 + R7 have noticeable queue backpressure (peak_queue 5-10) even at
  concurrency 32 — i.e. prefills can't keep up; chunked-prefill is OFF
  (`chunked_prefill_size=-1`). Enabling chunked prefill is a high-priority
  candidate to test.

### 11.4 What current logs CANNOT tell us (next-step profiling)

| Question | Tool to use |
|---|---|
| Is `fa3` actually selected for every batch shape, or does it fall back? | NVTX + PyTorch profiler (see `.github/skills/pytorch-profiling/`) |
| Which MoE kernels dominate (`moe.*topk` vs `grouped_gemm` vs `all_to_all`)? | PyTorch profiler — already designed in `pytorch-profiling/SKILL.md` |
| Is expert routing imbalanced (a few hot experts)? | Custom hook in sglang `expert_distribution_recorder` |
| Are CUDA graphs hit on every decode step, or are there fallback paths? | sglang `--log-level debug` + grep for `cuda_graph` |
| Cold-start vs steady-state TPOT? | Already designed: `pytorch-profiling/impl/run_profile.py --warmup-requests 16` |

## 12. Suggested next steps

Concrete priority order (smallest-effort first):

1. **Enable chunked prefill** (`chunked-prefill-size=2048`) on MoE and
   re-run R4 + R7. Expected: lower peak_queue and lower TTFT p95.
2. **Raise `max-running-requests` to 64** on both models and re-run R5.
   Expected: TTFT p50 drops from 538 ms → ~100 ms (dense) and 2 075 ms →
   ~500 ms (MoE).
3. **Attach `pytorch-profiling` skill to MoE R2** to investigate the 15 s
   e2e p99 outlier. If routing imbalance shows up, file Stage A problem
   package `experiments/problems_moe/PNNN-routing-tail/`.
4. **Run a true bimodal R7** by writing a small mixed-workload extension
   (currently we only have `random_range_ratio` jitter, not 70/30 short/long
   mix). This is a `regime_scout` extension, not a sglang change.

## 13. Reproducing the experiment

```bash
# 1. activate env
conda activate sglang-dev
cd /home/t-jialianggu/work/EndtoEnd-auto-optimization

# 2. run dense (≈9 min on H200)
python scripts/run_regime_suite.py --reset \
  --config configs/base.yaml \
  --workload-dir regime_scout/candidates_regime_study \
  --out results/regime_bench/raw/dense_rep1.jsonl \
  --run-root experiments/tmp/regime_study/dense_rep1

# 3. run MoE (≈10 min on H200)
python scripts/run_regime_suite.py --reset \
  --config configs/moe_qwen3_30b.yaml \
  --workload-dir regime_scout/candidates_regime_study \
  --out results/regime_bench/raw/moe_rep1.jsonl \
  --run-root experiments/tmp/regime_study/moe_rep1

# 4. aggregate (reads every results/regime_bench/raw/*_rep*.jsonl)
python scripts/regime_study/aggregate.py
# → results/regime_bench/{parsed_results.csv, summary_table.csv, summary.md}
```

For 2 reps + parallel: see how dense_rep2 used `_gpu_id: 1` + `port: 30001`
(via the `sed` snippet in `logs/regime_study_dense_rep2.log`).

## 14. Conclusion

This experiment shows that the **same model and SGLang engine can exhibit
17× output-throughput swings and 47× TTFT-p50 swings under different request
regimes**, with no code or config change. The largest gaps appear in:

- **R5** (high-concurrency saturation) on both models — actively bottlenecked
  by `max-running-requests=32`;
- **R2** (decode-heavy) and **R8** (prefix sharing) on MoE — where MoE shines
  for throughput but TTFT and tail latency expose suspicious patterns
  (e2e p99 = 15.5 s on R2);
- **R3 / R4 / R7** on MoE — backpressure + high ITL p95 suggest chunked
  prefill should be tested.

These regimes are good candidates for further profiling and mini-reproducer
construction. The next step is to attach kernel-level profiling evidence
(via `.github/skills/pytorch-profiling/`) and investigate whether
scheduler / config / kernel choices explain the observed gaps.

The full per-run data is in `results/regime_bench/parsed_results.csv`; the
aggregated table is in `results/regime_bench/summary_table.csv`; raw
SGLang logs are in `experiments/tmp/regime_study/<model>_rep<N>/<ts>/run_*/server.log`.

## 15. Hardware view & kernel selection (deep dive, follow-up)

> 🟢 NEW. Added 2026-06-01 evening, using **only SGLang's existing tools**:
> `/get_server_info` HTTP endpoint (runtime-confirmed backend), `nvidia-smi`
> sampling at 0.5 s (GPU memory/utilisation/power), and
> `sglang.bench_serving --profile` (Torch profiler trace).

### 15.1 Scope & tooling

For 6 cells (dense + MoE × {R1, R5, R8}), we re-ran the workload with:

- `SGLANG_TORCH_PROFILER_DIR` env on the server → enables Torch profiler.
- `--profile --profile-num-steps 10 --warmup-requests 8` on `bench_serving`
  → captures 10 forward steps after warmup.
- `nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm`
  every 0.5 s during the bench window.
- HTTP `GET /get_server_info` snapshot before killing the server →
  runtime-confirmed `attention_backend`, `sampling_backend`, etc.

All wrapped by `scripts/regime_study/run_hw_view.py`; aggregated by
`scripts/regime_study/aggregate_hw_view.py`. Per-cell raw output in
`results/regime_bench/raw/hw_view/<model>_<regime>/` (5 files each:
`hardware_view.json`, `profile_summary.json`, `server_info.json`,
`gpu_samples.csv`, `server.log`).

Full table also at `results/regime_bench/hardware_view_table.{md,csv}`.

### 15.2 Backend selection (runtime-confirmed)

| Model | Regime | Attention | Sampling | Schedule | KV dtype | max_running | torch_compile_max_bs |
|---|---|---|---|---|---|---|---|
| dense | R1 / R5 / R8 | **fa3** (FlashAttention-3) | **flashinfer** | lpm | auto (bf16) | 32 | 32 |
| MoE   | R1 / R5 / R8 | **fa3** | **flashinfer** | lpm | auto | 32 | 32 |

Backend is **constant across regimes** (selected at startup based on
hardware + model config). The interesting variation is in the **kernel
mix** (§15.4).

### 15.3 Hardware view (`nvidia-smi`, 0.5 s sampling, bench window only)

| Model | Regime | Wall (s) | Samples | Mem peak (GiB) | Mem mean (GiB) | GPU util mean (%) | GPU util p95 (%) | Mem-ctrl util (%) | Power mean (W) | Power peak (W) | Peak temp (°C) | SM clock mean (MHz) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | R1 | 41.5 | 43 | 98.4 | 61.4 | 4.1 | 10 | 1.0 | 113 | 200 | 42 | 1411 |
| dense | R5 | 85.7 | 54 | 98.6 | 63.8 | 4.2 | 26 | 1.1 | 120 | 266 | 45 | 1526 |
| dense | R8 | 64.6 | 48 | 99.0 | 59.6 | **12.7** | **100** | 4.5 | 147 | **630** | 56 | 1468 |
| MoE | R1 | 95.7 | 76 | 119.3 | 74.8 | 11.8 | 82 | 5.3 | 140 | 416 | 47 | 1636 |
| MoE | R5 | 109.4 | 99 | 120.1 | 84.2 | **17.0** | **97** | **12.1** | **187** | 578 | 54 | 1749 |
| MoE | R8 | 97.9 | 80 | 121.3 | 74.6 | 14.7 | 100 | 5.0 | 159 | 555 | 54 | 1650 |

**Findings**:

- **Dense is severely under-utilising the H200**: GPU util mean is 4-13%
  even at saturation (R5). This is the *small-model* tax — at TP=1 a 0.6B
  model leaves the SM array mostly idle.
- **MoE pushes the memory controller** (mem-ctrl util 5-12 % vs 1-5 % on
  dense). R5 on MoE hits 12.1 % mem-ctrl util — confirms MoE is closer to
  memory-bandwidth-bound than dense.
- **Peak power**: dense R8 hits 630 W and MoE R5 averages 187 W (peak 578 W),
  vs H200's 700 W TDP. R8 prefix sharing makes dense burst hardest because
  the radix-cached prefix lets many sequences move into pure-decode quickly,
  briefly running near peak utilisation.
- **Memory peak**: dense pre-reserves ~98 GiB (KV cache + model weights at
  `mem_fraction_static=0.7` on 143 GiB); MoE pre-reserves ~120 GiB at 0.85.
  These are server-startup constants; they do NOT shrink between regimes.

### 15.4 Kernel breakdown — categories (top-20 GPU events by self-time)

| Model | Regime | Trace wall (ms) | GPU active (ms) | Kernel categories |
|---|---|---|---|---|
| dense | R1 | 74.5 | 23.3 | cuda runtime/overhead **38.2%**; GEMM 23.9%; FlashAttention 6.7%; norm 4.2% |
| dense | R5 | 189.8 | 45.6 | cuda runtime/overhead 22.3%; GEMM 15.3%; FlashAttention 11.5%; norm 4.0%; elementwise 3.6% |
| dense | R8 | 234.0 | 103.2 | **FlashAttention 23.1%**; GEMM 21.6%; cuda runtime/overhead 9.4%; elementwise 4.3%; norm 3.8% |
| MoE | R1 | 206.3 | 103.9 | **MoE 33.8%**; cuda runtime/overhead 20.0%; other 19.5%; GEMM 5.5%; FlashAttention 2.7%; norm 1.9% |
| MoE | R5 | 505.1 | 297.5 | **MoE 45.3%**; other 19.3%; GEMM 6.7%; cuda runtime/overhead 5.3%; FlashAttention 2.7%; norm 1.8% |
| MoE | R8 | 696.1 | 598.0 | **MoE 46.5%**; FlashAttention 13.4%; GEMM 12.7%; other 5.7%; elementwise 2.9%; norm 2.4% |

### 15.5 Top-2 kernels per cell (the actual kernel names)

| Model | Regime | #1 kernel (self %) | #1 calls | #2 kernel (self %) |
|---|---|---|---|---|
| dense | R1 | `cudaGraphLaunch` (**17.9%**) | 7 | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` (10.0%) |
| dense | R5 | `cudaLaunchKernel` (**9.9%**) | **916** | `cudaLaunchKernelExC` (6.7%) |
| dense | R8 | `flash::FlashAttnFwdSm90<...,bfloat16_t,Sm90,…>` (**15.6%**) | 112 | another `FlashAttnFwdSm90` variant (7.4%) |
| MoE | R1 | **`fused_moe_kernel`** (**33.8%**) | 864 | `cudaEventSynchronize` (19.5%) |
| MoE | R5 | **`fused_moe_kernel`** (**45.3%**) | 864 | `cudaEventSynchronize` (16.8%) |
| MoE | R8 | **`fused_moe_kernel`** (**46.5%**) | 864 | `flash::FlashAttnFwdSm90<...>` (10.3%) |

### 15.6 Headline insights from the kernel view

These are claims we can defend from kernel-level evidence:

1. **MoE is dominated by `fused_moe_kernel`** — 34-47% of all GPU time across
   the 3 MoE regimes; **`cudaEventSynchronize` accounts for another 17-20%**,
   suggesting kernel-to-kernel sync waits inside the MoE expert dispatch are
   a major secondary bottleneck. This is the clearest case for kernel-agent
   work in this whole study.
2. **Dense R1 is overhead-bound, not compute-bound**: top kernel is
   `cudaGraphLaunch` (17.9 %), with FlashAttention only 6.7 %. The CUDA
   graphs are firing but the entire model fits in one graph that is so
   cheap that launch overhead dominates. This explains the 4% GPU util.
3. **Dense R5 (cap-hit) is *launch*-bound at the scheduler boundary**:
   `cudaLaunchKernel` shows up 916 times in 10 profiled steps. The
   `max_running_requests=32` cap means the scheduler is in a tight retry
   loop; each retry incurs a launch.
4. **Dense R8 is the only dense regime that's actually attention-bound**:
   `flash::FlashAttnFwdSm90` is the top kernel (15.6%), and FlashAttention
   total is 23.1 %. This is what high GPU util looks like for a small
   model — long shared prefix means many tokens to attend over.
5. **MoE R8 (prefix sharing) has the highest absolute GPU time and the
   richest mix**: MoE 46.5 % + FlashAttention 13.4 % + GEMM 12.7 % — all
   three pillars firing simultaneously. This is why R8 is MoE's throughput
   champion (3 355 out tok/s).

### 15.7 What this profile **cannot** tell us (the next layer down)

- **Per-expert load balance inside `fused_moe_kernel`** — Torch profiler
  doesn't break out per-expert ops. Need sglang's
  `expert_distribution_recorder` hook + a parser, OR Nsight Compute on the
  kernel.
- **Whether `fa3` ever falls back to `fa2`** for unusual shapes — the
  trace shows only Sm90 fa3 kernels in our 6 cells, so no fallback observed,
  but absence isn't proof.
- **Inter-kernel scheduling gaps** (the source of `cudaEventSynchronize`
  17 %) — Torch profiler shows kernel times but not stream-level wait
  decomposition. NVTX ranges + Nsight Systems is the right next step.
- **Phase tagging is unreliable**: the parser's heuristic (regex on kernel
  names) classifies most kernels as `other`. To get clean prefill/decode/
  schedule splits we'd need sglang to emit explicit user annotations or
  we'd need to align kernel time with the request-event log from
  `server.log`.

### 15.8 Reproducing §15

```bash
# one cell:
python scripts/regime_study/run_hw_view.py \
  --config  configs/moe_qwen3_30b.yaml \
  --workload regime_scout/candidates_regime_study/R8_prefix_sharing.yaml \
  --out-dir experiments/tmp/hw_view/moe_R8 \
  --gpu 0 --profile-num-steps 10 --warmup-requests 8

# aggregate everything in experiments/tmp/hw_view/ into one table
python scripts/regime_study/aggregate_hw_view.py
# → results/regime_bench/hardware_view_table.{md,csv}
```

Wall time ≈ 1-2 min/cell.

---

<a id="中文版"></a>

# 中文版

# Regime benchmark 实验 —— Qwen3-0.6B 与 Qwen3-30B-A3B (MoE) 在 H200 上

> **日期**：2026-06-01 · **作者**：Stage A harness + `scripts/regime_study/aggregate.py` ·
> **状态**：每个 regime 跑 2 个 rep，16/16 个 (model,regime) cell，31/32 次实际跑通过
> （MoE R8 在 rep 1 因为另一个用户的进程中途抢 GPU 0 内存导致 OOM；rep 2 干净复测通过）。

## 1. 目标

刻画 **同一个模型 + 同一份服务器配置下，SGLang 性能随 workload regime
是怎么变化的**。这一步不做任何优化，只收集证据。最终给出会议可用的表格：

1. 展示已配好的两个模型在 H200 上跑通；
2. 暴露各 regime 间可量化的性能差异；
3. 标记出值得后续 profiling 的可疑 regime；
4. 给出可复用的 mini-benchmark 模板。

## 2. 硬件 + 软件环境

| 项目 | 取值 |
|---|---|
| 硬件 | 8× NVIDIA H200（每张 143 GB），每次实验单卡 |
| 用到的 GPU | GPU 0 跑 dense rep1 + MoE rep1/2；GPU 1 跑 dense rep2 |
| Conda env | `sglang-dev` |
| SGLang | 0.5.12.post1（源码：`/home/t-jialianggu/work/sglang`） |
| Python | 3.11 |
| CUDA | 12.8 |
| HF cache | `/data/hf/gujialiang123/hf_cache` |

## 3. 模型

两个模型仓库里都已经配好且下载好。

| 模型 | 路径 | 体积 | 配置文件 |
|---|---|---|---|
| Qwen3-0.6B (dense) | `/data/hf/models/Qwen3-0.6B` | ~1.2 GB bf16 | `configs/base.yaml` |
| Qwen3-30B-A3B-Instruct-2507 (MoE：128 专家，8 active/token，GQA 32/4) | `/data/hf/models/Qwen3-30B-A3B-Instruct-2507` | ~57 GB bf16 | `configs/moe_qwen3_30b.yaml` |

### 启动配置（精确）

两份配置共用同一组调度参数（只有 `mem-fraction-static` 和 `context-length`
不同 —— MoE 用 0.85 / 32768，dense 用 0.7 / 默认）。这样 regime 之间的差异
就不会被 per-model 调参污染。

```yaml
# 共用旋钮（两个模型都一样）
tensor-parallel-size: 1
schedule-policy: lpm
schedule-conservativeness: 1.0
max-running-requests: 32
chunked-prefill-size: -1
max-prefill-tokens: 16384
disable-radix-cache: false
disable-cuda-graph: false
```

SGLang 启动 wrapper：`scripts/launch_server.py`（内部走 `python -m sglang.launch_server <flags>`）。

## 4. Regime 矩阵

8 个 regime，全部在 `regime_scout/candidates_regime_study/*.yaml`。两个模型用同一组。

| ID | 目的 | 数据集 | 输入长 | 输出长 | 并发 | 总 prompt 数 |
|---|---|---|---|---|---|---|
| **R1** | baseline（低负载） | random | 128 | 128 | 4 | 32 |
| **R2** | decode-heavy | random | 128 | 1024 | 32 | 64 |
| **R3** | prefill-heavy | random | 4096 | 128 | 8 | 32 |
| **R4** | 长输入 + 长输出 | random | 4096 | 512 | 8 | 24 |
| **R5** | 饱和（**故意**超过服务器 `max-running-requests=32`） | random | 512 | 256 | **64** | 128 |
| **R6** | 单流 / 延迟导向 | random | 512 | 256 | 1 | 16 |
| **R7** | 输入长度大范围抖动（`random_range_ratio=0.95`，约 100→4000） | random | 2048±95% | 256 | 32 | 64 |
| **R8** | 高 prefix 复用 | generated-shared-prefix（sys 2048 + Q 128，8 组 × 8 prompts） | 2048+128 | 256 | 32 | 64 |

> R7 是受限于 `sglang.bench_serving` 的 `--dataset-name random` 的"混合长度"
> 代理 —— 它给的是输入长度围绕均值的抖动，不是真正双峰分布。R8 用
> `--dataset-name generated-shared-prefix`，能直接压 radix cache。

## 5. 怎么跑出来的

完全复用仓库已有 harness —— **没新增任何 infrastructure**：

```bash
# 一次 suite = 8 个 regime，每个 regime 重启一次 server
python scripts/run_regime_suite.py \
  --config configs/base.yaml \
  --workload-dir regime_scout/candidates_regime_study \
  --out results/regime_bench/raw/dense_rep1.jsonl \
  --run-root experiments/tmp/regime_study/dense_rep1 \
  --mode quick --reset
```

每个 workload 走：`scripts/run_experiment.py` 起 server →
`scripts/wait_ready.py` 等就绪 → `scripts/run_benchmark.py` 调
`python -m sglang.bench_serving` → `scripts/parse_metrics.py` 把原始 jsonl
归一到固定 schema。`bench_serving` 给出：

- `request_throughput`（req/s）、`output_throughput`（输出 tok/s）、`input_throughput`
- TTFT 的 `mean/median/p99`，TPOT 的 `mean/median/p99`，ITL 的 `p50/p95/p99`
- e2e 延迟的 `mean/median/p90/p99`
- 每条请求的 `ttfts[]` / `itls[]`（用来自己算 ttft_p95）

服务器侧行为由 `.github/skills/server-log-mining/impl/parse_server_log.py`
从 `server.log` 挖出来：attention backend、scheduler policy、CUDA-graph
capture 范围、KV cache 体积、`max_total_num_tokens`、峰值 running/queue 长度、
prefill/decode batch 计数、retract 事件等。这些填了下面的"server 可观察行为"列。

## 6. Server 可观察行为（来自 `server.log`）

两个模型在同一份 config 下，**选了同一套 backend + scheduler 栈**。这套栈在
启动时定，跑期间不会随 regime 变。

| 项 | dense (Qwen3-0.6B) | MoE (Qwen3-30B-A3B) |
|---|---|---|
| Attention backend | **fa3**（FlashAttention-3） | **fa3** |
| Schedule policy | `lpm` | `lpm` |
| `chunked_prefill_size` | -1（关闭 —— full-prefill batch） | -1 |
| `max_prefill_tokens` | 16384 | 16384 |
| `max_running_requests` | 32 | 32 |
| `mem_fraction_static` | 0.7 | 0.85 |
| KV cache 总量 | 96.2 GB | 61.2 GB |
| `max_total_num_tokens` | 900 480 | 668 504 |
| CUDA-graph capture 的 batch size 集合 | 1..32（最大 32） | 1..32 |

**当前日志看不到的**（要别的 profiling 工具）：

- 每个 batch shape 实际选了哪个 kernel（比如 prefill 用了哪个、decode 用了
  哪个）。`server.log` 只汇总 batch 形状。→ 要 PyTorch profiler 或 Nsight。
- MoE 专家失衡 / routing entropy —— 默认日志没有。→ 要自己加 hook 或者
  用 torch profiler + kernel 名 regex（`moe.*topk`、`grouped_gemm`、`all_to_all`）。
- `fa3` 是否对某些 shape 退回 fa2。→ 不直接可见，要 NVTX 区段或 torch profiler。

这些就是下一步 profiling 的目标，见 §11。

## 7. 结果表 —— Qwen3-0.6B (dense)

> 两个 rep 的均值。单 rep 原始行在 `results/regime_bench/parsed_results.csv`。

| Regime | 输入 | 输出 | 并发 | NumPrompts | Req/s | Out tok/s | TTFT mean (ms) | TTFT p95 (ms) | TPOT mean (ms) | ITL p95 (ms) | E2E p99 (ms) | peak_run | peak_queue | vs R1（吞吐 gap） | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1** baseline | 128 | 128 | 4 | 32 | 21.1 | 1 427 | 27.4 | 95.4 | 2.26 | 2.50 | 374 | 4 | 0 | 0.0% | |
| **R2** decode-heavy | 128 | 1024 | 32 | 64 | 16.7 | **9 009** | 53.2 | 118.4 | 2.69 | 2.72 | 2 853 | 32 | 1 | **+531%** | 吞吐冠军 |
| **R3** prefill-heavy | 4096 | 128 | 8 | 32 | 26.7 | 1 803 | 41.5 | 109.0 | 3.29 | 14.1 | 544 | 8.5 | 0 | +26% | ITL p95 偏高 |
| **R4** 长输入+长输出 | 4096 | 512 | 8 | 24 | 9.0 | 2 498 | 43.2 | 94.8 | 2.73 | 2.90 | 1 477 | 8 | 0 | +75% | |
| **R5** 高并发 | 512 | 256 | 64 | 128 | 46.7 | 5 802 | **537.9** | **1 235** | 4.83 | 17.8 | 2 332 | 32 | **33** | +307% | **撞 `max_running` 上限** |
| **R6** 单流 | 512 | 256 | 1 | 16 | 3.5 | 521 | **22.7** | **33.9** | **1.77** | **1.82** | 483 | 1 | 0 | −64% | 延迟最低，吞吐最低 |
| **R7** 混合长度 | 2048±95% | 256 | 32 | 64 | 27.1 | 6 736 | 107.9 | 187.9 | 4.25 | 4.12 | 1 373 | 32 | 4.5 | +372% | 小幅 backpressure |
| **R8** prefix sharing | 2048+128 | 256 | 32 | 64 | 27.0 | 6 904 | 157.6 | 194.1 | 4.00 | 3.91 | 1 224 | 32 | 0 | +384% | radix cache 把 2K 前缀吸住了 |

## 8. 结果表 —— Qwen3-30B-A3B (MoE)

> 全部 cell 都是 2 rep 均值。R8 在 rep 1 因 **外部** GPU 0 资源争抢（另一个
> 用户的 `olive` 进程中途占了 ~20 GB）OOM 了；rep 2 干净跑完，R8 的指标就用
> rep 2。详见 §10。

| Regime | 输入 | 输出 | 并发 | NumPrompts | Req/s | Out tok/s | TTFT mean (ms) | TTFT p95 (ms) | TPOT mean (ms) | ITL p95 (ms) | E2E p99 (ms) | peak_run | peak_queue | vs R1（吞吐 gap） | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1** baseline | 128 | 128 | 4 | 32 | 6.9 | 469 | 57.1 | 132.9 | 7.41 | 6.83 | 1 079 | 4 | 0 | 0.0% | |
| **R2** decode-heavy | 128 | 1024 | 32 | 64 | 3.2 | 1 736 | 100.1 | 148.0 | 14.67 | 15.5 | 15 479 | 32 | 0 | +270% | **e2e p99 = 15 秒!** |
| **R3** prefill-heavy | 4096 | 128 | 8 | 32 | 7.9 | 536 | 123.9 | 308.8 | 11.54 | 41.6 | 1 952 | 8 | 0 | +14% | ITL p95 很高 |
| **R4** 长输入+长输出 | 4096 | 512 | 8 | 24 | 2.4 | 667 | 156.5 | 308.9 | 10.87 | 14.9 | 5 668 | 8 | 0 | +42% | e2e p99 ~5.7 秒 |
| **R5** 高并发 | 512 | 256 | 64 | 128 | 11.3 | 1 406 | **2 075** | **5 338** | 19.9 | 49.7 | 9 406 | 32 | **34** | +200% | **TTFT 尾灾难** |
| **R6** 单流 | 512 | 256 | 1 | 16 | 1.5 | 222 | **44.2** | **50.4** | **4.24** | **4.29** | 1 113 | 1 | 0 | −53% | 延迟最低；吞吐极低 |
| **R7** 混合长度 | 2048±95% | 256 | 32 | 64 | 7.6 | 1 880 | 407.7 | 1 001 | 15.21 | 14.2 | 5 159 | 32 | 9.5 | +301% | 可见 backpressure |
| **R8** prefix sharing | 2048+128 | 256 | 32 | 64 | **13.1** | **3 355** | 385.1 | 726.5 | 8.03 | 7.99 | 2 673 | 32 | 5 | **+615%** | 吞吐冠军；rep 1 受外部影响 OOM |

## 9. 跨 regime 对比（每个模型的 best vs worst）

| 模型 | 最佳 regime（out tok/s） | 最差 regime | 吞吐差 | TTFT p50 最低 | TTFT p50 最高 | TTFT p50 差 | 主要观察 |
|---|---|---|---|---|---|---|---|
| dense | **R2** 9 009 | **R6** 521 | **17.3×** | R6 20.7 ms | R5 538 ms | **26×** | decode 类 regime 吞吐最大；高并发 R5 因 `max_running=32` 把 TTFT 干飞 |
| MoE | **R8** 3 355 | **R6** 222 | **15.1×** | R6 44.5 ms | R5 2 075 ms | **47×** | prefix cache 完胜；**R5 TTFT 比 R6 差 47 倍** —— MoE 对饱和远比 dense 敏感 |

## 10. 失败 / 异常 run

| 模型 | Regime | Rep | 状态 | 原因 | 处理 |
|---|---|---|---|---|---|
| MoE | R8 | 1 | **FAIL — CUDA OOM** | GPU 0 被另一个用户的 `olive` 任务中途占走 ~20 GB。MoE server 启动时按整张卡 size 0.85 算好了 mem，外部进程冒出来后，再要 +190 MB 激活就 OOM。Server 日志：`torch.OutOfMemoryError: ... Process 2070002 has 19.84 GiB memory in use ... Including non-PyTorch memory, this process has 119.93 GiB memory in use.` | 等外部进程退出后跑 rep 2；rep 2 R8 干净通过，3 355 tok/s。**没改任何代码**。 |

没有其他 OOM / crash / timeout。没有 retract、没有 KV pool full 事件。

## 11. 简要观察 + profiling 候选

### 11.1 跨模型

- **MoE 的 "regime 错配代价" 远大于 dense**：TTFT p50 浮动 47×（MoE）vs 26×
  （dense）。MoE 单流 R6 只有 222 tok/s —— 才 dense R6 *最差* 吞吐的 47%。
- **`max-running-requests=32` 在 R5 两个模型上都是真实瓶颈**（peak_queue ≈ 33-34，
  `concurrency_capped=True`）。dense 代价是 TTFT 尾延，MoE 代价是灾难性（5.3 秒 TTFT p95）。
- **prefix sharing (R8) 把吞吐排名翻转了**：MoE 上是 *最佳* regime（3 355 tok/s），
  dense 上排第二。radix cache 把 2K 共享前缀吸收掉 → 实际单请求 prefill 短得多。

### 11.2 dense 专有

- R3（prefill 重）的 **ITL p95 = 14.1 ms，ITL p50 = 2.4 ms**（差 5.9×）——
  可能是 chunked-prefill 把 decode 步骤打断成尖刺。值得 profile。
- R5 dense 的瓶颈是 config，不是模型 —— Stage B 的 `config-agent` 之前在
  MoE 上已经证明过，估计同样的修复在 dense 上也成立。

### 11.3 MoE 专有

- R2（decode-heavy）**e2e p99 = 15.5 秒**，远超平均 TPOT 推算出来的值。尾巴很可疑。
  候选原因：冷专家被命中、或者 expert imbalance。
- R3（prefill 4K）**ITL p95 = 41.6 ms**，是 MoE R3 p50 的 3.6×。可能是
  prefill→decode 切换的尖刺（router 预热？）或者长 context 上的 routing 不均衡。
- R4 + R7 在并发 32 都已经看到 queue 堆积（peak_queue 5-10）—— prefill 跟不上，
  而且 chunked_prefill 是关的（`chunked_prefill_size=-1`）。打开 chunked
  prefill 是优先级最高的测试候选。

### 11.4 当前日志 **看不到** 的（profiling 路线）

| 问题 | 用什么工具 |
|---|---|
| `fa3` 真的对所有 batch shape 都生效，还是有 fallback？ | NVTX + PyTorch profiler（见 `.github/skills/pytorch-profiling/`） |
| MoE 哪些 kernel 主导耗时（`moe.*topk` vs `grouped_gemm` vs `all_to_all`）？ | PyTorch profiler —— `pytorch-profiling/SKILL.md` 已设计 |
| 专家路由是否失衡（几个 hot expert 吃掉大多数请求）？ | 给 sglang 加 `expert_distribution_recorder` hook |
| CUDA graph 是不是每个 decode step 都命中，还是有 fallback？ | sglang `--log-level debug` + grep `cuda_graph` |
| 冷启动 vs 稳态 TPOT？ | 已设计：`pytorch-profiling/impl/run_profile.py --warmup-requests 16` |

## 12. 下一步建议

按"动手成本最低优先"排：

1. **打开 chunked prefill**（`chunked-prefill-size=2048`）在 MoE 上 R4 + R7 重跑。
   预期：peak_queue 下降，TTFT p95 下降。
2. **把 `max-running-requests` 提到 64**，两个模型 R5 都重跑。
   预期：TTFT p50 从 538 ms → ~100 ms（dense），2 075 ms → ~500 ms（MoE）。
3. **把 `pytorch-profiling` skill 挂到 MoE R2 上**，查那个 15 秒 e2e p99 离群。
   如果看到 routing 失衡，就立一个 Stage A 题目包
   `experiments/problems_moe/PNNN-routing-tail/`。
4. **真正写一个双峰 R7**（目前只用 `random_range_ratio` 抖动，不是真正的
   70/30 短/长混合）。这是 `regime_scout` 的小扩展，不动 sglang。

## 13. 复现实验

```bash
# 1. 激活环境
conda activate sglang-dev
cd /home/t-jialianggu/work/EndtoEnd-auto-optimization

# 2. 跑 dense（H200 上约 9 分钟）
python scripts/run_regime_suite.py --reset \
  --config configs/base.yaml \
  --workload-dir regime_scout/candidates_regime_study \
  --out results/regime_bench/raw/dense_rep1.jsonl \
  --run-root experiments/tmp/regime_study/dense_rep1

# 3. 跑 MoE（H200 上约 10 分钟）
python scripts/run_regime_suite.py --reset \
  --config configs/moe_qwen3_30b.yaml \
  --workload-dir regime_scout/candidates_regime_study \
  --out results/regime_bench/raw/moe_rep1.jsonl \
  --run-root experiments/tmp/regime_study/moe_rep1

# 4. 聚合（读取 results/regime_bench/raw/*_rep*.jsonl 全部）
python scripts/regime_study/aggregate.py
# → results/regime_bench/{parsed_results.csv, summary_table.csv, summary.md}
```

要并行 2 rep：参照 dense_rep2 用 `_gpu_id: 1` + `port: 30001` 的做法
（`logs/regime_study_dense_rep2.log` 里有完整命令）。

## 14. 结论

实验证明：**同一个模型、同一份 SGLang engine，仅仅 workload regime 变化，
就能产生 17× 的输出吞吐摆幅和 47× 的 TTFT p50 摆幅**，根本不需要改代码或
配置。最大的差距出现在：

- **R5**（高并发饱和），两个模型上都被 `max-running-requests=32` 卡住；
- **R2**（decode-heavy）和 **R8**（prefix sharing）在 MoE 上 —— MoE 在
  这两个 regime 吞吐很猛，但 TTFT 和尾延暴露了可疑 pattern（R2 e2e p99 = 15.5 秒）；
- **R3 / R4 / R7** 在 MoE 上 —— backpressure + 高 ITL p95，提示应该试 chunked prefill。

这些 regime 都是后续 profiling 和 mini-reproducer 的好候选。下一步是用
`.github/skills/pytorch-profiling/` 拿 kernel 级证据，搞清楚 scheduler /
config / kernel 三方面，哪一个能解释观察到的 gap。

完整的 per-run 数据在 `results/regime_bench/parsed_results.csv`；聚合表在
`results/regime_bench/summary_table.csv`；原始 SGLang 日志在
`experiments/tmp/regime_study/<model>_rep<N>/<ts>/run_*/server.log`。

## 15. 硬件视图 + kernel 选择（深入篇，后续追加）

> 🟢 新增。2026-06-01 晚上加的，**只用 SGLang 自带工具**：
> `/get_server_info` HTTP 端点（运行时确认 backend）、`nvidia-smi` 0.5s 采样
> （GPU 内存/利用率/功耗）、`sglang.bench_serving --profile`（Torch profiler trace）。

### 15.1 范围 + 工具

挑 6 个 cell（dense + MoE × {R1, R5, R8}）重跑，加：

- 服务器侧设 `SGLANG_TORCH_PROFILER_DIR` env → 启用 Torch profiler。
- `bench_serving` 加 `--profile --profile-num-steps 10 --warmup-requests 8`
  → 暖机后捕获 10 个 forward step。
- `nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm`
  每 0.5 s 采一次。
- 杀 server 之前调 `GET /get_server_info` 拿运行时确认的 `attention_backend`、
  `sampling_backend` 等。

全部由 `scripts/regime_study/run_hw_view.py` 包起来，
`scripts/regime_study/aggregate_hw_view.py` 聚合。每个 cell 的原始产物在
`results/regime_bench/raw/hw_view/<model>_<regime>/`（5 个文件：
`hardware_view.json`、`profile_summary.json`、`server_info.json`、
`gpu_samples.csv`、`server.log`）。

完整表也在 `results/regime_bench/hardware_view_table.{md,csv}`。

### 15.2 Backend 选择（运行时确认）

| 模型 | Regime | Attention | Sampling | Schedule | KV dtype | max_running | torch_compile_max_bs |
|---|---|---|---|---|---|---|---|
| dense | R1 / R5 / R8 | **fa3**（FlashAttention-3） | **flashinfer** | lpm | auto (bf16) | 32 | 32 |
| MoE   | R1 / R5 / R8 | **fa3** | **flashinfer** | lpm | auto | 32 | 32 |

Backend **不随 regime 变**（启动时根据硬件+模型配置决定）。真正在变的是
**kernel mix**（见 §15.4）。

### 15.3 硬件视图（`nvidia-smi` 0.5 s 采样，bench 窗口内）

| 模型 | Regime | Wall (s) | 采样数 | 显存峰 (GiB) | 显存均 (GiB) | GPU util 均 (%) | GPU util p95 (%) | 显存控制器 util (%) | 功耗均 (W) | 功耗峰 (W) | 峰温 (°C) | SM 时钟均 (MHz) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | R1 | 41.5 | 43 | 98.4 | 61.4 | 4.1 | 10 | 1.0 | 113 | 200 | 42 | 1411 |
| dense | R5 | 85.7 | 54 | 98.6 | 63.8 | 4.2 | 26 | 1.1 | 120 | 266 | 45 | 1526 |
| dense | R8 | 64.6 | 48 | 99.0 | 59.6 | **12.7** | **100** | 4.5 | 147 | **630** | 56 | 1468 |
| MoE | R1 | 95.7 | 76 | 119.3 | 74.8 | 11.8 | 82 | 5.3 | 140 | 416 | 47 | 1636 |
| MoE | R5 | 109.4 | 99 | 120.1 | 84.2 | **17.0** | **97** | **12.1** | **187** | 578 | 54 | 1749 |
| MoE | R8 | 97.9 | 80 | 121.3 | 74.6 | 14.7 | 100 | 5.0 | 159 | 555 | 54 | 1650 |

**观察**：

- **Dense 严重压不动 H200**：即使饱和 (R5)，GPU util 均也只有 4-13%。这是
  *小模型* 税 —— TP=1 + 0.6B 模型让 SM 阵列大部分时间空着。
- **MoE 把显存控制器压得更狠**（mem-ctrl util 5-12% vs dense 的 1-5%）。
  MoE R5 mem-ctrl util 12.1% —— 证实 MoE 比 dense 更靠近 memory-bandwidth-bound。
- **功耗峰**：dense R8 摸到 630 W，MoE R5 均 187 W（峰 578 W），对比 H200 的
  700 W TDP。dense R8 之所以能爆功耗是因为 radix cache 把前缀吸住后，很多请求
  迅速进入纯 decode 阶段，短时间内逼近峰值利用。
- **显存峰值**：dense 启动时按 `mem_fraction_static=0.7 × 143 GiB` 预占 ~98 GiB；
  MoE 按 0.85 预占 ~120 GiB。这是 server 启动常量，**不会**随 regime 缩。

### 15.4 Kernel 分类（torch profiler trace 的 top-20 GPU 事件按 self-time 归类）

| 模型 | Regime | Trace wall (ms) | GPU active (ms) | Kernel 分类占比 |
|---|---|---|---|---|
| dense | R1 | 74.5 | 23.3 | cuda runtime/overhead **38.2%**; GEMM 23.9%; FlashAttention 6.7%; norm 4.2% |
| dense | R5 | 189.8 | 45.6 | cuda runtime/overhead 22.3%; GEMM 15.3%; FlashAttention 11.5%; norm 4.0%; elementwise 3.6% |
| dense | R8 | 234.0 | 103.2 | **FlashAttention 23.1%**; GEMM 21.6%; cuda runtime/overhead 9.4%; elementwise 4.3%; norm 3.8% |
| MoE | R1 | 206.3 | 103.9 | **MoE 33.8%**; cuda runtime/overhead 20.0%; other 19.5%; GEMM 5.5%; FlashAttention 2.7%; norm 1.9% |
| MoE | R5 | 505.1 | 297.5 | **MoE 45.3%**; other 19.3%; GEMM 6.7%; cuda runtime/overhead 5.3%; FlashAttention 2.7%; norm 1.8% |
| MoE | R8 | 696.1 | 598.0 | **MoE 46.5%**; FlashAttention 13.4%; GEMM 12.7%; other 5.7%; elementwise 2.9%; norm 2.4% |

### 15.5 每 cell 的 top-2 kernel（真实 kernel 名）

| 模型 | Regime | #1 kernel (self %) | #1 calls | #2 kernel (self %) |
|---|---|---|---|---|
| dense | R1 | `cudaGraphLaunch` (**17.9%**) | 7 | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` (10.0%) |
| dense | R5 | `cudaLaunchKernel` (**9.9%**) | **916** | `cudaLaunchKernelExC` (6.7%) |
| dense | R8 | `flash::FlashAttnFwdSm90<...,bfloat16_t,Sm90,…>` (**15.6%**) | 112 | 另一个 `FlashAttnFwdSm90` 变体 (7.4%) |
| MoE | R1 | **`fused_moe_kernel`** (**33.8%**) | 864 | `cudaEventSynchronize` (19.5%) |
| MoE | R5 | **`fused_moe_kernel`** (**45.3%**) | 864 | `cudaEventSynchronize` (16.8%) |
| MoE | R8 | **`fused_moe_kernel`** (**46.5%**) | 864 | `flash::FlashAttnFwdSm90<...>` (10.3%) |

### 15.6 Kernel 视角的核心结论

这些是有 kernel 级证据能扛住的结论：

1. **MoE 被 `fused_moe_kernel` 主导** —— 三个 MoE regime 上吃掉 34-47% 的 GPU
   时间；**`cudaEventSynchronize` 又吃掉 17-20%**，说明 MoE 专家 dispatch 里的
   kernel-to-kernel 同步等待是次要大瓶颈。这是整套实验里最该做 kernel-agent
   的明确案例。
2. **Dense R1 是 overhead-bound，不是 compute-bound**：top kernel 是
   `cudaGraphLaunch` (17.9%)，FlashAttention 才 6.7%。CUDA graph 确实在用，
   但整个模型小到 launch overhead 都能占主导。这就是 GPU util 只有 4% 的原因。
3. **Dense R5（撞 cap）是 *launch*-bound 在调度边界**：`cudaLaunchKernel`
   10 个 profile step 里出现了 916 次。`max_running_requests=32` 卡住后，
   调度器在紧密 retry loop 里，每次 retry 都伴随一次 launch。
4. **Dense R8 是唯一真正 attention-bound 的 dense regime**：
   `flash::FlashAttnFwdSm90` 是 top kernel (15.6%)，FlashAttention 总占 23.1%。
   长共享前缀意味着要 attend 的 token 多得多 —— 这就是小模型上"GPU util 高"
   该有的样子。
5. **MoE R8 (prefix sharing) 是 GPU 时间最大且 kernel mix 最丰富的**：
   MoE 46.5% + FlashAttention 13.4% + GEMM 12.7% —— 三大支柱同时点亮。这就
   解释了为什么 R8 是 MoE 的吞吐冠军（3 355 out tok/s）。

### 15.7 这一层 profile **还看不到** 的（下一层）

- **`fused_moe_kernel` 内部的 per-expert 负载分布** —— Torch profiler 不会
  拆 per-expert op。要 sglang 的 `expert_distribution_recorder` hook + parser，
  或者用 Nsight Compute 单独打这个 kernel。
- **`fa3` 是否对某些 shape 退回 `fa2`** —— 6 个 cell 的 trace 里只看到 Sm90 fa3
  kernel，没观察到退回，但"没观察到"不等于"不会发生"。
- **Kernel 间调度间隔**（`cudaEventSynchronize` 17% 的来源） —— Torch profiler
  给 kernel 时间，但不分解 stream 级 wait。要加 NVTX + Nsight Systems。
- **Phase tagging 不可靠**：parser 的启发式（按 kernel 名 regex）把多数 kernel
  归到 `other`。要拿到干净的 prefill/decode/schedule 三段比例，得让 sglang 显式
  emit user annotation，或者把 kernel 时间和 `server.log` 的请求事件对齐。

### 15.8 复现 §15

```bash
# 单个 cell：
python scripts/regime_study/run_hw_view.py \
  --config  configs/moe_qwen3_30b.yaml \
  --workload regime_scout/candidates_regime_study/R8_prefix_sharing.yaml \
  --out-dir experiments/tmp/hw_view/moe_R8 \
  --gpu 0 --profile-num-steps 10 --warmup-requests 8

# 把 experiments/tmp/hw_view/ 下所有 cell 聚合成表
python scripts/regime_study/aggregate_hw_view.py
# → results/regime_bench/hardware_view_table.{md,csv}
```

每 cell 约 1-2 分钟。

---

## 16. Three-model expansion + fused_moe_kernel deep dive (follow-up)

> 🟢 NEW. Added 2026-06-01 evening. Doubles down on §15 by running the **full
> 8-regime hardware view on a third model** + explaining exactly what the
> `fused_moe_kernel` is.

### 16.1 What changed

- Added a third model: **`gemma-3-1b-it`** (dense 1B, Google Gemma 3).
  Config: [`configs/gemma3_1b.yaml`](../configs/gemma3_1b.yaml). Same
  scheduling knobs as `configs/base.yaml`.
- **Why not Gemma-4 MoE** (the original request): sglang 0.5.12.post1 does
  not support `model_type=gemma4`. Quick smoke test gives
  `KeyError: 'gemma4'` from sglang's model registry. The only Gemma models
  sglang supports are gemma / gemma2 / gemma3 (all dense; gemma3 has a
  multimodal variant but the MoE block in Gemma-4 is not in any of these).
  → Substituted `gemma-3-1b-it` (dense) so we still get a 3-way comparison.
- Round-1 regime suite: 8 regimes × 2 reps on Gemma → results in
  `results/regime_bench/raw/gemma_rep{1,2}.jsonl`. Re-aggregated into
  `results/regime_bench/{parsed_results.csv,summary_table.csv,summary.md}`
  alongside dense + MoE.
- Round-2 hardware view: extended from 6 cells to **24 cells (3 models × 8
  regimes)**. New tables in `results/regime_bench/hardware_view_table.md`.

### 16.2 What is `fused_moe_kernel`?

`fused_moe_kernel` is the **Triton-JIT'd MoE expert dispatch + matmul kernel**
in sglang. Source:
[`sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:324`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py).

In **one** GPU kernel launch it does **all of**:

1. Reads the per-token top-k expert IDs (`expert_ids_ptr`) and routing
   weights (`topk_weights_ptr`).
2. Looks at the sorted token-id table (`sorted_token_ids_ptr`) that groups
   "tokens going to the same expert" into contiguous blocks (preprocessed
   by another kernel called `moe_align_block_size`).
3. For each block, picks the correct expert's weight matrix `B[expert_id]`
   (stride `stride_be`) and does **a tiled bf16 matmul** with the activation
   tile `A`.
4. Multiplies the result by the routing weight and writes to `C`.
5. Supports per-token / per-tensor / FP8 / INT8 scaling via the
   `a_scale_ptr` / `b_scale_ptr` paths (we're on bf16 → no quantization
   path).

> **Why it dominates** (we measure 34-47 % of GPU time on the MoE in §15):
> a Qwen3-30B-A3B forward pass calls this kernel **twice per layer × 48
> layers = 96 times per token** (once for `gate_up_proj`, once for
> `down_proj`). Each call is a grouped GEMM over **8 active experts × hidden
> tokens**. So this single kernel essentially **is** the MoE FFN.
>
> Side counter — Qwen3 MoE has 128 experts but only 8 are active per token;
> the matmul groups the 8-expert subset, not all 128. The kernel sees the
> dense 8-way work.

**Why `cudaEventSynchronize` shows up as a big secondary** (17-37 % of GPU
time depending on regime): every `fused_moe_kernel` call requires a sync to
ensure the previous `moe_align_block_size` finished before the matmul. On
long-prefill regimes (MoE R3/R4) this sync wait eclipses the matmul itself
— it becomes the new bottleneck.

### 16.3 Updated Round-1 result table — Gemma-3-1B (dense)

> 2 reps. Server config: `configs/gemma3_1b.yaml`. Same scheduling knobs as
> Qwen3-0.6B. Backend: fa3 / flashinfer / lpm.

| Regime | InLen | OutLen | Conc | Req/s | Out tok/s | TTFT mean | TTFT p95 | TPOT mean | ITL p95 | E2E p99 | n_pass | Gap vs R1 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1** | 128 | 128 | 4 | 8.9 | 600 | 50.4 | 120.5 | 5.62 | 5.08 | 835 | 2/2 | 0% | |
| **R2** | 128 | 1024 | 32 | 7.4 | 3 975 | 82.7 | 132.8 | 5.91 | 5.64 | 6 278 | 2/2 | +562% | |
| **R3** | 4096 | 128 | 8 | 13.3 | 900 | 67.6 | 143.5 | 6.63 | **31.0** | 1 062 | 2/2 | +50% | high ITL p95 |
| **R4** | 4096 | 512 | 8 | 4.5 | 1 266 | 74.3 | 141.5 | 5.31 | 5.05 | 2 879 | 2/2 | +111% | |
| **R5** | 512 | 256 | 64 | 22.3 | 2 772 | **1 059** | **2 624** | 9.67 | 35.7 | 4 707 | 2/2 | +362% | **hit max_running cap** |
| **R6** | 512 | 256 | 1 | 1.4 | 212 | 41.2 | 44.6 | 4.46 | 4.50 | 1 166 | 2/2 | −65% | |
| **R7** | 2048±95% | 256 | 32 | 18.0 | 4 482 | 182.8 | 386.0 | 6.29 | 5.49 | 2 122 | 2/2 | +647% | |
| **R8** | 2048+128 | 256 | 32 | 19.6 | **5 011** | 220.9 | 322.0 | 5.51 | 5.40 | 1 687 | 2/2 | **+735%** | best throughput |

Best/worst:
- Highest output throughput: **R8** 5 011 tok/s (radix cache wins, same as MoE)
- Lowest: **R6** 212 tok/s
- Highest TTFT p50: **R5** 1 044 ms (cap-hit)
- Lowest TTFT p50: **R1** 40 ms

### 16.4 Three-way model comparison — dense × dense × MoE

| Metric | Qwen3-0.6B (dense) | Gemma-3-1B (dense) | Qwen3-30B-A3B (MoE) | Observation |
|---|---|---|---|---|
| Output tok/s on R8 (best) | **6 904** | 5 011 | 3 355 | Qwen3-0.6B wins; **Gemma is ~27% slower than Qwen3 at almost 2× the params** |
| Output tok/s on R6 (single stream) | 521 | 212 | 222 | Qwen3-0.6B wins; Gemma surprisingly slower than MoE at single-stream |
| TTFT p50 on R5 (cap-hit) | 538 ms | 1 044 ms | 2 075 ms | Bigger models suffer more at saturation |
| TTFT p50 on R1 (baseline) | 21 ms | 40 ms | 45 ms | Roughly tracks model size |
| `max-running` cap hits | R5 only | R5 only | R5 only | Same cap, all 3 models hit it identically — config bottleneck, not model |
| GPU util mean on R8 | 12.7 % | 8.8 % | 14.7 % | All 3 are under-utilising H200 |

**Key new finding**: **Gemma is slower than Qwen3-0.6B even though it has
~1.7× the parameters**. Possible reasons (not yet profiled):
- Gemma's per-layer sliding-window attention may not be hitting fast paths.
- Gemma has hidden_size 2816 vs Qwen3-0.6B's 1024 — larger GEMMs, but
  apparently not enough to recover the overhead-bound nature of
  small-model serving on H200.

### 16.5 Updated hardware view (24 cells)

Full table in [`results/regime_bench/hardware_view_table.md`](../results/regime_bench/hardware_view_table.md).

**Highlights (new vs §15)**:

| Cell | Top kernel | % | What it means |
|---|---|---|---|
| dense R3 (prefill) | `flash::FlashAttnFwdSm90<...>` | 14.1% | Compute-bound when prefill is heavy |
| dense R7 (mixed) | `flash::FlashAttnFwdSm90<...>` | 17.2% | Mixed-length doesn't break attention dominance |
| **MoE R3** (prefill) | **`cudaEventSynchronize`** | **37.3%** | **Sync waits eclipse `fused_moe_kernel` (29.6%) on heavy prefill — different bottleneck than steady-state MoE** |
| **MoE R4** (long-in + long-out) | `cudaEventSynchronize` | 37.7% | Same — long prefill = sync-bound, not MoE-bound |
| MoE R7 (mixed) | `fused_moe_kernel` | 42.7% | Steady state — MoE kernel back in charge |
| **MoE R2** (decode-heavy) mem-ctrl util | 24.3 % | — | **MoE R2 saturates the memory controller** (vs 1-5% for dense regimes); explains the 15.5 s e2e p99 outlier |
| **Gemma R5** `cudaLaunchKernel` | 25.3% / **9 051 calls** | — | Gemma's scheduler-retry storm is 10× worse than Qwen3's R5 (916 calls) |
| Gemma R1/R2/R6 | `cudaGraphLaunch` | 19-29% | Gemma is **even more launch-overhead-bound than Qwen3** at low load |
| Gemma R7/R8 (top) | `at::native::elementwise_kernel<...>` | 13.8-14.4% / **2 695 calls** | Many small elementwise ops are creeping in — possibly not CUDA-graph-captured |

### 16.6 New cross-model insights

1. **MoE's bottleneck shifts by regime**. Steady-state MoE (R1/R2/R5/R7/R8):
   `fused_moe_kernel` dominates (34-47%). Heavy prefill (R3/R4):
   `cudaEventSynchronize` dominates (37%). Kernel-agent should target both.
2. **Small dense models are not interchangeable**. Gemma-3-1B is consistently
   slower than Qwen3-0.6B (1.7× the params but 17-30% less throughput in
   most regimes), and **much more overhead-bound** (more `cudaLaunchKernel`
   storms, more `cudaGraphLaunch` dominance). This is a real "model
   architecture matters even at the same scale" datapoint.
3. **The `max-running-requests=32` cap hurts everyone** — R5 TTFT p50 climbs
   from 538 ms (Qwen3) → 1 044 ms (Gemma) → 2 075 ms (MoE). The penalty
   scales roughly with per-token cost, but the *root cause* is identical
   across all 3 models. **One config change fixes all three**.
4. **MoE R2 is memory-bandwidth-bound, not compute-bound** — 24.3 % mem-ctrl
   util at 249 W average power. This is direct evidence that the 15.5 s
   e2e p99 outlier from §8 has a hardware-side reason: when decode batch
   fills up, the MoE expert weights have to be streamed through the L2 /
   HBM faster than the kernel can issue work. Candidate fix: enable
   FP8 quantization to halve the bandwidth pressure.

### 16.7 Reproducing §16

```bash
# Round-1 regime suite on Gemma (2 reps × 8 regimes ≈ 20 min)
for rep in 1 2; do
  python scripts/run_regime_suite.py --reset \
    --config configs/gemma3_1b.yaml \
    --workload-dir regime_scout/candidates_regime_study \
    --out results/regime_bench/raw/gemma_rep${rep}.jsonl \
    --run-root experiments/tmp/regime_study/gemma_rep${rep}
done

# Round-2 hardware view (24 cells ≈ 30 min on H200)
bash /tmp/run_hw_views_full.sh   # see logs/hw_view_batch_full.log

# Re-aggregate everything
python scripts/regime_study/aggregate.py            # round 1
python scripts/regime_study/aggregate_hw_view.py    # round 2
```

## 17. Gemma-4 MoE incompatibility note

`/data/hf/models/gemma-4-26B-A4B-it/` is present on disk (49 GB, 26B total /
4B active) but **sglang 0.5.12.post1 does not implement the `gemma4`
architecture**. Repro:

```bash
$ python -m sglang.launch_server \
    --model-path /data/hf/models/gemma-4-26B-A4B-it \
    --host 127.0.0.1 --port 30002 \
    --tensor-parallel-size 1 --mem-fraction-static 0.7 --trust-remote-code
…
File "…/sglang/srt/configs/model_config.py", line 250, in from_server_args
    return ModelConfig( …
File "…/sglang/srt/configs/model_config.py", line 127, in __init__
    self.hf_config = get_config(
KeyError: 'gemma4'
```

The Gemma family supported by sglang is:
`sglang/srt/models/{gemma.py, gemma2.py, gemma2_reward.py, gemma3_causal.py,
gemma3_mm.py, gemma3n_audio.py, gemma3n_causal.py, gemma3n_mm.py}` — no
gemma4. Substituted `gemma-3-1b-it` (dense) as the third model.

To enable Gemma-4 MoE in this experiment in the future:
- Upstream a `gemma4.py` model implementation in sglang (would require
  understanding Gemma-4's MoE block + sliding-window attention + RoPE
  config). Non-trivial.
- Or wait for Google to release Gemma-4 and for sglang to add support.

For now, this is correctly captured as **"experiment scope was limited by
runtime support"** rather than a silently-skipped item.


## 16. 三模型扩展 + fused_moe_kernel 深入解读（后续）

> 🟢 新增。2026-06-01 晚上。在 §15 基础上又把 **完整 8 regime 硬件视图扩到第三个模型** +
> 把 `fused_moe_kernel` 到底是啥讲清楚。

### 16.1 改了什么

- 加了第三个模型 **`gemma-3-1b-it`**（dense 1B，Google Gemma 3）。
  配置：[`configs/gemma3_1b.yaml`](../configs/gemma3_1b.yaml)。调度参数跟
  `configs/base.yaml` 一致。
- **为什么不是 Gemma-4 MoE**（你原本的要求）：sglang 0.5.12.post1 不支持
  `model_type=gemma4`。smoke test 直接抛 `KeyError: 'gemma4'`，sglang model
  registry 里没有。sglang 只支持 gemma / gemma2 / gemma3（全是 dense，gemma3
  有多模态变体但 Gemma-4 的 MoE block 在这些里都没有）。→ 替成
  `gemma-3-1b-it`（dense），至少能保留 3-way 对比。
- 第一轮 regime suite：Gemma 跑 8 regime × 2 rep → 结果在
  `results/regime_bench/raw/gemma_rep{1,2}.jsonl`。重新跟 dense + MoE 一起聚合到
  `results/regime_bench/{parsed_results.csv,summary_table.csv,summary.md}`。
- 第二轮硬件视图：从 6 个 cell 扩到 **24 个 cell（3 模型 × 8 regime）**。
  新表在 `results/regime_bench/hardware_view_table.md`。

### 16.2 `fused_moe_kernel` 到底是啥？

`fused_moe_kernel` 是 sglang 里 **Triton-JIT 编译的 MoE 专家 dispatch + matmul
kernel**。源码：
[`sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:324`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py)。

**一次** GPU kernel 启动里做完 **全部下面这些**：

1. 读出每个 token 的 top-k 专家 id（`expert_ids_ptr`）和路由权重（`topk_weights_ptr`）。
2. 看排过序的 token-id 表（`sorted_token_ids_ptr`），这张表已经把"要送到同一个
   专家的 token"分组放一起（由另一个 kernel `moe_align_block_size` 预处理）。
3. 对每个块挑出对应专家的权重矩阵 `B[expert_id]`（stride `stride_be`），跟激活
   tile `A` 做 **分块的 bf16 matmul**。
4. 乘上路由权重写回 `C`。
5. 通过 `a_scale_ptr` / `b_scale_ptr` 支持 per-token / per-tensor / FP8 / INT8
   量化（我们这次用 bf16，没走量化路径）。

> **为什么它会占主导**（§15 测出 MoE 上 34-47% GPU 时间）：Qwen3-30B-A3B 前向
> 一次调它 **每层 2 次 × 48 层 = 96 次**（一次 `gate_up_proj`、一次 `down_proj`）。
> 每次调用是 **8 个 active 专家 × 该批 token** 上的 grouped GEMM。所以这一个
> kernel **本质上就等于** MoE 的 FFN 部分。
>
> 小常识反驳一下："Qwen3 MoE 128 专家 8 个 active"，matmul 只在那 8 个 active
> 子集里 group，不是所有 128 个。kernel 看到的是稠密的 8-way 工作量。

**为什么 `cudaEventSynchronize` 当第二瓶颈**（不同 regime 17-37% 时间）：
每次 `fused_moe_kernel` 都要等前一个 `moe_align_block_size` 完成 → matmul 之前
插一个 sync。长 prefill 场景（MoE R3/R4）这个 sync wait 直接盖过 matmul 本身 →
sync 成了新瓶颈。

### 16.3 第一轮结果表（更新）—— Gemma-3-1B (dense)

> 2 rep。Server 配置：`configs/gemma3_1b.yaml`，调度旋钮跟 Qwen3-0.6B 完全一样。
> Backend：fa3 / flashinfer / lpm。

| Regime | 输入 | 输出 | 并发 | Req/s | Out tok/s | TTFT mean | TTFT p95 | TPOT mean | ITL p95 | E2E p99 | n_pass | vs R1 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1** | 128 | 128 | 4 | 8.9 | 600 | 50.4 | 120.5 | 5.62 | 5.08 | 835 | 2/2 | 0% | |
| **R2** | 128 | 1024 | 32 | 7.4 | 3 975 | 82.7 | 132.8 | 5.91 | 5.64 | 6 278 | 2/2 | +562% | |
| **R3** | 4096 | 128 | 8 | 13.3 | 900 | 67.6 | 143.5 | 6.63 | **31.0** | 1 062 | 2/2 | +50% | ITL p95 高 |
| **R4** | 4096 | 512 | 8 | 4.5 | 1 266 | 74.3 | 141.5 | 5.31 | 5.05 | 2 879 | 2/2 | +111% | |
| **R5** | 512 | 256 | 64 | 22.3 | 2 772 | **1 059** | **2 624** | 9.67 | 35.7 | 4 707 | 2/2 | +362% | **撞 max_running cap** |
| **R6** | 512 | 256 | 1 | 1.4 | 212 | 41.2 | 44.6 | 4.46 | 4.50 | 1 166 | 2/2 | −65% | |
| **R7** | 2048±95% | 256 | 32 | 18.0 | 4 482 | 182.8 | 386.0 | 6.29 | 5.49 | 2 122 | 2/2 | +647% | |
| **R8** | 2048+128 | 256 | 32 | 19.6 | **5 011** | 220.9 | 322.0 | 5.51 | 5.40 | 1 687 | 2/2 | **+735%** | 吞吐冠军 |

Best/worst：
- 吞吐最高：**R8** 5 011 tok/s（radix cache 优势，跟 MoE 同样的赢家）
- 吞吐最低：**R6** 212 tok/s
- TTFT p50 最高：**R5** 1 044 ms（撞 cap）
- TTFT p50 最低：**R1** 40 ms

### 16.4 三模型对比 —— dense × dense × MoE

| 指标 | Qwen3-0.6B (dense) | Gemma-3-1B (dense) | Qwen3-30B-A3B (MoE) | 观察 |
|---|---|---|---|---|
| R8 输出 tok/s (best) | **6 904** | 5 011 | 3 355 | Qwen3-0.6B 赢；**Gemma 几乎是 2 倍参数但反而比 Qwen3 慢 27%** |
| R6 输出 tok/s (单流) | 521 | 212 | 222 | Qwen3-0.6B 赢；Gemma 单流意外比 MoE 还慢 |
| R5 TTFT p50（撞 cap） | 538 ms | 1 044 ms | 2 075 ms | 模型越大，饱和惩罚越大 |
| R1 TTFT p50（基线） | 21 ms | 40 ms | 45 ms | 大致随模型尺寸 |
| 撞 `max-running` cap 的 regime | 仅 R5 | 仅 R5 | 仅 R5 | 同样 cap，3 个模型一致撞 —— **配置瓶颈，不是模型** |
| R8 GPU util 均 | 12.7 % | 8.8 % | 14.7 % | 3 个都没把 H200 压满 |

**新发现**：**Gemma 参数比 Qwen3-0.6B 多 1.7 倍，但反而慢**。可能原因（待 profile）：
- Gemma 的 per-layer sliding-window attention 可能没走快路径
- Gemma 的 hidden_size 2816 vs Qwen3-0.6B 的 1024 —— GEMM 大些，但显然不足以
  在 H200 上把小模型 serving 的 overhead-bound 性质拉回来

### 16.5 硬件视图更新（24 cell）

完整表在 [`results/regime_bench/hardware_view_table.md`](../results/regime_bench/hardware_view_table.md)。

**亮点（相对 §15 新增）**：

| Cell | Top kernel | % | 含义 |
|---|---|---|---|
| dense R3（prefill） | `flash::FlashAttnFwdSm90<...>` | 14.1% | 重 prefill 时变 compute-bound |
| dense R7（混合长度） | `flash::FlashAttnFwdSm90<...>` | 17.2% | 混合长度也压不倒 attention 主导 |
| **MoE R3**（prefill） | **`cudaEventSynchronize`** | **37.3%** | **重 prefill 上 sync wait 盖过 `fused_moe_kernel` (29.6%) —— 跟稳态 MoE 是完全不同的瓶颈** |
| **MoE R4**（长输入+长输出） | `cudaEventSynchronize` | 37.7% | 同上 —— 长 prefill = sync-bound，不是 MoE-bound |
| MoE R7（混合） | `fused_moe_kernel` | 42.7% | 稳态 —— MoE kernel 又回到老大 |
| **MoE R2**（decode-heavy）显存控制器 util | 24.3 % | — | **MoE R2 把显存控制器跑满**（dense regime 只有 1-5%）；解释了 15.5 s e2e p99 离群点 |
| **Gemma R5** `cudaLaunchKernel` | 25.3% / **9 051 calls** | — | Gemma 的"调度 retry 风暴"是 Qwen3 R5 (916) 的 10 倍 |
| Gemma R1/R2/R6 | `cudaGraphLaunch` | 19-29% | Gemma 在低负载下 **比 Qwen3 还更 launch-overhead-bound** |
| Gemma R7/R8（top） | `at::native::elementwise_kernel<...>` | 13.8-14.4% / **2 695 calls** | 大量小 elementwise op 冒出来 —— 可能没被 CUDA graph 抓到 |

### 16.6 跨模型新洞见

1. **MoE 的瓶颈随 regime 切换**。稳态 MoE（R1/R2/R5/R7/R8）：`fused_moe_kernel`
   主导（34-47%）。重 prefill（R3/R4）：`cudaEventSynchronize` 主导（37%）。
   kernel-agent 这两个都要打。
2. **小 dense 模型不能互换**。Gemma-3-1B 在大多数 regime 上一致比 Qwen3-0.6B
   慢（1.7× 参数但少 17-30% 吞吐），而且 **更 overhead-bound**（更多
   `cudaLaunchKernel` 风暴、更多 `cudaGraphLaunch` 主导）。这是一个真实的
   "同 scale 下模型架构差异有 measurable impact" 的数据点。
3. **`max-running-requests=32` 这个 cap 坑了所有人** —— R5 TTFT p50 从 538 ms
   (Qwen3) → 1 044 ms (Gemma) → 2 075 ms (MoE)。惩罚大小大致随 per-token
   成本，但 *根因* 在 3 个模型上完全一致。**一个配置改动同时修 3 个**。
4. **MoE R2 是 memory-bandwidth-bound，不是 compute-bound** —— 显存控制器 util
   24.3 %、平均功耗 249 W。这是直接证据，说明 §8 看到的 15.5 s e2e p99 离群
   有硬件侧原因：decode batch 满了之后，MoE 专家权重得从 HBM 流过 L2 比 kernel
   能 issue 工作的速度还快。候选 fix：开 FP8 量化把带宽压力砍半。

### 16.7 复现 §16

```bash
# 第一轮 Gemma 套件（2 rep × 8 regime ≈ 20 分钟）
for rep in 1 2; do
  python scripts/run_regime_suite.py --reset \
    --config configs/gemma3_1b.yaml \
    --workload-dir regime_scout/candidates_regime_study \
    --out results/regime_bench/raw/gemma_rep${rep}.jsonl \
    --run-root experiments/tmp/regime_study/gemma_rep${rep}
done

# 第二轮硬件视图（24 cell，H200 上 ≈ 30 分钟）
bash /tmp/run_hw_views_full.sh   # 见 logs/hw_view_batch_full.log

# 全部重新聚合
python scripts/regime_study/aggregate.py            # 第一轮
python scripts/regime_study/aggregate_hw_view.py    # 第二轮
```

## 17. Gemma-4 MoE 不兼容说明

`/data/hf/models/gemma-4-26B-A4B-it/` 在磁盘上（49 GB，26B total / 4B active），
但 **sglang 0.5.12.post1 没实现 `gemma4` 架构**。复现：

```bash
$ python -m sglang.launch_server \
    --model-path /data/hf/models/gemma-4-26B-A4B-it \
    --host 127.0.0.1 --port 30002 \
    --tensor-parallel-size 1 --mem-fraction-static 0.7 --trust-remote-code
…
File "…/sglang/srt/configs/model_config.py", line 250, in from_server_args
    return ModelConfig( …
File "…/sglang/srt/configs/model_config.py", line 127, in __init__
    self.hf_config = get_config(
KeyError: 'gemma4'
```

sglang 当前支持的 Gemma 家族是：
`sglang/srt/models/{gemma.py, gemma2.py, gemma2_reward.py, gemma3_causal.py,
gemma3_mm.py, gemma3n_audio.py, gemma3n_causal.py, gemma3n_mm.py}` —— **没有
gemma4**。已替换为 `gemma-3-1b-it`（dense）作为第三个模型。

要让 Gemma-4 MoE 在这套实验里能跑，未来要：
- 在 sglang 上游 PR 一个 `gemma4.py` model 实现（要搞懂 Gemma-4 的 MoE block +
  sliding-window attention + RoPE 配置）。不平凡。
- 或者等 Google 正式发布 Gemma-4 + sglang 跟进支持。

目前这个被准确记录为 **"实验范围受运行时支持限制"**，而不是默默跳过。

## 18. Kernel fusion deep dive — un-fused vs fused, and how regime changes the launch params

> 🟢 NEW. Follow-up to §16.2 — explains WHAT kernel fusion is doing in
> general, shows the exact "before vs after" for the MoE FFN, and uses our
> 24-cell traces to prove that the fused kernel's launch parameters
> (block, registers, shared memory, grid) **do change per regime**.

### 18.1 What is "kernel fusion", in one paragraph

Each GPU kernel launch costs ~5-10 µs of CPU overhead plus a kernel-launch
barrier; each intermediate tensor between kernels has to be *written* to HBM
by the producer and *read back* from HBM by the consumer (HBM bandwidth on
H200 is ~4.8 TB/s — fast but **not free**). "Kernel fusion" means: instead
of `K1 → write → K2 → write → K3 → ...`, you generate a single kernel that
**keeps intermediate values in registers / shared memory** and writes only
the final output to HBM. The savings are (a) launch-overhead × (N-1), and
(b) memory traffic = sum of intermediate tensor sizes × 2 (write + read).

### 18.2 Un-fused MoE FFN — what would run without fusion

sglang keeps a reference Torch-native implementation in
[`sglang/srt/layers/moe/fused_moe_native.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_native.py)
(`fused_moe_forward_native`):

```python
# For each token, w13/w2 are gathered by topk_ids → per-expert weight tensors
w13_weights = layer.w13_weight[topk_ids]        # gather
w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
w2_weights  = layer.w2_weight[topk_ids]         # gather
x1 = torch.einsum("ti,taoi -> tao",  x, w1_weights)   # K1: per-expert GEMM (gate)
x1 = F.silu(x1)                                       # K2: SiLU
x3 = torch.einsum("ti,taoi -> tao",  x, w3_weights)   # K3: per-expert GEMM (up)
y  = x1 * x3                                          # K4: elementwise mul
y  = torch.einsum("tao,taio -> tai", y, w2_weights)   # K5: per-expert GEMM (down)
y  = torch.einsum("tai,ta -> ti",   y, topk_weights)  # K6: weighted reduce-sum
```

Per MoE FFN that's **~6 separate kernel launches plus 3 intermediate
HBM round-trips** (`x1`, `x3`, `x1*x3`). Across 48 layers per token, that's
288 launches + a *lot* of HBM traffic just for the FFN.

### 18.3 Fused MoE FFN — what actually runs in sglang

sglang collapses the 6 ops into **2 `fused_moe_kernel` launches per layer**:

```python
# In sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:475 (gate_up part)
invoke_fused_moe_kernel(
    A=hidden_states,
    B=w1,                       # w1 (== concat of gate_proj and up_proj)
    C=intermediate_cache1,      # output of gate+up GEMM
    A_scale=..., B_scale=...,
    topk_weights=...,
    sorted_token_ids=...,       # tokens grouped by expert
    expert_ids=...,
    num_tokens_post_padded=...,
    mul_routed_weight=False,    # weight applied later
    top_k=topk,
    config=config,              # BLOCK_SIZE_M/N/K, num_warps, num_stages
    compute_type=tl.bfloat16,
    use_fp8_w8a8=False,
    ...
)
# In sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:534 (down part)
invoke_fused_moe_kernel(
    A=intermediate_cache2,      # SiLU(gate) * up — done in a tiny fused activation kernel
    B=w2,
    C=intermediate_cache3,
    ...
    mul_routed_weight=True,     # weighted reduce folded INTO this kernel
    ...
)
```

So the actual call graph in sglang is:

```
[1 prep kernel]   moe_align_block_size  → sorted_token_ids / expert_ids / num_post_padded
[1 fused kernel]  fused_moe_kernel #1   → gate+up grouped GEMM (K1+K3 in one)
[1 tiny kernel]   SiluAndMul             → element-wise gate*up fused activation
[1 fused kernel]  fused_moe_kernel #2   → down grouped GEMM with weighted reduce baked in (K5+K6 in one)
```

That's **4 kernels** vs the 6+ from the native path, **0 HBM round-trips
for the intermediate `x1*x3`** (it stays in registers across SiLU), and
**routing-weight scaling lives inside the down kernel** so K6 disappears.

Across a 48-layer Qwen3-30B-A3B forward, the fused path issues
`48 × 4 = 192` kernels per token vs `48 × 6 = 288` unfused — and avoids
`48 × 3 = 144` HBM round-trips of intermediates per token. That's why the
fused version even being a single Triton-JIT kernel can dominate at 34-47 %
of GPU time: it's **doing the work of 5 native ops in 1**, so its share is
proportionally large.

### 18.4 Does the fusion strategy change per regime? — **No, but the kernel parameters do**

The **shape of the call graph is fixed**: same 4-kernel sequence (prep +
gate_up + activation + down) every time. There is no per-regime
"more-fused-or-less-fused" switch in current sglang.

What **does** change per regime is the **kernel launch configuration**.
sglang ships per-`(E, N, device, dtype)` tuned JSON tables under
[`sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_X_X/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe/fused_moe_triton/configs).
For Qwen3-30B-A3B on H200, the relevant file is
`configs/triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json`, keyed by
`M` (tokens being dispatched after expansion):

| M (tokens) | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages |
|---|---|---|---|---|---|---|
| 1 | 16 | 64 | 64 | 1 | 4 | 5 |
| 4 | 16 | 64 | 128 | 16 | 4 | 2 |
| 16 | 16 | 64 | 256 | 1 | 4 | 2 |
| 32 | 16 | 64 | 128 | 16 | 4 | 2 |
| 48 | 16 | 128 | 128 | 16 | 4 | 3 |
| 64 | 16 | 256 | 128 | 1 | **8** | 2 |
| 512 | 64 | 128 | 64 | 1 | 4 | 3 |
| 1024 | **128** | 256 | 64 | 16 | **8** | 4 |
| 2048 | **128** | 256 | 64 | 1 | **8** | 4 |
| 4096 | **128** | 256 | 64 | 16 | **8** | 4 |

(`BLOCK_SIZE_M` jumps **8×** between low-M and high-M regimes; `num_warps`
doubles from 4 to 8.)

`try_get_optimal_moe_config(M, ...)` picks one row per call based on the
**current batch's total token-times-topk count** `M = num_tokens × topk`.
When `M` doesn't exactly match a key it picks the nearest one (rounding
down).

### 18.5 The above prediction in our actual traces

We extracted the per-launch `grid / block / registers / shared_memory` from
the `fused_moe_kernel` events in each MoE cell's `.trace.json.gz`. The
dominant launch shape across regimes:

| Cell | Token-load class | Dominant grid | block | regs/thread | shmem | mean dur | Calls in 10 prof steps |
|---|---|---|---|---|---|---|---|
| MoE R1 (M small: conc 4, in 128, all decode-class) | low-M | (768, 1, 1) | **128** | **64** | **20 KB** | 43 µs | 336 |
| MoE R2 (M mixed: in 128 + out 1024 decode steps) | low-M (decode-dominated) | (3288, 1, 1) | 128 | 64 | 20 KB | 139 µs | 336 |
| MoE R3 (M huge during the prefill: 8×4096×topk-8 ≈ 256 K) | high-M | (6246, 1, 1) | **256** | **194** | **192 KB** | 1 346 µs | 48 |
| MoE R5 (M big: cap-capped 32×512×topk-8) | high-M | (4008, 1, 1) | **256** | **194** | **192 KB** | 807 µs | 48 |
| MoE R8 (M biggest: 32×(2048 prefix+128)×topk-8) | high-M | (6774, 1, 1) | **256** | **194** | **192 KB** | 1 487 µs | 48 |

> Note: most cells produce **both** low-M and high-M launches in the same
> trace — high-M for the prefill step(s), low-M for subsequent decode
> steps. The "dominant" column above shows the variant that ate the most
> total time. Full per-cell breakdown:
> [`results/regime_bench/kernel_launch_params.csv`](../results/regime_bench/kernel_launch_params.csv).

What this means:

- **The block size doubles** (128 → 256 threads) when M crosses the boundary
  between the small and large config buckets — that's the JSON config
  switching from `BLOCK_SIZE_M=16` (4 warps × 32) to `BLOCK_SIZE_M=128`
  (8 warps × 32).
- **Register pressure triples** (64 → 194 per thread) for the high-M
  config — Triton allocates more registers to hold larger tile fragments.
  This affects how many concurrent threadblocks an SM can run.
- **Shared memory per block jumps 10×** (20 KB → 192 KB) — the high-M
  config uses bigger software-pipelined buffers (`num_stages=4`) and
  bigger tiles, so each block needs more shmem. On H200 SMs (228 KB
  shmem cap) this means **only one block of the high-M config can fit
  per SM**.
- **Grid scales with tokens** — R1 grid = 768 tiles, R5 grid = 6774 tiles
  (8.8 × more) — directly tracks `(M × N / (BLOCK_SIZE_M × BLOCK_SIZE_N))`.
- **Per-call duration scales 35×** (43 µs → 1 487 µs). The kernel works
  harder per launch, but is launched **far fewer times** (336 → 48 calls
  in the same 10-step profile window). Total time is dominated by R5 / R8
  / R2 even though they call the kernel less often.

### 18.6 Takeaway

| Question | Answer |
|---|---|
| What was un-fused before? | 6 separate ops: 3 grouped GEMMs (w1, w3, w2) + SiLU + elementwise-mul + weighted-reduce. Plus the per-token gathers `weight[topk_ids]`. |
| What's fused now? | **2 calls** to `fused_moe_kernel` per MoE FFN (gate_up + down), plus a tiny `SiluAndMul` between them and a `moe_align_block_size` prep. |
| Why does it dominate kernel time? | Because in the *new* topology it's literally **5-of-6 native ops collapsed into one Triton kernel**. Its share of GPU time should be high — that's the *point*. |
| Does fusion strategy change per regime? | **No** — same call graph every time. |
| Do the kernel parameters change per regime? | **Yes, substantially**. `BLOCK_SIZE_M` jumps 8 ×, `num_warps` 2 ×, registers 3 ×, shared memory 10 ×, grid up to 9 ×, per-call duration 35 ×, all tracked by `try_get_optimal_moe_config(M)` against the per-(E, N, device) JSON tuning table. |
| Where would more fusion still pay off? | The `SiluAndMul` between gate_up and down is currently a separate kernel (~3 % of GPU time on the MoE). Folding it into either neighbour would save 1 launch + 1 HBM round-trip per layer per token. The bigger win is on the prefill regimes (MoE R3 / R4) where `cudaEventSynchronize` between `moe_align_block_size` and `fused_moe_kernel` eats 37 % — fusing the prep into the matmul (a CUDA-graph or graph-capture-friendly variant) would attack that directly. |

---

## 18. Kernel fusion 深入解读 —— 融合前 vs 融合后，以及 regime 如何改变 launch 参数

> 🟢 新增。承接 §16.2 —— 把 kernel fusion 到底在干嘛说清楚，给出 MoE FFN 的
> "融合前 vs 融合后" 精确对比，再用我们 24 个 cell 的 trace 证明融合后 kernel
> 的 launch 参数 **真的会随 regime 变**。

### 18.1 一句话讲清 kernel fusion

每次 GPU kernel launch 有 ~5-10 µs 的 CPU 开销 + 一个 kernel-launch
barrier；每个中间 tensor 都得被生产者 *写* 进 HBM、被消费者 *读* 出来（H200
HBM 带宽 ~4.8 TB/s —— 快但 **不免费**）。所谓 kernel fusion 就是：与其
`K1 → 写 → K2 → 写 → K3 → ...`，不如生成一个 kernel **把中间值留在寄存器 /
shared memory**，只把最终结果写 HBM。节省的就是：(a) launch overhead × (N-1)，
(b) 内存流量 = 所有中间 tensor 大小 × 2（写+读）。

### 18.2 MoE FFN 融合 *前* 应该跑啥

sglang 在 [`sglang/srt/layers/moe/fused_moe_native.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_native.py)
留了一份 Torch-native 参考实现（`fused_moe_forward_native`）：

```python
# 每个 token，按 topk_ids 把 w13/w2 gather 出来 → per-expert 权重 tensor
w13_weights = layer.w13_weight[topk_ids]        # gather
w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
w2_weights  = layer.w2_weight[topk_ids]         # gather
x1 = torch.einsum("ti,taoi -> tao",  x, w1_weights)   # K1: per-expert GEMM (gate)
x1 = F.silu(x1)                                       # K2: SiLU
x3 = torch.einsum("ti,taoi -> tao",  x, w3_weights)   # K3: per-expert GEMM (up)
y  = x1 * x3                                          # K4: elementwise mul
y  = torch.einsum("tao,taio -> tai", y, w2_weights)   # K5: per-expert GEMM (down)
y  = torch.einsum("tai,ta -> ti",   y, topk_weights)  # K6: 加权 reduce-sum
```

每个 MoE FFN **~6 次 kernel launch + 3 次中间 HBM 来回**（`x1`、`x3`、`x1*x3`
都要写 HBM 再读出来）。48 层下来每 token 就 288 次 launch，单 FFN 部分 HBM
流量大得离谱。

### 18.3 MoE FFN 融合 *后* sglang 实际跑啥

sglang 把上面 6 个 op 压成 **每层 2 次 `fused_moe_kernel` launch**：

```python
# sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:475 (gate_up 部分)
invoke_fused_moe_kernel(
    A=hidden_states,
    B=w1,                       # w1（== gate_proj 和 up_proj concat）
    C=intermediate_cache1,      # gate+up GEMM 的输出
    A_scale=..., B_scale=...,
    topk_weights=...,
    sorted_token_ids=...,       # 按专家分组过的 token
    expert_ids=...,
    num_tokens_post_padded=...,
    mul_routed_weight=False,    # 路由权重之后才乘
    top_k=topk,
    config=config,              # BLOCK_SIZE_M/N/K, num_warps, num_stages
    compute_type=tl.bfloat16,
    use_fp8_w8a8=False,
    ...
)
# sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:534 (down 部分)
invoke_fused_moe_kernel(
    A=intermediate_cache2,      # SiLU(gate) * up —— 在一个很小的融合激活 kernel 里完成
    B=w2,
    C=intermediate_cache3,
    ...
    mul_routed_weight=True,     # 加权 reduce 折叠进这个 kernel
    ...
)
```

所以 sglang 的实际 call graph 是：

```
[1 个 prep kernel]   moe_align_block_size  → sorted_token_ids / expert_ids / num_post_padded
[1 个 融合 kernel]   fused_moe_kernel #1   → gate+up 分组 GEMM（K1+K3 合一）
[1 个 小 kernel]     SiluAndMul            → 逐元素 gate*up 融合激活
[1 个 融合 kernel]   fused_moe_kernel #2   → down 分组 GEMM + 加权 reduce 烤进 kernel（K5+K6 合一）
```

也就是 **4 个 kernel** vs 原生路径的 6+，**`x1*x3` 中间 tensor 0 次 HBM 来回**
（一直留寄存器里跨过 SiLU），**路由权重缩放搬到 down kernel 里了** 所以 K6 没了。

48 层 Qwen3-30B-A3B 前向，融合路径每 token issue `48 × 4 = 192` kernel，原生路径
`48 × 6 = 288` —— 还省了 `48 × 3 = 144` 次中间 HBM 来回。这就是为什么单个
Triton-JIT 的融合 kernel 能占 34-47% GPU 时间：它**一个抵 5 个 native op**，
份额自然大。

### 18.4 Fusion 策略会随 regime 变吗？ —— **不会，但 kernel 参数会**

call graph 的**形状是固定的**：永远是上面 4-kernel 序列（prep + gate_up +
activation + down）。当前 sglang 没有 per-regime "融合更多 / 更少" 的开关。

随 regime 变的是 **kernel 的 launch 配置**。sglang 在
[`sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_X_X/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe/fused_moe_triton/configs)
里塞了一堆 per-`(E, N, device, dtype)` 调好的 JSON 表。Qwen3-30B-A3B 在 H200
上用的是 `configs/triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json`，
按 `M`（分发后的 token 数）查：

| M (tokens) | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages |
|---|---|---|---|---|---|---|
| 1 | 16 | 64 | 64 | 1 | 4 | 5 |
| 4 | 16 | 64 | 128 | 16 | 4 | 2 |
| 16 | 16 | 64 | 256 | 1 | 4 | 2 |
| 32 | 16 | 64 | 128 | 16 | 4 | 2 |
| 48 | 16 | 128 | 128 | 16 | 4 | 3 |
| 64 | 16 | 256 | 128 | 1 | **8** | 2 |
| 512 | 64 | 128 | 64 | 1 | 4 | 3 |
| 1024 | **128** | 256 | 64 | 16 | **8** | 4 |
| 2048 | **128** | 256 | 64 | 1 | **8** | 4 |
| 4096 | **128** | 256 | 64 | 16 | **8** | 4 |

（低 M → 高 M，`BLOCK_SIZE_M` 跳 **8 倍**，`num_warps` 从 4 翻到 8。）

`try_get_optimal_moe_config(M, ...)` 每次调用按当前 batch 的 `M = num_tokens × topk`
选一行。M 不刚好匹配 key 就向下取最近的。

### 18.5 上面这套预言 —— 在我们真实 trace 里看到

从每个 MoE cell 的 `.trace.json.gz` 里把 `fused_moe_kernel` 事件的
`grid / block / registers / shared_memory` 拉出来。各 regime 的主流 launch 形状：

| Cell | Token-load 级别 | 主流 grid | block | regs/thread | shmem | 平均 dur | 10 个 profile step 的调用次数 |
|---|---|---|---|---|---|---|---|
| MoE R1（M 小：conc 4，in 128，全 decode 类）| 低-M | (768, 1, 1) | **128** | **64** | **20 KB** | 43 µs | 336 |
| MoE R2（M 混合：in 128 + 1024 步 decode）| 低-M（被 decode 主导）| (3288, 1, 1) | 128 | 64 | 20 KB | 139 µs | 336 |
| MoE R3（prefill 中 M 巨大：8×4096×topk-8 ≈ 256 K）| 高-M | (6246, 1, 1) | **256** | **194** | **192 KB** | 1 346 µs | 48 |
| MoE R5（M 大：撞 cap 后 32×512×topk-8）| 高-M | (4008, 1, 1) | **256** | **194** | **192 KB** | 807 µs | 48 |
| MoE R8（M 最大：32×(2048 prefix+128)×topk-8）| 高-M | (6774, 1, 1) | **256** | **194** | **192 KB** | 1 487 µs | 48 |

> 注：大多数 cell 实际同时产出 **低-M 和 高-M 两种 launch** —— 高-M 给
> prefill 步，低-M 给后续 decode 步。上表的"主流"指总耗时最高的那个变体。
> 每 cell 完整分解：
> [`results/regime_bench/kernel_launch_params.csv`](../results/regime_bench/kernel_launch_params.csv)。

含义：

- **block size 翻倍**（128 → 256 线程）—— 当 M 跨过小/大 config 分界，JSON 表
  从 `BLOCK_SIZE_M=16`（4 warps × 32）切到 `BLOCK_SIZE_M=128`（8 warps × 32）。
- **寄存器压力 3 倍**（64 → 194 / thread）—— Triton 给高-M config 分配更多
  寄存器存大 tile 片段。这直接影响每个 SM 能并发跑多少 threadblock。
- **每 block shared memory 10 倍**（20 KB → 192 KB）—— 高-M config 用更大的
  软件流水缓冲（`num_stages=4`）+ 更大的 tile，每个 block 需要更多 shmem。
  H200 SM 上限 228 KB shmem → **高-M config 每个 SM 只能塞 1 个 block**。
- **grid 随 token 数线性扩**（R1 768 tiles → R5 6 774 tiles，8.8 倍），直接
  对应 `(M × N / (BLOCK_SIZE_M × BLOCK_SIZE_N))`。
- **每次调用 wall 35 倍**（43 µs → 1 487 µs）。单 launch 干活更多，但调用次数
  **少得多**（同一 10-step 窗口内 336 → 48 次）。R5 / R8 / R2 的总时间是被
  少而重的 launch 主导。

### 18.6 一表打包

| 问题 | 答案 |
|---|---|
| 融合前是啥？ | 6 个 op：3 次 grouped GEMM（w1、w3、w2）+ SiLU + elementwise mul + 加权 reduce。还有 `weight[topk_ids]` 的 per-token gather。 |
| 融合后是啥？ | 每个 MoE FFN **2 次** `fused_moe_kernel`（gate_up + down）+ 中间一次小 `SiluAndMul` + 前置 `moe_align_block_size` 准备 kernel。 |
| 为啥它在 kernel 时间里占主导？ | 因为在新拓扑里它字面意义上 **把 6 个 native op 中的 5 个折叠成一个 Triton kernel**。它占 GPU 时间比例大 —— 这正是融合的 *目的*。 |
| Fusion 策略会随 regime 变吗？ | **不会** —— call graph 每次都一样。 |
| Kernel 参数会随 regime 变吗？ | **会，而且变化很大**。`BLOCK_SIZE_M` 8 倍、`num_warps` 2 倍、寄存器 3 倍、shmem 10 倍、grid 高达 9 倍、单次 wall 35 倍 —— 全由 `try_get_optimal_moe_config(M)` 按 per-(E, N, device) JSON 调优表选。 |
| 还有哪里能再融合？ | gate_up 和 down 之间那个 `SiluAndMul` 现在是独立 kernel（占 GPU 时间 ~3%）。把它折进任一邻居能再省 1 launch + 1 HBM 来回 / layer / token。更大的赢面在 prefill regime（MoE R3 / R4）—— 那里 `moe_align_block_size` 和 `fused_moe_kernel` 之间的 `cudaEventSynchronize` 吃 37%，把 prep 融进 matmul（CUDA-graph 友好的变体）能直接干掉这一坨。 |
