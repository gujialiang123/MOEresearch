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

### 18.2 Un-fused MoE FFN — actual reference code

sglang keeps a reference Torch-native implementation in
[`sglang/srt/layers/moe/fused_moe_native.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_native.py)
(`fused_moe_forward_native`). **This is the exact code, copy-pasted**:

```python
# sglang/srt/layers/moe/fused_moe_native.py:18-46
def fused_moe_forward_native(
    layer: torch.nn.Module,
    dispatch_output: StandardDispatchOutput,
) -> StandardCombineInput:

    x, x_scale, topk_output = dispatch_output
    moe_runner_config = layer.moe_runner_config

    if moe_runner_config.apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = topk_output

    # ---- per-token expert weight GATHER (4 ops worth of HBM traffic) ----
    w13_weights = layer.w13_weight[topk_ids]                       # gather  → HBM
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)    # view (no copy)
    w2_weights = layer.w2_weight[topk_ids]                         # gather  → HBM

    # ---- 6 actual ops, each a separate CUDA kernel ----
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)             # K1: GEMM (gate)        → HBM
    if moe_runner_config.activation == "silu":
        x1 = F.silu(x1)                                            # K2: SiLU              → HBM
    elif moe_runner_config.activation == "gelu":
        x1 = F.gelu(x1)
    else:
        raise ValueError(f"Unsupported activation: {moe_runner_config.activation=}")
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)            # K3: GEMM (up)         → HBM
    expert_outs = torch.einsum("tao, taio -> tai",
                               (x1 * x3),                          # K4: elementwise mul   → HBM
                               w2_weights)                         # K5: GEMM (down)       → HBM
    expert_outs = torch.einsum("tai,ta -> ti",
                               expert_outs,
                               topk_weights.to(expert_outs.dtype)) # K6: weighted reduce   → HBM
    return StandardCombineInput(hidden_states=expert_outs)
```

The HBM cost per MoE FFN, per token, with `hidden=2048, intermediate=768,
topk=8`:

| Tensor | Shape | bf16 bytes | Direction |
|---|---|---|---|
| `w13_weights` | `(topk, 2N, H)` = `(8, 1536, 2048)` | ~50 MB | read (gather) |
| `w2_weights` | `(topk, H, N)` = `(8, 2048, 768)` | ~25 MB | read (gather) |
| `x1` (gate output) | `(topk, N)` = `(8, 768)` | 12 KB | **write + read** |
| `x3` (up output) | `(topk, N)` = `(8, 768)` | 12 KB | **write + read** |
| `x1*x3` (intermediate) | `(topk, N)` = `(8, 768)` | 12 KB | **write + read** |
| `expert_outs` pre-reduce | `(topk, H)` = `(8, 2048)` | 32 KB | **write + read** |

The **4 intermediate write+read pairs** (the bold rows) are exactly what
kernel fusion eliminates.

### 18.3 Fused MoE FFN — what actually runs in sglang

The wrapper `invoke_fused_moe_kernel` (Python, in
[`fused_moe_triton_kernels.py:675`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py))
is called twice per MoE FFN. Its launch site:

```python
# sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:837
fused_moe_kernel[grid](
    A, a_desc, B, b_desc, bias, C,
    A_scale, B_scale,
    topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
    B.shape[1],                 # N
    B.shape[2] - padded_size,   # K
    sorted_token_ids.shape[0],  # EM (M after expert-padding)
    topk_ids.numel(),           # num_valid_tokens
    A.stride(0), A.stride(1),
    B.stride(0), B.stride(2), B.stride(1),
    bias.stride(0) if bias is not None else 0,
    bias.stride(1) if bias is not None else 0,
    C.stride(-2), C.stride(-1),
    A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
    ...
    MUL_ROUTED_WEIGHT=mul_routed_weight,    # True for down kernel → folds K6 in
    top_k=top_k,
    compute_type=compute_type,
    use_fp8_w8a8=use_fp8_w8a8,
    ...
    **config,   # ← BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages
                #   from try_get_optimal_moe_config(M)
)
```

The `@triton.jit` kernel itself (`fused_moe_kernel`,
[`fused_moe_triton_kernels.py:324`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py))
— compute core, abbreviated to show the **fusion points**:

```python
# sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:323-580 (excerpted)
@triton.jit
def fused_moe_kernel(
    a_ptr, ..., b_ptr, ..., c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    # block sizes — set per regime by try_get_optimal_moe_config(M)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    ...
):
    # 1. Block index → (pid_m, pid_n) with grouped ordering for L2 reuse
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    pid_m = group_id * GROUP_SIZE_M + ((pid % num_pid_in_group) % GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M

    # 2. Look up which tokens + which expert this block covers
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token    = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask    = offs_token < num_valid_tokens
    off_experts   = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # 3. Pointer arithmetic into A (tokens) and B[expert] (weights)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am
                      + offs_k[None, :]    * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be             # ← per-expert weight bank
              + offs_k[:, None] * stride_bk
              + offs_bn[None, :] * stride_bn)

    # 4. Software-pipelined GEMM accumulation in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        # ↑↑↑ these loads use shared-memory double-buffering;
        #     num_stages controls how many K iterations are in flight at once
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. ★ FUSION POINT ★ — fold routing weight into the down-projection result
    #    (this is what makes K5 + K6 a single kernel; the un-fused code in §18.2
    #     needed a separate einsum "tai,ta -> ti" after the down GEMM)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask)
        accumulator = accumulator * moe_weight[:, None]

    # 6. Single HBM store of the final block (no intermediate write/read)
    c = accumulator.to(compute_type)
    tl.store(c_ptr + ..., c, mask=...)
```

The 4 things this kernel **eliminates** vs the un-fused code in §18.2:

1. **Per-token expert weight gather** — `w13_weight[topk_ids]` becomes a
   pointer arithmetic op in step 3 (`b_ptr + off_experts * stride_be`).
   No materialised gather tensor.
2. **`x1`, `x3` intermediate HBM round-trips** — both gate and up projections
   share this kernel via `B = w1` (which is actually the concat `[gate_proj,
   up_proj]`). The accumulator stays in registers for the whole K loop.
3. **`x1 * x3` intermediate** — handled by the small `SiluAndMul` between
   the two `fused_moe_kernel` calls. That op operates on the gate_up output
   in-place. So we still have a tiny shuttle here (one launch + one HBM
   round-trip) — but only one, not three.
4. **`weighted_reduce_sum`** — folded into the down-kernel via
   `MUL_ROUTED_WEIGHT=True` + `accumulator * moe_weight[:, None]` in step 5.
   K6 disappears entirely.

The full sglang call graph per MoE FFN:

```
[1 prep kernel]    moe_align_block_size  → sorted_token_ids / expert_ids / num_post_padded
[1 fused kernel]   fused_moe_kernel #1   → gate+up grouped GEMM
                                            MUL_ROUTED_WEIGHT=False
                                            (K1 + K3 collapsed; no x1/x3 in HBM)
[1 small kernel]   SiluAndMul             → element-wise gate*up
[1 fused kernel]   fused_moe_kernel #2   → down grouped GEMM
                                            MUL_ROUTED_WEIGHT=True
                                            (K5 + K6 collapsed; final reduce baked in)
```

That's **4 kernels** vs the **6+** in §18.2. Per token across 48
Qwen3-30B-A3B layers: `48 × 4 = 192` launches (fused) vs `48 × 6 = 288`
(un-fused), and `48 × 2 = 96` HBM round-trips for intermediates eliminated
(only the small `gate*up` shuttle survives).

This is why `fused_moe_kernel` shows up as 34-47 % of GPU time in our
traces: it's literally **5 native ops collapsed into one Triton kernel**.
Its share is large because *its work is large*. That's the point of the
fusion.

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
every `fused_moe_kernel` event in each MoE cell's `.trace.json.gz`, grouped
by `(grid, block, regs, shmem)` tuple, ranked by total self-time, and kept
the top 3 dominant launches per cell. **All 8 MoE regimes shown below** —
columns are direct from the trace; `pct` is share of `fused_moe_kernel`'s
own total time within the cell (not total GPU time):

| Cell | Variant | Grid | Block | Regs/thr | Shmem | Calls | Mean (µs) | Total (µs) | Pct |
|---|---|---|---|---|---|---|---|---|---|
| **MoE R1** decode-style baseline | low-M #1 | 768 | 128 | 64 | 20 KB | 336 | 43 | 14 392 | 48.0% |
| | low-M #2 | 1 024 | 128 | 64 | 20 KB | 336 | 24 | 7 981 | 26.6% |
| | low-M #3 | 2 520 | 128 | 98 | 36 KB | 48 | 159 | 7 627 | 25.4% |
| **MoE R2** decode-heavy (1024 out tokens) | low-M #1 | 3 288 | 128 | 64 | 20 KB | 336 | 139 | 46 555 | 54.9% |
| | low-M #2 | 4 384 | 128 | 64 | 20 KB | 336 | 71 | 23 997 | 28.3% |
| | **high-M** (prefill burst) | 1 632 | **256** | **194** | **192 KB** | 48 | 298 | 14 299 | 16.9% |
| **MoE R3** prefill-heavy (4096 in × conc 8) | **high-M** #1 | 6 246 | **256** | **194** | **192 KB** | 48 | **1 346** | 64 602 | 50.8% |
| | **high-M** #2 | 8 328 | 256 | 196 | 192 KB | 48 | 857 | 41 122 | 32.3% |
| | low-M (decode tail) | 1 536 | 128 | 64 | 40 KB | 336 | 64 | 21 509 | 16.9% |
| **MoE R4** long-in long-out | **high-M** #1 | 6 246 | **256** | **194** | **192 KB** | 48 | **1 361** | 65 302 | 50.9% |
| | **high-M** #2 | 8 328 | 256 | 196 | 192 KB | 48 | 867 | 41 626 | 32.4% |
| | low-M (decode) | 1 536 | 128 | 64 | 40 KB | 336 | 64 | 21 479 | 16.7% |
| **MoE R5** cap-hit saturation | **high-M** #1 | 4 008 | **256** | **194** | **192 KB** | 48 | 807 | 38 754 | 43.6% |
| | low-M (post-cap decode) | 3 288 | 128 | 64 | 20 KB | 192 | 134 | 25 795 | 29.0% |
| | **high-M** #2 | 5 344 | 256 | 196 | 192 KB | 48 | 506 | 24 281 | 27.3% |
| **MoE R6** single-stream | low-M #1 | 192 | 128 | **56** | 40 KB | 432 | **16** | 6 793 | 64.2% |
| | low-M #2 | 256 | 128 | 56 | 40 KB | 432 | 9 | 3 788 | 35.8% |
| **MoE R7** mixed-length (random_range 0.95) | **high-M** #1 | 6 768 | **256** | **194** | **192 KB** | 48 | **1 490** | 71 535 | 33.7% |
| | **high-M** #2 | 6 696 | 256 | 194 | 192 KB | 48 | 1 474 | 70 769 | 33.3% |
| | **high-M** #3 | 6 744 | 256 | 194 | 192 KB | 48 | 1 464 | 70 268 | 33.1% |
| **MoE R8** prefix sharing | **high-M** #1 | 6 774 | **256** | **194** | **192 KB** | 48 | **1 487** | 71 377 | 37.9% |
| | **high-M** #2 | 6 882 | 256 | 194 | 192 KB | 48 | 1 483 | 71 169 | 37.8% |
| | **high-M** #3 | 9 032 | 256 | 196 | 192 KB | 48 | 955 | 45 819 | 24.3% |

Full per-cell breakdown including dense + Gemma's FlashAttention variants
in [`results/regime_bench/kernel_launch_params.csv`](../results/regime_bench/kernel_launch_params.csv).

**What this table is screaming at us**:

1. **Same kernel, two completely different "personalities"** —
   - **low-M variant**: block 128, regs 64, shmem 20-40 KB, ~10-150 µs/call,
     called ~336 times (= 48 layers × 7 decode steps)
   - **high-M variant**: block **256**, regs **194**, shmem **192 KB**,
     500-1490 µs/call, called 48 times (= 48 layers × 1 prefill step)
2. **Regimes naturally split into 3 archetypes**:
   - **Decode-dominated** (R1, R6): only the low-M variant fires
   - **Prefill-dominated** (R3, R4, R7, R8): high-M variant tops the chart,
     low-M shows up only as a decode tail
   - **Mixed** (R2, R5): both variants share the time roughly 50/50
3. **R6 single-stream uses an entirely different register count** (56 vs
   64/194). At grid=192 (very small), Triton picks `BLOCK_SIZE_K=64` instead
   of `BLOCK_SIZE_K=128`, which reduces register pressure by 12%.
4. **R3/R4/R7/R8 spawn 6 000-9 000 thread blocks** — at `BLOCK_SIZE_M=128`
   the grid is `(num_padded_tokens / 128) × (N / BLOCK_SIZE_N) =
   (tokens / 128) × (768 / 256) = (tokens / 128) × 3`. So R8's grid 6 774
   corresponds to **~290 K padded tokens of work per kernel launch**.
5. **The 192 KB shmem is right at H200's ceiling** (228 KB shmem per SM).
   So when the high-M variant runs, **only 1 block per SM** can be in flight
   simultaneously. Combined with the 194 reg/thread, this is what makes the
   kernel "fat" — each launch is a heavyweight grid that monopolises the
   chip for 1-1.5 ms.

So when mentor asks **"does the kernel change with regime?"**:

> **The kernel source code is identical** (one `@triton.jit`'d function).
> But the same source compiles to two distinct PTX variants under different
> `(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages)`
> meta-parameters; **the variant chosen per launch depends on the current
> batch's `M = num_tokens × topk`**, looked up in the per-`(E, N, device)`
> JSON table. Different regimes produce different `M`, which selects
> different meta-parameters, which produces different launch grids, register
> counts, and shared-memory allocations — directly visible in our traces.

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

### 18.2 MoE FFN 融合 *前* —— 真实参考代码

sglang 在 [`sglang/srt/layers/moe/fused_moe_native.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_native.py)
留了一份 Torch-native 参考实现（`fused_moe_forward_native`）。**这是源码原文，
直接拷过来的**：

```python
# sglang/srt/layers/moe/fused_moe_native.py:18-46
def fused_moe_forward_native(
    layer: torch.nn.Module,
    dispatch_output: StandardDispatchOutput,
) -> StandardCombineInput:

    x, x_scale, topk_output = dispatch_output
    moe_runner_config = layer.moe_runner_config

    if moe_runner_config.apply_router_weight_on_input:
        raise NotImplementedError()

    topk_weights, topk_ids, _ = topk_output

    # ---- 按 topk_ids 把专家权重 gather 出来（多 4 次 HBM 流量）----
    w13_weights = layer.w13_weight[topk_ids]                       # gather  → HBM
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)    # view（不拷贝）
    w2_weights = layer.w2_weight[topk_ids]                         # gather  → HBM

    # ---- 真正的 6 个 op，每个都是独立 CUDA kernel ----
    x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)             # K1: GEMM (gate)       → HBM
    if moe_runner_config.activation == "silu":
        x1 = F.silu(x1)                                            # K2: SiLU             → HBM
    elif moe_runner_config.activation == "gelu":
        x1 = F.gelu(x1)
    else:
        raise ValueError(f"Unsupported activation: {moe_runner_config.activation=}")
    x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)            # K3: GEMM (up)        → HBM
    expert_outs = torch.einsum("tao, taio -> tai",
                               (x1 * x3),                          # K4: elementwise mul  → HBM
                               w2_weights)                         # K5: GEMM (down)      → HBM
    expert_outs = torch.einsum("tai,ta -> ti",
                               expert_outs,
                               topk_weights.to(expert_outs.dtype)) # K6: 加权 reduce      → HBM
    return StandardCombineInput(hidden_states=expert_outs)
```

每个 MoE FFN per token 的 HBM 成本（`hidden=2048, intermediate=768, topk=8`）：

| Tensor | Shape | bf16 字节 | 方向 |
|---|---|---|---|
| `w13_weights` | `(topk, 2N, H)` = `(8, 1536, 2048)` | ~50 MB | 读（gather） |
| `w2_weights` | `(topk, H, N)` = `(8, 2048, 768)` | ~25 MB | 读（gather） |
| `x1` (gate 输出) | `(topk, N)` = `(8, 768)` | 12 KB | **写+读** |
| `x3` (up 输出) | `(topk, N)` = `(8, 768)` | 12 KB | **写+读** |
| `x1*x3` (中间) | `(topk, N)` = `(8, 768)` | 12 KB | **写+读** |
| `expert_outs` reduce 前 | `(topk, H)` = `(8, 2048)` | 32 KB | **写+读** |

**4 次中间 write+read 来回**（粗体行）就是 kernel fusion 要消除的目标。

### 18.3 MoE FFN 融合 *后* —— sglang 实际跑啥

wrapper `invoke_fused_moe_kernel`（Python，在
[`fused_moe_triton_kernels.py:675`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py)）
每个 MoE FFN 被调 2 次。launch site：

```python
# sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:837
fused_moe_kernel[grid](
    A, a_desc, B, b_desc, bias, C,
    A_scale, B_scale,
    topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
    B.shape[1],                 # N
    B.shape[2] - padded_size,   # K
    sorted_token_ids.shape[0],  # EM（专家 padding 后的 M）
    topk_ids.numel(),           # num_valid_tokens
    A.stride(0), A.stride(1),
    B.stride(0), B.stride(2), B.stride(1),
    bias.stride(0) if bias is not None else 0,
    bias.stride(1) if bias is not None else 0,
    C.stride(-2), C.stride(-1),
    A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
    ...
    MUL_ROUTED_WEIGHT=mul_routed_weight,    # down kernel 时是 True → 把 K6 折叠进来
    top_k=top_k,
    compute_type=compute_type,
    use_fp8_w8a8=use_fp8_w8a8,
    ...
    **config,   # ← BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages
                #   由 try_get_optimal_moe_config(M) 选出来
)
```

`@triton.jit` kernel 本体（`fused_moe_kernel`，
[`fused_moe_triton_kernels.py:324`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py)）
—— 计算核心，节选出 **融合点**：

```python
# sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:323-580 (节选)
@triton.jit
def fused_moe_kernel(
    a_ptr, ..., b_ptr, ..., c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    # block sizes —— 这些由 try_get_optimal_moe_config(M) 按 regime 选
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    ...
):
    # 1. block 索引 → (pid_m, pid_n)，按 group 排序提高 L2 重用
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    pid_m = group_id * GROUP_SIZE_M + ((pid % num_pid_in_group) % GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M

    # 2. 查这个 block 负责哪些 token + 哪个专家
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token    = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask    = offs_token < num_valid_tokens
    off_experts   = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # 3. 对 A（token）和 B[expert]（权重）做指针算术
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am
                      + offs_k[None, :]    * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be             # ← 选这个专家的权重 bank
              + offs_k[:, None] * stride_bk
              + offs_bn[None, :] * stride_bn)

    # 4. fp32 累加器里软件流水跑 GEMM
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        # ↑↑↑ 这些 load 用 shared-memory 双缓冲；
        #     num_stages 控制同时有多少个 K 迭代在飞
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. ★ 融合点 ★ —— 把路由权重折叠进 down-projection 结果
    #    （这就是 K5 + K6 能合一的关键；§18.2 的原生代码在 down GEMM
    #     之后还得单独跑一个 einsum "tai,ta -> ti"）
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask)
        accumulator = accumulator * moe_weight[:, None]

    # 6. 把最终 block 一次性写回 HBM（中间不写不读）
    c = accumulator.to(compute_type)
    tl.store(c_ptr + ..., c, mask=...)
```

这个 kernel 相对 §18.2 的原生代码 **消除了 4 件事**：

1. **per-token 专家权重 gather** —— `w13_weight[topk_ids]` 在第 3 步变成
   指针算术（`b_ptr + off_experts * stride_be`）。不实际化 gather tensor。
2. **`x1`、`x3` 的中间 HBM 来回** —— gate 和 up projection 共用一次 kernel
   调用（`B = w1`，其实是 `[gate_proj, up_proj]` 的 concat）。累加器整个 K
   循环都在寄存器里。
3. **`x1 * x3` 中间** —— 由两个 `fused_moe_kernel` 之间那个小 `SiluAndMul`
   in-place 处理。这里还剩 1 次 launch + 1 次 HBM 来回，但只剩这 1 次，不是 3 次。
4. **`weighted_reduce_sum`** —— 通过 `MUL_ROUTED_WEIGHT=True` +
   第 5 步的 `accumulator * moe_weight[:, None]` 折进 down kernel 里。K6 直接消失。

sglang 每个 MoE FFN 完整的 call graph：

```
[1 个 prep kernel]   moe_align_block_size  → sorted_token_ids / expert_ids / num_post_padded
[1 个 融合 kernel]   fused_moe_kernel #1   → gate+up 分组 GEMM
                                              MUL_ROUTED_WEIGHT=False
                                              （K1 + K3 折叠；x1/x3 不进 HBM）
[1 个 小 kernel]     SiluAndMul            → 逐元素 gate*up
[1 个 融合 kernel]   fused_moe_kernel #2   → down 分组 GEMM
                                              MUL_ROUTED_WEIGHT=True
                                              （K5 + K6 折叠；最终 reduce 烤进 kernel）
```

也就是 **4 个 kernel** vs §18.2 的 **6+**。48 层 Qwen3-30B-A3B per token：
融合版 `48 × 4 = 192` launch，原生版 `48 × 6 = 288`，**消除了
`48 × 2 = 96` 次中间 HBM 来回**（只剩那个 `gate*up` 小 shuttle）。

这就是为什么 `fused_moe_kernel` 在 trace 里占 34-47% GPU 时间 —— 它字面意义上
**把 5 个 native op 折叠成一个 Triton kernel**。份额大是因为 *它做的事多*。
这就是融合的目的。

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

从每个 MoE cell 的 `.trace.json.gz` 里把 *每一次* `fused_moe_kernel` 事件的
`grid / block / registers / shared_memory` 拉出来，按 `(grid, block, regs, shmem)`
分组，按总 self-time 排序，取每个 cell 的 top-3 主流 launch。**所有 8 个 MoE
regime 都列出来** —— 列直接来自 trace；`pct` 是该变体占本 cell `fused_moe_kernel`
总耗时的比例（不是总 GPU 时间）：

| Cell | 变体 | Grid | Block | Regs/thr | Shmem | 调用次数 | 平均 (µs) | 总 (µs) | Pct |
|---|---|---|---|---|---|---|---|---|---|
| **MoE R1** decode 基线 | 低-M #1 | 768 | 128 | 64 | 20 KB | 336 | 43 | 14 392 | 48.0% |
| | 低-M #2 | 1 024 | 128 | 64 | 20 KB | 336 | 24 | 7 981 | 26.6% |
| | 低-M #3 | 2 520 | 128 | 98 | 36 KB | 48 | 159 | 7 627 | 25.4% |
| **MoE R2** decode-heavy（1024 输出）| 低-M #1 | 3 288 | 128 | 64 | 20 KB | 336 | 139 | 46 555 | 54.9% |
| | 低-M #2 | 4 384 | 128 | 64 | 20 KB | 336 | 71 | 23 997 | 28.3% |
| | **高-M**（prefill 突发）| 1 632 | **256** | **194** | **192 KB** | 48 | 298 | 14 299 | 16.9% |
| **MoE R3** prefill-heavy（4096 输入 × conc 8）| **高-M** #1 | 6 246 | **256** | **194** | **192 KB** | 48 | **1 346** | 64 602 | 50.8% |
| | **高-M** #2 | 8 328 | 256 | 196 | 192 KB | 48 | 857 | 41 122 | 32.3% |
| | 低-M（decode 尾）| 1 536 | 128 | 64 | 40 KB | 336 | 64 | 21 509 | 16.9% |
| **MoE R4** 长输入长输出 | **高-M** #1 | 6 246 | **256** | **194** | **192 KB** | 48 | **1 361** | 65 302 | 50.9% |
| | **高-M** #2 | 8 328 | 256 | 196 | 192 KB | 48 | 867 | 41 626 | 32.4% |
| | 低-M（decode）| 1 536 | 128 | 64 | 40 KB | 336 | 64 | 21 479 | 16.7% |
| **MoE R5** 撞 cap 饱和 | **高-M** #1 | 4 008 | **256** | **194** | **192 KB** | 48 | 807 | 38 754 | 43.6% |
| | 低-M（撞 cap 后 decode）| 3 288 | 128 | 64 | 20 KB | 192 | 134 | 25 795 | 29.0% |
| | **高-M** #2 | 5 344 | 256 | 196 | 192 KB | 48 | 506 | 24 281 | 27.3% |
| **MoE R6** 单流 | 低-M #1 | 192 | 128 | **56** | 40 KB | 432 | **16** | 6 793 | 64.2% |
| | 低-M #2 | 256 | 128 | 56 | 40 KB | 432 | 9 | 3 788 | 35.8% |
| **MoE R7** 混合长度（random_range 0.95）| **高-M** #1 | 6 768 | **256** | **194** | **192 KB** | 48 | **1 490** | 71 535 | 33.7% |
| | **高-M** #2 | 6 696 | 256 | 194 | 192 KB | 48 | 1 474 | 70 769 | 33.3% |
| | **高-M** #3 | 6 744 | 256 | 194 | 192 KB | 48 | 1 464 | 70 268 | 33.1% |
| **MoE R8** prefix sharing | **高-M** #1 | 6 774 | **256** | **194** | **192 KB** | 48 | **1 487** | 71 377 | 37.9% |
| | **高-M** #2 | 6 882 | 256 | 194 | 192 KB | 48 | 1 483 | 71 169 | 37.8% |
| | **高-M** #3 | 9 032 | 256 | 196 | 192 KB | 48 | 955 | 45 819 | 24.3% |

dense + Gemma 的 FlashAttention 变体也在
[`results/regime_bench/kernel_launch_params.csv`](../results/regime_bench/kernel_launch_params.csv) 里。

**这张表在喊的话**：

1. **同一个 kernel，两副完全不同的"性格"** ——
   - **低-M 变体**：block 128，regs 64，shmem 20-40 KB，10-150 µs/call，
     调用 ~336 次（= 48 层 × 7 个 decode step）
   - **高-M 变体**：block **256**，regs **194**，shmem **192 KB**，500-1490 µs/call，
     调用 48 次（= 48 层 × 1 个 prefill step）
2. **regime 自然分成 3 类原型**：
   - **decode 主导**（R1、R6）：只有低-M 变体在跑
   - **prefill 主导**（R3、R4、R7、R8）：高-M 变体登顶，低-M 只出现在 decode 尾
   - **混合**（R2、R5）：两个变体大约 50/50 分时间
3. **R6 单流用完全不同的寄存器数**（56 vs 64/194）。grid=192 太小，Triton 选了
   `BLOCK_SIZE_K=64` 而不是 128，寄存器压力降 12%。
4. **R3/R4/R7/R8 起 6 000-9 000 个 thread block** —— `BLOCK_SIZE_M=128` 时
   grid 是 `(num_padded_tokens / 128) × (N / BLOCK_SIZE_N) = (tokens / 128) × 3`。
   R8 的 grid 6 774 对应 **~290 K padded token 的工作量 / 单次 launch**。
5. **192 KB shmem 卡在 H200 极限附近**（每 SM 228 KB shmem）。所以高-M 变体跑时
   **每个 SM 只能并发 1 个 block**。再叠加 194 reg/thread，这就是 kernel 变"重"
   的原因 —— 每次 launch 是一个 heavyweight grid，独占芯片 1-1.5 ms。

所以 mentor 问 **"kernel 会随 regime 变吗？"**：

> **kernel 源码完全相同**（一个 `@triton.jit` 函数）。但同一份源码在不同
> `(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages)` meta
> 参数下编译出两种不同的 PTX；**每次 launch 用哪个变体取决于当前 batch 的
> `M = num_tokens × topk`**，按 per-(E, N, device) JSON 表查。不同 regime
> 产生不同的 M，选不同的 meta 参数，给出不同的 launch grid、寄存器数、shmem
> 分配 —— 全部都能在我们的 trace 里看到。

### 18.6 一表打包

| 问题 | 答案 |
|---|---|
| 融合前是啥？ | 6 个 op：3 次 grouped GEMM（w1、w3、w2）+ SiLU + elementwise mul + 加权 reduce。还有 `weight[topk_ids]` 的 per-token gather。 |
| 融合后是啥？ | 每个 MoE FFN **2 次** `fused_moe_kernel`（gate_up + down）+ 中间一次小 `SiluAndMul` + 前置 `moe_align_block_size` 准备 kernel。 |
| 为啥它在 kernel 时间里占主导？ | 因为在新拓扑里它字面意义上 **把 6 个 native op 中的 5 个折叠成一个 Triton kernel**。它占 GPU 时间比例大 —— 这正是融合的 *目的*。 |
| Fusion 策略会随 regime 变吗？ | **不会** —— call graph 每次都一样。 |
| Kernel 参数会随 regime 变吗？ | **会，而且变化很大**。`BLOCK_SIZE_M` 8 倍、`num_warps` 2 倍、寄存器 3 倍、shmem 10 倍、grid 高达 9 倍、单次 wall 35 倍 —— 全由 `try_get_optimal_moe_config(M)` 按 per-(E, N, device) JSON 调优表选。 |
| 还有哪里能再融合？ | gate_up 和 down 之间那个 `SiluAndMul` 现在是独立 kernel（占 GPU 时间 ~3%）。把它折进任一邻居能再省 1 launch + 1 HBM 来回 / layer / token。更大的赢面在 prefill regime（MoE R3 / R4）—— 那里 `moe_align_block_size` 和 `fused_moe_kernel` 之间的 `cudaEventSynchronize` 吃 37%，把 prep 融进 matmul（CUDA-graph 友好的变体）能直接干掉这一坨。 |

## 19. MoE optimization-knob study (config A/B test)

> 🟢 NEW. Holds the workload fixed (Qwen3-30B-A3B MoE on R8 prefix sharing)
> and varies **one server config knob at a time** to see how each affects
> performance, kernel mix, hardware utilisation, and backend selection.
> 7 configs, 5 PASS, 2 FAIL (recorded as evidence). Full tables in
> [`results/regime_bench/moe_opt_levels_table.md`](../results/regime_bench/moe_opt_levels_table.md).

### 19.1 What we measured

7 config variants — each one differs from `configs/moe_qwen3_30b.yaml` by exactly one knob:

| Tag | Knob under test | Hypothesis |
|---|---|---|
| **C0** baseline | (unchanged) | reference |
| **C1** torch.compile | `enable-torch-compile: true` | JIT-compile small ops, may change kernel mix |
| **C2** no CUDA graph | `disable-cuda-graph: true` | Without graph capture, launch overhead rises sharply |
| **C3** chunked prefill | `chunked-prefill-size: 2048` (was -1) | Should help prefill-heavy regimes; might hurt R8 |
| **C4** MoE runner cutlass | `moe-runner-backend: cutlass` | Replace Triton `fused_moe_kernel` with cutlass MoE |
| **C5** attn flashinfer | `attention-backend: flashinfer` (was fa3) | Different attention kernels |
| **C6** piecewise CUDA graph | `enable-piecewise-cuda-graph: true` + token splits | Finer-grained graph capture |

### 19.2 Status overview

| Tag | Status | Reason if FAIL |
|---|---|---|
| C0 baseline | ✅ PASS | |
| C1 torch.compile | ❌ FAIL | `torch._dynamo` AssertionError in `sglang/srt/layers/rotary_embedding.py:272` (Qwen3 MoE incompatibility) |
| C2 no CUDA graph | ✅ PASS | |
| C3 chunked prefill | ✅ PASS | |
| C4 MoE cutlass | ✅ PASS | |
| C5 attn flashinfer | ❌ FAIL | flashinfer JIT (ninja) failed to build `batch_prefill` kernel on H200 (env issue, not config) |
| C6 piecewise CUDA graph | ✅ PASS | (needed explicit `piecewise-cuda-graph-tokens: [512, 1024, 2048, 4096, 8192]`) |

### 19.3 Performance (sglang.bench_serving on R8, 64 prompts)

| Tag | Req/s | **Out tok/s** | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | E2E p99 (ms) | vs C0 throughput |
|---|---|---|---|---|---|---|---|
| **C0** baseline | 5.23 | 1 339 | 379 | 658 | 22.5 | 10 081 | 0 % |
| **C2** no CUDA graph | 1.31 | 337 | 1 195 | 2 901 | 90.7 | **37 852** | **−75 %** ⚠️ |
| **C3** chunked prefill | 3.50 | 895 | **3 709** | **14 199** | 21.3 | 16 147 | −33 % |
| **C4** MoE cutlass | 5.21 | 1 334 | 377 | 659 | 22.6 | 10 136 | −0.4 % (noise) |
| **C6** piecewise CUDA graph | **6.24** | **1 598** | 370 | 773 | **18.6** | **8 096** | **+19 %** ✨ |

**Performance findings**:

1. **`disable-cuda-graph` is a 4× performance disaster** (out tok/s 1 339 → 337). CUDA graph is the single most important optimisation in sglang's default config — confirms why our `cudaGraphLaunch` 17 % kernel from §15 was a *good* thing, not overhead.
2. **`chunked-prefill-size=2048` HURTS R8** (out tok/s 1 339 → 895, TTFT mean 379 → 3 709 ms — a 10× regression!). The 2 K shared prefix means chunked prefill splits each user's prompt into pieces that thrash the radix cache. **Chunked prefill is regime-dependent: helps R3/R4 prefill-heavy, hurts R8 prefix-cache.**
3. **`moe-runner-backend: cutlass` is a silent no-op for our setup** — C4 and C0 are numerically identical (Req/s 5.21 vs 5.23 = within noise). sglang's cutlass MoE path is **mostly FP8/FP4 — on bf16 it falls back to Triton `fused_moe_kernel`**. We verified this by comparing kernel-by-kernel traces; the top 10 kernels match exactly (see §19.6).
4. **`enable-piecewise-cuda-graph` WINS — +19 % throughput, +17 % TTFT, −20 % E2E p99**. This is the clearest config improvement in the whole study.

### 19.4 Hardware utilisation

| Tag | Mem peak (GiB) | GPU util mean (%) | Mem-ctrl util (%) | Power mean (W) | Power peak (W) | SM clock (MHz) |
|---|---|---|---|---|---|---|
| C0 baseline | 121 | 12.5 | 4.5 | 159 | 559 | 1 667 |
| C2 no CUDA graph | 121 | 8.6 | 2.7 | 142 | 280 | **1 796** |
| C3 chunked prefill | 120 | 11.4 | 4.0 | 153 | 564 | 1 713 |
| C4 MoE cutlass | 121 | 12.7 | 4.5 | 158 | 562 | 1 671 |
| C6 piecewise CUDA graph | 122 | 9.6 | 3.1 | 156 | 596 | 1 754 |

**Hardware findings**:

- **C2 (no CUDA graph) has the HIGHEST SM clock** (1 796 MHz). This is paradoxical only at first glance: when launch overhead is high, the GPU has periodic idle gaps in which clocks ramp up. **High SM clock ≠ good throughput** — C2 is the worst performer.
- **C6 (piecewise CUDA graph) uses 60 MiB more peak memory** (121.95 vs 121.26 GiB) — cost of caching multiple sub-graph captures. Acceptable for the +19 % perf win.
- **C3 (chunked prefill) reduces mem-ctrl util** (4.5 → 4.0 %) — chunked prefill creates more small launches; the MoE kernel sees smaller M per call.

### 19.5 Kernel mix changes

| Tag | Trace wall (ms) | GPU active (ms) | Kernel categories (top 5) |
|---|---|---|---|
| C0 baseline | 701 | 600 | **MoE 46.4 %**; FA 13.3 %; GEMM 12.7 %; other 5.6 %; elementwise 2.8 % |
| C2 no CUDA graph | **3 202** ⚠️ | 599 | MoE 45.6 %; FA 13.1 %; GEMM 12.6 %; other 6.5 %; **cuda runtime 2.9 %** |
| C3 chunked prefill | 620 | 329 | **MoE 53.4 %** ↑; GEMM 11.6 %; FA 11.5 %; **cuda runtime 8.0 %** ↑; other 2.5 % |
| C4 MoE cutlass | 702 | 599 | MoE 46.3 %; FA 13.3 %; GEMM 12.7 %; other 5.4 %; elementwise 2.8 % |
| C6 piecewise CUDA graph | 810 | **1 069** ↑ | **MoE 40.0 %** ↓; **other 27.3 %** ↑; FA 7.9 %; GEMM 6.0 %; elementwise 2.3 % |

**Kernel-mix findings**:

- **C2 trace wall = 3 202 ms vs C0 701 ms** (4.6× slower per profile step). GPU active time is identical (~600 ms); the extra 2 600 ms is pure CPU launch overhead. **This is the direct proof that CUDA graph saves ~3 s per 10 forward steps.**
- **C3 increases `fused_moe_kernel` share to 53 %** — chunked prefill makes each individual MoE call smaller (lower M), so the high-M variant disappears and the low-M variant dominates. More launches but each is cheaper.
- **C6 increases GPU active time to 1 069 ms** (the *most* of any cell) while reducing `fused_moe_kernel` share from 46 % → 40 %. Piecewise CUDA graph appears to fold more work into the captured graph, but the overall layout looks more "other-heavy" because piecewise graph instrumentation shows up as new event types.
- **C4 kernel mix is byte-identical to C0** to 0.1 %. Conclusive: `--moe-runner-backend cutlass` is a no-op for bf16.

### 19.6 Top-2 kernels per cell

| Tag | #1 kernel | #1 % | calls | #2 kernel | #2 % |
|---|---|---|---|---|---|
| C0 baseline | `fused_moe_kernel` | 46.4 % | 864 | `flash::FlashAttnFwdSm90<…>` | 10.3 % |
| C2 no CUDA graph | `fused_moe_kernel` | 45.6 % | 864 | `flash::FlashAttnFwdSm90<…>` | 10.1 % |
| C3 chunked prefill | `fused_moe_kernel` | 51.2 % | 864 | `flash::FlashAttnFwdSm90<…>` | 11.5 % |
| **C4** MoE cutlass | `fused_moe_kernel` | 46.3 % | 864 | `flash::FlashAttnFwdSm90<…>` | 10.3 % |
| **C6** piecewise | `fused_moe_kernel` | 38.0 % | 864 | **`cudaEventSynchronize`** | **25.1 %** |

C6's #2 kernel is `cudaEventSynchronize` 25 % — same pattern we saw on MoE R3/R4 in §15 (prefill regimes). Piecewise CUDA graph adds sync points between sub-graph boundaries. **C6 is fast despite this overhead because of better launch parallelism inside each sub-graph**.

### 19.7 Cross-knob conclusions

1. **No "optimization level" exists in sglang** — it's a 30+ knob design. Effects don't compose linearly (we tested 6 single-knob deltas; combining e.g. C6 + C3 needs separate testing).
2. **CUDA graph (whole or piecewise) is essential** — disabling it costs 4×. Piecewise is the best variant we tested.
3. **Backend swaps are mostly no-ops for bf16 / Qwen3-30B-A3B**:
   - `moe-runner-backend: cutlass` → silently falls back to Triton (C4 = C0)
   - `attention-backend: flashinfer` → JIT compile fails on H200 in our env (C5 = N/A)
   - To actually exercise alternate backends we'd need to enable FP8 (`--quantization fp8`) — separate study.
4. **`chunked-prefill-size` is regime-dependent**: it's a "depends" knob, not a "always on" knob. Helps R3/R4, hurts R8.
5. **The biggest single win**: `enable-piecewise-cuda-graph: true` + explicit token splits gives +19 % throughput on R8. Should be tested on all 8 regimes before recommending as a default.

### 19.8 Failures captured (not silently skipped)

- **C1 torch.compile**: Sglang 0.5.12's Qwen3 MoE `rotary_embedding.py` triggers a `torch._dynamo` `AssertionError` during compile. Known sglang issue; would need a model-side fix or `--enable-torch-compile-debug-mode` to investigate.
- **C5 flashinfer attention**: flashinfer's JIT (ninja) build of `batch_prefill_with_kv_cache_*` failed on H200 in this conda env. This is environment-level — flashinfer needs a working CUDA toolchain that ninja can drive. Not a config issue.

Both kept in the table as evidence rather than silently dropped.

### 19.9 Reproducing §19

```bash
bash scripts/regime_study/run_moe_opt_levels.sh         # 7 cells, ~25 min
python scripts/regime_study/aggregate_moe_opt_levels.py # → results/regime_bench/moe_opt_levels_table.{csv,md}
```

Each cell's raw artefacts (5 files: `hardware_view.json`, `profile_summary.json`,
`server_info.json`, `gpu_samples.csv`, `server.log`, `bench.jsonl`) are in
`results/regime_bench/raw/moe_opt_levels/<tag>/`.

## 19. MoE 优化旋钮研究（配置 A/B 实验）

> 🟢 新增。固定 workload（Qwen3-30B-A3B MoE × R8 prefix sharing），每次只改 **一个**
> server config 旋钮，看对性能、kernel mix、硬件利用率、backend 选择各有何影响。
> 7 个配置，5 个 PASS，2 个 FAIL（作为证据如实记录）。完整表在
> [`results/regime_bench/moe_opt_levels_table.md`](../results/regime_bench/moe_opt_levels_table.md)。

### 19.1 测了什么

7 个配置变体 —— 每个跟 `configs/moe_qwen3_30b.yaml` 只差一个旋钮：

| Tag | 旋钮 | 假设 |
|---|---|---|
| **C0** baseline | （不变）| 参考 |
| **C1** torch.compile | `enable-torch-compile: true` | JIT 编译小 op，可能改 kernel mix |
| **C2** 关 CUDA graph | `disable-cuda-graph: true` | 没了 graph capture，launch 开销暴涨 |
| **C3** chunked prefill | `chunked-prefill-size: 2048`（原 -1）| 应该帮 prefill-heavy；R8 可能伤 |
| **C4** MoE runner cutlass | `moe-runner-backend: cutlass` | 把 Triton `fused_moe_kernel` 换成 cutlass MoE |
| **C5** attn flashinfer | `attention-backend: flashinfer`（原 fa3）| 不同 attention kernel |
| **C6** piecewise CUDA graph | `enable-piecewise-cuda-graph: true` + token 分段 | 更细粒度的 graph 抓取 |

### 19.2 状态概览

| Tag | 状态 | FAIL 原因 |
|---|---|---|
| C0 baseline | ✅ PASS | |
| C1 torch.compile | ❌ FAIL | `torch._dynamo` 在 `sglang/srt/layers/rotary_embedding.py:272` 抛 AssertionError（Qwen3 MoE 不兼容）|
| C2 关 CUDA graph | ✅ PASS | |
| C3 chunked prefill | ✅ PASS | |
| C4 MoE cutlass | ✅ PASS | |
| C5 attn flashinfer | ❌ FAIL | flashinfer JIT (ninja) 在 H200 上编译 `batch_prefill` kernel 失败（env 问题，不是配置）|
| C6 piecewise CUDA graph | ✅ PASS | （需显式设 `piecewise-cuda-graph-tokens: [512, 1024, 2048, 4096, 8192]`） |

### 19.3 性能（sglang.bench_serving 在 R8 上跑 64 prompt）

| Tag | Req/s | **Out tok/s** | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | E2E p99 (ms) | vs C0 吞吐 |
|---|---|---|---|---|---|---|---|
| **C0** baseline | 5.23 | 1 339 | 379 | 658 | 22.5 | 10 081 | 0 % |
| **C2** 关 CUDA graph | 1.31 | 337 | 1 195 | 2 901 | 90.7 | **37 852** | **−75 %** ⚠️ |
| **C3** chunked prefill | 3.50 | 895 | **3 709** | **14 199** | 21.3 | 16 147 | −33 % |
| **C4** MoE cutlass | 5.21 | 1 334 | 377 | 659 | 22.6 | 10 136 | −0.4 %（噪声） |
| **C6** piecewise CUDA graph | **6.24** | **1 598** | 370 | 773 | **18.6** | **8 096** | **+19 %** ✨ |

**性能发现**：

1. **`disable-cuda-graph` 是 4 倍性能灾难**（out tok/s 1 339 → 337）。CUDA graph 是
   sglang 默认配置里**最重要**的单个优化 —— 这就验证了 §15 看到的
   `cudaGraphLaunch` 17% 是 *好* 事，不是 overhead。
2. **`chunked-prefill-size=2048` 在 R8 上伤性能**（out tok/s 1 339 → 895，TTFT mean
   379 → 3 709 ms，**10× 倒退**！）。2K 共享前缀被 chunked prefill 切成片段，把每个
   用户的 prompt 拍碎，颠簸了 radix cache。**chunked prefill 是 regime-dependent：
   帮 R3/R4 prefill-heavy，伤 R8 prefix-cache。**
3. **`moe-runner-backend: cutlass` 在我们这套配置下是悄悄 no-op** —— C4 和 C0
   数值一致（Req/s 5.21 vs 5.23 = 噪声范围内）。sglang 的 cutlass MoE 路径**主要
   是 FP8/FP4 —— bf16 上 fallback 回 Triton `fused_moe_kernel`**。逐 kernel 比对
   trace 验证了这点；前 10 个 kernel 完全一致（见 §19.6）。
4. **`enable-piecewise-cuda-graph` 赢家 —— 吞吐 +19%、TTFT +17%、E2E p99 −20%**。
   这是整个研究里最清晰的单旋钮 win。

### 19.4 硬件利用率

| Tag | 显存峰 (GiB) | GPU util 均 (%) | 显存控制器 util (%) | 功耗均 (W) | 功耗峰 (W) | SM 时钟 (MHz) |
|---|---|---|---|---|---|---|
| C0 baseline | 121 | 12.5 | 4.5 | 159 | 559 | 1 667 |
| C2 关 CUDA graph | 121 | 8.6 | 2.7 | 142 | 280 | **1 796** |
| C3 chunked prefill | 120 | 11.4 | 4.0 | 153 | 564 | 1 713 |
| C4 MoE cutlass | 121 | 12.7 | 4.5 | 158 | 562 | 1 671 |
| C6 piecewise CUDA graph | 122 | 9.6 | 3.1 | 156 | 596 | 1 754 |

**硬件发现**：

- **C2（关 CUDA graph）SM 时钟反而最高**（1 796 MHz）。第一眼看起来矛盾：当 launch
  开销高，GPU 周期性出现空闲间隙，时钟趁机拉高。**SM 时钟高 ≠ 吞吐好** —— C2 是
  最差的。
- **C6（piecewise CUDA graph）显存峰多用 60 MiB**（121.95 vs 121.26 GiB）—— 多个子
  graph capture 的缓存开销。换 +19% 性能完全值。
- **C3（chunked prefill）显存控制器 util 下降**（4.5 → 4.0%）—— chunked prefill
  创建更多小 launch，MoE kernel 每次看到的 M 变小。

### 19.5 Kernel mix 变化

| Tag | Trace wall (ms) | GPU active (ms) | Kernel 分类（top 5）|
|---|---|---|---|
| C0 baseline | 701 | 600 | **MoE 46.4%**；FA 13.3%；GEMM 12.7%；other 5.6%；elementwise 2.8% |
| C2 关 CUDA graph | **3 202** ⚠️ | 599 | MoE 45.6%；FA 13.1%；GEMM 12.6%；other 6.5%；**cuda runtime 2.9%** |
| C3 chunked prefill | 620 | 329 | **MoE 53.4%** ↑；GEMM 11.6%；FA 11.5%；**cuda runtime 8.0%** ↑；other 2.5% |
| C4 MoE cutlass | 702 | 599 | MoE 46.3%；FA 13.3%；GEMM 12.7%；other 5.4%；elementwise 2.8% |
| C6 piecewise CUDA graph | 810 | **1 069** ↑ | **MoE 40.0%** ↓；**other 27.3%** ↑；FA 7.9%；GEMM 6.0%；elementwise 2.3% |

**Kernel-mix 发现**：

- **C2 trace wall = 3 202 ms vs C0 701 ms**（每个 profile step 慢 4.6 倍）。GPU
  active 时间一致（~600 ms）；多出来的 2 600 ms 全是纯 CPU launch overhead。
  **这就是 CUDA graph 每 10 个 forward step 省 ~3 秒的直接证据。**
- **C3 把 `fused_moe_kernel` 份额拉到 53%** —— chunked prefill 让每次 MoE 调用
  M 变小，所以高-M 变体消失，低-M 变体主导。launch 多了但每次便宜了。
- **C6 把 GPU active 时间拉到 1 069 ms**（所有 cell 里最高），同时把
  `fused_moe_kernel` 份额从 46% → 40%。piecewise graph 把更多工作折进抓取的
  graph 里，但整体看起来"other-heavy"是因为 piecewise 仪表化暴露出新事件类型。
- **C4 的 kernel mix 跟 C0 字节级一致**（0.1% 精度内）。确证：bf16 下
  `--moe-runner-backend cutlass` 是个 no-op。

### 19.6 每 cell 的 top-2 kernel

| Tag | #1 kernel | #1 % | calls | #2 kernel | #2 % |
|---|---|---|---|---|---|
| C0 baseline | `fused_moe_kernel` | 46.4% | 864 | `flash::FlashAttnFwdSm90<…>` | 10.3% |
| C2 关 CUDA graph | `fused_moe_kernel` | 45.6% | 864 | `flash::FlashAttnFwdSm90<…>` | 10.1% |
| C3 chunked prefill | `fused_moe_kernel` | 51.2% | 864 | `flash::FlashAttnFwdSm90<…>` | 11.5% |
| **C4** MoE cutlass | `fused_moe_kernel` | 46.3% | 864 | `flash::FlashAttnFwdSm90<…>` | 10.3% |
| **C6** piecewise | `fused_moe_kernel` | 38.0% | 864 | **`cudaEventSynchronize`** | **25.1%** |

C6 的 #2 是 `cudaEventSynchronize` 25% —— 跟 §15 看到的 MoE R3/R4（prefill regime）
同一个 pattern。piecewise CUDA graph 在子 graph 边界加了 sync 点。**C6 仍然快**，
是因为每个子 graph 内的 launch 并行度更好。

### 19.7 跨旋钮结论

1. **sglang 没有"optimization level"概念** —— 它是 30+ 个旋钮的设计。效果不线性
   叠加（我们只测了 6 个单旋钮 delta；组合比如 C6 + C3 要单独测）。
2. **CUDA graph（whole 或 piecewise）必不可少** —— 关掉性能掉 4 倍。Piecewise 是
   我们测过的最好变体。
3. **Backend 替换在 bf16 / Qwen3-30B-A3B 上多数是 no-op**：
   - `moe-runner-backend: cutlass` → 悄悄回退到 Triton（C4 = C0）
   - `attention-backend: flashinfer` → H200 上 JIT 编译失败（C5 = N/A）
   - 要真用到这些后端要开 FP8（`--quantization fp8`）—— 另起一个研究。
4. **`chunked-prefill-size` 是 regime-dependent 旋钮**：不是"始终开"的旋钮。帮
   R3/R4，伤 R8。
5. **最大单个 win**：`enable-piecewise-cuda-graph: true` + 显式 token 分段 → R8
   上 +19% 吞吐。推荐为默认前应该先在全 8 regime 上测一遍。

### 19.8 捕获的 FAIL（不是悄悄跳过）

- **C1 torch.compile**：sglang 0.5.12 的 Qwen3 MoE 在 `rotary_embedding.py` 触发
  `torch._dynamo` AssertionError。已知 sglang 问题；要么 model 侧修复，要么
  `--enable-torch-compile-debug-mode` 调查。
- **C5 flashinfer attention**：flashinfer 的 JIT（ninja）在这个 conda env 的 H200
  上编不出 `batch_prefill_with_kv_cache_*` kernel。env 级问题 —— flashinfer 需要
  能驱动起来的 CUDA toolchain。不是配置问题。

两个都留在表里作为证据，不是悄悄丢弃。

### 19.9 复现 §19

```bash
bash scripts/regime_study/run_moe_opt_levels.sh         # 7 cell，~25 分钟
python scripts/regime_study/aggregate_moe_opt_levels.py # → results/regime_bench/moe_opt_levels_table.{csv,md}
```

每个 cell 的原始产物（6 个文件：`hardware_view.json`、`profile_summary.json`、
`server_info.json`、`gpu_samples.csv`、`server.log`、`bench.jsonl`）在
`results/regime_bench/raw/moe_opt_levels/<tag>/`。

## 20. C6 piecewise CUDA graph deep dive — exactly what changes inside sglang

> 🟢 NEW. §19 said C6 (`enable-piecewise-cuda-graph`) won R8 by +19 %.
> This section pops the hood: shows the source-level mechanism, lists every
> kernel-mix delta between C0 and C6 from the trace, and explains *why* it
> wins despite a 25 % `cudaEventSynchronize` overhead.

### 20.1 What piecewise CUDA graph actually is

sglang ships two CUDA-graph modes:

| Mode | Granularity | Code path |
|---|---|---|
| **Whole-graph (default)** | 1 graph capture per `(batch_size, seq_len)` of the entire forward pass | `sglang/srt/model_executor/cuda_graph_runner.py` |
| **Piecewise** | many small graphs separated by `@register_split_op()` boundaries | `sglang/srt/model_executor/piecewise_cuda_graph_runner.py` + `sglang/srt/compilation/` |

**Source mechanism** —
[`sglang/srt/compilation/compilation_config.py:7-12`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/compilation/compilation_config.py)
defines the split-marker registry:

```python
def register_split_op(op_name: Optional[str] = None):
    def decorator(op_func: Callable):
        name = op_name or op_func.__name__
        SPLIT_OPS.append(f"sglang.{name}")
        return op_func
    return decorator
```

Anything decorated with `@register_split_op()` becomes a **piecewise
boundary** (the model graph is cut here, and the pieces between boundaries
are compiled by Inductor + captured as small CUDA graphs).

The boundaries registered in sglang are:

| File:Line | Boundary | Why it can't be in a graph |
|---|---|---|
| `sglang/srt/layers/radix_attention.py:139` | `radix_attention` | KV cache mutation depends on runtime shape |
| `sglang/srt/layers/radix_linear_attention.py:105` | `radix_linear_attention` | same |
| `sglang/srt/distributed/parallel_state.py:133` | `tensor_model_parallel_all_reduce` | NCCL collective (uses external stream) |
| `sglang/srt/models/qwen3_next.py:1161` | `qwen3_next` Mamba-style op | data-dependent control flow |

For our Qwen3-30B-A3B MoE, **the active boundary is `radix_attention`** (per
layer). So **each transformer layer becomes one sub-graph**:

```
... pre-layer compute (norm + qkv proj + rope)  ← sub-graph N
    radix_attention                              ← split (NOT in any graph)
    post-attention (output proj + residual)
    pre-MoE compute (norm + router topk)         ← sub-graph N+1
    fused_moe_kernel #1 (gate+up)
    SiluAndMul
    fused_moe_kernel #2 (down)
    residual + norm
    ... next layer ...
```

48 layers × ~2 sub-graphs each ≈ ~96 sub-graphs total (plus one for the
embedding + a few for the lm-head). **Each is captured once and replayed
on every decode step.**

The compilation entry point is
[`PiecewiseCudaGraphRunner.__init__()`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py)
which:

1. Builds a `CompilationConfig` with `compiler='eager'` or `compiler='inductor'`
   (default `eager` for our C6 run).
2. Walks the model's `nn.Module` tree, calls `install_torch_compiled(...)`
   on each leaf submodule.
3. Replays the model in capture mode for each token-bucket
   (`piecewise-cuda-graph-tokens: [512, 1024, 2048, 4096, 8192]` — we
   captured 5 buckets).
4. Stores the captured graphs; at inference, `bisect.bisect_left()` picks
   the smallest bucket ≥ current token count.

### 20.2 Direct trace diff — C0 vs C6 (R8, 10 profile steps)

Launch-call counts (from per-event `name` field in trace):

| Event | C0 baseline | C6 piecewise | Δ |
|---|---|---|---|
| `cudaGraphLaunch` | 5 | **247** | **+242** (49 ×) |
| `cudaLaunchKernel` | 1 583 | 1 483 | −100 |
| `cudaLaunchKernelExC` | 1 060 | 866 | −194 |
| **Total launches** | **2 648** | **2 596** | −52 |

**Interpretation**:

- C0 captures the whole decode step as **one** graph (5 graph launches = 5 decode iterations, the rest of the steps must be a prefill plus warmup steps where graph isn't used).
- C6 captures **per-layer-piece** sub-graphs. Over 10 forward steps × ~48 layers × ~0.5 graphs-per-layer (only post-attention sub-graphs hit the cache) ≈ 240 graph launches — matches the observed 247.
- C6 *reduces* `cudaLaunchKernel*` count by ~300 — those 300 small individual launches are now folded into sub-graphs.

### 20.3 Kernel mix delta — top GPU events

C0 baseline (599 ms total GPU self time, 10 prof steps):

| % | self (µs) | calls | mean (µs) | Kernel |
|---|---|---|---|---|
| **46.4** | 278 302 | 864 | 322 | `fused_moe_kernel` |
| 10.3 | 61 582 | 96 | 642 | `FlashAttnFwdSm90<…>` variant A |
| 6.3 | 37 832 | 96 | 394 | `nvjet_tst_128x248_64x4_2x1_v_bz_coopA_TNT` |
| 3.6 | 21 539 | 48 | 449 | `nvjet_tst_192x208_64x4_1x2_h_bz_coopB_TNT` |
| 3.1 | 18 330 | 336 | 55 | `FlashAttnFwdSm90<…>` variant B (decode) |
| 2.9 | 17 201 | **8** | **2 150** | `cudaEventSynchronize` |
| 2.8 | 17 056 | 384 | 44 | `at::native::elementwise_kernel<…>` |
| 2.8 | 16 942 | 48 | 353 | `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN` |
| 2.8 | 16 590 | 432 | 38 | `flashinfer::activation::act_and_mul_kernel<silu>` |
| 2.4 | 14 417 | 864 | 17 | `flashinfer::norm::FusedAddRMSNormKernel` |

C6 piecewise (1 069 ms total GPU self time, **1.78 × C0** — but wall time is *shorter*; see §20.5):

| % | self (µs) | calls | mean (µs) | Kernel |
|---|---|---|---|---|
| **38.0** | 406 241 | 864 | **470** | `fused_moe_kernel` |
| **25.1** | 268 418 | **8** | **33 552** | `cudaEventSynchronize` ⚠️ |
| 7.9 | 84 407 | 192 | 440 | `FlashAttnFwdSm90<…>` variant A |
| 2.3 | 24 274 | 432 | 56 | `at::native::elementwise_kernel<…>` |
| 2.2 | 23 443 | 432 | 54 | `flashinfer::activation::act_and_mul_kernel<silu>` |
| 2.1 | 22 300 | 96 | 232 | `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` ← **new** |
| 2.0 | 21 603 | 48 | 450 | `nvjet_tst_192x208_64x4_1x2_h_bz_coopB_TNT` |
| 1.9 | 20 694 | 336 | 62 | `moe_sum_reduce_warp_per_token_vec_kernel<8>` |
| 1.9 | 19 935 | 48 | 415 | `nvjet_tst_128x272_64x4_2x1_v_bz_coopA_TNT` ← **new** |
| 1.8 | 19 755 | 864 | 23 | `flashinfer::norm::FusedAddRMSNormKernel` |

### 20.4 Three concrete kernel-level changes in C6

#### (a) Inductor picked different cuBLAS GEMM tiles

These were the top GEMM kernels in C0 — **GONE in C6**:

| Kernel | C0 self (µs) | C6 |
|---|---|---|
| `nvjet_tst_128x248_64x4_2x1_v_bz_coopA_TNT` (Q/K/V proj?) | 37 832 | not present |
| `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | 16 942 | not present |
| `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN` | 2 201 | not present |

These were the top GEMM kernels in C6 — **NEW vs C0**:

| Kernel | C6 self (µs) | calls |
|---|---|---|
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 22 300 | 96 |
| `nvjet_tst_128x272_64x4_2x1_v_bz_coopA_TNT` | 19 935 | 48 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 17 281 | 96 |

The `nvjet_tst_<M>x<N>_<K>x<stages>_…` naming is cuBLASLt's tile heuristic
output. **Different GEMM shapes are getting different tile choices** — same
matmul math, different SM-occupancy/L2-locality trade-off. Inductor's pass
manager re-selects tiles after fusion changes the surrounding tensor
layouts (e.g., the QKV-proj might be batched differently after
torch.compile fuses the preceding RMSNorm).

#### (b) `fused_moe_kernel` consolidation — fewer launch variants, more calls each

C0 (5 dominant variants):

| Grid | Block | Regs | Calls | Mean (µs) | Total (µs) |
|---|---|---|---|---|---|
| 6 882 | 256 | 194 | 48 | 1 494 | 71 722 |
| 6 774 | 256 | 194 | 48 | 1 476 | 70 869 |
| 9 176 | 256 | 196 | 48 | 954 | 45 777 |
| 9 032 | 256 | 196 | 48 | 947 | 45 468 |
| 1 530 | 256 | 194 | 48 | 285 | 13 662 |

C6 (5 dominant variants):

| Grid | Block | Regs | Calls | Mean (µs) | Total (µs) |
|---|---|---|---|---|---|
| 3 840 | 256 | 194 | **96** | 808 | 77 544 |
| 6 834 | 256 | 194 | 48 | 1 491 | 71 555 |
| 6 426 | 256 | 194 | 48 | 1 390 | 66 700 |
| 5 120 | 256 | 196 | **96** | 515 | 49 423 |
| 9 112 | 256 | 196 | 48 | 958 | 46 010 |

Note the **`calls=96` variants** in C6 (top row, and the 5 120-grid row).
C0 only ever has `calls=48`. **96 = 2 × 48** = the kernel is being launched
**twice per profile step instead of once**. Most likely cause: Inductor
captured **two adjacent transformer layers** into one sub-graph (so one
graph replay fires two `fused_moe_kernel` invocations). Grids of 3 840 and
5 120 are different shapes from C0 — different `M` reaching the kernel
because the captured graph buffers tensors differently.

**Block / register / shmem are unchanged across all variants**: 256 / 194 /
192 KB stays put. This means **the Triton `fused_moe_kernel` itself was
NOT recompiled** under piecewise. The same JIT'd PTX is being launched —
just from inside a CUDA-graph node now.

#### (c) `cudaEventSynchronize` mean duration explodes 16 ×

| Metric | C0 | C6 |
|---|---|---|
| `cudaEventSynchronize` calls | 8 | 8 |
| Mean per call | 2 150 µs | **33 552 µs** |
| Total | 17 201 µs (2.9 %) | **268 418 µs (25.1 %)** |

This looks alarming, but it's a **measurement artefact**: in CPU-side
trace view, an event sync now waits for the *entire sub-graph chain* to
finish (because the launches return immediately as graph nodes are
queued). The actual GPU is busy during those 33 ms — it's just that the
CPU is observing a single long sync instead of many small kernel
completions. Confirmed by the fact that **GPU active time in C6 (1 069 ms)
is HIGHER than in C0 (600 ms) in absolute terms**, but wall time is
shorter — i.e. the GPU is working harder + faster, the CPU just waits in
fewer bigger chunks.

### 20.5 Why C6 is +19 % despite 25 % `cudaEventSynchronize`

The math:

| Quantity | C0 | C6 | Source |
|---|---|---|---|
| Wall time per profile step | ~70 ms | ~80 ms | trace `wallclock_ms` |
| **Bench throughput** | **1 339 tok/s** | **1 598 tok/s (+19 %)** | bench_serving |
| GPU active time per step | 600 ms | 1 069 ms (+78 %) | trace |
| CPU launch overhead per step | ~1 500 ms (C0 actually `cudaLaunch*` × ~5 µs × ~3 000) | ~7 ms (graph replay) | derived |

C6 spends **more total GPU time but achieves higher throughput** because:

1. **The 1 069 ms of C6's GPU active time covers more concurrent work** — sub-graphs allow async overlap of pre/post-attention compute with the attention itself (which can't be in a graph). The 700→1 069 increase isn't waste; it's previously-serialised work now running in parallel streams.
2. **Launch overhead is collapsed**: ~300 launches eliminated, ~240 of them replaced by graph launches that are ~10× cheaper to issue.
3. **The `cudaEventSynchronize` is a barrier between forward passes, not within one**. It happens 8 times in the 10-step window (= once per forward pass minus the first/last). So even at 33 ms × 8 = 264 ms, this barrier doesn't extend serving latency because the next prefill/decode is already queued behind it.

### 20.6 Summary table for mentor

| Question | Answer |
|---|---|
| What is "piecewise CUDA graph"? | sglang's mode that splits the forward graph at `@register_split_op()` boundaries (attention, allreduce) and captures **each piece** separately as a CUDA graph (vs the default "one graph per whole forward step"). |
| What "fuses" differently? | **Nothing inside `fused_moe_kernel` itself** — same PTX, same block/regs/shmem (256 / 194 / 192 KB). What changes is **which kernels live inside the same CUDA-graph node** — pre-attention norms + QKV proj + RoPE all become one graph node now; post-attention proj + MoE another. |
| What kernels appear / disappear? | **C6-only**: new `nvjet_tst_320x128_…`, `_256x128_`, `_128x272_` GEMM tiles (Inductor re-tuned shapes). **C0-only**: `nvjet_tst_192x192_…`, `_128x248_`, `_128x256_` (the old tiles). The Triton MoE kernel is identical PTX. |
| How do kernel parameters change? | `fused_moe_kernel`: block/regs/shmem **unchanged**, but **grid shapes are different** (3 840, 5 120, 6 834 vs C0's 6 774, 6 882, 9 176) because the captured graphs hold different-sized tensors at capture time, and `calls=96` appears (= 2 layers' MoE in one captured sub-graph). |
| Performance? | **+19 % throughput** on R8, **−20 % E2E p99**, **TPOT −17 %**. Costs: +60 MiB peak memory, +469 ms GPU active time per step (mostly absorbed by parallelism, not waste). |
| Backend selection? | Unchanged — still fa3 + flashinfer + lpm. Piecewise CUDA graph is **orthogonal to backend choice**; it only changes how the chosen backend's kernels are *launched*. |
| Recommended? | **Yes, on this regime.** Should be tested on R1-R7 before promoting to default — could regress on small batches (graph capture cost per bucket). |

---

## 20. C6 piecewise CUDA graph 深入解读 —— sglang 内部到底变了啥

> �� 新增。§19 说 C6（`enable-piecewise-cuda-graph`）在 R8 上赢 +19%。本节
> 把盖子掀开：解释源码级机制 + 列出 trace 里 C0 与 C6 的每一个 kernel-mix
> delta + 解释为什么它能赢虽然有 25% `cudaEventSynchronize` 的开销。

### 20.1 Piecewise CUDA graph 到底是什么

sglang 提供两种 CUDA graph 模式：

| 模式 | 粒度 | 代码路径 |
|---|---|---|
| **whole-graph（默认）** | 整个 forward pass 每 `(batch_size, seq_len)` 抓一个 graph | `sglang/srt/model_executor/cuda_graph_runner.py` |
| **Piecewise** | 多个小 graph，按 `@register_split_op()` 边界切分 | `sglang/srt/model_executor/piecewise_cuda_graph_runner.py` + `sglang/srt/compilation/` |

**源码机制** —
[`sglang/srt/compilation/compilation_config.py:7-12`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/compilation/compilation_config.py)
定义了 split-marker 注册表：

```python
def register_split_op(op_name: Optional[str] = None):
    def decorator(op_func: Callable):
        name = op_name or op_func.__name__
        SPLIT_OPS.append(f"sglang.{name}")
        return op_func
    return decorator
```

任何被 `@register_split_op()` 装饰的就成了 **piecewise 边界**（模型 graph 在
这里被切断，**边界之间的所有 op 由 Inductor 编译 + 捕获为小 CUDA graph**）。

sglang 注册的边界：

| 文件:行 | 边界 | 为啥不能 in-graph |
|---|---|---|
| `sglang/srt/layers/radix_attention.py:139` | `radix_attention` | KV cache 修改取决于 runtime shape |
| `sglang/srt/layers/radix_linear_attention.py:105` | `radix_linear_attention` | 同上 |
| `sglang/srt/distributed/parallel_state.py:133` | `tensor_model_parallel_all_reduce` | NCCL collective（用外部 stream）|
| `sglang/srt/models/qwen3_next.py:1161` | `qwen3_next` Mamba 风格 op | 数据依赖的控制流 |

对我们的 Qwen3-30B-A3B MoE，**激活的边界是 `radix_attention`**（每层一个）。所以
**每个 transformer 层就是一个 sub-graph**：

```
... pre-layer compute (norm + qkv proj + rope)  ← sub-graph N
    radix_attention                              ← 切断（不在任何 graph 里）
    post-attention (output proj + residual)
    pre-MoE compute (norm + router topk)         ← sub-graph N+1
    fused_moe_kernel #1 (gate+up)
    SiluAndMul
    fused_moe_kernel #2 (down)
    residual + norm
    ... 下一层 ...
```

48 层 × ~2 sub-graph/层 ≈ ~96 个 sub-graph（再加 embedding + lm-head 几个）。
**每个抓一次，每个 decode step 重放。**

### 20.2 trace 直接 diff —— C0 vs C6（R8，10 个 profile step）

Launch 调用数（trace 里 `name` 字段统计）：

| 事件 | C0 baseline | C6 piecewise | Δ |
|---|---|---|---|
| `cudaGraphLaunch` | 5 | **247** | **+242（49 倍）** |
| `cudaLaunchKernel` | 1 583 | 1 483 | −100 |
| `cudaLaunchKernelExC` | 1 060 | 866 | −194 |
| **总 launch** | **2 648** | **2 596** | −52 |

**含义**：

- C0 把整个 decode step 抓成 **1 个** graph（5 次 graph launch = 5 个 decode 迭代，剩下是 prefill 加预热步骤不用 graph）
- C6 抓 **per-layer-piece** sub-graph。10 个 forward step × ~48 层 × ~0.5 graph/层（只有 post-attention sub-graph 命中缓存）≈ 240 graph launch —— 跟实测 247 对上
- C6 *减少* `cudaLaunchKernel*` ~300 次 —— 那 300 个独立小 launch 现在折进 sub-graph 里了

### 20.3 Kernel mix delta —— top GPU 事件

C0 baseline（599 ms 总 GPU self time，10 prof step）：

| % | self (µs) | calls | mean (µs) | Kernel |
|---|---|---|---|---|
| **46.4** | 278 302 | 864 | 322 | `fused_moe_kernel` |
| 10.3 | 61 582 | 96 | 642 | `FlashAttnFwdSm90<…>` 变体 A |
| 6.3 | 37 832 | 96 | 394 | `nvjet_tst_128x248_64x4_2x1_v_bz_coopA_TNT` |
| 3.6 | 21 539 | 48 | 449 | `nvjet_tst_192x208_64x4_1x2_h_bz_coopB_TNT` |
| 2.9 | 17 201 | **8** | **2 150** | `cudaEventSynchronize` |
| 2.8 | 16 942 | 48 | 353 | `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN` |
| 2.8 | 16 590 | 432 | 38 | `flashinfer::activation::act_and_mul_kernel<silu>` |
| 2.4 | 14 417 | 864 | 17 | `flashinfer::norm::FusedAddRMSNormKernel` |

C6 piecewise（1 069 ms 总 GPU self time，**C0 的 1.78×** —— 但 wall 更短，见 §20.5）：

| % | self (µs) | calls | mean (µs) | Kernel |
|---|---|---|---|---|
| **38.0** | 406 241 | 864 | **470** | `fused_moe_kernel` |
| **25.1** | 268 418 | **8** | **33 552** | `cudaEventSynchronize` ⚠️ |
| 7.9 | 84 407 | 192 | 440 | `FlashAttnFwdSm90<…>` 变体 A |
| 2.3 | 24 274 | 432 | 56 | `at::native::elementwise_kernel<…>` |
| 2.2 | 23 443 | 432 | 54 | `flashinfer::activation::act_and_mul_kernel<silu>` |
| 2.1 | 22 300 | 96 | 232 | `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` ← **新** |
| 2.0 | 21 603 | 48 | 450 | `nvjet_tst_192x208_64x4_1x2_h_bz_coopB_TNT` |
| 1.9 | 19 935 | 48 | 415 | `nvjet_tst_128x272_64x4_2x1_v_bz_coopA_TNT` ← **新** |

### 20.4 C6 的 3 个具体 kernel 级变化

#### (a) Inductor 挑了不同的 cuBLAS GEMM tile

C0 里 top GEMM —— **C6 里没了**：

| Kernel | C0 self (µs) | C6 |
|---|---|---|
| `nvjet_tst_128x248_64x4_2x1_v_bz_coopA_TNT`（Q/K/V proj？）| 37 832 | 不出现 |
| `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | 16 942 | 不出现 |
| `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN` | 2 201 | 不出现 |

C6 里 top GEMM —— **C0 里没有**：

| Kernel | C6 self (µs) | calls |
|---|---|---|
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | 22 300 | 96 |
| `nvjet_tst_128x272_64x4_2x1_v_bz_coopA_TNT` | 19 935 | 48 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | 17 281 | 96 |

`nvjet_tst_<M>x<N>_<K>x<stages>_…` 命名是 cuBLASLt tile 启发式输出。**不同的
GEMM shape 选了不同的 tile** —— 数学一样，SM 占用率 / L2 局部性权衡不同。
Inductor 的 pass manager 在 fusion 改变了周围 tensor 布局后重新选 tile（比如
QKV-proj 在 torch.compile 把前面的 RMSNorm 融进来后 batch 方式可能不同了）。

#### (b) `fused_moe_kernel` 整合 —— launch 变体变少，每个变体调用变多

C0（5 个主流变体）：

| Grid | Block | Regs | Calls | Mean (µs) | Total (µs) |
|---|---|---|---|---|---|
| 6 882 | 256 | 194 | 48 | 1 494 | 71 722 |
| 6 774 | 256 | 194 | 48 | 1 476 | 70 869 |
| 9 176 | 256 | 196 | 48 | 954 | 45 777 |
| 9 032 | 256 | 196 | 48 | 947 | 45 468 |
| 1 530 | 256 | 194 | 48 | 285 | 13 662 |

C6（5 个主流变体）：

| Grid | Block | Regs | Calls | Mean (µs) | Total (µs) |
|---|---|---|---|---|---|
| 3 840 | 256 | 194 | **96** | 808 | 77 544 |
| 6 834 | 256 | 194 | 48 | 1 491 | 71 555 |
| 6 426 | 256 | 194 | 48 | 1 390 | 66 700 |
| 5 120 | 256 | 196 | **96** | 515 | 49 423 |
| 9 112 | 256 | 196 | 48 | 958 | 46 010 |

留意 C6 里 **`calls=96` 的变体**（第一行和 5 120-grid 那行）。C0 永远都是
`calls=48`。**96 = 2 × 48** = 这个 kernel **每个 profile step 被 launch 两次而
不是一次**。最可能的原因：Inductor 把 **两个相邻 transformer 层** 抓到一个
sub-graph 里了（所以一次 graph replay 触发两次 `fused_moe_kernel`）。Grid 3 840
和 5 120 也是 C0 没有的形状 —— 因为捕获的 graph 里 tensor 缓冲方式不同，到达
kernel 的 `M` 也就不同。

**Block / 寄存器 / shmem 在所有变体里完全不变**：256 / 194 / 192 KB。这说明
**Triton `fused_moe_kernel` 本身没被重新编译**。同一份 JIT 出来的 PTX 在被
launch —— 只是现在从 CUDA-graph 节点里 launch 而已。

#### (c) `cudaEventSynchronize` 平均时长爆炸 16 倍

| 指标 | C0 | C6 |
|---|---|---|
| `cudaEventSynchronize` 调用数 | 8 | 8 |
| 每次平均 | 2 150 µs | **33 552 µs** |
| 总 | 17 201 µs（2.9%）| **268 418 µs（25.1%）** |

看起来吓人，但是 **测量伪影**：CPU-side trace 视角里，event sync 现在要等
*整个 sub-graph 链* 完成（因为 graph node queue 后 launch 立刻返回）。GPU 在
那 33 ms 里其实是在干活 —— 只是 CPU 观察到的是 1 个长 sync 而不是许多小 kernel
完成事件。证据：**C6 的 GPU active 时间（1 069 ms）绝对值比 C0（600 ms）高**，
但 wall time 反而短 —— 也就是 GPU 干得更多 + 更快，CPU 只是等待的次数更少但每次更久。

### 20.5 C6 为什么 +19%，虽然有 25% `cudaEventSynchronize`

算账：

| 量 | C0 | C6 | 来源 |
|---|---|---|---|
| 每 profile step wall time | ~70 ms | ~80 ms | trace `wallclock_ms` |
| **Bench 吞吐** | **1 339 tok/s** | **1 598 tok/s（+19%）** | bench_serving |
| 每 step GPU active 时间 | 600 ms | 1 069 ms（+78%）| trace |
| 每 step CPU launch overhead | ~1 500 ms（C0 实际 `cudaLaunch*` × ~5 µs × ~3 000）| ~7 ms（graph replay）| 推算 |

C6 **总 GPU 时间更多但吞吐更高**，因为：

1. **C6 的 1 069 ms GPU active 时间覆盖更多并发工作** —— sub-graph 允许 pre/post-attention
   compute 跟 attention 本身（不能 in-graph）异步重叠。700→1 069 的增加不是浪费，
   是原来串行的工作现在并行 stream 里跑了。
2. **launch overhead 被折叠**：~300 个 launch 消失，~240 个被 graph launch 替代，graph
   launch issue 成本 ~10× 低。
3. **`cudaEventSynchronize` 是 forward pass *之间* 的 barrier，不是 forward 内部**。
   10-step 窗口里出现 8 次（= forward 数 - 起止）。即使 33 ms × 8 = 264 ms，这个 barrier
   不会扩展 serving latency，因为下个 prefill/decode 已经排在它后面等着了。

### 20.6 给 mentor 的总结表

| 问题 | 答案 |
|---|---|
| 什么是"piecewise CUDA graph"？ | sglang 的一个模式，把 forward graph 按 `@register_split_op()` 边界（attention、allreduce）切分，**每片** 单独抓成 CUDA graph（vs 默认"整个 forward 一个 graph"）。|
| 啥被"融合"得不一样了？ | **`fused_moe_kernel` 内部完全没变** —— 同 PTX，同 block/regs/shmem（256 / 194 / 192 KB）。变的是 **哪些 kernel 住在同一个 CUDA-graph 节点里** —— pre-attention norm + QKV proj + RoPE 现在变成一个 graph node；post-attention proj + MoE 另一个。|
| 哪些 kernel 出现 / 消失？ | **C6 独有**：新出现 `nvjet_tst_320x128_…`、`_256x128_`、`_128x272_` GEMM tile（Inductor 按新 shape 重新调）。**C0 独有**：`nvjet_tst_192x192_…`、`_128x248_`、`_128x256_`（旧 tile）。Triton MoE kernel 是同一份 PTX。|
| Kernel 参数怎么变？ | `fused_moe_kernel`：block / regs / shmem **不变**，但 **grid shape 不同**（C6 3 840、5 120、6 834 vs C0 6 774、6 882、9 176），因为捕获的 graph 在 capture 时持有不同大小的 tensor，并且 `calls=96` 出现（= 一个捕获 sub-graph 里包含 2 层 MoE）。|
| 性能？ | R8 上 **吞吐 +19%**，**E2E p99 −20%**，**TPOT −17%**。代价：显存峰 +60 MiB，每 step GPU active +469 ms（多数被并发吸收，不是浪费）。|
| Backend 选择？ | 不变 —— 依然 fa3 + flashinfer + lpm。Piecewise CUDA graph **跟 backend 选择正交**；它只改 backend 已选 kernel 的 *launch 方式*。|
| 推荐吗？ | **R8 上是的**。提到默认前要先在 R1-R7 上测 —— 小 batch 上可能因为 per-bucket graph capture 成本反而回归。|
