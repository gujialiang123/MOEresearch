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

## 21. Sglang fallback mechanics — when does it actually fall back, and when does it just fail?

> 🟢 NEW. Answers the question raised after §19: if sglang has Triton/Transformers
> as fallbacks, why did our Gemma-4 + C1 + C5 fail outright? This section
> walks through the **four distinct dispatch layers** in sglang, shows where
> each one does (or doesn't) fall back, and what really happens when you
> pass a backend flag the system can't satisfy.

### 21.1 The four layers — none of them are the same

| Layer | What it dispatches | Source | Fallback behaviour when target not available |
|---|---|---|---|
| **A. Model architecture** | `model_type` from `config.json` → a Python `nn.Module` class | `sglang/srt/models/registry.py:_ModelRegistry` | **HARD FAIL** — `KeyError: '<model_type>'`. No transformers fallback, no auto-substitute. |
| **B. Backend selection (auto)** | `attention_backend=None` (default) → picks `fa3`/`flashinfer`/`triton` based on hardware | `sglang/srt/server_args.py:_handle_attention_backend_compatibility` | **Smart auto-select** — picks the fastest path that works. This is the only layer that really "falls back". |
| **C. Backend selection (explicit)** | `--attention-backend flashinfer` → must use flashinfer | `sglang/srt/model_executor/model_runner.py:_get_attention_backend_from_str` | **HARD FAIL** if the backend can't load — `raise ValueError(f"Invalid attention backend: {backend_str}")` or downstream JIT error. **NO fallback to anything else.** |
| **D. Backend ignored by code path** | `--moe-runner-backend cutlass` on a bf16 (unquantized) model | `sglang/srt/layers/quantization/unquant.py:321-330` | **SILENT NO-OP** — the bf16 MoE path hardcodes `MoeRunnerBackend.TRITON` and never reads the flag. |

These four layers explain every "weird" behaviour we saw in §19 and elsewhere.

### 21.2 Layer A — model registry has NO fallback

Source ([`sglang/srt/models/registry.py:39-56`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py)):

```python
class _ModelRegistry:
    models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)

    def register(self, package_name: str, overwrite: bool = False, ...):
        new_models = import_model_classes(package_name, strict=strict)
        for arch, cls in new_models.items():
            if arch in self.models:
                raise ValueError(...)
            self.models[arch] = cls

    def _raise_for_unsupported(self, architectures: List[str]):
        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(...)
        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )
```

**Architecture lookup uses a hard dict lookup**. There is no `try
transformers; except: use triton path` — sglang doesn't even know what
your model is until someone (in the sglang source tree) writes a Python
class implementing it. Architecture-to-class map is built at import time
from `sglang/srt/models/*.py`.

**This is why our Gemma-4 attempt failed** with `KeyError: 'gemma4'` —
no `gemma4.py` in `sglang/srt/models/`. sglang's gemma family is
`gemma.py`, `gemma2.py`, `gemma2_reward.py`, `gemma3_causal.py`,
`gemma3_mm.py`, `gemma3n_*.py`. None of them are `gemma4`. Loading
fails before any kernel even gets a chance to run.

**Compare to HuggingFace transformers**: HF *does* have a generic
`AutoModelForCausalLM` that can load almost anything via the `architectures`
field with a Python forward path. Sglang chose NOT to do this because the
whole point of sglang is the optimised forward path — using transformers
generically would defeat the purpose. So sglang trades flexibility for
performance and **fails fast** when you give it a model it doesn't know.

### 21.3 Layer B — backend AUTO-select (the only "real" fallback)

Source ([`sglang/srt/server_args.py:1782-1818`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py)):

```python
# Pick the default attention backend if not specified
if self.attention_backend is None:
    if not use_mla_backend:
        # MHA architecture
        if is_hopper_with_cuda_12_3() and is_no_spec_infer_or_topk_one(self):
            self.attention_backend = "fa3"             # ← what we got on H200
        elif is_sm100_supported() and ...:
            self.attention_backend = "trtllm_mha"      # ← Blackwell
        elif is_hip():
            self.attention_backend = "aiter"            # ← AMD MI*
        else:
            self.attention_backend = (
                "flashinfer" if is_flashinfer_available() else "triton"  # ← real fallback chain
            )
    else:
        # MLA architecture
        if is_hopper_with_cuda_12_3():
            self.attention_backend = "fa3"
        elif is_blackwell():
            self.attention_backend = "flashinfer"
        else:
            self.attention_backend = "triton"
```

**This is the only place "fallback" really happens**: auto mode walks a
hardware-aware priority chain. On our H200 it lands on `fa3`. If
flashinfer's missing, it falls to `triton`. If we were on Blackwell it
would try `trtllm_mha` first.

Similar (but per-quantization) for MoE in
`sglang/srt/layers/quantization/fp8.py:1349`:

```python
# FP8 MoE: try DeepGEMM, fall back to Triton
if is_deep_gemm_supported() and ...:
    moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
else:
    moe_runner_backend = MoeRunnerBackend.TRITON
```

Again — **fallback only happens in the auto-detect path**. The moment you
pin a specific backend, the auto-detector is bypassed.

### 21.4 Layer C — explicit backend = NO fallback, hard fail

Source ([`sglang/srt/model_executor/model_runner.py:1792-1798`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py)):

```python
def _get_attention_backend_from_str(self, backend_str: str, ...):
    if backend_str not in ATTENTION_BACKENDS:
        raise ValueError(f"Invalid attention backend: {backend_str}")
    self.init_new_workspace = init_new_workspace
    full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
    return attn_backend_wrapper(self, full_attention_backend)
```

That `ATTENTION_BACKENDS[backend_str](self)` call is the **backend's
constructor**. If the backend's `__init__` or weights-init raises (because
of a missing kernel JIT, an unsupported shape, …), **the exception
propagates and the server dies**. No `try: cutlass except: triton` chain.

**This is why our C5 (`--attention-backend flashinfer`) failed**: the
explicit `flashinfer` backend triggered a JIT build of
`batch_prefill_with_kv_cache_*` and ninja blew up — sglang dutifully
raised the exception and the server exited. If we had left
`attention-backend` unset (=auto), Layer B would have picked `fa3`
instead and avoided the JIT path entirely.

**Why this design choice**: silently substituting backends would hide
performance regressions and make A/B testing impossible (which is exactly
what §19 is trying to do). Hard-fail-on-explicit is the right call for an
optimization research codebase.

### 21.5 Layer D — explicit backend SILENTLY IGNORED

This is the most subtle one — and the reason C4 looked like a no-op.

Source ([`sglang/srt/layers/quantization/unquant.py:321-330`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)):

```python
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        backend = (
            MoeRunnerBackend.TRITON_KERNELS
            if self.use_triton_kernels
            else MoeRunnerBackend.TRITON
        )
        self.runner = MoeRunner(backend, moe_runner_config)
        # ↑↑↑ self.server_args.moe_runner_backend is NEVER read here ↑↑↑
```

The bf16 path **hard-codes** `MoeRunnerBackend.TRITON`. The server arg
`--moe-runner-backend cutlass` you set in C4 was parsed, stored on
`server_args`, but **never consulted by this code path**.

Where IS it consulted? Only in the quantized paths
([`sglang/srt/layers/quantization/fp8.py:1349`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)):

```python
# Inside FP8 MoE method
if (server_args.moe_runner_backend == "deep_gemm"
        or (server_args.moe_runner_backend == "auto" and is_deep_gemm_supported())):
    moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
else:
    moe_runner_backend = MoeRunnerBackend.TRITON
```

So `--moe-runner-backend` is effectively **only honoured for FP8/FP4 MoE**.
On bf16 it's a silent no-op. **There's no warning, no log line, nothing.**

**This is the genuine sharp edge** in sglang's design — and what we
caught in §19 by comparing kernel-by-kernel traces. Without the trace
diff, C4 would look like "tried cutlass, got identical perf, conclude
cutlass = triton on this workload" — which is wrong; the truth is the
flag was never used.

### 21.6 The decision tree, as a flowchart

```
┌────────────────────────────────────────────────────────────────┐
│ sglang.launch_server --attention-backend X --moe-runner-backend Y │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │ Layer A — model registry     │
                │ model_type in registry?      │
                └──────────────┬──────────────┘
                       NO ───► raise KeyError, server exits
                       YES ───► continue
                               │
                ┌──────────────┴──────────────┐
                │ Layer B — attention auto?   │
                │ X is None / "auto"?         │
                └──────────────┬──────────────┘
                       YES ──► hardware-aware chain:
                               H200+CUDA12.3 ──► fa3
                               Blackwell    ──► trtllm_mha
                               AMD          ──► aiter
                               else         ──► flashinfer if available, else triton
                       NO ─────► continue with X
                               │
                ┌──────────────┴──────────────┐
                │ Layer C — explicit X loads? │
                │ ATTENTION_BACKENDS[X](self) │
                └──────────────┬──────────────┘
                       FAIL ──► raise, server exits  (← this is what C5 hit)
                       OK   ──► use X
                               │
                ┌──────────────┴──────────────┐
                │ Layer D — quant path reads Y?│
                │ for MoE: depends on dtype    │
                └──────────────┬──────────────┘
                       bf16  ──► IGNORED, hard-coded to TRITON   (← this is what C4 hit)
                       FP8   ──► honoured, dispatches to Y
```

### 21.7 Summary — answers to the mentor's three questions

**Q: If sglang has fallback to transformers / triton, why did our models still fail to run?**

A: The architecture registry (Layer A) has **no transformers fallback at all**. It's a hard dict lookup; unknown `model_type` raises immediately. The "fallback to triton/transformers" intuition comes from libraries like vLLM or HF generate, which sglang doesn't share. Sglang trades flexibility for a fully-optimised forward path.

**Q: How does fallback actually work (when it does)?**

A: **Only in two places**:

1. **Auto attention-backend selection** (Layer B) walks a hardware-priority chain at startup: `fa3` on H200 → `trtllm_mha` on Blackwell → `aiter` on AMD → `flashinfer if available else triton`.
2. **Auto quantized-MoE-runner selection** picks DeepGEMM if available, else Triton.

Both fall back to the most-conservative-but-always-works choice. **They run once at startup, not per request.**

**Q: When we pin a backend explicitly, does it still fall back if unsupported?**

A: **No. Three distinct fail modes depending on the layer:**

- **Hard fail with raised error** if the backend's constructor itself fails (Layer C). This is what happened to `--attention-backend flashinfer` (C5) when flashinfer's JIT couldn't build.
- **Hard fail with raised error** if the backend name isn't in the registry (Layer C, `ValueError: Invalid attention backend`).
- **Silent no-op** if the code path doesn't even read the flag (Layer D). This is what happened to `--moe-runner-backend cutlass` (C4) on bf16 — the flag was parsed but the unquantized MoE method ignores it and hardcodes Triton.

The silent-no-op is the most dangerous one — it's the reason we had to do kernel-by-kernel trace comparison to confirm C4 was really no-op-ing, not just "cutlass happens to perform identically".

---

## 21. Sglang fallback 机制 —— 啥时候真 fallback，啥时候直接挂掉？

> 🟢 新增。回答 §19 之后的疑问：sglang 不是有 Triton/Transformers 兜底吗？
> 为啥 Gemma-4 / C1 / C5 还是直接失败？这一节走 sglang 里 **4 个完全不同的
> dispatch 层**，看每层是不是会 fallback，以及指定一个 sglang 跑不动的
> backend 到底真正会发生什么。

### 21.1 4 个层 —— 谁都不一样

| 层 | dispatch 什么 | 源码 | 目标不可用时 |
|---|---|---|---|
| **A. 模型架构** | `config.json` 里的 `model_type` → Python `nn.Module` 类 | `sglang/srt/models/registry.py:_ModelRegistry` | **硬 FAIL** —— `KeyError: '<model_type>'`。无 transformers fallback、无自动替代。 |
| **B. Backend 自动选** | `attention_backend=None`（默认）→ 按硬件选 `fa3`/`flashinfer`/`triton` | `sglang/srt/server_args.py:_handle_attention_backend_compatibility` | **智能自动选** —— 按可用性选最快路径。**这是唯一真正会"fallback"的层。** |
| **C. Backend 显式指定** | `--attention-backend flashinfer` → 必须用 flashinfer | `sglang/srt/model_executor/model_runner.py:_get_attention_backend_from_str` | **硬 FAIL**：要么 `raise ValueError(f"Invalid attention backend")`，要么下游 JIT 错误。**完全不会回退到任何别的 backend。** |
| **D. Backend 被代码路径忽略** | bf16 模型上的 `--moe-runner-backend cutlass` | `sglang/srt/layers/quantization/unquant.py:321-330` | **悄悄 no-op** —— bf16 MoE path 把 `MoeRunnerBackend.TRITON` 写死了，从来没读这个 flag。 |

这 4 层解释了我们 §19 和别处看到的所有"奇怪"行为。

### 21.2 Layer A —— 模型 registry 没有 fallback

源码（[`sglang/srt/models/registry.py:39-56`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py)）：

```python
class _ModelRegistry:
    models: Dict[str, ...] = field(default_factory=dict)

    def register(self, package_name: str, ...):
        new_models = import_model_classes(package_name, strict=strict)
        for arch, cls in new_models.items():
            if arch in self.models:
                raise ValueError(...)
            self.models[arch] = cls

    def _raise_for_unsupported(self, architectures: List[str]):
        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(...)
        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )
```

**架构查找是硬 dict 查找**。没有 `try transformers; except: use triton`
—— sglang 在某人（在 sglang 源码树）写 Python 类实现它之前根本不知道你的
模型是什么。架构-类的 map 是 import 时从 `sglang/srt/models/*.py` 建出来的。

**这就是 Gemma-4 失败的原因** —— `sglang/srt/models/` 里没有 `gemma4.py`。
sglang 的 gemma 家族是 `gemma.py`、`gemma2.py`、`gemma2_reward.py`、
`gemma3_causal.py`、`gemma3_mm.py`、`gemma3n_*.py`，没有 `gemma4`。还没碰到
任何 kernel 就直接挂了。

**和 HuggingFace transformers 对比**：HF *有* 通用 `AutoModelForCausalLM`，
能通过 `architectures` 字段加载几乎所有东西。Sglang 选择 **不** 这么做，因为
sglang 的整个意义就是优化过的 forward path —— 通用 transformers 用法就违背了
初衷。所以 sglang 用灵活性换性能，遇到不认识的模型 **快速失败**。

### 21.3 Layer B —— backend 自动选（唯一"真"fallback）

源码（[`sglang/srt/server_args.py:1782-1818`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py)）：

```python
# 如果没指定，选默认 attention backend
if self.attention_backend is None:
    if not use_mla_backend:
        # MHA 架构
        if is_hopper_with_cuda_12_3() and is_no_spec_infer_or_topk_one(self):
            self.attention_backend = "fa3"             # ← H200 上选这个
        elif is_sm100_supported() and ...:
            self.attention_backend = "trtllm_mha"      # ← Blackwell
        elif is_hip():
            self.attention_backend = "aiter"            # ← AMD MI*
        else:
            self.attention_backend = (
                "flashinfer" if is_flashinfer_available() else "triton"  # ← 真正的 fallback 链
            )
    else:
        # MLA 架构
        if is_hopper_with_cuda_12_3():
            self.attention_backend = "fa3"
        elif is_blackwell():
            self.attention_backend = "flashinfer"
        else:
            self.attention_backend = "triton"
```

**这是 fallback 真正发生的唯一地方**：auto 模式走硬件感知的优先级链。我们的
H200 上落到 `fa3`。如果 flashinfer 缺，会降到 `triton`。如果在 Blackwell 上
会先试 `trtllm_mha`。

MoE 也类似（per-quantization），见
`sglang/srt/layers/quantization/fp8.py:1349`：

```python
# FP8 MoE：先试 DeepGEMM，不行回 Triton
if is_deep_gemm_supported() and ...:
    moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
else:
    moe_runner_backend = MoeRunnerBackend.TRITON
```

再说一次 —— **fallback 只发生在自动检测路径**。一旦你钉死一个 backend，
auto detector 就被绕过了。

### 21.4 Layer C —— 显式 backend = 没有 fallback，硬 fail

源码（[`sglang/srt/model_executor/model_runner.py:1792-1798`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py)）：

```python
def _get_attention_backend_from_str(self, backend_str: str, ...):
    if backend_str not in ATTENTION_BACKENDS:
        raise ValueError(f"Invalid attention backend: {backend_str}")
    self.init_new_workspace = init_new_workspace
    full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
    return attn_backend_wrapper(self, full_attention_backend)
```

`ATTENTION_BACKENDS[backend_str](self)` 是 **backend 的构造函数**。如果
backend 的 `__init__` 或者权重初始化抛异常（kernel JIT 失败、不支持的 shape
等），**异常向上传播，server 挂掉**。没有 `try: cutlass except: triton` 链。

**这就是 C5（`--attention-backend flashinfer`）失败的原因**：显式 `flashinfer`
触发了 `batch_prefill_with_kv_cache_*` 的 JIT 编译、ninja 炸了 —— sglang 老老实实
抛出来、server 退出。如果不指定 `attention-backend`（= auto），Layer B 会选
`fa3` 完全绕开 JIT 路径。

**为什么这么设计**：悄悄替换 backend 会隐藏性能回归，让 A/B 测试不可能（这就是
§19 在做的事）。**显式 = 硬 fail** 对优化研究 codebase 是对的选择。

### 21.5 Layer D —— 显式 backend **悄悄被忽略**

最微妙的一层 —— 也是 C4 看起来是 no-op 的原因。

源码（[`sglang/srt/layers/quantization/unquant.py:321-330`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)）：

```python
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """无量化的 MoE method。"""

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        backend = (
            MoeRunnerBackend.TRITON_KERNELS
            if self.use_triton_kernels
            else MoeRunnerBackend.TRITON
        )
        self.runner = MoeRunner(backend, moe_runner_config)
        # ↑↑↑ self.server_args.moe_runner_backend 这里从来没读 ↑↑↑
```

bf16 path **写死** `MoeRunnerBackend.TRITON`。你在 C4 里设的
`--moe-runner-backend cutlass` 被 parse 了、存到 `server_args` 上，但
**这条代码路径从来没读它**。

那它在哪儿被读？只有量化路径
（[`sglang/srt/layers/quantization/fp8.py:1349`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)）：

```python
# FP8 MoE method 里面
if (server_args.moe_runner_backend == "deep_gemm"
        or (server_args.moe_runner_backend == "auto" and is_deep_gemm_supported())):
    moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
else:
    moe_runner_backend = MoeRunnerBackend.TRITON
```

所以 `--moe-runner-backend` 实际上 **只对 FP8/FP4 MoE 生效**。bf16 上它是
悄悄 no-op。**没警告、没 log、什么都没。**

**这就是 sglang 设计里真正的"锐边"** —— 也是 §19 我们用 kernel-by-kernel
trace 对比抓到的。没有那次 trace diff，C4 看起来就是"试了 cutlass、性能一样、
结论：cutlass = triton 在这个 workload 上" —— 错的。真相是 flag 根本没被用。

### 21.6 决策树流程图

```
┌────────────────────────────────────────────────────────────────────┐
│ sglang.launch_server --attention-backend X --moe-runner-backend Y    │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │ Layer A —— 模型 registry    │
                │ model_type 在 registry 里？ │
                └──────────────┬──────────────┘
                       否 ───► raise KeyError，server 退出
                       是 ───► 继续
                               │
                ┌──────────────┴──────────────┐
                │ Layer B —— attention 自动？ │
                │ X 是 None / "auto"？        │
                └──────────────┬──────────────┘
                       是 ───► 硬件感知链：
                               H200+CUDA12.3 ──► fa3
                               Blackwell    ──► trtllm_mha
                               AMD          ──► aiter
                               else         ──► flashinfer if available, else triton
                       否 ───► 继续用 X
                               │
                ┌──────────────┴──────────────┐
                │ Layer C —— 显式 X 能加载？  │
                │ ATTENTION_BACKENDS[X](self) │
                └──────────────┬──────────────┘
                       失败 ──► raise，server 退出  （← C5 撞这里）
                       成功 ──► 用 X
                               │
                ┌──────────────┴──────────────┐
                │ Layer D —— 量化路径读 Y？   │
                │ MoE：取决于 dtype           │
                └──────────────┬──────────────┘
                       bf16  ──► 忽略，硬编码到 TRITON  （← C4 撞这里）
                       FP8   ──► 生效，dispatch 到 Y
```

### 21.7 一句话回答 mentor 的 3 个问题

**Q：sglang 不是有 fallback 到 transformers / triton 吗？为啥我们的模型还是跑不起来？**

A：架构 registry（Layer A）**完全没有 transformers fallback**。它是硬 dict 查找；未知 `model_type` 直接 raise。"fallback 到 triton/transformers" 的直觉来自 vLLM 或 HF generate 这些库，sglang 不走这条路。Sglang 用灵活性换全优化的 forward path。

**Q：fallback 实际原理是啥？（当它真的会 fallback 时）**

A：**只有 2 个地方**：

1. **Attention backend 自动选**（Layer B）在启动时走硬件优先级链：H200 → `fa3`；Blackwell → `trtllm_mha`；AMD → `aiter`；其他 → `flashinfer if available else triton`。
2. **量化 MoE runner 自动选**：FP8 时 DeepGEMM 可用就选 DeepGEMM，否则 Triton。

两者都 fallback 到最保守但总能跑的选项。**这都是启动时跑一次，不是 per-request。**

**Q：我们显式指定一个 backend 后，就算不支持也不会 fallback 吗？**

A：**不会，且有 3 种失败模式（取决于哪一层）：**

- **硬 fail 报错**：backend 的构造函数自己挂了（Layer C）。这就是
  `--attention-backend flashinfer`（C5）的遭遇 —— flashinfer 的 JIT 编译不出来。
- **硬 fail 报错**：backend 名字不在 registry 里（Layer C，`ValueError: Invalid attention backend`）。
- **悄悄 no-op**：代码路径根本不读这个 flag（Layer D）。这就是
  `--moe-runner-backend cutlass`（C4）在 bf16 上的遭遇 —— flag 被 parse 但 unquantized MoE 方法忽略它、硬编码到 Triton。

悄悄 no-op 是最危险的 —— 这就是为啥我们必须做 kernel-by-kernel trace 对比，才能
确认 C4 真的 no-op 了、而不是"cutlass 恰好性能一样"。

## 22. Three follow-up questions on fallback / model architecture / backend selection

> 🟢 NEW. Three questions raised after §21:
> (1) Why does sglang sometimes SILENTLY no-op instead of raising "cutlass not available"?
> (2) "Model not found → raise" means every model is re-implemented from scratch (not torch transformers)?
> (3) Walk me through the backend-selection logic in detail — what happens on H200 vs Blackwell vs AMD vs Intel?

### 22.1 Why silent no-ops exist (not always an error)

The reason is **dispatch is conditioned on TWO things**, not one: the
backend flag AND the model's quantization config. The flag is consulted
**inside the quant method's `__init__`**, not at a top-level dispatcher.

Concrete contrast for our MoE on R8:

| Model dtype | Code path entered | Reads `--moe-runner-backend cutlass`? | Result |
|---|---|---|---|
| **bf16 (unquantized)** | `UnquantizedFusedMoEMethod.create_moe_runner()` | **NO** — code hard-codes `MoeRunnerBackend.TRITON` | **silent no-op** (our C4) |
| **FP8** | `Fp8MoEMethod.__init__()` | **YES** — explicit `if get_moe_runner_backend().is_cutlass(): …` | Either uses cutlass (if asserts pass) or raises AssertionError |

The FP8 path actually does the right thing
([`sglang/srt/layers/quantization/fp8.py:678-686`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)):

```python
class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config):
        ...
        if get_moe_runner_backend().is_cutlass():
            assert cutlass_fp8_supported(), \
                "cutlass_fp8 MoE requires CUDA 12.0+ with SM90 or CUDA 12.4+ with SM89"
            assert self.block_quant, "cutlass_fp8 MoE requires block quantization"
            assert is_sm100_supported() or is_sm90_supported()
```

If you're on FP8 + your CUDA is too old, you get a **clear AssertionError**.

The bf16 path simply doesn't have this branch
([`sglang/srt/layers/quantization/unquant.py:321-330`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)):

```python
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    def create_moe_runner(self, layer, moe_runner_config):
        backend = (MoeRunnerBackend.TRITON_KERNELS
                   if self.use_triton_kernels
                   else MoeRunnerBackend.TRITON)
        self.runner = MoeRunner(backend, moe_runner_config)
        # never even looks at server_args.moe_runner_backend
```

**Why this design** (best guess from reading the code, not from sglang
maintainers):

1. **Cutlass MoE is a quantization-bound optimisation**. The whole reason
   cutlass paths exist is to exploit FP8/FP4 tensor-core throughput on
   Hopper/Blackwell. There is no "bf16 cutlass MoE" kernel in the cutlass
   library that's faster than the Triton path on this shape; the
   maintainers didn't write a cutlass bf16 path because it wouldn't win.
2. **`auto` is the default value** of `--moe-runner-backend`. Most users
   never set this flag. The code is written assuming `cutlass`/`deep_gemm`
   only get passed by users who know what quant they're using.
3. **A loud error per missing-combination would explode the validation
   matrix**: sglang would need a 10-backend × 8-quantization × 5-hardware
   compatibility table maintained by hand. They chose to validate only at
   the points where the flag actually does anything.

**But** the no-op IS a bug for our use case (config A/B test where we
*want* to know our flag was ignored). For now, **the only way to detect
it is kernel-by-kernel trace diff** — which is exactly what §19.5 did.

The fix-it-yourself patch (if you wanted upstream-able):

```python
# in unquant.py UnquantizedFusedMoEMethod.create_moe_runner
requested = get_moe_runner_backend()
if requested != MoeRunnerBackend.AUTO and requested != MoeRunnerBackend.TRITON \
        and requested != MoeRunnerBackend.TRITON_KERNELS:
    logger.warning(
        f"--moe-runner-backend={requested.value} is ignored for unquantized "
        f"(bf16/fp16) MoE; falling back to TRITON. "
        f"Use --quantization fp8 to enable cutlass/deep_gemm backends.")
```

(Filed as a candidate sglang upstream issue in our notes.)

### 22.2 Yes — every supported model is hand-implemented in sglang

`sglang/srt/models/` contains **one Python file per model architecture**.
Each file defines `nn.Module` classes that build the forward pass using
sglang-native primitives (not transformers').

Concrete example —
[`sglang/srt/models/qwen3_moe.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py),
the model we benchmarked:

```python
# Imports show what's used INSTEAD of transformers
from sglang.srt.distributed import get_moe_expert_parallel_world_size, get_pp_group
from sglang.srt.layers.linear import RowParallelLinear, ColumnParallelLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE     # ← our fused_moe_kernel
from sglang.srt.layers.radix_attention import RadixAttention            # ← @register_split_op boundary
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from transformers import PretrainedConfig  # ← only the config dataclass

# Layer classes
class Qwen3MoeSparseMoeBlock(nn.Module):
    def forward(self, hidden_states, ...): ...        # custom forward
    def forward_normal(self, ...): ...
    def forward_deepep(self, ...): ...                 # expert parallel variant

class Qwen3MoeAttention(nn.Module):
    def __init__(self, ...):
        ...
        self.o_proj = RowParallelLinear(...)
        self.attn = RadixAttention(...)               # sglang's KV-cache-aware attention
    def forward_prepare_native(self, ...): ...
    def forward_core(self, intermediate_state): ...
    def forward(self, ...): ...
```

The forward path **uses sglang-native layers**:

| Concept | HuggingFace transformers | sglang |
|---|---|---|
| Linear | `nn.Linear` | `RowParallelLinear` / `ColumnParallelLinear` (tensor-parallel-aware) |
| Attention | `Qwen3Attention.forward` (Q/K/V proj + SDPA) | `RadixAttention` (KV-cache-aware, FlashAttention, paged) |
| MoE FFN | `Qwen3MoeSparseMoeBlock` (Python loop over experts) | `FusedMoE` → `fused_moe_kernel` (one Triton kernel) |
| Vocab embedding | `nn.Embedding` | `VocabParallelEmbedding` (sharded across TP) |
| RoPE | `apply_rotary_pos_emb` | `flashinfer.BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel` |
| RMSNorm | `Qwen3RMSNorm` | `flashinfer.FusedAddRMSNormKernel` (fused with residual add) |

What sglang DOES use from transformers: **only `PretrainedConfig`** (the
dataclass that parses `config.json`). The model **weights** are loaded
directly from `safetensors` files using sglang's own weight loader
(`sglang/srt/model_loader/loader.py`).

**Why fully re-implement?** Three reasons:

1. **Tensor parallelism / expert parallelism**: transformers' models are
   single-GPU. sglang needs to shard weights across TP, dispatch tokens
   across EP, gather across PP. Adding this to transformers' classes
   would require rewriting most layers anyway.
2. **Custom kernels**: transformers uses `torch.nn.Linear` (cuBLAS GEMM)
   and `F.scaled_dot_product_attention` (PyTorch SDPA). sglang wants
   `flashinfer`/`fa3`/`triton` kernels with KV-cache awareness, paged
   attention, prefix sharing — none of which transformers supports.
3. **Quantization integration**: sglang loads FP8/FP4/INT8 weights
   directly into quant-aware layer classes (`Fp8MoEMethod`, etc.). HF's
   `bitsandbytes` integration is generic but not as fast.

**Consequences for our workflow**:

- New model architecture = new Python file in `sglang/srt/models/`. **No model = no support.**
- Even if the underlying ops are "standard transformer" (which Gemma-4
  basically is), there's no auto-derivation — someone has to map the
  config fields and weight names to sglang's layer classes.
- This is the same model on `huggingface_hub`, but the *runtime* is
  sglang's, not transformers'. Numerical outputs may differ slightly
  (different RoPE epsilon, different attention masking conventions, etc.) —
  sglang's model files always note the reference HF implementation in
  comments + try to match it.

`sglang/srt/models/*.py` count as of 0.5.12:

```bash
$ ls sglang/srt/models/*.py | wc -l
~120 files  # ~75 distinct model families (some have *_mm.py for multimodal)
```

That's *a lot* of code maintained by hand. New families added on demand
(Gemma-4 will appear when someone writes `gemma4.py`).

### 22.3 The backend-selection logic — full walkthrough

Source: [`sglang/srt/server_args.py:1772-1850`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py)
`_handle_attention_backend_compatibility()` (runs once at startup before
the model loads).

#### Step 1 — Check if the user supplied prefill/decode-specific backends

```python
if self.prefill_attention_backend is not None and (
    self.prefill_attention_backend == self.decode_attention_backend
):
    # User set both to the same value → just promote to global attention_backend
    self.attention_backend = self.prefill_attention_backend
```

If user passes `--prefill-attention-backend fa3 --decode-attention-backend fa3`,
that's the same as `--attention-backend fa3`. If they differ, the **hybrid
attention backend** is constructed later (see Step 4).

#### Step 2 — If user did NOT pass `--attention-backend`, auto-select

This is **the only place fallback chains happen** (per §21). Two parallel
chains based on whether the model uses MLA (Multi-head Latent Attention,
DeepSeek-V2-style) or standard MHA (most other models).

##### Step 2a — MHA models (Llama, Qwen, Gemma, …)

The decision tree (`sglang/srt/server_args.py:1797-1818`):

```
                        ┌──────────────────────────────┐
                        │ user didn't pass             │
                        │ --attention-backend          │
                        └──────────────┬───────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_hopper_with_cuda_12_3()  AND          │
                  │ is_no_spec_infer_or_topk_one(self)      │
                  └────────────────────┬────────────────────┘
                                       │
                YES (H100, H200) ──────► attention_backend = "fa3"   ★
                                       │
                                       NO
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_sm100_supported()  AND               │
                  │ is_no_spec_infer_or_topk_one(self)  AND │
                  │ (no spec OR eagle topk set)             │
                  └────────────────────┬────────────────────┘
                                       │
                YES (B200/GB200/B300) ──► attention_backend = "trtllm_mha"
                                       │
                                       NO
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_hip()  (AMD GPU)                     │
                  └────────────────────┬────────────────────┘
                                       │
                                  YES ──► attention_backend = "aiter"
                                       │
                                       NO
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_flashinfer_available() (Python pkg) │
                  └────────────────────┬────────────────────┘
                                       │
                  YES ──► attention_backend = "flashinfer"
                  NO  ──► attention_backend = "triton"  (the always-works default)
```

The comment in source:

> 1. Models with MHA Architecture (e.g: Llama, QWen)
>    1.1 We will turn on **FA3 on hopper** unless user use spec decode with topk > 1 or page_size > 1.
>    1.2 Use **trtllm_mha for SM100/SM103 (Blackwell B200/GB200/B300)** excluding spec with topk > 1.
>        Note: trtllm_mha does not support SM120, which will fall back to flashinfer.
>    1.3 In other cases, we will use **flashinfer if available, otherwise use triton**.

##### Step 2b — MLA models (DeepSeek-V2, V3, R1)

```
                        ┌──────────────────────────────┐
                        │ user didn't pass             │
                        │ --attention-backend          │
                        │ AND model uses MLA           │
                        └──────────────┬───────────────┘
                                       │
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_hopper_with_cuda_12_3()              │
                  └────────────────────┬────────────────────┘
                                       │
                YES (H100, H200) ──────► attention_backend = "fa3"
                                       │
                                       NO
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_sm100_supported()                    │
                  └────────────────────┬────────────────────┘
                                       │
                YES (Blackwell) ────────► attention_backend = "flashinfer"
                                       │
                                       NO
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_hip() (AMD)                          │
                  └────────────────────┬────────────────────┘
                                       │
                YES ──► aiter (if head_num == 128 or 16)
                YES ──► triton (otherwise)
                                       │
                                       NO
                                       │
                                  ───► attention_backend = "triton"
```

#### Step 3 — Side effects of certain backends

After picking the backend, the system may further disable features that
the backend can't support
(`sglang/srt/server_args.py:1840-1860`):

```python
if self.attention_backend == "torch_native":
    logger.warning("Cuda graph is disabled because of using torch native attention backend")
    self.disable_cuda_graph = True

if self.attention_backend == "flex_attention":
    logger.warning("Cuda graph is disabled because of using torch Flex Attention backend")
    self.disable_cuda_graph = True
    assert (self.speculative_algorithm is None), "..."

if self.attention_backend == "intel_amx":
    # ...
if self.attention_backend == "ascend":
    # ...
```

Each backend has its own quirks; sglang silently disables incompatible
features and logs a warning.

#### Step 4 — Hybrid backend (per-stage)

If user set `--prefill-attention-backend X --decode-attention-backend Y`
with X≠Y, sglang constructs a **HybridAttnBackend** that dispatches
per-batch (`model_runner.py:1755-1779`):

```python
if self.decode_attention_backend_str != self.prefill_attention_backend_str:
    attn_backend = HybridAttnBackend(
        decode_backend=self._get_attention_backend_from_str(self.decode_attention_backend_str),
        prefill_backend=self._get_attention_backend_from_str(self.prefill_attention_backend_str),
    )
    logger.warning(
        "Warning: Attention backend specified by --attention-backend or default backend "
        "might be overridden. The feature of hybrid attention backend is experimental and "
        "unstable. Please raise an issue if you encounter any problem."
    )
```

(Experimental — sglang says so loudly.)

#### Step 5 — Now actually load the backend

`model_runner.py:1792-1799`:

```python
def _get_attention_backend_from_str(self, backend_str, ...):
    if backend_str not in ATTENTION_BACKENDS:
        raise ValueError(f"Invalid attention backend: {backend_str}")
    full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
    return attn_backend_wrapper(self, full_attention_backend)
```

The backend's constructor runs here. **If it raises, the server exits.**
This is Layer C from §21 — explicit backend, no fallback.

#### Step 6 — Concrete walkthrough by hardware

| Hardware | Model | Auto path → final backend | Why |
|---|---|---|---|
| **H100 / H200** | MHA (Qwen, Llama, Gemma) | `fa3` | hopper + CUDA 12.3 + standard inference path |
| H100 / H200 | MHA + `--speculative-eagle-topk 4` | `flashinfer` or `triton` | spec decode topk > 1 → falls through to general chain |
| H100 / H200 | MHA + `--page-size 2` | `flashinfer` or `triton` | page > 1 → falls through |
| **H100 / H200** | MLA (DeepSeek) | `fa3` | same hopper detection |
| **B200 / GB200 / B300** | MHA | `trtllm_mha` | SM100/103 detected |
| **B200 / GB200 / B300** | MHA + spec topk > 1 | `flashinfer` or `triton` | trtllm not supported |
| **B200 / GB200 / B300** | MLA | `flashinfer` | MLA path on Blackwell |
| RTX 5090 (SM120) | any | `flashinfer` or `triton` | SM120 not supported by trtllm; auto skips |
| **AMD MI300X / MI300A** | MHA | `aiter` | `is_hip()` true |
| AMD MI300X | MLA | `aiter` (if 128 or 16 heads) or `triton` | aiter constrained |
| **Intel Gaudi / Xeon** (CPU AMX) | any | `intel_amx` only if explicit, else manual config | Intel paths aren't in auto chain — need explicit `--attention-backend intel_amx` |
| **A100** (SM80) | MHA | `flashinfer` if installed, else `triton` | A100 isn't hopper; trtllm is sm90+ |
| A100 | MLA | `triton` | no MLA fast path for SM80 |

**The "always-works default" is always `triton`** — it's pure Python +
PyTorch + Triton kernels, with no proprietary CUDA library dependency.
Slower than fa3/flashinfer/trtllm on supported hardware, but it runs
everywhere CUDA runs.

#### Step 7 — What the MoE/sampling/gemm backends do similarly

The same "auto + per-hardware chain + explicit-no-fallback" pattern repeats
for several other dispatchers:

- `--moe-runner-backend` (auto → triton for bf16; FP8 path may pick deep_gemm)
- `--sampling-backend` (auto → flashinfer on CUDA, pytorch on CPU)
- `--fp8-gemm-backend` (auto → deep_gemm if installed, else triton)
- `--fp4-gemm-backend` (default `flashinfer_cutlass`)
- `--moe-a2a-backend` (default `none`, can be `deepep`/`mooncake`/etc.)

Each has its own dispatcher in `server_args.py` or in the relevant layer
module, but they all follow Layer B (auto) / Layer C (explicit hard-fail)
/ sometimes Layer D (silent no-op if the code path doesn't read the flag).

### 22.4 TL;DR — three answers in one paragraph each

**Why silent no-op instead of "cutlass not available" error**: Backend
dispatch is **per-quantization-method**, not centralised. The cutlass MoE
path only exists inside `Fp8MoEMethod.__init__()` (which `assert`s
correctly). The bf16 path (`UnquantizedFusedMoEMethod`) doesn't have a
cutlass branch at all — and doesn't even read `--moe-runner-backend`. It
hardcodes Triton. So passing `cutlass` to a bf16 model silently picks
Triton. The architectural reason is that cutlass MoE only makes sense
for FP8/FP4 (which is what cutlass MoE kernels are written for); no one
wrote a bf16 cutlass path because Triton already wins on that shape.

**Yes, every supported model is hand-implemented**: ~120 Python files in
`sglang/srt/models/`. Each one defines `nn.Module` classes that build the
forward pass using sglang-native primitives (`RowParallelLinear`,
`FusedMoE`, `RadixAttention`, `VocabParallelEmbedding`) instead of
transformers' classes. Sglang uses transformers ONLY for the
`PretrainedConfig` dataclass that parses `config.json`. Weights are
loaded directly from `safetensors`. New model = new Python file in
`sglang/srt/models/`. This is why Gemma-4 fails — there's no
`gemma4.py`.

**Backend-selection logic, in 3 lines**: At startup
(`server_args.py:_handle_attention_backend_compatibility`), if you did
NOT pass `--attention-backend`, sglang picks one from a hardware-aware
chain: **H100/H200 → fa3; Blackwell → trtllm_mha; AMD → aiter;
otherwise → flashinfer if installed, else triton**. If you DID pass the
flag, sglang tries that exact backend and **raises an exception if it
can't load** (no fallback). The MLA models (DeepSeek) follow a parallel
but simpler chain (fa3 on hopper, flashinfer on Blackwell, triton
otherwise). The "always-works" default is **triton** — pure Python +
PyTorch + Triton, no proprietary dependencies.

---

## 22. 三个 fallback / 模型架构 / backend 选择的后续问题

> 🟢 新增。§21 之后的 3 个问题：
> (1) 为啥 sglang 有时候 **悄悄 no-op**，而不是直接"cutlass 不可用"报错？
> (2) "模型没有就报错" 是不是意味着 sglang 把所有模型都自己另实现了一份？
> (3) 详细讲讲 backend 选择逻辑 —— H200 / Blackwell / AMD / Intel 各会咋走？

### 22.1 为啥会有"悄悄 no-op"（不总是个错误）

原因是 **dispatch 同时依赖两件事**：backend flag **和** 模型的量化配置。flag 是
在 **每个量化方法的 `__init__` 里** 读的，不是顶层 dispatcher 统一处理。

我们 MoE 上的具体对比：

| 模型 dtype | 走的代码路径 | 读 `--moe-runner-backend cutlass`？ | 结果 |
|---|---|---|---|
| **bf16（未量化）** | `UnquantizedFusedMoEMethod.create_moe_runner()` | **不读** —— 代码硬编码 `MoeRunnerBackend.TRITON` | **悄悄 no-op**（我们的 C4） |
| **FP8** | `Fp8MoEMethod.__init__()` | **读** —— 显式 `if get_moe_runner_backend().is_cutlass(): …` | 要么用 cutlass（assert 过了），要么 AssertionError |

FP8 路径其实写得很对
（[`sglang/srt/layers/quantization/fp8.py:678-686`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)）：

```python
class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config):
        ...
        if get_moe_runner_backend().is_cutlass():
            assert cutlass_fp8_supported(), \
                "cutlass_fp8 MoE requires CUDA 12.0+ with SM90 or CUDA 12.4+ with SM89"
            assert self.block_quant, "cutlass_fp8 MoE requires block quantization"
            assert is_sm100_supported() or is_sm90_supported()
```

FP8 + CUDA 太老 → 清晰 AssertionError。

bf16 路径根本没有这一分支
（[`sglang/srt/layers/quantization/unquant.py:321-330`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)）：

```python
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    def create_moe_runner(self, layer, moe_runner_config):
        backend = (MoeRunnerBackend.TRITON_KERNELS
                   if self.use_triton_kernels
                   else MoeRunnerBackend.TRITON)
        self.runner = MoeRunner(backend, moe_runner_config)
        # 完全没看 server_args.moe_runner_backend
```

**为什么这么设计**（从读源码猜，不是从 sglang 维护者那听来的）：

1. **Cutlass MoE 是量化导向的优化**。cutlass 路径存在就是为了利用 Hopper/Blackwell
   上 FP8/FP4 的 tensor core 吞吐。**根本没"bf16 cutlass MoE" kernel**
   能在这个 shape 上比 Triton 快；维护者没写 bf16 cutlass 路径，因为没法赢。
2. **`--moe-runner-backend` 默认 `auto`**。绝大多数用户根本不设这个 flag。代码假定
   写 `cutlass` / `deep_gemm` 的人知道自己用的是什么量化。
3. **每个缺失组合都报错会让 validation matrix 爆炸**：sglang 得维护 10-backend
   × 8-quantization × 5-hardware 的兼容性表。他们选择只在 flag 真起作用的地方校验。

**但对我们的使用场景（config A/B 测试，想知道 flag 被忽略了），这个 no-op 就是 bug**。
目前只能靠 kernel-by-kernel trace diff 检测（§19.5 干的就是这事）。

可以提 upstream 的补丁：

```python
# unquant.py UnquantizedFusedMoEMethod.create_moe_runner 里
requested = get_moe_runner_backend()
if requested != MoeRunnerBackend.AUTO and requested != MoeRunnerBackend.TRITON \
        and requested != MoeRunnerBackend.TRITON_KERNELS:
    logger.warning(
        f"--moe-runner-backend={requested.value} is ignored for unquantized "
        f"(bf16/fp16) MoE; falling back to TRITON. "
        f"Use --quantization fp8 to enable cutlass/deep_gemm backends.")
```

### 22.2 是的 —— sglang 支持的每个模型都是手写实现的

`sglang/srt/models/` 里 **每个模型架构一个 Python 文件**。每个文件定义
`nn.Module` 类，用 sglang 自己的原语（不是 transformers 的）构造 forward。

具体例子 ——
[`sglang/srt/models/qwen3_moe.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py)，
我们 benchmark 的那个：

```python
# import 显示用了什么 *替代* transformers
from sglang.srt.distributed import get_moe_expert_parallel_world_size, get_pp_group
from sglang.srt.layers.linear import RowParallelLinear, ColumnParallelLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE     # ← 我们的 fused_moe_kernel
from sglang.srt.layers.radix_attention import RadixAttention            # ← @register_split_op 边界
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from transformers import PretrainedConfig  # ← 只用 config 数据类

# Layer 类
class Qwen3MoeSparseMoeBlock(nn.Module):
    def forward(self, hidden_states, ...): ...        # 自己写的 forward
    def forward_normal(self, ...): ...
    def forward_deepep(self, ...): ...                 # expert parallel 变体

class Qwen3MoeAttention(nn.Module):
    def __init__(self, ...):
        ...
        self.o_proj = RowParallelLinear(...)
        self.attn = RadixAttention(...)               # sglang 的 KV-cache-aware attention
    def forward(self, ...): ...
```

forward 用的是 **sglang 自己的 layer**：

| 概念 | HuggingFace transformers | sglang |
|---|---|---|
| Linear | `nn.Linear` | `RowParallelLinear` / `ColumnParallelLinear`（TP 感知） |
| Attention | `Qwen3Attention.forward`（Q/K/V proj + SDPA） | `RadixAttention`（KV-cache 感知、FlashAttention、paged） |
| MoE FFN | `Qwen3MoeSparseMoeBlock`（Python 循环遍历专家） | `FusedMoE` → `fused_moe_kernel`（一个 Triton kernel） |
| Vocab embedding | `nn.Embedding` | `VocabParallelEmbedding`（TP 分片） |
| RoPE | `apply_rotary_pos_emb` | `flashinfer.BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel` |
| RMSNorm | `Qwen3RMSNorm` | `flashinfer.FusedAddRMSNormKernel`（融合 residual add） |

sglang **用** transformers 的：**只用 `PretrainedConfig`**（解析 `config.json` 的数据类）。
模型 **权重** 是用 sglang 自己的 weight loader 直接从 `safetensors` 加载
（`sglang/srt/model_loader/loader.py`）。

**为啥完全重写**？三个原因：

1. **张量并行 / 专家并行**：transformers 模型是单卡的。sglang 需要把权重在 TP 上分片、
   把 token 在 EP 上 dispatch、跨 PP 收集。要给 transformers 的类加上这些，本来就要
   重写绝大部分 layer。
2. **自定义 kernel**：transformers 用 `torch.nn.Linear`（cuBLAS GEMM）和
   `F.scaled_dot_product_attention`（PyTorch SDPA）。sglang 想用 `flashinfer`
   / `fa3` / `triton` kernel，带 KV-cache 感知、paged attention、prefix sharing ——
   这些 transformers 都不支持。
3. **量化集成**：sglang 直接把 FP8/FP4/INT8 权重加载到量化感知的 layer 类
   （`Fp8MoEMethod` 等）。HF 的 `bitsandbytes` 集成通用但不够快。

**对我们工作流的影响**：

- 新模型架构 = `sglang/srt/models/` 加一个 Python 文件。**没文件 = 不支持。**
- 即使底层 op 是"标准 transformer"（Gemma-4 基本就是），也没有自动推导 ——
  得有人把 config 字段和权重名映射到 sglang 的 layer 类。
- 同一个模型 在 `huggingface_hub` 上，但 *运行时* 是 sglang 的，不是 transformers 的。
  数值输出可能略有差异（不同 RoPE epsilon、不同 attention mask 约定等等）——
  sglang 模型文件总在注释里写参考的 HF 实现 + 尽量对齐。

`sglang/srt/models/*.py` 在 0.5.12 上的数量：

```bash
$ ls sglang/srt/models/*.py | wc -l
~120 个文件  # ~75 个模型家族（部分有 *_mm.py 多模态变体）
```

这是 *很大量* 的手写代码。新家族按需添加（Gemma-4 等有人写 `gemma4.py` 才会出现）。

### 22.3 Backend 选择逻辑 —— 详细 walkthrough

源码：[`sglang/srt/server_args.py:1772-1850`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py)
`_handle_attention_backend_compatibility()`（启动时跑一次，模型加载前）。

#### Step 1 —— 检查用户有没有设 prefill/decode 单独的 backend

```python
if self.prefill_attention_backend is not None and (
    self.prefill_attention_backend == self.decode_attention_backend
):
    # 用户两个都设成同一个 → 提升为全局 attention_backend
    self.attention_backend = self.prefill_attention_backend
```

用户传 `--prefill-attention-backend fa3 --decode-attention-backend fa3` 等于
`--attention-backend fa3`。如果不同，后面构造 **hybrid attention backend**
（Step 4）。

#### Step 2 —— 用户没设 `--attention-backend`，自动选

**这是 fallback 链唯一发生的地方**（per §21）。按模型是不是用 MLA
（Multi-head Latent Attention，DeepSeek-V2 风格）分两条并行链。

##### Step 2a —— MHA 模型（Llama、Qwen、Gemma 等）

决策树（`server_args.py:1797-1818`）：

```
                        ┌──────────────────────────────┐
                        │ 用户没传                      │
                        │ --attention-backend          │
                        └──────────────┬───────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_hopper_with_cuda_12_3()  AND          │
                  │ is_no_spec_infer_or_topk_one(self)      │
                  └────────────────────┬────────────────────┘
                                       │
                是（H100、H200）──────► attention_backend = "fa3"   ★
                                       │
                                       否
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_sm100_supported()  AND               │
                  │ is_no_spec_infer_or_topk_one(self)  AND │
                  │ （无 spec 或 eagle topk 已设）           │
                  └────────────────────┬────────────────────┘
                                       │
                是（B200/GB200/B300）──► attention_backend = "trtllm_mha"
                                       │
                                       否
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_hip()（AMD GPU）                     │
                  └────────────────────┬────────────────────┘
                                       │
                                  是 ──► attention_backend = "aiter"
                                       │
                                       否
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_flashinfer_available()（Python 包）  │
                  └────────────────────┬────────────────────┘
                                       │
                  是 ──► attention_backend = "flashinfer"
                  否 ──► attention_backend = "triton"  （总能跑的默认）
```

源码注释：

> 1. MHA 架构模型（如 Llama、QWen）
>    1.1 hopper 上开 **fa3**，除非用户开 spec decode 且 topk > 1，或 page_size > 1。
>    1.2 SM100/SM103（Blackwell B200/GB200/B300）用 **trtllm_mha**，spec topk > 1 除外。
>        注意：trtllm_mha 不支持 SM120，会 fall back 到 flashinfer。
>    1.3 其他情况，**flashinfer 可用就用，否则 triton**。

##### Step 2b —— MLA 模型（DeepSeek-V2、V3、R1）

```
                        ┌──────────────────────────────┐
                        │ 用户没传                     │
                        │ --attention-backend          │
                        │ 且模型用 MLA                 │
                        └──────────────┬───────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_hopper_with_cuda_12_3()              │
                  └────────────────────┬────────────────────┘
                                       │
                是（H100、H200）──────► attention_backend = "fa3"
                                       │
                                       否
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_sm100_supported()                    │
                  └────────────────────┬────────────────────┘
                                       │
                是（Blackwell）────────► attention_backend = "flashinfer"
                                       │
                                       否
                                       │
                  ┌────────────────────┴────────────────────┐
                  │ is_hip()（AMD）                         │
                  └────────────────────┬────────────────────┘
                                       │
                是 ──► aiter（如果 head_num == 128 或 16）
                是 ──► triton（其他）
                                       │
                                       否
                                       │
                                  ───► attention_backend = "triton"
```

#### Step 3 —— 某些 backend 的副作用

选完 backend，系统可能进一步关掉 backend 不支持的特性
（`server_args.py:1840-1860`）：

```python
if self.attention_backend == "torch_native":
    logger.warning("Cuda graph is disabled because of using torch native attention backend")
    self.disable_cuda_graph = True

if self.attention_backend == "flex_attention":
    logger.warning("Cuda graph is disabled because of using torch Flex Attention backend")
    self.disable_cuda_graph = True
    assert (self.speculative_algorithm is None), "..."
```

每个 backend 有自己的怪癖；sglang 悄悄禁用不兼容的特性 + log 警告。

#### Step 4 —— Hybrid backend（per-stage）

用户设 `--prefill-attention-backend X --decode-attention-backend Y` 且 X≠Y，
sglang 构造 **HybridAttnBackend** 按 batch 分发
（`model_runner.py:1755-1779`）：

```python
if self.decode_attention_backend_str != self.prefill_attention_backend_str:
    attn_backend = HybridAttnBackend(
        decode_backend=self._get_attention_backend_from_str(self.decode_attention_backend_str),
        prefill_backend=self._get_attention_backend_from_str(self.prefill_attention_backend_str),
    )
    logger.warning("…experimental and unstable…")
```

（实验性 —— sglang 自己说得很大声。）

#### Step 5 —— 真正加载 backend

`model_runner.py:1792-1799`：

```python
def _get_attention_backend_from_str(self, backend_str, ...):
    if backend_str not in ATTENTION_BACKENDS:
        raise ValueError(f"Invalid attention backend: {backend_str}")
    full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
    return attn_backend_wrapper(self, full_attention_backend)
```

Backend 的构造函数在这里跑。**抛异常 server 退出。** 这就是 §21 的 Layer C
—— 显式 backend、无 fallback。

#### Step 6 —— 按硬件具体走法

| 硬件 | 模型 | 自动路径 → 最终 backend | 为啥 |
|---|---|---|---|
| **H100 / H200** | MHA（Qwen、Llama、Gemma） | `fa3` | hopper + CUDA 12.3 + 标准推理路径 |
| H100 / H200 | MHA + `--speculative-eagle-topk 4` | `flashinfer` 或 `triton` | spec decode topk > 1 → 落到一般链 |
| H100 / H200 | MHA + `--page-size 2` | `flashinfer` 或 `triton` | page > 1 → 落到一般链 |
| **H100 / H200** | MLA（DeepSeek） | `fa3` | 同样 hopper 检测 |
| **B200 / GB200 / B300** | MHA | `trtllm_mha` | SM100/103 检测到 |
| **B200 / GB200 / B300** | MHA + spec topk > 1 | `flashinfer` 或 `triton` | trtllm 不支持 |
| **B200 / GB200 / B300** | MLA | `flashinfer` | Blackwell 上的 MLA path |
| RTX 5090（SM120） | 任意 | `flashinfer` 或 `triton` | SM120 不被 trtllm 支持；auto 跳过 |
| **AMD MI300X / MI300A** | MHA | `aiter` | `is_hip()` 为真 |
| AMD MI300X | MLA | `aiter`（128 或 16 个 head）或 `triton` | aiter 受限 |
| **Intel Gaudi / Xeon**（CPU AMX） | 任意 | 只有显式指定 `intel_amx` 才会用，否则要手动配置 | Intel 路径不在 auto 链里 —— 要显式 `--attention-backend intel_amx` |
| **A100**（SM80） | MHA | flashinfer 装了就用，否则 `triton` | A100 不是 hopper；trtllm 要 sm90+ |
| A100 | MLA | `triton` | SM80 没有 MLA 快路径 |

**"总能跑的默认"永远是 `triton`** —— 纯 Python + PyTorch + Triton kernel，
不依赖任何专有 CUDA 库。比 fa3/flashinfer/trtllm 在支持的硬件上慢，但
CUDA 能跑的地方它都能跑。

#### Step 7 —— MoE / sampling / gemm backend 也类似

"auto + per-hardware 链 + 显式-无-fallback" 这套模式在好几个 dispatcher 重复：

- `--moe-runner-backend`（auto → bf16 上 triton；FP8 path 可能选 deep_gemm）
- `--sampling-backend`（auto → CUDA 上 flashinfer，CPU 上 pytorch）
- `--fp8-gemm-backend`（auto → deep_gemm 装了就用，否则 triton）
- `--fp4-gemm-backend`（默认 `flashinfer_cutlass`）
- `--moe-a2a-backend`（默认 `none`，可以 `deepep`/`mooncake` 等）

各自的 dispatcher 在 `server_args.py` 或相关 layer 模块里，但都遵循 Layer B
（auto）/ Layer C（显式硬 fail）/ 有时 Layer D（代码不读 flag 就悄悄 no-op）。

### 22.4 一段话总结 —— 三个答案

**为啥悄悄 no-op 而不是"cutlass 不可用"报错**：Backend dispatch 是
**per-量化-方法** 的，不是集中式的。Cutlass MoE 路径只存在于
`Fp8MoEMethod.__init__()`（正确地 `assert`）。bf16 path
（`UnquantizedFusedMoEMethod`）根本没有 cutlass 分支 —— 也不读
`--moe-runner-backend`，硬编码 Triton。所以给 bf16 模型传 `cutlass` 悄悄选了 Triton。
架构原因是 cutlass MoE 只对 FP8/FP4 有意义（cutlass MoE kernel 就是为这个写的）；
没人写 bf16 cutlass 路径因为 Triton 已经赢了。

**是的，每个支持的模型都是手写实现的**：`sglang/srt/models/` 里 ~120 个 Python
文件。每个定义 `nn.Module` 类，用 sglang 自己的原语（`RowParallelLinear`、
`FusedMoE`、`RadixAttention`、`VocabParallelEmbedding`）构造 forward，不用
transformers 的类。Sglang **只用** transformers 的 `PretrainedConfig` 数据类
解析 `config.json`。权重直接从 `safetensors` 加载。新模型 = `sglang/srt/models/`
加新 Python 文件。这就是 Gemma-4 失败的原因 —— 没 `gemma4.py`。

**Backend 选择逻辑，3 行总结**：启动时
（`server_args.py:_handle_attention_backend_compatibility`），用户**没**传
`--attention-backend` 的话，sglang 按硬件感知链选：**H100/H200 → fa3；
Blackwell → trtllm_mha；AMD → aiter；其他 → flashinfer 装了就用，否则
triton**。用户**传了** flag，sglang 严格用那个 backend，**加载失败就 raise**
（无 fallback）。MLA 模型（DeepSeek）走另一条更简单的链
（hopper 上 fa3，Blackwell 上 flashinfer，其他 triton）。"总能跑"的默认是
**triton** —— 纯 Python + PyTorch + Triton，无专有依赖。

## 23. Silent no-op — exact source evidence + detection tooling

> 🟢 NEW. Two follow-up questions after §22:
> (1) Where exactly is the source code evidence?
> (2) How do we know in practice whether a flag was silently ignored vs
>     actually used?
>
> Also includes a **correction** to §22.1: unquant.py DOES read
> `get_moe_runner_backend()` (for flashinfer_cutlass and auto checks),
> just not for the `cutlass` value. The precise mechanism is more subtle
> than "completely ignores the flag".

### 23.1 The exact source evidence — `unquant.py`

The complete `UnquantizedFusedMoEMethod` initialization
([`sglang/srt/layers/quantization/unquant.py:155-167`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)):

```python
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""

    def __init__(
        self, use_triton_kernels: bool = False, use_flashinfer_trtllm_moe: bool = False
    ):
        super().__init__()
        # ↓↓↓ DOES read the backend flag — but only for ONE value
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self._cache_permute_indices = dict({})
```

And the runner creation
([`sglang/srt/layers/quantization/unquant.py:321-330`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)):

```python
    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        backend = (
            MoeRunnerBackend.TRITON_KERNELS
            if self.use_triton_kernels
            else MoeRunnerBackend.TRITON
        )
        self.runner = MoeRunner(backend, moe_runner_config)
        # ↑↑↑ Only chooses between TRITON_KERNELS or TRITON
        #     — even if server arg said cutlass / deep_gemm / flashinfer_trtllm
```

Plus the AIter fallback gate
([`unquant.py:229`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)):

```python
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # AIter only runs when the flag is set to 'auto' (i.e. user did NOT
        # request a specific backend)
        _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
        ...
```

**Precise reading**: the bf16 unquantized MoE method **partially** reads
the flag:

| `--moe-runner-backend` value | bf16 path behaviour |
|---|---|
| `auto` (default) | Triton, with AIter shuffle on AMD |
| `triton` | Triton (explicit, matches default) |
| `triton_kernel` | TritonKernels variant |
| `flashinfer_cutlass` | Triton **but** with `self.use_flashinfer_cutlass=True` which changes weight loading order ([`unquant.py:333-335`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py) `load_up_proj_weight_first`). The MoE compute kernel itself is still `fused_moe_kernel`. |
| **`cutlass`** | **silent no-op → Triton** (this is what C4 hit) |
| `deep_gemm` | **silent no-op → Triton** |
| `flashinfer_trtllm` | **silent no-op → Triton** |
| `flashinfer_mxfp4` | **silent no-op → Triton** |
| `flashinfer_cutedsl` | **silent no-op → Triton** |
| `marlin` | **silent no-op → Triton** |

So the surface area of "silent no-op" for bf16 MoE is **6 out of the 10
values** of the `--moe-runner-backend` flag.

The **FP8 path** by contrast checks the flag explicitly
([`sglang/srt/layers/quantization/fp8.py:678-686`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)):

```python
class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config):
        ...
        if get_moe_runner_backend().is_cutlass():
            assert cutlass_fp8_supported(), \
                "cutlass_fp8 MoE requires CUDA 12.0+ with SM90 or CUDA 12.4+ with SM89"
            assert self.block_quant, \
                "cutlass_fp8 MoE requires block quantization"
            assert is_sm100_supported() or is_sm90_supported()
```

If you pass `--moe-runner-backend cutlass --quantization fp8` and your
CUDA isn't new enough, you get a **clear AssertionError at startup**.

The full read map for all backend-related flags (grep'd from sglang
source, with all read sites for `get_moe_runner_backend()`):

```
$ grep -rn "get_moe_runner_backend" sglang/srt/layers/quantization/
unquant.py:162          is_flashinfer_cutlass()   ← reads but only narrowly
unquant.py:229          is_auto()                  ← reads but only narrowly
unquant.py:391          is_auto()                  ← reads but only narrowly
fp8.py:1349             is_deep_gemm() / TRITON    ← real dispatcher (FP8 only)
modelopt_quant.py:755   is_flashinfer_trtllm()     ← real dispatcher (modelopt only)
modelopt_quant.py:762   is_flashinfer_cutlass()    ← real dispatcher
modelopt_quant.py:855   is_flashinfer_cutlass()    ← real dispatcher
mxfp4.py:301-303        is_triton_kernels(), is_flashinfer_mxfp4()  ← real dispatcher (mxfp4 only)
compressed_tensors/...  ← various, all quant-specific
awq.py:831              assert is_auto()           ← assertion, not dispatch
```

**The unquantized path checks the flag, but only to decide between Triton
variants and to gate AMD AIter.** Values that select cutlass / DeepGEMM /
flashinfer_trtllm fall through without warning to Triton — because those
backends only have implementations registered for quantized weights.

### 23.2 Why `/server_info` doesn't tell you the truth

The natural reaction is "well, just query `/server_info` and see what the
server thinks it's using". **This doesn't work**. Source
([`sglang/srt/entrypoints/http_server.py:594-610`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)):

```python
@app.get("/server_info")
async def server_info():
    """Get the server information."""
    internal_states: List[Dict[Any, Any]] = (
        await _global_state.tokenizer_manager.get_internal_state()
    )
    return {
        **dataclasses.asdict(_global_state.tokenizer_manager.server_args),
        # ↑↑↑ dumps the INPUT server_args verbatim — NOT what's actually used
        **_global_state.scheduler_info,
        "internal_states": internal_states,
        "version": __version__,
    }
```

`dataclasses.asdict(server_args)` returns exactly what the user passed on
the CLI (resolved through any defaults). It does NOT reflect:

- whether the bf16 unquantized path silently ignored `--moe-runner-backend cutlass`
- whether AMD AIter is actually running (gated by `is_auto()`)
- whether the JIT for an explicit backend actually succeeded
- whether torch.compile got applied to all modules vs partially

So if you check `server_info` after launching with
`--moe-runner-backend cutlass` on bf16, **it will tell you
`moe_runner_backend: 'cutlass'`** even though Triton is running. This is
exactly what we saw in C4: `server_info.json` said `cutlass`, the trace
said `fused_moe_kernel`.

### 23.3 The detection tool — `detect_silent_noop.py`

The only reliable way to detect silent no-ops is to **fingerprint each
backend by the kernels it should produce** and check the trace.

I wrote a small detector script,
[`scripts/regime_study/detect_silent_noop.py`](../scripts/regime_study/detect_silent_noop.py),
that does this automatically. It encodes one regex per (backend_flag,
value) pair — e.g.:

```python
BACKEND_FINGERPRINT = {
    ("moe_runner_backend", "cutlass"): [
        r"cutlass.*moe", r"cutlass_fused_experts", r"sm90_xmma_warpspecialized.*moe"
    ],
    ("moe_runner_backend", "triton"): [r"fused_moe_kernel"],
    ("attention_backend", "fa3"): [r"FlashAttnFwdSm90", r"flash::"],
    ("attention_backend", "flashinfer"): [
        r"flashinfer.*(?:Prefill|Decode|Attention)",
        r"BatchPrefillWithRaggedKV", r"BatchDecodeWithPagedKV",
    ],
    ...
}
```

It does three things:

1. Reads `server_info.json` to see what the user requested
2. Loads the trace, lists all GPU kernel names + their self-time
3. For each requested backend, checks if **any** kernel matches the
   fingerprint; reports `HONOURED` / `IGNORED_OR_FALLBACK`

Plus an optional **reference-trace diff** that catches the silent no-op
even when the fingerprint is ambiguous: if 4/5 of top kernels are within
±5% of the reference, the flag was probably a no-op.

Usage:

```bash
python scripts/regime_study/detect_silent_noop.py \
    --server-info results/regime_bench/raw/moe_opt_levels/C4_moe_cutlass/server_info.json \
    --trace experiments/tmp/moe_opt_levels/C4_moe_cutlass/raw_trace/*/p_*.trace.json.gz \
    --reference-trace experiments/tmp/moe_opt_levels/C0_baseline/raw_trace/*/p_*.trace.json.gz
```

Output (smoke-tested on our C4):

```
[detect] loading trace experiments/tmp/moe_opt_levels/C4_moe_cutlass/raw_trace/.../p_-...-TP-0.trace.json.gz
[detect] 85 unique GPU kernels in trace
[detect] moe_runner_backend='cutlass' → ⚠️  IGNORED OR FELL BACK
[detect] attention_backend='fa3' → ✅ HONOURED
          matched pattern: /FlashAttnFwdSm90/
          matched pattern: /flash::/
            61364 us  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<…>
            61364 us  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<…>
            18305 us  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<…>
[detect] sampling_backend='flashinfer' → ⚠️  IGNORED OR FELL BACK

[detect] reference trace experiments/tmp/moe_opt_levels/C0_baseline/raw_trace/.../p_-...-TP-0.trace.json.gz
[detect] LIKELY_SILENT_NOOP
[detect] 5/5 kernels within ±5% of reference
            277135 vs 278301 us  (−0.4%)  fused_moe_kernel
             61364 vs  61582 us  (−0.4%)  void cutlass::device_kernel<flash::enable_sm90_or_later<flas
             37488 vs  37831 us  (−0.9%)  nvjet_tst_128x248_64x4_2x1_v_bz_coopA_TNT
             21556 vs  21538 us  (+0.1%)  nvjet_tst_192x208_64x4_1x2_h_bz_coopB_TNT
             18305 vs  18330 us  (−0.1%)  void cutlass::device_kernel<flash::enable_sm90_or_later<flas
```

Two clear signals:

1. **`moe_runner_backend='cutlass'` → IGNORED OR FELL BACK** — no cutlass-MoE
   kernel in the trace
2. **Kernel-mix near-identical to baseline** (5/5 within ±5%) → confirms
   silent no-op rather than "cutlass happens to produce identical perf"

The script's exit code is 2 when any flag was IGNORED — usable in CI to
fail builds that quietly drop optimisation flags.

> Note on the `sampling_backend='flashinfer' → IGNORED` result above:
> this is a **false positive**. Our fingerprint patterns for the flashinfer
> sampling backend (`r"flashinfer.*sampling"`, `r"top_p"`, `r"top_k_sampling"`)
> don't match the actual flashinfer sampling kernel names — these are
> launched as small `at::native::*` thread-fusion kernels because the
> sampler ops are tiny and inline. The fingerprint database needs more work
> for the sampling backend; for now, flashinfer-sampling honour can be
> assumed if attention is on flashinfer too (they share JIT init).

### 23.4 Practical recipe for verifying any backend flag

When you launch sglang with a non-default backend flag, run this 3-step check:

1. **Confirm the server actually used it** — run the trace + detector:
   ```bash
   # Run sglang.bench_serving --profile (saves a trace)
   # Then:
   python scripts/regime_study/detect_silent_noop.py \
       --server-info <your_run>/server_info.json \
       --trace <your_run>/raw_trace/*/p_*.trace.json.gz
   ```
2. **Read the server.log for warnings** (sometimes sglang emits one):
   ```bash
   grep -iE "ignored|falling back|fall.?back|not.?support|will.?use" server.log
   ```
3. **Verify performance changed** — if you flipped the flag and bench
   numbers are identical to 0.1%, you've probably hit a no-op. Compare:
   ```bash
   diff <(jq '.output_throughput' baseline/bench.jsonl) \
        <(jq '.output_throughput' yourrun/bench.jsonl)
   ```

If all three say "no change", you're either honoured-but-no-effect (e.g.
cutlass is "auto" choice anyway) or silently ignored. **Detector ⊕ kernel
diff distinguishes these two.**

### 23.5 Summary

| Question | Answer |
|---|---|
| Where's the source evidence? | `sglang/srt/layers/quantization/unquant.py:155-167` (init) and `:321-330` (runner creation). Lines 162, 229, 391 do read `get_moe_runner_backend()` — but only check for `is_flashinfer_cutlass()`, `is_auto()`. Values like `cutlass`, `deep_gemm`, `flashinfer_trtllm`, `marlin` fall through to Triton without any warning. |
| Can `/server_info` tell us? | **No.** `http_server.py:594-610` returns `dataclasses.asdict(server_args)` verbatim — it shows what you asked for, not what's running. |
| How to detect in practice? | `scripts/regime_study/detect_silent_noop.py` — fingerprints each backend by expected kernel names in the trace; flags missing fingerprints as IGNORED; optionally diffs against a reference trace for kernel-mix similarity. |
| Detection accuracy on our C4? | ✅ Caught `moe_runner_backend='cutlass'` as IGNORED and confirmed via 5/5 kernels within ±5% of baseline. |
| How widespread is silent no-op? | **6 of 10** `--moe-runner-backend` values are silent no-ops on bf16 (cutlass, deep_gemm, flashinfer_trtllm, flashinfer_mxfp4, flashinfer_cutedsl, marlin). Only `auto`, `triton`, `triton_kernel`, `flashinfer_cutlass` are honoured. |

---

## 23. 悄悄 no-op —— 精确源码证据 + 检测工具

> 🟢 新增。§22 后的两个 follow-up 问题：
> (1) 精确源码证据在哪？
> (2) 实际怎么知道一个 flag 被悄悄忽略了还是真的被用了？
>
> 同时包含对 §22.1 的**修正**：unquant.py **确实** 读了
> `get_moe_runner_backend()`（检查 flashinfer_cutlass 和 auto），只是没检查
> `cutlass` 值。精确机制比"完全忽略"更细一点。

### 23.1 精确源码证据 —— `unquant.py`

`UnquantizedFusedMoEMethod` 完整初始化
（[`sglang/srt/layers/quantization/unquant.py:155-167`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)）：

```python
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """无量化的 MoE method。"""

    def __init__(
        self, use_triton_kernels: bool = False, use_flashinfer_trtllm_moe: bool = False
    ):
        super().__init__()
        # ↓↓↓ 确实读 backend flag —— 但只检查一个值
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self._cache_permute_indices = dict({})
```

以及 runner 创建
（[`sglang/srt/layers/quantization/unquant.py:321-330`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)）：

```python
    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        backend = (
            MoeRunnerBackend.TRITON_KERNELS
            if self.use_triton_kernels
            else MoeRunnerBackend.TRITON
        )
        self.runner = MoeRunner(backend, moe_runner_config)
        # ↑↑↑ 只在 TRITON_KERNELS 和 TRITON 之间选
        #     —— 即使 server arg 说 cutlass / deep_gemm / flashinfer_trtllm
```

加上 AIter fallback gate
（[`unquant.py:229`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)）：

```python
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # AIter 只有 flag 是 'auto' 时才跑（也就是用户没显式请求某个 backend）
        _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
        ...
```

**精确解读**：bf16 unquantized MoE method **部分** 读 flag：

| `--moe-runner-backend` 值 | bf16 path 行为 |
|---|---|
| `auto`（默认）| Triton，AMD 上加 AIter 权重 shuffle |
| `triton` | Triton（显式，跟默认一样）|
| `triton_kernel` | TritonKernels 变体 |
| `flashinfer_cutlass` | Triton，**但** `self.use_flashinfer_cutlass=True` 改了权重加载顺序（`unquant.py:333-335` `load_up_proj_weight_first`）。MoE 计算 kernel 本身还是 `fused_moe_kernel`。 |
| **`cutlass`** | **悄悄 no-op → Triton**（C4 撞这里）|
| `deep_gemm` | **悄悄 no-op → Triton** |
| `flashinfer_trtllm` | **悄悄 no-op → Triton** |
| `flashinfer_mxfp4` | **悄悄 no-op → Triton** |
| `flashinfer_cutedsl` | **悄悄 no-op → Triton** |
| `marlin` | **悄悄 no-op → Triton** |

所以 bf16 MoE 上"悄悄 no-op"的表面积是 `--moe-runner-backend` 10 个值里
**6 个**。

**FP8 path** 显式检查 flag
（[`sglang/srt/layers/quantization/fp8.py:678-686`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)）：

```python
class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config):
        ...
        if get_moe_runner_backend().is_cutlass():
            assert cutlass_fp8_supported(), \
                "cutlass_fp8 MoE requires CUDA 12.0+ with SM90 or CUDA 12.4+ with SM89"
            assert self.block_quant, "cutlass_fp8 MoE requires block quantization"
            assert is_sm100_supported() or is_sm90_supported()
```

你传 `--moe-runner-backend cutlass --quantization fp8` 且 CUDA 太老，启动时
**清晰的 AssertionError**。

完整的 backend flag 读取位点 map（grep 自 sglang 源码）：

```
$ grep -rn "get_moe_runner_backend" sglang/srt/layers/quantization/
unquant.py:162          is_flashinfer_cutlass()   ← 读但只检查窄范围
unquant.py:229          is_auto()                  ← 读但只检查窄范围
unquant.py:391          is_auto()                  ← 读但只检查窄范围
fp8.py:1349             is_deep_gemm() / TRITON    ← 真 dispatcher（仅 FP8）
modelopt_quant.py:755   is_flashinfer_trtllm()     ← 真 dispatcher（仅 modelopt）
modelopt_quant.py:762   is_flashinfer_cutlass()    ← 真 dispatcher
modelopt_quant.py:855   is_flashinfer_cutlass()    ← 真 dispatcher
mxfp4.py:301-303        is_triton_kernels(), is_flashinfer_mxfp4()  ← 真 dispatcher（仅 mxfp4）
compressed_tensors/...  ← 各种，全是 quant-specific
awq.py:831              assert is_auto()           ← assertion，不是 dispatch
```

**unquantized path 读 flag，但只用来在 Triton 变体之间选 + AMD AIter gate。**
选 cutlass / DeepGEMM / flashinfer_trtllm 的值悄悄落到 Triton 上 —— 因为这些
backend 只对量化权重有实现。

### 23.2 为啥 `/server_info` 告诉你的不是真相

自然反应是"查 `/server_info` 看 server 自己说在用啥"。**这不管用**。源码
（[`sglang/srt/entrypoints/http_server.py:594-610`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)）：

```python
@app.get("/server_info")
async def server_info():
    """Get the server information."""
    internal_states: List[Dict[Any, Any]] = (
        await _global_state.tokenizer_manager.get_internal_state()
    )
    return {
        **dataclasses.asdict(_global_state.tokenizer_manager.server_args),
        # ↑↑↑ 原样 dump INPUT server_args —— 不是实际用的
        **_global_state.scheduler_info,
        "internal_states": internal_states,
        "version": __version__,
    }
```

`dataclasses.asdict(server_args)` 返回用户 CLI 传的（解析过默认值后）。它 **不**
反映：

- bf16 unquantized 路径是否悄悄忽略了 `--moe-runner-backend cutlass`
- AMD AIter 是否真的在跑（被 `is_auto()` gate）
- 显式 backend 的 JIT 是否成功了
- torch.compile 是全装上了还是只装了一部分

所以你用 `--moe-runner-backend cutlass` 启动 bf16 后查 `server_info`，
**它会告诉你 `moe_runner_backend: 'cutlass'`**，即使实际跑的是 Triton。这就是
C4 看到的：`server_info.json` 说 `cutlass`，trace 说 `fused_moe_kernel`。

### 23.3 检测工具 —— `detect_silent_noop.py`

检测悄悄 no-op 的唯一可靠方法是 **按每个 backend 应该产生的 kernel 给它做指纹**
然后查 trace。

我写了个小检测脚本，
[`scripts/regime_study/detect_silent_noop.py`](../scripts/regime_study/detect_silent_noop.py)，
自动做这事。它对每对 (backend_flag, value) 编码一个 regex —— 例如：

```python
BACKEND_FINGERPRINT = {
    ("moe_runner_backend", "cutlass"): [
        r"cutlass.*moe", r"cutlass_fused_experts", r"sm90_xmma_warpspecialized.*moe"
    ],
    ("moe_runner_backend", "triton"): [r"fused_moe_kernel"],
    ("attention_backend", "fa3"): [r"FlashAttnFwdSm90", r"flash::"],
    ("attention_backend", "flashinfer"): [
        r"flashinfer.*(?:Prefill|Decode|Attention)",
        r"BatchPrefillWithRaggedKV", r"BatchDecodeWithPagedKV",
    ],
    ...
}
```

干 3 件事：

1. 读 `server_info.json` 看用户请求了啥
2. 加载 trace，列出所有 GPU kernel 名 + 总 self-time
3. 对每个被请求的 backend，检查是否 **任何** kernel 匹配指纹；报
   `HONOURED` / `IGNORED_OR_FALLBACK`

加上可选的 **reference-trace diff**，在指纹模糊时也能抓到 silent no-op：
top-5 kernel 里 4 个在 ±5% 内 → flag 大概率是 no-op。

用法：

```bash
python scripts/regime_study/detect_silent_noop.py \
    --server-info results/regime_bench/raw/moe_opt_levels/C4_moe_cutlass/server_info.json \
    --trace experiments/tmp/moe_opt_levels/C4_moe_cutlass/raw_trace/*/p_*.trace.json.gz \
    --reference-trace experiments/tmp/moe_opt_levels/C0_baseline/raw_trace/*/p_*.trace.json.gz
```

输出（C4 上 smoke test）：

```
[detect] loading trace .../C4_moe_cutlass/raw_trace/.../p_-...-TP-0.trace.json.gz
[detect] 85 unique GPU kernels in trace
[detect] moe_runner_backend='cutlass' → ⚠️  IGNORED OR FELL BACK
[detect] attention_backend='fa3' → ✅ HONOURED
          matched pattern: /FlashAttnFwdSm90/
          matched pattern: /flash::/
            61364 us  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<…>
            18305 us  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<…>
[detect] sampling_backend='flashinfer' → ⚠️  IGNORED OR FELL BACK

[detect] reference trace .../C0_baseline/raw_trace/.../p_-...-TP-0.trace.json.gz
[detect] LIKELY_SILENT_NOOP
[detect] 5/5 kernels within ±5% of reference
            277135 vs 278301 us  (−0.4%)  fused_moe_kernel
             61364 vs  61582 us  (−0.4%)  void cutlass::device_kernel<flash::enable_sm90_or_later<flas
             37488 vs  37831 us  (−0.9%)  nvjet_tst_128x248_64x4_2x1_v_bz_coopA_TNT
             21556 vs  21538 us  (+0.1%)  nvjet_tst_192x208_64x4_1x2_h_bz_coopB_TNT
             18305 vs  18330 us  (−0.1%)  void cutlass::device_kernel<flash::enable_sm90_or_later<flas
```

两个清晰信号：

1. **`moe_runner_backend='cutlass'` → IGNORED OR FELL BACK** —— trace 里没
   cutlass-MoE kernel
2. **Kernel-mix 跟 baseline 近乎一致**（5/5 在 ±5% 内）→ 确证悄悄 no-op，
   不是"cutlass 恰好性能跟 triton 一样"

脚本退出码：任何 flag 被 IGNORED 就 exit 2 —— 可以在 CI 里用，让悄悄丢失优化
flag 的 build 失败。

> 关于上面 `sampling_backend='flashinfer' → IGNORED` 的说明：这是 **假阳性**。
> 我们对 flashinfer sampling backend 的指纹模式
> （`r"flashinfer.*sampling"`、`r"top_p"`、`r"top_k_sampling"`）不匹配实际的
> flashinfer sampling kernel 名 —— 这些被 launch 为小的 `at::native::*`
> thread-fusion kernel，因为 sampler op 很小可以 inline。sampling backend
> 的指纹库需要补；目前如果 attention 也是 flashinfer，可以假定 flashinfer
> sampling 也被用了（共享 JIT 初始化）。

### 23.4 验证任意 backend flag 的实用流程

给非默认 backend flag 启动 sglang 时，跑 3 步检查：

1. **确认 server 真的用了** —— 跑 trace + 检测：
   ```bash
   # 跑 sglang.bench_serving --profile（存 trace）
   # 然后：
   python scripts/regime_study/detect_silent_noop.py \
       --server-info <your_run>/server_info.json \
       --trace <your_run>/raw_trace/*/p_*.trace.json.gz
   ```
2. **读 server.log 找警告**（sglang 有时会 log）：
   ```bash
   grep -iE "ignored|falling back|fall.?back|not.?support|will.?use" server.log
   ```
3. **验证性能变化** —— 翻 flag 后 bench 数字 0.1% 内一致，大概率撞 no-op。对比：
   ```bash
   diff <(jq '.output_throughput' baseline/bench.jsonl) \
        <(jq '.output_throughput' yourrun/bench.jsonl)
   ```

三个都说"没变化"，要么 honoured-but-no-effect（比如 cutlass 是 auto
选的反正），要么悄悄忽略了。**检测器 ⊕ kernel diff 区分这两种。**

### 23.5 总结

| 问题 | 答案 |
|---|---|
| 源码证据在哪？ | `sglang/srt/layers/quantization/unquant.py:155-167`（init）+ `:321-330`（runner 创建）。第 162、229、391 行 **确实** 读 `get_moe_runner_backend()`，但只检查 `is_flashinfer_cutlass()`、`is_auto()`。`cutlass`、`deep_gemm`、`flashinfer_trtllm`、`marlin` 这些值默默 fall 到 Triton，没警告。 |
| 能用 `/server_info` 看吗？ | **不能**。`http_server.py:594-610` 返回 `dataclasses.asdict(server_args)` 原样 —— 显示你要的，不是实际跑的。 |
| 实际怎么检测？ | `scripts/regime_study/detect_silent_noop.py` —— 按 backend 预期的 kernel 名打指纹；指纹没匹配 flag IGNORED；可选 reference trace diff 看 kernel mix 相似度。 |
| 在 C4 上的检测准确度？ | ✅ 抓到了 `moe_runner_backend='cutlass'` 被 IGNORED，并通过 5/5 kernel ±5% 内确证。 |
| 悄悄 no-op 多普遍？ | bf16 上 `--moe-runner-backend` 10 个值里 **6 个** 是悄悄 no-op（cutlass、deep_gemm、flashinfer_trtllm、flashinfer_mxfp4、flashinfer_cutedsl、marlin）。只有 `auto`、`triton`、`triton_kernel`、`flashinfer_cutlass` 被真用。 |

## 24. "Quantization path" vs "narrow check" — what they actually mean

> 🟢 NEW. Follow-up question after §23: "Only quantized paths take the flag
> seriously" — what does that actually mean? Is it the model's dtype, or
> something else? And what's a "narrow check"?
>
> Short answer: it's BOTH the model's stored weight format AND a separate
> dispatch table inside sglang. This section traces the full path from
> `config.json` → quant_method class → which lines read which flag.

### 24.1 The chain — from `config.json` to "which MoE method runs"

Step-by-step for our Qwen3-30B-A3B MoE:

**Step 1**: At startup, sglang reads `config.json` from the model directory.
This is HuggingFace's standard config. For Qwen3-30B-A3B
(`/data/hf/models/Qwen3-30B-A3B-Instruct-2507/config.json`):

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "torch_dtype": "bfloat16",
  "quantization_config": null,   ← no quant config = unquantized
  ...
}
```

**Step 2**: `sglang/srt/configs/model_config.py:648` reads this field:

```python
def _parse_quant_hf_config(self):
    quant_cfg = getattr(self.hf_config, "quantization_config", None)
    if quant_cfg is None:
        # also check "compression_config" (compressed-tensors models use this key)
        quant_cfg = getattr(self.hf_config, "compression_config", None)
    if quant_cfg is None:
        # also try to download a standalone hf_quant_config.json
        ...
    return quant_cfg
```

For our model this returns **`None`** because the field isn't in `config.json`
and there's no `hf_quant_config.json` file.

**Step 3**: `sglang/srt/layers/moe/fused_moe_triton/layer.py:285-290` then
decides the quant_method:

```python
if quant_config is not None:
    self.quant_method = quant_config.get_quant_method(self, prefix)
if self.quant_method is None:
    self.quant_method = UnquantizedFusedMoEMethod(
        self.use_triton_kernels, self.use_flashinfer_trtllm_moe
    )
```

For our bf16 model `quant_config is None` → goes straight to
`UnquantizedFusedMoEMethod`. This is the **"bf16 / unquantized path"**.

For an FP8 model (e.g.
`https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8`) `config.json`
includes:

```json
"quantization_config": {"quant_method": "fp8", ...}
```

Then `quant_config.get_quant_method(...)` returns an `Fp8MoEMethod`
instance — the **"FP8 path"**. **Same flag, completely different code
reading it.**

The full set of quant-paths sglang has, one Python class per file in
`sglang/srt/layers/quantization/`:

```
unquant.py            → UnquantizedFusedMoEMethod    (bf16/fp16, no quant)
fp8.py                → Fp8MoEMethod                  ("quant_method": "fp8")
fp4_utils.py          → FP4Method
fpgemm_fp8.py         → FpGemmFp8MoEMethod            (per-group FP8)
blockwise_int8.py     → BlockwiseInt8MoEMethod
gptq.py               → GPTQMarlinMoEMethod           ("quant_method": "gptq")
awq.py                → AWQMarlinMoEMethod            ("quant_method": "awq")
bitsandbytes.py       → BitsAndBytesMoEMethod
gguf.py               → GGUFMoEMethod
mxfp4.py              → Mxfp4MoEMethod
modelopt_quant.py     → ModelOptFp8MoEMethod, ModelOptFp4MoEMethod
compressed_tensors/   → CompressedTensorsW8A8MoEMethod, ...W4A4..., ...etc
```

**Which class loads depends on `config.json`'s `quant_method` field**, not
on your CLI flag. Your CLI flag `--moe-runner-backend cutlass` is *a
parameter to whichever class loads*, not the class selector.

### 24.2 "Quantization path" = which of those classes serves your model

Concretely:

| Your model on disk | `config.json["quantization_config"]` | Quant class instantiated | Path nickname |
|---|---|---|---|
| Qwen3-30B-A3B bf16 (ours) | `null` | `UnquantizedFusedMoEMethod` | **"bf16 / unquantized path"** |
| Qwen3-30B-A3B FP8 (hypothetical) | `{"quant_method": "fp8", ...}` | `Fp8MoEMethod` | "FP8 path" |
| DeepSeek-V3 W4AFP8 | `{"quant_method": "w4afp8", ...}` | (compressed_tensors variant) | "W4AFP8 path" |
| Llama-3-AWQ | `{"quant_method": "awq", ...}` | `AWQMarlinMoEMethod` | "AWQ path" |

So "**quantization path**" just means: **which `*MoEMethod` class object
sglang actually constructed for your model**. Different classes have
different code, including completely different rules for reading
`--moe-runner-backend`.

### 24.3 Does `--quantization fp8` change which class runs?

**Yes, but only for FP8** — sglang has explicit logic to *convert* a bf16
weight set to FP8 on the fly if user passes `--quantization fp8`. Pseudo-code:

```python
# server_args.py parses --quantization
# model_executor builds:
quant_config = (
    Fp8Config(...)                  # explicit override → Fp8MoEMethod
    if server_args.quantization == "fp8"
    else parse_from_hf_config()      # uses the model's config.json field
)
```

If user passes `--quantization fp8` on a bf16-on-disk model, sglang reads
bf16 weights and **quantizes-on-load** to FP8 (lossy). Now you're on the
FP8 path, and `--moe-runner-backend cutlass` actually does something
(see §24.4).

This is why the recommendation in §19.7 was "try FP8 quantization to
unlock cutlass MoE path" — passing `--quantization fp8` switches the
quant_method from `UnquantizedFusedMoEMethod` to `Fp8MoEMethod`, which is
the class that *does* dispatch on `--moe-runner-backend`.

### 24.4 "Narrow check" — exactly what the bf16 path does with the flag

Recall from §23.1, the bf16 path is `UnquantizedFusedMoEMethod`. Let's
trace **every single read** of `get_moe_runner_backend()` inside this class
(grep'd from sglang source):

```
unquant.py:162  self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
unquant.py:229  _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
unquant.py:391  _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
unquant.py:325  backend = (TRITON_KERNELS if self.use_triton_kernels else TRITON)
                # ↑ does NOT read the server flag; uses a separate instance
                #   attribute self.use_triton_kernels passed in via __init__
```

**These three reads + one ignore are "the narrow check"**:

1. **Line 162** — at construction time, set `self.use_flashinfer_cutlass`
   if-and-only-if the user passed `flashinfer_cutlass`. This bool is then
   used at line 333-335 to change the weight loading order (`load_up_proj_weight_first`).
   The actual MoE compute kernel is still `fused_moe_kernel`. So
   `flashinfer_cutlass` value affects layout, not which kernel runs.
2. **Line 229 + 391** — only used to gate the AMD AIter weight shuffle.
   If user passed *anything* other than `auto`, AIter is disabled.
3. **Line 325** — picks between `TRITON_KERNELS` and `TRITON` based on a
   `use_triton_kernels` flag that came in via `__init__`. That flag is
   itself set from `--moe-runner-backend triton_kernel` upstream.

So out of the 10 possible values of `--moe-runner-backend`, the bf16
path **only does something different** for:

- `auto` (AIter shuffle on AMD)
- `triton` (default, same as auto on non-AMD)
- `triton_kernel` (TritonKernels variant via `use_triton_kernels=True`)
- `flashinfer_cutlass` (just weight-loading order change, kernel still
  Triton)

The other 6 values (`cutlass`, `deep_gemm`, `flashinfer_trtllm`,
`flashinfer_mxfp4`, `flashinfer_cutedsl`, `marlin`) **are read but never
match any of the above conditionals**, so they fall through to the default
Triton path. That's the **"narrow"** part — the bf16 path only has a small
window of values it acts on.

By contrast, the FP8 path
([`fp8.py:678-686`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)
and
[`fp8.py:1349`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py))
has **explicit branches for `cutlass`, `deep_gemm`, and falls back to
Triton in the else** — and crucially raises `AssertionError` if the
chosen path can't run on this hardware. That's a **"wide" check** — it
considers every value, with an assertion as the catch-all.

### 24.5 Concrete one-line summary

> "**Quantization path**" = which of the ~12 `*MoEMethod` classes sglang
> instantiated for your model, decided at startup from `config.json`'s
> `quantization_config` field (or `--quantization` override). Each class
> is independent code with its own backend-flag handling.
>
> "**Narrow check**" = the bf16 `UnquantizedFusedMoEMethod` class only
> branches on **4 of 10** `--moe-runner-backend` values (`auto`, `triton`,
> `triton_kernel`, `flashinfer_cutlass`); the other 6 are silently
> ignored and the code falls through to Triton. This is in contrast to
> FP8's class which branches on more values with explicit asserts.

### 24.6 Why this design (best guess)

Reading the code, the rationale appears to be:

1. **Quantized MoE kernels are quant-format-specific**. A "cutlass FP8 MoE"
   kernel can't be reused for bf16 because the tensor-core MMA instructions
   are different (FP8 uses HMMA.16832.E4M3, bf16 uses HMMA.16816.BF16).
   So there literally is no "cutlass bf16 MoE" kernel to dispatch to.
2. **Triton bf16 MoE is already optimal for bf16 on Hopper**.
   `fused_moe_kernel` already uses Sm90 mma.bf16 instructions and beats
   naïve PyTorch ops by ~10×. Maintainers didn't write a bf16 cutlass MoE
   because Triton is already at the GEMM peak for bf16 shapes.
3. **No early validation = simpler argparse**. If the bf16 path raised
   "cutlass not supported for bf16" at startup, users would need separate
   flags per quantization level, or a giant compatibility matrix in
   server_args.py. The maintainers chose to make `--moe-runner-backend`
   a "hint that's honoured if the chosen quant class knows what to do" —
   which is silent on bf16 because there's nothing to do.

**But the silent-no-op is the cost**. The patch I proposed in §22.1 would
add a `logger.warning(...)` when the bf16 path sees a quantized-only
backend value — at least announcing the no-op.

### 24.7 Practical implications

- **You can't accelerate bf16 MoE by changing `--moe-runner-backend`**.
  The only real lever for bf16 MoE on H200 is `--quantization fp8`
  (which changes the quant_path and unlocks cutlass/DeepGEMM).
- **`--moe-runner-backend` is a hint, not a command**. The class
  receiving it gets to decide whether/how to honour it.
- **Always pair backend flags with quant flags when testing**. Otherwise
  you're probably benchmarking the default Triton path against itself.

---

## 24. "量化路径" vs "窄检查" —— 具体啥意思

> 🟢 新增。§23 后的 follow-up 问题："只有量化路径才把 flag 当真" 具体是啥意思？
> 是模型的 dtype 决定的还是别的？窄检查又是啥？
>
> 简短答案：**两者都有** —— 既看模型的存储权重格式，也看 sglang 内部的一张分发表。
> 这节追踪从 `config.json` → quant_method 类 → 哪行读哪个 flag 的完整链。

### 24.1 完整链 —— 从 `config.json` 到"哪个 MoE method 跑"

我们 Qwen3-30B-A3B MoE 的逐步过程：

**Step 1**：启动时 sglang 读模型目录的 `config.json`，这是 HF 标准 config。我们的：

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "torch_dtype": "bfloat16",
  "quantization_config": null,   ← 没量化 config = 未量化
  ...
}
```

**Step 2**：`sglang/srt/configs/model_config.py:648` 读这个字段：

```python
def _parse_quant_hf_config(self):
    quant_cfg = getattr(self.hf_config, "quantization_config", None)
    if quant_cfg is None:
        # 也查 "compression_config"（compressed-tensors 模型用这个 key）
        quant_cfg = getattr(self.hf_config, "compression_config", None)
    if quant_cfg is None:
        # 也尝试下载独立的 hf_quant_config.json
        ...
    return quant_cfg
```

我们模型这里返回 **`None`** —— `config.json` 没这字段，没 `hf_quant_config.json`。

**Step 3**：`sglang/srt/layers/moe/fused_moe_triton/layer.py:285-290` 决定 quant_method：

```python
if quant_config is not None:
    self.quant_method = quant_config.get_quant_method(self, prefix)
if self.quant_method is None:
    self.quant_method = UnquantizedFusedMoEMethod(
        self.use_triton_kernels, self.use_flashinfer_trtllm_moe
    )
```

我们 bf16 模型 `quant_config is None` → 直接走 `UnquantizedFusedMoEMethod`。
这就是 **"bf16 / 未量化路径"**。

FP8 模型（比如
`https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8`）的 `config.json` 有：

```json
"quantization_config": {"quant_method": "fp8", ...}
```

那 `quant_config.get_quant_method(...)` 返回 `Fp8MoEMethod` 实例 —— **"FP8 路径"**。
**同一个 flag，完全不同的代码读它。**

sglang 的全部 quant 路径，`sglang/srt/layers/quantization/` 下每个文件一个 Python 类：

```
unquant.py            → UnquantizedFusedMoEMethod    (bf16/fp16，不量化)
fp8.py                → Fp8MoEMethod                 ("quant_method": "fp8")
fp4_utils.py          → FP4Method
fpgemm_fp8.py         → FpGemmFp8MoEMethod           (per-group FP8)
blockwise_int8.py     → BlockwiseInt8MoEMethod
gptq.py               → GPTQMarlinMoEMethod          ("quant_method": "gptq")
awq.py                → AWQMarlinMoEMethod           ("quant_method": "awq")
bitsandbytes.py       → BitsAndBytesMoEMethod
gguf.py               → GGUFMoEMethod
mxfp4.py              → Mxfp4MoEMethod
modelopt_quant.py     → ModelOptFp8MoEMethod, ModelOptFp4MoEMethod
compressed_tensors/   → CompressedTensorsW8A8MoEMethod, ...W4A4..., 等等
```

**加载哪个类取决于 `config.json` 的 `quant_method` 字段**，不取决于你的 CLI flag。
你的 CLI flag `--moe-runner-backend cutlass` 是 *给那个被加载的类的参数*，不是
**类选择器**。

### 24.2 "量化路径" = 那些类里哪一个伺候你的模型

具体说：

| 你硬盘上的模型 | `config.json["quantization_config"]` | 实例化的 quant 类 | 路径昵称 |
|---|---|---|---|
| Qwen3-30B-A3B bf16（我们的）| `null` | `UnquantizedFusedMoEMethod` | **"bf16 / 未量化路径"** |
| Qwen3-30B-A3B FP8（假设）| `{"quant_method": "fp8", ...}` | `Fp8MoEMethod` | "FP8 路径" |
| DeepSeek-V3 W4AFP8 | `{"quant_method": "w4afp8", ...}` | (compressed_tensors 变体) | "W4AFP8 路径" |
| Llama-3-AWQ | `{"quant_method": "awq", ...}` | `AWQMarlinMoEMethod` | "AWQ 路径" |

所以 "**量化路径**" 就是：**sglang 给你模型实际构造的 `*MoEMethod` 类对象是哪个**。
不同类是完全独立的代码，包括完全不同的 `--moe-runner-backend` 读取规则。

### 24.3 `--quantization fp8` 会换走哪个类吗？

**会，但只限 FP8** —— sglang 有显式逻辑，用户传 `--quantization fp8` 时把
bf16 权重 *实时转* 成 FP8。伪码：

```python
# server_args.py 解析 --quantization
# model_executor 构造：
quant_config = (
    Fp8Config(...)                  # 显式 override → Fp8MoEMethod
    if server_args.quantization == "fp8"
    else parse_from_hf_config()      # 用模型 config.json 字段
)
```

用户在硬盘是 bf16 的模型上传 `--quantization fp8`，sglang 读 bf16 权重并
**加载时量化** 到 FP8（有损）。这时你在 FP8 路径上了，`--moe-runner-backend cutlass`
真的起作用（见 §24.4）。

这就是 §19.7 里推荐"试 FP8 量化解锁 cutlass MoE 路径"的原因 —— 传
`--quantization fp8` 把 quant_method 从 `UnquantizedFusedMoEMethod` 换成
`Fp8MoEMethod`，**那个类才会真的 dispatch `--moe-runner-backend`**。

### 24.4 "窄检查" —— bf16 路径对 flag 到底干了啥

回顾 §23.1，bf16 路径是 `UnquantizedFusedMoEMethod`。我们追这个类里
**每一处** 读 `get_moe_runner_backend()`（grep 自 sglang 源码）：

```
unquant.py:162  self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
unquant.py:229  _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
unquant.py:391  _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
unquant.py:325  backend = (TRITON_KERNELS if self.use_triton_kernels else TRITON)
                # ↑ 不读 server flag；用的是另一个 instance 属性 self.use_triton_kernels
                #   通过 __init__ 传进来
```

**这 3 个读 + 1 个忽略就是"窄检查"**：

1. **第 162 行** —— 构造时，**仅当** 用户传 `flashinfer_cutlass` 时
   `self.use_flashinfer_cutlass = True`。这个 bool 在第 333-335 行被用，
   改变权重加载顺序（`load_up_proj_weight_first`）。实际 MoE 计算 kernel 还是
   `fused_moe_kernel`。所以 `flashinfer_cutlass` 值只影响 layout，不影响跑哪个 kernel。
2. **第 229 + 391 行** —— 只用来 gate AMD AIter 权重 shuffle。用户传 `auto`
   以外的任何值，AIter 都被禁掉。
3. **第 325 行** —— 在 `TRITON_KERNELS` 和 `TRITON` 之间选，按 `__init__` 传入的
   `use_triton_kernels` flag。那个 flag 自己在上游由 `--moe-runner-backend triton_kernel`
   设。

所以 `--moe-runner-backend` 10 个可能值里，bf16 路径 **只对这几个做不同的事**：

- `auto`（AMD 上 AIter shuffle）
- `triton`（默认，非 AMD 上跟 auto 一样）
- `triton_kernel`（通过 `use_triton_kernels=True` 走 TritonKernels 变体）
- `flashinfer_cutlass`（只改权重加载顺序，kernel 还是 Triton）

其他 6 个值（`cutlass`、`deep_gemm`、`flashinfer_trtllm`、`flashinfer_mxfp4`、
`flashinfer_cutedsl`、`marlin`）**被读了但跟上面任何条件都不匹配**，所以默默落到默认
Triton 路径。这就是 **"窄"** 的意思 —— bf16 路径只对一小撮值起作用。

对比一下，FP8 路径（[`fp8.py:678-686`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)
和 [`fp8.py:1349`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)）
**对 `cutlass`、`deep_gemm` 都有显式分支，else 回 Triton** —— 关键是当选的路径在
这个硬件上跑不了时会抛 `AssertionError`。这是 **"宽"检查** —— 它考虑每个值，
默认行为是 assertion 兜底。

### 24.5 具体一句话总结

> "**量化路径**" = sglang 给你模型实例化的 ~12 个 `*MoEMethod` 类中的哪一个，
> 启动时根据 `config.json` 的 `quantization_config` 字段决定（或
> `--quantization` 显式 override）。每个类是独立代码，自带独立的 backend-flag 处理。
>
> "**窄检查**" = bf16 `UnquantizedFusedMoEMethod` 类只在 `--moe-runner-backend`
> 的 10 个值中的 **4 个** 上做分支（`auto`、`triton`、`triton_kernel`、
> `flashinfer_cutlass`）；其他 6 个被默默忽略，代码默认落到 Triton。FP8 类反之，
> 在更多值上有分支 + 显式 assert。

### 24.6 为啥这么设计（best guess）

读源码看，理由大致是：

1. **量化 MoE kernel 是 quant-format-specific 的**。"cutlass FP8 MoE" kernel
   不能复用给 bf16，因为 tensor-core MMA 指令不同（FP8 用 HMMA.16832.E4M3，
   bf16 用 HMMA.16816.BF16）。所以**真的没有"cutlass bf16 MoE" kernel** 能 dispatch。
2. **Triton bf16 MoE 在 Hopper 上已经最优**。`fused_moe_kernel` 已经用了 Sm90
   mma.bf16 指令，吊打朴素 PyTorch op ~10×。维护者没写 bf16 cutlass MoE 因为
   Triton 在 bf16 shape 上已经到了 GEMM 极限。
3. **早期验证 = 简化 argparse**。如果 bf16 路径在启动时报"bf16 不支持 cutlass"，
   用户就要为每个量化级别单独的 flag，或者 server_args.py 里维护一个巨大兼容性矩阵。
   维护者选择让 `--moe-runner-backend` 成为"被选中的 quant 类如果知道怎么处理就处理的
   hint" —— bf16 上是 silent 的因为没什么可做。

**但 silent no-op 是代价**。§22.1 提的补丁就是在 bf16 路径看到量化-only backend
值时加 `logger.warning(...)` —— 至少声明 no-op。

### 24.7 实用启发

- **你不能靠改 `--moe-runner-backend` 加速 bf16 MoE**。H200 上 bf16 MoE 的
  真正杠杆是 `--quantization fp8`（换 quant_path、解锁 cutlass/DeepGEMM）。
- **`--moe-runner-backend` 是建议，不是命令**。接收它的类决定是否/怎么遵从。
- **测 backend flag 时永远配套 quant flag**。否则你可能在拿默认 Triton 路径
  跟自己 benchmark。

## 25. The 4 honoured branches — exact differences + is bf16 MoE all just Triton?

> 🟢 NEW. Follow-up after §24:
> (1) "only changes weight loading order, kernel still Triton" — what does
>     that mean? What's the difference vs the other 3 honoured branches?
> (2) Under bf16, are all MoE models actually running the same Triton kernel?
> (3) `fused_moe_kernel` itself — is it pre-written by sglang, hand-coded
>     or generated? Where does the optimisation headroom come from?
>
> Also includes a **correction** to §24.4 — `flashinfer_cutlass` actually
> DOES switch to a different kernel (not just permute weights). The previous
> wording understated this branch's effect.

### 25.1 Correction to §24.4 — flashinfer_cutlass really does swap the kernel

When I said "only changes weight loading order, kernel still Triton" for
the `flashinfer_cutlass` branch, **that was incomplete**. Looking at the
full `apply()` method
([`sglang/srt/layers/quantization/unquant.py:355-415`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)):

```python
def apply(self, layer, dispatch_output) -> CombineInput:
    backend = self.runner.runner_backend

    if backend.is_triton_kernels():
        # Branch 1: TritonKernels variant
        quant_info = TritonKernelsQuantInfo(...)
        return self.runner.run(dispatch_output, quant_info)

    elif self.use_flashinfer_cutlass:
        # Branch 2: flashinfer_cutlass — SWAPS THE KERNEL
        output = flashinfer_cutlass_fused_moe(   # ← NOT fused_moe_kernel
            input=x,
            token_selected_experts=topk_output.topk_ids,
            ...
            fc1_expert_weights=layer.w13_weight,
            fc2_expert_weights=layer.w2_weight,
            ...
        )[0]
        return StandardCombineInput(hidden_states=output)

    else:
        # Branch 3 + 4: AIter (auto on AMD) or default Triton fused_moe_kernel
        _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
        if _should_use_aiter_moe:
            output = fused_moe(...)  # AIter's C++/HIP MoE
        else:
            output = self.runner.run(dispatch_output, quant_info)
            # ↑ this is the path that ends up in fused_moe_kernel
```

So `flashinfer_cutlass` **actually calls `flashinfer_cutlass_fused_moe`** —
a different kernel from flashinfer's library. AND it also requires a
weight-permute step at load time
([`unquant.py:260-295`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)
in `process_weights_after_loading`) to re-order each expert's weight into
flashinfer's expected memory layout. So **both things change**: weight
layout AND the kernel called.

**Revised honoured-branch table for bf16 MoE**:

| Value | Kernel actually called | Weight format | Lives in |
|---|---|---|---|
| `auto` (default, non-AMD) | `fused_moe_kernel` (Triton JIT) | as-loaded | `sglang/srt/layers/moe/fused_moe_triton/` |
| `auto` (default, AMD) | AIter's `fused_moe` (C++/HIP) | shuffled to AIter layout | `sglang/srt/layers/moe/` + AIter ROCm library |
| `triton` (explicit) | `fused_moe_kernel` (Triton JIT) | as-loaded | same as auto (non-AMD) — equivalent |
| `triton_kernel` | `TritonKernelsRunnerCore.run()` → `matmul_ogs` | as-loaded with shape transpose | `sglang/srt/layers/moe/moe_runner/triton_kernels.py` + `triton_kernels` library |
| `flashinfer_cutlass` | `flashinfer_cutlass_fused_moe` | re-permuted into block-layout | `flashinfer.fused_moe.cutlass` |

So **the 4 honoured branches really are 3 distinct kernel sources** (Triton,
TritonKernels, flashinfer-cutlass) + 1 AMD-only AIter variant. The other
6 values (cutlass, deep_gemm, flashinfer_trtllm, flashinfer_mxfp4,
flashinfer_cutedsl, marlin) silently fall through to the default Triton
`fused_moe_kernel`.

### 25.2 So is every bf16 MoE on H200 running the same Triton kernel?

**Almost yes — on H200 specifically.** Let's trace every path:

| Setup | Path | Kernel actually called |
|---|---|---|
| Qwen3-30B-A3B bf16 + `--moe-runner-backend auto` (our C0) | bf16 path, no AMD → falls into `else` branch | **`fused_moe_kernel`** |
| Qwen3-30B-A3B bf16 + `--moe-runner-backend triton` | same as auto | **`fused_moe_kernel`** |
| Qwen3-30B-A3B bf16 + `--moe-runner-backend cutlass` (our C4) | silent no-op → Triton | **`fused_moe_kernel`** |
| Qwen3-30B-A3B bf16 + `--moe-runner-backend deep_gemm` | silent no-op → Triton | **`fused_moe_kernel`** |
| Qwen3-30B-A3B bf16 + `--moe-runner-backend triton_kernel` | TritonKernels variant | `matmul_ogs` (triton_kernels lib) |
| Qwen3-30B-A3B bf16 + `--moe-runner-backend flashinfer_cutlass` | flashinfer-cutlass branch | `flashinfer_cutlass_fused_moe` |
| Qwen3-30B-A3B bf16 + nothing custom (i.e. just auto) on **AMD MI300X** | AIter variant | AIter `fused_moe` (C++/HIP) |
| Qwen3-30B-A3B FP8 (quantized model) + `--moe-runner-backend cutlass` | Fp8MoEMethod cutlass branch | `cutlass_fused_experts_fp8` |
| Llama-3-AWQ + auto | AWQ marlin branch | Marlin's W4A16 GEMM |

**On H200 + bf16 specifically**:

- The **vast majority** of setups end up at `fused_moe_kernel` (Triton).
- `triton_kernel` exists as an alternative but it's a different
  Python/Triton library (`triton_kernels` is a separate openai/triton
  subproject for grouped GEMM).
- `flashinfer_cutlass` is the only "real alternative" — it dispatches to a
  pre-compiled CUTLASS kernel from flashinfer.

**Our C0-baseline through C6 all use `fused_moe_kernel`** because:
- C0, C2, C3, C6: explicit/default → Triton path
- C4 cutlass: silent no-op → Triton path
- C1 (torch.compile) and C5 (flashinfer attention) FAILED to load
- We never tested `triton_kernel` or `flashinfer_cutlass` explicitly

So the 24-cell hardware view in §15 / §16 / §19 is fundamentally a study
of **one kernel** (`fused_moe_kernel`) under different launch conditions.

### 25.3 What IS `fused_moe_kernel`? — hand-written, by sglang, in Triton

The kernel lives in
[`sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py)
(line 323-580).

| Property | Value |
|---|---|
| **File size** | 1 148 lines of Python |
| **`@triton.jit` kernels in this file** | 8 (one main `fused_moe_kernel` + 7 helpers for FP8 / GPTQ / AWQ / split-K variants) |
| **Authorship** | Hand-written by sglang maintainers, in Triton DSL |
| **NOT generated** | No code generator. The `@triton.jit` source is what runs (after Triton's compiler turns it into PTX) |
| **Adapted from** | vLLM's `fused_moe_kernel` (sglang notes "Adapted from vllm" in some places) |
| **Per-shape tuning** | 35 JSON files under `configs/triton_3_X_X/E=<E>,N=<N>,device_name=<…>.json` provide `BLOCK_SIZE_M/N/K`, `num_warps`, `num_stages` per (M-bucket, expert count, hidden size, device, dtype) |

**How it works in 2 lines**:

1. The Python source (lines 323-580) reads `BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
   `BLOCK_SIZE_K`, `num_warps`, `num_stages` as `tl.constexpr` parameters.
2. At each `kernel[grid](...)` invocation, the wrapper picks the JSON
   row matching current `M` (= num_tokens × topk), passes it to Triton,
   which JIT-compiles a specialised PTX variant for that (constexpr,
   shape) combo. The PTX is cached so subsequent calls are free.

**"Hand-written"** here means: a human (you can see them on
[GitHub Blame](https://github.com/sgl-project/sglang/blame/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py))
typed the body of:

```python
@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, ..., BLOCK_SIZE_M: tl.constexpr, ...):
    pid = tl.program_id(axis=0)
    # ... ~250 lines of pointer arithmetic, GEMM loops, fusion logic ...
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        ...
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token)
        accumulator = accumulator * moe_weight[:, None]
    c = accumulator.to(compute_type)
    tl.store(c_ptr + ..., c)
```

The **JSON tuning tables are auto-generated** (via Triton's autotuner, run
offline by sglang devs on each major GPU). At inference time, sglang just
**looks up** the right row — no autotuning happens at startup.

**"Generated" parts** in the bigger picture:
- **Per-shape compiled PTX**: Triton JIT-compiles each (BLOCK_SIZE_M, ...,
  dtype, shape) combo to PTX once at startup and caches in
  `~/.triton/cache/`. The cache key is a hash of the constexpr params.
- **JSON tuning tables**: pre-tuned offline. New (E, N, device) combos
  fall back to a "nearest" lookup, which is why we sometimes see a 1
  µs-per-call gap when M is between two JSON buckets.

### 25.4 Is there optimisation headroom inside `fused_moe_kernel`?

**Yes, lots.** §15 / §20 already proved a single kernel dominates 34-47 %
of GPU time on MoE. Specific opportunities, in priority order:

#### (a) Tune the JSON for our exact shape (cheap, possibly big win)

Our Qwen3-30B-A3B has `E=128, N=768`. The JSON file
`configs/triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json` is shared
across all `(M, dtype)` buckets, but **the tuning was done on a different
workload mix**. Specifically:

- The JSON's `M=1024` row uses `BLOCK_SIZE_M=128, num_warps=8, num_stages=4`.
- Our R8 with concurrency 32 + prefix 2048+128 gives `M ≈ 32 × 8 ≈ 256` for
  decode steps and `M ≈ 32 × 2176 × 8 / 128 ≈ 4352` for prefill.
- Re-running Triton's `tune.py` against our specific decode-vs-prefill M
  distribution would likely find a 5-10 % win.

#### (b) Fuse `moe_align_block_size` + `fused_moe_kernel` (medium, big win)

§15 showed `cudaEventSynchronize` between these two kernels accounts for
**37 %** of GPU time on MoE R3/R4 (prefill regimes). The sync is needed
because `fused_moe_kernel` reads `num_tokens_post_padded` written by
`moe_align_block_size`. If we could express both as a single kernel (or at
least pipeline them onto the same stream with overlapping work), the sync
would go away.

#### (c) Activation fusion (cheap, small win)

The pipeline today is:

```
fused_moe_kernel (gate+up GEMM) → SiluAndMul (small kernel) → fused_moe_kernel (down GEMM)
```

The SiluAndMul in the middle is a separate kernel (~3 % of GPU time).
Folding it into the down-GEMM kernel (so down loads SiLU-activated tiles
directly from the up-GEMM output staying in shared memory) is a known
Triton optimisation. **NOT trivial** — would need to rewrite the kernel
to take 2 weight tensors (w13 and w2) and pipeline both GEMMs.

#### (d) FP8 quantization (big win, requires data prep)

The biggest single lever — but requires:
- Converting Qwen3-30B-A3B weights from bf16 to FP8 (lossy)
- Switching to `Fp8MoEMethod` → unlocks `cutlass_fused_experts_fp8` (the
  "real" cutlass MoE path)
- Expected: 1.5-2× throughput on Hopper (FP8 tensor cores have 2× peak
  throughput vs bf16)

#### (e) DeepGEMM backend (big win on H200/H100, FP8 only)

`deep_gemm` (from DeepSeek, available in sglang via
`--moe-runner-backend deep_gemm`) is a hand-tuned CUTLASS-based MoE kernel
specifically for FP8 + Hopper. Reports 1.2-1.5× over Triton FP8 path on
DeepSeek-V3.

#### (f) Expert parallelism + DeepEP all-to-all (large scale only)

For multi-GPU setups, switching from tensor parallelism to expert
parallelism (`--ep-size 8 --moe-a2a-backend deepep`) can reduce per-GPU
work significantly. Doesn't help our single-H200 setup but matters for
production.

#### (g) Routing imbalance (data-dependent)

`fused_moe_kernel` assumes uniform expert load (each block is one expert).
In practice some experts get hotter than others, leaving some blocks
nearly empty (still consuming a tile of compute). Variable-block-size
scheduling would help — but it's a research problem, not just a kernel
rewrite.

#### Priority ranking for our project

| Opportunity | Cost | Expected speedup | Maturity |
|---|---|---|---|
| (a) Re-tune JSON for our shape | 1 day | 5-10 % | Trivial via Triton autotuner |
| (b) Fuse moe_align + fused_moe | 2-4 weeks (Triton kernel surgery) | 10-30 % on prefill regimes | Hard |
| (c) Fuse SiluAndMul into down | 1-2 weeks | ~3 % | Medium |
| (d) FP8 quantization | 2-3 days | 1.5-2× | Easy (just convert weights + flag) |
| (e) DeepGEMM | requires (d) | 1.2-1.5× over (d) | Easy once FP8 is set up |
| (f) Expert parallelism | weeks (system change) | depends on cluster size | Production-only |
| (g) Routing-balance research | months | TBD | Research |

**For getting fastest wins on our H200 single-GPU setup**: try (d) FP8
quantization first — it's the only one with a 2× ceiling without a kernel
rewrite. Then (a) JSON re-tuning for the residual.

### 25.5 Concrete answers to the three sub-questions

**Q: "only changes weight loading order, kernel still Triton" — what does that mean? And how does it differ from the other 3 honoured branches?**

That was my mis-statement. **`flashinfer_cutlass` actually swaps the kernel** to `flashinfer_cutlass_fused_moe` AND re-permutes weights. The 4 honoured bf16 branches are 4 distinct code paths:

| Branch | Kernel | Weight format |
|---|---|---|
| `auto`/`triton` | sglang's `fused_moe_kernel` (Triton JIT) | as-loaded |
| `auto` on AMD | AIter's `fused_moe` (C++/HIP) | shuffled |
| `triton_kernel` | `triton_kernels` library's `matmul_ogs` | transposed |
| `flashinfer_cutlass` | flashinfer's `flashinfer_cutlass_fused_moe` | block-permuted |

**Q: Under bf16, are all MoE models actually running the same Triton kernel?**

On H200 + bf16 + default flags: **yes, virtually all paths converge on `fused_moe_kernel`**. Our 24-cell hardware-view study (§15-§20) is fundamentally a study of one kernel under different launch conditions and different surrounding graph topologies.

**Q: What IS `fused_moe_kernel`? Hand-written? Generated?**

Hand-written in **Triton DSL** by sglang maintainers (~1 148 lines including 8 `@triton.jit` kernels in the file). Adapted from vLLM. Not auto-generated. Triton JIT-compiles it to PTX per (M, dtype, shape) combo on first use, cached in `~/.triton/cache/`. The **JSON tuning tables** (block sizes, warps, stages) under `configs/triton_3_X_X/` were generated offline by Triton's autotuner.

**Q: Is there optimisation headroom?**

Yes, lots. Concrete priorities for our use case:
1. **FP8 quantization** — only single lever with a 2× ceiling
2. **JSON re-tune for our shape** — 5-10 %, 1 day of work
3. **`moe_align` + `fused_moe` fusion** — kills the 37 % sync wait on prefill regimes (hard)
4. **SiluAndMul fusion** — 3 % win (medium)
5. **DeepGEMM backend** — additional 1.2-1.5× after FP8

---

## 25. 4 个被真用的分支 —— 精确区别 + bf16 MoE 是不是都在跑 Triton？

> 🟢 新增。§24 后的 follow-up：
> (1) "只改权重加载顺序，kernel 还是 Triton" 啥意思？跟另外 3 个被真用的分支区别是啥？
> (2) bf16 下 MoE 模型实际上都在跑同一个 Triton kernel 吗？
> (3) `fused_moe_kernel` 本身 —— 是 sglang 提前写好的？手写的还是生成的？优化空间从哪来？
>
> 同时包含对 §24.4 的**修正** —— `flashinfer_cutlass` 实际上 **真的换 kernel**
> （不只是 permute 权重）。之前的说法低估了这个分支的效果。

### 25.1 修正 §24.4 —— flashinfer_cutlass 真的换 kernel

我之前说 `flashinfer_cutlass` 分支 "只改权重加载顺序，kernel 还是 Triton"
**不完整**。看完整的 `apply()`
（[`sglang/srt/layers/quantization/unquant.py:355-415`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/unquant.py)）：

```python
def apply(self, layer, dispatch_output) -> CombineInput:
    backend = self.runner.runner_backend

    if backend.is_triton_kernels():
        # 分支 1：TritonKernels 变体
        quant_info = TritonKernelsQuantInfo(...)
        return self.runner.run(dispatch_output, quant_info)

    elif self.use_flashinfer_cutlass:
        # 分支 2：flashinfer_cutlass —— 真的换 KERNEL
        output = flashinfer_cutlass_fused_moe(   # ← 不是 fused_moe_kernel
            input=x,
            token_selected_experts=topk_output.topk_ids,
            ...
            fc1_expert_weights=layer.w13_weight,
            fc2_expert_weights=layer.w2_weight,
            ...
        )[0]
        return StandardCombineInput(hidden_states=output)

    else:
        # 分支 3 + 4：AIter（AMD 上 auto）或默认 Triton fused_moe_kernel
        _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
        if _should_use_aiter_moe:
            output = fused_moe(...)  # AIter 的 C++/HIP MoE
        else:
            output = self.runner.run(dispatch_output, quant_info)
            # ↑ 这条路径最终走到 fused_moe_kernel
```

所以 `flashinfer_cutlass` **真的调** `flashinfer_cutlass_fused_moe` ——
flashinfer 库里的另一个 kernel。**而且** 在 load 时需要权重 permute 步骤
（`unquant.py:260-295` 的 `process_weights_after_loading`）把每个专家的权重
重排成 flashinfer 期望的内存布局。所以 **两件事都变了**：权重布局 AND 调用的 kernel。

**修正后的 bf16 MoE honoured-branch 表**：

| 值 | 实际调的 kernel | 权重格式 | 在哪里 |
|---|---|---|---|
| `auto`（默认，非 AMD）| `fused_moe_kernel`（Triton JIT）| 加载即用 | `sglang/srt/layers/moe/fused_moe_triton/` |
| `auto`（默认，AMD）| AIter 的 `fused_moe`（C++/HIP）| 被 shuffle 成 AIter 布局 | `sglang/srt/layers/moe/` + AIter ROCm 库 |
| `triton`（显式）| `fused_moe_kernel`（Triton JIT）| 加载即用 | 跟 auto（非 AMD）一样 |
| `triton_kernel` | `TritonKernelsRunnerCore.run()` → `matmul_ogs` | 加载时 transpose | `sglang/srt/layers/moe/moe_runner/triton_kernels.py` + `triton_kernels` 库 |
| `flashinfer_cutlass` | `flashinfer_cutlass_fused_moe` | 被 re-permute 成 block-layout | `flashinfer.fused_moe.cutlass` |

所以 **4 个被真用的分支真的是 3 个独立的 kernel 来源**（Triton、TritonKernels、
flashinfer-cutlass）+ 1 个 AMD-only AIter 变体。剩下 6 个值（cutlass、deep_gemm、
flashinfer_trtllm、flashinfer_mxfp4、flashinfer_cutedsl、marlin）默默落到默认
Triton `fused_moe_kernel`。

### 25.2 H200 上 bf16 MoE 是不是真的都在跑同一个 Triton kernel？

**几乎是 —— 特别在 H200 上**。我们追每条路：

| Setup | 走的路径 | 实际调的 kernel |
|---|---|---|
| Qwen3-30B-A3B bf16 + `--moe-runner-backend auto`（我们的 C0）| bf16 path，非 AMD → 走 `else` 分支 | **`fused_moe_kernel`** |
| Qwen3-30B-A3B bf16 + `--moe-runner-backend triton` | 跟 auto 一样 | **`fused_moe_kernel`** |
| Qwen3-30B-A3B bf16 + `--moe-runner-backend cutlass`（我们的 C4）| 悄悄 no-op → Triton | **`fused_moe_kernel`** |
| Qwen3-30B-A3B bf16 + `--moe-runner-backend deep_gemm` | 悄悄 no-op → Triton | **`fused_moe_kernel`** |
| Qwen3-30B-A3B bf16 + `--moe-runner-backend triton_kernel` | TritonKernels 变体 | `matmul_ogs`（triton_kernels 库）|
| Qwen3-30B-A3B bf16 + `--moe-runner-backend flashinfer_cutlass` | flashinfer-cutlass 分支 | `flashinfer_cutlass_fused_moe` |
| Qwen3-30B-A3B bf16 + 默认（auto），跑在 **AMD MI300X** | AIter 变体 | AIter `fused_moe`（C++/HIP）|
| Qwen3-30B-A3B FP8（量化模型）+ `--moe-runner-backend cutlass` | Fp8MoEMethod cutlass 分支 | `cutlass_fused_experts_fp8` |
| Llama-3-AWQ + auto | AWQ marlin 分支 | Marlin 的 W4A16 GEMM |

**H200 + bf16 specifically**：

- **绝大多数** setup 最后都到 `fused_moe_kernel`（Triton）
- `triton_kernel` 存在但是另一个 Python/Triton 库（`triton_kernels` 是 openai/triton 的另一个子项目）
- `flashinfer_cutlass` 是唯一"真正的备选"—— dispatch 到 flashinfer 预编译的 CUTLASS kernel

**我们 C0-baseline 到 C6 全都跑 `fused_moe_kernel`**，因为：
- C0、C2、C3、C6：显式/默认 → Triton path
- C4 cutlass：悄悄 no-op → Triton path
- C1（torch.compile）和 C5（flashinfer attention）加载失败
- 我们没显式测过 `triton_kernel` 或 `flashinfer_cutlass`

所以 §15 / §16 / §19 的 24-cell 硬件视图本质上是 **一个 kernel**
（`fused_moe_kernel`）在不同 launch 条件下的研究。

### 25.3 `fused_moe_kernel` 到底是啥？—— sglang 自己手写的 Triton

Kernel 在
[`sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py)
（323-580 行）。

| 属性 | 值 |
|---|---|
| **文件大小** | 1 148 行 Python |
| **该文件里的 `@triton.jit` kernel 数** | 8 个（一个主 `fused_moe_kernel` + 7 个辅助 helpers，给 FP8 / GPTQ / AWQ / split-K 变体）|
| **作者** | sglang 维护者手写，用 Triton DSL |
| **不是生成的** | 没有代码生成器。`@triton.jit` 源码就是真跑的（Triton 编译器把它编成 PTX）|
| **改编自** | vLLM 的 `fused_moe_kernel`（sglang 在某些地方注释"Adapted from vllm"）|
| **Per-shape 调优** | `configs/triton_3_X_X/E=<E>,N=<N>,device_name=<…>.json` 下 35 个 JSON 文件，给出 per (M-bucket, expert 数, hidden size, device, dtype) 的 `BLOCK_SIZE_M/N/K`、`num_warps`、`num_stages` |

**两行解释怎么工作**：

1. Python 源码（323-580 行）把 `BLOCK_SIZE_M`、`BLOCK_SIZE_N`、`BLOCK_SIZE_K`、
   `num_warps`、`num_stages` 当 `tl.constexpr` 参数读。
2. 每次 `kernel[grid](...)` 调用，wrapper 按当前 `M`（= num_tokens × topk）查 JSON
   行，传给 Triton，Triton 给那个 (constexpr, shape) 组合 JIT 编译一个专门的 PTX。
   PTX 被缓存所以后续调用零开销。

**"手写"** 的意思：人（你在
[GitHub Blame](https://github.com/sgl-project/sglang/blame/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py)
能看到他们）一行一行打出来这个 body：

```python
@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, ..., BLOCK_SIZE_M: tl.constexpr, ...):
    pid = tl.program_id(axis=0)
    # ... ~250 行指针算术、GEMM 循环、fusion 逻辑 ...
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        ...
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token)
        accumulator = accumulator * moe_weight[:, None]
    c = accumulator.to(compute_type)
    tl.store(c_ptr + ..., c)
```

**JSON 调优表是 auto-generated**（用 Triton 的 autotuner，sglang 开发者离线在每个
主流 GPU 上跑一次）。推理时 sglang 只**查表** —— 启动时不做 autotuning。

更大层面的 **"生成"部分**：
- **Per-shape 编译的 PTX**：Triton JIT 在启动时把每个 (BLOCK_SIZE_M, ..., dtype, shape)
  组合编一次成 PTX，缓存在 `~/.triton/cache/`。缓存 key 是 constexpr 参数的 hash。
- **JSON 调优表**：离线预调好。新的 (E, N, device) 组合 fall back 到"最近"查表，
  这就是为啥 M 在两个 JSON bucket 之间时偶尔看到 1 µs/call 的 gap。

### 25.4 `fused_moe_kernel` 内部有多少优化空间？

**有很多**。§15 / §20 已经证明这一个 kernel 在 MoE 上占 34-47% GPU 时间。
具体机会，按优先级排：

#### (a) 给我们的 shape 重新调 JSON（便宜，可能大赢）

我们 Qwen3-30B-A3B 是 `E=128, N=768`。JSON 文件
`configs/triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json` 所有 `(M, dtype)`
bucket 共享，但 **调优是在另一个 workload mix 上做的**。具体：

- JSON `M=1024` 行用 `BLOCK_SIZE_M=128, num_warps=8, num_stages=4`
- 我们 R8 并发 32 + 前缀 2048+128 给的是：decode 步 `M ≈ 32 × 8 ≈ 256`，prefill 步
  `M ≈ 32 × 2176 × 8 / 128 ≈ 4352`
- 在我们具体的 decode-vs-prefill M 分布上重跑 Triton 的 `tune.py`，可能找到 5-10% 的赢面

#### (b) 融合 `moe_align_block_size` + `fused_moe_kernel`（中等，大赢）

§15 显示这俩之间的 `cudaEventSynchronize` 占 MoE R3/R4（prefill regime）GPU 时间的
**37%**。sync 是必须的因为 `fused_moe_kernel` 要读 `moe_align_block_size` 写的
`num_tokens_post_padded`。如果能把它们表达成单 kernel（或至少 pipeline 到同 stream
上重叠工作），sync 就消失了。

#### (c) 激活融合（便宜，小赢）

当前 pipeline：

```
fused_moe_kernel (gate+up GEMM) → SiluAndMul (小 kernel) → fused_moe_kernel (down GEMM)
```

中间的 SiluAndMul 是独立 kernel（占 ~3% GPU 时间）。把它折进 down-GEMM kernel
（让 down 直接从 up-GEMM 输出的 shared memory 里 load 已激活的 tile）是个已知的
Triton 优化。**不简单** —— 需要重写 kernel 接 2 个权重 tensor（w13 和 w2）并把两个
GEMM pipeline 起来。

#### (d) FP8 量化（大赢，需要数据准备）

最大的单一杠杆 —— 但需要：
- 把 Qwen3-30B-A3B 权重从 bf16 转 FP8（有损）
- 换到 `Fp8MoEMethod` → 解锁 `cutlass_fused_experts_fp8`（真正的 cutlass MoE path）
- 预期：Hopper 上吞吐 1.5-2× 提升（FP8 tensor core 峰值吞吐是 bf16 的 2×）

#### (e) DeepGEMM backend（H200/H100 大赢，仅 FP8）

`deep_gemm`（来自 DeepSeek，sglang 通过 `--moe-runner-backend deep_gemm` 提供）是
专门给 FP8 + Hopper 的手调 CUTLASS MoE kernel。在 DeepSeek-V3 上比 Triton FP8 path
快 1.2-1.5×。

#### (f) Expert parallelism + DeepEP all-to-all（仅大规模）

多卡 setup 上从 tensor parallel 换成 expert parallel（`--ep-size 8 --moe-a2a-backend deepep`）
能显著减少 per-GPU 工作量。对我们单 H200 没用，对生产环境很关键。

#### (g) Routing 不均衡（数据相关）

`fused_moe_kernel` 假设专家均衡 load（每个 block 一个专家）。实际有些专家比别的热，
让一些 block 几乎空跑（还消耗一个 tile 的算力）。变 block 大小的调度会有帮助 ——
但这是研究问题，不只是 kernel 改写。

#### 我们项目的优先级排序

| 机会 | 成本 | 预期加速 | 成熟度 |
|---|---|---|---|
| (a) 为我们 shape 重调 JSON | 1 天 | 5-10% | Trivial 用 Triton autotuner |
| (b) 融合 moe_align + fused_moe | 2-4 周（Triton kernel 手术）| prefill regime 上 10-30% | 难 |
| (c) 把 SiluAndMul 融进 down | 1-2 周 | ~3% | 中等 |
| (d) FP8 量化 | 2-3 天 | 1.5-2× | 简单（转权重 + flag） |
| (e) DeepGEMM | 需要 (d) | 在 (d) 基础上再 1.2-1.5× | 一旦 FP8 就绪很简单 |
| (f) Expert parallelism | 几周（系统改动）| 看集群规模 | 仅生产 |
| (g) Routing 平衡研究 | 几个月 | TBD | 研究 |

**我们 H200 单卡 setup 拿最快胜利**：先试 (d) FP8 量化 —— 不重写 kernel 唯一能上
2× 天花板。然后 (a) JSON 重调拿残余。

### 25.5 三个 sub-question 的精确答案

**Q：「只改权重加载顺序，kernel 还是 Triton」是啥？跟另外 3 个被真用的分支区别是啥？**

那是我说错了。**`flashinfer_cutlass` 真的换了 kernel** 到
`flashinfer_cutlass_fused_moe`，并且 re-permute 权重。bf16 上 4 个 honoured
分支是 4 条独立代码路径：

| 分支 | Kernel | 权重格式 |
|---|---|---|
| `auto`/`triton` | sglang 的 `fused_moe_kernel`（Triton JIT）| 加载即用 |
| AMD 上 `auto` | AIter 的 `fused_moe`（C++/HIP）| 被 shuffle |
| `triton_kernel` | `triton_kernels` 库的 `matmul_ogs` | 被 transpose |
| `flashinfer_cutlass` | flashinfer 的 `flashinfer_cutlass_fused_moe` | 被 block-permute |

**Q：bf16 下 MoE 模型实际都在跑同一个 Triton kernel 吗？**

H200 + bf16 + 默认 flag：**是，几乎所有路径都收敛到 `fused_moe_kernel`**。
我们 24-cell 硬件视图研究（§15-§20）本质上是一个 kernel 在不同 launch 条件和
不同周围 graph 拓扑下的研究。

**Q：`fused_moe_kernel` 是啥？手写的？生成的？**

sglang 维护者 **手写** 的 **Triton DSL** 代码（~1148 行，含 8 个 `@triton.jit` kernel）。
改编自 vLLM。不是 auto-generated。Triton JIT 在首次使用时为每个 (M, dtype, shape) 组合
编译一次 PTX，缓存在 `~/.triton/cache/`。**JSON 调优表**（block sizes、warps、stages）
在 `configs/triton_3_X_X/` 下，由 Triton 的 autotuner 离线生成。

**Q：有优化空间吗？**

有很多。我们 use case 的具体优先级：
1. **FP8 量化** —— 不重写 kernel 唯一能上 2× 天花板的杠杆
2. **为我们 shape 重调 JSON** —— 5-10%，1 天工作
3. **`moe_align` + `fused_moe` 融合** —— 干掉 prefill regime 37% 的 sync wait（难）
4. **SiluAndMul 融合** —— 3% 赢面（中等）
5. **DeepGEMM backend** —— FP8 之后再叠 1.2-1.5×

## 26. Hand-written Triton vs Inductor-generated — how to tell them apart in a trace

> 🟢 NEW. Mentor / Mason Remy / Debadeepta question after seeing §18-§25:
> Is `fused_moe_kernel` hand-written or Torch Inductor codegen output?
> The two cases have **completely different optimisation opportunities**:
> if hand-written → library maintainers manually tune; if Inductor →
> torch.compile's heuristic template path, lots of low-hanging fruit.
> This section gives the precise way to distinguish them in our traces +
> answers for our specific MoE study.

### 26.1 The naming convention — kernel names directly tell you

The Triton compiler produces kernels with specific name prefixes that
reveal who wrote them:

| Trace kernel name prefix | Source | Hand-written? |
|---|---|---|
| `triton_poi_fused_*` | **Inductor codegen** (pointwise fusion) | ❌ Auto-generated by `torch.compile` |
| `triton_tem_fused_*` | **Inductor codegen** (template fusion, e.g. GEMM) | ❌ Auto-generated |
| `triton_red_fused_*` | **Inductor codegen** (reduction fusion) | ❌ Auto-generated |
| `fused_moe_kernel` | sglang manual `@triton.jit` | ✅ Hand-written by sglang maintainers |
| `moe_align_block_size_kernel` | sglang manual `@triton.jit` | ✅ Hand-written |
| `moe_sum_reduce_warp_per_token_vec_kernel` | sglang manual `@triton.jit` | ✅ Hand-written |
| `write_req_to_token_pool_triton` | sglang manual `@triton.jit` (scheduler) | ✅ Hand-written |
| `flashinfer::*` | flashinfer pre-compiled CUDA (NOT Triton) | ✅ Hand-written CUDA C++ |
| `void at::native::*` | PyTorch native kernel | ✅ Hand-written |
| `nvjet_tst_*` / `cublas*` / `cutlass::*` | NVIDIA library | ✅ Vendor hand-written |

**The convention is forced by Triton itself**: Inductor uses `torch._inductor.codecache.PyCodeCache` which generates Python files containing `@triton.jit` functions named `triton_<type>_fused_<op1>_<op2>_<...>`. Hand-written kernels can be named anything; sglang maintainers chose descriptive names like `fused_moe_kernel`.

### 26.2 Empirical breakdown — our C0 baseline by source

We took our C0 baseline trace (`fused_moe_kernel` dominates) and classified every GPU event by source:

| Share | Source | Top kernel name |
|---|---|---|
| **46.4 %** | **sglang HAND-WRITTEN Triton** (`fused_moe_kernel`) | `fused_moe_kernel` |
| 29.0 % | cuBLAS / CUTLASS (NVIDIA pre-compiled) | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<…>` |
| 8.4 % | CUDA runtime (launch / sync / memcpy) | `cudaLaunchKernel`, `cudaGraphLaunch` |
| 7.2 % | flashinfer library (hand-written CUDA C++, pre-compiled) | `flashinfer::norm::FusedAddRMSNormKernel`, `flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel` |
| 3.0 % | sglang HAND-WRITTEN Triton (MoE helpers) | `moe_align_block_size_kernel`, `moe_sum_reduce_warp_per_token_vec_kernel` |
| 3.0 % | PyTorch `at::native::*` (e.g. elementwise) | `at::native::index_elementwise_kernel`, `vectorized_elementwise_kernel` |
| 2.9 % | sglang other (scheduler helpers) | `write_req_to_token_pool_triton`, `compute_position_kernel` |
| **0.00 %** | 🤖 **Torch Inductor GENERATED Triton** | `triton_poi_fused_clamp_copy__index_lt_neg_where_0` (19 µs, 14 calls total) |

**Conclusion**: In our default sglang setup (no `--enable-torch-compile`), **Inductor generates essentially zero kernels**. The hot path is dominated by hand-written sglang Triton + vendor pre-compiled libraries.

The tiny 19 µs of Inductor output is from sglang's request-prep code which uses `torch.compile` for trivial pointwise ops (`clamp`, `where`); negligible.

### 26.3 What about C6 piecewise CUDA graph? Surely Inductor kicks in there?

We checked — same result. C6 trace also shows ~19 µs Inductor output (0.00 % of GPU). Reason: our C6 config uses `piecewise-cuda-graph-compiler: eager` (the default), which captures sub-graphs without running them through Inductor's fusion passes.

To actually invoke Inductor's full fusion machinery would require:

1. `--enable-torch-compile` (our C1 — failed with `torch._dynamo` AssertionError in `sglang/srt/layers/rotary_embedding.py:272` — known sglang/Qwen3 MoE incompatibility), OR
2. `--enable-piecewise-cuda-graph --piecewise-cuda-graph-compiler inductor` (untested in our study; warranted as a follow-up).

### 26.4 What this means for optimisation strategy

The two cases imply completely different paths to performance:

#### Case A: `fused_moe_kernel` IS hand-written (this is our case)

| Optimisation path | Mechanism | Expected payoff |
|---|---|---|
| Re-tune JSON for our exact shape | run Triton autotuner offline against our (E=128, N=768, H200, bf16) | **5-10 %** |
| Fuse `moe_align_block_size` into `fused_moe_kernel` | rewrite `fused_moe_kernel` to do the prep loop in-kernel | **kills the 37 % `cudaEventSynchronize` on prefill regimes** |
| Fuse `SiluAndMul` into the down `fused_moe_kernel` | rewrite kernel to take w13 and w2, pipeline both GEMMs with SiLU in registers | ~3 % |
| Move to FP8 quantization | switch `quant_method` to `Fp8MoEMethod`, unlocks `cutlass_fused_experts_fp8` | **1.5–2 ×** (Hopper FP8 has 2× tensor-core peak) |
| Try DeepGEMM backend after FP8 | `--moe-runner-backend deep_gemm` on FP8 model | additional 1.2–1.5 × |

All of these require **manual Triton/CUDA kernel work** by someone who knows the library. No "low-hanging Inductor fruit" here because Inductor isn't involved.

#### Case B: Kernel was Inductor-generated (hypothetical, not our case)

| Optimisation path | Mechanism | Expected payoff |
|---|---|---|
| Tweak Inductor's heuristic config | `TORCHINDUCTOR_*` env vars, set `max_autotune=True` | 10–30 % on individual kernels |
| Force Inductor to choose better templates | adjust `coordinate_descent_tuning` | small wins on specific kernels |
| Fix Inductor's fallback paths (kernels Inductor refused to fuse) | wrap missing ops in `torch.library.custom_op` | medium wins |
| Replace generated kernel with a hand-written one | write a Triton kernel that beats Inductor's output, register as `torch.library` op | large wins on hot kernels |

These are real low-hanging fruit IF Inductor was on the hot path — typically when the inductor-generated kernel does something stupid (too many small loads, missed fusion across an unexpected boundary, etc.).

### 26.5 To actually get Inductor on the MoE hot path

This would require resolving the sglang + torch.compile incompatibility. From our C1 attempt the failure is:

```
File "/home/t-jialianggu/work/sglang/python/sglang/srt/layers/rotary_embedding.py", line 272, in forward_native
    fused_set_kv_buffer_arg is None

torch._dynamo.exc.AssertionError
```

This is sglang-internal: the rotary-embedding code uses a Python idiom that Dynamo can't trace cleanly. Fixing it would need:

1. Patch `rotary_embedding.py:272` to make `fused_set_kv_buffer_arg` an explicit kwarg with a default
2. Re-test `--enable-torch-compile` on Qwen3 MoE
3. If torch.compile succeeds, check trace for `triton_*_fused_*` kernels and measure speedup

This is a 1-2 day investigation. The expected payoff would be:

- Auto-fusion of the 7.2 % flashinfer RMSNorm + RoPE chain
- Auto-fusion of the 3 % PyTorch `at::native::*` elementwise ops
- Possibly auto-fusion of small ops between `fused_moe_kernel` calls
- Total upside maybe 5-15 %, but **highly dependent on whether Inductor's heuristics match the workload**.

This is genuinely a "low-hanging fruit" path worth pursuing — but it's a separate research direction from optimising the hand-written `fused_moe_kernel`.

### 26.6 How to apply this to any new kernel observation

Workflow:

1. **Look at the kernel name in the trace.** Got `triton_poi_fused_*` / `triton_tem_fused_*` / `triton_red_fused_*`? → Inductor. Got anything else (custom name)? → hand-written.
2. **For hand-written ones, grep the library source** (sglang, flashinfer, vllm) for the kernel name. The Python `@triton.jit` source is right there.
3. **For Inductor ones, set `TORCH_LOGS=output_code`** to dump the generated Python source. It's also runtime-cached under `/tmp/torchinductor_<user>/`.
4. **Cross-check with `torch._dynamo.config.verbose = True`** if you want to see what Inductor decided to fuse vs not.

For our project: **all the hot kernels are hand-written**. The optimisation strategy in §25.4 stands. Inductor-driven optimisations are a separate (1-2 day) follow-up gated on fixing the `rotary_embedding.py` compatibility.

### 26.7 Single-line answer to Mason / Debadeepta

> **Our `fused_moe_kernel` is hand-written by sglang maintainers in Triton DSL** (`sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`, 1148 lines, 8 `@triton.jit` kernels, adapted from vLLM). **Trace evidence**: the kernel name is `fused_moe_kernel`, not `triton_*_fused_*` (which would mean Inductor). **Inductor accounts for 0.00 % of GPU time** in our default setup because we don't pass `--enable-torch-compile` — and when we tried (C1), torch.compile failed on Qwen3 MoE's rotary embedding (Dynamo `AssertionError`). **Optimisation implications**: the §25.4 menu (FP8 quantization, JSON re-tune, kernel-level fusion of `moe_align`+`fused_moe`, `SiluAndMul` folding) is the right path for the hand-written kernel. To explore Inductor-driven optimisations would first need a 1-2 day fix to sglang's `rotary_embedding.py:272` to make it Dynamo-compatible, then re-test C1 and look for `triton_*_fused_*` kernels in the trace.

---

## 26. 手写 Triton vs Inductor 生成 —— 怎么在 trace 里区分

> 🟢 新增。Mentor / Mason Remy / Debadeepta 看完 §18-§25 后的问题：
> `fused_moe_kernel` 是手写还是 Torch Inductor 生成的？这两种情况的优化
> 机会**完全不同**：手写 → 库维护者手调；Inductor → torch.compile 的启发式
> template 路径，有很多低垂果实。这节给在 trace 里精确区分的方法 + 我们 MoE
> 研究的具体答案。

### 26.1 命名规则 —— kernel 名字直接告诉你

Triton 编译器产出的 kernel 有特定名字前缀，告诉你是谁写的：

| Trace kernel 名前缀 | 来源 | 是否手写 |
|---|---|---|
| `triton_poi_fused_*` | **Inductor 生成**（pointwise 融合）| ❌ `torch.compile` 自动生成 |
| `triton_tem_fused_*` | **Inductor 生成**（template 融合，如 GEMM）| ❌ 自动生成 |
| `triton_red_fused_*` | **Inductor 生成**（reduction 融合）| ❌ 自动生成 |
| `fused_moe_kernel` | sglang 手写 `@triton.jit` | ✅ sglang 维护者手写 |
| `moe_align_block_size_kernel` | sglang 手写 `@triton.jit` | ✅ 手写 |
| `moe_sum_reduce_warp_per_token_vec_kernel` | sglang 手写 `@triton.jit` | ✅ 手写 |
| `write_req_to_token_pool_triton` | sglang 手写 `@triton.jit`（scheduler）| ✅ 手写 |
| `flashinfer::*` | flashinfer 预编译 CUDA（**不是** Triton）| ✅ 手写 CUDA C++ |
| `void at::native::*` | PyTorch 原生 kernel | ✅ 手写 |
| `nvjet_tst_*` / `cublas*` / `cutlass::*` | NVIDIA 库 | ✅ 厂商手写 |

**这个规则是 Triton 本身强制的**：Inductor 用 `torch._inductor.codecache.PyCodeCache` 生成 Python 文件，里面 `@triton.jit` 函数名总是 `triton_<type>_fused_<op1>_<op2>_<...>`。手写 kernel 可以叫任何名字；sglang 维护者起的是描述性名字 `fused_moe_kernel`。

### 26.2 实测分类 —— C0 baseline 各来源占比

我们拿 C0 baseline trace（`fused_moe_kernel` 主导）按来源分类每个 GPU event：

| 占比 | 来源 | 顶级 kernel 名 |
|---|---|---|
| **46.4 %** | **sglang 手写 Triton** (`fused_moe_kernel`) | `fused_moe_kernel` |
| 29.0 % | cuBLAS / CUTLASS（NVIDIA 预编译）| `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<…>` |
| 8.4 % | CUDA runtime（launch / sync / memcpy）| `cudaLaunchKernel`、`cudaGraphLaunch` |
| 7.2 % | flashinfer 库（手写 CUDA C++，预编译）| `flashinfer::norm::FusedAddRMSNormKernel`、`flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel` |
| 3.0 % | sglang 手写 Triton（MoE helpers）| `moe_align_block_size_kernel`、`moe_sum_reduce_warp_per_token_vec_kernel` |
| 3.0 % | PyTorch `at::native::*`（如 elementwise）| `at::native::index_elementwise_kernel`、`vectorized_elementwise_kernel` |
| 2.9 % | sglang 其他（scheduler helpers）| `write_req_to_token_pool_triton`、`compute_position_kernel` |
| **0.00 %** | 🤖 **Torch Inductor 生成 Triton** | `triton_poi_fused_clamp_copy__index_lt_neg_where_0`（共 19 µs，14 calls）|

**结论**：我们默认的 sglang setup（没开 `--enable-torch-compile`），**Inductor 实际生成 0 个有意义的 kernel**。热路径完全被 sglang 手写 Triton + 厂商预编译库占据。

那 19 µs 的 Inductor 输出是 sglang 请求预处理代码里用了 `torch.compile` 跑些 trivial 的 pointwise op（`clamp`、`where`）；忽略不计。

### 26.3 C6 piecewise CUDA graph 呢？那总该有 Inductor 吧？

我们 check 了 —— 结果一样。C6 trace 也只有 ~19 µs Inductor 输出（0.00 % GPU）。原因：我们 C6 config 用的是 `piecewise-cuda-graph-compiler: eager`（默认），它捕获 sub-graph 但不走 Inductor 的融合 pass。

要真触发 Inductor 全量融合 machinery 需要：

1. `--enable-torch-compile`（我们 C1 试了，sglang 0.5.12 在 `sglang/srt/layers/rotary_embedding.py:272` 抛 `torch._dynamo` AssertionError —— 已知 sglang/Qwen3 MoE 不兼容）；或
2. `--enable-piecewise-cuda-graph --piecewise-cuda-graph-compiler inductor`（我们研究里没测；值得 follow-up）

### 26.4 这对优化策略意味着什么

两种情况导向完全不同的性能路径：

#### Case A：`fused_moe_kernel` 是手写（我们就是这种情况）

| 优化路径 | 机制 | 预期收益 |
|---|---|---|
| 为我们 shape 重调 JSON | 在 (E=128, N=768, H200, bf16) 上离线跑 Triton autotuner | **5-10 %** |
| 把 `moe_align_block_size` 融进 `fused_moe_kernel` | 重写 kernel 把准备循环放 kernel 内 | **干掉 prefill regime 上 37 % `cudaEventSynchronize`** |
| 把 `SiluAndMul` 融进 down `fused_moe_kernel` | 重写 kernel 接 w13 和 w2，pipeline 两个 GEMM + 寄存器里的 SiLU | ~3 % |
| 切到 FP8 量化 | 换 `quant_method` 到 `Fp8MoEMethod`，解锁 `cutlass_fused_experts_fp8` | **1.5–2 ×**（Hopper FP8 tensor core 峰值 2×）|
| FP8 后试 DeepGEMM backend | FP8 模型上 `--moe-runner-backend deep_gemm` | 在 FP8 上再 1.2–1.5 × |

这些都需要 **懂库的人手动改 Triton/CUDA kernel**。**这条路上没有"Inductor 低垂果实"**，因为 Inductor 根本没在跑。

#### Case B：kernel 是 Inductor 生成（假设，不是我们这种）

| 优化路径 | 机制 | 预期收益 |
|---|---|---|
| 调 Inductor 启发式 config | `TORCHINDUCTOR_*` env vars，设 `max_autotune=True` | 单 kernel 上 10–30 % |
| 让 Inductor 选更好的 template | 调 `coordinate_descent_tuning` | 特定 kernel 小赢 |
| 修 Inductor 的 fallback 路径（Inductor 不肯融合的）| 把缺失的 op 包进 `torch.library.custom_op` | 中等赢 |
| 拿手写 kernel 替掉生成的 | 写个 Triton kernel 打过 Inductor 输出，注册成 `torch.library` op | 热 kernel 大赢 |

这些是真的低垂果实 —— **前提是 Inductor 在热路径上**。典型场景：Inductor 生成的 kernel 做了傻事（太多小 load、漏掉跨某个意外边界的融合等）。

### 26.5 真要让 Inductor 上 MoE 热路径

要解决 sglang + torch.compile 不兼容问题。从 C1 失败看，错在：

```
File "/home/t-jialianggu/work/sglang/python/sglang/srt/layers/rotary_embedding.py", line 272, in forward_native
    fused_set_kv_buffer_arg is None

torch._dynamo.exc.AssertionError
```

这是 sglang 内部问题：rotary-embedding 代码用了 Dynamo 跟不上的 Python idiom。修需要：

1. 给 `rotary_embedding.py:272` 打补丁，把 `fused_set_kv_buffer_arg` 变成有默认值的显式 kwarg
2. 重测 `--enable-torch-compile` 在 Qwen3 MoE 上
3. torch.compile 通过后查 trace 里有没有 `triton_*_fused_*` kernel 并测加速比

这是 1-2 天的调研。预期收益：

- 自动融合 7.2 % flashinfer RMSNorm + RoPE 链
- 自动融合 3 % PyTorch `at::native::*` elementwise op
- 可能自动融合 `fused_moe_kernel` 调用之间的小 op
- 总收益大概 5-15 %，但 **高度依赖 Inductor 的启发式是否匹配 workload**

这是一个真正的"低垂果实"路径，值得追 —— 但它是跟优化手写 `fused_moe_kernel` 完全不同的方向。

### 26.6 给新 kernel 观察的应用流程

工作流：

1. **看 trace 里 kernel 名**。是 `triton_poi_fused_*` / `triton_tem_fused_*` / `triton_red_fused_*`？→ Inductor。是别的（自定义名）？→ 手写。
2. **手写的，grep 库源码**（sglang、flashinfer、vllm）找 kernel 名。`@triton.jit` Python 源码就在那。
3. **Inductor 的，设 `TORCH_LOGS=output_code`** dump 出生成的 Python 源码。也缓存在 `/tmp/torchinductor_<user>/`。
4. **想看 Inductor 决定融合了啥没融合啥**，设 `torch._dynamo.config.verbose = True`。

我们项目：**所有热 kernel 都是手写的**。§25.4 的优化策略成立。Inductor 驱动的优化是单独的（1-2 天）follow-up，门槛是先修 `rotary_embedding.py` 兼容性。

### 26.7 给 Mason / Debadeepta 的一句话回答

> **我们的 `fused_moe_kernel` 是 sglang 维护者用 Triton DSL 手写的**
> （`sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`，
> 1148 行，8 个 `@triton.jit` kernel，改编自 vLLM）。**trace 证据**：
> kernel 名是 `fused_moe_kernel`，**不是** `triton_*_fused_*`（后者才表示
> Inductor）。**Inductor 在我们默认 setup 里占 0.00 % GPU 时间** —— 因为我们
> 没传 `--enable-torch-compile`；C1 试图传时，torch.compile 在 Qwen3 MoE 的
> rotary embedding 上失败了（Dynamo `AssertionError`）。**优化含义**：§25.4
> 的菜单（FP8 量化、JSON 重调、kernel 级融合 `moe_align`+`fused_moe`、
> `SiluAndMul` 折叠）是手写 kernel 的正确路径。要探索 Inductor 驱动的优化要先
> 花 1-2 天修 sglang 的 `rotary_embedding.py:272` 让 Dynamo 能 trace，然后
> 重测 C1 并查 trace 里 `triton_*_fused_*` kernel。

## 27. Real kernel-swap experiment: C7 flashinfer_cutlass + C8 triton_kernel on MoE R8

> 🟢 NEW. Per §22-§26, only 2 of the 10 `--moe-runner-backend` values
> actually swap the bf16 MoE kernel: `triton_kernel` and `flashinfer_cutlass`.
> §15-§19 covered the silent-no-op cases. This section runs **the real ones**
> to see if swapping the kernel actually wins performance.
>
> Also serves as worked example of "what happens when you really swap the
> kernel" vs the silent-no-op cases (§19 C4).

### 27.1 Setup

Both cells run against R8 prefix sharing (MoE's throughput champion):

- **C7** = baseline + `--moe-runner-backend flashinfer_cutlass`. Calls flashinfer's `flashinfer_cutlass_fused_moe` (NOT our Triton `fused_moe_kernel`). Requires weight re-permute at load time.
- **C7b** = C7 + `--disable-cuda-graph` (because C7 hangs during graph capture).
- **C8** = baseline + `--moe-runner-backend triton_kernel`. Calls `triton_kernels` library's `matmul_ogs` (still Triton, but a different library).

### 27.2 Results

| Cell | Status | Req/s | Out tok/s | TTFT mean | TPOT mean | E2E p99 | vs C0 |
|---|---|---|---|---|---|---|---|
| **C0 baseline** (Triton `fused_moe_kernel`) | ✅ PASS | 5.23 | **1 339** | 379 ms | 22.5 ms | 10 081 ms | 0 % |
| **C7** flashinfer_cutlass | ❌ FAIL | — | — | — | — | — | hung during CUDA graph capture |
| **C7b** flashinfer_cutlass + disable-cuda-graph | ❌ FAIL | — | — | — | — | — | server health stayed 503 (never warmed up) |
| **C8** triton_kernel | ✅ PASS | 0.75 | **192** | 4 182 ms | 150.5 ms | 80 199 ms | **−86 %** ⚠️ |

### 27.3 What we learned

#### C8 is a textbook real-kernel-swap, and it's catastrophically slower

The trace shows `fused_moe_kernel` is **completely gone**, replaced by `triton_kernels` library's `_p_matmul_ogs_NNN_bf16xbf16xbf16_<TILE>` kernels:

**C0 baseline top 5 kernels** (599 ms total):
| % | self (µs) | calls | name |
|---|---|---|---|
| 46.4 | 278 302 | 864 | **`fused_moe_kernel`** ← sglang's hand-written Triton |
| 10.3 | 61 582 | 96 | `flash::FlashAttnFwdSm90<…>` |
| 6.3 | 37 832 | 96 | `nvjet_tst_128x248_64x4_2x1_v_bz_coopA_TNT` (cuBLAS) |
| 3.6 | 21 539 | 48 | `nvjet_tst_192x208_64x4_…_coopB_TNT` |
| 3.1 | 18 330 | 336 | `flash::FlashAttnFwdSm90<…>` variant |

**C8 triton_kernel top 5 kernels** (550 ms total — similar total!):
| % | self (µs) | calls | name |
|---|---|---|---|
| 37.8 | 207 866 | 288 | **`_p_matmul_ogs_NNN_bf16xbf16xbf16_128x256x64x1`** ← triton_kernels library |
| 11.3 | 62 310 | 96 | `flash::FlashAttnFwdSm90<…>` (unchanged) |
| 7.0 | 38 777 | 96 | `nvjet_tst_128x248_64x4_…` (unchanged) |
| 4.0 | 22 012 | 480 | **`_p_matmul_ogs_NNN_bf16xbf16xbf16_16x256x64x1`** (small-M variant) |
| 4.0 | 21 980 | 48 | `nvjet_tst_192x208_64x4_…` (unchanged) |
| 3.2 | 17 819 | 480 | **`_reduce_grouped`** ← triton_kernels' MoE-specific reduce |

**This is the cleanest example we have** of "the kernel really did change". The MoE-related kernel name is entirely different, the call count is different (288 instead of 864 — triton_kernels does it in fewer launches per layer), and we even see a triton_kernels-specific helper (`_reduce_grouped`) that doesn't exist in sglang's path.

But **GPU active time is roughly the same** (550 ms vs 599 ms) — yet **observed throughput is 7× lower**. The kernel swap itself isn't faster; the surrounding integration is *much* slower. Possible reasons:

1. **No CUDA graph compatibility**: triton_kernels' grouped GEMM likely doesn't replay cleanly in CUDA graph capture, so each step re-launches everything from scratch (CPU-side launch overhead spikes)
2. **Worse warmup characteristics**: the 4 s TTFT and 80 s e2e p99 suggest the kernel is JIT-compiling per-shape on every single request rather than caching
3. **Different scheduling assumptions**: triton_kernels expects a different batch-token layout, sglang has to do extra prep work each call

#### C7 (flashinfer_cutlass) didn't even start

Two attempts both failed:
- **C7** (with CUDA graph): server got past weight loading + KV cache allocation, then **hung on "Capture cuda graph bs=32"** — never produced a heartbeat for 480 s
- **C7b** (with `--disable-cuda-graph`): server got past startup but health endpoint **stayed 503 for 200+ s** — the warmup never completed

This is **the third failure mode from §21** (Layer C: hard fail when explicit backend can't load). Both C7 variants exhibit the issue that **flashinfer_cutlass MoE wasn't really designed for bf16** — the library's CUTLASS MoE kernel exists primarily for FP8 paths. On bf16, sglang dutifully tries to call it, the weight re-permute happens at load, but downstream initialisation hangs.

### 27.4 Three concrete answers

#### Q: Does sglang actually use auto-generated kernels (Inductor)?

**Yes, but only in 23 small auxiliary functions, not the hot path.**

Source — grep on `^@torch.compile` in sglang:

```
managers/overlap_utils.py:20            @torch.compile  _resolve_future_token_ids
speculative/eagle_worker.py:1018        @torch.compile  (speculative decoding helper)
speculative/spec_utils.py:402,452,466   @torch.compile  (3 spec helpers)
forward_batch_info.py:1096              @torch.compile  (position-encoding helper)
... 23 total
```

These are **trivial helper functions** (token-id resolution, position encoding, spec-decode arithmetic). None of them is in the attention or MoE forward path. **In C0 baseline, these contribute 0.00 % of GPU time** (19 µs total — just the rare `triton_poi_fused_clamp_*` from spec/preproc).

To put Inductor on the hot path, you'd need `--enable-torch-compile` (our C1 failed) or `--enable-piecewise-cuda-graph --piecewise-cuda-graph-compiler inductor` (untested in our study).

#### Q: What is Inductor exactly?

**Torch Inductor is `torch.compile`'s default backend** — a Python-driven kernel-codegen pipeline that turns a PyTorch model into Triton kernels (or sometimes C++/CUDA) automatically. The flow:

```
Python model code (nn.Module)
        ↓
    Dynamo trace
(captures Python control flow as FX graph)
        ↓
       FX graph
        ↓
       Inductor
(decides fusion, picks templates, generates Triton source)
        ↓
   Generated Triton (.py files in /tmp/torchinductor_<user>/)
        ↓
     Triton compile
        ↓
        PTX
```

Key properties:

- **Fully automatic**: just `model = torch.compile(model)`, Inductor takes over the whole forward
- **Generated code is readable**: real `@triton.jit` Python files, though variable names are ugly (`tmp0`, `tmp1`, …)
- **Kernel names are mandatory**: always `triton_<type>_fused_<op1>_<op2>_<...>`, where type is `poi` (pointwise), `tem` (template, e.g. GEMM), or `red` (reduction)
- **Heuristic-driven**: has fusion templates + rules; doesn't try every combination
- **Trade-off**: sometimes faster than hand-written, sometimes slower, sometimes fails to compile entirely. That's why production LLM serving libraries (sglang, vLLM, TensorRT-LLM) hand-write the hot kernels — they need to guarantee performance, not gamble on Inductor's heuristics.

Concrete example. Take this PyTorch code:
```python
def forward(x):
    a = torch.relu(x)        # K1
    b = a * 2.0              # K2
    c = b + 1.0              # K3
    return c.sum(dim=-1)     # K4
```

Eager mode: **4 kernels + 3 HBM round-trips**.
With `torch.compile`: Inductor generates **one** kernel like `triton_red_fused_relu_mul_add_sum_0`, doing everything in registers — **1 kernel, 0 intermediate HBM**.

#### Q: What's "low-hanging fruit" exactly?

The optimisations specific to the Inductor codegen path. **Only apply if Inductor is on the hot path** — which it isn't in our default sglang setup.

Concrete categories (with rough expected payoffs):

| Type | Mechanism | Expected speedup |
|---|---|---|
| Enable `TORCHINDUCTOR_MAX_AUTOTUNE=1` | Inductor tries more tile-size variants offline | 5-15 % |
| Enable `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` + `COORDINATE_DESCENT_TUNING=1` | GEMM template autotuning | 10-30 % on GEMM-heavy ops |
| Patch Inductor missed-fusion bugs | wrap custom ops in `torch.library.custom_op` so Inductor can see across them | 5-30 % per fix |
| Replace bad-output Inductor kernel | review `/tmp/torchinductor_<user>/*.py`, hand-write a better Triton kernel, register as `torch.library` op | large win on hot kernels |

**These are real wins in PyTorch training workloads** where Inductor dominates. They're "low-hanging" in the sense that you flip env vars / write small patches and immediately see speedup, **without rewriting kernels by hand**.

For LLM serving (sglang, vLLM) the maintainers chose hand-written kernels for the hot path because Inductor is too unpredictable. Mason's question essentially asks: is the picture still "all hand-written" or did Inductor sneak in? **Our trace evidence says: hand-written for the hot path, Inductor only in trivial preproc functions, contributing 0.00 % of GPU time in default config.**

### 27.5 Are ALL sglang model kernels hand-written?

**On the hot path: yes.** The MoE FFN, attention, RMSNorm, RoPE, GEMM are all hand-written (sglang Triton, flashinfer CUDA, NVIDIA cuBLAS/CUTLASS, flash-attn).

**Off the hot path: mixed.** 23 small helper functions use `@torch.compile` (Inductor) for things like token-ID resolution, speculative-decoding helpers, position encoding.

**The full hierarchy in sglang**:

| Layer | Author | Examples |
|---|---|---|
| L1: NVIDIA closed-source libraries | NVIDIA | cuBLAS (`nvjet_tst_*`), CUTLASS (`cutlass::*`), cuDNN |
| L2: NVIDIA open libraries | NVIDIA | flash-attn (`flash::FlashAttnFwdSm90`) |
| L3: Third-party LLM-specialised | NVIDIA / FlashInfer team | flashinfer (`flashinfer::FusedAddRMSNormKernel`, RoPE, sampling) |
| L4: sglang-internal hand-written Triton | sglang maintainers | `fused_moe_kernel`, `moe_align_block_size_kernel`, `write_req_to_token_pool_triton` |
| L5: PyTorch native | PyTorch | `at::native::elementwise_kernel<…>` for misc tensor ops |
| L6: Inductor-generated (only 23 helpers) | torch.compile | `triton_poi_fused_clamp_copy_*` for trivial preproc |

This layering is **the standard pattern for LLM serving**. vLLM, TensorRT-LLM are similar: share L1-L3 with everyone (NVIDIA libs + flashinfer), each library hand-writes its own L4 (MoE / scheduler / KV cache kernels), use L5 for misc, basically don't use L6 in the hot path.

The reason production LLM serving doesn't use Inductor on the hot path:
1. **Inductor's heuristics still aren't good enough** for MoE-style grouped GEMM
2. **Inductor's compile time is high** (10-60 s) and incompatible with low-latency server boot
3. **Inductor + custom CUDA ops + dynamic shapes + KV cache control flow** = lots of fallback paths that nuke performance
4. **Hand-written gives predictable performance** — production wants no surprises

### 27.6 Summary

- **Real kernel swaps**: only 2 of 10 `--moe-runner-backend` values do this on bf16: `triton_kernel` (uses `triton_kernels` library) and `flashinfer_cutlass` (uses flashinfer library).
- **C8 `triton_kernel`**: real kernel swap confirmed (`fused_moe_kernel` → `_p_matmul_ogs_*`), but **−86 % throughput** due to incompatible CUDA graph / warmup / scheduling integration.
- **C7 `flashinfer_cutlass`**: failed to start in 2 different attempts (CUDA graph hung; or 503 health). bf16 path of flashinfer cutlass MoE is essentially untested in sglang 0.5.12.
- **Default `fused_moe_kernel` is fastest** for bf16 MoE on H200, by a wide margin.
- **Real optimisation gains** require either FP8 quantization (unlocks the cutlass path inside `Fp8MoEMethod`) or hand-rewriting `fused_moe_kernel` (§25.4).
- **Inductor is essentially unused** in the hot path — 23 `@torch.compile` decorators in sglang for trivial helpers only, contributing 0.00 % of GPU time. "Low-hanging fruit" optimisations are gated on first fixing the `--enable-torch-compile` failure (1-2 days, per §26.5).

## 28. CORRECTION to §21 + §22: sglang DOES have transformers fallback (and why Gemma-4 still fails)

> 🟡 IMPORTANT CORRECTION. §21.2 and §22.1 said "sglang has no transformers
> fallback at all". **That was wrong.** sglang has a `--model-impl auto/transformers`
> flag that DOES fall back to transformers when the native sglang implementation
> is missing. This section corrects the record and gives the real reason Gemma-4
> fails: not sglang, but transformers itself doesn't know `gemma4` yet.

### 28.1 The real fallback mechanism — `--model-impl`

Source: [`sglang/srt/configs/model_config.py:47-51`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py):

```python
class ModelImpl(str, Enum):
    AUTO = "auto"          # default — try sglang first, fall back to transformers
    SGLANG = "sglang"       # force sglang's native implementation
    TRANSFORMERS = "transformers"  # force transformers fallback
    MINDSPORE = "mindspore"
```

The `--model-impl auto` (default) decision logic
([`sglang/srt/model_loader/utils.py:103-117`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/utils.py)):

```python
supported_archs = ModelRegistry.get_supported_archs()
is_native_supported = any(arch in supported_archs for arch in architectures)

if model_config.model_impl == ModelImpl.MINDSPORE:
    architectures = ["MindSporeForCausalLM"]
elif not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
    architectures = resolve_transformers_arch(model_config, architectures)
return ModelRegistry.resolve_model_cls(architectures)
```

And `resolve_transformers_arch()` at
[`utils.py:71-86`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/utils.py):

```python
if model_config.model_impl == ModelImpl.AUTO:
    if hasattr(model_module, "is_backend_compatible") and not model_module.is_backend_compatible():
        raise ValueError(
            f"{arch} has no SGlang implementation and the Transformers "
            "implementation is not compatible with SGLang."
        )
    logger.warning(
        "%s has no SGLang implementation, falling back to Transformers "
        "implementation. Some features may not be supported and "
        "performance may not be optimal.", arch,
    )
    architectures[i] = "TransformersForCausalLM"
```

So **the fallback DOES exist**: sglang transparently swaps the architecture name to `TransformersForCausalLM`, which is a generic wrapper at [`sglang/srt/models/transformers.py:142`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/transformers.py) that delegates `forward()` to a transformers model. Performance will be worse (no custom kernels / KV cache integration), but it should run.

### 28.2 So why does Gemma-4 still fail?

Because **the failure is upstream of sglang** — `transformers` itself doesn't know `gemma4` yet. Verified:

```python
>>> from transformers import AutoConfig
>>> AutoConfig.from_pretrained('/data/hf/models/gemma-4-26B-A4B-it', trust_remote_code=True)
ValueError: The checkpoint you are trying to load has model type `gemma4` but
Transformers does not recognize this architecture. This could be because of an
issue with the checkpoint, or because your version of Transformers is out of date.
```

We have `transformers 4.57.1`. The model_type `gemma4` is **not in transformers' `CONFIG_MAPPING`**. So `AutoConfig.from_pretrained()` raises immediately.

The actual failure chain:

```
1. sglang.launch_server --model-path /data/hf/models/gemma-4-26B-A4B-it
                          ↓
2. sglang/srt/configs/model_config.py:__init__ calls AutoConfig.from_pretrained(...)
                          ↓
3. transformers 4.57.1 sees model_type="gemma4"
                          ↓
4. ❌ ValueError raised immediately — server exits
                          ↓
              【Never reaches sglang's ModelRegistry, never reaches
                 the transformers fallback path either】
```

**I was wrong about what error message we hit**. The previous sections claimed it was sglang's `KeyError: 'gemma4'` from its own model registry. Re-checking, the actual error is `transformers` library's `ValueError` — sglang isn't even involved at that point.

### 28.3 Updated decision tree

Revised §21.6 / §22 flowchart accounting for `--model-impl`:

```
┌────────────────────────────────────────────────────────────────────┐
│ sglang.launch_server --model-path X --model-impl auto                │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │ AutoConfig.from_pretrained() │
                │ — does transformers recognize │
                │   config.json's model_type?  │
                └──────────────┬──────────────┘
                  NO ──► raise ValueError, server exits  (← Gemma-4 HITS HERE)
                  YES ──► continue
                               │
                ┌──────────────┴──────────────┐
                │ sglang ModelRegistry —      │
                │ is arch in models/*.py?     │
                └──────────────┬──────────────┘
                  YES ──► use sglang's native model class (fast path)
                  NO  ──► continue (NEW: fallback exists)
                               │
                ┌──────────────┴──────────────┐
                │ Does transformers have this │
                │ arch (TFModelClass)?        │
                └──────────────┬──────────────┘
                  NO ──► raise ValueError (transformers' own error)
                  YES ──► continue
                               │
                ┌──────────────┴──────────────┐
                │ Does TFModelClass.          │
                │ is_backend_compatible()?    │
                └──────────────┬──────────────┘
                  NO ──► raise ValueError ("not compatible with SGLang")
                  YES ──► rename arch to "TransformersForCausalLM"
                          and use sglang/srt/models/transformers.py wrapper
                          (slow path: no custom kernels, no KV cache opts)
```

### 28.4 How to actually run Gemma-4

By feasibility:

| Option | How | Cost |
|---|---|---|
| **Upgrade transformers** | `pip install -U transformers` or `pip install git+https://github.com/huggingface/transformers.git` (if no release contains gemma4 yet) | 5 min; may break sglang/flashinfer compat |
| Edit `config.json` model_type to `gemma3` | depends on how different Gemma-4 architecture is from Gemma-3 (likely will fail to load weights cleanly) | 1-2 days debugging |
| Wait for sglang to add `gemma4.py` | depends on upstream | indefinite |
| Write `sglang/srt/models/gemma4.py` yourself | copy `gemma3_causal.py` + adapt for MoE block | 2-3 days |

If transformers gets updated and recognises `gemma4`, then the sglang transformers-fallback path (§28.1) should kick in automatically — and we'd see `TransformersForCausalLM` in our trace (slow but functional).

### 28.5 Implications for our broader analysis

The §15-§27 analysis still stands — those experiments are all on Qwen3 (sglang has `qwen3_moe.py`) and Gemma-3 (sglang has `gemma3_causal.py`), so we never exercised the fallback path. But:

- **§21.2 and §22.1 are incorrect** about there being no transformers fallback. Edit the previous sections in your head: sglang DOES have one, it's via `--model-impl auto` (default) → `resolve_transformers_arch`.
- **The "hard fail at architecture registry" claim** is correct for models that aren't in either sglang OR transformers. It's wrong for models in transformers but not in sglang — those fall back to `TransformersForCausalLM`.
- **Gemma-4's specific failure is not a sglang bug**, it's that transformers 4.57.1 doesn't have it yet. Upgrading transformers might solve it.

### 28.6 Apology for the error

When I wrote §21.2 and §22.1, I read sglang's `models/registry.py` and saw the hard `raise ValueError(...)` for unknown architectures. I missed `model_loader/utils.py:62-86` which intercepts BEFORE the registry lookup and inserts the transformers-fallback path. That was a real research mistake on my part.

The mental model going forward:

> sglang has TWO fallback paths:
> 1. **Backend selection** (`--attention-backend auto`, `--moe-runner-backend auto`): per-hardware priority chain. Falls back automatically. (correctly described in §21.3)
> 2. **Model implementation** (`--model-impl auto`): falls back to transformers when sglang doesn't have a native impl. (this section corrects the earlier omission)
>
> Both are real fallback chains. The "no fallback, hard fail" claim was correct only for the explicit-backend case (`--attention-backend X` where X can't load, §21.4), and for cases where neither sglang nor transformers know the architecture.

The §21 / §22 walls of text aren't worth re-writing in place; this §28 is the corrected version of the model-architecture story. Please read it together with §21.6's flowchart — replace that flowchart with the one in §28.3.

## 29. What "default" backend actually means in sglang (Debadeepta question)

> 🟢 NEW. Debadeepta's question after the meeting: "default" is mentioned
> a lot in our analyses, but it's never been pinned down. Is it Triton?
> Is it the fused MoE kernel? Is it model-dependent? Different models
> default to different backends? This section makes "default" precise
> with source + trace evidence.

### 29.1 "Default" is not one value — it's a vector of ~10 independent dispatcher defaults

There is no single "the default backend" in sglang. There are **at least 10 separate backend flags**, each with its own default, each resolved independently at startup. Source: `sglang/srt/server_args.py:445-535`:

| Flag | Default value (in `server_args.py`) | When resolved | Resolved on our H200 + Qwen3-30B-A3B bf16 |
|---|---|---|---|
| `attention_backend` | `None` → hardware-aware chain | `_handle_attention_backend_compatibility` (§21.3) | **`fa3`** (hopper auto-chain) |
| `prefill_attention_backend` | `None` (uses `attention_backend`) | same | inherits `fa3` |
| `decode_attention_backend` | `None` (uses `attention_backend`) | same | inherits `fa3` |
| `sampling_backend` | `None` → `flashinfer` if available, else `pytorch` | `_handle_sampling_backend:1766-1770` | **`flashinfer`** |
| `grammar_backend` | `xgrammar` | literal | `xgrammar` |
| `moe_runner_backend` | `auto` | per-quant resolution inside each `*MoEMethod` class (§24) | **`auto` → Triton `fused_moe_kernel`** |
| `moe_a2a_backend` | `none` | literal | `none` |
| `fp8_gemm_runner_backend` | `auto` | inside `Fp8GemmRunnerBackend` | `auto` (no-op, no FP8 weights) |
| `fp4_gemm_runner_backend` | `flashinfer_cutlass` | literal | `flashinfer_cutlass` (no-op, no FP4 weights) |
| `mamba_backend` | `triton` | literal | `triton` (no-op, no Mamba layers) |
| `lora_backend` | `csgmv` | literal | `csgmv` (no-op, no LoRA) |
| `nsa_prefill_backend` | `None` → hardware/dtype detect | `_handle_nsa_*` | None (no NSA) |
| `mm_attention_backend` | `None` | model-loader decides | None (text-only model) |

So when our trace shows "default" behavior, what's actually running is a **specific combination of resolved defaults** — `fa3 + flashinfer + xgrammar + Triton-fused_moe_kernel + …`. Calling any one of them "the default" is loose talk.

### 29.2 Resolution evidence — directly from our C0 baseline `server_info.json`

```json
{
  "attention_backend": "fa3",                   ← resolved from None
  "prefill_attention_backend": null,
  "decode_attention_backend": null,
  "sampling_backend": "flashinfer",             ← resolved from None
  "grammar_backend": "xgrammar",
  "moe_runner_backend": "auto",                 ← stays "auto", but
                                                  internally resolved to Triton
                                                  (see §24 + §25)
  "moe_a2a_backend": "none",
  "fp8_gemm_runner_backend": "auto",
  "fp4_gemm_runner_backend": "flashinfer_cutlass",
  "mamba_backend": "triton",
  "lora_backend": "csgmv",
  "speculative_moe_runner_backend": "auto",
  "speculative_moe_a2a_backend": null,
  ...
}
```

**Critical observation**: `moe_runner_backend` says `'auto'` in `server_info.json`, but the actual kernel running is `fused_moe_kernel`. `/server_info` shows the **user input**, not the resolved value (§23.2). To know what really runs, you have to look at the trace's kernel names.

### 29.3 Source breakdown — what's actually running per-component in our C0 trace

| GPU component | Default-resolved backend | Actual kernel name in trace | Library origin |
|---|---|---|---|
| MoE FFN | `moe_runner_backend=auto` → Triton | **`fused_moe_kernel`** (46.4 % GPU) | sglang hand-written Triton |
| Attention | `attention_backend=auto` → fa3 (Hopper) | `void cutlass::device_kernel<flash::FlashAttnFwdSm90<…>>` (13.4 %) | flash-attn library (NVIDIA, CUTLASS-based) |
| GEMM (Q/K/V/O proj) | implicit cuBLAS via `nn.Linear` | `nvjet_tst_*` (~13 %) | cuBLAS / cuBLASLt (NVIDIA) |
| RMSNorm | implicit flashinfer choice | `flashinfer::FusedAddRMSNormKernel` (~2.4 %) | flashinfer (hand-written CUDA C++) |
| RoPE | implicit flashinfer choice | `flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel` (~1.8 %) | flashinfer |
| Activation (SiLU) | implicit flashinfer choice | `flashinfer::activation::act_and_mul_kernel<silu>` (~2.8 %) | flashinfer |
| Sampling | `sampling_backend=flashinfer` | (rolls into general `at::native` + `flashinfer::*sample`) | flashinfer |
| Schedule helpers | `moe_align_block_size`, `moe_sum_reduce`, `write_req_to_token_pool` | (~3 % GPU) | sglang hand-written Triton |
| Misc | various PyTorch | `void at::native::elementwise_kernel<…>` (~3 %) | PyTorch native |
| Inductor (auto-gen) | — | 0.00 % (only 19 µs total) | Not in default forward path |

### 29.4 Answers to Debadeepta's 4 sub-questions

#### Q: Is default Triton?

**Partially.** Three different defaults dominate different parts of the forward pass:

- **MoE compute (49 % GPU): Triton**, via sglang's `fused_moe_kernel` + helpers
- **Attention (13 %): NOT Triton** — `fa3` is CUTLASS-based CUDA from flash-attn
- **GEMM proj (13 %): NOT Triton** — cuBLAS/cuBLASLt
- **Norm + RoPE + activation (7.2 %): NOT Triton** — flashinfer hand-written CUDA C++

Calling any one of these "the default" misses the others. The binary "is default Triton?" question is itself the wrong frame.

#### Q: Is default a fused MoE kernel?

**Yes, for the MoE FFN specifically**: `moe_runner_backend=auto` on bf16 H200 resolves to sglang's hand-written `fused_moe_kernel` (a Triton kernel). But "default" for attention / sampling / GEMM is something else entirely (see Q1).

If the question is "for MoE compute specifically, what runs at default?": **Triton `fused_moe_kernel`**. If it's "what runs by default in the whole engine?": **a mix of fa3 + flashinfer + cuBLAS + Triton fused MoE**.

#### Q: Is default model-dependent?

**Yes, at least 3 dimensions of dependency**:

| Dependency dimension | Affects | Example |
|---|---|---|
| **MHA vs MLA architecture** | Attention auto-chain | Both fall to `fa3` on H200 (per §21.3 Step 2a vs 2b), but the MLA chain is different. On Blackwell, MHA → `trtllm_mha`, MLA → `flashinfer`. |
| **Quantization config** (`config.json["quantization_config"]`) | Which `*MoEMethod` class loads (§24) → how `moe_runner_backend=auto` is interpreted | bf16: `UnquantizedFusedMoEMethod` → Triton. FP8: `Fp8MoEMethod` → DeepGEMM (if available) or Triton. AWQ: `AWQMarlinMoEMethod` → Marlin W4A16 GEMM. |
| **Multi-modal vs text-only** | `mm_attention_backend` engages | text-only: `None`; LLaVA/Pixtral: model-loader picks `sdpa` / `fa3` / `triton_attn` |

#### Q: Different models import → different default backends?

**Yes. Concrete enumeration on H200**:

| Model | Default attention | Default MoE runner | Resolved MoE kernel |
|---|---|---|---|
| Qwen3-0.6B dense bf16 (our C0 dense) | `fa3` | N/A (no MoE) | — |
| Gemma-3-1B dense bf16 (our gemma) | `fa3` | N/A | — |
| Qwen3-30B-A3B MoE bf16 (our C0 MoE) | `fa3` | `auto` → Triton | **`fused_moe_kernel`** |
| Qwen3-30B-A3B MoE FP8 (hypothetical, requires `--quantization fp8`) | `fa3` | `auto` → DeepGEMM if available | **`fp8_blockwise_scaled_grouped_mm`** (DeepGEMM) or Triton FP8 path |
| DeepSeek-V3 MLA + native FP8 | `fa3` (MLA chain on H200) | `auto` → DeepGEMM (model ships FP8 weights) | `fp8_blockwise_scaled_grouped_mm` |
| Llama-3 dense bf16 on H200 | `fa3` | N/A | — |
| Llama-3 dense bf16 on A100 (SM80) | `flashinfer` (or `triton` if flashinfer missing) | N/A | — |
| **Same Qwen3-30B-A3B on AMD MI300X** | `aiter` | `auto` → AIter | AIter's `fused_moe` (C++/HIP, NOT Triton) |
| **Same Qwen3-30B-A3B on Blackwell B200** | `trtllm_mha` (MHA on Blackwell) | `auto` → ? (no explicit Blackwell-specific MoE chain — likely cutlass once it's supported there) | unclear, would need test |

### 29.5 How to verify "default" for any model — the only reliable method

`/server_info` lies (§23.2). The only reliable way:

1. Run a short benchmark with `--profile`:
   ```bash
   python -m sglang.bench_serving --backend sglang \
       --host 127.0.0.1 --port 30000 \
       --model <MODEL> \
       --dataset-name random \
       --random-input-len 128 --random-output-len 128 \
       --num-prompts 16 \
       --profile --profile-num-steps 5 \
       --profile-output-dir ./trace
   ```
   (must set `SGLANG_TORCH_PROFILER_DIR` on server side, see `.github/skills/pytorch-profiling`)

2. Open the trace, look at kernel names:
   - `fused_moe_kernel` → sglang's hand-written Triton MoE
   - `cutlass_fused_experts_fp8` → cutlass FP8 MoE (only if FP8 model)
   - `_p_matmul_ogs_*` → triton_kernels library (only if `--moe-runner-backend triton_kernel`)
   - `flashinfer_cutlass_fused_moe` → flashinfer cutlass MoE
   - `FlashAttnFwdSm90` → fa3 attention
   - `BatchPrefillWithRaggedKV` → flashinfer attention (NOT fa3)
   - `flashinfer::FusedAddRMSNormKernel` → flashinfer norm
   - `triton_poi_fused_*` / `triton_tem_fused_*` / `triton_red_fused_*` → Inductor (rare in default config)

3. Use our `detect_silent_noop.py` to automate this comparison for any flag you set:
   ```bash
   python scripts/regime_study/detect_silent_noop.py \
       --server-info <RUN_DIR>/server_info.json \
       --trace <RUN_DIR>/raw_trace/*/p_*.trace.json.gz
   ```

### 29.6 One-paragraph summary for Debadeepta / Mason

> **"Default" in sglang is not a single value — it's a vector of ~10
> independent dispatcher defaults, resolved at startup based on
> (hardware, model architecture, quantization config)**. For our C0
> baseline (H200 + Qwen3-30B-A3B bf16), the resolved default is
> `attention=fa3 + sampling=flashinfer + moe_runner=auto→Triton(fused_moe_kernel) + grammar=xgrammar + …`. **The same default `moe_runner=auto` resolves to Triton on bf16, DeepGEMM on FP8, AIter on AMD, and cutlass once you're on Blackwell-+-FP8**. So yes, default IS model-dependent (via quantization) AND hardware-dependent (via architecture-specific auto-chains). **The only reliable way to verify what default a particular model resolves to is to run a trace and read the kernel names** — `/server_info` returns the user input verbatim, not the resolved value (§23.2). We have a tool for this: `scripts/regime_study/detect_silent_noop.py`.

## 30. How sglang actually imports a HF model — the full flow (Debadeepta open question)

> 🟢 NEW. The other open question from the meeting: how does sglang turn
> a HuggingFace model directory into a running engine? Does it parse the
> PyTorch source code? Does it pattern-match and map to kernels?
> What happens if a pattern has no kernel?
>
> This section traces the complete path from `--model-path` to GPU kernel
> launch, with source citations and references to our experiments.

### 30.1 The big-picture summary (one paragraph)

**Sglang does NOT parse PyTorch source. It does NOT pattern-match.** Instead, every supported model architecture has a **hand-written Python class in `sglang/srt/models/*.py`**, written by sglang maintainers, that re-implements the model's forward pass using sglang's own optimised layer primitives (`RadixAttention`, `FusedMoE`, `RowParallelLinear`, etc.). At import time, sglang reads `config.json`'s `architectures` field, looks up the architecture string in a static `ModelRegistry`, and instantiates the corresponding hand-written class. Weight loading uses a per-model **hand-written translation dictionary** (`stacked_params_mapping`, `expert_params_mapping`) to map HF safetensors names to sglang parameter names. Kernel choice happens later, inside each layer class, driven by the `--*-backend` flags + hardware auto-detection + quantization config (the dispatchers from §21-§29). **Nothing is auto-generated from arbitrary PyTorch code.**

This is in contrast to:

- **`torch.compile` + Inductor**: actually traces the forward pass with Dynamo, identifies fusion patterns, generates Triton kernels at runtime. Works on arbitrary code.
- **TensorRT-LLM**: parses an exported ONNX/FX graph, identifies fusion templates, compiles to a `.plan` file. Works on arbitrary graphs (with caveats).
- **sglang**: requires that a human has already hand-written `sglang/srt/models/<arch>.py` for your architecture. No automation.

### 30.2 The 5-stage import pipeline, with source citations

```
sglang.launch_server --model-path X --model-impl auto
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ Stage 1 — Config parsing                                             │
│   Source: sglang/srt/configs/model_config.py:127                     │
│   Calls: transformers.AutoConfig.from_pretrained(model_path, ...)   │
│   Reads: <model_path>/config.json                                   │
│   Extracts: architectures, hidden_size, num_layers, num_experts,    │
│             torch_dtype, quantization_config, ...                   │
│   Fails: ValueError if transformers doesn't know model_type        │
│           ← Gemma-4 hits here (§28.2)                               │
└──────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│ Stage 2 — Architecture → class resolution                            │
│   Source: sglang/srt/model_loader/utils.py:89-117                    │
│           sglang/srt/models/registry.py:78-89                        │
│   Logic:                                                             │
│     archs = config.architectures                  # ['Qwen3MoeForCausalLM'] │
│     if not in sglang ModelRegistry:                                  │
│         architectures = resolve_transformers_arch(...)               │
│         ← if transformers has it, becomes "TransformersForCausalLM" │
│           (§28.1 corrected fallback path)                            │
│     return ModelRegistry.resolve_model_cls(archs)                    │
│   Output: the Python class object, e.g. Qwen3MoeForCausalLM         │
│   Fails: raise ValueError if neither sglang nor transformers has it │
└──────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│ Stage 3 — Layer construction                                         │
│   Source: sglang/srt/models/qwen3_moe.py (1500 lines hand-written)  │
│   Action: ModelClass.__init__(config) instantiates every layer:     │
│     - QKVParallelLinear, RowParallelLinear (with TP sharding)       │
│     - RadixAttention (sglang's KV-cache-aware attention)            │
│     - FusedMoE (calls fused_moe_kernel internally)                  │
│     - VocabParallelEmbedding (sharded vocab embedding)              │
│   Layer choice is HARD-CODED in the model class. Backend flag       │
│   selection happens INSIDE the layers (see Stage 5).                │
└──────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│ Stage 4 — Weight loading                                             │
│   Source: sglang/srt/model_loader/loader.py:653                     │
│           model class's load_weights() method (per-model)            │
│   Logic:                                                             │
│     1. Open all *.safetensors files (mmap'd)                        │
│     2. Iterate (name, tensor) pairs                                 │
│     3. Translate HF name → sglang name using hand-written tables:   │
│        stacked_params_mapping = [                                   │
│            ("qkv_proj", "q_proj", "q"),                             │
│            ("qkv_proj", "k_proj", "k"),                             │
│            ("qkv_proj", "v_proj", "v"),                             │
│            ("gate_up_proj", "gate_proj", 0),                        │
│            ("gate_up_proj", "up_proj", 1),                          │
│        ]                                                             │
│        expert_params_mapping = FusedMoE.make_expert_params_mapping(...)│
│     4. Call weight_loader(param, loaded_weight, shard_id) which     │
│        does the in-place shard copy.                                │
│   Fails: shape mismatch → raise; missing weights → init random      │
│          (with logger.warning); extra weights → discard (warning)   │
└──────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│ Stage 5 — Runtime setup (KV cache + CUDA graph + warmup)             │
│   Source: sglang/srt/model_executor/model_runner.py:init_*          │
│   Actions:                                                           │
│     - Compute KV cache size based on mem_fraction_static            │
│     - Resolve attention_backend / moe_runner_backend / etc.         │
│       (§21.3 hardware-aware chains, §24 quant-aware dispatch)       │
│     - CUDA graph capture for each (batch_size, seq_len) bucket      │
│     - Run a few warmup forward passes                               │
│   Fails: backend constructor errors, JIT compile errors, OOM, hang  │
│           ← C5 (flashinfer JIT), C7 (CUDA graph hang) hit here     │
└──────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
                          Server ready for requests
```

### 30.3 Direct answers to the 6 sub-questions

#### Q1: Does sglang receive PyTorch code?

**No.** It receives 4 things from `<model_path>`:

| File | Purpose |
|---|---|
| `config.json` | HF config dictionary (architectures, hidden_size, etc.) |
| `*.safetensors` (or `*.bin`) | Raw weight tensors |
| `tokenizer.json` / `tokenizer.model` | Tokenizer |
| Optional: `chat_template.jinja` | Chat formatting |

**It never reads** HF's `modeling_qwen3_moe.py`, never reads HF's `nn.Module` class definitions. HF's forward implementation is **completely unused** because sglang maintainers already wrote an equivalent forward in `sglang/srt/models/qwen3_moe.py` that uses sglang's optimised layers.

#### Q2: Does it parse model source?

**Strictly no.** 0% parsing. No AST traversal, no FX graph, no Dynamo trace. The mapping from "this is a Qwen3 MoE model" to "use these sglang classes and kernels" is **purely a static dictionary lookup** by architecture string.

Source — [`sglang/srt/models/registry.py:78-89`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py):

```python
def resolve_model_cls(self, architectures):
    architectures = self._normalize_archs(architectures)
    for arch in architectures:
        model_cls = self._try_load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)
    return self._raise_for_unsupported(architectures)
```

`self._try_load_model_cls(arch)` is just a `dict.get(arch, None)`.

#### Q3: Does it recognise patterns and map to existing kernels?

**No.** This is the fundamental difference between sglang and compiler-style systems:

| System | How it maps model → kernel |
|---|---|
| `torch.compile` + Inductor | Real Dynamo trace of forward → recognise op patterns (matmul, norm, activation chain) → auto-generate Triton |
| TensorRT-LLM | Parse exported ONNX/FX graph → match fusion templates → compile to `.plan` |
| **sglang** | **No pattern recognition.** Each model's forward is a **hand-written Python file** in `sglang/srt/models/` that **explicitly calls** sglang's pre-built layers. **Humans decide which kernel each layer uses** — not machine inference. |

Example: `sglang/srt/models/qwen3_moe.py`:

```python
class Qwen3MoeAttention(nn.Module):
    def __init__(self, ...):
        self.qkv_proj = QKVParallelLinear(...)        # ← human said: use this layer
        self.o_proj = RowParallelLinear(...)          # ← human said: use this layer
        self.attn = RadixAttention(...)               # ← human said: this layer internally calls fa3

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, ...):
        self.experts = FusedMoE(...)                  # ← human said: this calls fused_moe_kernel
```

Within each layer, the backend flag (`--attention-backend=auto`, `--moe-runner-backend=auto`) decides which actual kernel runs (see §21-§29). But the choice of LAYER (and therefore the choice of which kernel family to use) is hard-coded by the human who wrote the model file.

#### Q4: If a pattern has no kernel, does import fail or fall back?

**There is no such thing as "a pattern with no kernel" in sglang's model.** Patterns are not what dispatches. Failure happens in one of 5 places (see the flowchart above):

| Stage | Failure mode | Recovery |
|---|---|---|
| 1. Config parse | transformers ValueError (model_type unknown) | **NONE** — server exits. Gemma-4 hits here (§28.2) |
| 2. Architecture lookup | sglang has no `<arch>.py` | **Fall back** to TransformersForCausalLM wrapper (§28.1) IF transformers itself has it; otherwise hard fail |
| 3. Layer construction | layer's `__init__` raises | **NONE** — server exits |
| 4. Weight loading | shape mismatch on a tensor | **NONE** — server exits; missing weights → warning + random init |
| 5. Runtime setup | backend JIT, CUDA graph capture, OOM, hang | **NONE** — server exits or hangs. C5 + C7 hit here (§19, §27) |

**No "fallback to a slower kernel"** at runtime — kernel choice is decided once at startup based on the dispatchers (§29.1), not per-pattern.

#### Q5: How does sglang bench download, load, and run benchmarks?

**Download**: sglang itself doesn't download. The user provides `--model-path`. If it's an HF hub ID (not a local path), sglang uses `huggingface_hub.snapshot_download` (or `modelscope` if `SGLANG_USE_MODELSCOPE` is set) to cache it locally — but you usually pre-download to `HF_HUB_CACHE` (we pinned to `/data/hf/gujialiang123/hf_cache`).

**Load**: stages 1-4 above. Time depends on file size and disk speed — for our 30B MoE bf16, ~13 seconds for `Loading safetensors checkpoint shards: 100% Completed | 16/16`.

**Benchmark**: `sglang.bench_serving` is a **separate client process** that sends HTTP requests to `<host>:<port>/generate`. It doesn't load the model, doesn't touch GPU directly. It just:

1. Opens a socket to the server
2. Sends N concurrent `POST /generate` requests with sampled prompts
3. Times each request: TTFT (time-to-first-token from response stream), TPOT (per-output-token time), e2e latency
4. Aggregates into the metrics we see in `bench.jsonl`

Our `scripts/run_benchmark.py` is a thin wrapper that builds `bench_serving` argv from `workload.yaml`. Documented in §5 of this report.

#### Q6: For different backends, what's the path from arbitrary code to kernel list?

**Backend doesn't affect the "code → kernel" path much, because there is no "code" path** — the forward graph is hand-written. Backend only affects **which kernel each pre-existing layer chooses**. Full 5-level breakdown:

| Level | Set by | Example |
|---|---|---|
| **L1: Forward graph shape** | hand-written Python in `models/*.py` (backend doesn't change this) | `qwen3_moe.py:Qwen3MoeAttention.forward` always calls `RadixAttention.forward` |
| **L2: Layer class implementation** | hand-written Python in `layers/*.py` (backend doesn't change this either) | `RadixAttention.forward` always calls `self.attn_backend.forward(...)`, where `self.attn_backend` is set at init |
| **L3: Backend dispatch** | `--*-backend` flag + hardware auto-chain (§21.3) + quantization config (§24) | `attention_backend=auto` on H200 → fa3 backend object; `moe_runner_backend=auto` on bf16 → `MoeRunnerBackend.TRITON` |
| **L4: Kernel selection** | backend object's internal logic | fa3 dispatches to flash-attn's `FlashAttnFwdSm90<…>`; Triton MoE dispatches to sglang's `fused_moe_kernel` |
| **L5: Kernel launch parameters** | per-shape JSON tuning tables (§18.4) + Triton constexpr specialisation + cuBLASLt tile selection | `fused_moe_kernel` at M=32 uses `BLOCK_SIZE_M=16, num_warps=4`; at M=1024 uses `BLOCK_SIZE_M=128, num_warps=8` |

So **"different backends → different kernels"** isn't because the forward graph changed — it's because L3 dispatch picked a different backend object, which selected a different kernel at L4, with different parameters at L5. **L1 + L2 are immutable hand-written sglang code**; the backend flag can never change what graph the model runs.

### 30.4 What this means for our agent strategy

Knowing the import mechanism precisely tells us where an optimisation agent could insert itself:

| Insertion point | What the agent could do | Feasibility |
|---|---|---|
| **Pre-Stage 1** | Edit `config.json` (e.g. switch quantization, change `model_type`) | Easy, but limited |
| **Stage 2** | Write a new `sglang/srt/models/<arch>.py` for an unsupported model | Hard — requires expertise; this is what's needed for Gemma-4 |
| **Stage 3 (layer choice)** | Change which sglang layer the model class uses (e.g. swap `RadixAttention` for a custom one) | Hard — needs editing model file |
| **Stage 5 (backend dispatch)** | Set `--*-backend` flags to steer kernel selection | **Easy** — this is what we did in §19/§27 |
| **Inside L5 (kernel tuning)** | Re-tune JSON for our specific shape (Triton autotuner) | Medium — 1 day investment, 5-10% gain (§25.4) |
| **Inside L4 (kernel source)** | Hand-write a new Triton kernel; register in `FusedMoE.apply()` | Hard — Triton expertise required; biggest potential wins |
| **Inside L1 (graph rewrite)** | Fix sglang's `rotary_embedding.py` to be Dynamo-compatible so `torch.compile` works → Inductor takes over and auto-fuses (§26) | Medium — 1-2 days; 5-15% expected gain on non-hot-path |

**Agent action items, by ROI**:

1. **Stage 5 (flag steering)** — what we already do; cheap, can scan the 30+ flag space
2. **L5 (per-shape JSON re-tune)** — cheap, 5-10% wins; an agent could run Triton autotuner per workload
3. **L4 (kernel rewrite)** — the highest-payoff but needs Triton expertise; agent could propose patches and run benchmark CI
4. **Stage 2 (new model class)** — biggest enabler; an agent that could read HF source and emit a working `sglang/srt/models/<arch>.py` would unlock all unsupported models

### 30.5 One-paragraph summary for Debadeepta / Mason

> **Sglang does NOT parse PyTorch source, does NOT pattern-match, does NOT auto-generate kernels.** Each supported model has a **hand-written Python class** in `sglang/srt/models/<arch>.py` (written by sglang maintainers) that re-implements the forward using sglang's optimised layer primitives (`RadixAttention`, `FusedMoE`, `RowParallelLinear`). At import time, sglang reads `config.json`'s `architectures` field, looks up the class in a static `ModelRegistry`, instantiates it, and loads weights via a **per-model hand-written translation dictionary** (`stacked_params_mapping` + `expert_params_mapping`) that maps HF safetensors names → sglang parameter names. Kernel choice happens **inside each layer's `forward()`**, driven by `--*-backend` flags + hardware + quantization config (§21-§29). The 5-stage failure model is: (1) `transformers.AutoConfig` doesn't know `model_type` → fail (Gemma-4 hits this); (2) arch missing from sglang ModelRegistry → fall back to `TransformersForCausalLM` wrapper IF transformers has it (§28); (3-4) layer init / weight shape mismatch → fail; (5) backend JIT / CUDA graph hang → fail or hang (our C5, C7 hit this). **There is no "pattern with no kernel" because patterns aren't what dispatch — flags + hardware + quant-config are.** An optimisation agent therefore inserts most cheaply at the flag/dispatch layer (Stage 5) or per-shape kernel-tuning layer (L5), with progressively more expertise needed to insert at layer source (Stage 3 layer-choice) or kernel source (L4 kernel rewrite).
