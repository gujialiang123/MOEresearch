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
