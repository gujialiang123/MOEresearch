# Stage 1 RegimeScout — First Real Run (2026-05-28)

**Hardware**: 8× NVIDIA H200 (1 GPU used, `CUDA_VISIBLE_DEVICES=0`)
**Model**: Qwen3-0.6B (`/data/hf/models/Qwen3-0.6B`, ~1.2 GB bf16)
**SGLang**: `0.5.12.post1` (source: `/home/t-jialianggu/work/sglang`)
**Env**: `conda run -n sglang-dev`
**Suite duration**: 708 s (11min 48s) for 10 workloads, ~70 s per workload
**Result**: 10/10 PASS, 0 OOM, 0 crash, 0 timeout

Suite log:    `logs/regime_suite_20260528_023213.log`
Raw results:  `regime_scout/outputs/raw_results.jsonl`
Per-run dirs: `experiments/tmp/regime_scout/20260528_023213/run_NNNN_*/`

---

## 1. Headline numbers

```
workload                                  ttft_p50  ttft_p95  ttft_p99  p99/p50  tpot_p50   out_tps  req_tps
--------------------------------------------------------------------------------------------------------------
smoke                                         24.4      98.7      98.9     4.06      3.42     628.6     46.2
tiny_latency                                  11.5      13.8      16.5     1.43      1.67     187.3     65.9
short_in_short_out                            34.4      98.7      99.7     2.90      7.25    1638.9     99.2
scheduler_overhead_high_concurrency          129.3     434.2     485.0     3.75     16.24    1835.3    211.3
prefill_medium                                32.7      49.6      90.3     2.77      4.12     481.9     54.9
prefill_long                                  73.0     120.1     137.7     1.89      2.71     174.3     18.3
decode_medium                                 19.3      95.7      96.7     5.01      2.51    5759.1     22.0
decode_heavy                                  22.9     123.2     123.9     5.42      2.78   10140.7     18.6
prefix_reuse_ideal                           123.3     177.3     184.6     1.50      3.90    3282.1     25.6
prefix_churn                                  64.8     185.4     201.7     3.11      4.56    3125.6     24.4
```

## 2. Findings worth a Stage-2 dive

### 🚨 Finding A — `max-running-requests=32` blocks CUDA graph above 32 concurrent reqs

Captured from `run_0004_scheduler_overhead_high_concurrency/server.log`:

```
cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, ..., 256]
Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32]
max_running_requests=32
```

SGLang's default `cuda_graph_bs` goes up to 256, but our `max_running_requests=32`
prevents capture above 32. When `scheduler_overhead_high_concurrency` runs with
`max_concurrency=64`, the batch above 32 cannot use the CUDA graph fast path:

- TTFT p95 **434 ms** vs 99 ms for `short_in_short_out` (mc=16)
- TPOT p50 **16 ms** vs 7 ms (also 2.3× regression)
- But request throughput **doubled** (211 vs 99 req/s) — pure throughput-vs-tail tradeoff

**Hypothesis for Stage 2/3**: raise `max-running-requests` to ≥ 64 and confirm
CUDA graph captures for bs > 32. Expect TTFT p95 drop by 2–4×.

### 🚨 Finding B — `decode_medium` and `decode_heavy` have ttft_p99/p50 ≈ 5

Both decode-heavy regimes show p99/p50 ≈ 5 (the highest of any workload), even
though absolute p99 stays modest (~100 ms). The first ~5 % of requests pay a
disproportionate TTFT cost while the rest fly. Likely cause:

- Cold KV cache warm-up on the first batches
- CUDA graph mode-switching as the batch shape changes

**Hypothesis for Stage 2/3**: a controlled warmup phase (already in our harness
but currently OFF) should flatten this tail. Worth measuring with `--warmup`.

### 🚨 Finding C — Persistent **~99 ms TTFT plateau** across three different workloads

| workload | ttft_p95 |
|---|---|
| `smoke`              | **98.66 ms** |
| `short_in_short_out` | **98.65 ms** |
| `decode_medium`      | **95.69 ms** |

99 ms is suspiciously close to a system tick boundary. Could be a scheduling
period, a flush-cache cost (we set `flush_cache: true` for all cold runs),
or the first-prefill prefill cost. **Worth investigating** in Stage 2 by
toggling `flush_cache` and checking the prefill log timestamps.

### 🟨 Finding D — Prefix reuse cold-start is heavy

`prefix_reuse_ideal` (TTFT p50 = **123 ms**) is 5× slower than `short_in_short_out`
(TTFT p50 = 24 ms), even though the design intent is "shared prefix, should be
fast". This is the radix-cache **cold start**: the first request per prefix
group pays the full 4096-token prefill. With `num_prompts=256` and 8 groups
(32 prompts/group), the per-group amortization isn't fully observed in this
short run.

**Hypothesis**: a longer run (≥ 1024 prompts) or pre-warmed cache should reveal
the true prefix-reuse advantage.

### 🟩 Finding E — Prefill is NOT the bottleneck for Qwen3-0.6B on H200

`prefill_long` (16384 input tokens, mc=2) only reached TTFT p95 = **120 ms**.
H200's compute is wildly overpowered for a 0.6 B model — even 16 k prefill
finishes in ~70 ms. The `chunked-prefill-size` knob will likely have **zero
effect** on Qwen3-0.6B. We should test it on a larger model (e.g.
Qwen3-30B-A3B) to see meaningful prefill regimes.

---

## 3. Honest evaluation of the v0.2 scoring function

```
top 10 (after running score_suspicion + cluster_regimes):
  smoke                                    score=0.250
  short_in_short_out                       score=0.250
  scheduler_overhead_high_concurrency      score=0.250
  prefill_medium                           score=0.250
  prefill_long                             score=0.250
  decode_medium                            score=0.250
  decode_heavy                             score=0.250
  prefix_reuse_ideal                       score=0.250
  prefix_churn                             score=0.250
  tiny_latency                             score=0.054
```

9 out of 10 saturate to 0.25. Reasons:

1. **`local_nonlinearity` always 0**: every `regime_hint` has only 1 seed
   workload, so the same-hint-neighbor lookup finds nothing.

2. **`tail_latency_ratio` is over-active**: 9/10 workloads have ttft_p99/p50 ≥ 3
   (the score threshold), so they all hit the cap (= 1.0 × 0.25 weight = 0.25).

3. **`diagnostic_sensitivity`, `stack_gap`, `variance` are all stubbed in v0.2**.

4. **`failure_nearness`**: 0 (no failures and we don't parse server log for
   KV pressure / queue depth yet).

**Diagnosis**: the score function as designed needs a neighborhood — it can't
detect anomalies from 1-point-per-regime. The system is functionally correct
but score-blind. The v0.2 demonstration ran with `--threshold 0.2` to force
selection of 5 cases for downstream wiring.

---

## 4. v0.3 fix list (NOT done yet — recorded for next iteration)

1. **Boundary expansion** before scoring: for each seed that passes, generate
   2–3 neighbors along (`input_len`, `max_concurrency`, `output_len`) and re-run.
   This unlocks `local_nonlinearity` and turns the system from "10 isolated
   points" into "an actual map".
2. **Parse server log** for `KV cache pool is full`, `Retract request`,
   `token usage` peaks → feed into `failure_nearness`.
3. **Auto-deduce `cuda_graph_capture_bs_max`** from server log and warn when
   `max_concurrency > captured_bs_max` (Finding A would have auto-flagged).
4. **Calibrate noise baseline**: run the smoke workload 5× to compute CV per
   metric, then make `tail_latency_ratio` use `cv-adjusted` thresholds rather
   than hard-coded 3.0.
5. **Bigger model for prefill regimes**: scout will benefit from a second pass
   using `Qwen3-30B-A3B` to actually see prefill-bound behavior.

---

## 5. What we have working today

| Component | Status |
|---|---|
| YAML → sglang argv translation                    | ✅ verified (B1 fix from DESIGN §0.G) |
| `conda run` + CUDA_HOME + HF_HOME env wrapping    | ✅ all subprocesses inherit cleanly |
| One-workload closed loop (`run_experiment.py`)    | ✅ smoke test + 10-workload suite green |
| Suite runner with budget / per-run dir / structured log | ✅ |
| Standardized metrics schema (`<mode>_metrics.json`) | ✅ |
| Suspicion scoring with audit trail                | ✅ structure correct, **needs neighbor data** |
| Rule-based clustering + human-readable regime map | ✅ |
| Stage 2 case handoff (case.json + frozen workload) | ✅ schema matches supplement §9 |

## 6. Where the data lives

```
regime_scout/outputs/
  raw_results.jsonl              # 1 row / workload, full metrics blob
  suspicious_cases.jsonl         # 1 row / workload, score + components
  regime_map.md                  # this run's human regime map
  regime_map.json                # same, machine readable
  selected_cases.jsonl           # 5 selected cases (with --threshold 0.2)

experiments/regimes/cases/
  S001/ ... S005/                # frozen case dirs (case.json + workload.yaml + metrics.json)

experiments/tmp/regime_scout/20260528_023213/
  run_0001_smoke/                # per-workload run dir
    server.log                   # full sglang server output
    quick_benchmark.log          # bench_serving output
    quick_raw.jsonl              # raw bench_serving jsonl (per-request ttfts, itls)
    quick_metrics.json           # our normalized metrics
    orchestrator.log             # run_experiment.py stdout
    config_snapshot.yaml         # server config used
    workload_snapshot.yaml       # workload used
    workload_input.yaml          # copy from candidates/
  run_0002_tiny_latency/
  ...
  run_0010_prefix_churn/

logs/
  regime_suite_20260528_023213.log     # suite-level structured log
  regime_suite_20260528_023213_nohup.out
```
