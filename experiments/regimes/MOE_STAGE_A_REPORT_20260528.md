# Stage A — Problem-Setter Report: Qwen3-30B-A3B MoE (2026-05-28)

> First real MoE run. 5 seeds + 2 boundary neighbors, total 9 min wall
> time on a single H200. One problem package produced (P001 MoE), one
> idea filed (R-001) for follow-up.

## Environment

| Item | Value |
|---|---|
| Hardware | NVIDIA H200 (143 GB), GPU 0 |
| Model | Qwen3-30B-A3B-Instruct-2507 (30B total, 128 experts, 8 active/token, GQA 32/4) |
| Disk | 57 GB bf16 |
| SGLang | 0.5.12.post1 |
| Config | `configs/moe_qwen3_30b.yaml` (TP=1, mem-fraction-static=0.85, context-length=32768) |
| Conda env | sglang-dev |
| HF cache | /data/hf/gujialiang123/hf_cache |

Wave 0: 5 seeds (smoke / short_in_short_out / scheduler_overhead_high_concurrency / prefill_long / decode_heavy)
Wave 1: 2 boundary neighbors (mc=16, mc=32) on the scheduler workload

Total: 7 benchmarks, ~9 min wall time. 0 failures.

---

## Wave 0 + Wave 1 numbers

| workload | mc | num_prompts | TTFT p50 | TTFT p95 | TPOT p50 | out_tps | req_tps |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke                                          |  4 |  16 |  79 |  **621** | 18.5 |  121 |  5.6 |
| short_in_short_out                             | 16 |  64 | 458 |  **975** | 80.5 |  160 |  9.4 |
| scheduler_overhead_high_concurrency (target)   | 64 | 128 | 925 | **2282** | 44.0 |  324 | 41.3 |
| ├─ neighbor con_16 (wave 1)                    | 16 |  32 |  96 |    142  | 32.4 |  388 | 44.2 |
| └─ neighbor con_32 (wave 1)                    | 32 |  64 | 136 |    147  | 43.3 |  615 | 70.0 |
| prefill_long (8k input)                        |  2 |   8 | 155 |   944   |  5.4 |   52 |  4.7 |
| decode_heavy (1024 output)                     | 16 |  32 | 145 |   260   | 12.4 | 1024 |  1.9 |

---

## Findings

### Finding A (reproduces) — `max-running-requests=32` caps concurrency, cliff is much steeper on MoE

Same `concurrency_capped` signal as the Qwen3-0.6B run, but the cliff is
much sharper:

| metric | mc=16 | mc=32 | mc=64 | mc=32 → mc=64 jump |
|---|---:|---:|---:|---:|
| TTFT p95 (ms)        | 142 | 147 | **2282** | **15.5×** |
| out_tps              | 388 | **615** | **324** | **−47% (regression!)** |
| TPOT p50 (ms)        | 32  | 43  | 44   | +2% |

By contrast, on 0.6B the mc=64 TTFT p95 was only 3.6× the mc=32. **MoE
collapses harder when queueing kicks in**, AND throughput **regresses**
(was monotonically increasing on 0.6B). This is the most actionable
problem: the same fix (raise `max-running-requests` to 64) likely yields
even bigger gains on MoE than on 0.6B.

Server log evidence:
- `peak_running_reqs = 30`, `peak_queue_reqs = 38`, `max_running_requests = 32` → load shedding confirmed
- `concurrency_capped = True`
- Classification: `load_shed_concurrency`

→ Packaged as `experiments/problems_moe/P001/`. Suspicion score 0.735.

### Finding B (NEW — idea pool, not packaged) — MoE cold-start tail is 6× worse than 0.6B

| workload | model | TTFT p50 | TTFT p95 | p95/p50 |
|---|---|---:|---:|---:|
| smoke           | 0.6B | 24  | 99  | 4.0 |
| smoke           | MoE  | 79  | 621 | **7.8** |
| short_in_short_out | 0.6B | 34  | 99  | 2.9 |
| short_in_short_out | MoE  | 458 | 975 | 2.1 |

The MoE `smoke` workload (mc=4, 16 prompts) has p95 = 621 ms — that's
more than the 0.6B mc=64 scheduler workload's p95 (434 ms). For a
4-concurrent 32-token output benchmark on a 30B MoE, that's a lot of
warmup cost.

The v0.4 rule-based scorer **does not flag this** (classification =
clean_pass, no concurrency cap, no KV pressure). But the
ratio jump vs 0.6B (4.0 → 7.8) is large enough to be a real MoE
characteristic worth investigating.

→ Filed as `experiments/ideas/from_setter/idea_001.json` (R-001).
Suggested follow-up: probe `num_prompts ∈ {4, 16, 64, 256}` to see how
fast the warmup tail amortizes.

### Finding C (informational) — MoE TPOT is 11× the 0.6B baseline

| workload | TPOT p50 — 0.6B | TPOT p50 — MoE | ratio |
|---|---:|---:|---:|
| smoke               | 3.4 ms | 18.5 ms | 5.4× |
| short_in_short_out  | 7.2 ms | 80.5 ms | **11.2×** |
| scheduler (mc=64)   | 16.2 ms | 44.0 ms | 2.7× |
| prefill_long        | 2.7 ms | 5.4 ms  | 2.0× |
| decode_heavy        | 2.8 ms | 12.4 ms | 4.4× |

11× TPOT on a "1 GB → 57 GB on disk, 0.6B → 3B active" jump is within
expected MoE overhead range. Not a finding by itself, but contextualizes
all latency numbers.

### Finding D — `prefill_long` (8k input) is *cheaper* than `short_in_short_out`

On MoE, `prefill_long` (mc=2, 8 prompts, 8192 input) gave TTFT p50 = 155
ms while `short_in_short_out` (mc=16, 64 prompts, 128 input) gave TTFT
p50 = 458 ms. That's counterintuitive — long prefill should be expensive.

The likely explanation: `prefill_long` only has 2 concurrent requests, so
each gets the GPU more often (less queue). `short_in_short_out` at mc=16
already shows admission-cap-adjacent behavior on MoE even though
max_running_requests=32 isn't violated.

This suggests **even sub-cap concurrency on MoE pays a queueing tax** —
worth probing whether MoE's expert routing serializes batches more than
on a dense model.

→ Could be filed as an additional idea later if it shows up again.

---

## What landed where

```
experiments/problems_moe/P001/                          (the MoE-specific Finding A package)
├── problem.json
├── workload.yaml                            scheduler_overhead_high_concurrency, mc=64
├── baseline_metrics.json
├── server_features.json
├── classification.json
├── neighbors/
│   ├── scheduler_overhead_high_concurrency__con_16.{yaml,baseline_metrics.json,server_features.json,classification.json}
│   └── scheduler_overhead_high_concurrency__con_32.{yaml,baseline_metrics.json,server_features.json,classification.json}
├── controls/
│   └── prefill_long.{yaml,baseline_metrics.json,server_features.json,classification.json}
├── hypothesis.md                            (auto-drafted)
├── acceptance_criteria.json                 (target: TTFT p95 ≥ 30% improvement)
└── attempts/                                (empty; solver writes here)

experiments/ideas/from_setter/idea_001.json  (R-001 — MoE cold-start tail follow-up)

regime_scout/outputs/
├── moe_raw_results.jsonl                    (7 rows)
├── moe_suspicious.jsonl                     (scored)
└── moe_selected_problems.jsonl              (1 row)

logs/
├── moe_wave0_20260528_172134.log
└── moe_wave1_20260528_172928.log
```

---

## Honest assessment of v0.4 scoring on MoE

The rule-based scorer **correctly** identified the load-shed problem
(Finding A reproduces with the same score 0.735) but **missed** the
two more MoE-characteristic findings (B and C/D). Not a scorer bug —
those findings need either:

1. Cross-model comparison (the scorer only sees one model's data at a
   time).
2. New rules sensitive to "absolute TPOT > X for cheap input" (no such
   rule yet).
3. Stronger noise calibration to know if a 7.8 p99/p50 ratio is
   abnormal for this model (requires noise baseline run — not done).

For now, the v0.4 setup correctly produces a problem package on the
strongest signal AND surfaces the second-tier finding as an idea pool
entry for follow-up. That's the right behavior given the contract.

---

## Suggested next actions

1. **Send P001 (MoE) to a config-agent** (when Stage B lands) to verify
   the predicted ~60% TTFT p95 improvement from raising
   `max-running-requests`.
2. **Probe Finding B** by running a `num_prompts` sweep on smoke
   (16, 64, 256, 1024) to characterize how the warmup tail amortizes.
3. **Run noise calibration** (`calibrate_noise.py`) on MoE smoke to
   determine whether the 7.8 p95/p50 ratio is signal or repeatable.
4. **Verify Finding D** by adding a "moe_mid_concurrency" probe at
   mc=8 to see if queue effects appear before max_running_requests is
   actually hit.
