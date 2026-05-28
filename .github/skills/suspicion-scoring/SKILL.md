---
name: suspicion-scoring
description: Combine server-log-mining, failure-classification, noise-aware-scoring, and local-nonlinearity into one suspicion score per workload run, with full evidence audit trail.
version: 2
stage: [1]
inputs:
  - raw_results.jsonl    (one row per workload run)
  - server_features.json per run
  - classifications.json per run
  - noise_baseline.json  (optional; if missing, defaults used)
outputs:
  - suspicious_cases.jsonl
triggers:
  - "after every wave of run_regime_suite.py"
depends_on: [server-log-mining, failure-classification, noise-aware-scoring]
---

# suspicion-scoring

## WHEN

Call after every wave of benchmarks (seeds and each expanded wave). The
score depends on the **current population** of runs — adding new neighbor
runs may move scores up or down.

## WHY

The v0.2 score function had the right shape but the wrong inputs:

1. It only saw `parse_metrics.py` output → missed server-log signals
   (Finding A).
2. It used hard-coded ratio thresholds → 9/10 workloads saturated.
3. `local_nonlinearity` needed neighbors and had none.

This v2 fixes all three by **composing skills** instead of reimplementing
their logic:

```
suspicion-scoring (v2)
  ├── server-log-mining       → fields.cuda_graph_too_small, concurrency_capped, ...
  ├── failure-classification  → near_failure_kv / load_shed_concurrency / clean_pass / ...
  ├── noise-aware-scoring     → adjusted_threshold(metric, base) for tail ratios
  └── local_nonlinearity      → in-skill, but only fires when neighbors exist
```

## HOW

Implementation: `impl/score.py`. Score formula (v2):

```
score = w1 * local_nonlinearity_v2          # only > 0 when a same-hint neighbor exists
      + w2 * tail_latency_ratio_v2          # uses adjusted_threshold from noise baseline
      + w3 * server_log_signal              # NEW: from server-log-mining derived booleans
      + w4 * failure_class_score            # NEW: from failure-classification enum
      + w5 * local_nonlinearity_v2_secondary  # secondary axis (e.g. throughput when primary=ttft)
```

`server_log_signal` component (NEW in v2):

| derived flag | contribution |
|---|---|
| `cuda_graph_too_small=true`     | +1.0 |
| `concurrency_capped=true`       | +0.7 |
| `at_capacity=true`              | +0.9 |
| `near_capacity=true` (and not at_capacity) | +0.4 |
| `retract_events > 0`            | +0.8 |
| `kv_pool_full_events > 0`       | +1.0 |
| `max_running_above_cuda_graph`  | +0.5 |

Clamped to [0, 1] then weighted.

`failure_class_score` (NEW in v2):

| classification | score |
|---|---|
| `clean_pass`                | 0.0 |
| `load_shed_concurrency`     | 0.7 |
| `partial_success`           | 0.8 |
| `near_failure_retract`      | 0.85 |
| `near_failure_kv`           | 0.9 |
| `oom`                       | 1.0 |
| `server_crash`              | 1.0 |
| `benchmark_timeout`         | 0.9 |
| others                      | 0.3 |

Default weights (v2):

```yaml
w1 local_nonlinearity_primary    = 0.20
w2 tail_latency_ratio            = 0.15
w3 server_log_signal             = 0.30   # biggest weight; reflects skill priority
w4 failure_class                 = 0.25
w5 local_nonlinearity_secondary  = 0.10
```

## OUTPUT CONTRACT

Same per-row schema as v1 but each `components.*` block now carries
`evidence` referencing the upstream skill's output JSON:

```json
{
  "run_id": "run_0004",
  "workload_name": "scheduler_overhead_high_concurrency",
  "score": 0.59,
  "components": {
    "server_log_signal": {
      "weight": 0.30,
      "score": 0.7,
      "evidence": {
        "concurrency_capped": true,
        "max_running_requests": 32,
        "peak_queue_reqs": 36,
        "feature_source": "experiments/tmp/.../server_features.json"
      }
    },
    "failure_class": {
      "weight": 0.25,
      "score": 0.7,
      "evidence": {
        "classification": "load_shed_concurrency",
        "feature_source": "experiments/tmp/.../classification.json"
      }
    },
    ...
  }
}
```

## FAILURE MODES

- Upstream skill JSON missing → that component reports `score=0` with
  `evidence.reason = "feature_missing"`. The aggregate score still computes.
- Noise baseline missing → falls back to v0.2 hard thresholds and prints a
  warning.

## ROADMAP

- v3: Bayesian update — combine prior (from regime hint) with empirical evidence.
- v3: incremental scoring — only re-score new runs, not the whole population.
