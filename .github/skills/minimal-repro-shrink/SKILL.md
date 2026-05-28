---
name: minimal-repro-shrink
description: Binary-shrink a workload along (num_prompts, max_concurrency, input_len, output_len) until the symptom disappears, producing the smallest workload that still reproduces the suspicious metric.
version: 1
stage: [1, 2]
inputs:
  - case workload yaml
  - target symptom (metric name + direction + observed value)
  - baseline config
outputs:
  - shrunk workload yaml
  - shrink_log.jsonl
triggers:
  - "before handing a case from Stage 1 to Stage 2"
  - "Stage 2 wants a faster repro for iteration"
depends_on: [server-log-mining, failure-classification]
---

# minimal-repro-shrink

## WHEN

Call when (a) Stage 1 has selected a suspicious case and we want a cheaper
workload to iterate on in Stage 2/3, or (b) Stage 2 finds the baseline
repro too slow for one-knob-at-a-time experiments.

## WHY

Suspicious cases identified by Stage 1 are usually full-size workloads
(e.g. `num_prompts=320`, `max_concurrency=64`). Iterating Stage 3 fixes on
them costs 60–120 s per attempt. A minimal repro that still triggers the
symptom (often `num_prompts=80, max_concurrency=64`) cuts iteration cost
in half without affecting outcomes.

It's also a **diagnostic tool**: if `max_concurrency` can be cut from 64 to
32 and the symptom **disappears**, that proves the symptom is concurrency-
dependent and not workload-shape-dependent. The shrink path is evidence.

## HOW (deferred to v0.4 — only SKILL.md exists in v0.3)

Shrink order, by cost-of-removal:

```
1. num_prompts:      halve until benchmark < 30 s OR symptom disappears
2. max_concurrency:  decrease one search_space step until disappears
3. input_len:        halve until below the regime's expected boundary
4. output_len:       halve (only for decode-bound regimes)
5. prefix groups:    halve (only for prefix_reuse / cache_churn regimes)
```

Stop when:
- symptom no longer reproduces (the metric returns to within `noise CV` of
  the cluster baseline), OR
- a hard floor is hit (`num_prompts >= 8`, `max_concurrency >= 1`).

The "symptom" is preserved if any of:
- primary metric is still worse than the regime baseline by ≥ adjusted_threshold
- `concurrency_capped` flag still True
- `retract_events > 0` still observed
- failure classification still in the same bucket

## OUTPUT CONTRACT (planned)

```yaml
# minimal repro workload yaml: same schema as the parent, with one or more
# axes reduced. Adds:
shrink_provenance:
  parent: "experiments/regimes/cases/S003/workload.yaml"
  iterations: 4
  preserved_symptom: "load_shed_concurrency"
  final_metric:
    name: "ttft_p95_ms"
    parent_value: 434.2
    shrunk_value: 421.1
    relative_change_pct: -3.0
```

And a structured log:

```jsonl
{"step": 1, "axis": "num_prompts", "from": 320, "to": 160, "symptom_preserved": true, "primary": 432.0}
{"step": 2, "axis": "num_prompts", "from": 160, "to": 80, "symptom_preserved": true, "primary": 425.4}
{"step": 3, "axis": "num_prompts", "from": 80, "to": 40, "symptom_preserved": false, "primary": 178.2}
{"step": 4, "axis": "num_prompts", "from": 80, "to": 80, "decision": "stop_at_last_preserved"}
```

## FAILURE MODES

- Symptom never reproduces even at parent size → `ok=false, error="symptom
  not present even at parent"`. The case was probably flaky; recompute
  noise baseline.
- All axes reach floor → return the floor workload but mark
  `shrink_provenance.hit_floor = true`.

## ROADMAP

- v1 impl (deferred to v0.4): straightforward sequential halving.
- v2: simultaneous-axis search with a small budget of `(2 axis)^k` trials.
- v3: use a learned proxy model (per regime) to predict "would shrinking
  this axis still preserve symptom?" and only re-benchmark when the proxy
  is uncertain.
