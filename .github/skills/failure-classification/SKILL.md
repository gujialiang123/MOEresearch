---
name: failure-classification
description: Classify a benchmark run failure (or near-failure) into a typed category that downstream scoring/agents can branch on.
version: 1
stage: [1, 2, 3]
inputs:
  - metrics: parse_metrics.py output (or null on missing file)
  - server_features: server-log-mining output
  - bench_log_text: optional, for benchmark-side errors
outputs:
  - classification.json
triggers:
  - "every benchmark run, regardless of pass/fail"
depends_on: [server-log-mining]
---

# failure-classification

## WHEN

Call after `parse_metrics.py` and `server-log-mining` for every run. Even
passed runs go through this skill so that `near_failure` and `partial_success`
can be detected.

## WHY

The v0.2 pipeline only knew three failure modes: `oom`, `server_crash`,
`timeout`. But H200's giant memory lets a run "succeed" while burning warning
signs that any human would call a near-failure:

- `peak_token_usage >= 0.9` (KV pool almost full)
- `retract_events > 0` (sglang dropped in-flight requests)
- `success_rate < 1.0` (some requests errored but rest completed)
- `peak_queue_reqs > max_running_requests` (concurrency cap shed load)

These need a name so agents can branch on them.

## HOW

Pure Python; no IO beyond reading the two input JSONs. Implementation:
`impl/classify.py`.

```python
def classify(metrics: dict | None, features: dict, bench_log: str = "") -> dict:
    if metrics is None or not metrics.get("passed", False):
        if features.get("oom_events", 0) > 0:           return "oom"
        if features.get("crash_events", 0) > 0:         return "server_crash"
        if metrics and metrics.get("timeout"):          return "benchmark_timeout"
        if metrics and metrics.get("parse_error"):      return "parse_error"
        return "unknown_failure"
    # Passed but near-failure:
    if features.get("at_capacity"):                     return "near_failure_kv"
    if features.get("retract_events", 0) > 0:           return "near_failure_retract"
    if (metrics.get("success_rate") or 1.0) < 1.0:      return "partial_success"
    if features.get("concurrency_capped"):              return "load_shed_concurrency"
    return "clean_pass"
```

## OUTPUT CONTRACT

```json
{
  "schema_version": 1,
  "classification": "load_shed_concurrency",
  "_classification_enum": [
    "clean_pass",
    "near_failure_kv", "near_failure_retract", "partial_success",
    "load_shed_concurrency",
    "oom", "server_crash", "benchmark_timeout", "parse_error",
    "unknown_failure"
  ],
  "evidence": {
    "passed": true,
    "success_rate": 1.0,
    "peak_token_usage": 0.00,
    "retract_events": 0,
    "concurrency_capped": true,
    "max_running_requests": 32,
    "peak_queue_reqs": 36
  }
}
```

## FAILURE MODES

- Missing both `metrics` and `features` → `classification = "unknown_failure"`
  with `evidence.reason = "no_data"`.
- Disagreement: metrics says passed but server_features has crash_events>0 →
  prefer `server_crash` (server log is more authoritative than bench_serving).

## ROADMAP

- v2: introduce a `score` (0–1) on top of the categorical label so partial
  failures rank within their category.
- v2: detect "tokenizer error" vs "model error" subtypes inside `parse_error`.
