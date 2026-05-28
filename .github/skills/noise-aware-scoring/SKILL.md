---
name: noise-aware-scoring
description: Compute per-metric coefficient-of-variation from N repeated baseline runs, and provide noise-adjusted thresholds for downstream scoring.
version: 1
stage: [1, 3]
inputs:
  - baseline_metrics: list of normalized metrics JSONs from N repeats of one workload
outputs:
  - noise_baseline.json
  - utility function `adjusted_threshold(metric_name, base_threshold) -> float`
triggers:
  - "before first scoring pass in a session"
  - "after every 20 experiments (drift re-check)"
depends_on: []
---

# noise-aware-scoring

## WHEN

- One-time per (model, server config, hardware) tuple at the start of a
  scouting session.
- Re-run after large fleet changes or every 20 experiments to detect drift.

## WHY

The v0.2 scoring used a hard threshold `ttft_p99/p50 ≥ 3.0` to declare "high
tail". On Qwen3-0.6B + H200, **9 of 10 workloads naturally exceed 3.0** —
not because they're suspicious, but because the model is so small that even
trivial warmup variance dominates the distribution. The threshold was
mis-calibrated and the score function saturated.

The fix is empirical: run the same baseline workload N times, measure the
coefficient of variation (CV = std / mean) per metric, and use
`max(base_threshold, k * CV)` as the actual decision boundary.

## HOW

Implementation: `impl/calibrate_noise.py`.

```bash
# CLI
python .github/skills/noise-aware-scoring/impl/calibrate_noise.py \
    --config configs/base.yaml \
    --workload regime_scout/candidates/seed_00_smoke.yaml \
    --repeats 5 \
    --out experiments/noise_baseline.json
```

The skill spins the server once and runs the same `bench_serving` invocation
N times back-to-back (after one warmup pass). For each numeric metric in
`parse_metrics.py`'s schema, it records mean, std, CV, min, max.

Downstream scorers import `adjusted_threshold(metric, base)` from
`impl/threshold.py`:

```python
def adjusted_threshold(metric: str, base_threshold: float,
                       baseline: dict, k: float = 2.0) -> float:
    cv_pct = (baseline.get(metric, {}).get("cv_pct") or 0.0)
    # Express CV-derived floor in the same units as base_threshold (which is
    # a ratio, e.g. 3.0 for tail ratio). 1 + k*CV ≈ "k sigma above mean".
    cv_floor = 1.0 + (k * cv_pct / 100.0)
    return max(base_threshold, cv_floor)
```

## OUTPUT CONTRACT

```json
{
  "schema_version": 1,
  "ok": true,
  "config": "configs/base.yaml",
  "workload": "regime_scout/candidates/seed_00_smoke.yaml",
  "repeats": 5,
  "metrics": {
    "ttft_p50_ms":        {"n": 5, "mean": 24.1, "std": 0.8, "cv_pct": 3.3, "min": 23.2, "max": 25.0},
    "ttft_p95_ms":        {"n": 5, "mean": 99.0, "std": 6.2, "cv_pct": 6.3, ...},
    "output_throughput":  {"n": 5, "mean": 630, "std": 9, "cv_pct": 1.4, ...}
  }
}
```

## FAILURE MODES

- A repeat run fails (oom/crash) → that repeat is excluded; `n` < repeats.
- All repeats fail → `ok=false`, callers fall back to default thresholds.

## ROADMAP

- v2: per-regime baselines (one calibration per regime hint, not just smoke).
- v2: percentile-based noise (use IQR instead of std for tail-heavy metrics).
- v3: continuous drift estimation — keep a rolling window of "best.yaml"
  reruns and alert on CV change.
