"""Helper consumed by suspicion-scoring and any downstream skill."""
from __future__ import annotations

import json
from pathlib import Path


def load_baseline(path: str | Path | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f).get("metrics") or {}


def cv_pct(baseline: dict, metric: str) -> float:
    return float((baseline.get(metric) or {}).get("cv_pct") or 0.0)


def adjusted_threshold(metric: str, base_threshold: float,
                       baseline: dict, k: float = 2.0) -> float:
    """Return a noise-aware floor for a 'ratio' threshold.

    `base_threshold` is e.g. 3.0 meaning "p99/p50 >= 3 is suspicious".
    If a metric's CV is large, the threshold floor rises so noise alone
    doesn't trigger a false positive.
    """
    floor = 1.0 + (k * cv_pct(baseline, metric) / 100.0)
    return max(base_threshold, floor)
