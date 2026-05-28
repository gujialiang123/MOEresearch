---
name: boundary-expansion
description: Given one workload + a hypothesized axis (input_len / max_concurrency / output_len), generate N neighbor workload YAMLs along that axis to probe for nonlinear regime boundaries.
version: 1
stage: [1]
inputs:
  - parent_workload: a single workload YAML
  - axis: one of {input_len, output_len, max_concurrency, num_prompts}
  - search_space: regime_scout/search_space.yaml (provides allowed axis values)
outputs:
  - one workload YAML per neighbor
triggers:
  - "after seed suite finishes, when a regime has ≥1 suspicious passed run"
  - "when local_nonlinearity component fires score < 0.1 on a regime (insufficient neighbors)"
  - "when a hint cluster has only one member"
depends_on: []
---

# boundary-expansion

## WHEN

Call this skill in the second wave of a Stage 1 explore loop. The first wave
runs seeds (one workload per regime hint). The second wave looks at first-wave
results, picks the axis most likely to reveal a nonlinearity, and expands.

Concrete triggers (encoded in `explore_regimes.py`):

1. The seed run **passed** AND its `concurrency_capped` flag is True → expand
   along `max_concurrency` to bracket the cliff.
2. The seed run **passed** AND its `cuda_graph_too_small` is True → same.
3. The seed run **passed** AND `peak_token_usage > 0.5` → expand along
   `input_len` × `max_concurrency` upward (probe near KV pressure).
4. The seed run **failed** → expand DOWNWARD on the dominant axis to find a
   minimal repro (or escalate to `minimal-repro-shrink`).
5. The seed run is **lonely** (only member of its regime hint cluster) → expand
   along the axis that defines the hint (prefill_long → input_len; decode_heavy
   → output_len; scheduler_tail → max_concurrency).

## WHY

The 2026-05-28 run failed to flag Finding A's nonlinearity (TTFT 99 → 434 ms
between mc=16 and mc=64) because we had ONE point at mc=16 and ONE at mc=64.
`local_nonlinearity` can't trigger without intermediate neighbors.

This skill exists to make `local_nonlinearity` useful by guaranteeing
**every regime hint has ≥ 3 points along each interesting axis**, generated
adaptively rather than as a pre-baked grid (which would waste budget).

## HOW

Implementation: `impl/expand.py`.

```bash
python .github/skills/boundary-expansion/impl/expand.py \
    --parent regime_scout/candidates/seed_03_scheduler_overhead_high_concurrency.yaml \
    --axis max_concurrency \
    --strategy bracket \
    --search-space regime_scout/search_space.yaml \
    --neighbors-out regime_scout/candidates/expanded/
```

Strategies:

| name | what it does |
|---|---|
| `bracket`   | If parent value = V, pick the 2 nearest **smaller** and 2 nearest **larger** values from search_space.axes[axis].values, drop V itself. Use when the parent is suspicious and we want to bracket the cliff. |
| `geometric` | log-uniform spacing between min and max of axis. Use for first-time probing of an axis. |
| `downward`  | Pick values strictly smaller than parent. Use after a failure (shrink). |
| `upward`    | Strictly larger. Use to find capacity limits. |

The output yaml inherits everything from parent and overrides only the
axis field. Filenames: `expanded/<parent_stem>__<axis>_<val>.yaml`.

## OUTPUT CONTRACT

For each generated neighbor:

```yaml
name: "scheduler_overhead_high_concurrency__mc_48"
regime_hint: "scheduler_tail"          # inherits
source:
  generated_by: "boundary-expansion"
  parent: "regime_scout/candidates/seed_03_..."
  axis: "max_concurrency"
  axis_value: 48
  strategy: "bracket"
dataset: { ... copy from parent ... }
traffic:
  max_concurrency: 48                  # only this changed
  num_prompts: 320                     # copied
cache: { ... copy ... }
seed: 1234
```

A summary JSON is also emitted to `--summary-json`:

```json
{
  "ok": true,
  "parent": "...",
  "axis": "max_concurrency",
  "strategy": "bracket",
  "generated": [
    {"path": "...mc_32.yaml", "axis_value": 32},
    {"path": "...mc_48.yaml", "axis_value": 48},
    ...
  ]
}
```

## FAILURE MODES

- Axis not in search_space → `ok=false`, no neighbors written.
- Parent value already at the extreme end → `bracket` returns whatever it
  can (1 or 2 neighbors instead of 4) with a warning.
- Generated neighbor would duplicate an existing candidate file → skip + warn.

## ROADMAP

- v2: 2D expansion (e.g. expand input_len × max_concurrency jointly when both
  are implicated).
- v2: pruning — reject neighbors predicted by a simple model to be redundant.
- v3: budget-aware ordering — return neighbors sorted by expected information
  gain so the explore loop can stop early.
