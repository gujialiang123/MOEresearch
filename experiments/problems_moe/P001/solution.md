# Solution for P001

**Auto-written by `config_agent.py --exhaustive`.**

**Problem**: R_scheduler_tail on `/data/hf/models/Qwen3-30B-A3B-Instruct-2507`.

**Strategy**: `S001` — knob `max-running-requests`. Rationale: If concurrency_capped, raising admission cap should drain the queue.

**Values swept**: [128, 96, 64]

**Baseline ttft_p95_ms**: 2282.48


## Per-value results

| Attempt | Value | Decision | New | Δ% | Violations | also_solved |
|---|---:|---|---:|---:|---|---:|
| attempt_003 | 128 | keep | 161.66 | +92.9 | 0 | 1 |
| attempt_002 | 96 | keep | 164.78 | +92.8 | 0 | 1 |
| attempt_001 | 64 | keep | 168.68 | +92.6 | 0 | 1 |

## Best attempt

**`attempt_001`** — set `max-running-requests = 64`

> **Note**: 3 values ([64, 96, 128]) give effectively the same improvement (within ±1%). Picked the smallest value to minimize memory/risk.

- ttft_p95_ms: **2282.48 → 168.68** (+92.6%)
- Decision: **keep**
- Constraint violations: none

- Side-solved problems:

  - `prefill_long` (ttft_p95_ms: +79.7%)


## Recommended config change

Edit `/home/t-jialianggu/work/EndtoEnd-auto-optimization/configs/moe_qwen3_30b.yaml` to set:

```yaml
max-running-requests: 64
```


## Reproducibility

```bash
# Apply the recommended fix and re-run target:
python scripts/run_experiment.py \
    --config /home/t-jialianggu/work/EndtoEnd-auto-optimization/experiments/problems_moe/P001/attempts/attempt_001/candidate_config.yaml \
    --workload /home/t-jialianggu/work/EndtoEnd-auto-optimization/experiments/problems_moe/P001/workload.yaml \
    --mode quick
```
