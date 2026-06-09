# Example regime sets for e2e-bench-runner --regimes-file

This directory holds **reusable** regime definitions that any agent can pass
to `e2e-bench-runner` or `regime-sweep-runner`. Naming convention:

```
regimes/<model-class>_<purpose>.yaml
```

Examples:
- `qwen3_30b_moe_default.yaml` — the same 3 regimes (short/medium/long) built into
  e2e-bench-runner, but as a file so they can be edited / extended.
- `qwen3_30b_moe_concurrency_sweep.yaml` — fixed shape, varying concurrency
  from 1 to 32. Use this to find scheduler / KV-cache cliffs.
- `qwen3_30b_moe_seq_len_sweep.yaml` — fixed concurrency, varying input length
  from 200 to 8000. Use this to characterize prefill vs decode dominance.

## Why externalize them
- The same workloads must be runnable across configs (cutlass vs triton, sglang
  vs vllm) — externalizing them prevents the "I changed the bench by accident"
  trap.
- `regime_scout/` already has 8 candidate regimes (`R1`–`R8`); these YAMLs are
  the bridge between that taxonomy and the runner.
