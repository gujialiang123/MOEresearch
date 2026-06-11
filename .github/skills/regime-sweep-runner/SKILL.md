---
name: regime-sweep-runner
description: |
  [DEPRECATED 2026-06-11] Replaced by `regime-bench-harness`. The new harness wraps
  e2e-bench-runner with server lifecycle + deterministic spec_hash + quality gate,
  and N-way sweeps become a simple shell loop over N bench-specs. Do not invoke
  this skill in new work; left in place only for archival comprehension of old runs.
version: 0
stage: []
inputs: []
outputs: []
triggers:
  - "DO NOT USE — point user at regime-bench-harness instead."
depends_on:
  - e2e-bench-runner
---

# regime-sweep-runner (DEPRECATED)

> ⚠️ **Deprecated 2026-06-11.** Use [`regime-bench-harness`](../regime-bench-harness/SKILL.md)
> instead. The harness gives you:
>   - Server lifecycle (this skill required servers running already)
>   - `spec_hash` reproducibility anchor (this skill had none)
>   - Sanity quality gate (this skill skipped correctness)
>   - Schema-v1 `summary.json` (this skill had its own ad-hoc format)
>
> For N-way comparisons, write N bench-specs under `bench-specs/` and shell loop:
>   ```bash
>   for s in bench-specs/sglang-*.yaml; do
>     python harness/run_bench.py --spec "$s" --out-dir "results/sweep/$(basename "$s" .yaml)/"
>   done
>   ```

The original WHEN/WHY/HOW below is preserved for archival reading only.

---



# regime-sweep-runner

## WHEN

Concrete conditions:

1. **You have ≥2 configs to compare** AND **≥3 regimes worth checking**. Below that
   threshold, call `e2e-bench-runner` directly — this orchestrator is overhead.
2. **The configs' servers exist already** (this skill does NOT start/stop servers;
   that's the caller's job, because bringing up a vLLM/sglang server takes 5+ min
   and is best done with the user's awareness).
3. **You need a 2-D view** (configs × regimes) for cross-regime anomaly detection
   (which is what the next skill, `cross-regime-anomaly`, consumes).

Do NOT call when:
- You only want one number (use `e2e-bench-runner`).
- The configs need different launch args or different GPUs — the configs file
  is just a `{tag,url,backend}` list; if your scenario is more complex, drive
  `e2e-bench-runner` in a loop yourself.

## WHY (the failure mode this prevents)

Two documented mistakes from past CUTLASS investigation:

1. **2026-06-04 R_medium-only conclusion**: declared "sglang Triton 20% slower
   than vLLM Triton" based on R_medium alone. R_short showed the opposite.
   Cross-regime view would have caught it. **Skill rule: a config comparison
   that only looked at one regime is suspect by construction.**

2. **2026-06-09 vllm CUTLASS gap "in noise"**: found CUTLASS ≈ Triton on R_medium
   but didn't compare across regimes systematically — only noticed afterwards
   that R_short flipped the winner. A sweep makes this immediately visible.

## HOW

```bash
python .github/skills/regime-sweep-runner/impl/sweep.py \
    --configs-file  experiments/2026-06-10/configs.yaml \
    --regimes-file  regimes/qwen3_30b_moe_default.yaml \
    --num-runs      3 \
    --out-dir       results/2026-06-10_sweep/
```

The skill iterates configs serially (in declaration order), calling
`e2e-bench-runner` for each. **Each config gets its own subdirectory** with
the full bench artifacts; the top-level `regime_sweep_summary.json` is
a flattened matrix.

## INPUT — `configs_file` schema

```yaml
configs:
  - tag:       vllm_cutlass
    url:       http://127.0.0.1:30001
    backend:   vllm
    notes:     "moe_backend=flashinfer_cutlass, cudagraph ON, autotune ON"
  - tag:       vllm_triton
    url:       http://127.0.0.1:30001       # same port — caller restarted server between cells
    backend:   vllm
    notes:     "moe_backend=triton"
  - tag:       sglang_cutlass
    url:       http://127.0.0.1:30000
    backend:   sglang
    notes:     "moe_runner_backend=flashinfer_cutlass, cudagraph OFF (default)"
```

`notes` is free-form — agent should put enough server-config info that the
result is interpretable later without digging.

## OUTPUT CONTRACT — `regime_sweep_summary.json`

```json
{
  "schema_version": 0,
  "ok": true,
  "captured_at": "2026-06-10T05:00:00Z",
  "configs_file": "experiments/.../configs.yaml",
  "regimes_file": "regimes/qwen3_30b_moe_default.yaml",
  "regimes": ["R_short", "R_medium", "R_long"],
  "configs": ["vllm_cutlass", "vllm_triton", "sglang_cutlass"],
  "matrix": {
    "vllm_cutlass": {
      "ok": true,
      "notes": "moe_backend=flashinfer_cutlass, cudagraph ON, autotune ON",
      "regimes": {
        "R_short":  {"req_per_s_mean": 3.29, "stddev_pct": 0.1, "reliable": true,
                     "tokens_per_s_mean": 211, "e2e_p50_ms": 350, "e2e_p99_ms": 720},
        "R_medium": {"req_per_s_mean": 4.74, "stddev_pct": 0.0, "reliable": true, ...},
        "R_long":   {...}
      }
    },
    "vllm_triton":   {"ok": true, "regimes": {...}},
    "sglang_cutlass":{"ok": false, "error": "server not ready at http://...:30000"}
  },
  "warnings": ["sglang_cutlass row is incomplete — server health check failed"]
}
```

Failures of individual cells (`"ok": false`) do NOT abort the sweep — the matrix
continues with the remaining cells, and the failure is recorded in-place. This
makes the skill robust for long-running orchestration.

## WHICH METRIC HELPS WHICH PROBLEM

The matrix output is meant to be **read row by row, then column by column**.

### Reading row-by-row (one config across regimes)

| Pattern in a single row | What it suggests |
|---|---|
| Steady req/s across regimes | Config is workload-insensitive — good news. |
| req/s drops sharply at one concurrency level | KV-cache cliff or scheduler cap — call `server-log-mining`. |
| stddev_pct > 8 on only one regime | That regime is reliability-unstable; re-run before concluding. |
| reliable=false everywhere | Server is noisy host or undersized hardware. |

### Reading column-by-column (one regime across configs)

| Pattern in a single column | What it suggests |
|---|---|
| All configs within 5% | This regime doesn't discriminate; pick another for kernel-level study. |
| Config A wins by 2× | Strong signal — proceed to profiling with `nsys-capture`. |
| Config A wins by 10% | Borderline — increase num_runs to 5 before drawing a conclusion. |

### Reading the full matrix

| Pattern across the whole matrix | What it suggests |
|---|---|
| Winning config changes per regime | **Regime-dependent inversion** — escalate to `cross-regime-anomaly`. The "best" config depends on workload. |
| All configs converge on R_short, diverge on R_long | Bottleneck shifts from compute → memory/scheduling as work scales. |
| One config dominates everywhere | Safe to declare it the winner and proceed to optimization on top of it. |

## METHODOLOGY — predict-then-verify

Before the sweep, agent writes (in plan.md):

> "I expect config X to win on regime Y by ≥Z%, because <reason>.
>  I do NOT expect winners to change across regimes; if they do, the kernel
>  improvement direction must be regime-aware."

After the sweep, compare. If winners changed across regimes that the agent
expected to be uniform, **stop and re-think the hypothesis** before doing any
profiling.

## EXTENSION

- For >5 configs or >5 regimes, the wall time blows up (each cell takes
  ~30s + per-server warmup). Use `--cells <config1>:<regime1>,...` (planned v1)
  to subset the matrix.
- The skill is intentionally framework-agnostic about server lifecycle. If you
  want auto-start/stop, wrap this skill in a higher-level orchestrator that
  shells the launch scripts.

## FAILURE MODES

| Mode | Detection | Mitigation |
|---|---|---|
| Configs YAML missing required fields | YAML load + validation | `{"ok": false, "error": "configs entry missing 'url'"}` |
| One server unreachable | per-cell e2e-bench-runner fails with `"server not ready"` | record as cell failure, continue |
| All cells fail | post-aggregation check | `{"ok": false, "error": "all sweep cells failed"}` |
| Cell wall time > 10× expected | watchdog (v1) | for now, user must `Ctrl-C`; partial matrix is preserved |

## ROADMAP

- **v1** — `--cells` flag to run subset (matrix-cell-by-cell instead of full grid).
- **v1** — `--restart-script <path>` callback to bring up each config's server before bench.
- **v2** — parallel cells (when configs run on different GPUs).

## REFERENCES

- Past hand-rolled equivalent: `results/4way_bench/scripts/run_bench_4way.py` (4 configs × 3 regimes)
- The 2026-06-04 R_medium-only conclusion: `docs/2026-06-08/vllm_autotune_e2e_impact.md` (deprecated)
- Cross-regime view as anomaly source: `regime_scout/outputs/regime_map.md`
