# regime-bench-harness — Layer-1 deterministic e2e bench harness

> **What this is.** A single command that takes one `bench-spec.yaml` and
> produces one schema-v1 `summary.json`. The agent decides *which* spec to run;
> this harness handles *what it actually does* — launch the server, wait for it,
> run the regime sweep, kill it on exit, emit a self-contained result.

## TL;DR

```bash
python harness/run_bench.py \
    --spec bench-specs/sglang-triton-bf16-baseline.yaml \
    --out-dir results/<experiment>/sglang-triton/
```

Exit codes: `0=ok`, `1=hard fail`, `2=quality gate failed`, `3=unreliable stddev`.

## Architecture (the layer Mason asked for)

```
┌──────────────────────────────────────────────────────────┐
│  Layer 2: Agent (LLM)                                    │
│  - Decides which bench-spec to run next                  │
│  - Reads summary.json, draws conclusions                 │
│  - Calls profiling skills (nsys, ncu) for deeper digs    │
└─────────────────┬────────────────────────────────────────┘
                  │  exec
                  ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 1: regime-bench-harness (PURE, no LLM)            │
│                                                           │
│  run_bench.py                                             │
│    ├── spec.py          BenchSpec + spec_hash             │
│    ├── env_snapshot.py  GPU / driver / cuda / lib / git   │
│    ├── lifecycle.py     port check → launch → /health     │
│    │                    → force-kill on exit              │
│    ├── executor.py      drive .github/skills/e2e-bench-   │
│    │                    runner/impl/run_bench.py          │
│    ├── quality.py       sanity gate (PPL deferred to v2)  │
│    └── output.py        SUMMARY_SCHEMA v1 + writer        │
└──────────────────────────────────────────────────────────┘
```

**Deterministic property**: same `bench-spec.yaml` + same referenced base config
+ same regimes YAML → same `spec_hash` → same expected output (modulo run-to-run
noise reported as `stddev_pct`).

## Inputs

A **bench-spec** is a YAML file under `bench-specs/`. See
[`bench-specs/_template.yaml`](../bench-specs/_template.yaml) for the full schema
and per-field docs. Quick mental model:

```yaml
submission_id: <kebab-case-id>                    # required, unique
description: "free-form prose, NOT in spec_hash"
tags: [free-form]                                  # NOT in spec_hash

server:
  config: configs/<base.yaml>                      # base flags
  overrides:                                       # per-experiment diff
    moe-runner-backend: triton
    _gpu_id: 4
  conda_env: sglang-dev
  health_url: http://127.0.0.1:31100/health
  base_url:   http://127.0.0.1:31100
  startup_timeout_s: 600

regimes:
  file: regimes/<workload-sweep.yaml>              # workload defs

bench:
  num_runs: 3                  # run 1 is always dropped (cold)
  reliable_stddev_pct: 8       # stddev_pct > this → reliable:false
  per_request_timeout_s: 600
  backend: sglang              # sglang | vllm (vllm support in v1.1)

quality_gate:
  type: sanity                 # sanity | none. ppl is v2
```

## Outputs

Inside `out_dir/`:

| file | what |
|---|---|
| `summary.json` | **the canonical result.** Schema v1. See `harness/output.py`. |
| `server.log` | stdout+stderr of the launched server |
| `server_config_used.yaml` | exact resolved flags after override merge |
| `regimes_resolved.yaml` | exact regimes dict after `only:` filter |
| `server.pid` | pid of the launched server (cleaned up on exit) |
| `bench_summary.json` | raw output of the underlying e2e-bench-runner skill |
| `per_run/<regime>_run<N>.json` | per-request records from each run |

### `summary.json` (schema v1) — key fields

```json
{
  "schema_version": 1,
  "ok": true,
  "submission_id": "...",
  "spec_hash": "sha256:<64hex>",
  "captured_at": "2026-06-11T18:00:00Z",
  "spec_resolved": {
    "server_config": { /* full resolved flags inline */ },
    "regimes":       { /* full regimes dict inline */ },
    "bench":         { "num_runs": 3, "backend": "sglang", ... },
    "quality_gate":  { "type": "sanity" }
  },
  "environment": {
    "hostname": "...",
    "gpu":      { "name": "NVIDIA H200", "id": 4, "sm": "90", "uuid": "..." },
    "driver":   "580.x",
    "cuda":     "12.8",
    "engine_version": { "sglang": "0.5.12.post1", "flashinfer": "0.6.3", ... },
    "git":      { "commit": "...", "dirty": false }
  },
  "server": { "startup_wall_s": 87.3, "first_health_at_s": 87.3, "log_path": "..." },
  "regimes": {
    "R_medium": {
      "num_prompts": 16, "concurrency": 8, ...,
      "req_per_s":    { "mean": 4.49, "stddev": 0.10, "stddev_pct": 2.2, "runs": [...] },
      "tokens_per_s": { ... },
      "e2e_ms":       { "p50": 1820, "p99": 2950, "count": 32 },
      "wall_s":       { ... },
      "completion_rate": 1.0,
      "reliable": true
    }
  },
  "quality_gate": { "type": "sanity", "passed": true, "checks": {...} },
  "warnings": []
}
```

If `ok: false`, an `error: {phase, message}` field is added. Schema validates
either way — downstream consumers never need `try/except` on shape.

## CLI reference

```
python harness/run_bench.py --spec PATH --out-dir PATH [flags]

  --dry-run         load spec, compute spec_hash, exit. No server launch.
                    Useful for pre-commit validation of new specs.
  --no-server-start assume server already running at spec.server.base_url
                    (skip launch + cleanup). Pair with manual --keep-server runs.
  --keep-server     don't kill server on exit. Use to attach nsys/ncu after.
```

## How `spec_hash` is computed

```
spec_hash = sha256(canonical_json({
  "stable_spec":            { submission_id, server.*, regimes.*, bench.*, quality_gate.type },
  "resolved_server_config": <merged base + overrides as dict>,
  "resolved_regimes":       <regimes YAML loaded + 'only' filter applied>
}))
```

**In** the hash: every flag actually fed to sglang, every regime parameter,
the structural spec fields.

**NOT in** the hash: `description`, `tags`, `_spec_path`. Edit prose freely
without invalidating prior runs.

## When NOT to use this harness

- **Profiling**: use `nsys-capture` / `ncu-microarch` against the server. Tip:
  run the harness with `--keep-server`, then run profiling tools against the
  surviving server, then `kill -- -$(cat <out_dir>/server.pid)` to clean up.
- **Custom one-off bench shapes** (non-standard prompts, streaming, etc.): call
  `.github/skills/e2e-bench-runner/impl/run_bench.py` directly with `--url`.
- **Server-side log mining**: use `server-log-mining` on `out_dir/server.log`
  after the harness finishes.

## Relation to existing skills

- **`e2e-bench-runner`** — unchanged; harness shells out to its CLI.
- **`regime-sweep-runner`** — deprecated 2026-06-11; pointed at harness.
- **`nsys-capture`, `ncu-microarch`** — independent; orthogonal to harness.
- **`profile-summary-unified`** — independent; reads its own inputs.

## v1 limitations (deferred to v2)

- **vLLM launcher**: `scripts/launch_server.py` is sglang-specific. Adding vLLM
  support means writing `scripts/launch_vllm.py` and a `bench.backend → launcher`
  dispatcher. For now, run vLLM by `--no-server-start` against a manually
  launched vLLM server.
- **PPL quality gate**: `quality_gate.type: ppl` is reserved but not implemented.
  v2 will use `lm-eval-harness`.
- **Profiling integration**: nsys / ncu remain separate skills. v2 may add
  `bench.profile_levels: [nsys, ncu]` to capture during the bench window.

## See also

- `bench-specs/_template.yaml` — bench-spec schema reference (with comments)
- `bench-specs/sglang-triton-bf16-baseline.yaml` — production-grade example
- `bench-specs/sglang-cutlass-bf16-patched.yaml` — experimental spec used to
  validate the `model_runner.py:1841` 1-line patch (see
  `docs/2026-06-11/ofer_meeting_findings_draft.md` §8.6)
- `.github/skills/regime-bench-harness/SKILL.md` — agent-facing entry point
