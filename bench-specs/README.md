# bench-specs/

Bench-specs are the deterministic inputs to `python harness/run_bench.py`. Each
file describes **one server config × one regime set** and produces one
`summary.json` per run.

> See `harness/README.md` for the harness architecture, schema v1, and CLI.

## Quick start

```bash
# Run a spec
python harness/run_bench.py \
    --spec bench-specs/sglang-triton-bf16-baseline.yaml \
    --out-dir results/2026-06-11_harness/sglang-triton/

# Validate a spec without launching anything
python harness/run_bench.py \
    --spec bench-specs/_template.yaml \
    --out-dir /tmp/dry \
    --dry-run

# N-way sweep
for s in bench-specs/sglang-*.yaml; do
    name=$(basename "$s" .yaml)
    python harness/run_bench.py --spec "$s" --out-dir "results/sweep/$name/"
done
```

## Writing a new spec

1. Copy [`_template.yaml`](_template.yaml) to `bench-specs/<your-id>.yaml`.
2. Fill in `submission_id` (kebab-case, unique).
3. Decide the **base** server config under `configs/` to reference.
4. List only the **diff** under `server.overrides` (backend flag, GPU id, port, etc.).
5. Pick the regimes YAML under `regimes/` (or write a new one).
6. Run with `--dry-run` first to validate.

## Naming convention

`<engine>-<backend>-<dtype>-<variant>.yaml`, lowercase, kebab-case:

- `sglang-triton-bf16-baseline.yaml`
- `sglang-cutlass-bf16-patched.yaml`
- `sglang-triton-fp8-baseline.yaml` (when we add fp8)

Matching `submission_id` inside the file. The `submission_id` is what shows up
in `summary.json` and the only thing the harness uses to identify the run.

## Available specs (2026-06-11)

| spec | purpose | requires |
|---|---|---|
| `sglang-triton-bf16-baseline.yaml` | sglang default config; the "control" any comparison runs against | nothing |
| `sglang-cutlass-bf16-patched.yaml` | validate the `model_runner.py:1841` 1-line patch unblocks cutlass on H200 | sglang source patch (see spec description) |

## What goes in `server.overrides` vs `server.config`

- **`server.config`** (the referenced base YAML in `configs/`): things that are
  stable across many experiments — model path, served-model-name,
  context-length, mem-fraction-static, scheduling defaults, host.
- **`server.overrides`**: things you're *experimenting* with — the backend flag,
  the GPU id, the port, cudagraph on/off, autotune on/off.

This split keeps spec files short ("here's what's different from baseline") and
makes it cheap to update the baseline for everybody (edit `configs/<base>.yaml`
once → all specs reflect it after re-hashing).

## What affects `spec_hash` (and what doesn't)

In the hash (changes invalidate prior runs):
- `submission_id`
- `server.config` *content* (the base YAML, dereferenced)
- `server.overrides`
- `server.conda_env`, `server.health_url`, `server.base_url`, `server.startup_timeout_s`
- `regimes.file` *content* (the regimes YAML, dereferenced)
- `regimes.only` (subset filter)
- `regimes.inline` (when used instead of `file`)
- `bench.{num_runs,reliable_stddev_pct,per_request_timeout_s,backend}`
- `quality_gate.type`

NOT in the hash (free to edit):
- `description`
- `tags`
- File names / paths (only the base-name + content is hashed, not the path)

## Port conventions (avoid collisions)

We use `31000+` for harness-managed servers to avoid clashing with hand-run
sglang servers (which default to `30000`):

| spec | port |
|---|---|
| sglang-triton-bf16-baseline | 31100 |
| sglang-cutlass-bf16-patched | 31101 |
| (future) sglang-triton-fp8-baseline | 31102 |
| ... | ... |

Pick an unused port for new specs. If you launch the same spec on two GPUs at
once, override the port via `server.overrides.port` (and `server.{health,base}_url`)
in your local copy.
