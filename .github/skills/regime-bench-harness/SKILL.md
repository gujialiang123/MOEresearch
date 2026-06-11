---
name: regime-bench-harness
description: |
  Run a complete regime-based e2e benchmark from a single bench-spec.yaml. Manages
  server lifecycle (port check + launch + /health wait + force-kill), drives the
  underlying e2e-bench-runner skill across all regimes, applies a sanity quality gate,
  and writes a deterministic schema-v1 summary.json with spec_hash for replay.
  This is the Layer-1 "deterministic harness" — agent decides WHICH spec to run,
  harness decides WHAT it actually does.
version: 1
stage: [1, 2, 3]
inputs:
  - spec:    path to a bench-spec.yaml under bench-specs/
  - out_dir: where to write summary.json + per_run/ + server.log
outputs:
  - summary.json   schema v1 (see harness/output.py SUMMARY_SCHEMA)
  - server.log     stdout+stderr from the launched server
  - per_run/<regime>_run<N>.json   raw per-request records
  - regimes_resolved.yaml          the actual regimes dict used (post 'only:' filter)
  - server_config_used.yaml        exact server flags after override merge
triggers:
  - "User wants to bench a specific (model, engine, backend, dtype) config end-to-end"
  - "Reproducing a past experiment from its spec_hash"
  - "N-way comparison: write N bench-specs, call this skill N times"
  - "Replaces ad-hoc multi-step bench scripts that manually launch_server + run_benchmark"
depends_on: [e2e-bench-runner]
---

# regime-bench-harness

## WHEN

Use this skill whenever the prior step would otherwise be:

> "Start the sglang server with these flags, wait for it, run the bench across these
> regimes, save results somewhere, then kill the server."

That's exactly what this skill encapsulates. The agent only decides **what's in the
bench-spec** (which is the experimental hypothesis); the harness handles
launch/wait/bench/kill deterministically.

Do **not** use this skill for:

- Profiling (nsys / ncu): use `nsys-capture` + `ncu-microarch` against the server
  this harness leaves running with `--keep-server`.
- Server-side log mining: use `server-log-mining` on `out_dir/server.log` after.
- Custom one-off bench shapes: use `e2e-bench-runner` directly with `--url`.

## WHY

Three concrete failure modes this prevents:

1. **Manual command-line drift.** Hand-typing `sglang.launch_server --moe-runner-backend X
   --enable-Y --port Z` across 4 experiments → typo, results not reproducible. Spec file fixes that.
2. **Server leak.** "Bench died, sglang still on GPU." The harness force-kills on
   any exit path (success, fail, Ctrl-C).
3. **Spec drift.** "I'm comparing run-from-yesterday vs run-from-today but did I
   change the model path?" `spec_hash` in `summary.json` answers this in 1 second.

See also `docs/2026-06-08/agent_profiling_capability_audit.md` Part E for the
historical mistakes that motivated this.

## HOW

```bash
python harness/run_bench.py \
    --spec bench-specs/sglang-triton-bf16-baseline.yaml \
    --out-dir results/<experiment>/<config>/
```

Optional flags:

- `--dry-run`: load spec, compute spec_hash, exit. Use to validate a spec without
  launching anything (good for `pre-commit`-style checks on edited specs).
- `--no-server-start`: assume server already running at `spec.server.base_url`.
  Use to bench a server you launched yourself with extra debug flags.
- `--keep-server`: don't kill the server on exit. Use to attach nsys/ncu after.

Exit codes (matters for shell scripting):

- `0` — bench OK + quality gate passed + all regimes reliable
- `1` — hard failure (spec invalid / server didn't start / executor crashed)
- `2` — bench completed but quality gate failed
- `3` — bench OK, quality OK, but stddev_pct unreliable on at least one regime

## OUTPUT CONTRACT

See `harness/output.py::SUMMARY_SCHEMA` (jsonschema draft-2020-12).

Critical fields downstream tools read:

- `spec_hash` — sha256, includes resolved base config + regimes content
- `regimes.<id>.req_per_s.mean` — primary throughput metric
- `regimes.<id>.reliable` — `false` ⇒ don't quote the number
- `quality_gate.passed` — gate sentinel
- `environment.engine_version.{sglang,flashinfer,vllm}` — what library versions ran
- `environment.git.commit` — what commit of THIS repo ran
- `spec_resolved.server_config` — full resolved YAML inline (self-contained replay)

## RELATION TO OTHER SKILLS

This skill **wraps** `e2e-bench-runner`:

- `e2e-bench-runner` = run bench against an already-running URL
- `regime-bench-harness` = manage server lifecycle + run bench + verify + cleanup

Prefer the harness for any "real experiment." Use `e2e-bench-runner` directly only
for unusual lifetimes (already-running server, custom regime shapes, etc.).

This skill **supersedes** the deprecated `regime-sweep-runner` (which had similar
goals but no spec-hash determinism, no server lifecycle, no quality gate).
