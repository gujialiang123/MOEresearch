---
name: pytorch-profiling
description: Capture a sglang Torch profile (CPU+GPU activities) for a problem package's target workload, then reduce the trace to a structured profile_summary.json (top kernels, phase breakdown, MoE expert dispatch overhead, CUDA-graph fallback count).
version: 0
stage: [1, 2]   # used inside Stage A (problem-setter L4 evidence) and Stage B (solver diagnosis)
inputs:
  - candidate_config: path to sglang launch config (yaml)
  - workload:         path to workload yaml (defines bench_serving args)
  - profile_num_steps: int, default 10  (how many decode steps to profile)
outputs:
  - profile_summary.json   (structured aggregates — agent-readable)
  - raw_trace/*.json.gz    (raw torch profiler trace, optional)
triggers:
  - "Stage A: setter is preparing a problem package, has L1-L3 evidence, but root cause is not config-shaped (e.g. server log clean, neighbors don't bracket a config boundary, throughput drop > 20% with no obvious knob)."
  - "Stage B: solver's first config-agent sweep returned all reverts / needs_more_evidence on the suggested_strategies; the solver-orchestrator wants per-kernel data before dispatching to scheduler-agent or kernel-agent."
  - "Stage B: kernel-agent receives a problem package without an existing profile_summary.json — it MUST capture one before proposing source-level changes."
depends_on:
  - server-log-mining       # we still grab server_features.json from the profile run
---

# pytorch-profiling

## WHEN

Call this skill **only when L1–L3 evidence is insufficient**. The profile is
expensive (server restart + 10–30 s of overhead + multi-MB trace), so it must
be agent-decided, not always-on.

Concrete triggers:

1. **Stage A setter, L4 path** — setter has run server-log mining (L2) and
   failure classification (L3) on the symptom run, and:
   - `server_features.json` shows no max_running_requests cap, no KV pressure,
     no retracts; AND
   - the regime has no obvious axis-boundary explanation (neighbors don't
     bracket a cliff); AND
   - the gap vs. a closely related healthy regime is still ≥ 20 % on the
     primary metric.
   The setter packages the resulting `profile_summary.json` into
   `problem_package/evidence/profile_summary.json` so the solver doesn't
   have to redo it.

2. **Stage B solver, escalation path** — config-agent has swept all
   `suggested_strategies[*].values_to_try` and the best attempt did **not**
   pass `acceptance_criteria.required_improvement_pct`. The orchestrator
   captures a profile to decide between:
   - scheduler-agent (high % time in scheduler/coordinator threads)
   - workload-shape-agent (high % in tokenization or HTTP)
   - kernel-agent (single kernel dominates self-time, or MoE routing > 25 %)

3. **Stage B kernel-agent, mandatory** — kernel-agent will refuse to propose
   any source-level patch without a profile_summary.json captured against the
   exact `candidate_config` it is about to modify.

## WHY (the failure mode this prevents)

We learned three things from the 2026-05-28 MoE run:

1. The setter was good at finding **config-shaped** problems
   (`max-running-requests=32` cap) because L2 server-log mining surfaces them
   directly.
2. The setter has **no signal whatsoever** for kernel-shaped problems:
   bench_serving aggregates can't tell "slow because of bad scheduler" apart
   from "slow because attention kernel is mis-tiled for this seq-len".
3. Without per-kernel data, kernel-agent would be **guessing**, and guessing
   source changes is the fastest way to break a working sglang install.

The skill closes that gap: any agent that needs "what is actually consuming
GPU time during this workload?" can call it and get a structured answer
without writing trace-parsing code.

## HOW

Two-step procedure, both wrapped by `impl/run_profile.py`:

### Step 1 — Capture
```bash
python .github/skills/pytorch-profiling/impl/run_profile.py \
    --config        <candidate_config.yaml> \
    --workload      <workload.yaml> \
    --profile-num-steps 10 \
    --out-dir       <attempt_dir>/profile/
```

Internally:
- Sets `SGLANG_TORCH_PROFILER_DIR=<out-dir>/raw_trace` on the server's env.
- Launches the sglang server using the same launcher as
  `scripts/run_experiment.py` (so config parity is guaranteed).
- Waits for `/health` to be ready.
- Sends a small warmup volley (`--warmup-requests 16`) to bypass cold-start —
  this is the Finding-B lesson; profiles must measure steady-state, not warmup.
- Runs `sglang.bench_serving --profile --profile-num-steps N
  --profile-activities CPU GPU --profile-output-dir <out-dir>/raw_trace`.
- Tears down the server, stops the profile, locates the produced
  `*.pt.trace.json` file under `<out-dir>/raw_trace/`.

### Step 2 — Reduce
```bash
python .github/skills/pytorch-profiling/impl/parse_trace.py \
    --trace <out-dir>/raw_trace/*.pt.trace.json \
    --out   <out-dir>/profile_summary.json
```

Internally:
- Streams the JSON trace (`events` array, Chrome Trace format).
- Groups events by `name` and aggregates `dur` (us) into `self_time_us` and
  `total_time_us` (subtract children).
- Identifies phase via the sglang-emitted user-annotations
  (`Prefill`, `Decode`, `Schedule`, `Tokenize`) when present, otherwise via
  kernel-name heuristics (`flashinfer*decode*` → decode, etc.).
- Detects MoE routing overhead by summing time in `*moe*topk*`,
  `*moe*dispatch*`, `*all_to_all*`.
- Counts CUDA-graph fallback markers (`launch_kernel` events that occur
  inside a "Decode" range when a graph for that batch size should exist).

## OUTPUT CONTRACT — `profile_summary.json`

```json
{
  "schema_version": 0,
  "ok": true,
  "captured_at": "2026-05-28T18:00:00Z",
  "candidate_config_sha256": "<hash of yaml content>",
  "workload": {
    "regime_id": "scheduler_overhead_high_concurrency__con_64",
    "num_prompts": 64,
    "max_concurrency": 64,
    "profile_num_steps": 10
  },
  "totals": {
    "wallclock_ms": 1234.0,
    "gpu_active_ms": 980.0,
    "gpu_idle_ms":   254.0,
    "gpu_utilization_pct": 79.4
  },
  "phase_breakdown_pct": {
    "prefill":   18.2,
    "decode":    65.4,
    "scheduler": 11.0,
    "tokenize":   2.1,
    "other":     3.3
  },
  "top_kernels": [
    {"rank": 1, "name": "flashinfer::SinglePrefillWithKVCacheKernel",
     "self_time_pct": 22.1, "calls": 320, "avg_us": 350.0,
     "phase": "prefill"},
    {"rank": 2, "name": "fused_moe::moe_align_block_size",
     "self_time_pct": 14.8, "calls": 12800, "avg_us": 5.9,
     "phase": "decode"}
    /* up to top 20 */
  ],
  "moe_overhead": {
    "applicable": true,
    "topk_pct":     6.2,
    "dispatch_pct": 8.4,
    "all_to_all_pct": 0.0,
    "total_routing_pct": 14.6,
    "verdict": "moderate"   /* low<10, moderate<25, high>=25 */
  },
  "cuda_graph": {
    "captured_bs_range": [1, 32],
    "decode_steps_in_graph_pct":   71.0,
    "decode_steps_outside_graph_pct": 29.0,
    "fallback_reason_guess": "max_running_requests=32 capped capture range"
  },
  "warnings": [
    /* e.g. "trace truncated at 256 MB", "profile_by_stage unavailable on this build" */
  ]
}
```

If anything fails, emit:
```json
{ "schema_version": 0, "ok": false, "error": "<one sentence>" }
```

The agent that called the skill must check `ok` before consuming any other
field.

## FAILURE MODES

| Mode | Detection | Mitigation |
|---|---|---|
| Server didn't honor `SGLANG_TORCH_PROFILER_DIR` (build lacks profiler) | `raw_trace/` empty after run | `{"ok": false, "error": "no trace produced; rebuild sglang with profiler enabled"}` |
| Trace > 1 GB (long decode + many ops) | file size pre-check | reduce `--profile-num-steps` to 3 and retry once before failing |
| Phase annotations missing (older sglang) | no `Prefill`/`Decode` ranges in trace | fall back to kernel-name heuristics; emit warning |
| MoE model but moe_overhead all zeros | model is MoE per config, no moe_* kernels seen | warning "MoE routing not captured — likely fused into a single GEMM" |
| Cold-start dominated (Finding B) | first 30 % of trace has ≥ 2× kernel time of last 30 % | warning "cold-start tail detected; profile is biased toward warmup" |
| Single GPU profiled in TP > 1 deployment | server_features says tp_size > 1 | warning "tp_size=N but only GPU 0 profiled"; future: profile all ranks |

## ROADMAP

- **v1** — multi-GPU trace merge (one summary across all TP ranks)
- **v1** — distinguish CUDA-graph capture cost from steady-state launch cost
- **v2** — call `kineto` / `nsys` instead of pure-Python parsing for traces > 1 GB
- **v2** — auto-attach the profile capture to every Stage A run that finishes
  with `triage = "needs_l4_investigation"` (currently the agent must request)
- **v3** — diff mode: given two `profile_summary.json` (before/after a fix),
  emit the regression/improvement table per kernel so the solver-orchestrator
  can verify the fix landed in the expected kernel rather than elsewhere.
