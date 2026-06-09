---
name: profile-summary-unified
description: Define a single canonical JSON schema that merges outputs from e2e-bench-runner, nsys-timeline-sql, pytorch-profiling, and any framework-specific torch profiler into one profile_unified.json. Carries an evidence_chain that records which field came from which skill, so downstream agents (especially coding agents reading handoff prompts) have a single, auditable source of truth.
version: 0
stage: [1, 2, 3]
inputs:
  - bench_summary:       optional path to e2e-bench-runner's bench_summary.json
  - timeline_summary:    optional path to nsys-timeline-sql's timeline_summary.json
  - torch_profile_text:  optional path to a torch.profiler text summary (vLLM-style)
  - sglang_profile:      optional path to pytorch-profiling's profile_summary.json
  - subject:             dict — what was profiled: framework, model, server config, hardware
  - workload:            dict — regime_id, regime_yaml_sha, etc.
  - out:                 path for profile_unified.json
outputs:
  - profile_unified.json   (per schema/profile_unified.schema.json)
triggers:
  - "End of any investigation cycle, before writing a handoff: produce one canonical artifact summarising all the profiling done."
  - "When the analysis agent wants to hand off to a coding agent — the unified summary is the basis of the handoff's `Evidence chain` section."
  - "When comparing two investigations (baseline vs. patched): unify each, then diff."
depends_on:
  - e2e-bench-runner
  - nsys-timeline-sql
  - pytorch-profiling
---

# profile-summary-unified

## WHEN

Concrete conditions:

1. **End of an investigation**, before writing the conclusion doc or handoff.
2. You have **at least 2 different profiling skill outputs** to merge.
   (If you only have one, just read that file — this skill is overhead for a
   single source.)
3. You want a **traceable evidence chain** — the resulting JSON's
   `evidence_chain` field records which numeric field originated from which
   skill / source file. This is what makes "skill attribution" in reports
   mechanical.

## WHY (the failure mode this prevents)

Two real problems we hit on 2026-06-09 CUTLASS investigation:

1. **4 different output formats glued by hand**. The investigation pulled from
   `bench_summary.json` (e2e-bench-runner), torch.profiler text (custom regex),
   `microbench results_2026-06-08.md` (manual table), and sglang source line
   numbers (grep). Each was parsed inline in the writeup; if any field changed,
   the writeup would silently go stale. A unified intermediate makes the
   coupling explicit.

2. **No way to prove "this skill contributed"**. The mentor explicitly asked for
   "skill attribution" — which skill produced which finding? Without a unified
   schema, the proof lives only in the doc's prose, which is brittle.
   `evidence_chain` records this mechanically and machine-checkably.

## HOW

```bash
python .github/skills/profile-summary-unified/impl/unify.py \
    --bench-summary       results/.../bench/bench_summary.json \
    --timeline-summary    results/.../nsys/timeline_summary.json \
    --torch-profile-text  results/.../torch_trace/profiler_out_0.txt \
    --subject-yaml        experiments/2026-06-09/subject_cutlass.yaml \
    --workload-yaml       regimes/qwen3_30b_moe_default.yaml \
    --out                 results/.../profile_unified.json
```

All `--*` inputs are optional; the adapter pulls fields from whichever sources
are provided and marks missing ones as `null` (with a `gap_reason` in
`evidence_chain`).

## OUTPUT CONTRACT — `profile_unified.json`

See `schema/profile_unified.schema.json` for the JSON Schema. High-level shape:

```json
{
  "schema_version": 0,
  "ok": true,
  "captured_at": "2026-06-09T05:50:00Z",
  "subject": {
    "framework": "vllm",
    "framework_version": "0.22.1",
    "model": "qwen3-30b-a3b-moe",
    "config_summary": {
      "moe_backend": "flashinfer_cutlass",
      "cudagraph": true, "autotune": true, "tp_size": 1
    },
    "hardware": {"name": "NVIDIA H200", "sm": "9.0", "num_sms": 132}
  },
  "workload": {
    "regime_id": "R_medium",
    "regime_yaml_path": "regimes/qwen3_30b_moe_default.yaml",
    "regime_yaml_sha256": "..."
  },
  "e2e": {                         // ← from bench_summary.json
    "req_per_s": {"mean": 4.74, "stddev_pct": 0.0, "reliable": true},
    "tokens_per_s_mean": 1182,
    "e2e_p50_ms": 200, "e2e_p99_ms": 600,
    "completion_rate": 1.0
  },
  "gpu_macro": {                    // ← from nsys-timeline-sql
    "wall_ms": 3500.0,
    "gpu_active_ms": 3320.0,
    "gpu_idle_ms":    180.0,
    "gpu_util_pct":    94.8,
    "launch_count": 24672,
    "launch_ratio_graph_to_eager": 0.95,
    "verdict": "graph_dominated"
  },
  "kernel_breakdown": [             // ← merged from nsys-timeline-sql + torch.profiler
    {
      "category": "moe_gemm",
      "kernel_pattern": "fused_moe::run_global",
      "self_ms": 1040.0, "self_pct": 31.55,
      "calls": 24672, "avg_us": 44.9,
      "source": "torch.profiler"
    },
    {"category": "dense_gemm", "kernel_pattern": "cutlass::device_kernel<GemmUniversal>", ...},
    {"category": "moe_routing", "kernel_pattern": "trtllm_kernels::*", ...}
  ],
  "kernel_micro": {                 // ← reserved for ncu; null until unlocked
    "available": false,
    "reason": "ncu not yet wired in (pending mentor permission unlock — see audit Gap)"
  },
  "evidence_chain": [               // ← THE attribution field
    {"field": "e2e.req_per_s",       "source_skill": "e2e-bench-runner",
     "source_file": "bench/bench_summary.json", "ok": true},
    {"field": "gpu_macro.gpu_util_pct", "source_skill": "nsys-timeline-sql",
     "source_file": "nsys/timeline_summary.json", "ok": true},
    {"field": "kernel_breakdown",   "source_skill": "torch.profiler+manual",
     "source_file": "torch_trace/profiler_out_0.txt", "ok": true,
     "note": "vllm-specific text format; categorization heuristic in unify.py"},
    {"field": "kernel_micro",       "source_skill": null,
     "source_file": null, "ok": false,
     "gap_reason": "ncu unavailable"}
  ],
  "warnings": []
}
```

`evidence_chain` is **the** field that downstream agents (including the
handoff-prompt-template skill's `Evidence chain` section) consume to write
correct attribution.

## WHICH METRIC HELPS WHICH PROBLEM

This skill **does not** add new metrics — it merges existing ones from upstream
skills. The metric-→-problem mappings continue to live in those upstream
SKILL.mds (`e2e-bench-runner`, `nsys-timeline-sql`, `pytorch-profiling`).

What changes is that an agent looking at one `profile_unified.json` can:
1. Quickly answer "what do we know?" by reading `e2e`, `gpu_macro`,
   `kernel_breakdown` in any order.
2. Quickly answer "what do we NOT know?" by scanning `evidence_chain` for
   entries with `ok: false` or `kernel_micro.available: false`.
3. Quickly cite sources by copying `evidence_chain` rows into a handoff.

## EXTENSION — adapter recipes

Different framework outputs need different parsers:
- vLLM torch.profiler text: regex-parsed (current implementation)
- sglang profile_summary.json: structured, direct copy
- nsys timeline_summary.json: structured, direct copy
- Future: NCU csv export, proton hatchet JSON

To add a new source, write a `_from_<source>()` function in `impl/unify.py` that
returns a list of `kernel_breakdown` rows + an `evidence_chain` entry. Keep
upstream contracts stable; this skill is a downstream sink, not a definer.

## FAILURE MODES

| Mode | Detection | Mitigation |
|---|---|---|
| All input files missing | input validation | `{"ok": false, "error": "no profile inputs given"}` |
| One source unparseable | per-source try/except | record `ok: false` + `gap_reason` in evidence_chain, continue |
| Conflicting metrics (e.g. e2e wall_ms ≠ nsys wall_ms by > 20%) | post-merge consistency check | emit warning, prefer e2e source for wall time |
| Schema version mismatch | future-proofing | reject upstream files with `schema_version > N` and refuse to extrapolate |

## ROADMAP

- **v1** — `diff` sub-command: given two `profile_unified.json`, emit a
  Markdown table with per-field delta + delta% (for paste-into-handoff or
  paste-into-PR).
- **v1** — JSON Schema validation step (`jsonschema` lib) before write.
- **v2** — when NCU is unlocked, add `_from_ncu_csv()` adapter and a
  `kernel_micro` filler.

## REFERENCES

- Upstream SKILL.mds: `e2e-bench-runner`, `nsys-timeline-sql`, `pytorch-profiling`
- Downstream consumer: `handoff-prompt-template` (the `Evidence chain` section)
- Real-world need this addresses: `docs/2026-06-09/cutlass_vs_triton_e2e_investigation.md` had a manual "skill attribution" table at the end — this skill makes that table the *machine-generated* basis instead.
