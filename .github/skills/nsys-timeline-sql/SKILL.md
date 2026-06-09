---
name: nsys-timeline-sql
description: Reduce an nsys-exported SQLite database to a curated `timeline_summary.json` (GPU active/idle, top-10 kernels, top idle gaps, CPU launch counts, cudagraph vs eager ratio, memcpy stats). Also exposes a `query` sub-command for arbitrary read-only SQL plus a recipe library of pre-written diagnostic queries.
version: 0
stage: [1, 2, 3]
inputs:
  - sqlite_path:  path  (produced by nsys-capture; must come from `nsys export --type sqlite`)
  - out_dir:      path  (where timeline_summary.json is written)
  - stream_id:    int   (optional; default = the busiest CUDA stream)
  - top_n:        int   (default 10; how many top kernels / largest gaps to emit)
  - window_ns:    "start_ns,end_ns"  (optional; restrict analysis to a sub-window)
outputs:
  - timeline_summary.json     (≤10 KB; the only thing downstream skills should read by default)
  - recipes_used.json         (which recipe SQLs ran + their row counts; for debugging)
triggers:
  - "After nsys-capture has produced a .sqlite. This skill is what turns raw events into agent-readable numbers."
  - "When comparing two captures (before vs after a patch): run on each, then diff the resulting JSONs."
depends_on:
  - nsys-capture
---

# nsys-timeline-sql

## WHEN

Concrete conditions:

1. **`nsys_capture.json` exists with `ok: true`.** This skill consumes its `files.sqlite`.
2. **You have a specific question** — even if it's "which kernel dominates?", that's
   specific. Don't call this skill for "tell me everything about the profile" — the
   default summary is intentionally small. For deep dives, use the `query` sub-command.
3. **You're verifying a patch.** Call once on baseline.sqlite, once on patched.sqlite,
   then diff the two `timeline_summary.json` files. The skill ships a `diff` sub-command
   that does this.

## WHY (the failure mode this prevents)

We made **three** documented mistakes interpreting nsys output before this skill:

1. **Trusted `nsys stats cuda_gpu_kern_sum` blindly** (2026-06-05). It folds cudagraph
   replays into a single bucket, so AT_OFF_CG_ON showed only 0.42s GPU active when
   the real number (counted from `CUPTI_ACTIVITY_KIND_KERNEL` rows including graph
   nodes) was much higher. Wrong conclusion about "GPU mostly idle". **This skill
   queries the raw SQLite table, not the stats CSV, so graph nodes are included.**
2. **No idle-gap analysis** (2026-06-06). Looked at top kernel = 22% of GPU active,
   declared "GPU is well-utilized" — but missed that GPU was 60% idle. Top kernel
   was 22% of *active* time, not 22% of wall. **Skill rule: always report
   `gpu_active_ms` AND `gpu_idle_ms` AND their ratio.**
3. **No stream awareness** (2026-06-07). Summed kernel durations across all streams,
   double-counting parallel work. **Skill rule: aggregate per-stream, report streams
   separately, and only sum across streams when explicitly computing "GPU-busy-anywhere".**

## HOW

### Default mode — produce `timeline_summary.json`

```bash
python .github/skills/nsys-timeline-sql/impl/summarize.py \
    --sqlite     results/<exp>/nsys/<config>/profile.sqlite \
    --out-dir    results/<exp>/nsys/<config>/ \
    --top-n      10
```

### Diff mode — compare two captures

```bash
python .github/skills/nsys-timeline-sql/impl/summarize.py diff \
    --baseline   results/<exp>/baseline/timeline_summary.json \
    --patched    results/<exp>/patched/timeline_summary.json \
    --out        results/<exp>/diff.json
```

Emits per-kernel before/after self-time + delta + signed % change, sorted by
`abs(delta_ms)`. Useful for "did my patch actually move the kernel I thought it would?"

### Query mode — arbitrary read-only SQL

```bash
python .github/skills/nsys-timeline-sql/impl/summarize.py query \
    --sqlite     profile.sqlite \
    --sql        "SELECT name, COUNT(*) FROM ..."
    --limit      50
```

The skill enforces:
- only `SELECT`/`WITH` statements (no INSERT/UPDATE/DELETE/PRAGMA write)
- automatic `LIMIT` if not present
- output as JSON rows

This is the agent's escape hatch for any metric not in the default summary.

## OUTPUT CONTRACT — `timeline_summary.json`

```json
{
  "schema_version": 0,
  "ok": true,
  "captured_from": "results/.../profile.sqlite",
  "analysis_window_ns": [12345000, 98765000],
  "wall_ns": 86420000,
  "gpu": {
    "device_name": "NVIDIA H200",
    "sm_arch": "9.0",
    "num_sms": 132
  },
  "streams": {
    "primary_stream_id": 7,
    "active_streams": [7, 13, 21],
    "per_stream": {
      "7":  {"active_ms": 980.0, "idle_ms": 254.0, "kernel_count": 4500},
      "13": {"active_ms":  42.0, "idle_ms": 100.0, "kernel_count":   12}
    }
  },
  "totals_primary_stream": {
    "gpu_active_ms":   980.0,
    "gpu_idle_ms":     254.0,
    "gpu_util_pct":     79.4,
    "kernel_count":   4500
  },
  "top_kernels": [
    {"rank": 1, "short_name": "flashinfer::SinglePrefillWithKVCacheKernel",
     "self_ms": 220.5, "self_pct_of_active": 22.5,
     "calls": 320, "avg_us": 689.0, "max_us": 1240.0,
     "register_per_thread": 168, "max_grid": [200, 1, 1], "max_block": [128, 1, 1]}
    /* up to top_n */
  ],
  "largest_idle_gaps": [
    {"rank": 1, "gap_ms": 46.0, "start_ns": 12350000, "end_ns": 12396000000,
     "before_kernel": "cudaMemcpyAsync (DtoH)", "after_kernel": "fused_moe::...",
     "likely_cause": "host roundtrip (memcpy DtoH precedes idle)"}
    /* up to top_n */
  ],
  "cuda_api": {
    "launch_kernel_count": 850000,
    "launch_kernel_total_ms": 1200.0,
    "launch_kernel_avg_us": 1.41,
    "graph_launch_count":   200,
    "graph_launch_total_ms": 12.0,
    "launch_ratio_graph_to_eager": 0.000235,
    "verdict": "eager_dominated"   /* eager_dominated <0.1 | mixed <0.9 | graph_dominated */
  },
  "memcpy": {
    "h2d_bytes": 1.2e9, "h2d_ms": 8.5, "h2d_avg_gb_s": 141,
    "d2h_bytes": 2.1e6, "d2h_ms": 0.4, "d2h_avg_gb_s": 5.2,
    "d2d_bytes": 0.0,   "d2d_ms": 0.0
  },
  "recipes_run": ["top_kernels_by_self_time", "largest_idle_gaps", "api_launch_counts",
                  "memcpy_aggregate", "per_stream_breakdown"],
  "warnings": []
}
```

On failure: `{ "schema_version": 0, "ok": false, "error": "..." }`.

## WHICH METRIC HELPS WHICH PROBLEM

This is the section the agent should read **before deciding what to look at next**.

### GPU utilization metrics

| Look at | Says what | Action if outside healthy range |
|---|---|---|
| `gpu_util_pct` (= active / (active+idle)) | How busy the GPU is overall | <60% → CPU bound or scheduling stall (next: `largest_idle_gaps` + `cuda_api.launch_kernel_count`). >95% → GPU bound (next: `top_kernels`). |
| `kernel_count` over `wall_ns` (launches/sec) | Launch rate | >50k/s → likely CPU-launch-overhead bound; consider cudagraph. <500/s → very few large kernels; per-kernel optimization more valuable than launching less. |
| `launch_ratio_graph_to_eager` | Whether cudagraph is replaying | "eager_dominated" + low gpu_util_pct → cudagraph is configured but not actually being used (matches sglang R_short case). |

### Kernel-level metrics

| Look at | Says what | Action |
|---|---|---|
| `top_kernels[0].self_pct_of_active` | Whether one kernel dominates | >40% → single-kernel hot spot, source-level optimization or autotuning worth pursuing. <15% → flat distribution, scheduler/launch overhead more likely than kernel choice. |
| `top_kernels[].avg_us` vs `max_us` | Kernel time variance | `max_us > 3 * avg_us` → variable inputs (likely different batch sizes hitting the same kernel); autotuning may help. Stable → single tile config used. |
| `top_kernels[].register_per_thread` | Register pressure | >128 → likely register spill; SM occupancy will be low (need ncu to confirm, but a strong hint). |
| `top_kernels[].max_grid[0] * max_block[0]` | Total threads | Compare to `num_sms * 2048` (max threads in flight on H200). Much smaller → underutilizing GPU, consider larger batch / different tile. |
| Top kernel name pattern `cutlass::Kernel` vs `sm80_xmma_gemm` vs Triton-generated | Which backend | Discrepancies between expected and observed backend = "wrong code path" bug. |

### Idle gap metrics

| Look at | Says what | Action |
|---|---|---|
| `gpu_idle_ms / wall_ns * 100` | Total idle fraction | >40% → ANY work to reduce idle pays off. <10% → focus on kernel speed instead. |
| `largest_idle_gaps[0].gap_ms` | Worst single stall | >100ms → very likely a host roundtrip or sync; check `before_kernel`. >10ms across many gaps → scheduler overhead / Python loop. |
| `before_kernel == cudaMemcpy*` for top gaps | Host transfer caused stall | Use pinned memory / async transfer / co-locate buffer on device. |
| `before_kernel == cudaStreamSynchronize` | Explicit sync stalled GPU | Identify where in Python this came from; often a `.item()` / `.cpu()` call. |
| `before_kernel == nothing` (period > 100ms) | GPU was just waiting for next launch | CPU bound; cudagraph / batching / async dispatch helps. |

### CUDA API metrics

| Look at | Says what | Action |
|---|---|---|
| `launch_kernel_count` total | How many host launches | >500k for a 30s run → very chatty; cudagraph is the obvious lever. |
| `launch_kernel_avg_us` | Per-launch CPU cost | >5us → CPU contention or hot Python path. <2us → launches are cheap; reducing them helps but not as much as elsewhere. |
| `graph_launch_count` > 0 but `launch_kernel_count` also high | Mixed graph + eager | Some path is bypassing graph capture; find which model layer is eager (NVTX helps). |

### Memcpy metrics

| Look at | Says what | Action |
|---|---|---|
| `h2d_avg_gb_s` < 20 | H2D bandwidth wasted | Likely unpinned host memory; allocate with `cudaMallocHost`. |
| `d2h_bytes` significant per inference step | Output is being copied back per step | KV cache or logits leaking to host; pin them on device. |
| `d2d_bytes` very high | Internal copies (tensor reshape, view contiguity) | Profile with NVTX to find which op causes the copies. |

### When NONE of the above moves

The default summary doesn't show: SM occupancy, L2 hit rate, mem-controller throughput
over time. These need `ncu` (Nsight Compute), which we don't currently have set up.
Document the gap, escalate to mentor.

## METHODOLOGY — recipe library + extension

The default summary runs a fixed set of "recipe" SQLs. Each recipe is a separate
`.sql` file under `recipes/` so the agent can read what it does:

```
.github/skills/nsys-timeline-sql/recipes/
  top_kernels_by_self_time.sql
  largest_idle_gaps.sql
  api_launch_counts.sql
  memcpy_aggregate.sql
  per_stream_breakdown.sql
  gpu_active_per_stream.sql
```

To add a new recipe (when the agent finds a query worth running every time):
1. Drop a new `<name>.sql` under `recipes/`.
2. Update `impl/summarize.py` to wire its result into the output JSON.
3. Bump `version:` if the new field is mandatory for downstream.

For one-off queries, **do not** add a recipe — use the `query` sub-command.

## EXTENSION — `query` sub-command (the escape hatch)

The `query` sub-command runs arbitrary read-only SQL against the SQLite file
exported by nsys. Useful when the default summary doesn't answer your question.

**Common SQL building blocks** the agent should know about:

```sql
-- Every GPU kernel as a row, with timestamps and config
SELECT s.value AS name,
       k.start AS start_ns,
       k.end AS end_ns,
       (k.end - k.start) AS dur_ns,
       k.streamId, k.gridX, k.gridY, k.gridZ,
       k.blockX, k.blockY, k.blockZ,
       k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory,
       k.gridId  -- nonzero = cudagraph node
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
ORDER BY k.start;

-- Every CUDA API call (host side)
SELECT s.value AS name,
       r.start AS start_ns, r.end AS end_ns,
       (r.end - r.start) AS dur_ns,
       r.correlationId
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
ORDER BY r.start;

-- Join host launch with GPU kernel via correlationId
-- to measure "how long after cudaLaunchKernel does the kernel actually run?"
SELECT s_api.value AS api,
       s_k.value   AS kernel,
       r.start AS api_start_ns,
       k.start AS gpu_start_ns,
       (k.start - r.end) AS queue_lag_ns
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN CUPTI_ACTIVITY_KIND_KERNEL  k ON r.correlationId = k.correlationId
JOIN StringIds s_api ON r.nameId = s_api.id
JOIN StringIds s_k   ON k.shortName = s_k.id
ORDER BY queue_lag_ns DESC LIMIT 20;

-- Memcpy operations
SELECT m.copyKind, m.bytes,
       (m.end - m.start) AS dur_ns,
       m.streamId
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
ORDER BY dur_ns DESC LIMIT 20;
```

If the agent reuses a query across investigations, promote it to a recipe.

## FAILURE MODES

| Mode | Detection | Mitigation |
|---|---|---|
| sqlite file missing required table (rare; only if `nsys export` was incomplete) | catch sqlite3 OperationalError | `{"ok": false, "error": "table CUPTI_ACTIVITY_KIND_KERNEL not found; sqlite was not produced by nsys export"}` |
| All kernel rows have `streamId=0` (no GPU work captured) | post-query check | `{"ok": false, "error": "no GPU kernels in profile; target_cmd likely didn't run any GPU work"}` |
| Single stream dominates 99% — assume primary, but warn if more than 5 streams have >5% each | warning | normal multi-stream caches will trip this; emit info |
| Query sub-command rejects non-SELECT | input parse | `{"ok": false, "error": "only SELECT/WITH allowed"}` |
| Window restricts to 0 kernels | post-window count | warn "window contained no kernel events"; emit empty stats |

## ROADMAP

- **v1** — auto-detect "interesting windows" via NVTX boundaries and emit per-window summaries
  (e.g. "prefill window" vs "decode window" if those NVTX ranges are present).
- **v1** — recipe: per-layer breakdown using NVTX layer-N ranges (requires sglang/vllm
  to be instrumented, but the skill side is free).
- **v2** — diff sub-command emits human-readable Markdown table for direct paste-into-PR.
- **v2** — when ncu becomes available, add `ncu-followup` recipe that points to which
  kernels are worth ncu-profiling.

## REFERENCES

- nsys deep-dive (the SQLite tables): `docs/2026-06-08/nsys_deep_dive_and_proton.md`
- 2x2 validation that used proto-recipes by hand: `docs/2026-06-08/nsys_2x2_validation_and_nsys_usage.md`
- Capability audit, Gap 1 (corrected — we CAN see timeline data via SQL): `docs/2026-06-08/agent_profiling_capability_audit.md`
