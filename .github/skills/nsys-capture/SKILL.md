---
name: nsys-capture
description: Wrap an arbitrary action (a bench run, a single curl, an N-second sleep) with `nsys profile`, then immediately export the .nsys-rep to SQLite so downstream skills can SQL-query it without reopening the binary trace.
version: 0
stage: [1, 2, 3]
inputs:
  - target_cmd:    string   # shell command to run while profiling is active
  - duration_s:    float    # how long to keep the profile open (capped at 120s by default)
  - gpu_id:        int      # which GPU to limit profiling to (CUDA_VISIBLE_DEVICES-style)
  - out_dir:       path
  - extra_nsys:    string   # optional extra args appended to nsys profile (e.g. "--cuda-graph-trace=node")
outputs:
  - nsys_capture.json   (path metadata + exit codes; downstream skills read this)
  - profile.nsys-rep    (binary trace, 50–200 MB typical)
  - profile.sqlite      (exported event database, ~same size or slightly larger)
triggers:
  - "When a bench shows >20% gap and `server-log-mining` is clean (no config-shaped cause)."
  - "When the agent wants any of: GPU active time, idle gap, per-kernel cost, CPU launch count, stream parallelism — i.e. anything the bench JSON can't see."
  - "Before/after any source-code patch that targets a hot kernel — capture both sides."
depends_on: []
---

# nsys-capture

## WHEN

Concrete conditions:

1. **Bench gap is unexplained.** `bench_summary.json` shows X vs Y differ ≥20%,
   `server-log-mining` returned no config-shaped culprit (no cap, no KV pressure,
   no retract). The next layer of evidence is per-kernel timing → this skill.
2. **Kernel-level hypothesis.** Agent has a specific guess like "autotune is producing
   a different tile size" or "cudagraph isn't actually replaying". Both need timeline
   inspection. Call this skill against the candidate config to get raw data, then
   call `nsys-timeline-sql` to extract.
3. **Patch verification.** Before applying a source patch, capture a baseline. After,
   capture an identical workload. Diff via `nsys-timeline-sql`. **Never** trust an
   e2e bench alone to attribute a fix to the patched kernel.

Do **not** call when:
- The gap is small (<10%); nsys overhead itself is 5–15%, signal-to-noise too low.
- You don't yet have a hypothesis about which window matters. Sample-profiling 30s
  of mixed warmup + steady state gives garbage. Always run the target_cmd long enough
  that ≥80% of profile time is in steady state.
- Multiple users are heavy on the same GPU. CUPTI events bleed across processes
  for cudagraph-related metrics. Use `gpu_id` to pin.

## WHY (the failure mode this prevents)

Three concrete past mistakes:

1. **Profiled too short** (2026-06-05). Captured 5s of a sglang start → measured
   profile dominated by graph capture, not steady-state inference. Wrong conclusion
   about which kernel was hot. **Skill rule: minimum target_cmd duration 20s of
   steady state, asserted via `bench_summary.json` `wall_s.mean`.**
2. **Lost the .nsys-rep** (2026-06-07). Profile worked, then user deleted the
   ~80 MB file before agent re-queried. Had to re-run the entire bench.
   **Skill rule: always export sqlite *immediately* (within the same skill call).
   The sqlite copy is what downstream uses; the .nsys-rep can be deleted to save disk.**
3. **Forgot to kill the nsys daemon** (2026-06-04). Bench finished but `nsys profile`
   was wrapping `sleep 600`. The .nsys-rep file was 0 bytes until SIGINT.
   **Skill rule: send SIGINT after `duration_s`, wait up to 60s for flush, then
   confirm file size > 1 MB before returning success.**

## HOW

```bash
python .github/skills/nsys-capture/impl/run_capture.py \
    --target-cmd  'python results/4way_bench/scripts/run_bench_4way.py http://127.0.0.1:30000 sglang_cutlass /tmp/x' \
    --duration-s  60 \
    --gpu-id      1 \
    --out-dir     results/<exp>/nsys/<config>/ \
    --extra-nsys  "--cuda-graph-trace=node"
```

Internal sequence:
1. **Resolve nsys binary** — search `which nsys`, fall back to known path
   `/home/t-chendili/cuda/12.6/bin/nsys`. Write the version into output.
2. **Launch `nsys profile`** with these defaults:
   - `-t cuda,nvtx,osrt` (CUDA + user NVTX ranges + OS runtime)
   - `-s none` (no sampling — cuts overhead by 5x)
   - `--cuda-memory-usage=true` (catches host-side allocator behavior)
   - `--force-overwrite=true -o <out_dir>/profile`
   - target = the user's `target_cmd` (wrapped via `sh -c`)
   - `CUDA_VISIBLE_DEVICES=<gpu_id>` in the launched env
3. **Wait** up to `duration_s + 30s` for `target_cmd` to exit naturally.
   If still running, SIGINT the nsys wrapper, wait 60s for flush.
4. **Verify** `profile.nsys-rep` exists and is >1 MB.
5. **Export sqlite immediately**: `nsys export --type sqlite --force-overwrite=true
    --output profile.sqlite profile.nsys-rep`.
6. Write `nsys_capture.json`.

## OUTPUT CONTRACT — `nsys_capture.json`

```json
{
  "schema_version": 0,
  "ok": true,
  "captured_at": "2026-06-09T04:00:00Z",
  "nsys_binary": "/home/t-chendili/cuda/12.6/bin/nsys",
  "nsys_version": "2024.5.1",
  "gpu_id": 1,
  "target_cmd": "python ... /tmp/x",
  "target_exit_code": 0,
  "target_duration_s": 38.4,
  "files": {
    "nsys_rep": "results/.../profile.nsys-rep",
    "nsys_rep_size_mb": 87.2,
    "sqlite":   "results/.../profile.sqlite",
    "sqlite_size_mb": 91.6
  },
  "warnings": [
    /* e.g. "target_cmd exited before duration_s — profile may be short" */
  ]
}
```

On failure:
```json
{ "schema_version": 0, "ok": false, "error": "<one sentence>",
  "stderr_tail": "<last 1KB of nsys stderr>" }
```

## WHICH METRIC HELPS WHICH PROBLEM

This skill itself produces **no analysis metrics** — it produces the *data*.
The metrics-to-problem mapping lives in the next skill: `nsys-timeline-sql`.

What this skill **does** enable for downstream:

| If `extra_nsys` includes... | Downstream gets... | Useful for |
|---|---|---|
| (default: no extras) | per-kernel timestamps + per-CUDA-API timestamps | Most cases. GPU idle gap, top kernels, launch counts. |
| `--cuda-graph-trace=node` | per-kernel rows inside cudagraph replays, not folded | Diagnosing whether cudagraph is hiding the real bottleneck (sglang R_short case) |
| `--gpu-metrics-device=all` | SM occupancy, mem throughput sampled at 100 Hz | "Why is this kernel slow?" — but adds 5–10% overhead, only enable when targeted |
| `-t cuda,nvtx,osrt,nccl` | NCCL collectives (allreduce, alltoall) timing | Multi-GPU / TP > 1 only |
| `-t cuda,nvtx,osrt,cublas,cusparse,cudnn` | library-level annotations | When you suspect cuBLAS dispatch overhead |

Default is intentionally minimal — extras cost overhead and disk. The agent should
add them only when the question requires them.

## METHODOLOGY — predict-then-verify

Before capturing, the agent **must** write down (in plan.md or call context):

> "I expect to see in the profile: <kernel X> taking >Y% of GPU time, OR <N>
>  cudaLaunchKernel calls per second, OR <stream Z> idle for >Wms gaps."

After `nsys-timeline-sql` returns, compare. If the profile shows none of those:
- Either the hypothesis is wrong (don't backfill a new story to fit the data).
- Or the capture missed the relevant window (run again with longer duration / different
  target_cmd phase).

## EXTENSION

Two escape hatches when the default capture isn't enough:

1. **Custom nsys args** via `--extra-nsys` — anything `nsys profile --help` accepts.
   Examples: `--sample=cpu`, `--cudabacktrace=true`, `--capture-range=cudaProfilerApi`.
2. **Capture multiple windows** — call this skill N times with N target_cmds that
   each `sleep K` before different phases (warmup vs steady vs teardown). The skill
   is intentionally stateless to enable this.

## FAILURE MODES

| Mode | Detection | Mitigation |
|---|---|---|
| `nsys` not on PATH and fallback path missing | first `which nsys` + stat fallback | `{"ok": false, "error": "nsys not found; install or set NSYS_PATH"}` |
| `.nsys-rep` is 0 bytes after duration | post-flush size check | retry once with longer flush wait (90s); if still 0, fail |
| sqlite export fails (corrupt trace) | exit code of `nsys export` | keep .nsys-rep, fail loudly so user can debug manually |
| target_cmd crashes during capture | non-zero exit | record `target_exit_code`, attempt flush+export anyway (partial profile may still be useful), set `ok: true` with warning |
| Disk full during capture | check df pre-flight, OSError mid-flight | abort cleanly, fail with "disk full at <path>" |
| Another nsys profile already running on host | `pgrep nsys` returns >0 | warn, proceed (concurrent profiling works but doubles overhead) |
| CUDA_VISIBLE_DEVICES already set by target_cmd | env var conflict detection | warn, ours wins (skill overrides) |

## ROADMAP

- **v1** — auto-detect the "steady state" window inside the profile (skip first 10%
  of timeline as warmup) and emit `steady_state_start_ns` into capture metadata,
  so downstream queries default to that window.
- **v1** — compress .nsys-rep to .nsys-rep.zst after sqlite export (typically 3-5×
  smaller, since sqlite has the per-event data anyway).
- **v2** — multi-rank capture (one .nsys-rep per TP rank, merged into one sqlite).

## REFERENCES

- nsys deep-dive (what each report contains): `docs/nsys_deep_dive_and_proton.md`
- Past nsys-based validation: `docs/nsys_2x2_validation_and_nsys_usage.md`
- Capability audit (what nsys can / can't see): `docs/agent_profiling_capability_audit.md` Part B
