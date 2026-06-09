---
name: ncu-microarch
description: Run `sudo ncu` (Nsight Compute) against a specific kernel regex while a target command executes, then reduce the CSV to ncu_summary.json with curated microarchitectural metrics (SM occupancy, DRAM throughput, L1/L2 hit rate, tensor-core utilization, top warp-stall reason). The "why is this kernel slow?" skill that complements nsys-timeline-sql's "which kernel is slow?".
version: 0
stage: [2, 3]
inputs:
  - target_cmd:         shell command to run while NCU profiles
  - kernel_regex:       regex matching kernel short names to profile (REQUIRED — profiling all kernels is prohibitively slow)
  - launch_count:       int, default 12 (number of kernel launches to sample; higher = more reliable but slower)
  - metrics_file:       optional path to a metric set file (default: metric_sets/default.txt — 8 most-actionable metrics)
  - gpu_id:             int, which GPU to limit profiling to
  - out_dir:            path
outputs:
  - ncu_summary.json     (curated per-kernel metrics + interpretive verdict)
  - ncu_raw.csv          (raw NCU CSV — kept for replay and ad-hoc analysis)
triggers:
  - "After nsys-timeline-sql has identified a hot kernel (>15% of GPU active time) AND the agent has a hypothesis the skill must falsify: 'why slow?'."
  - "Before claiming 'this kernel is X-bound' — without ncu, that claim is a guess."
  - "Before recommending a source-level kernel optimization (D1/D2/D3 etc.): ncu data justifies the direction."
depends_on:
  - nsys-timeline-sql   # ncu is expensive — only invoke after timeline-sql narrows the target
---

# ncu-microarch

## WHEN

Concrete conditions:

1. **You have ≥1 named hot kernel** (from `nsys-timeline-sql top_kernels`).
   This skill ALWAYS requires `--kernel-regex` — there is no "profile everything"
   mode because ncu replays each kernel ~10× to collect counters; on real
   workloads with 25k+ launches, profiling all kernels takes hours.
2. **You can re-run the workload deterministically** (same prompts → same
   kernels). The target_cmd is run **once** under ncu; reproducibility matters
   because counter values aren't averaged across runs.
3. **The hypothesis you want to test is microarchitectural**:
   - "is this kernel memory-bound or compute-bound?"
   - "are tensor cores actually firing?"
   - "what's causing warp stalls?"
   - "how close to peak FLOPs?"
   For "is this kernel slow?" use `nsys-timeline-sql` instead — ncu is overkill.

Do NOT call when:
- The workload runs <10 launches of the target kernel — ncu has nothing to sample.
- You only want timing data — that's nsys, not ncu.
- The hot kernel is wrapped in a cudagraph but you want to profile it eagerly
  — ncu won't see graph-captured kernels by default. Use `nsys-capture` with
  `--extra-nsys --cuda-graph-trace=node` instead, then come back.

## WHY (the failure mode this prevents)

We hit this exact gap in the 2026-06-09 CUTLASS investigation:

> "CUTLASS MoE GEMM 44.9µs vs Triton 2×29.8µs — CUTLASS wins per layer by 25%.
>  But the win is only ~10% of total time, so e2e is in the noise. Could we
>  push CUTLASS further? Without NCU, we don't know if it's already at HBM
>  peak, or has 50% more headroom. So the improvement direction D4 says
>  'investigate why CUTLASS doesn't have more headroom — needs ncu, currently
>  unavailable'."

The skill closes that gap. Once you have ncu output for the CUTLASS kernel,
"D4 is dead because it's already at 95% peak" becomes a concrete claim that
can be verified, not handwaved.

The 5 wrong root causes documented in `docs/2026-06-08/agent_profiling_capability_audit.md`
Part E mostly bottomed out at "we don't know if this kernel is the actual
bottleneck or just busy" — ncu answers that.

## HOW

```bash
python .github/skills/ncu-microarch/impl/run_ncu.py \
    --target-cmd  'python /tmp/bench_for_ncu.py http://127.0.0.1:30001' \
    --kernel-regex 'fused_moe::run_global' \
    --launch-count 12 \
    --gpu-id 1 \
    --out-dir results/<exp>/ncu/<config>/
```

Internal sequence:
1. Resolve NCU path (`/home/t-chendili/.conda/pkgs/nsight-compute-*/bin/ncu`)
   and verify `sudo -n <ncu> --version` works (NCU requires sudo for GPU
   counters; chendi unlocked it via NOPASSWD whitelist on 2026-06-09).
2. Load metric set from `--metrics-file` (default: `metric_sets/default.txt`).
3. Invoke `sudo -n <ncu> --target-processes all --launch-count <N>
   --kernel-name regex:<regex> --metrics <csv-list> --csv <target_cmd>`.
4. Capture stdout to `ncu_raw.csv`.
5. Parse CSV → per-kernel aggregated metrics → write `ncu_summary.json` with
   an interpretive `verdict` per kernel.

### Cost
- Per kernel sample: ~5–10s on H200 (replays the kernel ~10× with counters).
- 12 samples per kernel × 1 kernel regex = ~1-2 min total wall.
- If you profile 4 kernels, multiply by 4.

## OUTPUT CONTRACT — `ncu_summary.json`

```json
{
  "schema_version": 0,
  "ok": true,
  "captured_at": "2026-06-09T18:30:00Z",
  "ncu_path": "/home/t-chendili/.conda/pkgs/.../ncu",
  "ncu_version": "2026.1.1.0",
  "target_cmd": "python ...",
  "kernel_regex": "fused_moe::run_global",
  "launch_count": 12,
  "gpu_id": 1,
  "metrics_file": "metric_sets/default.txt",
  "metrics_collected": ["sm__throughput.avg.pct_of_peak_sustained_elapsed", ...],
  "kernels": [
    {
      "kernel": "fused_moe::run_global<fused_moe::Fused_Moe_Kern...>",
      "samples": 12,
      "metrics": {
        "sm_throughput_pct":         82.4,
        "dram_throughput_pct":       45.7,
        "warps_active_pct":          78.0,
        "l1_hit_pct":                89.3,
        "l2_hit_pct":                72.1,
        "tensor_pipe_active_pct":    74.5,
        "stall_long_scoreboard_avg": 1.20,
        "stall_math_pipe_throttle_avg": 0.45
      },
      "verdict": "compute_bound",
      "headroom_estimate_pct": 17.6,
      "notes": [
        "Tensor Cores firing at 74.5% — good for bf16 on Hopper.",
        "Warp stall dominated by long_scoreboard but only 1.2 warps/issue — moderate.",
        "DRAM at 45% — not memory-bound."
      ]
    }
  ],
  "warnings": []
}
```

### `verdict` enum (derived from metric thresholds)

| `verdict`              | Rule                                                       | Implication |
|---|---|---|
| `compute_bound`        | `sm_throughput_pct ≥ 70` AND `dram_throughput_pct < 50`     | Tune tile size / data layout to push sm higher (D4-style work). |
| `memory_bound`         | `dram_throughput_pct ≥ 70`                                  | Reduce data movement — tile fusion, on-chip reuse, smaller dtype. |
| `latency_bound`        | `sm_throughput_pct < 30` AND `dram_throughput_pct < 30`     | Likely launch overhead or warp stalls — check `warps_active_pct`. |
| `balanced`             | both 40–70%                                                | Hard to optimize one axis without hurting another; small wins only. |
| `low_occupancy`        | `warps_active_pct < 30`                                    | Grid/block shape wrong; fix occupancy first. |
| `tensor_core_idle`     | `tensor_pipe_active_pct < 10` on a GEMM-shaped kernel       | RED FLAG — Tensor Cores not being used. Bug. |

`headroom_estimate_pct` = `100 - max(sm_throughput_pct, dram_throughput_pct)`
— a rough "how much faster could this be" upper bound, **not** a promise.

## WHICH METRIC HELPS WHICH PROBLEM

| Metric | Says what | Action if outside healthy range |
|---|---|---|
| `sm_throughput_pct` | Overall SM busy-ness | <50 → likely launch/memory bound, not a compute-tune target. >85 → kernel is already squeezed; only algorithmic changes will help. |
| `dram_throughput_pct` | HBM bandwidth usage | >70 → memory-bound. Algorithmic data-reuse (tile fusion, persistent kernels) > tile tuning. |
| `warps_active_pct` | Warp slots filled | <50 → grid/block too small; raise. Cudagraph won't fix occupancy. |
| `l2_hit_pct` | Reuse across SMs | <30 on a kernel with reused data → bad tiling. >80 → already tiled well. |
| `tensor_pipe_active_pct` | Tensor Core utilization | 0–10 on a GEMM/MM kernel = wrong dtype or no TMA layout. 50–90 = healthy. >95 = peak. |
| `stall_long_scoreboard_avg` | Warps waiting on memory load | >2.0 = severe memory-bound; consider prefetch / persistent. |
| `stall_math_pipe_throttle_avg` | Math pipeline back-pressure | >1.5 = tensor cores saturated; consider smaller tile. |

## METHODOLOGY — predict-then-verify

Before ncu: agent records prediction in plan.md:

> "I predict `fused_moe::run_global` is compute_bound with sm_throughput_pct
>  ≥ 70 (because I see CUTLASS Hopper TMA in the kernel name). If it's
>  memory_bound instead, the D4 direction shifts from 'tune more tiles' to
>  'reduce expert weight traffic'."

After ncu: compare verdict + metrics to prediction. If they disagree, the
hypothesis was wrong — drop the planned improvement direction and re-think.

## EXTENSION

- Custom metric sets under `metric_sets/`. Each file is a newline-separated
  metric name list. Comments OK with `#`. Default keeps capture cheap (8
  metrics); use `metric_sets/roofline.txt` for arithmetic intensity (when added).
- For Tensor Core arithmetic intensity, request:
  `sm__sass_thread_inst_executed_op_hfma_pred_on.sum`,
  `sm__sass_thread_inst_executed_op_dfma_pred_on.sum` etc. — these are large
  metric sets (slower capture); only enable when needed.
- For "diff two runs" use `unify.py` (in `profile-summary-unified`): it has
  a `kernel_micro` field that adapts ncu output.

## FAILURE MODES

| Mode | Detection | Mitigation |
|---|---|---|
| `sudo -n ncu` fails (no NOPASSWD) | preflight | `{"ok": false, "error": "ncu NOPASSWD not configured; ask t-chendili to add /etc/sudoers.d entry"}` |
| Kernel regex matches 0 kernels in trace | CSV has 0 data rows | `{"ok": false, "error": "no kernels matched regex '<X>' — list available with --list-only"}` |
| `--launch-count` exceeds actual launches | fewer samples than requested | record `samples_actual < launch_count` in warning; verdict still computed |
| Counter overflow / NaN | CSV cell unparseable | per-metric fallback to None; mention in warnings |
| Target cmd crashes mid-profile | non-zero exit | save partial CSV, mark `target_exit_code` non-zero, still try to parse |

## ROADMAP

- **v1** — `--list-only` mode that runs `ncu` with `--list-only` to enumerate
  matching kernels without profiling (useful for "which kernels match this regex?").
- **v1** — automatic kernel selection from a `timeline_summary.json` top-N
  list (avoid hand-typing regex).
- **v2** — roofline plot generation (HBM bandwidth × FLOPs intensity) — needs
  `roofline` metric section + a plotting library.
- **v2** — register spill detection via `--metrics sass__inst_executed_op_local_*`
  — surface in `verdict: register_spilled` kind.

## REFERENCES

- Unlock history: chendi added NOPASSWD entry 2026-06-09 evening for
  `/home/t-chendili/.conda/pkgs/nsight-compute-2026.1.1.2-h1ff7d1d_0/bin/ncu`.
- Why we need this skill: `docs/2026-06-08/agent_profiling_capability_audit.md` B.2
  (currently flagged 🟨; this skill closes the gap).
- Investigation that motivated D4 needing ncu: `docs/2026-06-09/cutlass_vs_triton_e2e_investigation.md`
- Upstream skill that picks targets for ncu: `nsys-timeline-sql` → `top_kernels[*].short_name`
- Downstream consumer: `profile-summary-unified` `_from_ncu()` adapter fills `kernel_micro` field.
