# sglang Triton MoE — 4-regime sweep — file index

Generated 2026-06-09 night. See `docs/2026-06-09/sglang_triton_4regime_profiling.md`
for the narrative writeup.

## Where to find what

### Primary narrative
- `docs/2026-06-09/sglang_triton_4regime_profiling.md` — the writeup with cross-regime tables

### Per-regime human-readable NCU reports (NEW — start here for kernel detail)
- `ncu/R_short_decode/ncu_report.md` — 30 kernels, full table + per-kernel notes
- `ncu/R_medium_balanced/ncu_report.md` — 30 kernels
- `ncu/R_long_prefill/ncu_report.md` — 50 kernels (NCU `--launch-count 50` for this regime)
- `ncu/R_concurrent_decode/ncu_report.md` — 30 kernels

### Canonical merged artifact per regime (machine-readable, evidence-chain attribution)
- `unified/<regime>/profile_unified.json` — single JSON per regime, all 3 sources merged

### Per-regime raw artifacts

| Layer | Per-regime files |
|---|---|
| e2e bench (req/s, ttft, etc.) | `bench/bench_summary.json` (all 4 regimes in one JSON) + `bench/per_run/<regime>_runN.json` |
| nsys timeline summary (top-15 kernels, idle gaps, etc.) | `nsys/<regime>/timeline_summary.json` |
| ncu structured (8 metrics + verdict + headroom per kernel) | `ncu/<regime>/ncu_summary.json` |
| ncu wide-format raw (~7000 metrics × N kernels) | `ncu/<regime>/ncu_raw_full.csv` |
| ncu binary (open with `ncu-ui` GUI) | `ncu/<regime>/<regime>_ncu.ncu-rep` (gitignored — too big) |
| sglang server / bench logs | `ncu/<regime>/bench.log`, `nsys/server.log`, `server.log` |

### Workload definition
- `regimes/qwen3_30b_moe_sglang_perf_sweep.yaml` (under repo root, not under this dir)

### Scripts that built these
- `scripts/bench_ncu_one_regime.sh` — drives one regime through sglang.bench_one_batch + sudo ncu
- `scripts/bench_ncu_all_regimes.sh` — serial 4-regime batch driver
- `scripts/ncu_csv_wide_to_summary.py` — wide-CSV → ncu_summary.json adapter
- `scripts/unify_sweep.py` — calls profile-summary-unified per regime
- `scripts/generate_ncu_reports.py` — generates the per-regime ncu_report.md

### Skills exercised
- `e2e-bench-runner` v1 (`--regimes-file` YAML)
- `nsys-timeline-sql` (sliced single .nsys-rep into 4 windows)
- `ncu-microarch` style (--set full, no kernel filter, --profile-from-start off + CUDA_PROFILER trigger)
- `profile-summary-unified` (4 evidence_chain entries × 4 fields all ok=true)
