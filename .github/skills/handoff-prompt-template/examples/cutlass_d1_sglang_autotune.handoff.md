# Re-enable flashinfer_cutlass in sglang autotune allowlist (D1)

## Problem statement

sglang with `moe_runner_backend=flashinfer_cutlass` runs the CUTLASS MoE GEMM
kernel at its untuned fallback (tactic 0) because sglang's autotune allowlist
excludes `flashinfer_cutlass`, causing 5-6× kernel slowdown vs the autotuned path.

## Evidence chain

- `e2e-bench-runner` → R_medium req/s 1.26 (sglang_cutlass) vs 4.74 (vllm_cutlass) → `results/2026-06-09_cutlass_investigation/cutlass/bench/bench_summary.json`
- `cross-regime-anomaly` → "large_uniform_gap: sglang_triton beats sglang_cutlass on all 3 regimes (~60%)" → `(synthetic test on the matrix)`
- `pytorch-profiling` (methodology) → CUTLASS MoE GEMM 44.9µs avg in vLLM (with autotune ON) → `results/2026-06-09_cutlass_investigation/cutlass/torch_trace/profiler_out_0.txt`
- microbench → fallback tactic 0 ≈ 44µs vs tuned tactic ≈ 8µs → `results/cutlass_microbench/results_2026-06-08.md`
- source: `sglang/python/sglang/srt/model_executor/model_runner.py:1838-1845`
- source (root cause): `flashinfer/.../flashinfer_cutlass_fused_moe_binding.cu:638` (fallback = `mAllProfiles.front()`)

## Hypothesis

sglang's `_should_run_flashinfer_autotune()` allowlist omits `flashinfer_cutlass`
(TODO comment cites "flashinfer compilation errors"). Without autotune entry,
the CUTLASS kernel runs `mAllProfiles.front()` = tactic 0 fallback every time.
Microbench proved 5-6× difference between fallback and tuned. Therefore re-enabling
autotune should restore sglang_cutlass to within 30% of sglang_triton on all regimes.

## Suggested change

- **file**: `sglang/python/sglang/srt/model_executor/model_runner.py`
- **lines**: `1838-1845`
- **type**: source_edit
- **patch**:

```diff
- backends = ["fa3", "flashinfer", "flashinfer_trtllm"]
+ backends = ["fa3", "flashinfer", "flashinfer_trtllm", "flashinfer_cutlass"]
  return self.moe_runner_backend in backends
```

## Acceptance test

- **call**: `python .github/skills/regime-sweep-runner/impl/sweep.py --configs-file experiments/2026-06-10/sglang_only_configs.yaml --regimes-file regimes/qwen3_30b_moe_default.yaml --num-runs 3 --out-dir results/2026-06-10/sglang_cutlass_autotune/`
  - configs.yaml must contain BOTH the baseline (autotune OFF) and the patched (autotune ON) servers.
- **expect**: patched sglang_cutlass R_medium `req_per_s_mean ≥ 3.0` (≥ 2.4× improvement over baseline 1.26).
- **revert if**: `req_per_s_mean < 2.0` (less than 60% of the predicted gain) OR server fails to start (compilation errors materialize).

## Known risks

- risk: TODO comment cites "flashinfer compilation errors" — sglang might fail to launch.
  - mitigation: capture server.log; if engine init fails, revert immediately.
- risk: `_dummy_run()` shape might not match real inference shape → autotune cache miss → tuned tactic not actually used at runtime.
  - mitigation: verify by re-running `pytorch-profiling` on the patched server; the MoE GEMM avg µs must drop ≥ 3×. If not, the cache key needs fixing (separate handoff).
- risk: enabling autotune adds 2-30s to startup time.
  - mitigation: acceptable — this is one-time at server start, not per-request.

## What NOT to do

- do NOT modify the FlashInfer CUTLASS kernel source itself.
- do NOT change `cuda_graph_*` flags (orthogonal axis; would confound the test).
- do NOT add new autotune flags to ServerArgs — just edit the allowlist.
- do NOT bump sglang/flashinfer versions.
- do NOT touch `triton` or `flashinfer_trtllm` backends.

## Cross-references

- analysis doc: `docs/2026-06-09/cutlass_vs_triton_e2e_investigation.md` (improvement direction D1)
- prior failed attempt at fix1 (different hypothesis): `docs/2026-06-08/fix1_invalidated.md`
- prior failed attempt at bug A (this exact fix but cache-key issue): `docs/2026-06-08/buga_fix_validation.md` — autotune ran but e2e unchanged due to dummy_run shape mismatch; the risk above is concrete here.

## Predicted outcome

R_medium req_per_s_mean: 1.26 → ≥ 3.0 (≥ 2.4× improvement).
R_short and R_long should improve similarly because the bottleneck is the same.
If patched cutlass overtakes sglang_triton (3.02), that's strong confirmation
the entire 2.4× gap was autotune-only.
