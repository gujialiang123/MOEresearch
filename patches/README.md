# patches/

Source patches applied to upstream dependencies for our experiments.

Patches here are **not** automatically applied — apply manually with `git apply`
in the relevant upstream repo, then run the corresponding bench-spec.

## Patches

### `sglang_cutlass_autotune_allowlist.diff`

**Target**: `/home/t-jialianggu/work/sglang/python/sglang/srt/model_executor/model_runner.py`

**What it does**: adds `"flashinfer_cutlass"` to the
`_should_run_flashinfer_autotune()` allowlist. The original TODO comment dating
back several months claimed this caused "flashinfer compilation errors". As of
2026-06-11 with sglang 0.5.12.post1 + flashinfer 0.6.3, that TODO is **stale**:
the patch applies cleanly and produces a 4.7-8.4× speedup over the unpatched
baseline (see results below).

**Why it matters**: this single allowlist entry simultaneously fixes both bugs
flagged in `docs/2026-06-11/ofer_meeting_findings_draft.md` §8.6:
- (a) the JIT × cudagraph capture hang (autotune triggers JIT during warmup
      before cudagraph capture begins)
- (b) the tactic-0 fallback (autotune populates the AutoTuner cache so
      subsequent inference calls find a tuned tactic instead of falling back)

**To apply**:
```bash
cd /home/t-jialianggu/work/sglang
git apply /home/t-jialianggu/work/EndtoEnd-auto-optimization/patches/sglang_cutlass_autotune_allowlist.diff
# verify
sed -n '1838,1850p' python/sglang/srt/model_executor/model_runner.py
```

**Bench specs that require this patch**:
- `bench-specs/sglang-cutlass-bf16-patched.yaml`
- `bench-specs/sglang-cutlass-fp8-patched.yaml`

**Measured impact** (bf16, on H200, same regimes as `2026-06-09_sglang_triton_sweep`):

| regime | unpatched triton baseline | cutlass+patch | speedup |
|---|---|---|---|
| R_short_decode | 0.10 req/s | 0.83 | **8.4×** |
| R_medium_balanced | 0.74 | 4.40 | **6.0×** |
| R_long_prefill | 2.52 | 13.66 | **5.4×** |
| R_concurrent_decode | 2.94 | 13.86 | **4.7×** |

Startup time: +45s (autotune warmup window — as predicted).

See `results/2026-06-11_harness-v1/sglang-cutlass-bf16-patched/summary.json`
for the full validated run.

## Upstream issues filed

- **#27951** [filed 2026-06-11] `[Bug] --moe-runner-backend flashinfer_cutlass + FP8 weights crashes with AttributeError`
  https://github.com/sgl-project/sglang/issues/27951

## Update 2026-06-11 20:35: autotune patch already on sglang main

The `sglang_cutlass_autotune_allowlist.diff` patch in this directory is
**already shipped on sglang main HEAD** via
[PR #26496](https://github.com/sgl-project/sglang/pull/26496) (Brayden Zhong,
2026-06-04). Our local sglang at `/home/t-jialianggu/work/sglang` is on
`study/v0.5.9` (a 6-month-old snapshot) which still needs the patch; sglang
upstream users do not.

We retain the patch file as historical record of independent diagnosis +
quantified validation (4.7-8.4x speedup, see
`docs/2026-06-11/harness_v1_4way_findings.md`).
