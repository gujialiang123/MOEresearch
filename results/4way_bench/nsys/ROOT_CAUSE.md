# ⚠ DEPRECATED 2026-06-08 — 整个文件结论被实验推翻

> Fix 1 (`tune_max_num_tokens=8192`) 实测无效 (-6%)。
> 真正的 9x kernel launch 差距来源是 **cudagraph 覆盖度**,不是 AutoTuner re-benchmark。
> `flashinfer/autotuner.py:432-451` 显示 runtime `choose_one` 在 `is_tuning_mode=False`
> (默认) 时直接返回 fallback tactic,根本不 launch 候选 kernel sweep。
> 详见 `docs/fix1_invalidated.md`。

本文以下内容已过时,保留只是 history。

---


# Root cause — why sglang flashinfer_cutlass is 3.4-4.7× slower than vLLM's

## NSys kernel count: smoking gun

R_medium (16 reqs × 256 tokens), same `flashinfer.fused_moe.cutlass_fused_moe`:

| metric | vLLM CUTLASS | sglang CUTLASS | ratio |
|---|---|---|---|
| `cutlass::device_kernel<...sm90...gemm...>` calls | 10,802 | 97,774 | **sglang 9.05×** |
| avg kernel time | 178 us | 142 us | sglang slightly faster per-kernel |
| Total `cutlass-sm90-gemm` time | 1924 ms | 13951 ms | sglang 7.25× more |
| GPU kernel time / token | 2.6 calls/token | **23.9 calls/token** | 9.2× |

So **per-kernel** sglang isn't slower — it's launching **9× more kernels** for the same work.

## Source code diff — same flashinfer entry point

Both engines call the same `flashinfer.fused_moe.cutlass_fused_moe`:

| arg | sglang (`unquant.py:373-386`) | vLLM (`flashinfer_cutlass_moe.py:378-403`) |
|---|---|---|
| `output=` | **omitted** (allocates new each call) | preallocated buffer |
| `tune_max_num_tokens` | `next_power_of_2(x.shape[0])` — **changes per forward** | `max(self.max_capture_size, 1)` — **fixed** |
| `fc1/fc2_expert_biases` | omitted | explicit None |
| `swiglu_alpha/beta/limit` | omitted | explicit |
| `activation_type` | omitted | explicit |

## The actual bug: AutoTuner re-benchmarks on every call

`flashinfer/fused_moe/core.py:490-578` — every call to `cutlass_fused_moe`:

```python
def cutlass_fused_moe(...):
    tuner = AutoTuner.get()
    MoERunner.refine_tuning_config(tune_max_num_tokens)   # cached on tune_max value

    # GEMM1 tuning — picks best tactic, benchmarking candidates if cache misses
    _, gemm_tactic_1 = tuner.choose_one(
        "trtllm::fused_moe::gemm1", [moe_runner],
        MoERunner.tuning_config, [...inputs...], gemm_idx=1,
    )
    # GEMM2 tuning — same
    _, gemm_tactic_2 = tuner.choose_one(
        "trtllm::fused_moe::gemm2", [moe_runner],
        MoERunner.tuning_config, [...inputs...], gemm_idx=2,
    )
    # Actual MoE run, using the chosen tactics
    run_moe(...)
```

When the tuner's cache key changes, `choose_one` runs the candidate kernels to benchmark them — these are real GEMM kernel launches, captured by nsys.

**vLLM**: `tune_max_num_tokens` is constant (`max_capture_size`); shapes hit cudagraph's fixed capture_sizes. Cache hits → tuning runs once per shape, never again.

**sglang**: `tune_max_num_tokens = next_power_of_2(x.shape[0])` changes per forward; no cudagraph means shapes can vary too. Cache misses → tuning re-benchmarks frequently.

This produces the ~9× extra cutlass-sm90-gemm kernel launches we observed.

## Proposed fixes (ordered by effort/risk)

### Fix 1 — one-line change to `tune_max_num_tokens` (Lowest risk)

In `python/sglang/srt/layers/quantization/unquant.py:385`:

```python
# Before
tune_max_num_tokens=next_power_of_2(x.shape[0]),
# After
tune_max_num_tokens=8192,  # match flashinfer default, fixed for tuning cache stability
```

Expected gain: significant reduction in re-tune frequency.
Risk: 8192 may not be optimal for very small batches; consider `max(8192, next_power_of_2(x.shape[0]))` or expose as a config.

### Fix 2 — wrap call in `autotune(False)` context (medium effort)

Disable AutoTuner re-benchmarking after first warm-up pass:

```python
from flashinfer.autotuner import autotune  # check actual import path

# after first warmup
with autotune(False):
    output = flashinfer_cutlass_fused_moe(...)
```

Expected gain: completely eliminates `choose_one` benchmark kernels after warm-up.
Risk: tactic chosen during warm-up applies to all subsequent shapes (may be suboptimal for shapes very different from warm-up).

### Fix 3 — fix sglang's cudagraph hang with flashinfer_cutlass (Best long-term)

When `--moe-runner-backend flashinfer_cutlass` is set with cudagraph enabled,
the detokenizer process freezes after "Capture cuda graph end" log line.
Reproduced 2× during checkpoint 005 work. With cudagraph fixed:
- shapes are bounded to capture_sizes → tune cache hits
- per-step launch overhead drops to graph-replay cost
- closes most of the 3.4-4.7× gap

This is what vLLM does (`cudagraph_mode=FULL_AND_PIECEWISE`).

## Verification plan

Apply Fix 1 only (5-min change), re-run 3 sglang_cutlass runs, expect:
- sglang_cutlass R_medium: 1.31 req/s → projected 2-3 req/s
- nsys: cutlass-sm90-gemm calls drop from 97774 toward ~10000

If Fix 1 closes most of the gap, Fix 2/3 are optional. If not, the gap is also coming from missing cudagraph (Fix 3).
