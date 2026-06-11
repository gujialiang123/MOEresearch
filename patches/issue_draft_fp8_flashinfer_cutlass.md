# Issue draft — to be posted to sgl-project/sglang
#
# This file is for review. Do not post until reviewed.
# Once approved, post via:
#   gh issue create -R sgl-project/sglang \
#     --title "<title from below>" \
#     --body-file patches/issue_draft_fp8_flashinfer_cutlass.md \
#     --label bug

## Title

[Bug] `--moe-runner-backend flashinfer_cutlass` + FP8 weights crashes with `AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'`

## Body

### Summary

When launching sglang with an FP8 quantized MoE model and explicitly selecting
`--moe-runner-backend flashinfer_cutlass`, the server passes startup and CUDA
graph capture cleanly, but **crashes on the first forward pass** with:

```
AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'
```

The combination is silently accepted at config time and only fails when an actual
request arrives, which is a poor user experience. The expected behavior is either:

1. A clear `NotImplementedError`/`ValueError` at server startup explaining that
   `flashinfer_cutlass` is not supported on the FP8 path (and pointing users at
   `--moe-runner-backend cutlass` instead), **or**
2. A real implementation of FP8 + flashinfer cutlass MoE.

The lightweight first option would already remove a sharp edge for users.

### Environment

- sglang: main HEAD (commit `10219bd9d` at time of repro)
- flashinfer: 0.6.3
- torch: 2.9.1 / triton: 3.5.1
- transformers: 5.8.1
- Hardware: 1× NVIDIA H200 (SM 9.0)
- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`

### Reproducer

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --moe-runner-backend flashinfer_cutlass \
    --tensor-parallel-size 1 \
    --port 31103
```

Then any request:
```bash
curl http://127.0.0.1:31103/generate -H 'content-type: application/json' \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 8}}'
```

### Server log (truncated to the relevant frame)

```
File "sglang/python/sglang/srt/layers/quantization/fp8.py", line 1447, in apply
    if self.runner.runner_backend.is_deep_gemm():
       ^^^^^^^^^^^
AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'
```

### Root cause analysis

In `sglang/srt/layers/quantization/fp8.py`, `Fp8MoEMethod.create_moe_runner`
only assigns `self.runner` for a limited set of backends:

```python
# python/sglang/srt/layers/quantization/fp8.py (main HEAD, ~line 1779)
if (
    moe_runner_backend.is_deep_gemm()
    or moe_runner_backend.is_triton()
    or moe_runner_backend.is_aiter()
    or moe_runner_backend.is_flashinfer_trtllm()
    or moe_runner_backend.is_flashinfer_trtllm_routed()
):
    self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
else:
    # TODO(cwan): refactor other backends
    pass
```

When the user passes `--moe-runner-backend flashinfer_cutlass`, none of the
`is_*()` checks match, so the `else: pass` branch silently runs and
`self.runner` is never set.

Later in the same module, `Fp8MoEMethod.apply()` (~line 1938 on main)
dispatches by reading `self.runner.runner_backend`:

```python
if self.runner.runner_backend.is_deep_gemm():
    ...
elif self.runner.runner_backend.is_flashinfer_trtllm() ...:
    ...
```

Hitting `self.runner` then raises `AttributeError`.

Note: the FP8 path already has a separate `cutlass` branch a few lines earlier
that uses sglang's native `cutlass_fused_experts_fp8`:

```python
if get_moe_runner_backend().is_cutlass():
    output = cutlass_fused_experts_fp8(...)
    return StandardCombineInput(hidden_states=output)
```

So the working path for FP8 + cutlass is `--moe-runner-backend cutlass`, not
`flashinfer_cutlass`. Users hitting the bug are likely confused by the BF16
path, where `flashinfer_cutlass` is the right (and now well-tuned, since
#26496) choice.

### Suggested fix (minimal)

Add an explicit failure mode in `Fp8MoEMethod.create_moe_runner` so the
incompatibility surfaces at startup, not at first request:

```python
        if (
            moe_runner_backend.is_deep_gemm()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_aiter()
            or moe_runner_backend.is_flashinfer_trtllm()
            or moe_runner_backend.is_flashinfer_trtllm_routed()
        ):
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        elif moe_runner_backend.is_flashinfer_cutlass():
            raise NotImplementedError(
                "FP8 MoE does not currently support --moe-runner-backend "
                "flashinfer_cutlass. Use --moe-runner-backend cutlass for FP8 "
                "(sglang's native cutlass_fused_experts_fp8 implementation). "
                "flashinfer_cutlass is currently only supported on the BF16 "
                "path (unquant.py)."
            )
        else:
            # TODO(cwan): refactor other backends
            pass
```

This is a strictly user-experience fix; the underlying implementation gap
(no FP8 path through flashinfer's cutlass MoE kernel) is a separate
discussion.

### Related

- PR #26496 ("Changes for SM120 perf and usability for NVFP4", merged
  2026-06-04) recently re-enabled `flashinfer_cutlass` in the FlashInfer
  autotune allowlist, restoring a 4.7-8.4× speedup on H200 BF16 models when
  users explicitly request `flashinfer_cutlass`. This is excellent — but it
  also makes it more likely users will reach for `flashinfer_cutlass` on
  FP8 models, which is exactly the path that hits this AttributeError.
  See our independent validation: PR #26496 with FP8 attempt also crashes the
  same way (would have validated this issue earlier if we'd tried FP8 first).

### Discovery context

Found while benchmarking sglang BF16 vs FP8 throughput on H200 with a
deterministic bench harness. The BF16 path with `flashinfer_cutlass` worked
beautifully (5-8× over the triton default once #26496-style autotune is
enabled); migrating the same flag to FP8 produced this crash. Happy to share
the per-regime benchmark numbers and the bench-spec setup if that helps narrow
down the failure mode.

### Willingness to contribute

I'm happy to send a PR for the minimal user-experience fix (option 1 above) if
maintainers agree that's the right framing. The real implementation (option 2)
is out of scope for me — I don't know whether flashinfer's cutlass MoE has a
working FP8 path on Hopper that sglang could plug into.
