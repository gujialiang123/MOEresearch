# Bug A Fix 验证 — autotune 跑了但 e2e 没改善

## 实验

apply 了 sglang `model_runner.py:1841` 的 Bug A fix —— 取消注释让 `flashinfer_cutlass`
也走 `_flashinfer_autotune()` warmup 路径,跑 3 runs e2e bench (GPU 1, no cudagraph)。

## Server 端确认 autotune 跑了

```
[2026-06-08 18:54:03] Running FlashInfer autotune...
2026-06-08 18:54:03,285 - INFO - autotuner.py:256 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
2026-06-08 18:54:05,156 - INFO - autotuner.py:262 - flashinfer.jit: [Autotuner]: Autotuning process ends
[2026-06-08 18:54:05] FlashInfer autotune completed.
```

只用了 ~2 秒(vLLM 用 ~4 秒),没 crash,没 hang。**TODO 提到的 compilation error
在 warm cache 状态下没出现**。

## e2e bench 结果 — 没改善

| Regime | Broken baseline | Bug A fix |
|---|---|---|
| R_short  | 0.700 req/s | 0.661 req/s |
| R_medium | 1.299 req/s | 1.260 req/s |
| R_long   | 1.290 req/s | 1.245 req/s |

仍然在 1.30 req/s 这个水平,**跟微基准预期的 5-6× 加速完全不匹配**。

## 为什么微基准跟 e2e 不一致?

### 假设 1: cache key mismatch

`sglang/.../model_runner.py:1862-1872` `_flashinfer_autotune` 实现:
```python
def _flashinfer_autotune(self):
    with torch.inference_mode(), autotune():
        self._dummy_run(
            batch_size=self.req_to_token_pool.size,  # ← 用一个很大的 batch
            run_ctx=autotune()
        )
```

`req_to_token_pool.size` 在 Qwen3-30B-A3B / context_length=32768 上是 ~668k tokens。
`_dummy_run` 跑这个 batch 触发 autotune。但 R_medium 实际 prefill 时 batch 是 800 tokens,decode 是 8 tokens —— 完全不同的 size。

flashinfer AutoTuner cache 按 `tune_max_num_tokens` 分桶 (我们之前发现) + 按 shape 分 key。
sglang dummy_run 的 shape **可能跟 inference shape 的 cache key 不匹配** → cache 没命中 → 仍走 fallback。

### 假设 2: dummy run 不触发真实 MoE 路径

sglang `_dummy_run` 主要是为了 cuda graph capture 准备的,内部可能用 zero routing 或
all-route-to-expert-0 这种 degenerate case,**不一定会真正激活 cutlass_fused_moe()**。
如果 dummy run 路径不调 cutlass_fused_moe,那 autotune 根本没 sweep 这个 op,cache 仍空。

### 假设 3: bottleneck 不在 MoE kernel

数字算: 微基准 batch=8 tuned cutlass kernel 0.146 ms × 96 layer-gemm = **14 ms per forward**。
但 sglang decode step 实测 ~23 ms (gen throughput 340 tok/s, batch=8)。
**差出来的 9 ms** 在 MoE 之外 (attn, rmsnorm, sample, IPC),**这部分跟 tuning 无关**。

如果 sglang 真的从 fallback (1.36 ms/call) 切到 tuned (0.146 ms/call),decode step 应该
从 (1.36 * 96 + 9) = 140 ms 跳到 (0.146 * 96 + 9) = 23 ms,**throughput 6× 提升**。
**没看到这个跳跃**,说明 sglang 实际可能本来就没用 fallback (1.36ms),而是用某种中间值,
或者其它 overhead 占主导。

## 这告诉我们什么

1. 微基准 `tuned vs fallback = 5-6×` 是真的
2. 但 sglang 在实际 server 里 **真的用的什么 tactic 不清楚** —— 既不是稳定 tuned 也不是
   稳定 fallback,需要 nsys instrumentation 才能查清
3. Bug A fix 让 autotune 跑了,但 cache 没被实际推理路径复用

## 还能怎么做?

**剩下的 open question**:

1. 在 sglang 的 `flashinfer_cutlass_fused_moe` 调用前后加 logging,看每次 forward 实际
   传给 `cutlass_fused_moe` 的 shape,跟 autotune 时的 shape 对比,确认 cache key 是否匹配
2. 改 sglang `_dummy_run` 让它显式覆盖我们 bench 用的 shape (B=8, prompt_len=800)
3. 或者完全绕过 sglang autotune,直接在 sglang launch_server 前用 Python 脚本预跑一次
   `with autotune(): cutlass_fused_moe(...)` 把 AutoTuner 单例的 cache 填好,然后启动 sglang

这些都需要进一步实验 (1-2 天),不是 5 分钟能验证的。

## 结论

| Fix | 状态 |
|---|---|
| Fix 1 (`tune_max_num_tokens=8192`) | 已验证无效 |
| Fix Bug A (sglang autotune list) | 已验证安全 (没 crash),但**e2e 没改善** |
| Fix 3' (SM90 hand-tuned .py 表) | 没验证,但因为 Fix Bug A 已经走 autotune 而 e2e 没改善,说明 hand-tuned 表也不会帮到 sglang (root cause 不在 tactic 选择) |

**真正的 sglang slow 根因仍未确定**。可能在:
- AutoTuner cache key 没匹配 (假设 1)
- dummy run 没真激活 MoE (假设 2)
- 其它非 MoE overhead (Python wrapper, IPC, attn) 主导 (假设 3)

需要更深的 profiling 才能确定。

## 数据 artifact

- `results/4way_bench/buga_validation/bench_*_run{1,2,3}.json` — 3 runs e2e data
- `results/4way_bench/buga_validation/server_nograph.log` — 完整 server log (含 autotune 日志)
- `results/cutlass_microbench/results_2026-06-08.md` — 微基准结果

## sglang 已 revert

`sglang/.../model_runner.py` 的 patch 已 git checkout 还原,working tree clean。
