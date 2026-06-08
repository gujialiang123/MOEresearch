# Fix 1 验证 — 失败,原 ROOT_CAUSE 分析错了

## 实验结果

测试: 在 GPU 1 上 apply `unquant.py:385` patch (`tune_max_num_tokens=8192`),
3 runs × 3 regimes,nograph mode (跟原 baseline 一致):

| Regime | Broken baseline | Fix 1 | 变化 |
|---|---|---|---|
| R_short  | 0.700 req/s | 0.656 req/s | **-6.2%** (变慢) |
| R_medium | 1.299 req/s | 1.218 req/s | **-6.2%** (变慢) |
| R_long   | 1.290 req/s | 1.205 req/s | **-6.6%** (变慢) |

nsys kernel count 对比 (R_medium 同样 workload):
- Broken: 97774 cutlass-sm90 calls
- Fix 1: 94896 calls (只少 3%, 远不是预期的 ~10000)

**Fix 1 没改善 9x 差距,反而略微变差**。

## 原 ROOT_CAUSE 分析错在哪

我之前 (`ROOT_CAUSE.md`, `analysis.md` 4.3) 说:
> "每次 cutlass_fused_moe() 调用都跑 AutoTuner.choose_one() 两次,
>  cache miss 时会真的 launch 候选 kernel 做 micro-benchmark"

**实际不是**。看 `flashinfer/autotuner.py:432-451` 源码:

```python
input_shapes = tuple(self._get_input_sizes(inputs))

# Early return if it's not tuning, use cache found one or fallback one
if not self.is_tuning_mode:
    is_cache_hit, runner_id, tactic, stored_profile = self.search_cache(...)
    runner = runners[runner_id]
    return runner, tactic    # 直接返回,不 launch 任何 kernel
```

`AutoTuner.is_tuning_mode` 默认 False。**只有显式包在 `with autotune(): ...` ctx 里才会进入 sweep 分支**。

Runtime 时 `choose_one()` 在 cache miss 时只是返回 fallback tactic (-1),让 runner 用默认 kernel,**不 sweep**。

所以 sglang 上观察到的 97774 cutlass-sm90 call 跟 vLLM 10802 的 9x 差距,**根本不是 AutoTuner re-benchmark 造成的**。

## 真正的差距来源 (cudagraph 覆盖度)

数字算 (R_medium):
- decode tokens: 16 reqs x 256 = 4096
- prefill tokens: 16 x 800 ~ 12800
- forward calls: ~512 个 forward (prefill + decode)
- layers x gemms per forward: 48 x 2 = 96

| | total cutlass calls | calls / forward | calls / (layer * gemm * forward) |
|---|---|---|---|
| vLLM cutlass | 10802 | 21 | **0.22** |
| sglang cutlass | 97774 | 191 | **1.99** |

**sglang ~ 2 calls/layer-gemm/forward**: 正常预期,每 layer 真的 launch 2 个 kernel (gate+up, down)
**vLLM ~ 0.22**: 远低于 1,因为 **cudagraph 把整个 forward 压成几个 graph launch**,nsys 看到的 "call" 是 graph replay,不是每个 kernel 单独 launch

**所以差距是 cudagraph 造成的,不是 AutoTuner**。

## 这把所有结论翻盘了吗?

部分翻盘。

### 仍然成立的

- vLLM cutlass ~ vLLM triton on R_medium (e2e)
- sglang cutlass 慢 sglang triton 3.4-4.7x (实测)
- B200/GB200 hand-tuned 表是 SM100 独有的优势 (B 部分仍正确)
- SM100 vs SM90 候选 kernel 搜索空间 ~12x 差距 (C 部分仍正确)
- conc=64 上 vLLM cutlass +17% (12 部分实测)

### 翻盘的

- ~"sglang 慢 3.4-4.7x 的根因是 AutoTuner re-benchmark 每个 forward 触发"~
- **真正根因**: sglang flashinfer_cutlass 在 nograph mode 下,每个 forward 必须走 Python wrapper + 真 launch 每个 kernel;vLLM 有 cudagraph 把这些都压在 graph 里

### Fix 1 判决: 完全无用

不再推荐做。改 `tune_max_num_tokens` 对实际 runtime 行为没影响。

### Fix 3' (给 SM90 写 hand-tuned 表) 重新评估

Fix 3' 解决的是 "AutoTuner 在 tuning_mode 下能不能挑到最优 tactic" 的问题。
但 runtime 默认 is_tuning_mode=False,根本不进入 sweep 分支,**hand-tuned 表也用不上**。

要让 hand-tuned 表生效,需要先让 sglang 在启动时显式包一次 `with autotune(): _dummy_run()`
(模式 2 的 startup warmup)。但 sglang 因为 Bug A (compilation error) 把 flashinfer_cutlass
从 _flashinfer_autotune() 路径里注释掉了。**所以 Fix 3' 实际起效要先修 Bug A**。

## 新的真正的 fix path

要让 sglang_cutlass 追到 vllm_cutlass 水平,**核心要让 sglang 的 cutlass 路径用 cudagraph**。

但 sglang cutlass + cudagraph 现在状态 (11 部分实测):
- **prefill 路径不走 cudagraph** (sglang 设计就是 decode-only graph capture)
- **decode 路径走 cudagraph** (我们 11 部分 checkpoint 007 实测,decode 端 cuda graph: True 出现 112 次)

R_medium 是 prefill-heavy (800 词 prompt vs 256 输出),所以 cudagraph 帮不大。

要彻底追上,要么:
1. 让 sglang 的 prefill 也走某种 graph (vLLM piecewise CUDA graph 思路) — 工程量大
2. 优化 sglang prefill 路径的 Python wrapper,减少 per-step 开销 — 需要 profile sglang 自己

**结论**: Fix 1 无效,Fix 3' 要先修 Bug A 才能起效,差距本质在 cudagraph 覆盖度,不在 tuning。

## 文档需要更新的地方

- `sglang_vs_vllm_flashinfer_cutlass_analysis.md` 4.3, 4.5 — 撤回 AutoTuner re-benchmark 根因
- `nsys/ROOT_CAUSE.md` — 整个文件需要 deprecate 或重写
- `triton_vs_cutlass_moe_kernel_source_comparison.md` — 没受影响 (源码对比仍正确)
- `server_lifecycle_and_sm100_tuning.md` C 部分 — SM100 vs SM90 候选差距仍正确,但 Fix 1 章节要删
