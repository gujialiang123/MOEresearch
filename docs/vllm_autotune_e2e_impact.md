# vLLM autotune 影响 e2e 实测 — 2.5-3.4× 真的

## 实验

在 GPU 1 上启动 vLLM with `--kernel-config '{"moe_backend": "flashinfer_cutlass", "enable_flashinfer_autotune": false/true}'`,3 runs × 3 regimes,对比 e2e 吞吐。

## 结果

### Warm-only (run 2 + 3 avg)

| Regime | autotune **关** | autotune **开** | autotune speedup |
|---|---|---|---|
| R_short  | 1.31 req/s | 3.29 req/s | **2.5×** |
| R_medium | 1.36 req/s | 4.66 req/s | **3.4×** |
| R_long   | 1.37 req/s | 4.47 req/s | **3.3×** |

### 跟其它配置对照

| 配置 | R_medium req/s |
|---|---|
| **vllm_cutlass autotune 开** | **4.66** |
| vllm_cutlass autotune 关 | 1.36 |
| sglang_cutlass (基线,无 autotune) | 1.30 |
| sglang_cutlass + Bug A fix | 1.26 |
| vllm_triton (cudagraph default) | 4.71 |
| sglang_triton (cudagraph default) | 4.51 |

## 含义

### 1. autotune 是 vllm_cutlass 性能的真正驱动

之前几篇分析 (`fix1_invalidated.md`, `buga_fix_validation.md`) 一直在猜:
- "差距是 cudagraph 覆盖度"?
- "差距是 AutoTuner cache miss"?
- "差距是 wrapper overhead"?

**直接对照**告诉我们: **autotune 关掉,vLLM 跟 sglang 一样慢**。
autotune 开,vLLM 跳 3.4×。

### 2. sglang 慢的本质 = "没真用 autotune"

- vllm 关 autotune ≈ sglang 现状 (1.36 vs 1.30 — 同一水平)
- 说明 sglang 跟 "vLLM autotune-off 模式" 是等价的

### 3. Bug A fix 为什么没用 — 现在能解释

`buga_fix_validation.md` 里我们 apply 了 Bug A fix,server log 显示 autotune 跑了,
但 e2e 没改善。**对比这次实验**:

| | autotune 跑了? | e2e 收益 |
|---|---|---|
| vLLM autotune ON | ✓ | 3.4× ✓ |
| sglang Bug A fix | ✓ (log 确认) | **0×** ✗ |

差别: vLLM 的 autotune 真的 prime 了 cache 给后续推理用; sglang 的 autotune 跑了
但 cache 没被推理路径复用。这验证了之前的假设之一:**sglang `_dummy_run` 的 shape
跟实际推理 shape 不匹配**,或者**dummy_run 没真激活 cutlass_fused_moe 路径**。

### 4. 关于"扩大搜索空间"的判断

User 最初问 "扩大 SM90 autotuning 搜索空间能不能优化性能"。答案现在更清晰:

| 在哪里扩 | 收益预估 |
|---|---|
| vLLM (已经用 autotune) | **可能 10-30%** 进一步提升 (在 4.66 基础上,加 M=64 等候选可能再涨) |
| sglang (没真用 autotune) | **0% 收益** — 即使候选库扩大 1000×,sglang 不查就没用 |

**真要拿到这部分收益,顺序是**:
1. **先**修 sglang autotune 集成 bug(让 sglang 真的查 cache),拉到 4.66 req/s 水平
2. **然后**才有意义讨论 "扩大候选库" 能不能再涨 10-30%

### 5. cudagraph 的角色重新评估

之前以为 cudagraph 是 vLLM 比 sglang 快的主因。**不对**。
对比:
- vLLM autotune ON + cudagraph FULL: 4.66 req/s
- vLLM autotune OFF + cudagraph FULL: 1.36 req/s

cudagraph 在 ON 一直开着,**但 autotune off 时它救不了**。说明 cudagraph 帮助消除
**CPU launch overhead**,**但消除不了 GPU 本身 kernel 慢的时间**。

也就是:
- autotune 决定**GPU kernel 时间**(0.146 ms vs 1.36 ms)
- cudagraph 决定**CPU launch overhead** (~5 ms × 500 kernel = 2.5 ms or near-0)

两个都重要,但 e2e 性能在我们这个测试上,**autotune 影响更大**。

## 实验工件

- `results/4way_bench/vllm_autotune_impact/bench_vllm_cutlass_no_autotune_run{1,2,3}.json`
- `results/4way_bench/vllm_autotune_impact/bench_vllm_cutlass_with_autotune_run{1,2,3}.json`
- `results/4way_bench/vllm_autotune_impact/server_no_autotune.log` (autotune skipped 日志)
- `results/4way_bench/vllm_autotune_impact/server_with_autotune.log` (autotune ran ~3s 日志)

## 修订所有相关 docs

需要更新:
- `sglang_vs_vllm_flashinfer_cutlass_analysis.md` 的 TL;DR 和 §3 —— "cudagraph 是主因" 修正成 "autotune 是主因"
- `fix1_invalidated.md` 的"真正根因 = cudagraph 覆盖度" —— 修正成 "autotune 集成"
- `buga_fix_validation.md` 的三个假设 —— 现在有 #1 (cache key mismatch) 或 #2 (dummy_run 没激活 MoE) 的实证支持
