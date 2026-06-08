# vLLM 2×2 矩阵: autotune × cudagraph 实测,翻盘所有之前推论

## 实验设计

为了把 vLLM cutlass 性能的两个 factor 分开:
- `--enforce-eager` 关掉 cudagraph
- `enable_flashinfer_autotune: false` 关掉 startup autotune

4 个 cell × 3 runs × 3 regimes,GPU 1。

## 完整结果矩阵 (R_medium warm req/s)

|  | cudagraph **ON** | cudagraph **OFF** (eager) |
|---|---|---|
| **autotune ON** | **4.66** | 1.01 |
| **autotune OFF** | 1.36 | 0.93 |

R_short 矩阵:

|  | cudagraph **ON** | cudagraph **OFF** |
|---|---|---|
| autotune ON | 3.29 | 0.50 |
| autotune OFF | 1.31 | 0.46 |

R_long 矩阵:

|  | cudagraph **ON** | cudagraph **OFF** |
|---|---|---|
| autotune ON | 4.47 | 0.99 |
| autotune OFF | 1.37 | 0.93 |

## 因素隔离 (R_medium)

| 因素 | 单独贡献 |
|---|---|
| autotune,在 cudagraph **ON** 下 | **3.42×** |
| autotune,在 cudagraph **OFF** 下 | **1.09×** (几乎无效!) |
| cudagraph,在 autotune **ON** 下 | **4.63×** |
| cudagraph,在 autotune **OFF** 下 | 1.47× |
| 两个都 ON vs 都 OFF | **5.03×** |

## 关键洞察 — 两个 factor 必须配对才有用

**单开任一个几乎都没用**:
- 单开 autotune (没 cudagraph): 0.93 → 1.01 = **+9%** (几乎无效)
- 单开 cudagraph (没 autotune): 0.93 → 1.36 = **+47%** (有用但有限)
- **两个都开**: 0.93 → 4.66 = **+401%** (倍数级)

**这强烈暗示**: cudagraph 的优化机制依赖 autotune 提供的 tuned tactic。可能是:
- cudagraph capture 时把 tuned tactic "录"进 graph
- 后续 graph replay 时按 tuned tactic 跑
- 没 cudagraph 时,每个 forward 重新进 Python wrapper,虽然 AutoTuner cache hit 拿到 tuned tactic,但 Python wrapper 自身的 ~0.5 ms overhead 把 tuned kernel 的优势抵消掉
- 没 autotune 但有 cudagraph: graph 内是 fallback tactic,replay 快但 kernel 本身慢

更精确的猜测要 nsys profile 验证。

## 这把所有之前的结论翻盘

| 之前判断 | 现在 |
|---|---|
| "cudagraph 是 vLLM 比 sglang 快的主因" (fix1_invalidated.md) | 错: cudagraph 单独只给 1.47× |
| "autotune 是主因" (vllm_autotune_e2e_impact.md) | 错: autotune 单独只给 1.09× |
| **真实**: 两个互相依赖,必须配对 | 正确 |

## sglang 对照重新审视

| 配置 | R_medium req/s |
|---|---|
| vLLM cutlass: autotune ON + cudagraph ON | 4.66 |
| vLLM cutlass: autotune OFF + cudagraph ON | 1.36 |
| vLLM cutlass: autotune ON + cudagraph OFF | 1.01 |
| vLLM cutlass: autotune OFF + cudagraph OFF | 0.93 |
| sglang cutlass baseline (`--disable-cuda-graph`,autotune OFF) | 1.30 |
| sglang cutlass + cudagraph (warm cache, autotune OFF) | 1.35 |
| sglang cutlass + Bug A fix (autotune ON,`--disable-cuda-graph`) | 1.26 |

**重要观察**:
- sglang cutlass + cudagraph (1.35) ≈ vLLM autotune OFF + cudagraph ON (1.36) — **完美对应**
- sglang cutlass nograph baseline (1.30) > vLLM autotune OFF + cudagraph OFF (0.93) — **sglang 在 nograph 模式下比 vLLM eager 模式快 ~40%**!

**意外**: sglang 即使没 cudagraph,也比 vLLM eager 模式快。这说明 **sglang 的 nograph 路径有别的优化**(可能是 chunked-prefill / scheduler 优化),vLLM 完全 eager 时损失更大。

## 真正能让 sglang 追到 vLLM 4.66 的需求

需要**同时**:
1. sglang 走 cutlass 路径时 autotune 真的 prime 了正确的 cache key (修 Bug A 不够,因为 dummy_run 用的 batch shape 不对)
2. sglang 的 cudagraph 真的覆盖到 cutlass kernel 路径 (warm cache 状态 cudagraph 能开,但只 cover decode)

只满足一个,就只在 1.35 这个水平 (跟 vLLM autotune-off + cudagraph 一样)。

**这就是为什么 Bug A fix 实测 1.26 而不是 4.66** — 只满足了 autotune,没满足"autotune cache 跟 inference shape 匹配"。

## 修订之前所有相关 docs

- `fix1_invalidated.md` — "cudagraph 覆盖度是主因" → 改成 "两个 factor 互相依赖"
- `vllm_autotune_e2e_impact.md` — "autotune 是主因" → 同上修正
- `buga_fix_validation.md` — "Bug A 应该 +6×" → 现在解释: Bug A 即使生效也只能拉到 1.01 req/s (vLLM AT-ON + eager 对应),因为没 cudagraph + tuned 配对就没那么大优势

## 工件

- `results/4way_bench/vllm_autotune_impact/bench_vllm_AToff_eager_run{1,2,3}.json`
- `results/4way_bench/vllm_autotune_impact/bench_vllm_ATon_eager_run{1,2,3}.json`
- `results/4way_bench/vllm_autotune_impact/server_AToff_eager.log` (确认 enforce_eager + autotune disabled)
- `results/4way_bench/vllm_autotune_impact/server_ATon_eager.log` (确认 enforce_eager + autotune ran)

## 给"扩大搜索空间"问题的最终回答

User 最初问: 在 SM90 上扩大 cutlass autotuning 搜索空间能不能优化性能?

**答案**: 只在 **cudagraph 开** 的引擎里有用。
- vLLM (cudagraph + autotune 都 ON): 当前 4.66, 加 M=64 等候选可能再 +5-15% → ~5.0 req/s
- sglang (cudagraph 部分开 + autotune 没真用): 0 受益
- vLLM eager 模式 / sglang nograph: 0 受益 (autotune 不带 cudagraph 几乎没差)

**所以**:
- 不要在 sglang 上花时间(autotune 集成有更深的 bug)
- 不要在 eager 模式上花时间(autotune 跟 cudagraph 不配对没用)
- 真要榨这部分性能,**只能去给 flashinfer 上游提 PR 加 SM90 候选**,让 vLLM 用户受益
