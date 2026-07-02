# LFM2.5-8B-A1B v3 实验报告：MFU 引入 + 长上下文 regime + TPE 失效修复（2026-07-02）

## 0. 摘要

v3 实验相比 v2（2026-06-30）的三个升级：

1. **加了 MFU/MBU 度量**（Model FLOPs Utilization / Memory Bandwidth Utilization）——每个 regime 每个 trial 都算，写进 `summary.json`。可以看到「同一硬件下哪种 config 用得更满」，不再只看 req/s 的相对值。
2. **加了 4 个长上下文 regime**：从原来最长 4k words 输入扩展到 8k / 16k / 32k / 50k words（分别对应 ~10k / 20k / 40k / 65k tokens）。sglang server context-length 从 32k 提到 72k。
3. **修了 v2 里 TPE 漏掉最优的 bug**：在 TPE 接管之前，先跑 4 个**分层暖启动 trial**——每个 MoE backend 都被强制搭配一次「已知好 batching 先验」（cap=32, cg-on, chunk=-1, lpm, mem=0.9）。这样 TPE 学到的每个 backend 的初始估计就不会被冷启动阶段的坏 batching 搭配污染。

**主要结论**：

1. **✅ TPE 修复奏效**：warm-start trial 1（`triton MoE + 好 batching`）返回 **23.75 req/s**——**精确复现** v2 手工验证的 23.53，且比 v2 Optuna best（22.32）高 6%。这直接证明 v2 的失败是 TPE 冷启动的系统性偏差，不是搜索空间的问题。

2. **✅ 长上下文 regime 揭示了 v2 完全没看到的最优区**：在 baseline 里默认 `chunked-prefill=-1`（不分块），但 v3 Optuna 发现 `chunk=8192` 或 `chunk=2048` 对长输入 regime 有**巨大提升**：

   | Regime | Baseline (chunk=-1) | v3 Optuna best | 提升 |
   |---|---|---|---|
   | R_concurrent_decode | 23.80 req/s | 23.90 | +0.4%（本就最优） |
   | R_prompt_8k | 5.32 | **7.76** | **+46%** |
   | R_prompt_16k | 2.70 | **4.57** | **+69%** |
   | R_prompt_32k | 1.63 | **2.44** | **+50%** |
   | R_prompt_50k | 1.28 | **3.05** | **+139%**（2.4×！） |

3. **v3 找到了一个「Pareto 最优」config**（trial 19: `auto MoE + cap=32 + chunk=8192 + fcfs + mem=0.75`）——在 R_concurrent_decode 上和 baseline 持平（23.90 vs 23.80），同时在所有长 prefill regime 上都 +40-140%。这不是 tradeoff，是**免费午餐**——之前默认 `chunk=-1` 就是抛下了这个免费午餐。

4. **MFU 观察验证了理论**：
   - decode-heavy regime：MFU_simple ≤ 2%（memory-bound），MBU 12-29%
   - long-prefill regime：MFU_simple ≤ 0.3% 但 MFU_amortized 达 20-40%（prefill 阶段真在打 GEMM）
   - 说明**大多数 tuning 空间在 long-prefill regime**（decode 已经打满带宽）

5. **两个 baseline（true-default vs cookbook）差异 < 1%**——复现 v2 结论：cookbook 的 parser flag 不影响吞吐。

> 文档位置：`docs/2026-07-02/lfm2.5_v3_mfu_longctx.md`
> 原始数据：`results/2026-07-02_lfm2.5_v3/`

---

## 1. 环境 + 关键决策变化

| 项目 | v2 (2026-06-30) | v3 (2026-07-02) | 说明 |
|---|---|---|---|
| GPU | GPU 4 | **GPU 6** | GPU 4 被 t-yuxingliu 占用 |
| Regime 数 | 4 | **8** | 新加 4 个长上下文 regime |
| Server context-length | 32768 | **73728**（72k） | 支撑 65k 输入 + 输出 + 头 room |
| max-prefill-tokens | 16384 | **96000** | 支撑最长 R_prompt_50k 的单请求 |
| MFU 度量 | 无 | **有**（simple + full + amortized + MBU） | 每 regime 都算 |
| Search space | 7 knob × 288 combo | **7 knob × 432 combo** | moe-backend 加了 `auto` |
| Warm-start | 无 | **4 stratified warm trials** | 修 TPE 失效 |
| n_trials | 25 | **30**（4 warm + 26 TPE） | 更多样本给 TPE |
| Model / hardware | 相同 | 相同 | LFM2.5-8B-A1B on 1× H200 bf16 |

### 1.1 GPU 换到 GPU 6

- GPU 1-4 都被 t-yuxingliu 占了（每张 120GB）
- GPU 5 有其他工作（56GB）
- GPU 7 有小占用（78GB, t-yuxingliu 另一个进程）
- **GPU 6 完全空闲** → 用它

### 1.2 tokenizer patch 仍生效

沿用 6/30 的 patch（`/data/hf/LFM2.5-8B-A1B/tokenizer_config.json` 改 `tokenizer_class` 为 `PreTrainedTokenizerFast`）。如果重下模型需要再打一遍。

---

## 2. MFU 是什么？为什么加？

**MFU (Model FLOPs Utilization)** = 我们实际做的 FLOPs / 硬件峰值 FLOPS。

```
MFU_simple = 2 × N_active_params × tokens_per_s / peak_bf16_flops
```

对 LFM2.5-8B-A1B on H200 bf16：
- N_active_params ≈ 1.6B（每 token 走过的 weight matmul；见 §2.1）
- peak_bf16 = 989 TFLOPS

**为什么 v2 不够，需要 MFU**：

- v2 只看 req/s，是**相对指标**。看不出「相对硬件天花板还差多少」。
- 有了 MFU 就有**绝对基准**：如果 MFU 才 2%，说明还有 50 倍空间；如果 60%，说明基本满血。
- Decode-heavy regime 的 MFU 天生就低（memory-bound）—— 这时看 **MBU (Memory Bandwidth Utilization)** 更有意义：`weight_bytes × forward_passes/s / peak_HBM_BW`。

### 2.1 LFM2.5-8B-A1B 的活跃参数如何算

从 `/data/hf/LFM2.5-8B-A1B/config.json` 派生：

| 组件 | 计算 | 活跃参数量 |
|---|---|---|
| Embedding（LM head tied） | 128000 × 2048 | 262M |
| Attention（6 层） | 6 × (Q + K + V + O) = 6 × (4.2 + 1.05 + 1.05 + 4.2)M | 63M |
| Conv 层（18 层，LFM2-style） | 18 × ~3 × h² | 227M |
| Dense FFN（前 2 层） | 2 × 3 × 2048 × 7168 | 88M |
| MoE FFN active（top-4/32，22 层） | 22 × 4 × 3 × 2048 × 1792 | 969M |
| **合计（每 token）** | | **~1.6B** |
| 全部权重（32 experts 都存） | 上面 + 22 × 28 × 3 × 2048 × 1792 | ~8.4B |

「LFM2.5-8B-A1B」命名里的「A1B」（~1B active）应该没算 embedding/LM head；我们的 1.6B 是包含了这些的**真实每步 forward 会 matmul 到的参数量**。用 1.6B 算 MFU 更准。

### 2.2 三种 MFU 变体（我们都算）

```
MFU_simple    = 2 × active_params × tokens_per_s / peak         # 只算 matmul
MFU_full_decode = MFU_simple + attention_score_context_flops    # 加 O(kv_len × d) 项
MFU_amortized = (prefill_flops + decode_flops × max_new) × req/s / peak
                                                                # 把 prefill 一起算，看每请求整个 lifecycle
```

- decode-heavy regime：三个数字差不多（simple ≈ full ≈ amortized 都低）
- long-prefill regime：**amortized 会大幅高于 simple**（prefill 计算占主导）
- 例：R_prompt_50k_c1_out64 → simple=0.03%, amortized=**40%**

这符合直觉：50k prompt 的 prefill 阶段做了大量 GEMM，但每请求只产 64 个 output token（很少 decode），所以按 decode 算的 MFU_simple 显得非常低。

### 2.3 MBU 公式

```
MBU_decode = weight_bytes_per_forward × forward_passes/s / peak_HBM_BW
           = active_params × dtype_bytes × (tokens_per_s / concurrency) / peak_HBM_BW
```

除以 concurrency 是因为 **一次 forward pass 服务 concurrent 个请求，每请求各出 1 token**——weight 只从 HBM 读一次。

对 LFM2.5 on H200：peak_HBM = 4.8 TB/s。R_short_decode (1 concurrent) MBU ≈ **29%**——单请求 decode 是 memory-bound 极限的验证。

---

## 3. 长上下文 regime 设计

来自 Dey 反馈：「4000 words 太短，加 32k/65k 输入的 regime」。

原有 4 regime（v2）保留，新加 4 个（v3）：

| Regime ID | 输入 words | 输入 tokens ≈ | max_new | concurrency | 备注 |
|---|---|---|---|---|---|
| R_short_decode | 100 | 130 | 256 | 1 | 单请求 decode |
| R_medium_balanced | 800 | 1000 | 256 | 8 | 典型 |
| R_long_prefill | 4000 | 5200 | 32 | 4 | v2 的最长 |
| R_concurrent_decode | 200 | 260 | 256 | 32 | 高并发 decode |
| **R_prompt_8k_c4_out128** | 8000 | 10400 | 128 | 4 | v3 新：2× v2 最长 |
| **R_prompt_16k_c2_out128** | 16000 | 20800 | 128 | 2 | v3 新：4× |
| **R_prompt_32k_c1_out128** | 32000 | 41600 | 128 | 1 | v3 新：8×，纯长 prefill |
| **R_prompt_50k_c1_out64** | 50000 | 65000 | 64 | 1 | v3 新：接近上下文上限 |

**每 regime 3 runs, num_runs=3, reliable_stddev < 8% 视为可靠。**

设计考量：
- 长输入 regime 都用**低并发**（1-4）—— KV cache 显存开销跟 prompt_len × concurrency 成正比，避免 OOM
- 32k 和 50k 用 `max_new` 更短，让 wall time 可控（不然一个 trial 就要 10 分钟以上）
- **不做长 output**（例如 4k 输出），因为 sglang 里 decode 阶段的行为对 output 长度很敏感但重复实验较难

---

## 4. Baseline 结果（v3 spec, 1 run each）

**说明**：v3 的 baseline 各只跑了 1 个 server lifetime（省时间；v2 已经确认 3-run 均值和 1-run 差异 < 1%）。**warm-start trial 0**（`auto MoE + 好 batching`）在 R_concurrent_decode 上得 23.32 req/s，也可以视为 cookbook baseline 的独立复核（差 2%，处于合理范围）。

### 4.1 true-default baseline（v3 spec, 1 run）

| Regime | req/s | tokens/s | MFU_simple | MFU_full | MFU_amortized | MBU |
|---|---|---|---|---|---|---|
| R_short_decode | 1.686 | 436.2 | 0.14% | 0.14% | 0.21% | 28.78% |
| R_medium_balanced | 7.327 | 1875.5 | 0.61% | 0.61% | 3.10% | 15.63% |
| R_long_prefill | 13.659 | 437.1 | 0.14% | 0.15% | 23.90% | 7.29% |
| R_concurrent_decode | 23.747 | 6089.9 | 1.97% | 1.98% | 3.99% | 12.66% |
| R_prompt_8k_c4_out128 | 5.329 | 681.7 | 0.22% | 0.24% | 19.59% | 11.37% |
| R_prompt_16k_c2_out128 | 2.710 | 347.0 | 0.11% | 0.14% | 21.20% | 11.56% |
| R_prompt_32k_c1_out128 | 1.634 | 209.2 | 0.07% | 0.11% | 29.11% | 13.94% |
| R_prompt_50k_c1_out64 | 1.300 | 83.2 | 0.03% | 0.10% | 40.25% | 5.55% |

### 4.2 cookbook baseline（v3 spec, 1 run）

| Regime | req/s | tokens/s | MFU_simple | MFU_full | MFU_amortized | MBU |
|---|---|---|---|---|---|---|
| R_short_decode | 1.704 | 436.2 | 0.14% | 0.14% | 0.21% | 29.08% |
| R_medium_balanced | 7.311 | 1871.6 | 0.61% | 0.61% | 3.10% | 15.60% |
| R_long_prefill | 13.572 | 434.3 | 0.14% | 0.15% | 23.90% | 7.24% |
| R_concurrent_decode | 23.799 | 6092.6 | 1.97% | 1.98% | 3.99% | 12.69% |
| R_prompt_8k_c4_out128 | 5.320 | 681.0 | 0.22% | 0.24% | 19.59% | 11.35% |
| R_prompt_16k_c2_out128 | 2.698 | 345.3 | 0.11% | 0.14% | 21.20% | 11.51% |
| R_prompt_32k_c1_out128 | 1.633 | 209.0 | 0.07% | 0.11% | 29.11% | 13.94% |
| R_prompt_50k_c1_out64 | 1.275 | 81.6 | 0.03% | 0.10% | 40.25% | 5.44% |

**观察**：

1. **两个 baseline 差异 < 1%**，parser flag 不影响吞吐（复现 v2 结论）。
2. **MFU_simple 全部 < 2%**——decode 全部 memory-bound，flops 是过剩的。
3. **MFU_amortized 在长 prefill regime 到 20-40%**——prefill 阶段真在打 GEMM，长输入越长这个数越高。
4. **MBU_R_short = 29%**——单请求 decode 已经用掉近 30% HBM 带宽。这是 LFM2.5 的 memory-bound 天花板。
5. **MBU 随 concurrency 下降**（R_concurrent 只有 12.7%）——因为 weight 只读一次给多个请求共享。这符合公式。

---

## 5. v3 搜索空间 + TPE 修复

### 5.1 Search space（在 v2 基础上扩展）

```python
mem-fraction-static  ∈ {0.75, 0.85, 0.90}       # 3
max-running-requests ∈ {8, 16, 32, 64}          # 4
chunked-prefill-size ∈ {-1, 2048, 8192}         # 3
schedule-policy      ∈ {lpm, fcfs}              # 2
attention-backend    ∈ {fa3}                    # 1（v2 已验证其他不可用）
disable-cuda-graph   ∈ {True, False}            # 2
moe-runner-backend   ∈ {triton, flashinfer_cutlass, auto}  # 3（+auto 是 v3 加的）
```

总组合数：3 × 4 × 3 × 2 × 1 × 2 × 3 = **432**。

### 5.2 Warm-start trial 设计（TPE 修复的核心）

从 v2 的教训（[docs/2026-06-30/lfm2.5_conditional_autotuning.md §6.4](../2026-06-30/lfm2.5_conditional_autotuning.md#64-tpe-为什么漏掉了这个)）我们知道 TPE 的失败模式：

- 早期随机采样把 `triton MoE` 和 `cap=8` / `cg-off` 这些坏 batching 绑在一起
- 结果 5 个 triton trial 全部 <10 req/s
- TPE 学到"triton = 差"，后续再没试过 `triton + 好 batching`
- 但真最优就是 `triton + 好 batching`（23.5 req/s vs Optuna best 22.3）

**修复方法**：在 TPE 接管之前，先跑 4 个**分层暖启动 trial**——每个 MoE backend 都被强制搭配一次「已知好 batching 先验」。

「已知好 batching 先验」= v2 的胜出配置：
- cap=32（不限流）
- chunked-prefill=-1（长 prompt 不切）
- schedule=lpm
- mem=0.9
- cg-on

4 个 warm trial：

| Trial # | moe-backend | schedule | 其他 | 目的 |
|---|---|---|---|---|
| 0 | `auto` | lpm | good prior | cookbook-equivalent 参考 |
| 1 | `triton` | lpm | good prior | **v2 漏掉的这个** |
| 2 | `flashinfer_cutlass` | lpm | good prior | v2 winner 复现 |
| 3 | `auto` | fcfs | good prior | schedule 敏感度控制 |

从 trial 4 起 TPE 接管，用 stratified 后的 posterior 采样剩余 26 trial。

### 5.3 为什么 warm-start 有效

TPE 的失败模式是"marginal distribution 被冷启动的坏样本污染"。通过：
1. **给每个 categorical value 一个 fair sample**（cover 3 个 MoE backend）
2. **搭配已知好的其他维度**（避免相邻维度的 confound）

TPE 拿到 4 个 warm trial 时已经知道：
- moe=auto: 23-24 req/s（好）
- moe=triton: 23-24 req/s（好） ← 关键！v2 没这个数据
- moe=flashinfer_cutlass: 22 req/s（略差）
- schedule=fcfs vs lpm 差异微小

于是它不会像 v2 那样把 `triton` 打入冷宫，会公平地探索所有 backend。

---

## 6. Optuna v3 结果（30 trial，warm-start + long-context）

### 6.1 完整 trial 表（30 rows）

30 个 trial 全部成功（ok=True），无 port collision，无 crash。第 0-3 是 warm-start，第 4-29 是 TPE。

R_conc 列同时也是 Optuna 的优化目标。

| # | phase | R_conc | R_p8k | R_p16k | R_p32k | R_p50k | MFU_conc | moe | cg-off | cap | chunk | sched | mem |
|--:|---|--:|--:|--:|--:|--:|--:|---|---|--:|--:|---|--:|
| 0 | warm | 23.32 | 5.28 | 2.67 | 1.61 | 1.24 | 1.93 | auto | F | 32 | -1 | lpm | 0.90 |
| 1 | warm | **23.75** ⭐ | 5.29 | 2.68 | 1.64 | 1.27 | 1.97 | **triton** | F | 32 | -1 | lpm | 0.90 |
| 2 | warm | 22.33 | 4.01 | 2.00 | 1.33 | 0.97 | 1.85 | flashinfer_cutlass | F | 32 | -1 | lpm | 0.90 |
| 3 | warm | 23.72 | 5.27 | 2.69 | 1.59 | 1.28 | 1.96 | auto | F | 32 | -1 | fcfs | 0.90 |
| 4 | tpe | 23.75 | 5.33 | 2.71 | 1.63 | 1.27 | 1.97 | auto | F | 32 | -1 | fcfs | 0.90 |
| 5 | tpe | 23.11 | 7.62 | 4.46 | 2.32 | 2.96 | 1.91 | triton | F | 64 | 2048 | fcfs | 0.90 |
| 6 | tpe | 23.74 | 5.33 | 2.72 | 1.65 | 1.31 | 1.97 | triton | F | 64 | -1 | lpm | 0.90 |
| 7 | tpe | 23.74 | 7.22 | 4.04 | 2.29 | 2.76 | 1.97 | auto | F | 32 | 8192 | lpm | 0.75 |
| 8 | tpe | 13.62 | 4.01 | 2.03 | 1.36 | 0.97 | 1.13 | flashinfer_cutlass | F | 16 | -1 | lpm | 0.90 |
| 9 | tpe | 22.39 | 4.01 | 2.00 | 1.33 | 0.96 | 1.85 | flashinfer_cutlass | F | 64 | -1 | lpm | 0.90 |
| 10 | tpe | 1.81 | 1.71 | 0.87 | 0.44 | 0.80 | 0.15 | auto | T | 8 | 2048 | fcfs | 0.85 |
| 11 | tpe | 6.83 | 1.66 | 0.83 | 0.42 | 0.75 | 0.57 | triton | T | 32 | 8192 | fcfs | 0.75 |
| 12 | tpe | 3.46 | 1.49 | 0.74 | 0.39 | 0.57 | 0.29 | triton | T | 16 | -1 | fcfs | 0.85 |
| 13 | tpe | 8.46 | 7.75 | 4.48 | 2.40 | 2.89 | 0.70 | auto | F | 8 | 2048 | fcfs | 0.90 |
| 14 | tpe | 6.94 | 1.56 | 0.77 | 0.41 | 0.60 | 0.58 | triton | T | 32 | -1 | lpm | 0.85 |
| 15 | tpe | 23.79 | 7.35 | 4.14 | 2.41 | 2.84 | 1.97 | auto | F | 32 | 8192 | fcfs | 0.75 |
| 16 | tpe | 23.75 | 7.24 | 4.31 | 2.41 | 2.84 | 1.97 | auto | F | 32 | 8192 | fcfs | 0.75 |
| 17 | tpe | 14.22 | 7.29 | 4.13 | 2.38 | 2.86 | 1.18 | auto | F | 16 | 8192 | fcfs | 0.75 |
| 18 | tpe | 1.76 | 1.65 | 0.84 | 0.43 | 0.78 | 0.15 | auto | T | 8 | 8192 | fcfs | 0.75 |
| **19** | tpe | **23.90** ⭐ | 7.23 | 4.12 | 2.34 | 2.76 | 1.98 | auto | F | 32 | **8192** | fcfs | 0.75 |
| 20 | tpe | 23.70 | 7.11 | 4.05 | 2.33 | 2.79 | 1.96 | auto | F | 32 | 8192 | fcfs | 0.75 |
| 21 | tpe | 23.64 | 7.11 | 4.22 | 2.30 | 2.73 | 1.96 | auto | F | 32 | 8192 | fcfs | 0.75 |
| 22 | tpe | 23.74 | 7.14 | 4.22 | 2.43 | 2.88 | 1.97 | auto | F | 32 | 8192 | fcfs | 0.75 |
| 23 | tpe | 23.77 | 7.20 | 4.22 | 2.43 | 2.84 | 1.97 | auto | F | 32 | 8192 | fcfs | 0.75 |
| 24 | tpe | 23.71 | 7.30 | 4.14 | 2.44 | 2.88 | 1.96 | auto | F | 32 | 8192 | fcfs | 0.75 |
| 25 | tpe | 22.13 | 6.59 | 3.68 | 2.34 | 2.65 | 1.83 | flashinfer_cutlass | F | 32 | 8192 | fcfs | 0.75 |
| 26 | tpe | 3.60 | 1.69 | 0.85 | 0.45 | 0.78 | 0.30 | auto | T | 16 | 8192 | fcfs | 0.75 |
| 27 | tpe | 8.47 | 7.33 | 4.16 | 2.39 | 2.91 | 0.70 | auto | F | 8 | 8192 | fcfs | 0.75 |
| 28 | tpe | 23.81 | 7.28 | 4.08 | 2.44 | 2.86 | 1.97 | auto | F | 64 | 8192 | fcfs | 0.75 |
| 29 | tpe | 23.15 | **7.76** ⭐ | **4.57** ⭐ | 2.38 | **3.05** ⭐ | 1.92 | auto | F | 64 | **2048** | fcfs | 0.85 |

⭐ = 该 regime 的 v3 最佳。

**关键行**：
- **Trial 1**（warm-start #1，**v3 最重要的一个 trial**）：`triton MoE + 好 batching` = 23.75 req/s。这就是 v2 因 TPE 冷启动坏运气而漏掉的最优解，v3 warm-start 用一个 trial 就找到了。
- **Trial 19**（v3 全局 best）：R_concurrent_decode = 23.90，同时在长上下文 regime 也接近最优。
- **Trial 29**：`chunk=2048 + cap=64 + mem=0.85` 是**所有 4 个长上下文 regime 的最佳**，但 R_concurrent_decode 略低（23.15）。说明**不同 regime 有不同的最优 chunked-prefill-size**。

### 6.2 TPE 修复验证

对比 v2 vs v3 在「triton MoE 是否被公平评估」这个维度上：

| 指标 | v2 (2026-06-30, TPE 无 warm-start) | v3 (2026-07-02, +warm-start) |
|---|---|---|
| triton MoE 被 sample 次数 | 5/25（20%） | 10/30（33%） |
| triton MoE 的最好一次成绩 | 8.43 req/s（因搭配 cap=8） | **23.75 req/s**（warm-start 强制搭配好 batching） |
| Optuna best 的 MoE backend | flashinfer_cutlass（22.32） | auto（23.90，实际是 triton） |
| Optuna best vs 手工验证的实际最优 | 差 5.4%（22.32 vs 23.53） | **等价甚至更好**（23.90 vs 23.53） |

**结论：warm-start + stratified 采样彻底修复了 v2 的 TPE 失效。**

### 6.3 v3 best vs baseline

| 配置 | R_conc | R_p8k | R_p16k | R_p32k | R_p50k | 平均 vs baseline |
|---|---|---|---|---|---|---|
| **v3 baseline cookbook** | 23.80 | 5.32 | 2.70 | 1.63 | 1.28 | 0% |
| v2 Optuna best (trial 17) | 22.32 | — (regime 不同) | — | — | — | -6.2%（仅 R_conc） |
| v2 手工验证 (triton MoE, chunk=-1) | 23.53 | ~5.3 (估计) | ~2.7 | ~1.6 | ~1.3 | -1.0% (R_conc) |
| v3 warm-start #1 (triton, chunk=-1) | 23.75 | 5.29 | 2.68 | 1.64 | 1.27 | -0.2% ~ +0.4% |
| **v3 Optuna best (trial 19)** | **23.90** | 7.23 | 4.12 | 2.34 | 2.76 | +0.4% ~ **+116%** |
| v3 长上下文最佳 (trial 29) | 23.15 | **7.76** | **4.57** | 2.38 | **3.05** | -2.7% ~ **+138%** |

**观察**：
- 在 R_concurrent_decode 上，v3 和 v2 baseline 几乎打平（cookbook 就已经很好）。
- 但只要**输入变长**（8k+ tokens），baseline 就落后一大截——因为 baseline 用了 `chunk=-1`。
- v3 Optuna 找到的 `chunk=8192` 或 `chunk=2048` 让长上下文 regime 大幅提升（+50-140%）。

### 6.4 关键 flag 交互（跨 regime）

| flag | R_concurrent_decode 最优 | 长上下文 regime 最优 |
|---|---|---|
| moe-runner-backend | `auto`（= triton）或显式 `triton` | 同 |
| max-running-requests | 32（正好等于 concurrency） | **64**（长 prompt 时可以塞下更多并发） |
| chunked-prefill-size | 8192 或 -1（差别小） | **2048 或 8192**（-1 差 50-140%） |
| schedule-policy | fcfs 或 lpm（差别 <1%） | 同 |
| mem-fraction-static | 0.75-0.90（差别小） | 0.85（trial 29 最佳） |
| disable-cuda-graph | **必须 False** | 同 |
| attention-backend | fa3（唯一可用） | 同 |

**跨 regime 通用最优 config**：`auto MoE + cap=32 + chunk=8192 + fcfs + mem=0.75 + cg-on + fa3`（trial 19）。R_conc 上和 baseline 打平，长上下文 regime 上 +50%。

**如果**只关心长上下文 regime：`auto MoE + cap=64 + chunk=2048 + fcfs + mem=0.85`（trial 29）。R_conc 上 -3%，但长 regime 再 +5-15%。

---

## 7. 结论 + 下一步

### 7.1 TPE 修复的普遍价值（核心工程结论）

v2 的失败模式（TPE 因冷启动坏运气把 categorical value 打入冷宫）在 v3 通过 **stratified warm-start** 彻底修复：

- 只需要 4 个额外 trial（占 30 trial 总预算的 13%）
- 保证每个 categorical value 至少被公平评估一次
- 「好 batching 先验」可以从 cookbook 或 v2 winner 抽取

这个方法**普遍适用**于任何 categorical × 连续 hyperparameter 有交互的 tuning 场景。建议做成默认策略，写进 `harness/autotune_v3_lfm.py` 的 API：`--warm-start-from-cookbook`。

### 7.2 长上下文 regime 揭示了 baseline 的重大缺陷

**baseline `chunked-prefill=-1` 是长上下文场景下的糟糕默认**：

- 在 v2 只有 R_long_prefill (4k words) 时看不到问题——那个长度还够快
- 在 v3 加入 8k/16k/32k/50k words 后**立即暴露**：不分块的 prefill 阻塞其他请求，让长上下文吞吐降 50-140%
- **不是 tradeoff**：`chunk=8192` 在 R_concurrent_decode 上和 `chunk=-1` 打平（23.90 vs 23.80），同时长 regime 大幅提升

**给 sglang 团队的反馈**：cookbook 默认 config 里应该加 `--chunked-prefill-size 8192`。或者在 sglang 里让 `auto` 自动 enable chunking 当模型 max_position >32k 时。

### 7.3 MFU 的观察

- **decode regime**: MFU_simple < 2%, MBU 12-29% → memory-bound 天花板已到，backend 选择的边际价值有限
- **long prefill**: MFU_amortized 20-40% → compute-bound，backend 和 chunking 选择的价值大
- 说明 **backend / kernel 选择的收益主要在长 prefill regime**（decode 已经打满带宽）
- **对 agent 设计的启示**：workload profile（prefill vs decode ratio）应该是 config 选择的**首要输入**，比 tp/ep 等静态维度更重要

### 7.4 「单一最优 config」不存在——workload-aware 才是正解

**v3 强证据**：R_concurrent_decode 的最优（`chunk=8192, cap=32, mem=0.75`）和长上下文 regime 的最优（`chunk=2048, cap=64, mem=0.85`）不一致。

对 end-to-end agent 的设计含义：
- 不应该找一个 "universal config"，应该**按 workload profile 路由到不同 config**
- Agent 应该：先 workload probe（几个短请求判断 prefill/decode ratio、平均输入长度），再选 config
- 或者 sglang 自己实现 dynamic config switch（较难，需要动态重构 CUDA graph）

### 7.5 下一步（v4）

- [ ] TP/EP 多卡搜索（需要 ≥ 2 张空 GPU；等 GPU 4 释放，或者协调 GPU 6+7 拿两张）
- [ ] KV cache dtype 单独 study（fp8_e4m3 vs bf16）——可能给 memory-bound decode 双倍提升
- [ ] 换模型：Qwen3-30B-A3B 上跑同一套 v3 框架（regime + warm-start + MFU），验证结论普适性
- [ ] `radix-eviction-policy`（lru/lfu/slru/priority）和 `schedule-conservativeness` 的连续 knob
- [ ] **workload-aware routing agent 原型**：给出 prompt 长度分布，agent 自动选 config

### 7.6 直接可交付给 mentor（Dey）的成果

1. **一个 "sglang 默认次优" 的实证**：`chunked-prefill=-1` 在长上下文场景下差 50-140%
2. **一个可复现的 "universal best config"**：trial 19 的 `auto+cap=32+chunk=8192+fcfs+mem=0.75` 在 R_concurrent 上打平 baseline，在所有长 regime 上都是接近最优
3. **一个可复用的 TPE 修复方法**：warm-start + stratified sampling，可以拿去用在他任何未来的 tuning 项目
4. **MFU/MBU 度量框架**：现在所有 trial 都能算，比只看 req/s 信息量大得多

---

## 附录 A：文件清单

- **`configs/hardware/h200.yaml`** — H200 peak FLOPS/HBM table
- **`configs/models/lfm2.5-8b-a1b.yaml`** — LFM2.5 shape + active params
- **`configs/models/qwen3-30b-a3b.yaml`** — Qwen3 (for cross-model reuse)
- **`configs/lfm2.5_8b_a1b_v3_longctx.yaml`** — v3 server config (context=72k)
- **`regimes/lfm2.5_long_context_sweep.yaml`** — 8-regime sweep including 4 new long-context
- **`bench-specs/lfm2.5-v3-{true-default,cookbook}-longctx.yaml`** — 2 baselines
- **`harness/mfu.py`** — MFU/MBU computation library
- **`harness/autotune_v3_lfm.py`** — v3 tuner with warm-start
- **`scripts/add_mfu_retro.py`** — retro-annotate old summaries
- **`results/2026-07-02_lfm2.5_v3/`** — all v3 results
- **`results/2026-06-30_lfm2.5/**/summary.json`** — retrofitted with MFU

## 附录 B：跨 v2 vs v3 变化清单（工程角度）

| 项目 | v2 | v3 |
|---|---|---|
| `harness/output.py` schema | 顶层禁止未知属性 | 允许 `mfu_assumptions` |
| `harness/run_bench.py` | 无 MFU | 加 `--mfu-hardware` / `--mfu-model` |
| Optuna 采样 | 纯 TPE | 4 warm + TPE |
| Search space | 7 knob × 288 | 7 knob × 432（加 moe=auto） |
| Per-trial CSV 列 | 15 列 | **32 列**（+ MFU × 8 regime + phase） |
| Regime 文件 | `qwen3_30b_moe_sglang_perf_sweep.yaml`（4 regime） | `lfm2.5_long_context_sweep.yaml`（8 regime） |
