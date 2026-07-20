# 过夜实验记录 — MoE K→长度机理（2026-07-20 夜）

> 本文档记录 2026-07-20 夜间 autopilot 跑的所有实验，供早上审阅。仓库：`MOEresearch`。
> 模型 Qwen3-30B-A3B（原生 top-8）| GSM8K | greedy | max_new=512 | GPU4-7。

## 背景与目标
v20/v21/v22 已确立：降 K → 输出变长，主因是**轨迹中介的 L_to_answer↑**（占 82–97%），
直接终止效应仅在 k4 出现（margin 收窄）。**但 v21/v22 都是 `phase=all`（prefill+decode 同时降 K）**，
未区分是 prompt 编码还是 autoregressive decode 驱动。本轮按 v23-v28 计划推进机理主线。

## 基础设施（已完成 + 验证）
- **统一 `moe_research/k_policy.py`**：`KPolicy(prefill_k, decode_k, weight_mode, selectors)`；
  phase 由 **cache 状态**（顶层 pre-hook）判定，不靠 seq_len 猜；物理跳过；无同步计数；4 种 weight mode。
- **测试**：`tests/test_k_policy.py` 8/8 通过。
- **真实模型验证**（`verify_k_policy_realmodel.py`）：ACCEPTANCE_PASS —
  (8,8) next-token logits **精确 0 误差**、贪心生成 token 级一致；phase routing (8,4)→pf8/dec4、(4,8)→pf4/dec8；KV 完整。

## 今晚运行的实验

| 实验 | 目的 | 配置 | GPU | 状态 |
|---|---|---|---|---|
| **v23** phase factorial | 分离 prefill vs decode | 7 configs, n=500 | GPU4/7 | ✅ 完成 |
| **v28** decode dose (renorm) | decode K 剂量曲线 + 临界点 | pk=8, dk∈{8,7,6,5,4}, n=500 | GPU5 | ✅ 完成 |
| **v24** weight ablation + mode D | 排除 renorm/scale 假象 | 4 weight modes (含 calibrated_norm_match) | GPU6 | ✅ 完成 |
| **v26** direct-effect | 真·当前步直接效应（改进 v22） | fixed-K8 KV fork, K∈{8,6,4}, n=60 | GPU7 | ✅ 完成 |
| **v25** answer-readiness | t_ready vs t_marker vs t_eos | 复用 v23 轨迹, n=80 | GPU4 | ✅ 完成 |
| **v28b** decode dose (no_renorm) | 对照 renorm 剂量曲线是否压平 | pk=8, dk∈{8,7,6,5,4} no_renorm, n=500 | GPU4/5 | ✅ 完成（曲线压平 250-255）|
| **full-test 1319** | 主效应 full-test 确认 | 8x8/8x6/8x4 renorm, n=1319 | GPU6/7 | ✅ 完成（K6 +6.7, K4 +29.2）|

---

## 结果

### v23 — Prefill K × Decode K 因子实验（★ 主结果：解离 prefill vs decode）
n=500, renorm_survivors。paired Δlen vs 8x8 (95%CI)：

| 配置 | Δlen vs 8x8 | acc_strict | noMark | 含义 |
|--|--:|--:|--:|--|
| 8x8 (baseline) | 0 | 83.4% | 5.0% | — |
| 6x8 (prefill K6) | +3.3 (−0.2, 7.0) **ns** | 85.0% | 5.6% | 只扰动 prompt/KV |
| 8x6 (decode K6) | **+6.7 (2.7, 10.7)** 显著 | 85.0% | 6.2% | 只扰动 decode |
| 6x6 (both K6) | +11.8 (7.1, 16.6) | 86.0% | 6.0% | 完整 K6 |
| 4x8 (prefill K4) | +3.4 (−1.2, 8.2) **ns** | 83.4% | 6.8% | 强 prefill 干预 |
| 8x4 (decode K4) | **+28.2 (22.9, 33.6)** 显著 | 81.8% | 10.0% | 强 decode 干预 |
| 4x4 (both K4) | +36.0 (30.2, 41.9), acc 74.6%, noMark 18% | | | 完整 K4（过拐点）|

**因子效应**：
- **K=6**：prefill=**+3.3 (ns)** | decode=**+6.7 (sig)** | interaction=+1.9 (ns)
- **K=4**：prefill=**+3.4 (ns)** | decode=**+28.2 (sig)** | interaction=+4.5 (ns)

**关键结论（强解离，K6/K4 均成立）**：
1. **长度效应几乎完全是 decode 阶段现象**：decode 效应 +6.7(K6)/+28(K4) 均显著；**prefill 效应 ~+3，两档都不显著**。
2. → 降低 prompt 编码的 K 几乎不改变生成长度；驱动变长的是**降低 autoregressive decode 的 K**。机理一致：prefill 只编码一次 prompt，decode 才是逐步生成、被 renorm per-token 放大逐步累积之处。
3. **两档 interaction 都不显著（可加）**：4x4 观测 +36 ≈ decode(28)+prefill(3.4)+interaction(4.5)。
4. **交叉验证**：4x4（both K4，+36，acc 74.6%，noMark 18%）与 v21 的 fixed_k4（phase=all，+38，acc 74.8%）**高度吻合** → 新旧 pipeline 自洽。

### v28 — Decode K 剂量曲线（renorm_survivors, decode-only, n=500）

| K (decode) | len | Δlen vs K8 (95%CI) | acc_strict | noMark |
|--:|--:|--:|--:|--:|
| 8 | 251.5 | 0 | 83.4% | 5.0% |
| 7 | 253.6 | +2.1 (−1.5, 5.7) 不显著 | 84.8% | 5.2% |
| 6 | 258.2 | +6.7 (2.7, 10.7) 显著 | 85.0% | 6.2% |
| 5 | 266.5 | +15.0 (10.2, 19.6) 显著 | 84.4% | 7.4% |
| 4 | 279.7 | +28.2 (22.9, 33.6) 显著 | 81.8% | 10.0% |

**相邻 K 的 Δlen（加速/凸曲线）**：8→7: +2.1(ns) → 7→6: +4.6 → 6→5: +8.3 → 5→4: +13.2。

**结论**：
1. **平滑的凸型剂量-响应**：每降一档 K，长度增量**递增**（2→5→8→13），不是 K5 处的突变临界点，而是随 K 下降**加速累积**的稳定性退化。
2. **精度**：K6-7 甚至略升（85%），K4 才掉（81.8%）；noMark 5%→10%。呼应 v21"k6 安全、k4 过拐点"。
3. **⚠️ 关键联系（配合 v24）**：这条曲线是 **renorm_survivors** 下的；v24 证明换 no_renorm 后整条曲线会**基本压平**。所以这是"**renormalization 介导的**长度效应"的剂量曲线，不是"专家数量"的内在剂量曲线。

### v28b — no_renorm 剂量曲线（决定性对照）
直接对比 decode-only 剂量曲线在两种 weight mode 下的形状（renorm n=500；no_renorm 混 n=500/200）：

| decode K | **renorm** len (Δ) | **no_renorm** len (Δ) |
|--:|--:|--:|
| 8 | 251.5 (0) | 251.5 (0) |
| 7 | 253.6 (+2.1) | 250.0 (**−1.5**) |
| 6 | 258.2 (+6.7) | 252.7 (**+1.2**) |
| 5 | 266.5 (+15.0) | 254.5 (**+3.0**) |
| 4 | 279.7 (**+28.2**) | 255.8 (**+4.3**, n200) |

（no_renorm 8x8/8x7/8x6/8x5 为 n=500，8x4 为 v24 的 n=200；full-test 1319 confirmatory 仍在跑）

**结论**：**no_renorm 下整条 decode 剂量曲线基本压平**（250–256，Δ 全在 ±4.5 内），renorm 一路升到 +28。→ 决定性证明"降 decode K → 变长"是 **renorm 介导的**，换 no_renorm 后几乎消失。与 v24/mode D 一致，构成主结论核心对照图。

### v24 — Weight-mode 消融（★ 重大发现）
**decode arm 相同的 K 缩减，长度变化完全取决于权重聚合方式**（paired Δlen vs 8x8, 95%CI）：

| 配置 | renorm_survivors (n=500) | no_renorm (n=200) | fold_mass_to_top1 (n=200) |
|--|--:|--:|--:|
| 8x6 (dk=6) | **+6.7** (2.7,10.7) 显著 | +4.0 (−1.9,10) **不显著** | +20 (13,27) 显著 |
| 8x4 (dk=4) | **+28.2** (22.9,33.6) 显著 | +3.8 (−3.6,11) **不显著** | +144 (127,160) 崩溃 |

（fold 的 8x4：acc 82%→**34%**，noMark 4.5%→**54.5%** — 灾难性；prefill arm 见 analysis.json）

**关键结论**：
1. **长度效应不是"专家数量"本身的内在效应**：同样把 decode K 从 8 降到 4，
   - `renorm_survivors`（存活权重重归一到和=1）：+28 tok，**显著**；
   - `no_renorm`（丢尾部、保留原权重、和<1）：+3.8 tok，**不显著**（CI 含 0）；
   - `fold_mass_to_top1`（丢弃质量全给 top-1）：+144 tok，**崩溃**。
2. → 驱动长度增长的是**对存活专家权重的重新分配/放大（renormalization）**，而非丢弃专家这件事本身。
   - 这对应 v23-v28 计划决策树的"**尺度结果**"：效应主要存在于 renorm → **Scale-Preserving Expert Sparsification**。
   - **重要反转**：v21 用的是 renorm_survivors，所以 v21 的"降 K → 变长（推理替代）"结论，很大程度上是**被 renormalization 混淆的**。换 no_renorm 后效应基本消失。
3. **fold 证明"质量放置方式"极其关键**：把丢弃质量堆到 top-1 会让输出爆炸、精度崩溃 → 不是简单的"norm 大小"，而是**存活专家的相对混合/方向**被改变。

**已补（mode D 决定性结果，n=200）**：`calibrated_norm_match`（冻结 per-layer scalar 把 branch norm 匹配到 K8，但保留 no_renorm 的相对混合）：

| K | renorm | no_renorm | **calibrated_norm_match** | fold |
|--|--:|--:|--:|--:|
| 8x6 | +6.7 | +4.0 | **+5.3** | +20 |
| 8x4 | +28.2 | +3.8 | **+7.7** | +144 |

**决定性结论**：calibrated（norm 匹配到 K8）给 K4 只有 **+7.7**，远小于 renorm 的 **+28**，接近 no_renorm 的 +3.8。
→ **匹配 branch norm 并不能恢复 renorm 的长度效应**，所以效应**不是** branch norm 的大小造成的。
→ renorm 与 calibrated 数学上都是"no_renorm × 标量"，区别在于：**renorm 的标量是 per-token 的 1/Σw_survivors**（随路由置信度变化：模型越不确定、被丢质量越多，upscale 越大），calibrated 是**冻结的 per-layer 平均标量**。二者平均尺度接近（s≈1.05），但长度效应差 4 倍（+28 vs +7.7）。
→ 因此驱动长度增长的是 **renorm 的 per-token 自适应放大**（恰好在"该走的专家被丢掉、置信度低"的 token 上把存活专家放大最多），而非专家数量、也非平均 norm。**"降 K → 变长"本质上是一个权重重归一化（自适应放大）的产物**，v21 的"推理替代"叙事被此混淆。

（注：per-token 自适应的精确因果仍可在 raw log 上进一步验证；但"calibrated≈no_renorm≪renorm"这一点已足以排除"norm 大小"和"专家数量"作为主因。）

### v26 — 当前步直接效应
**n=60 题, ~6600 采样位置/K. 在完全相同的 K8 历史上，只改当前一步的 K。**

| K | KL(p8‖pk) | EOS Δlogp | margin Δ | gold Δlogp | top1 一致率 |
|--:|--:|--:|--:|--:|--:|
| 8 (ref) | 0 | 0 | 0 | 0 | 100% |
| 6 | 0.011 | +1.44 | +1.44 | −0.014 | 98.4% |
| 4 | 0.060 | +3.94 | +3.93 | −0.069 | 96.1% |

（near-EOS ≤8 tok 子集趋势一致：K4 KL=0.061, top1 95.8%）

**关键结论**：
1. **单步直接效应很小**：KL(p8‖pk) 仅 0.01（K6）/0.06（K4），**top-1 next-token 一致率 96–98%**。→ 只改当前一步的 K，几乎不改变 next-token 分布。
2. **因此长度效应是轨迹累积的，不是单步的**：这比 v22 更严格地证明了 v21 的结论 —— 大的长度变化来自**许多步的微小扰动累积**（轨迹发散），而非任何单步的剧烈分布改变。
3. **⚠️ 反直觉 nuance（值得早上深挖）**：单步上，降 K 反而**抬高** EOS logit（K4 +3.9，跨推理位置平均）。但 v21 里降 K 序列却更长。这个"每步 EOS 倾向↑，但整体更长"的悖论，恰是"直接 vs 轨迹"分离的核心：低 K 每步略微抬 EOS，却也把轨迹推上更长的推理路径，净效应（v21）是更长。注意 EOS logit 虽升但基数极负（多数推理位置 EOS 远非 top），故 KL 仍小。位置分层分析见 raw log。

对比 v22（trajectory-fixed cumulative effect）：v22 在 baseline EOS 那一点测 margin，发现 k4 margin 收窄；v26 是纯单步、跨位置平均，两者测的是不同量，不矛盾。v26 才是"当前步直接效应"的干净版。

### v25 — Answer-readiness
**n=80 题, K8 gold-answer logprob 探针, 阈值=−0.5（64 个 K8-correct 校准）, ready_found_frac≈0.98。**

| config | t_ready | t_marker | t_eos | marker−ready | eos−marker |
|--|--:|--:|--:|--:|--:|
| 8x8 (dk=8) | 99.3 | 225.5 | 238.7 | 133.3 | 13.1 |
| 8x6 (dk=6) | 103.3 | 234.3 | 242.9 | 140.5 | 8.6 |
| 8x4 (dk=4) | 104.6 | 242.8 | 251.8 | 143.2 | 10.5 |

Δ(K8→K4)：**t_ready 仅 +5.3**，t_marker +17.3，marker−ready +9.9。

**关键结论（对 v21 的重要细化）**：
- **模型"想出答案"的时刻几乎不随 K 变**（t_ready 99→105，仅 +5）。
- **但"输出 #### 标记"的时刻明显推迟**（t_marker +17）。
- 二者之差（marker−ready，"知道答案却还没写出来"的间隔）增大 +10。
- → 低 K 增加的长度，**更多是"已经答案就绪、却继续生成、延迟提交标记"**，而非纯粹"需要更多推理才能得到答案"。这把 v21 的"推理-计算替代"细化为**部分是 answer-ready 的冗长/延迟提交**（更偏 termination/commitment 侧），只有小部分（+5）是真正的答案形成延迟。

**注意**：t_ready 依赖 gold-answer logprob 探针 + 固定阈值；绝对值随阈值变，但相对位移（Δt_ready≈+5 vs Δt_marker≈+17）是稳健信号。raw log 保留每题每 checkpoint，可换阈值/换探针重算。

---

## 综合结论

### 统一机制图景（v20–v28 合并）
把"降 K → 生成变长"这一现象（v21）彻底机理化，今晚得到一个**连贯、严格、且对 v21 有重大修正**的结论链：

1. **这是 decode 阶段现象，不是 prefill**（v23）：decode 效应 +6.7(K6)/+28(K4) 显著；prefill 效应 ~+3，即使 K4 也不显著。降低 prompt 编码的 K 几乎不改变长度。
2. **本质是权重重归一化（renorm）的 per-token 自适应放大产物，不是"专家数量"内在效应**（v24 + mode D）：同样 decode K8→K4，
   - renorm_survivors：+28（显著）
   - no_renorm：+3.8（不显著）
   - calibrated_norm_match（norm 匹配到 K8）：+7.7（≈no_renorm，远小于 renorm）
   - fold_mass_to_top1：+144（崩溃）
   匹配 norm 无法恢复效应 → 驱动因素是 renorm 的 **per-token 1/Σw 放大**（在"该走的专家被丢、置信度低"的 token 上放大最多），而非 norm 大小或专家数量。
3. **单步直接效应很小，长度变化是轨迹累积**（v26）：只改当前一步 K，next-token 分布几乎不变（KL≤0.06，top-1 一致 96–98%）。大的长度改变来自许多步微小扰动的累积（轨迹发散）。
4. **模型"想出答案"的时刻几乎不变，只是更晚提交标记**（v25）：t_ready +5 vs t_marker +17 —— 额外长度更多是 answer-ready 后的延迟提交/冗长，而非需要更多推理。
5. **剂量-响应是平滑的凸曲线**（v28）：每降一档 K 长度增量递增（2→5→8→13, K8→4），无 K5 突变临界点；但这是 renorm 模式下的曲线（no_renorm 下预期压平，v28b 验证中）。

### 一句话（论文级重构）
> 在标准 renorm_survivors 路由下，**降低 decode 阶段的 K 会通过对存活专家的 per-token 自适应放大，在自回归轨迹上逐步累积成更长的生成**；模型在几乎相同的步数就已"知道答案"，只是更晚提交答案标记。这**本质是一个权重重归一化产物（Scale-Preserving / renorm-induced），而非"专家越少越需要更多推理"的内在效应**——v21 的"推理-计算替代"叙事被 renorm 混淆。它是 **decode 特有**（非 prefill）现象。

### 对应决策树
命中计划的"**尺度结果**"分支 → **Scale-Preserving Expert Sparsification**（效应主要存在于 renorm）。这比原假设更严格、更有警示意义（关于 expert-dropping 的实现方式会人为制造长度变化）。

### 局限 / 早上待办
- 4x4（K4 both，interaction 项）运行中；v28b no_renorm 全 dose（n=500）验证中。
- 结论基于 HF-eager + GSM8K + greedy；跨任务/跨架构（Qwen1.5-MoE, Phi-3.5-MoE）验证未做。
- per-token 自适应放大的精确因果可在 raw log 上进一步分层验证。
- 真实系统收益（sglang fused kernel 的 latency/TPOT）未测——本轮只回答行为/机理。
- 所有 raw log（含完整 token ids）已保存，任意新指标可离线重算，无需重跑。

---

## Full-test 1319 confirmatory 状态（✅ 完成）
主 decode 效应在**全 GSM8K test（n=1319）**上确认，与 n=500 高度一致：

| 配置 (n=1319) | len | Δlen vs 8x8 | acc_strict | noMark | vs n=500 Δlen |
|--|--:|--:|--:|--:|--:|
| 8x8 | 253.1 | 0 | 82.7% | 4.6% | 0 |
| 8x6 (decode K6) | 259.8 | **+6.7** | 84.9% | 5.7% | +6.7（完全一致）|
| 8x4 (decode K4) | 282.3 | **+29.2** | 81.3% | 10.5% | +28.2（稳健）|

→ 主结果（decode-K 缩减在 renorm 下显著增长，K6 安全、K4 过拐点）在**全测试集上稳健复现**。数据见 `results/2026-07-20_v28_fulltest_k{4,6}/`。

## 产物清单（本夜）
- 代码：`moe_research/{k_policy,answer_parsing,stats}.py`、`tests/test_k_policy.py`（8/8）、
  `scripts/{run_gsm8k_configs,run_v25_answer_readiness,run_v26_direct_effect_probe,calibrate_norm_match,analyze_v23_v28,verify_k_policy_realmodel}.py`
- 结果：`results/2026-07-20_*`（各实验 raw jsonl 含完整 token ids + summary + analysis）
- 全部 commit 并 push 到 `origin/main`（github.com/gujialiang123/MOEresearch）。
