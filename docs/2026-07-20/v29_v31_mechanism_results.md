# v29–v31 机制实验：renorm 因果链 kill-tests（2026-07-20）

> 目标：用受控因果实验判断三个可证伪问题(计划 P1)。仓库 `MOEresearch`,所有 raw data 已入 git。
> 模型 Qwen3-30B-A3B(原生 top-8)| GSM8K | greedy | max_new=512 | GPU 1/4/6。

## 背景与三个 kill-test 问题
昨晚(v20–v28)发现"降 decode K → 变长",主因指向 **renorm 的 per-token 放大**(mode-D 间接推断)。本轮严格检验:
1. **v29**:生成长度是否随 renorm 强度 β **单调**变化?(gain 是否**因果**控制长度)
2. **v30**:影响来自**平均 scale** 还是 **token-conditioned gain**?(shuffled/mean/clipped gain 区分)
3. **v31**:prefill 扰动是否**恢复**、decode 扰动是否被**自回归反馈放大**?(open-loop vs closed-loop)

## P0 基础设施(已完成,14/14 测试通过)
扩展 `moe_research/k_policy.py`:`partial_renorm(β)`(β=0≡no_renorm,β=1≡full_renorm,native-K 所有 β 等价)、`clipped_gain`、`fixed_gain`、`shuffled_gain`。新增 `gain_calibration.py`(只用 train split)、`trace_schema.py`(manifest/config-hash)。测试含:partial_renorm 三等价、gain 无泄漏、intervention window、native-K gain 等价、物理跳过、无同步。

---

## v29 — Partial-renorm 剂量曲线
**prefill=8, decode-only, greedy, n=500。partial_renorm β 连续插值 no_renorm(0)↔full_renorm(1)。Δlen vs k8_native(=251.5, 95%CI)。**

| decode K | β | len | Δ (95%CI) | acc | noMark |
|--:|--:|--:|--:|--:|--:|
| 4 | 0.00 | 255 | +3.7 (−0.7, 8.0) ns | 86.0% | 4.8% |
| 4 | 0.25 | 259 | +7.9 (4.3, 11.6) | 86.2% | 5.6% |
| 4 | 0.50 | 261 | +9.8 (5.2, 14.5) | 86.8% | 5.2% |
| 4 | 0.75 | 271 | +19.4 (14.7, 24.1) | 83.6% | 8.0% |
| 4 | 1.00 | 280 | +28.2 (22.9, 33.6) | 81.8% | 10.0% |
| 6 | 0.00 | 253 | +1.1 ns | 85.8% | 4.8% |
| 6 | 0.25 | 253 | +1.0 ns | 87.0% | 3.6% |
| 6 | 0.50 | 256 | +4.3 (1.1, 7.7) | 86.4% | 5.2% |
| 6 | 0.75 | 260 | +8.4 (4.4, 12.5) | 84.6% | 6.2% |
| 6 | 1.00 | 261 | +9.4 | 85.6% | 6.6% |

**β 单调趋势检验**:
- **decode K4:严格单调递增**(255→259→261→271→280),4/4 相邻对 concordant,β≥0.25 起 CI 全部 >0。
- **decode K6:近单调**(3/4;β0→0.25 的微小波动在噪声内,K6 的 gain 本就小 1/r≈1.05)。

**结论(kill-test #1 通过)**:
1. **renorm 强度 β 因果控制生成长度**:在完全相同的保留 top-K 下,只连续改变对存活专家的放大强度,长度就连续、单调地增长。β=0(no_renorm)几乎无效应,β=1(full)最大。
2. 满足计划的"强机制结果"标准:K4 下长度随 β 单调增(且 noMark/hit_max 同步升),K6 方向一致但更弱,accuracy 与 length 变化不完全同步(K4 β≤0.5 时 acc 不降甚至微升,长度已在涨)。
3. → 可以写:**"renormalization strength causally controls verbosity"**(而非退回到中性的"aggregation semantics alter generation nonlinearly")。

## v30 — Gain controls（平均 scale vs token-conditioned gain）
**固定 decode K4,prefill 8,greedy,n=200。gain 校准只用 GSM8K train(64 题)。Δlen vs k8_native(=252)。**

| 模式 | 定义 | len | Δ | acc | noMark |
|--|--|--:|--:|--:|--:|
| k8_native | 基线 | 252 | 0 | 82.5% | 4.5% |
| no_renorm | g=1(不放大) | 256 | **+3.8** | 87.0% | 3.0% |
| true_token_gain | g=1/r 每 token(=full renorm) | 280 | **+27.7** | 81.0% | 9.0% |
| layer_mean_gain | 冻结每层平均 gain(纯平均 scale) | 276 | **+24.5** | 85.5% | 7.0% |
| shuffled_gain | 从匹配池抽 gain(破坏 token 对应) | 296 | **+43.7** | 76.5% | 15.5% |
| clipped_gain_q90 | g=min(1/r, q90) | 281 | +28.6 | 82.5% | 10.5% |
| clipped_gain_q95 | g=min(1/r, q95) | 281 | +28.9 | 79.5% | 10.5% |

**结论(对昨晚 mode-D 框架的重要修正)**：
1. **gain 因果控制长度**:no_renorm(g=1)只有 +3.8,一旦施加放大就 +25~44。→ 放大(gain)是长度效应的**直接原因**。
2. **主要是平均 scale,不是 token 对应关系**:
   - `layer_mean_gain`(纯每层平均标量,无任何 token 条件)已复现 **+24.5**,占 true(+27.7)的 **88%**。
   - `shuffled_gain`(保留 gain 分布、**打破**与当前 token 的对应)不但没削弱,反而 **+43.7 更大**(且 noMark 15.5% 最高、acc 最低)。→ **token 级对应关系不是关键**;把大 gain 随机砸到"不需要"的 token 上反而更不稳定。
   - 对应计划的"结果 B/C":gain 的**幅度/分布**比其与 router state 的语义对应更重要。
3. **不是极端尾部**:`clipped_gain_q90/q95`(裁掉最大的 gain)几乎不降(+28.6/+28.9≈true)。→ 效应来自 gain 的**主体幅度**,不是少数极大 gain。

**与昨晚 mode-D 的关系(诚实澄清)**:mode-D 的 `calibrated_norm_match` 匹配的是 **branch 输出 norm**(标量≈1.05),给 +7.7;v30 的 `layer_mean_gain` 施加的是**平均 gain E[1/r]**(标量≈1.2-1.6,更大),给 +24.5。二者不矛盾 —— 都指向"平均放大幅度"是主因,只是"匹配 norm 所需的放大"小于"full-renorm 的平均 gain"。综合:**长度效应由 renorm 施加的平均放大幅度驱动,而非 token 级对应、也非极端尾部。** 这比昨晚"per-token 自适应"的表述更准确。

## v31 — Pulse-and-recovery（自回归放大）

**问题(kill-test #3)**:decode 阶段的局部 K 扰动为什么能改变最终长度?是**自回归反馈把一个瞬时扰动放大成永久的轨迹偏移**,还是只是逐步的静态偏差累加?判据:如果同一个扰动在 **open-loop(teacher-forced,喂 K8 参考 token)** 下会被系统"遗忘"(KL 迅速衰减),但在 **closed-loop(自由生成,喂自己的 token)** 下却导致轨迹永久翻转,则证明存在自回归放大。

**方法**:沿 K8 baseline 轨迹注入一个有限时长的 K4 "脉冲",脉冲结束后恢复 K8。
- **Part A(prefill recovery)**:整段 prefill 用 K4,decode 全程 K8 —— 测一个"上游一次性扰动"是否被 decode 消化。
- **Part B(decode pulse)**:decode 中在 early/late 位置注入 dur∈{1,16} 的 K4 脉冲。
  - *open-loop*:脉冲后强制喂 K8 参考 token,测脉冲期内 KL(`open_kl_in`)与脉冲后 8 token 的残余 KL(`recovery_first8`)。
  - *closed-loop*:自由生成,测最终轨迹是否翻转(`closed_flip_frac`)。

**结果**(n=40 problems,Part B 每 duration n=80 pulse 位置):

| | open-loop 脉冲内 KL | open-loop 脉冲后残余(8 tok) | closed-loop 翻转率 |
|---|---|---|---|
| **Part B dur=1** | 0.132 | **0.005**(几乎完全恢复) | **46.3%** |
| **Part B dur=16** | 0.110 | 0.031(部分恢复) | **75.0%** |

Part A(prefill 一次性扰动):open-loop KL 从 first16=0.049 衰减到 last16=0.014(decode 在遗忘上游扰动);但 closed-loop first-divergence 中位数仅 **16.5 token**,97.5% 的样本最终仍然发散。

**结论**:**同一个局部扰动,open-loop 下 8 个 token 内几乎完全恢复(KL→0.005),closed-loop 下却有 46–75% 概率使轨迹永久翻转。** 这是自回归放大的直接证据 —— 是"喂自己生成的 token"这个闭环、而非静态偏差累加,把瞬时 K 扰动放大成长度/终止的永久改变。且脉冲越长(dur 1→16),残余 KL 与翻转率都单调上升(0.005→0.031,46%→75%),符合"扰动越久越难被闭环拉回"。**Kill-test #3 通过。**

---

## 综合结论

三个 kill-test 共同锁定了"降低 K → 生成变长"的完整因果链,并逐一排除了主要的替代解释:

1. **v29(因果性)**:长度效应由 renorm 的**强度 β 单调、因果地控制**(K4:Δ 3.7→28.2,β 0→1,4/4 concordant,CI 全 >0)。不是"专家数量"本身,而是"删除专家后如何重整权重"决定了变长幅度。→ 排除"K 越小信息越少所以更啰嗦"的朴素解释。

2. **v30(机制定位)**:效应来自 renorm 施加的**平均放大幅度**(layer_mean_gain 复现 88%,shuffled 反而更大 +43.7,clip 尾部无削减),**不是** token 级 gain 对应、**也不是**极端尾部专家的贡献。→ 排除"被删专家携带特定 token 信息"和"尾部重要专家"两种解释。修正了昨晚 mode-D "per-token 自适应"的表述。

3. **v31(传播机制)**:局部扰动之所以能改变**最终**长度,是因为**自回归闭环放大**:open-loop 8 token 内恢复(KL 0.005),closed-loop 46–75% 永久翻转。→ 排除"逐步静态偏差线性累加"的解释,确立"闭环反馈"为放大通道。

**一句话论文主张**:降低 MoE 活跃专家数通过 renormalization 对 survivor 权重施加一个**平均上放大**,这个 residual-scale 扰动在**自回归解码闭环**中被放大,系统性地推迟终止、延长生成 —— 因此这是一个 **scale-preserving / renorm-mediated 的生成稳定性效应**,而非"专家—token 替代"式的语义信息损失。

**下一步(不阻塞初稿)**:v25 answer-readiness(区分"更晚知道答案" vs "只是更啰嗦")、v26 fixed-KV direct-effect、v28 完整剂量曲线的 confirmatory test-split 运行、以及第二个 MoE 架构(Phi-3.5-MoE / Qwen1.5-MoE)方向复现。这些属于增强项;P0+v29+v30+v31 已构成最小 kill-test 包并全部通过。
