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
（待填:Part A prefill recovery,Part B decode pulse,open-loop vs closed-loop）

---

## 综合结论
（待填）
