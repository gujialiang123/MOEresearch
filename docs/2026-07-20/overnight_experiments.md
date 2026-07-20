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
| **v23** phase factorial | 分离 prefill vs decode 对长度的贡献 | 7 configs (pk×dk ∈ 8/6/4), n=500 test | GPU4 | 运行中 |
| **v28** decode dose | decode K 剂量曲线 + 临界点 | pk=8, dk∈{8,7,6,5,4}, n=500 | GPU5 | 运行中 |
| **v24** weight ablation | 排除 renorm/residual-scale 假象 | (8,8)(8,6)(8,4)(6,8)(4,8) × {no_renorm, fold_top1}, n=200 | GPU6 | 运行中 |
| **v26** direct-effect | 真·当前步直接效应（改进 v22） | fixed-K8 KV fork, K∈{8,6,4}, n=60 | GPU7 | 运行中 |
| **v25** answer-readiness | t_ready vs t_marker vs t_eos | 复用 v23 轨迹, n=80 | 待 v23 完成 | 待运行 |

---

## 结果

### v23 — Prefill K × Decode K 因子实验
（待填）

### v28 — Decode K 剂量曲线
（待填）

### v24 — Weight-mode 消融
（待填）

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
（待填）
