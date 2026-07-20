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
（待填:decode K∈{6,4} × β∈{0,.25,.5,.75,1},n=500）

## v30 — Gain controls（平均 scale vs token-conditioned gain）
（待填:decode K4,7 种 gain 模式,calibration 只用 train）

## v31 — Pulse-and-recovery（自回归放大）
（待填:Part A prefill recovery,Part B decode pulse,open-loop vs closed-loop）

---

## 综合结论
（待填）
