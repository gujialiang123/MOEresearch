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
（待填）

### v25 — Answer-readiness
（待填）

---

## 综合结论
（待填）
