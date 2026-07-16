# v18：动态 top-k（置信度自适应）vs 固定 top-k — GSM8K 精度 × 平均激活专家

> ## ⚠️ ERRATUM（2026-07-16，勘误）
> v18 的实现与测量存在问题，**其速度/E2E 结论不能作为 dynamic-K 加速证据**：
> 1. **没有物理跳过被丢弃的 expert**：旧 forward 只把 routing weight 置零（`rw = rw * keep`），但 `expert_mask` 仍由完整 `selected_experts` 构造，被丢弃的 assignment 仍进入 `expert_layer(...)`，只是结果乘零。因此旧 `avg_k` 是**逻辑 K，不是实际执行 K**。
> 2. **每层 `.item()` 污染 timing**：`self._k_sum += int(keep.sum().item())` 在每层每 step 做 D2H 同步，使 wall/tok-s 不可信。
> 3. 因此本文档的 **wall_s / tok_per_s 列不能解释为加速**（HF-eager 逻辑置零，非真实少算，且 timing 被同步污染）。
> **质量结论**（准确率 vs avg_k）仍可视为"数学上的 zero-weight pruning"探索，但仍受答案解析（旧用 last-number，见 P0.5）、样本量（200 题，±3-4% 噪声）与生成长度影响，属**待验证现象**而非确定规律。
>
> **已修正版本**见 `scripts/dynamic_topk_utils.py`（物理跳过 + 无同步计数 + strict 解析）、`tests/test_dynamic_topk.py`（9/9 通过）、`scripts/run_v20_dynamic_topk_equivalence.py`（真实模型 keep-all 精确 0 误差）、`docs/2026-07-16/v20_dynamic_topk_validation.md`。

**日期**：2026-07-15
**目的**：验证"按 router 置信度自适应决定每 token 用几个专家"（top-p 式）能否在**更低的平均 k** 下达到与固定 top-k **同等的精度** —— 这是"动态 topk 值不值得做 / 能不能撑一篇 paper"的核心实验。
**脚本**：`scripts/run_v18_dynamic_topk.py`
**数据**：`results/2026-07-15_v18_dynamic_topk/dynamic_vs_tau.json`
**对照基线**：v17 固定 top-k（同模型/数据/解码设置）

## 方法
- 模型 `Qwen3-30B-A3B-Instruct-2507`（E=128，原生 top-8，48 层 MoE）；GSM8K test 前 200 题，贪心解码，`max_new=400`，batch=64，GPU4。
- **动态路由**（monkeypatch `Qwen3MoeSparseMoeBlock.forward`，约 30 行）：softmax 后取 top-8 候选池，按**归一化累积概率**从高到低保留专家，累积达到阈值 τ 即停（top-p 式）；`k_min=1`（至少 1 个）、`k_max=8`（至多 8）。丢弃专家权重置 0（等价于不选）。
- 每层每 token 记录**实际 k**，聚合成 `avg_k`。扫 τ ∈ {0.60,0.70,0.80,0.88,0.95}。

## 结果

| τ | 实测 avg_k | GSM8K acc | 平均生成 tok |
|--:|--:|--:|--:|
| 0.60 | 3.85 | 73.0% | 273.7 |
| 0.70 | 4.72 | 80.0% | 265.2 |
| 0.80 | 5.74 | 80.5% | 256.2 |
| 0.88 | 6.72 | 82.5% | 249.9 |
| 0.95 | 7.85 | 81.5% | 247.9 |

## 与固定 top-k（v17）对照 —— 核心结论

| 方法 | avg_k | GSM8K acc |
|---|--:|--:|
| 动态 τ=0.70 | **4.72** | 80.0% |
| 固定 top-5 | 5.00 | 80.0% |
| 动态 τ=0.80 | **5.74** | 80.5% |
| 固定 top-6 | 6.00 | 83.0% |
| 动态 τ=0.88 | 6.72 | 82.5% |
| 固定 top-7 | 7.00 | 83.0% |
| 动态 τ=0.95 | 7.85 | 81.5% |
| 固定 top-8（base） | 8.00 | 83.5% |

**诚实的结论（混合信号）：**
1. **低档动态有优势**：动态 τ=0.70 用 **avg_k=4.72** 达到 80.0%，与固定 top-5（k=5.0）**同精度但平均 k 更低**（省 ~0.3 专家）。方向验证成立 —— 难 token 保留更多专家、易 token 用更少确实有效。
2. **中档动态不占优**：动态 τ=0.80（k=5.74，80.5%）明显低于固定 top-6（k=6.0，83.0%）；即在 k≈6 这档，固定反而更好。τ=0.95 甚至（k=7.85，81.5%）低于 base top-8（83.5%），说明高 τ 时的归一化/掩码实现引入了轻微扰动。
3. **优势幅度有限**：本设置下动态相对固定的"等精度省 k"收益很小（~0.3 专家），不足以单独支撑强 paper 结论。

## Novelty 核查（联网检索）
- **已有工作**：Huang et al., *Harder Tasks Need More Experts: Dynamic Routing in MoE Models*（**ACL 2024**, arXiv:2403.07652, 有开源码）——**正是** top-p 式置信度自适应路由，声称比 top-2 提升 0.7%、激活参数 <90%。但它是**从头训练**一个用 top-p 路由的 MoE 模型（training-time 方法）。
- **我们的差异点（潜在 novelty）**：我们做的是**免训练、推理时**在**已训练好的固定 top-8 模型**上直接施加动态路由 —— 这是 post-hoc 干预，无需重训。这个"training-free 动态 topk 在预训练 MoE 上的精度-搬运权衡"角度，与 Huang 的训练时方法不同。
- **但**：本实验结果显示，免训练动态相对免训练固定 top-k 的**额外收益很小**（模型是按固定 top-8 训练的，路由分布未针对动态优化）。要成为强结果，需要：(a) 与"真实搬运节省/端到端加速"挂钩（非仅逻辑 k）；(b) 可能需要轻量校准/微调让动态发挥。

## 对项目主线的意义
- **作为独立 paper**：单靠"免训练动态 topk"目前证据不足（收益小、已有训练时工作）。
- **作为 decode 搬运优化的一环**：仍有价值 —— 它是"move less weight"方向的一种实现。更有前景的组合是**批内聚集（劝导到已激活专家）** + **动态 k**，并在 **sglang 端量真实搬运/latency**（而非 transformers 逻辑层）。
- **下一步**：(1) 在 sglang `custom_routing_function` 落地固定 top-6（安全档）测真实 decode latency；(2) 若继续动态方向，先做轻量校准，且必须报告真实搬运字节（用 v19b 的 `dram__bytes_read.sum`）而非仅 avg_k。

## 局限
- limit=200 有 ±3-4% 噪声，top-5~top-8 之间的小差异部分在噪声内。
- transformers eager，`avg_k` 是逻辑激活数，**不等于**真实搬运节省（需 sglang 侧确认）。
- 仅 GSM8K；未测 HellaSwag/其它任务的动态-vs-固定对照。
