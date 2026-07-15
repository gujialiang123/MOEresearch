# v14：批内 expert 聚集的搬运节省 vs 精度代价（模拟）

**日期**：2026-07-15
**执行**：GPU 4，HF transformers（Qwen3-30B-A3B router 输出）
**目的**：把用户 idea 从"有空间"（v13）推进到"值不值"——量化"批内把边缘 token 劝导到已激活专家"能省多少搬运、代价多大。
**脚本**：`scripts/run_v14_consolidation.py`（固定大批）、`scripts/run_v14b_consolidation_batch.py`（随 batch 扫描）

---

## 方法
对真实 agent prompt 前向，抓每 token 每层的 topk 专家 + 归一化权重。模拟"聚集"：每个 token 保留 rank<r 的核心专家，把 rank≥r 的**边缘专家劝导到该 (batch, layer) 已激活的核心专家**。测：
- **transfer_saved**：(激活专家数_before − after) / before = 搬运次数节省比例。
- **weight_redirected**：被改动的路由权重占比 = 精度代价的代理指标（动的权重越少，精度影响越小）。

---

## 结果一（v14）：固定大批（189–250 token）—— 聚集空间被压缩
| 劝导阈值 | 搬运节省 | 权重改动 |
|---|---|---|
| rank≥7（换最弱1个） | 3.8% | 7.2% |
| rank≥6（换最弱3个） | 8.3% | 15.1% |
| rank≥5 | 13.0% | 23.9% |
| rank≥4 | 19.5% | 34.0% |

**关键洞察**：大批（250 token）时专家已近全覆盖（102/128 激活），聚集能省的有限——因为本来就要搬大部分专家。**但真实 decode 并发才 6–20，专家覆盖不满，聚集空间应更大** → 见 v14b。

## 结果二（v14b）：★ 聚集收益随 batch size（劝导 rank≥6，换最弱 3 个）
| batch | 平均激活专家/层 | **搬运节省** | 权重改动 |
|---|---|---|---|
| 4 | 27 | **22.4%** | 14.9% |
| 8 | 44 | **19.7%** | 15.0% |
| 16 | 63 | **16.9%** | 14.8% |
| 32 | 81 | 13.3% | 14.9% |
| 64 | 96 | 10.3% | 14.8% |

**规律**：**batch 越小，聚集省得越多**（4→64：22.4%→10.3%），而**精度代价（权重改动）几乎不变（~15%）**。

---

## 结论：idea 在真实场景下有实际收益

1. **真实 agent decode 并发 6–20**（v7/v9d 实测）→ 落在 batch=4–16 → **搬运可省 17–22%**。
2. **代价是改动 ~15% 的路由权重**——但动的是每 token **最弱的 rank6-8 专家**（v13 测得 rank8 只贡献 7% 权重、router top1 置信度仅 8.9%）→ **实际精度损失应远小于 15%**（权重改动 ≠ 精度损失，因为动的是最不重要、router 最不确定的选择）。
3. **为什么小 batch 收益大**：小 batch 专家覆盖不满（batch=4 只激活 27/128），fringe 专家很多是"只被一个 token 选的冷门专家"，劝导掉直接省一次搬运；大 batch 专家近全覆盖，省无可省。**这与"decode 真实并发低"的痛点完美契合**——正是搬运摊销最差、最该优化的场景。

**一句话**：批内 expert 聚集在真实 agent decode（batch 6–20）上能省 **~17–22% 的专家搬运**，代价是改动最弱的 15% 路由权重（真实精度损失预计更小）。这是一个**针对 low-concurrency MoE decode 的、有量化依据的新优化方向**。

---

## 局限 & 下一步（真正落地前）
1. **精度代价目前是代理指标**（权重改动比例），**未实测 perplexity/下游精度**。下一步：真正执行聚集后的前向，测 next-token loss / perplexity 的实际变化，把"权重改动 15%"换成"精度掉 X%"。
2. router 分布抓的是 prefill 前向；decode 逐 token 的分布可能略不同。
3. 只用了 6 段 agent prompt（339 token）；扩到真实 toolagent 大样本更稳。
4. 聚集的**实现开销**（重路由逻辑、kernel 端如何利用减少的专家数）未评估——理论节省要落到真实 kernel 才算数。

## 产物
- `results/2026-07-15_v14_consolidation/consolidation_tradeoff.json`
- `results/2026-07-15_v14b_consolidation_batch/consolidation_vs_batch.json`
- `scripts/run_v14_consolidation.py`、`scripts/run_v14b_consolidation_batch.py`
