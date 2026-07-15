# MoE Routing 优化研究现状调研（回应"router 是否最优 + 改 route 换效率"）

**目的**：调研两个方向的真实学术工作——(1) router 不是最优、甚至不按 router 走能更好的证据；(2) 通过改变 route 排布、以损失部分性能换推理效率的工作。评估用户 idea 的新颖性。
**日期**：2026-07-15（含联网检索）

---

## 方向一：router 不是最优，甚至"不按 router 走反而更好"

### ★ 最直接命中：《When Are Experts Misrouted? Counterfactual Routing Analysis in MoE LMs》（Microsoft Research, arXiv:2605.07260）
**这篇几乎就是你的问题，而且用了我们的主线模型 Qwen3-30B-A3B。**

- **方法（counterfactual routing）**：冻结整个 MoE 模型，对每个 token：
  1. 给它**当前 router 选的专家路线**算 next-token 概率（对真实下一个 token 的 log-likelihood）；
  2. 采样若干**等算力的替代路线**（同样 topk 个专家，但换一组），各算 next-token 概率；
  3. 对比"实际路线"vs"最佳替代路线"，看 router 是否真的选到了最优。
- **发现**：
  - **相当比例的 token，存在比 router 所选更好（loss 更低）的替代路线** → router 并非最优。
  - 错误集中在**"fragile tokens"**（驱动困难推理的关键 token），这些 token 最容易被 misroute。
  - **只微调最后一层的 router**（专家权重全冻结），就能在 AIME 2024/2025、HMMT 2025 等数学推理基准上**明显提升 pass@K** → 证明"瓶颈在 routing 本身，不在专家容量"。
- **在哪些模型验证**：Qwen3-30B-A3B、GPT-OSS-20B、DeepSeek-V2-Lite、OLMoE-1B-7B。
- **对你的意义**：**"不按 router 走可以更好"这件事已被严谨证明，而且就在 Qwen3-30B-A3B 上。** 但注意：他们的收益来自"换更优路线"（提升精度），**不是**"换更省搬运的路线"（你的角度是拿精度换效率，方向相反但同源——都建立在"router 有次优空间"这个前提上）。

### 佐证工作
- **Expert Choice Routing（NeurIPS 2022, Google）**：让**专家选 token**而非 token 选专家 → 负载更均衡、收敛快 2×+、下游更好。说明标准 top-k token routing 有系统性缺陷。
- **StableMoE（ACL 2022）**：router 训练不稳定（波动、坍缩），固定路由策略反而更稳。
- **综述结论**：因为负载均衡和专家专化难以联合优化，学到的 router 常收敛到差的局部最优；**随机/周期性打乱路由有时能匹配甚至超过学到的 router**。

---

## 方向二：改 route 排布 / 减少激活专家，换推理效率（★ 你的思路）

**结论：你的思路是一个真实、活跃的方向，已有多类工作，但仍有空白。** 分几类：

### 类别 A：Token routing reordering（重排，★ 纯加速无精度损失）
- **做什么**：不改 token 选哪些专家，只把 batch 里去**同一专家的 token 分组排在一起**，减少内存碎片、提高每次搬进来的专家权重的复用。
- **精度**：**无损**（选择没变，只改执行顺序）——检索结果里标为 "Pure speedup, no accuracy risk"。
- **代表**：Tutel（Microsoft）、FasterMoE 等系统里的 token sorting / grouped dispatch。
- **和 sglang 的关系**：sglang 的 `moe_align_block_size` 已经做了按专家排序分组——**这层已经在做了**，我们的 fused_moe 就是分组后的 GEMM。所以"纯重排"这块基本已被现有实现吃掉。

### 类别 B：减少激活专家数 / 动态 topk（★ 拿精度换效率，正是你的角度）
- **Adaptive-K / Dynamic routing**：根据 token 的不确定性**动态减少 topk**——简单 token（标点、常见词）少激活几个专家。
- **精度-效率权衡**：低 K → 省内存/带宽，但精度可能降。检索标为 "Tunable, risk of modest accuracy loss"。
- **Blockwise Expert Routing（arXiv:2312.00284）**：讨论 latency/accuracy 权衡。
- **和你 idea 的关系**：这是你"以损失部分性能换效率"最接近的已有工作，**但它是 per-token 减 K，不是你说的"批内向更少专家聚集"**（见空白点）。

### 类别 C：Expert popularity caching（热门专家缓存，正交手段）
- 跟踪最近 batch 里**最常被选的专家**，预取/常驻高带宽内存；冷门专家按需搬。
- 或热门高精度、冷门低精度（量化）。
- 代表：《Efficient MoE Inference via Dynamic Expert Caching》(arXiv:2203.07262)。
- **对我们**：Qwen3 专家全在单卡 HBM，没有"从慢存储搬"的问题，所以缓存收益有限；但"冷门专家量化"这条对减搬运字节有效。

### 类别 D：Expert parallelism / locality-aware routing（多卡场景）
- 多卡时把 token 路由到**本卡的专家**，减少跨卡通信。
- **对我们单卡场景不直接适用**（我们是 HBM→SM，不是跨卡）。

---

## 你的 idea 的定位：新颖点在哪

你的具体想法：**"轻微调整 route，让一个 batch 里的 token 尽量集中到更少的专家 → 减少每 batch 激活的专家数 → 减搬运，代价是精度略降。"**

对照上面：
| 已有工作 | 和你的异同 |
|---|---|
| Counterfactual routing (MS) | 证明了 router 有次优空间（你的前提成立），但他们换路线是为**提精度**，不是省搬运 |
| Token reordering (Tutel) | 只重排不改选择，**无损但也不减激活专家数**；sglang 已做 |
| Dynamic topk | 减 K 换精度，但是 **per-token 独立减**，不是**批内向更少专家聚集** |
| Expert caching | 正交（缓存而非改 route） |

**→ 你的独特角度 = "批内 expert 聚集"（batch-level expert consolidation）**：
- 不是 per-token 减 K，而是**利用 batch 维度**——如果 batch 里某些边缘 token 的次优专家恰好是已被其他 token 激活的专家，就把它们"劝导"过去，**减少 batch 激活的不同专家总数**。
- 这直接攻击我们量化的痛点：batch=32 激活 111 个专家 → 若能压到 60 个，搬运砍 46%。
- **这个"利用 batch 内 expert 重叠来减总激活数"的角度，在检索到的工作里没有直接对应**（dynamic-K 是 per-token，reordering 不减数量，caching 不改选择）。**可能是一个真实的空白/创新点。**

**关键前提已被验证**：Counterfactual routing (MS, 就在 Qwen3-30B-A3B 上) 证明了"很多 token 有等价或更好的替代路线" → 说明**把边缘 token 劝导到已激活专家，精度损失可能很小**（因为替代路线本来就常常不差）。这正好给你的 idea 提供了理论弹药。

---

## 建议的下一步（验证你的 idea）
1. **先量化空间**：实测我们模型在不同 batch 下的**专家激活分布**——batch 里的 token 有多集中/多分散？有多少"边缘 token"的次优专家恰好是已激活的？这决定了"批内聚集"能省多少。（单卡、当天可做，纯分析不占大量 GPU。）
2. **精度-搬运权衡曲线**：人为把 X% 的边缘 token 劝导到已激活专家，测激活专家数↓ vs perplexity↑ 的曲线。
3. **对照 MS 论文**：他们的 counterfactual 框架可以直接借来量化"劝导到已激活专家"的 loss 代价。

---

## 参考文献（真实，可引用）
- **When Are Experts Misrouted? Counterfactual Routing Analysis in MoE LMs** — Microsoft Research, arXiv:2605.07260（用了 Qwen3-30B-A3B，★最相关）
- Mixture-of-Experts with Expert Choice Routing — NeurIPS 2022（Google）
- StableMoE: Stable Routing Strategy for MoE — ACL 2022
- Efficient MoE Inference via Dynamic Expert Caching — arXiv:2203.07262
- Blockwise Expert Routing for LLMs — arXiv:2312.00284
- 系统实现：Tutel (Microsoft)、FasterMoE

> 注：文献来自联网检索，arXiv 编号以实际检索结果为准（部分编号需在 arXiv 二次核对；MS 那篇标注 2605.07260，日期格式偏未来，建议直接搜标题确认）。
