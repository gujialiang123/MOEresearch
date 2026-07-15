# 动态 topk 可行计划：实现难度 + 轻量级精度 benchmark

**目的**：回应用户——(1) 做动态 topk（或先固定改 topk）的实现难度；(2) 因为有损，需要轻量级精度 benchmark；调研并给可行计划。
**日期**：2026-07-15（含联网检索 + 源码核查）

---

## Part 1：动态 topk 的实现难度

### 结论：非常低。两侧都有现成 hook。

### 1A. Transformers 侧（快速验证/研究用）—— 1 行
`Qwen3MoeSparseMoeBlock.top_k` 是普通属性，patch 即可（v15 已验证可行）：
```python
for block in model.modules():
    if isinstance(block, Qwen3MoeSparseMoeBlock):
        block.top_k = r   # 固定改
```
**动态版**（按 token 置信度自适应）：需改 `forward` 里 `torch.topk(..., self.top_k)` 之后，用 router 概率做一个掩码——比如"累积概率达到阈值 τ 就停"（top-p 式）或"低置信度 token 用小 k"。约 10-20 行，改一个 forward。

### 1B. sglang 侧（真实推理加速用）—— 也不难
- `sglang/srt/layers/moe/topk.py` 的 `TopKConfig.top_k` 控制；`TopK` 类持有 `self.topk_config.top_k`。改这个字段即可固定改 topk。
- **更干净的落点**：`TopKConfig.custom_routing_function`（现成 hook！）——传一个自定义路由函数，就能实现动态 topk / 批内聚集，不动核心代码。
- 难点不在"改 topk"，而在"**让 kernel 真正受益**"：sglang 的 fused_moe 按 BLOCK_SIZE_M 分组，减少激活专家数后要确保 `moe_align_block_size` 和 kernel 真的少搬权重（否则只是逻辑上少选，物理上没省）。这一步需要验证端到端 latency 真的降。

### 实现难度分级
| 目标 | 难度 | 工作量 |
|---|---|---|
| 固定改 topk（8→6→4）测精度 | ★☆☆☆☆ | v15 已做，1 行 |
| 动态 topk（按置信度自适应）测精度 | ★★☆☆☆ | 改 1 个 forward，~20 行 |
| 批内聚集（劝导到已激活专家）测精度 | ★★★☆☆ | 需 batch 级重路由逻辑 |
| sglang 端真实加速（kernel 受益） | ★★★★☆ | custom_routing_function + 验证 kernel 少搬 |

**建议**：先在 transformers 侧做 1A（固定 + 动态），拿精度-效率曲线；确认值得后再上 sglang 做真实加速。

---

## Part 2：轻量级精度 benchmark（回应"SWE-bench 太重"）

### 结论：用 `lm-evaluation-harness` + 小 limit，几分钟出结果。

### 2A. 推荐的轻量级 benchmark（按"轻+有区分度"排序）
| benchmark | 测什么 | 轻量做法 | 为什么选它 |
|---|---|---|---|
| **GSM8K** | 小学数学推理 | `--limit 100-200` | ★ 对"丢专家"最敏感（MS 论文发现 misroute 集中在推理 token）→ 能放大精度损失，是最好的探针 |
| **HellaSwag** | 常识/句子续写 | `--limit 200-500` | 快、稳定、多选题（无需生成），几分钟 |
| **MMLU（子集）** | 多领域知识 | `--limit 100` 或选几个 subject | 覆盖广，看知识损失 |
| **Perplexity（已做）** | 语言建模 | v15 已有 | 最快但最粗，作 sanity check |

### 2B. 工具：lm-evaluation-harness（EleutherAI）
```bash
pip install lm-eval
# 例：GSM8K 前 200 题
lm-eval --model hf --model_args pretrained=/data/hf/models/Qwen3-30B-A3B-Instruct-2507,dtype=bfloat16 \
        --tasks gsm8k --limit 200 --device cuda:0
```
- 支持 HF 模型直接跑；`--limit` 控制样本数（50-250 够看趋势）。
- **关键技巧**：要测"动态 topk 的精度损失"，需在加载后 patch `block.top_k`，再用 lm-eval 的 Python API 跑（把 patch 过的 model 传进去）：
```python
import lm_eval
from lm_eval.models.huggingface import HFLM
# ... load model, patch top_k ...
lm = HFLM(pretrained=model, tokenizer=tok)
res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k","hellaswag"], limit=200)
```

### 2C. 为什么 GSM8K 是最好的探针
MS《When Are Experts Misrouted》发现：misrouting 的伤害**集中在 fragile/推理 token**（数学、逻辑），简单 token 几乎不受影响。所以：
- 丢专家对 **GSM8K（纯推理）** 的伤害会被放大 → 最敏感的探针，能看到精度损失的上界。
- 对 HellaSwag（常识）伤害小 → 看下界。
- 两者一起，画出"不同任务类型对丢专家的敏感度"。

---

## Part 3：可行计划（3 步，递进）

### Step 1（当天，1A 固定 topk + lm-eval）：建立精度-效率曲线
- 装 lm-eval；对 topk=8/7/6/5/4 各跑 GSM8K(200) + HellaSwag(300)。
- 产出：**准确率 vs topk** 曲线（配合 v15 的 perplexity + v14 的搬运节省）。
- 预期：印证 v15 的拐点（top5-6 精度可接受，top4 以下崩）；GSM8K 掉得比 HellaSwag 多。
- 成本：每个 topk × 2 任务 ≈ 10-20 min，共约 2 小时。

### Step 2（动态 topk）：自适应减少激活
- 实现"按 router 置信度动态 k"：token 的 top-p 累积概率达阈值 τ 就停（简单 token 自然用少专家，难 token 保留多）。
- 用 v16 数据设 τ：选中 8 个总置信度才 38%，可设 τ=0.30-0.35，让平均 k 降到 5-6。
- 跑同样的 GSM8K + HellaSwag，对比"固定 topk"——**动态应在同等精度下省更多**（因为它保护了难 token）。
- 成本：改 ~20 行 + 同样的 eval，约半天。

### Step 3（可选，批内聚集 + sglang 真实加速）
- 实现"劝导到已激活专家"的批内聚集（比动态 topk 更保精度，因为保留计算）。
- sglang 侧用 custom_routing_function 落地，测端到端 decode latency 真降多少。
- 用 lm-eval 确认精度可接受。

---

## 建议的度量矩阵（最终产出一张表）
| 方法 | 搬运节省 | perplexity | GSM8K acc | HellaSwag acc | 端到端加速 |
|---|---|---|---|---|---|
| baseline (top8) | 0 | 5.77 | ? | ? | 1× |
| top6 固定 | ~13% | +5.5% | ? | ? | ? |
| top5 固定 | ~20% | +7.6% | ? | ? | ? |
| 动态 (τ=0.32) | ? | ? | ? | ? | ? |
| 批内聚集 | ? | ? | ? | ? | ? |

（? = 待 Step1-3 填）

---

## 风险 / 注意
1. **lm-eval 与 patch 的集成**：要确保 lm-eval 用的是 patch 过 top_k 的 model（用 Python API 传 model 对象，别用 CLI 重新加载）。
2. **GSM8K 需要生成 + 数值解析**：比 HellaSwag（多选）慢，但对推理最敏感，值得。
3. **transformers 精度 ≠ sglang 精度**：transformers 侧测的是"算法层"精度损失；sglang 端到端加速要另测。两者分开报告。
4. **小 limit 的噪声**：GSM8K limit=200 的准确率有 ±3-4% 噪声，看趋势不看绝对值；关键 config 可加大到 500。

## 产物（本调研）
- 本文档
- 依据：v14（搬运）、v15（perplexity）、v16（router 分布/置信度）
- 工具：lm-evaluation-harness（EleutherAI）
- 源码落点：transformers `Qwen3MoeSparseMoeBlock.top_k`；sglang `TopKConfig.top_k` / `custom_routing_function`
