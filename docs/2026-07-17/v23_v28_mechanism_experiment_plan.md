# Copilot Agent Prompt：MoE 的 K 对生成长度影响——机理实验计划

你正在修改仓库：

```text
https://github.com/gujialiang123/MOEresearch
```

目标不是继续优化推理 kernel，而是把 **MoE expert 数量 K 对自由生成长度的影响** 单独作为一条行为与机理研究线，建立可信、可复现、可统计检验的因果证据。

请先阅读并复用仓库中已经验证过的 v20/v21/v22 实现与报告，尤其是：

- 物理跳过 dropped experts，而不是只把 routing weight 置零；
- K=8 keep-all 与原模型 logits、MoE output、greedy generation 等价；
- benchmark/forward 关键路径中不做逐层 `.item()`、`.cpu()`、`.tolist()`；
- 已有 GSM8K K=4/6/8 自由生成结果和 teacher-forced EOS 分析。

不要推翻已经通过正确性验证的实现。请在现有代码上模块化扩展。

---

## 一、研究问题

需要依次回答以下问题：

1. K 对长度的影响主要来自 **prefill prompt 表征变化**，还是 **decode 期间的轨迹累积变化**？
2. 现象是否主要由 dropped weights 的重新归一化、MoE residual branch 尺度变化造成，而不是 K 本身？
3. 多出来的 token 是：
   - 模型更晚才形成正确答案；
   - 已经知道答案，但更晚输出答案标记或 EOS；
   - 更啰嗦、回退、自我修正或生成退化；
   - 早期微小扰动导致的自回归轨迹分叉？
4. 哪些 MoE layer、哪些 decode 时段对最终长度最敏感？
5. K 与长度之间是平滑 dose-response，还是存在一个临界 K，低于它之后终止和格式稳定性突然崩坏？
6. K>原生 K=8 时是否出现反向长度趋势？该实验仅作为 OOD mechanistic probe，不作为正常推理策略结论。

在报告中禁止未经证据直接写“低 K 让模型主动多思考”。在 answer-readiness 结果出来前，只能使用更保守的表述：

> 降低 K 改变了自回归轨迹和内生输出长度。

---

## 二、总体实验原则

### 1. 先固定 K，不先使用 confidence-based dynamic K

主机理实验先使用固定：

```text
K ∈ {4, 5, 6, 7, 8}
```

避免“token 难度同时决定 K 和长度”这一混杂因素。原来的动态 threshold policy 留到机理确认后再回测。

### 2. 第一阶段只用 greedy decoding

统一：

```text
do_sample=False
temperature=None
top_p=None
```

固定 tokenizer、chat template、max_new_tokens、EOS 配置和随机种子。后续再增加 sampling robustness，不要在第一阶段引入采样噪声。

### 3. 主数据集先使用完整 GSM8K test split

- smoke test：前 16 或 32 题；
- development run：固定 200 题；
- main run：完整 GSM8K test set；
- 默认行为分析 batch size 为 1，避免 batch 完成时间、padding 或其他样本影响解释；
- 所有配置必须作用于完全相同的样本顺序。

暂时不要同时铺大量新 benchmark。完成本 prompt 的主机理实验后，再按结果选择 MATH-500、MMLU/BoolQ 或 ShareGPT。

### 4. 每个结果必须保存逐样本原始记录

每条样本至少保存到 JSONL：

```text
sample_id
question
gold_answer
config_id
prefill_k
decode_k
weight_mode
seed
prompt_token_count
generated_token_ids
generated_text
output_token_count
first_marker_token_index
first_parsed_answer_token_index
eos_token_index
has_answer_marker
has_eos
hit_max_new_tokens
post_answer_token_count
strict_parsed_answer
tolerant_parsed_answer
strict_correct
tolerant_correct
first_divergence_from_k8
repetition_3gram
repetition_4gram
avg_k_prefill
avg_k_decode
assignments_prefill
assignments_decode
```

若记录 runtime，请将 model generation GPU time、generation wall time、CPU parse time 分开；本研究线暂不把 timing 作为首要结论。

### 5. 结果必须可恢复、可重入

所有长实验支持：

```text
--limit
--start-index
--resume
--output-dir
--overwrite
```

配置写入 `config.json`，同时保存 git commit、模型 revision、transformers/torch/CUDA 版本和 GPU 信息。已经完成的 `(sample_id, config_id)` 不重复计算。

---

## 三、基础代码重构

请建立统一的 policy 与 metrics 模块，不要为每个实验复制一套 monkeypatch。

建议结构，可按仓库现状调整：

```text
moe_research/
  k_policy.py
  generation_metrics.py
  answer_parsing.py
  stats.py
scripts/
  run_v23_phase_factorial.py
  run_v24_weight_ablation.py
  run_v25_answer_readiness.py
  run_v26_direct_effect_probe.py
  run_v27_layer_time_intervention.py
  run_v28_k_dose_response.py
  analyze_v23_v28.py
```

### K policy 接口

至少支持：

```python
policy = KPolicy(
    prefill_k=8,
    decode_k=6,
    weight_mode="renorm_survivors",
    layer_selector=None,
    decode_step_selector=None,
)
```

### Prefill/decode 阶段判定

不要仅凭 MoE 层中的 `hidden_states.shape[1]` 猜阶段。应在模型顶层 forward 或 generation wrapper 中根据 cache 状态显式设置当前 phase：

- 初次完整 prompt、past cache 为空：`prefill`；
- past cache 已存在的后续调用：`decode`；
- 兼容可能的 multi-token decode 或 chunked prefill；
- 所有 MoE 层从同一个 policy context 读取 phase。

优先实现显式 context，例如：

```python
with policy_context.phase("prefill"):
    ...
with policy_context.phase("decode"):
    ...
```

或者在模型顶层 forward 中设置 thread-local/contextvar。不要让每个 MoE 层独立判断。

### 必须新增的测试

1. `prefill_k=8, decode_k=8` 与原模型：
   - MoE output 等价；
   - router logits 等价；
   - next-token logits 等价；
   - greedy generation token-for-token 等价。
2. phase routing test：
   - 初次 forward 使用 `prefill_k`；
   - 后续 cached forward 使用 `decode_k`；
   - 每层统计不能混淆两个阶段。
3. physical skip call-counter test：dropped expert 不执行 FFN。
4. benchmark mode 中禁止关键路径 `.item()`/`.cpu()`/`.tolist()`。
5. K 不得小于 1，K 不得超过 expert 数；默认原生 K=8。

---

# 四、实验 1：Prefill K × Decode K 因子实验（最高优先级）

## 目的

隔离 prompt 编码和 autoregressive decode 各自对生成长度的贡献。

## 主配置

首先在 `renorm_survivors` 下运行：

| Prefill K | Decode K | 配置名 |
|---:|---:|---|
| 8 | 8 | baseline |
| 6 | 8 | prefill6_only |
| 8 | 6 | decode6_only |
| 6 | 6 | both6 |
| 4 | 8 | prefill4_only |
| 8 | 4 | decode4_only |
| 4 | 4 | both4 |

必要时增加：

```text
(7,8), (8,7), (5,8), (8,5)
```

但先完成上表。

## 主要统计量

对每个配置报告：

```text
mean / median / p90 / p95 / p99 output length
mean L_to_marker
mean L_post_marker
no-answer-marker rate
no-EOS rate
max_new_tokens hit rate
strict accuracy
tolerant accuracy
3-gram / 4-gram repetition
first-divergence position vs K8
```

右截尾样本（命中 `max_new_tokens`）不能简单等价为正常长度。至少保存 censoring flag；若依赖可用，绘制 Kaplan–Meier survival curve，否则实现基础 survival/ECDF 分析。

## 因子效应

分别计算 K=6 与 K=4 的配对效应：

```text
Prefill effect:
L(Kp, 8) - L(8, 8)

Decode effect:
L(8, Kd) - L(8, 8)

Interaction:
L(K, K) - L(K, 8) - L(8, K) + L(8, 8)
```

对长度差使用 paired bootstrap 95% CI，至少 10,000 次 resampling。对 accuracy flips 使用 exact McNemar test，同时报告：

```text
baseline correct -> intervention wrong
baseline wrong -> intervention correct
```

## 解释规则

- `decode-only` 已明显增长：支持 decode trajectory accumulation；
- `prefill-only` 已明显增长：说明 prompt 表征/KV cache 是主因之一；
- 两个单独影响小、both 明显：说明存在累积或非线性交互；
- 任何结论都需要 paired CI，不只看均值。

输出：

```text
docs/<date>/v23_phase_factorial.md
results/<date>_v23_phase_factorial/
```

---

# 五、实验 2：Dropped-weight / residual-scale 消融（第二优先级）

## 目的

排除“长度变化主要来自重新归一化和 residual branch 尺度改变”，而不是 K 数量本身。

## Weight modes

必须实现并比较：

### A. `renorm_survivors`

```math
w'_j = w_j / \sum_{m \in keep} w_m
```

### B. `no_renorm`

保留原始 top-8 权重，删除尾部后不重新归一化：

```math
w'_j = w_j
```

### C. `fold_mass_to_top1`

将 dropped mass 全部加到 top-1：

```math
w'_1 = w_1 + \sum_{j \in drop} w_j
```

其他 surviving weights 保持原值。

### D. `calibrated_norm_match`（机制分析模式）

在独立 calibration subset 上按 layer 和 K 估计一个固定 scalar，使低 K MoE branch 的平均 L2 norm 接近 K8：

```math
s_{l,K} = E[||y_{8,l}||_2] / E[||y_{K,l}||_2]
```

评测时只使用预先标定的 `s_{l,K}`，不要每个 token 运行完整 K8 作为 reference。

## 主配置

至少运行：

```text
(Kp,Kd) = (8,8), (8,6), (8,4), (6,8), (4,8)
```

每个配置比较四种 weight mode。若成本过高，先在固定 200 题上完成，再对最关键组合跑完整 test set。

## 机制记录

在抽样 token/layer 上保存：

```math
norm_ratio = ||y_K||_2 / ||y_8||_2
cosine = cos(y_K, y_8)
relative_error = ||y_K-y_8||_2 / ||y_8||_2
```

同时报告这些量与最终：

```text
Δlength
ΔL_to_marker
no-marker
no-EOS
correctness flip
```

之间的相关性。

## 解释规则

- 只有 renorm 明显变长：主要怀疑 branch scale/calibration；
- no-renorm、fold、norm-match 都仍随 K 变长：K/专家函数缺失本身更可能是主因；
- 不允许把不同 weight mode 混在同一主曲线中而不标注。

输出：

```text
docs/<date>/v24_weight_ablation.md
results/<date>_v24_weight_ablation/
```

---

# 六、实验 3：Answer-readiness probe（第三优先级）

## 目的

区分：

```text
A. 低 K 真的让模型更晚形成正确答案
B. 模型早已能答对，但更晚输出答案标记/EOS
```

当前 `L_to_answer` 只代表第一个 `####`/`FINAL:` 之前的长度，不能直接当作真实 reasoning length。

## 样本与 checkpoint

先在 100–200 个分层样本上运行：

- K8/K6/K4 均正常答题的样本；
- K8 正确但 K4/K6 错误的样本；
- 低 K 明显变长的上分位样本；
- 低 K no-marker 或 max-token 样本。

对每条自由生成轨迹保存 checkpoint：

```text
t = 0, 32, 64, 96, ...
```

并在实际 marker 前后额外加入更密集 checkpoint，例如：

```text
marker-32, marker-16, marker-8, marker
```

## Probe 1：Gold-answer conditional log-likelihood

在当前 prefix 后追加统一 cue：

```text
\nTherefore, the final answer is ####
```

使用原生 K8 作为 probe model，teacher-force gold numeric answer，记录：

```text
sum logprob
mean token logprob
minimum token logprob
```

使用 K8 probe 是为了测“当前 prefix 是否已包含足够答案信息”，避免把 probe 自身低 K 误差混入。

## Probe 2：Short greedy answer probe

对同一 prefix 使用 K8，最多生成 16–24 token，只允许输出最终答案格式，记录能否解析为正确答案。

建议同时实现两种上下文：

1. `full_context_probe`：原问题 + 当前 reasoning prefix；
2. `prefix_only_probe`：只给 reasoning prefix。

`t=0` 必须作为控制。若 full-context probe 在没有 reasoning 时就能直接答对，该样本不能用它证明 reasoning readiness，应同时参考 prefix-only 与 gold-answer logprob。

## t_ready 定义

至少给出两个版本：

```text
t_ready_greedy:
第一个 greedy probe 正确且后续至少一个 checkpoint 仍正确的位置

t_ready_logprob:
首次超过由 K8 正确样本 calibration 得到的阈值的位置
```

不要手工选一个对结果最有利的 threshold。阈值需要在独立 calibration subset 上固定。

## 核心分解

报告：

```text
t_ready
t_actual_marker
t_eos

t_actual_marker - t_ready
t_eos - t_actual_marker
```

## 解释规则

- 低 K 的 `t_ready` 也显著变晚：支持真正的 reasoning/answer formation delay；
- `t_ready` 基本不变、marker/EOS 变晚：支持 verbosity/format/termination effect；
- `t_ready` 波动大且 first divergence 很早：支持 autoregressive trajectory bifurcation；
- 不得仅凭生成文本看起来更长就称为“更多有效推理”。

输出：

```text
docs/<date>/v25_answer_readiness.md
results/<date>_v25_answer_readiness/
```

---

# 七、实验 4：真正的当前步 direct-effect probe

## 目的

改进现有 v22。v22 固定了 token IDs，但低 K 从整条序列开头累积计算，不能严格隔离“只改变当前 step 的 K”对 EOS/next-token logits 的直接影响。

## 方法

使用 K8 baseline token trace。对选定 step `t`：

1. 使用 K8 计算 prompt 和 `x_{<t}`，获得完全相同的 K8 `past_key_values`；
2. 固定同一个 input token `x_t` 和同一份 K8 past；
3. 仅当前 forward 分别使用 K=8、6、4；
4. 不让 K6/K4 的 past 继续污染后续历史；
5. 比较当前 next-token distribution。

记录：

```text
KL(K8 || Kk)
gold-next-token Δlogprob
EOS Δlogprob
EOS margin change
answer-marker token Δlogprob
next-token top1 agreement
logit L2 / cosine
```

重点采样：

- baseline EOS 前最后 64 个位置；
- first marker 前最后 64 个位置；
- 首次 K8/K6 或 K8/K4 自由生成分叉附近；
- 随机早期/中期位置作为控制。

这才可以称为：

> current-step direct effect given identical K8 history

将现有 v22 改称：

> trajectory-fixed cumulative distribution effect

不要删除旧结果，但在报告中澄清两者区别。

输出：

```text
docs/<date>/v26_direct_effect_probe.md
results/<date>_v26_direct_effect_probe/
```

---

# 八、实验 5：Layer × time 局部干预（第四优先级）

## 目的

定位长度效应来自哪些层、哪些 decode 时段，并为后续 layer-aware K policy 提供依据。

## 第一阶段：粗粒度 layer groups

按实际 MoE layer 列表划分：

```text
early third
middle third
late third
```

只在一个 group 中使用 K6，其余层 K8；prefill 先固定 K8，只干预 decode：

| Early | Middle | Late |
|---:|---:|---:|
| 6 | 8 | 8 |
| 8 | 6 | 8 |
| 8 | 8 | 6 |
| 6 | 6 | 8 |
| 8 | 6 | 6 |
| 6 | 8 | 6 |
| 6 | 6 | 6 |

先在 200 题上跑。若某组明显敏感，再细分到具体 layer。

## 第二阶段：粗粒度 decode windows

先使用在线可实现的绝对窗口：

```text
early: decode step 1–32
middle: step 33–96
late: step 97+
```

分别只在一个窗口内用 K6，其余时间恢复 K8。若样本通常较短，可根据现有长度分布调整窗口，但配置必须预先固定。

## 第三阶段：3×3 layer-time map

只在以下一个 block 使用 K6：

```text
{early, middle, late layer group}
×
{early, middle, late decode window}
```

其余全部 K8。

记录：

```text
Δlength
Δt_ready
Δmarker position
ΔEOS position
first divergence
accuracy flips
KL / hidden-state drift（抽样）
```

不要一开始全量逐层逐 token 扫描。先完成 3×3，再对高影响 block 细化。

## 可选精细干预

对敏感区域定义：

```math
I_{l,t} = E[L_{intervene(l,t)} - L_{K8}]
```

只在单个 layer 或短 time window 降 K，然后恢复 K8 继续生成。

输出：

```text
docs/<date>/v27_layer_time_intervention.md
results/<date>_v27_layer_time_intervention/
```

---

# 九、实验 6：K dose-response 与临界点（第五优先级）

## 主实验

在最干净的 phase 和 weight setting 下运行：

```text
K = 4, 5, 6, 7, 8
```

推荐优先：

```text
prefill K=8
decode K∈{4,5,6,7,8}
```

因为它能隔离 decode effect。若实验 1 表明 prefill 是主因，再增加 prefill dose-response。

报告：

```text
mean/median/p90/p95/p99 length
survival curve
L_to_marker
L_post_marker
t_ready
no-marker
no-EOS
max-token hit
strict/tolerant accuracy
paired correctness flips
first-divergence distribution
repetition
```

为每个相邻 K 计算 paired difference：

```text
K8 -> K7
K7 -> K6
K6 -> K5
K5 -> K4
```

## 临界点分析

拟合并比较：

1. 线性/平滑 dose-response；
2. piecewise linear / change-point model；
3. logistic model for `no_marker` / `hit_max_tokens`。

目标不是强行证明 phase transition，而是检验是否在 K5 左右出现非线性稳定性崩坏。

## Optional：super-native K probe

支持：

```text
K = 9, 10, 12
```

但必须单独输出到 `ood_super_native_k` section，并明确：

- Qwen 原生训练 K=8；
- K>8 是分布外 mixture intervention；
- 不能与 K≤8 的正常 dose-response 混为一谈；
- 只用于观察增加尾部 expert 是否反向改变长度、EOS margin 和 output scale。

还应增加更干净的连续 tail-restoration probe：

```math
y_α = \sum_{j=1}^{k} w_jF_j(h) + α\sum_{j=k+1}^{8}w_jF_j(h)
```

扫描：

```text
α ∈ {0, 0.25, 0.5, 0.75, 1.0}
```

优先使用该实验，而不是把 K 扩到 12 后直接解释为“更多专家让模型更短”。

输出：

```text
docs/<date>/v28_k_dose_response.md
results/<date>_v28_k_dose_response/
```

---

# 十、答案解析与长度指标要求

## 1. 双评分器

实现：

### strict parser

仅接受官方/目标格式，例如：

```text
#### 42
```

### tolerant parser

额外接受：

```text
#### $42
#### \boxed{42}
#### Answer: 42
FINAL: 42
```

保存 parser failure 类型，并人工导出 correctness flips 样本供审查。

## 2. 长度分解

至少定义：

```text
L_total
L_to_first_marker
L_to_first_parsed_answer
L_post_answer
L_to_eos
```

没有 marker/answer/EOS 时使用 `None`，不要默默替换成总长度；统计时单独报告 missing/censored。可以额外提供用于 survival analysis 的 censoring length。

## 3. First divergence

对相同 sample 的 K8 与干预输出 token IDs，从第一个 generated token 开始找到首个不同位置。记录：

```text
first_divergence_index
prefix_equal_fraction
```

并分析 first divergence 与最终 Δlength、correctness flip 的关系。

---

# 十一、统计与可视化

每份报告至少包含：

1. 汇总表；
2. paired length delta 分布；
3. survival/ECDF 曲线；
4. p50/p90/p99；
5. no-marker/no-EOS/max-token rate；
6. strict/tolerant accuracy 和 paired flips；
7. first-divergence 与 Δlength scatter/分桶；
8. answer-readiness 三时间点图：`t_ready`, `t_answer`, `t_eos`；
9. prefill/decode factorial interaction plot；
10. K=4..8 dose-response 与 change-point 分析。

统计方法：

```text
paired bootstrap 95% CI
exact McNemar test
Holm correction for multiple pairwise comparisons
```

避免只报告平均值和单次运行。

---

# 十二、结果决策树与后续应用原型

完成实验后，根据结果自动生成一个 `mechanism_summary.md`，按以下规则给出保守结论和下一步应用建议。

## 情况 A：Decode-only 导致主要长度增长，t_ready 也变晚

解释：每 token expert compute 减少可能延迟 answer formation。

下一步：

```text
K6 default
检测到高 entropy / 长轨迹 / 低 answer readiness 时升级 K8
```

研究目标：spatial-temporal compute allocation。

## 情况 B：t_ready 不变，但 marker/EOS 变晚

解释：主要是表达、格式遵循或终止失准。

下一步：

```text
termination-aware K restoration
接近答案/终止的 late layers 或 late decode steps 恢复 K8
runtime 在可靠 FINAL marker 后停止
```

研究目标：termination-preserving expert sparsification。

## 情况 C：Prefill-only 已造成主要变化

解释：prompt 编码/KV cache 质量敏感。

下一步：

```text
prefill 保持 K8
decode 使用低 K
```

研究目标：phase-aware expert allocation。

## 情况 D：Renorm 决定大部分效应

解释：主要是 residual branch scale/calibration。

下一步：

```text
no-renorm / fold-to-top1 / calibrated norm matching
```

研究目标：scale-preserving expert skipping。

## 情况 E：长度敏感性集中在少量 late layers/time windows

下一步：

```text
大部分 layer K6
敏感 layer/window K8
```

研究目标：layer/time selective native-K restoration。

## 情况 F：K5 以下 no-marker/max-token 突然上升

解释：存在稳定性临界区。

下一步：

```text
online safety floor K>=6
或根据风险信号禁止跨过临界 K
```

研究目标：stability-constrained dynamic K。

---

# 十三、跨 benchmark 扩展条件

不要立刻全铺。主机理结果明确后再选：

- 若 `t_ready` 变晚：增加 MATH-500，验证更长、更难 reasoning；
- 若 marker/EOS/格式主导：增加 MMLU、BoolQ 或 ARC 的严格短输出任务；
- 若 verbosity/repetition 主导：增加 ShareGPT/open-ended prompts；
- 最终至少增加一个 native top-2 或 top-4 的不同 MoE 模型，验证是否为 Qwen top-8 特例。

---

# 十四、验收标准

代码和报告只有同时满足以下条件才算完成：

1. K8/K8 与原模型严格等价；
2. prefill 和 decode 的 K 能独立控制并有单元测试；
3. dropped experts 物理不执行；
4. 不在关键 forward 路径逐层同步 CPU；
5. 每个配置有逐样本 JSONL、可恢复运行和完整环境元数据；
6. strict/tolerant parser 同时报告；
7. marker、answer、EOS、censoring 分开记录；
8. 主结论使用 paired CI/McNemar，不只看均值；
9. 在 `t_ready` 未证实前，不写“更多有效推理”；
10. 在 direct current-step probe 未证实前，不写“低 K 直接压低 EOS”；
11. K>8 结果明确标为 OOD probe；
12. 报告最后给出“证据支持什么 / 不支持什么 / 下一步应用”的三段式结论。

---

# 十五、建议执行顺序

严格按信息增益排序：

```text
0. 基础重构与正确性测试
1. Prefill K × Decode K factorial
2. Weight renorm / residual-scale ablation
3. Answer-readiness probe
4. Current-step fixed-KV direct-effect probe
5. Layer × time intervention
6. K=4..8 dose-response / change-point
7. 根据机制选择新 benchmark 和应用原型
```

先完成 1–4，就应能回答现象主要属于：

```text
prompt representation
trajectory accumulation
answer formation delay
verbosity/format delay
termination instability
residual-scale artifact
```

不要在这些因素尚未分离时直接实现复杂 learned controller。