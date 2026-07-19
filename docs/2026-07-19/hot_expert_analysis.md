# Hot-Expert 研究：emergent shared-expert 可行性验证

**日期**：2026-07-19 · 分支 `main` · 模型 `Qwen3-30B-A3B-Instruct-2507`（E=128, top-8, L=48）
**动机**：v13/v16 发现 router 强偏斜（Gini 0.646；热门 10% 专家吃 41% 流量；top1 置信仅 8.9%）。
本研究验证能否把这种"热偏移"变成 **lossless** 的系统优化（Idea 1：把 emergent 的热专家当隐式 shared-expert 常驻/加速）。

**方法**：收集 4 个 workload（gsm8k/coding/general/synthetic）在 **decode 位置**的每层每专家选择计数矩阵 [L,E]，
验证三个生死前提：Q1 跨层稳定性、Q2 跨 workload 稳定性、Q3 L2/常驻预算 fit。

**脚本**：`scripts/hot_expert/{run_h1_collect_traces,analyze_h2_hot_expert,plot_h3_hot_expert}.py`
**数据**：`results/2026-07-19_hot_expert/`

---

## Q3（权重尺寸算术，不依赖 trace，已确定）

| 量 | 值 |
|---|---|
| 每 expert 参数（gate+up+down, 3×768×2048） | 4,718,592 |
| 每 expert bf16 大小 | **9.44 MB** |
| 一层全部 128 experts | 1208 MB |
| 一层激活 top-8 | **75.5 MB** |
| H200 L2 (~50MB) 能放下 | **仅 5 个 expert** |
| 全模型 MoE 权重 | 58 GB |
| 单 token decode 全 48 层激活搬运 | 3.62 GB/token |
| → H200 ~4.8TB/s 理论下限 | ~0.75 ms/token（纯 MoE 搬运） |

**硬结论（L2-persisting 类 idea）**：H200 L2 只能放 5 个 expert，**连一层的激活集（75.5MB）都装不下**。
→ "热专家常驻 L2"基本不可行。剩下的希望是"热专家常驻 **VRAM** 并用更快物理路径（dense GEMM）计算"，
这取决于 Q1/Q2 的热专家稳定性（trace 分析中）。

---

## Q1 跨层稳定性 · Q2 跨 workload 稳定性

**数据**：4 workload（gsm8k/coding/general/synthetic），decode 位置,各 4.5–6.4k decode tokens,
每层每专家选择计数矩阵 [48,128]。指标:top-N 热集的跨层/跨workload Jaccard(1=完全相同,0=完全不同)。

### Q1 — 跨层:热专家几乎完全不同(❌ 无法全局常驻)

| top-N/层 | 跨层平均 Jaccard | 相邻层 Jaccard | 全局热集覆盖每层流量 | 每层自身热集覆盖 |
|--:|--:|--:|--:|--:|
| 8 | 0.048 | 0.033 | 8.5% | 19.2% |
| 16 | 0.084 | 0.066 | 16.1% | 32.6% |
| 32 | 0.157 | 0.134 | 30.3% | 53.2% |

- **跨层 Jaccard 极低(0.05–0.16)**:不同层的热专家几乎不重叠。
- 直观证据:layer0 top5 = [84,60,13,87,124]、layer24 = [115,98,56,95,44]、layer47 = [59,22,38,105,67]
  → **三层完全不相交**。
- 单一"全局热集"只覆盖每层 8–30% 的 decode 流量 → **不存在跨层通用的热专家**。

### Q2 — 跨 workload:热专家高度输入相关(❌ 无法静态预计算)

| workload 对 | top16 平均 Jaccard |
|---|--:|
| gsm8k vs coding | 0.116 |
| gsm8k vs general | 0.090 |
| gsm8k vs synthetic | 0.130 |
| coding vs general | 0.057 |
| coding vs synthetic | 0.089 |
| general vs synthetic | 0.025 |
| **平均** | **0.085** |

- gsm8k 学到的热集只覆盖其他 workload 15–20% 的流量。
- **热度是输入相关的,不是全局先验** → 无法离线预计算一个静态热集。

### 与 v16 的偏斜对比(重要修正)

v16 的"热偏移"是在 **5 个 agent prompt 的 prefill** 上测的(Gini 0.646,热门 10% 吃 41%)。
本研究在 **decode**(真正的瓶颈位置)上重测:

| | v16 prefill | 本研究 decode |
|---|--:|--:|
| per-layer Gini | 0.646 | **0.441**（min 0.329, max 0.546） |
| 热门 10%（13 专家）流量占比 | 41% | **27.9%** |

→ **decode 的偏斜明显弱于 prefill**。当初激励这个 idea 的强偏斜,部分是 prefill 假象;
decode 更平,进一步削弱"少数热专家垄断"的前提。

---

## 结论

### 三个生死前提全部证伪 ❌

| 前提 | 结果 | 判定 |
|---|---|---|
| Q3 L2 fit | H200 L2 只放 5 个 expert < 一层激活集 75.5MB | ❌ L2-persisting 不可行 |
| Q1 跨层稳定 | 热集跨层 Jaccard 0.05–0.16,层间几乎不相交 | ❌ 无法全局常驻 |
| Q2 跨workload稳定 | 热集跨workload Jaccard 0.085,高度输入相关 | ❌ 无法静态预计算 |

### 对各 idea 的裁决

- **Idea 1（emergent shared-expert 常驻/dense 化）**:❌ **否决**。DeepSeek 式 shared-expert 是训练时
  *设计*出一组**所有 token、所有层共用**的专家;Qwen 里**不存在**这样的 emergent 结构——热专家
  每层一套、每 workload 一套。没有可常驻的稳定热核。
- **Idea 2（热度先验的确定性预取/常驻）**:❌ **否决**。"免费 predictor = 热度先验"不成立,因为
  热度既不跨层也不跨输入稳定;要预取仍需每层每步的真实/预测路由(退回到已被否决的 predictor 路线)。
- **residency Pareto**:要靠常驻覆盖 80% decode 流量需 top-64/层 = **29GB VRAM**(见图4),
  等于把半个模型常驻,毫无意义。

### 正面收获(转向)

1. **decode 偏斜被高估**:真正瓶颈位置的偏斜(Gini 0.441)比 prefill 观察值温和,说明
   "利用偏斜"这条系统路线整体前景弱于预期。**统一叙事**:v16 的强偏斜是 prefill 现象,
   不应外推到 decode 优化。
2. **回到真正的 lossless 杠杆**:既然"热核常驻/预取"死路,单卡 decode 下唯一能提升算术强度的
   lossless 手段仍是 **speculative decoding**(一次 forward 服务 γ+1 token,摊薄专家搬运)——
   这不依赖热专家稳定性,收益来自 step 维度的复用。建议作为下一个方向。
3. **机理角度(Idea 6)仍成立**:router 低置信(8.9%)+ 热度输入相关,支持"router 选择部分被
   流行度而非语义驱动"的研究问题,可作为解释 v17–v22 结果的机理 paper,但**无直接系统收益**。

### 一句话

> **Qwen3-30B 的 MoE 不存在可被 lossless 利用的稳定热专家结构**:热专家每层不同、每 workload 不同,
> 且 decode 偏斜本就弱于 prefill。热专家常驻/预取方向应终止;lossless 系统收益应回到
> speculative decoding(step 维度摊薄搬运)。

**图**:`results/2026-07-19_hot_expert/plots/`（1 负载热图 / 2 跨层Jaccard矩阵 / 3 跨workload曲线 / 4 常驻Pareto）
**数据**:`results/2026-07-19_hot_expert/{traces,analysis.json}`

