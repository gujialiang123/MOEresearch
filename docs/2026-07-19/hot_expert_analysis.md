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

（待 trace 收集完成后由 h2 填入）

---

## 结论

（待补）
