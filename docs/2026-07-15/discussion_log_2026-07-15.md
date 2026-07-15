# 2026-07-15 讨论纪要：从"gap 看得见"到"gap 摸得着"再到"kernel 层根因"

**目的**：完整记录今天的对话逻辑链——干预实验证明 gap 可回收、scheduler 机制、MoE kernel 根因分析、搬运 vs 计算的本质。按讨论顺序整理，含所有数据、证据、问答。
**模型**：LFM2.5-8B-A1B、Qwen3-30B-A3B（bf16）；**硬件**：H200（132 SM，HBM 4.8 TB/s，bf16 TC 989 TFLOP/s）

---

## 阶段一：干预实验——证明两个 gap "摸得着"（可回收，非只看得见）

**背景**：之前（v6–v10）已证明两个 gap **存在（看得见）**：serving idle（真实单流 GPU ~86% 在等请求）、kernel SM 空转（decode 时 SM 67–78% 周期在干等内存）。今天要证明它们**摸得着**（用 config 以外的手段真的推动指标）。

### 实验 B2：多路并发流 → serving idle 摸得着（★强证据）
- **办法**：一个 server，同时打 N=1/2/4/8 条独立真实 toolagent 流（模拟多租户），nsys 测 GPU 利用率。
- **结果（LFM2.5）**：

| 并发流数 | GPU 利用率 | 合计吞吐 |
|---|---|---|
| 1 | 13% | 1087 tok/s |
| 2 | 18% | 2163 |
| 4 | 25% | 4217 |
| 8 | **32%** | **8062** |

- **结论**：利用率 13%→32%（2.5×）、吞吐 7.4×。streams=1 复现 v9d 的 14%（自洽）。**证明 serving idle 是负载不足、多租户能回收。**

### 实验 A2：换 attention backend → 负面对照
- fa3 8.71ms（最优）、triton 10.27ms（慢 18%）、flashinfer 起不来（JIT ld 错误）。
- **结论**：fa3 已是现有实现最优，换 backend 没空间。

### 实验 A1：EAGLE3 spec decoding → 负面对照
- Qwen3-32B + redhat EAGLE3 draft：TPOT 52→186ms（反而更慢），**accept length 仅 1.28**（理想 2–4）。
- **结论**：draft 与 target 不匹配 → 反而变慢。印证"投机源必须匹配"。

### 实验 A1b：n-gram spec decoding → kernel 层摸得着（★正向证据）
- **办法**：n-gram spec（**不需 draft 模型**，从已生成 context 找匹配；exact 方法）；直接在**主线 Qwen3-30B-A3B** 上做。
- **结果**：

| conc | TPOT 基线 | TPOT ngram | 改善 | accept |
|---|---|---|---|---|
| 1 | 4.33ms | 4.06ms | −6% | 1.85 |
| 32 | 18.64ms | 14.27ms | **−23%** | 2.08 |

- **结论**：主线模型 decode TPOT **降 23%**（exact 方法）。conc=32 收益>conc=1，与"batch 大 SM 空转多"诊断自洽。**证明 kernel/算法层 space 摸得着。**

**阶段一小结**：两个 gap 都拿到正向实证——serving idle 靠多租户（利用率 2.5×）、kernel 靠 n-gram spec（TPOT −23%）。两个负面对照（fa3 已最优、EAGLE3 draft 不匹配）说明**拿到收益需选对手段**。

---

## 阶段二：NCU 测 spec decoding 的机制——反直觉的重要发现

**问题**：spec decoding 具体减少了多少 SM 空转空间？
**做法**：NCU（WarpStateStats+SchedulerStats）测 ngram vs baseline 的 decode No-Eligible。

| 变体 | 时间加权 No-Eligible（SM 空转） | SM 利用 |
|---|---|---|
| baseline | 77.5% | 21.1% |
| ngram | 78.0% | 19.8% |

**发现**：**spec decoding 几乎没降低单 kernel 的 SM 空转率（77.5→78.0%）。**

**机制修正（重要）**：spec decoding 降 TPOT **不是靠"填满 SM 空转"，而是靠"减少前向次数"**——一次前向验证 ~2 token → 生成 N token 只需 ~N/2 次前向。每次前向仍 78% 空转，但次数减半 → TPOT 下降。
> 类比：不是"让厨师不发呆"，而是"让厨师一次多做几道菜"。
**含义**：decode 的 78% SM 空转是 memory-bound 本质，spec decoding **绕过而非消除**它。纯 kernel 层 gap 仍未被任何现成手段攻克。
> 注：bench_one_batch+NCU 使 decode latency 绝对值失真；TPOT 收益以 A1b server 实测（−23%）为准，NCU 仅用于 per-kernel 空转率。

---

## 阶段三：scheduler policy 机制（读源码）

**问题**：agent-aware 调度（另起项目）之前，先搞清现在的 scheduler 机制。

### 请求生命周期
```
recv → waiting_queue → [每 tick] get_next_batch_to_run → run_batch
```
单线程 tick 循环（`scheduler.py:1108`）。

### 两个层面的决策（影响力完全不同）
| 决策 | 谁控制 | 影响 |
|---|---|---|
| prefill 优先 / batch 多大 / 显存预算 | 硬编码 + PrefillAdder + max-running-requests、chunked-prefill | **GPU 一步算多少（利用率/吞吐）** |
| 等待队列排序（--schedule-policy） | lpm/fcfs/lof | 只是**先后顺序**（cache 命中/公平/单请求延迟） |

### 三个决定性机制
- **A. prefill 绝对优先于 decode**（`:1935`）：新请求能组 prefill 就插队，打断正在跑的 decode（保 TTFT，但 TBT 抖动）。硬编码，policy 改不了。
- **B. continuous batching**（`:1906,1919`）：每 tick 把当前所有在跑请求一起 decode，batch 动态；新请求 prefill 完 merge 进来，完成的 filter 出去。
- **C. waiting_queue 排序**：默认 **lpm**（按最长前缀，最大化 radix cache 复用），队列>128 自动退化 fcfs（`:159`）；还有 in-batch 去重（共享未缓存前缀时只放一个进去）。
- 兜底：显存满时 **retract**（把部分 decode 请求踢回队列重排，白算的 KV 作废）。

### 关键问答：这些对 SM 利用率 / 接近硬件上限有帮助吗？
**基本没有。** 关键区分：
- **GPU 利用率（忙/闲）** = 有没有活 → scheduler **能影响**（组大 batch → 减 idle）。
- **SM 利用率（忙时满不满）** = 单 kernel 内部 → scheduler **碰不到**（kernel 一旦启动，该等内存还是等内存）。

| 手段 | 能改善 | 碰不到 |
|---|---|---|
| scheduler policy（排序） | cache 命中/公平/延迟 | ❌ SM 利用率、GPU 利用率 |
| batch/并发 knob | GPU 利用率、吞吐 | ❌ SM 空转 |
| spec decoding | 每 token 延迟（减前向次数） | ❌ SM 空转（绕过不消除） |
| **kernel 优化** | ✅ **接近硬件上限** | — |

> 这也解释 v7 为什么 lpm↔fcfs 吞吐几乎无差别——排序不改变"每步算多满"，真正的杠杆是 max-running-requests（v8 证明的主导 knob）。

---

## 阶段四：triton MoE kernel 根因分析

**问题**：研究现在用的 kernel（triton MoE），找提升 SM 利用率的空间。

### kernel 结构
`fused_moe_kernel`（`fused_moe_triton_kernels.py:324`）= 分组 GEMM，每个 block 算 `[BLOCK_M×BLOCK_N]`，沿 K 循环做 Tensor Core MMA（`tl.dot`）。性能由 tile 尺寸 + num_warps + num_stages 决定。

### ★ 根因：decode 走"小 tile"分支
`get_default_config`（`fused_moe_triton_config.py:193`）：`if M <= E → BLOCK_SIZE_M=16`。
- M = 每专家平均 token 数。实测 decode 时：LFM batch=32 M≈4；Qwen3 batch=32 M≈**2**。**永远 M≪E，100% 命中小 tile。**
- 后果：BLOCK_M=16 但只有 2–4 真 token → **75–87% 是 padding**（Tensor Core 算零）；tile 小 + 浅流水线 → occupancy 低、SM 空转 78%。

### 实锤：从未 tuning 过
bench 日志：**"Using default MoE kernel config... Config file not found at E=32,N=1792,device=H200.json"**。现有 tuned config 只有 fp8 的，**我们的 bf16 MoE 从没 tuning 过**。

### 关键问答：H200 最小 tile 限制（用户指出，修正了乐观）
- H200 bf16 **WGMMA 最小 M=64**；BLOCK_M=16 只能用老的 mma.sync（用不上最快的 WGMMA）。
- 但 decode 每专家仅 2–4 token：小 tile（M=16）用不上 WGMMA；大 tile（M=64）93% 是 padding。**两条路在极小 M 下都不好** → 单纯 tile tuning 天花板低。

### 优化空间（分级）
- **A. 生成 tuned tile config**：零风险、当天可做，但受 M 太小限制，天花板低。
- **B. MoE-decode 专用 kernel 形状**（更深 num_stages / swap_ab 减 padding）。
- **C. 换 GEMV 专用 kernel**（见下）：最高收益。
- **D. spec decoding 增大有效 M**（正交）。

---

## 阶段五：MoE decode 执行流程模拟 + 搬运 vs 计算（本质）

**问题**：GEMM vs GEMV 是什么？"搬运 experts"什么意思？搬运和计算各占多少时间？

### GEMM vs GEMV（机会 C 的核心）
- **GEMM（prefill）**：M 大（几千 token），权重读一次被几千行复用 → 算术强度高 → compute-bound → TC 吃满。
- **GEMV（decode）**：M≈1（每专家 2–4 token），权重读一次只被几行用 → 算术强度≈1 → memory-bound → TC 用不上。
- 现在把 decode 的 GEMV 硬塞进 GEMM kernel（pad 到 BLOCK_M）→ 算 padding 零。机会 C = 换 GEMV 专用 kernel，不 pad。

### 一步 MoE 执行流程（Qwen3, decode, batch=32）
权重常驻显存（58GB）。"搬运" = **HBM→SM**，非跨卡。
1. 32 token 到达 → 2. 路由：每 token 选 8 专家 = 256 配对，摊到 128 专家（平均每专家 2 token）→ 3. 按专家分组，pad 到 BLOCK_M=16 → 4. **逐个专家：从 HBM 搬 9.4MB 权重 → 只算 2 token（其余乘零）→ 丢弃 → 下一个**（一层 ~1.21GB）→ 5. 加权合并 → 6. 重复 48 层。
**一句话**：为给 32 token 各算一步，要把 58GB 专家权重整个从 HBM 读一遍。

### ★ 搬运 vs 计算的时间占比（核心数字）
两者**重叠**，总时间 ≈ max(搬运, 计算)。

**第一性原理（各自打满硬件）**：
| 模型 (decode 1步, b=32) | 搬 | 算 | 纯搬运 | 纯计算 | **搬:算** |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 58.0 GB | 0.12 TFLOP | 12.08 ms | 0.117 ms | **103 : 1** |
| LFM2.5-8B-A1B | 15.5 GB | 0.062 TFLOP | 3.23 ms | 0.063 ms | **52 : 1** |

**NCU 实测**：Qwen3 fused_moe DRAM 75% busy / SM 16% / No-Eligible 80%。

**结论**：**decode MoE 几乎 100% 时间在搬运（HBM→SM），计算 <1%（103:1）。** memory-bound 是本质，SM 空转是正常物理结果。

### ★ 修正优化目标
- ❌ "提升 SM 利用率"是错误目标（计算只需 1% 时间，没活给 SM 算）。
- ✅ 正确目标：**搬更快**（DRAM 75→90%，~1.2×）+ **搬更少**（消 padding）+ **搬一次服务更多 token**（增大有效 batch）。
- 真实 kernel headroom ≈ roofline 的 **1.5–1.9×**，不是"填满 SM"。

---

## 阶段六：跨 request 合并减少搬运——可行性 + MoE 专家覆盖曲线

**问题**：跨 request 合并减少搬运次数可行吗？

**回答**：可行，本质就是增大 batch（权重搬一次，batch 内所有 token 共享）。**sglang 的 continuous batching 已在做。** 但 MoE 有个特殊衰减：token 分散到 128 专家。

### 专家覆盖曲线（Qwen3, E=128, topk=8，球扔盒子模型）
| batch | 激活专家 | 搬(GB/层) | **每 token 分摊(MB)** |
|---|---|---|---|
| 1 | 8/128 | 0.073 | 73.5 |
| 8 | 51/128 | 0.477 | 59.6 |
| 32 | 111/128 | 1.046 | 32.7 |
| 64 | 126/128 | 1.186 | 18.5 |
| **128** | **128/128** | 1.208 | 9.4 |
| 256 | 128/128 | 1.208 | 4.7 |
| 512 | 128/128 | 1.208 | 2.4 |

**两个阶段**：
- **batch 1→128**：合并有收益但"边搬边涨"（batch 涨，激活专家也涨），每 token 73.5→9.4MB。
- **batch ≥128（专家全覆盖后）**：★ 真正免费——再加 batch，搬运量封顶（1.208GB 不变），token 翻倍 → 纯赚（9.4→4.7→2.4MB）。

**结论**：
1. 对 MoE，**真正的合并红利在专家全覆盖之后（Qwen3 batch≥128）**——之后每多合并一个请求几乎零额外搬运。
2. 受两个墙限制：**凑不齐**（serving idle，真实并发 6–20 ≪ 128 甜蜜点）；**甜蜜点前搬运还在涨**。
3. 这解释了为什么单流 agent 负载（batch 6–20）在 MoE 上效率最差——**既没喂饱 GPU，又没覆盖满专家，搬运摊销最差。**
4. **可落地策略**：可控攒批（batching delay）——稍等几 ms 攒够接近全覆盖的 batch 再跑，拿一点 TBT 换大幅搬运下降（sglang 有 prefill-delayer 类机制）。

---

## 全局结论：三层手段 vs 两个 gap（今天的总图）

| 层 | 手段 | 能碰的 gap | 今天的证据 |
|---|---|---|---|
| **serving** | 多租户/攒批/调度 | serving idle（GPU 忙不忙） | B2：利用率 2.5×；专家覆盖：batch≥128 甜蜜点 |
| **算法** | spec decoding | 每 token 延迟（减前向次数） | A1b：TPOT −23%（但 NCU 证明绕过而非消除 SM 空转） |
| **kernel** | tile tuning / GEMV 专用 kernel | **接近硬件上限**（搬运效率） | MoE 根因：小 tile+padding；103:1 搬算比；headroom 1.5–1.9× |

**核心论断（可上报）**：
1. decode MoE 是**彻底的 memory-bound**（搬:算 = 103:1），SM 空转是本质，**"提升 SM 利用率"是错误目标**。
2. 真正的杠杆是**减少/加速搬运**：跨 request 合并（batch≥128 甜蜜点）、消 padding、打满带宽。
3. scheduler policy（排序）碰不到硬件上限；能碰的是 batch 大小和 kernel 本身。
4. 两个 gap 正交：serving idle 靠 policy/负载，kernel 搬运效率靠 kernel 工程；spec decoding 是算法层的第三条路（减前向次数）。

---

## 附：今日产物
| 文档 | 内容 |
|---|---|
| `v11_realize_gap_results.md` | B2/A2/A1/A1b 干预实验 |
| `v12_ncu_spec_mechanism.md` | spec decoding No-Eligible 机制发现 |
| `scheduler_mechanism_and_agent_aware_idea.md` | 调度机制 + agent-aware idea |
| `triton_moe_kernel_analysis.md` | MoE kernel 根因 + 流程模拟 + 103:1 + 优化目标 |
| 本文件 | 今日完整对话逻辑链纪要 |

**数据源**：`consolidated_v9_ncu.csv`、`v9b_stall_analysis.csv`、`v11b2_multistream_util.csv`、`2026-07-15_v11a1b_ngram/`、`2026-07-15_v12_ncu_spec/`
