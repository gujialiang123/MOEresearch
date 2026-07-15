# Performance Opportunity Gap 综合分析报告

**目的**：把 v6→v12 所有能证明"优化机会 gap"的实测证据，串成完整的证据链 + 逻辑链 + 结论，供与 Chendi 讨论、并作为上报 Dey/Ofer 的 meeting 稿。**本轮升级**：在 v6→v10"gap 看得见"的基础上，补入 v11–v12 的**干预实验**，把 gap 从"看得见（potential）"升级到"摸得着（可回收）"，并诚实记录一个机制修正。
**日期范围**：2026-07-08 ~ 2026-07-15
**模型**：LFM2.5-8B-A1B、Qwen3-30B-A3B-Instruct-2507（均 bf16）
**硬件**：NVIDIA H200（132 SM，HBM3e 4.8 TB/s，bf16 TC ~989 TFLOP/s）
**负载**：真实 agent workload —— mooncake `toolagent` trace（FAST'25 公开）+ generated-shared-prefix

---

## 0. 一句话结论（TL;DR）

在**真实 agent 负载**、**已调到最优的 config** 下，我们用 NCU（kernel 硬件计数器）+ nsys（server 时间线）分层测出**两个正交的、config tuning 都够不到的 opportunity gap**，并用**干预实验**证明它们不只是"看得见"、而是"摸得着"：

1. **Serving idle**：真实单流下 GPU ~85% 墙钟在等请求 → 靠 **server policy / 加负载** 回收（吞吐可提 3–4×，但换更高 TBT）。**已实测可回收**：多路并发流 1→8，GPU 利用率 13%→32%（2.5×）、吞吐 7.4×（v11-B2）。
2. **Kernel SM idle**：decode 阶段 SM 有 67–78% 周期在空转等内存 → 靠 **kernel / 算法层** 回收（标准 roofline 显示热点 kernel 距 HBM 屋顶 1.3–2.4×）。**已实测可回收**：n-gram spec decoding 让主线模型 decode TPOT −23%（v11-A1b）；但 v12 NCU 揭示机制修正——spec **不降 SM 空转率**，而是**减少前向次数**（见证据 H）。

两者独立、不可互相替代；config tuning 已被证明触及不到任一。**逻辑链完整闭环**：会调 → 调到头（v8）→ gap 看得见（v9，NCU/nsys）→ gap 摸得着（v11–v12，干预）→ 定位下一步入口（已确认 kernel 层 78% SM 空转的攻坚点在 MoE decode，且 MoE decode 是 memory-bound）。

---

## 1. 逻辑链总览（从"怎么调"到"调不动了，gap 在哪"）

```
v6   建立方法学：NCU 能测真实 sglang kernel（单进程 bench_one_batch 技巧）
 │
v7   真实负载画像：agent 负载是 prefill 主导（in:out ≈ 9-13:1），与人工 regime 相反
 │
v8   在真实负载上 tuning 到头：max-running-requests 是主导 knob，cap=128 是拐点
 │        └─ 结论：config tuning 已用尽
 ▼
v9   最优 config 下测硬件利用率：occupancy 12-25%，decode kernel SM/DRAM 各 ~50%
v9b  空转归因：decode 内 SM 有 67-78% 周期 No-Eligible（等内存）
v9c  时间分解：单批 GPU 时间 ~40% prefill / ~60% decode
v9d  server idle 实测（nsys）：真实单流 GPU 利用率仅 14-19%
 │
v10  ① 加负载能回收多少（load sweep）② 用标准 roofline 重述 gap
 │
v11  干预实验：gap 从"看得见"→"摸得着"
 │    ├─ B2 多流：GPU 利用率 13%→32%、吞吐 7.4×  → serving idle 可回收 ✓
 │    ├─ A1b n-gram spec：主线模型 decode TPOT −23%  → kernel/算法层可回收 ✓
 │    └─ 负面对照：A2 fa3 已最优、A1 EAGLE3 draft 不匹配 → 收益需选对手段
v12  NCU 机制修正：spec decoding 不降 SM 空转率（77.5%→78.0%），
 │    而是减少前向次数 → SM 空转仍是纯 kernel 层未攻克的硬骨头
 ▼
结论：两个正交 gap，两个正交 lever，且都已实测可回收（附机制界定）
 │
下一步指向：kernel 层 78% SM 空转的攻坚点在 MoE decode——
     已确认 MoE decode 是 memory-bound（搬权重:算≈103:1），是后续工作的入口
```

---

## 2. 证据逐条（证据 → 数字 → 出处）

### 证据 A：config tuning 已经调到头（v8）
**做法**：真实 toolagent + shared_prefix 上网格扫 `chunked-prefill {2048/4096/8192/16384} × max-running-requests {32/64/128/192/256}`。
**结果**：
- 主导 knob 是 `max-running-requests`：cap 32→128 吞吐 +48~89%；固定 cap 下改 chunked <10%。
- cap=128 是拐点：扩到 192/256 吞吐仅 ±1–2%。
- 之前在**人工合成 regime** 上调出的"最优"（flashinfer_cutlass+fcfs）在**真实负载**上反而最差（Qwen3 toolagent 5.14 vs 7.6 req/s）。
**结论**：**config 层的优化空间已耗尽**，且合成 regime 的调参不迁移到真实负载。
**出处**：`docs/2026-07-09/v8_tuning_on_real_workload.md`、`results/consolidated_v8_tuning.csv`（56 行）

### 证据 B：最优 config 下，硬件远未吃满（v9，NCU 实测）
**做法**：最优 config + 真实 agent 代表点（in≈2700/out≈207），NCU 测 prefill+decode 的每 kernel SM%/DRAM%/occupancy。
**结果（decode b=32，热点 kernel）**：
| 模型 | kernel | SM% | DRAM% | Occupancy% |
|---|---|---|---|---|
| LFM2.5 | flash_attn | 45.8 | 45.1 | 24.9 |
| LFM2.5 | nvjet_gemm | 10.1 | 68.1 | 14.2 |
| Qwen3 | flash_attn | 46.6 | 71.9 | 18.5 |
| Qwen3 | fused_moe | 16.1 | 75.3 | 37.2 |
**结论**：
- Occupancy 普遍 **12–25%** → Hopper warp 槽 75–88% 空闲（config 改不了 occupancy）。
- decode 关键 kernel SM 和 DRAM **都没到顶**（flash_attn 两维都 ~46%）→ 卡在延迟/stall。
- 最高 DRAM 仅 ~76%，离 HBM 峰值差 24%。
**出处**：`docs/2026-07-10/v9_ncu_hardware_ceiling_evidence.md`、`results/consolidated_v9_ncu.csv`（276 行）

### 证据 C：decode 的低利用率来自 SM 空转（v9b，NCU 调度器实测）
**做法**：重跑 NCU 加 `WarpStateStats + SchedulerStats`，取 **"No Eligible"** = 调度器无可发射 warp 的周期占比 = SM 真正空转。
**结果**：
| 模型 | kernel | SM%(平均) | No Eligible%(空转) |
|---|---|---|---|
| LFM2.5 | flash_attn | 44 | **50** |
| LFM2.5 | nvjet_gemm | 9.5 | **90** |
| Qwen3 | flash_attn | 44.6 | **68** |
| Qwen3 | fused_moe | 15.9 | **80** |

**时间加权**：decode GPU 时间里 **LFM 67% / Qwen3 78% 是 SM 空转**（在等内存/依赖）。
**结论**：低 SM% 不是"一直半忙"，而是"一半满载发射 + 一半完全空转"。根因 = occupancy 低，驻留 warp 太少，无法隐藏内存延迟。**这是 kernel 层可回收的空转，config/加负载都碰不到。**
**出处**：`docs/2026-07-10/v9b_walltime_and_stall_analysis.md`、`results/v9b_stall_analysis.csv`

### 证据 D：server idle 实测（v9d，nsys 时间线）
**做法**：nsys 用 delay/duration 精确圈住 bench 窗口，录 server 进程 GPU 时间线；GPU busy = 内核+memcpy 区间**并集**。真实到达（slowdown 1.0，200 请求）。
**结果**：
| 模型 | serving | 实测并发 | GPU busy | **GPU 利用率** | **server idle** |
|---|---|---|---|---|---|
| LFM2.5 | 38.1s | 6.2 | 5.36s | 14% | **86%** |
| Qwen3-30B | 43.3s | 19.7 | 8.03s | 19% | **81%** |
**结论**：真实单流 agent 负载下，GPU 大部分墙钟在等请求。原因：真实到达并发只有 6–20（vs server 容量 ~155），加上 toolagent 的 prefix 共享（radix cache）缩小了 prefill 计算。**这是负载/部署问题，不是 kernel/硬件 gap。**
**出处**：`docs/2026-07-10/v9b_walltime_and_stall_analysis.md`（Q4）、`results/v9d_server_idle_measured.csv`

### 证据 E：加负载能回收多少 serving idle（v10，load sweep）
**做法**：固定最优 config，客户端扫 offered concurrency 8→256。
**结果**：
| offered | LFM 吞吐(tok/s) | LFM TPOT | Qwen3 吞吐 | Qwen3 TPOT |
|---|---|---|---|---|
| 8 | 1663 | 4.0ms | 739 | 9.5ms |
| 64 | 4684 | 12.4ms | 1859 | 30.6ms |
| 256 | **7309** | 33.4ms | **2416** | 65.8ms |
**结论**：
- 加负载确实回收 serving idle 换来的吞吐：LFM **+4.4×**、Qwen3 **+3.3×**（8→256）。
- 但 **TBT/TPOT 单调恶化**（LFM 8.4×、Qwen3 6.9×），且收益递减、TTFT 在 256 急升。
- **吞吐-延迟不可兼得**，最优工作点取决于 SLA。
**出处**：`docs/2026-07-14/v10_loadsweep_and_roofline.md`、`results/consolidated_v10_load_sweep.csv`

### 证据 F：kernel gap 的 roofline 量化（v10，标准 roofline）
**做法**：用标准 roofline（Williams 2009）重述 kernel gap。achieved BW = 实测 DRAM% × 4.8 TB/s。
**结果（per-kernel，decode b=32）**：
| 模型 | kernel | achieved BW | 距 HBM 屋顶 |
|---|---|---|---|
| LFM2.5 | flash_attn | 2.17 TB/s | **2.2×** |
| Qwen3 | fused_moe | 3.61 TB/s | **1.3×** |
**时间加权整步距屋顶**：LFM **2.37×** / Qwen3 **1.87×**（纯实测，不依赖 MoE 假设）。
**结论**：decode 热点 kernel 全 memory-bound，距带宽屋顶 1.3–2.4×，即 kernel 优化的理论上界。此数字与我们早前自造的 "TBT headroom" 完全吻合，现用标准 roofline 表述后可信度更高。
**出处**：`docs/2026-07-14/v10_loadsweep_and_roofline.md`、`scripts/compute_v10_roofline.py`

### 证据 F2：roofline 的 compute 轴——achieved GFLOP/s（Chendi 要求补测）
**做法**：补上 roofline 的另一条轴。用 NCU 直接采 **tensor-core FLOP 计数器**（`sm__ops_path_tensor_op_hmma/hgmma_*` = 真实 tensor 运算数）+ 非 tensor FP 指令，除以 kernel 时长，得到每 kernel 已达到的算力吞吐。H200 bf16 tensor-core 峰值 = **989.5 TFLOP/s**。
**结果（Qwen3-30B decode，★真 FLOP 计数器，含 fused_moe）**：
| 点 | kernel | 时长 | SM% | **achieved 算力** | **占 bf16 峰值** |
|---|---|---|---|---|---|
| **b32** | flash_attn | 141 µs | 34.8 | 338 TFLOP/s | 34.2% |
| **b32** | **fused_moe（MoE GEMM）** | 119 µs | 17.7 | **68.9 TFLOP/s** | **7.0%** |
| **b32** | **整步（时间加权）** | — | — | **169.9 TFLOP/s** | **17.2%** |
| **b64** | flash_attn | 235 µs | 44.6 | 405 TFLOP/s | 40.9% |
| **b64** | **fused_moe（MoE GEMM）** | 84 µs | 15.9 | **75.4 TFLOP/s** | **7.6%** |
| **b64** | **整步（时间加权）** | — | — | **258.9 TFLOP/s** | **26.2%** |

**其余 decode 点（LFM2.5，tensor-pipe%外推估计；LFM 在 NCU 下加载挂起，无法精确重测）**：
| 模型 | 点 | 整步 achieved | 占峰值 |
|---|---|---|---|
| LFM2.5-8B | decode b32 | ~184 TFLOP/s | ~19% |
| LFM2.5-8B | decode b64 | ~274 TFLOP/s | ~28% |

**结论**：
- decode 整步算力仅 **17.2%（b32）/ 26.2%（b64）** 的 bf16 峰值——**从 compute 轴独立印证 memory-bound**（与证据 F 的"距带宽屋顶 1.3–2.4×"互为佐证：算力空、带宽近满 = 典型 memory-bound）。
- **MoE GEMM（`fused_moe`）算力最低，仅 7.0–7.6% 峰值**（两个 batch 一致）——直接量化"瓶颈在搬专家权重而非算力"，是 §5-11 "MoE decode 是 memory-bound" 结论的 compute-轴硬证据。
- batch 32→64 整步算力占比升高（17%→26%），主要来自 attention（34%→41%）；**MoE GEMM 几乎不随 batch 改善（7.0%→7.6%）**——因为 decode MoE 每 token 各选专家、batch 内难凑成大 GEMM，这正是"搬权重主导"的表现。
**方法说明（诚实标注）**：Qwen3 两个 decode 点用**真 FLOP 计数器**（`sm__ops_path_tensor_op_hmma/hgmma_*` + 非 tensor FP 指令，严谨值）；LFM 两点用 tensor-pipe 活跃%外推（NCU roofline compute 轴的标准代理，略偏乐观）——尝试对 LFM 做真 FLOP 精确重测时，LFM2.5 混合模型在 NCU 下加载挂起（已知 mamba/conv 混合架构 + NCU 兼容问题），故 LFM 保留代理值。**头条结论（decode memory-bound、MoE GEMM ~7% 峰值）已由 Qwen3 精确值坐实**，LFM 仅作趋势佐证。
**出处**：`scripts/parse_v18_ncu_long.py`（精确）、`scripts/compute_v18_gflops.py`（估计）、`results/2026-07-15_v18_gflops/gflops_accurate.json`、`gflops_estimate.json`

---

## 2b. 干预证据：gap 不只是"看得见"，而是"摸得着"（v11–v12）

前面 A–F 证明 gap **看得见**（potential，用 profiler 观测到）。下面用 **config 以外的干预手段**，实测某指标真的朝预测方向移动，把 gap 升级为 **摸得着**（可回收）。

### 证据 G：serving idle 可回收（v11-B2，多路并发流，★强证据）
**逻辑**：v9d 测到真实单流 GPU idle 86%。若 idle 真是"负载不足"造成，则同时跑 N 条独立真实到达流（模拟多租户），利用率应随 N 单调上升。
**做法**：一个 server（最优 config，max-running 256），N=1/2/4/8 条并发 toolagent 流（真实到达，各 200 请求，不同 seed），nsys 测 GPU busy（内核+memcpy 并集）。
**结果（LFM2.5）**：
| 并发流数 | GPU 利用率 | 合计吞吐 |
|---|---|---|
| 1 | **13%** | 1087 tok/s |
| 2 | 18% | 2163 tok/s |
| 4 | 25% | 4217 tok/s |
| 8 | **32%** | 8062 tok/s |
**结论**：利用率单调 13%→32%（**2.5×**）、吞吐 **7.4×**；streams=1 的 13% 复现 v9d 的 14%（方法自洽）。**直接证明 serving idle 是负载不足造成、可被多租户/多流回收**。注：8 流仍仅 32%，单条 toolagent 流很稀疏，填满 H200 需更多路并发（趋势明确，未饱和）。
**出处**：`docs/2026-07-15/v11_realize_gap_results.md`、`results/v11b2_multistream_util.csv`

### 证据 H：kernel/算法层可回收 + 机制修正（v11-A1b + v12，★重要且诚实）
**逻辑**：decode 的 SM 空转（67–78%）根因是"每步只算 1 个 token，权重读一遍只服务一个 token"。spec decoding（exact）让一次前向验证多 token → 应降每 token 延迟。
**做法**：主线模型 + 真实 toolagent，基线 vs n-gram spec decoding（exact，无需 draft 模型），测 server 端 decode TPOT。
**结果（A1b，主线模型）**：
| 并发 | 基线 TPOT | n-gram TPOT | 变化 | Accept length |
|---|---|---|---|---|
| 32 | 18.64 ms | 14.27 ms | **−23%** | 2.08 |
→ decode TPOT 降 23%（exact，不损精度）；conc=32 收益 > conc=1，与"batch 大 SM 空转多"诊断自洽。

**机制修正（v12，NCU 实测）—— 必须诚实标注**：
| 变体 | 时间加权 No-Eligible（SM 空转） | SM 利用 |
|---|---|---|
| baseline | 77.5% | 21.1% |
| n-gram spec | 78.0% | 19.8% |

**spec decoding 并不降低单 kernel 的 SM 空转率**（77.5%→78.0% 基本不变）。它降 TPOT 不是靠"让每个 kernel 更满"，而是靠**减少每 token 需要的前向次数**（一次验证 ~2 个 token → 前向次数减半）。
- **含义**：SM 空转（78%）是 decode memory-bound 的根本属性，spec decoding **绕过**而非**消除**它。要真正吃掉这块空转，需要**更高 occupancy 的 decode kernel**——这是纯 kernel 层尚未攻克的硬骨头。
**出处**：`docs/2026-07-15/v11_realize_gap_results.md`（A1b）、`docs/2026-07-15/v12_ncu_spec_mechanism.md`

### 证据 I：两个负面对照（诚实记录，本身有价值）
- **A2 attention backend**：换 fa3→triton，TBT 差 18%，但 **fa3 已是三种实现里最快**（flashinfer 起不来）→ 用现成 backend 无法证明 kernel 还有可回收空间（已用最优实现）。
- **A1 EAGLE3 spec**：现成 redhat EAGLE3 draft 与我们的 Qwen3-32B 权重不配套，**accept length 仅 1.28**（理想 2–4），反而更慢 → 用不匹配 draft 无法验证 spec 收益。
**价值**：两个负面结果印证"kernel 层收益不是随手换个现成组件就能拿到"，需要匹配的 draft / 定制 kernel / 算法级手段——支撑"kernel 层是真正深水区"，也印证 autotuner 必须先做兼容性/有效性裁剪。
**出处**：`docs/2026-07-15/v11_realize_gap_results.md`（A2、A1）

---

## 3. 完整时间预算（把所有证据拼成一张图）

以 LFM2.5、真实单流、serving 38.1s 为例（全实测）：
```
serving 墙钟 38.1s：
├─ ① server idle（GPU 没活，等请求）      32.7s  86%   → server policy（证据 D,E）
└─ ② GPU busy                             5.4s   14%
    ├─ decode SM 空转（No-Eligible 67%）  ~2.0s  ~5%   → kernel 优化（证据 C,F）
    └─ 真正在算                           ~3.4s  ~9%
```
**饱和场景（server 喂满，idle≈0）下**，decode 的 SM 空转仍占 GPU 计算时间的 **39%(LFM) / 46%(Qwen3)** —— 即使把 server idle 完全消除，这块 kernel 层浪费依然存在。

---

## 4. 两个 gap、两个 lever（核心结论）

| | Serving idle | Kernel SM 空转 |
|---|---|---|
| **是什么** | GPU 没活，在等请求 | GPU 在跑 kernel，但 SM 空转等内存 |
| **实测量** | 真实单流占墙钟 86%/81%（nsys） | decode GPU 时间 67%/78%（NCU No-Eligible） |
| **根因** | 请求到达率低（真实并发 6–20 vs 容量 155） | occupancy 低（12–25%），内存延迟未隐藏 |
| **回收手段** | server policy / 加负载 / 攒批 / 多租户 | ① 算法层：spec decoding（减前向次数）；② 纯 kernel：提 occupancy / 隐藏延迟 / 更大 tile |
| **已实测可回收？** | ✅ v11-B2：多流 1→8，利用率 13%→32%（2.5×）、吞吐 7.4× | ⚠️ 部分：算法层 ✅ v11-A1b spec TPOT −23%；纯 kernel（吃掉 78% SM 空转）❌ 尚未攻克 |
| **机制界定（v12）** | 多流真的减少 idle（利用率上升） | spec decoding **不降 SM 空转率**（77.5%→78.0%），靠**减少前向次数**降 TPOT；SM 空转需更高 occupancy kernel 才能消除 |
| **回收上界** | 吞吐 3–4×（但 TBT 恶化，SLA 权衡） | TBT ~1.8–2.4×（roofline，exact 方法） |
| **另一个 lever 有用吗** | kernel 优化对它无效 | 加负载对它无效 |
| **算不算硬件 gap** | ❌ 部署/负载选择 | ✅ kernel/硬件层的真实 gap |

---

## 5. 与 Chendi 讨论：哪些适合上报 Dey/Ofer

### 建议**明确上报**（证据硬、方法标准、结论稳健）
1. **方法学本身**：真实负载 + 最优 config + NCU/nsys 分层测量 —— 这是"我们怎么测 opportunity gap"的答案，本身就是交付物。
2. **两个正交 gap 的框架**：serving idle（policy）vs kernel SM idle（kernel）—— 清晰、可行动。
3. **config tuning 已到头**（v8）：cap=128 拐点、合成 regime 不迁移 —— 有力支撑"需要往 kernel/policy 层走"。
4. **kernel gap 的 roofline 量化**：decode 热点 kernel 距 HBM 屋顶 1.3–2.4× —— 标准框架，Chendi 认可的语言。
5. **load sweep 的吞吐-延迟权衡曲线**：回答"加负载能提升多少"，且点明 SLA 权衡。
6. **gap 摸得着的干预证据**（本轮新增，最强升级）：
   - serving idle 可回收——v11-B2 多流实测利用率 13%→32%（2.5×）、吞吐 7.4×；
   - 算法层可回收——v11-A1b n-gram spec decoding 让主线模型 decode TPOT −23%（exact，不损精度）。
7. **诚实的机制修正**（v12，反而增强可信度）：spec decoding **不消除** SM 空转，而是**减少前向次数**；纯 kernel 层的 78% SM 空转仍是未攻克的硬骨头。主动讲清楚"什么手段解决什么问题、什么还没解决"，比夸大更能取信。

### 建议**谨慎/加 caveat**（对假设敏感）
8. **端到端 first-principles floor（batch>1）**：对 MoE 专家激活假设高度敏感（Qwen3 batch=32 甚至算出 <1×，不合理）。**只用 batch=1 展示"小 batch 浪费带宽"**，batch>1 一律以实测 per-kernel roofline 为准。
9. **"TBT 可提升 ~2×" 这个绝对数字**：是 roofline **上界**（假设打满带宽屋顶），真实 kernel 达不到 100%。上报时须标注"上界、exact 方法、不含量化/投机解码"。
10. **两个负面对照**（A2 fa3 已最优、A1 EAGLE3 draft 不匹配）：可作为"为何 kernel 层是深水区"的支撑，但须标注是负面结果、不是失败。

### 建议**作为"下一步入口"点到为止**（不展开）
11. **kernel gap 的攻坚入口 = MoE decode**：v9/v12 已定位 decode 的 78% SM 空转是纯 kernel 层硬骨头；进一步测出 **MoE decode 是 memory-bound**（搬权重:算 ≈ 103:1），说明瓶颈在"搬专家权重"而非算力。**本稿到此为止**——具体怎么优化（减激活专家 / 批内聚集 / 量化等）是后续独立工作，不在本次 meeting 结论内。

### 建议**暂不上报 / 内部保留**
8. 早期自造的 "TBT headroom" 命名 —— 已被标准 roofline 取代，避免再提这个非标准词。
9. eager 模式 vs cudagraph 的 TBT 绝对值差异 —— 内部技术细节，比例结论不受影响即可。

---

## 6. 数据与可信度声明

| 数字 | 实测 or 估算 | 工具 |
|---|---|---|
| config sweep 吞吐/延迟 | 实测 | sglang bench_serving |
| kernel SM%/DRAM%/occupancy | 实测 | NCU 硬件计数器 |
| No-Eligible（SM 空转） | 实测 | NCU SchedulerStats |
| prefill/decode 时间分段 | 实测 | sglang bench_one_batch（cudagraph） |
| server idle | 实测 | nsys 时间线（内核并集） |
| load sweep 曲线 | 实测 | sglang bench_serving |
| 多流利用率回收（v11-B2） | 实测 | nsys 时间线（多路流并集） |
| spec decoding TPOT −23%（v11-A1b） | 实测 | sglang bench_serving（n-gram，exact） |
| spec 不降 SM 空转（v12） | 实测 | NCU SchedulerStats（baseline vs n-gram） |
| roofline 距屋顶（memory 轴） | 实测 DRAM% 外推 | NCU + roofline 公式 |
| achieved GFLOP/s（compute 轴，v18；Qwen b32+b64） | **实测**（真 FLOP 计数器） | NCU sm__ops_path_tensor_* |
| achieved GFLOP/s（LFM 2 点） | **代理估算**（tensor-pipe%×峰值；LFM 在 NCU 下挂起，无法精确重测） | NCU tensor-pipe active% |
| 端到端 floor（batch>1） | **估算**（MoE 敏感） | 第一性原理，仅供参考 |
| MoE decode memory-bound（搬:算≈103:1） | 实测 | HF forward 结构分析 + roofline |

**关键局限**：
- roofline 距屋顶是**理论上界**（假设 kernel 能打满带宽屋顶）；真实优化拿回的 < 此值。
- server idle 取决于负载；多用户/攒批可回收，非硬件极限。
- 所有 kernel 数据基于 NCU 每 kernel 抓 24 个 launch，未必覆盖一步全部 layer；但每 kernel 的利用率/空转是其自身实测值，不受影响。

---

## 附：全部产物索引
| 版本 | 内容 | 文档 | 数据 |
|---|---|---|---|
| v6 | NCU 方法学 + Qwen3/LFM decode | `docs/2026-07-08/v6_ncu_sglang_experiment_report.md` | `consolidated_v6_sglang_ncu.csv` |
| v7 | 真实负载画像 + config sweep | `docs/2026-07-09/v7_*.md` | `consolidated_v7_*.csv` |
| v8 | 真实负载 tuning | `docs/2026-07-09/v8_tuning_on_real_workload.md` | `consolidated_v8_tuning.csv` |
| v9 | 最优 config 下 NCU gap | `docs/2026-07-10/v9_ncu_hardware_ceiling_evidence.md` | `consolidated_v9_ncu.csv`, `v9_tbt_headroom.csv` |
| v9b/c/d | 空转 + 时间分解 + server idle | `docs/2026-07-10/v9b_walltime_and_stall_analysis.md` | `v9b_stall_analysis.csv`, `v9d_server_idle_measured.csv` |
| v10 | load sweep + roofline | `docs/2026-07-14/v10_loadsweep_and_roofline.md` | `consolidated_v10_load_sweep.csv` |
| v11 | 干预：B2 多流 / A1b n-gram spec / A2 backend / A1 EAGLE3 | `docs/2026-07-15/v11_realize_gap_results.md` | `v11b2_multistream_util.csv`, `results/2026-07-15_v11*` |
| v12 | NCU 机制修正：spec 不降 SM 空转 | `docs/2026-07-15/v12_ncu_spec_mechanism.md` | `results/2026-07-15_v12_ncu_spec/` |
| （下一步入口） | 定位 kernel 攻坚点：MoE decode 是 memory-bound（搬:算≈103:1） | `docs/2026-07-15/triton_moe_kernel_analysis.md` | `results/2026-07-15_v13_router/` |
