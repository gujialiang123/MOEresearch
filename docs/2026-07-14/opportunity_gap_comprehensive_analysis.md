# Performance Opportunity Gap 综合分析报告

**目的**：把 v6→v10 所有能证明"优化机会 gap"的实测证据，串成完整的证据链 + 逻辑链 + 结论，供与 Chendi 讨论、筛选哪些结论适合上报 Dey/Ofer。
**日期范围**：2026-07-08 ~ 2026-07-14
**模型**：LFM2.5-8B-A1B、Qwen3-30B-A3B-Instruct-2507（均 bf16）
**硬件**：NVIDIA H200（132 SM，HBM3e 4.8 TB/s，bf16 TC ~989 TFLOP/s）
**负载**：真实 agent workload —— mooncake `toolagent` trace（FAST'25 公开）+ generated-shared-prefix

---

## 0. 一句话结论（TL;DR）

在**真实 agent 负载**、**已调到最优的 config** 下，我们用 NCU（kernel 硬件计数器）+ nsys（server 时间线）分层测出**两个正交的、config tuning 都够不到的 opportunity gap**：

1. **Serving idle**：真实单流下 GPU ~85% 墙钟在等请求 → 靠 **server policy / 加负载** 回收（吞吐可提 3–4×，但换更高 TBT）。
2. **Kernel SM idle**：decode 阶段 SM 有 67–78% 周期在空转等内存 → 靠 **kernel 优化** 回收（标准 roofline 显示热点 kernel 距 HBM 屋顶 1.3–2.4×）。

两者独立、不可互相替代；config tuning 已被证明触及不到任一。

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
 ▼
结论：两个正交 gap，两个正交 lever
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
| **回收手段** | server policy / 加负载 / 攒批 / 多租户 | kernel 优化（提 occupancy、隐藏延迟、更大 tile） |
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

### 建议**谨慎/加 caveat**（对假设敏感）
6. **端到端 first-principles floor（batch>1）**：对 MoE 专家激活假设高度敏感（Qwen3 batch=32 甚至算出 <1×，不合理）。**只用 batch=1 展示"小 batch 浪费带宽"**，batch>1 一律以实测 per-kernel roofline 为准。
7. **"TBT 可提升 ~2×" 这个绝对数字**：是 roofline **上界**（假设打满带宽屋顶），真实 kernel 达不到 100%。上报时须标注"上界、exact 方法、不含量化/投机解码"。

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
| roofline 距屋顶 | 实测 DRAM% 外推 | NCU + roofline 公式 |
| 端到端 floor（batch>1） | **估算**（MoE 敏感） | 第一性原理，仅供参考 |

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
