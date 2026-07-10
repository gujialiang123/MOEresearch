# v9 实验报告：证明"单靠 tuning 不足以达到硬件上限"（真实负载 + NCU）

**日期**：2026-07-10
**执行**：双卡并行 —— GPU 1（LFM2.5）+ GPU 2（Qwen3-30B），Nsight Compute `2026.2.1`（sudo）
**脚本**：`scripts/run_v9_ncu_realworkload.py`
**产出**：`results/2026-07-10_v9_ncu_realworkload/`、`results/consolidated_v9_ncu.csv`
**用时**：LFM 55 min / Qwen3 72 min（并行），6 combo（2 模型 × 3 regime），零失败

---

## 1. 目的与逻辑

前面几轮已经确认：v7/v8 在真实负载上把 config tuning 做到了头（cap=128 是拐点，再调无收益）。**本轮要回答的是：把最优 config 用上之后，硬件到底吃满了没有？如果没吃满，就说明剩下的 gap 不是 config 能解决的，只能靠 kernel 优化。**

方法（v6 的单进程 NCU 口径）：
- `sglang.bench_one_batch`（单进程）+ `--profile-activities CUDA_PROFILER`
- `sudo ncu --profile-from-start off` 只测被 bench 的那一段
- **config = v8 tuned winner**（LFM chunked=4096、Qwen3 chunked=16384，均 triton MoE）→ 保证是在"最优 config"下测的
- **代表点来自真实 toolagent 画像**（v7）：input≈2700、output≈207
- NCU section 加了 `MemoryWorkloadAnalysis`，详细记录 memory 指标

## 2. 跑的 regime（真实 agent 代表点）

| regime | batch | input | output | 阶段 |
|---|---|---|---|---|
| agent_prefill_b1 | 1 | 2700 | 8 | prefill |
| agent_decode_b32 | 32 | 2700 | 32 | decode |
| agent_decode_b64 | 64 | 2700 | 32 | decode |

## 3. 显存占用（server 日志实测，H200 143GB）

| 模型 | 权重(bf16) | KV 池 | KV 容量 |
|---|---|---|---|
| LFM2.5-8B-A1B | 16.34 GB | 53.6 GB | 468 万 token |
| Qwen3-30B-A3B | 57.02 GB | 61.1 GB | 66.7 万 token |

---

## 4. 结果：热点 kernel 的硬件利用率（v8 最优 config 下）

指标：SM%（算力利用）、DRAM%（HBM 带宽利用）、Mem%（Memory pipe）、Occ%（实测 occupancy）、L2%。

### prefill 段（agent, in=2700, b=1）
| 模型 | kernel | SM% | DRAM% | Occ% |
|---|---|---|---|---|
| LFM2.5 | nvjet_gemm | 86.2 | 17.4 | **14.7** |
| LFM2.5 | act_and_mul | 42.0 | 78.3 | 73.9 |
| Qwen3 | fused_moe | 66.5 | 60.2 | **12.5** |
| Qwen3 | nvjet_gemm | 85.4 | 17.2 | **14.6** |
| Qwen3 | moe_sum | 21.6 | 77.3 | 41.0 |

### decode 段（agent, in=2700）
| 模型 | regime | kernel | SM% | DRAM% | Occ% |
|---|---|---|---|---|---|
| LFM2.5 | b32 | flash_attn | 45.8 | 45.1 | 24.9 |
| LFM2.5 | b32 | nvjet_gemm | 10.1 | 68.1 | 14.2 |
| LFM2.5 | b64 | flash_attn | 48.5 | 46.5 | 25.1 |
| LFM2.5 | b64 | nvjet_gemm | 19.1 | 64.3 | 13.7 |
| Qwen3 | b32 | flash_attn | 46.6 | 71.9 | 18.5 |
| Qwen3 | b32 | fused_moe | 16.1 | 75.3 | 37.2 |
| Qwen3 | b64 | flash_attn | 52.3 | 75.8 | 18.9 |
| Qwen3 | b64 | fused_moe | 12.7 | 59.7 | 23.6 |

---

## 5. 结论：单靠 tuning 达不到硬件上限（三条硬证据）

**即使用上 v8 tuning 出来的最优 config，热点 kernel 仍系统性地远离硬件天花板：**

1. **Occupancy 普遍只有 12–25%。**
   所有主导 GEMM（nvjet / fused_moe）的实测 occupancy 都在 12–15%，注意力 kernel 也只有 18–25%——也就是 Hopper 的 warp 调度槽有 **75–88% 是空的**。occupancy 是 kernel 的 launch/寄存器/tile 配置决定的，**config knob（batch、chunked、schedule）改不了它**。这是 tuning 触及不到的层面。

2. **decode 的关键 kernel SM% 和 DRAM% 都没到顶。**
   - LFM2.5 decode 最热的 flash_attn：**SM 48% / DRAM 46%**——既没被算力卡、也没被带宽卡，卡在延迟/依赖 stall 上。这是"gap 不在 config"的最直接信号：两个硬件维度都还剩一半。
   - 最高的 DRAM 也只到 **~76%**（Qwen3 decode flash_attn/fused_moe），离 HBM 峰值还差 ~24%。

3. **prefill 的 MoE 也没吃满。**
   Qwen3 prefill 的 fused_moe：SM 66% / DRAM 60% / Occ 12.5%——占用率极低，算力和带宽都只用了六成。大 GEMM（nvjet）虽然 SM 85%，但 occupancy 只有 14.6%，说明是靠少数满载的 wave 撑起来的，仍有结构性浪费。

**总结**：v7/v8 已经把 config 调到最优（cap=128、chunked 各自最佳），但 NCU 显示 **occupancy 12–25%、decode 关键 kernel SM/DRAM 各 ~50%、峰值 DRAM 仅 ~76%**。这些 gap 是 **kernel 层**的（occupancy、tile 大小、访存 pattern、stall），config tuning 无法触及。**→ 要继续逼近硬件上限，必须做 kernel 级优化（更高 occupancy 的 MoE/attention kernel、更好的 tiling），而不是继续调 config。**

---

## 6. 下一步

1. 对最卡的 kernel（LFM decode flash_attn、Qwen3 decode fused_moe）用 `--set full` 深挖 stall 原因（`--section WarpStateStats / SchedulerStats`），定位到底是 long-scoreboard（访存等待）还是 barrier/依赖。
2. 把"achieved DRAM GB/s vs HBM 峰值"算成绝对数字，量化 hardware-layer gap 的大小。
3. 用这批证据支撑 mentor 讨论："config autotuner 的天花板"与"需要 kernel 层介入"的分界。

## 7. 回答 Dey 的问题："每个 regime 在最优 config 下，TBT 还能再改进多少？"

TBT（time between tokens）= decode 阶段每步时间。用 v9 的 kernel 级数据可以给出一个**有依据的上界**：

**方法（roofline，仅限 exact 方法、不改字节数/算法）**：每个 kernel 若能把它最忙的那条 pipe（`max(SM%, Memory Throughput%)`）打满到 100%，时间可压到 `dur × busiest_util%`。按 duration 加权汇总，得到"整个 decode step 的时间加权 busiest-pipe 利用率"，其倒数就是 TBT 的**最大可改进倍数**。

| 模型 | regime | 实测 TBT（eager+NCU） | 时间加权 busiest-pipe | **TBT 最多再快** | 可压掉 |
|---|---|---|---|---|---|
| LFM2.5 | decode b32 | 29.7 ms | 42.3% | **~2.4×** | ~58% |
| LFM2.5 | decode b64 | 45.0 ms | 44.6% | **~2.2×** | ~55% |
| Qwen3-30B | decode b32 | 78.8 ms | 53.5% | **~1.9×** | ~47% |
| Qwen3-30B | decode b64 | 85.3 ms | 55.8% | **~1.8×** | ~44% |

**怎么读**：即使选了最优 config，LFM2.5 的 decode step 里"最忙的硬件资源"平均也只用了 ~43%，所以 TBT 理论上还能再快 ~2.4×；Qwen3 因为 fused_moe/attention 把带宽压得更高（busiest ~54%），headroom 小一些（~1.8×）。

**三条必须说明的 caveat**：
1. 这是**上界**：假设 kernel 优化能把最忙 pipe 打到 100%（占用率、tiling、访存 pattern 全部理想化）。真实 kernel 达不到 100%，所以实际可改进量**小于**这个数。
2. **只算 exact 方法**（同样的字节/FLOP，只提升硬件利用），符合 Dey"不引入量化等近似"的约束。若允许量化/投机解码，headroom 会更大（另算）。
3. 实测 TBT 绝对值是 `bench_one_batch`+NCU 的 **eager 模式**（profiling 期间关 cudagraph），比线上带 cudagraph 的 TBT 偏高；**headroom 比例**（倍数）不受此影响，是稳健的部分。

**一句话答复 Dey**：可以回答。用最优 config 时，decode 的 TBT 仍有 headroom——LFM2.5 约 **2.2–2.4×**、Qwen3-30B 约 **1.8–1.9×**（即分别可压掉 ~55–58% / ~44–47%），这是 exact-method、roofline 上界；这些 headroom 全在 kernel 层（occupancy 12–25%、busiest pipe 只有 ~43–56%），config tuning 拿不到。

## 附：产物
- `results/2026-07-10_v9_ncu_realworkload/<model>/<regime>/` — 每个含 ncu.ncu-rep + ncu_raw.csv（含 MemoryWorkloadAnalysis）+ bench.log
- `results/consolidated_v9_ncu.csv` — 276 行（每 kernel 的 SM/DRAM/Mem/Occ/L2/Duration/MaxBW）
- `results/v9_tbt_headroom.csv` — 每 regime 的 TBT headroom 汇总
- `scripts/run_v9_ncu_realworkload.py`
