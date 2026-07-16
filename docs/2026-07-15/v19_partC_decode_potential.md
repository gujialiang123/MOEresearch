# v19 Part C — Decode optimization potential & gaps (NCU 11-metric analysis)

**日期**：2026-07-15
**目的**：回答 Chendi 的问题 —— **decode 优化能拿多少收益、gap 在哪**。基于 Part A（wall 占比）+ Part B（NCU 11 指标，4 个 regime）。
**数据**：
- Part A CSV：`results/2026-07-15_v19_wall_sweep/qwen3-30b-a3b-bf16/wall_proportion.csv`
- Part B NCU：`results/2026-07-15_v19b_ncu_decode/qwen3-30b-a3b-bf16/{agent_decode_b32,b64,b128,agent_prefill_b1}/`（`ncu.ncu-rep` + `ncu_raw.csv`）
- 汇总：`results/2026-07-15_v19b_ncu_decode/qwen3-30b-a3b-bf16/ncu_summary.json`
- 脚本：`scripts/run_v19_wall_sweep.sh`, `scripts/run_v19b_ncu_decode.py`, `scripts/parse_v19b_ncu.py`

---

## 1. Decode 主导 wall（Part A）
Agent（toolagent）workload、Qwen3-30B、H200、in~2700 / out~194，扫并发。**Prefill wall ≈ TTFT；Decode wall ≈ E2E − TTFT。**

| max-conc | prefill TTFT(ms) | decode(ms) | E2E(ms) | **decode 占比** |
|--:|--:|--:|--:|--:|
| 1 | 91 | 854 | 945 | **90.4%** |
| 4 | 68 | 1400 | 1468 | **95.4%** |
| 8 | 81 | 1933 | 2014 | **96.0%** |
| 16 | 98 | 2295 | 2393 | **95.9%** |
| 32 | 300 | 2184 | 2484 | **87.9%** |
| 64 | 177 | 2013 | 2190 | **91.9%** |

**全并发档 decode 占 88–96% wall** → decode 是 wall-clock 的优化杠杆。

## 2. Decode kernel 分解（Part B，NCU 11 指标）

一个 decode step 有 13 个 kernel。按耗时聚合（b32），两大热点 = **FlashAttention + fused_moe（专家 GEMM）合计 ~65%**：

**agent_decode_b32（每 step 452 µs）**
| kernel | dur(µs) | 占比 | DRAM% | SM% | warps_active% | L2 hit% | occ 限制 |
|---|--:|--:|--:|--:|--:|--:|---|
| **FlashAttention** | 165.6 | 37% | 68.4 | 44.2 | 18.7 | 2.6 | warp |
| **fused_moe（专家 GEMM）** | 127.1 | 28% | 66.9 | 19.5 | **37.6** | 12.2 | warp |
| dense GEMM (nvjet) | 72.7 | 16% | 33.3 | 5.8 | 14.3 | 14.4 | warp |
| RMSNorm / RoPE / topk / act / setup | <22 each | — | 小 | 小 | — | — | — |

其余（norm、RoPE、router、activation、小 dense GEMM）都 <22 µs，不是时间所在。

## 3. Decode 能拿多少 & gap 在哪（核心回答）

### (a) 瓶颈本质：搬运受限 + 低 occupancy
- **两大热点都是 DRAM-bound**：FlashAttn DRAM% 68–79%，fused_moe 55–67%（vs SM% 只有 15–45%）。这是 memory-bound decode 的直接证据。
- **fused_moe 的 occupancy 被 warp 数卡住**：`sm__warps_active` 仅 **23–38%**，`launch__occupancy_limit_warps` 是限制项（每 SM block 数被 warp 上限约束，不是寄存器/smem）。→ 有升 occupancy 的空间，但要靠 kernel 层改（更多 resident warps 隐藏访存延迟）。

### (b) 最大杠杆：move-once-serve-more（batch↑ 直接改善搬运效率）
NCU 直接量到"增大 batch 提升权重复用"的证据 —— fused_moe 的 **L2 命中率随 batch 单调上升**：

| batch | fused_moe DRAM读(MB) | fused_moe L2 hit% | FA+MoE 占 step |
|--:|--:|--:|--:|
| 32 | 397.5 | **12.2%** | 65% |
| 64 | 322.2 | **25.9%** | 71% |
| 128 | 390.6 | **41.5%** | 83% |

**batch 32→128，fused_moe 的 L2 命中率从 12% → 41.5%**：更大 batch 让"搬一次专家权重服务更多 token"，权重更多命中 L2 而非每次都从 HBM 重搬 —— 这正是 103:1 搬算比可被 batch 改善的直接测量证据。**这是 decode 优化最大的现实杠杆**（spec-decode / 多租户并发 / expert 并行都走这条路）。

### (c) roofline 上界（exact 方法）
- decode TBT 上界 ~**1.8–1.9×**（Qwen3）/ ~2.2–2.4×（LFM），假设 kernel 打到 busiest-pipe 天花板。
- 已验证可达性：spec-decode +6%(b1)/+23%(b32) TBT；多流 util 13%→32%、吞吐 7.4×。

### (d) prefill 对照（证明 decode 的特殊性）
prefill（b1）里 fused_moe 是**compute-bound**（SM 58%、warps 仅 12%），与 decode 的 memory-bound 完全相反。→ 说明 decode 的搬运瓶颈是 decode 特有的（每步只算 1 token、权重读一遍只服务一个 token），不是模型固有属性。

## 4. 结论（给 Chendi）
1. **能拿多少**：exact 方法 decode TBT ~1.8–1.9×；真正的大头来自 **move-once-serve-more**（增大有效 batch），NCU 已量到 L2 复用随 batch 提升（12%→41.5%）。
2. **gap 在哪**：
   - **fused_moe（最大、搬运受限、occupancy 被 warp 卡）** → 提高有效 batch + kernel occupancy。
   - **FlashAttention（长 agent 上下文 ~2700 KV）** → KV 压缩 / 分页 / 更好 tiling。
   - 其余 kernel 小，不值得追。
3. **两条落地路线**：(i) move-once-serve-more（spec-decode/并发/expert-parallel）；(ii) move-less-weight（减专家：8→6 在 GSM8K 仅 −0.5pp）。都要在 sglang 端量真实 decode latency。

## 局限
- launch-count：b32/b64 采多个 decode step（40 launches），b128/prefill 用 13（一个完整 step），量级一致、families 相同。
- 单卡 H200、bf16、fa3+triton MoE 最优 config；数值为实测 NCU 硬件计数器。
- b32 是给 Chendi 的主交付 rep（已单独 push）；b64/b128/prefill 为逼近 decode 上界与 prefill 对照。
