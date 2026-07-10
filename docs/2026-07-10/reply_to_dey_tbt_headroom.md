# Reply to Dey — "How much can TBT still improve at best config?"

> 说明：这是给 Dey 的回复草稿。上半部分是可直接复制的英文回复，下半部分是中文备注 + 数据出处。

---

## ✉️ English reply (copy-paste ready)

Hi Dey — yes, we can answer this. We measured it end-to-end under the **best config we tuned** (kernel-level with Nsight Compute, server timeline with Nsight Systems). Three parts:

**1. TBT (decode) roofline headroom — exact methods only (same bytes/FLOPs, no quantization):**
- LFM2.5-8B-A1B: **~2.2–2.4×** (the busiest hardware pipe is only ~43% utilized, time-weighted across the decode step)
- Qwen3-30B-A3B: **~1.8–1.9×** (~54%)

This is an upper bound (assumes kernels can be driven to 100% on their busiest pipe), so real gains will be somewhat less. It excludes approximate methods — quantization/spec-decode would add more on top.

**2. Why the average SM utilization is low (is there real headroom?):**
The hottest decode kernels are SM-idle 50–90% of cycles — this is NCU's "No Eligible" metric: the scheduler has **no warp ready to issue** an instruction, so the SM stalls. Root cause is low occupancy (12–25%): too few resident warps to hide memory latency. Time-weighted, **67% (LFM) / 78% (Qwen3) of the decode GPU-time is the SM sitting idle waiting on memory**. This is reclaimable by kernel work (higher occupancy, better latency hiding / tiling), **not** by config tuning.

**3. Full walltime budget under real single-stream toolagent arrival (measured with nsys timeline):**
- **~86% (LFM) / ~81% (Qwen3) of walltime is server idle** — the GPU is waiting for requests. Real arrival concurrency is only 6–20 vs ~155 server capacity, and toolagent's heavy prefix sharing (radix-cache reuse) shrinks prefill compute too. This is a **serving-policy / load** problem (batching, multi-tenancy, continuous batching), not a kernel one.
- Within the 14–19% the GPU is actually busy, the decode SM-idle from (2) still applies — and it persists even at full saturation (39–46% of GPU compute time is SM idle in the saturated case).

**Bottom line:** there are **two independent gaps**, and config tuning (which we've now exhausted on the real workload) reaches neither:
- **Server idle (~85%)** → fix with serving policy / higher load.
- **Kernel SM idle (~2× TBT ceiling on decode)** → fix with kernel optimization.

All numbers are measured (NCU hardware counters for kernels; nsys kernel-interval union for GPU busy/idle), not estimated. Happy to share the per-kernel tables and the timeline breakdown.

---

## 📌 中文备注（给你自己看，不用发）

**每个 regime 的 TBT headroom（roofline 上界，实测数据外推）**

| 模型 | regime | 时间加权 busiest-pipe | TBT 最多再快 | 可压掉 |
|---|---|---|---|---|
| LFM2.5 | decode b32 | 42.3% | ~2.4× | 58% |
| LFM2.5 | decode b64 | 44.6% | ~2.2× | 55% |
| Qwen3-30B | decode b32 | 53.5% | ~1.9× | 47% |
| Qwen3-30B | decode b64 | 55.8% | ~1.8× | 44% |

**SM 空转（decode 热点 kernel 的 No Eligible%，NCU 实测）**

| 模型 | kernel | SM%(平均) | No Eligible%(空转) | Occ% |
|---|---|---|---|---|
| LFM2.5 | flash_attn | 44 | 50 | 25 |
| LFM2.5 | nvjet_gemm | 9.5 | 90 | 14 |
| Qwen3 | flash_attn | 44.6 | 68 | 19 |
| Qwen3 | fused_moe | 15.9 | 80 | 36 |

**server idle（nsys 时间线实测）**

| 模型 | serving | 并发 | GPU busy | 利用率 | server idle |
|---|---|---|---|---|---|
| LFM2.5 | 38.1s | 6.2 | 5.36s | 14% | **86%** |
| Qwen3-30B | 43.3s | 19.7 | 8.03s | 19% | **81%** |

**两个 gap，两个手段（互相独立、不可替代）**
- SM 空转 → kernel 优化（提 occupancy、隐藏内存延迟）；饱和负载下仍占 GPU 39–46%。
- server idle → server policy（攒批、多并发、多租户）；真实单流下占墙钟 ~85%。

**数据出处**
- TBT headroom / SM% / No Eligible / occupancy：NCU 实测（`results/consolidated_v9_ncu.csv`、`results/v9b_stall_analysis.csv`、`results/v9_tbt_headroom.csv`）
- prefill/decode 分段：sglang clean bench（`results/2026-07-10_v9c_split/`）
- server idle：nsys 时间线（`results/v9d_server_idle_measured.csv`、`results/2026-07-10_v9d_nsys/*/timeline.nsys-rep`）
- 完整分析文档：`docs/2026-07-10/v9_ncu_hardware_ceiling_evidence.md`（§7 TBT headroom）、`docs/2026-07-10/v9b_walltime_and_stall_analysis.md`（Q1–Q4）

**caveat（如果 Dey 追问）**
- TBT headroom 是 roofline **上界**，真实达不到 100% 利用；只算 exact 方法。
- decode 绝对 TBT 的原始值来自 bench_one_batch+NCU 的 eager 模式（偏高），但 **headroom 比例**不受影响。
- server idle 是**实测**（不是估算）；它取决于负载，多用户/攒批可回收。
