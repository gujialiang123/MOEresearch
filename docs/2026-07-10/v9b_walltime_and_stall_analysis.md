# v9b/v9c 补充：真实 workload 的时间占比 + decode 的空闲/等待分析

**日期**：2026-07-10（回答 Dey 追问：时间去哪了 + 45.8% 平均背后有多少空闲）
**产出**：`results/2026-07-10_v9b_stalls/`（NCU WarpState/Scheduler）、`results/2026-07-10_v9c_split/`（clean bench）、`results/v9b_stall_analysis.csv`

---

## Q1：真实 workload 里 prefill / decode / idle 各占多少？

### GPU 计算时间分解（干净 bench_one_batch，带 cudagraph，无 NCU 干扰）
代表点 in=2700 / out=207 / batch=32，最优 config：

| 模型 | prefill（1 次批量 prefill） | decode（206 步） | **prefill : decode** |
|---|---|---|---|
| LFM2.5 | 905 ms | 1230 ms | **42% : 58%** |
| Qwen3-30B | 1254 ms | 1794 ms | **41% : 59%** |

> 即：处理一批 agent 请求时，GPU 计算时间大致 **40% 在 prefill、60% 在 decode**。

### idle（等 request）—— 取决于负载，不是硬件属性
| 场景 | 并发 | GPU 利用 |
|---|---|---|
| 饱和闭环（v8, max-concurrency 128） | ~128 | 几乎无 idle（队列不空） |
| **真实到达时间戳**（v7, 单条 toolagent 流） | LFM 6.1 / Qwen3 25.4 | server decode 容量 ~155 → **大量 idle** |

> **关键**：单条真实 agent 流太稀疏，喂不满一张 H200——真实到达下并发只有 6–25，而 server 能同时跑 ~155。所以"等 request 的 idle"在真实单流下**很大**，但这是**部署/负载**问题（多用户并发就能填满），不是 kernel 或硬件的优化空间。**要区分两类 gap：idle 靠加负载解决，kernel 内的空转靠 kernel 优化解决（见 Q2）。**

---

## Q2：45.8% 是平均——背后瞬时/空闲多少？多少时间在等内存？

NCU 的 throughput 是**整个 kernel 执行期的平均**（sustained），没有"瞬时峰值"时间序列。但 `SchedulerStats` 的 **"No Eligible"** 直接给出答案：

> **No Eligible% = 调度器在多少比例的周期里"没有任何一个 warp 可以发射指令"** —— 也就是 SM 完全空转、在干等（等内存 load 回来 / 等依赖）的周期占比。

### decode（b=32, in=2700，最优 config）热点 kernel

| 模型 | kernel | dur | SM% | **No Eligible%（空转周期）** | Occ% |
|---|---|---|---|---|---|
| LFM2.5 | flash_attn | 88 µs | 44.0 | **50.4%** | 24.9 |
| LFM2.5 | nvjet_gemm | 20 µs | 9.5 | **90.0%** | 14.1 |
| Qwen3 | flash_attn | 54 µs | 44.6 | **67.6%** | 18.7 |
| Qwen3 | fused_moe | 44 µs | 15.9 | **80.5%** | 35.6 |
| Qwen3 | nvjet_gemm | 10 µs | 7.1 | **82.6%** | 14.2 |

### 怎么读（回答"45.8% 平均背后是什么"）
- LFM2.5 最热的 flash_attn：SM 平均 44%，但 **50% 的周期 SM 完全空转**（No Eligible）——所以"平均 44%"不是"一直半忙"，而是"一半时间在满负荷发射、一半时间在干等"。等的原因：occupancy 只有 25%，驻留的 warp 太少，一旦它们去等内存（long-scoreboard），调度器就没有别的 warp 顶上 → SM 空转。
- decode 的 GEMM（nvjet）更极端：**80–90% 周期空转**，因为 decode 的 batch 太小，GEMM 是访存/延迟受限，算力几乎没用上。

### 这证明了什么（是否真有优化空间）
**有，而且是可回收的空闲，不是"已经在满负荷干活"：**
1. decode 关键 kernel **50–90% 的周期是 SM 空转**（No Eligible），根因是 **occupancy 低（12–25%）+ 内存延迟没被隐藏**。
2. 这类空转正是 kernel 优化能回收的：提高 occupancy（更多驻留 warp）、更好的访存 overlap/prefetch、更大 tile → 让 warp 一直有活干、把等内存的时间藏起来。
3. **config tuning 改不了这个**（occupancy 是 kernel launch/寄存器/tile 决定的）——和 §5 的结论一致：剩余 gap 在 kernel 层。

> 与 §7 的 roofline 上界（LFM decode 可快 ~2.4×）互相印证：那 2.4× 的 headroom，物理来源就是这里的 50%+ 空转周期。

---

## 数据可信度说明
- prefill/decode 分解、SM%、No Eligible%、occupancy 全部为 **NCU/sglang 实测**（clean bench 带 cudagraph；stall 用 WarpStateStats+SchedulerStats section）。
- "No Eligible" 是调度器实测的空转周期占比，**不是估算**，直接对应"SM 在等（内存/依赖）"的时间。
- 局限：NCU 每 kernel 抓 24 个 launch，未必覆盖一个 decode step 的全部 layer；但每 kernel 的利用率/空转比例是该 kernel 自身的实测值，不受影响。

## 产物
- `results/v9b_stall_analysis.csv` — 每 kernel 的 SM% / No Eligible% / occ / issued-per-cycle
- `results/2026-07-10_v9b_stalls/<model>/agent_decode_b32/` — NCU 原始（WarpState+Scheduler）
- `results/2026-07-10_v9c_split/{lfm,qwen}.jsonl` — clean prefill/decode 分段计时

---

## Q3：一共浪费多少时间？多少是 SM 空转、多少是 server idle？

两种浪费在**两个层级**，靠**两种手段**回收：

### 场景 A —— 真实到达（单条 toolagent 流，LFM2.5，200 请求，墙钟 38s）
| 时间去向 | 秒 | 占墙钟 | 靠什么回收 |
|---|---|---|---|
| ① server idle（GPU 没活、等请求） | 24.7s | **65%** | server policy（攒批/多路并发） |
| ② GPU 跑 prefill | 5.6s | 15% | — |
| ③ GPU 跑 decode | 7.7s | 20% | — |
| &nbsp;&nbsp;└ 其中 **SM 空转**（No Eligible） | 5.2s | **14%** | kernel 优化 |
| &nbsp;&nbsp;└ 其中 SM 真正在算 | 2.5s | 7% | — |
| **浪费合计** | **29.9s** | **79%** | |
| &nbsp;&nbsp;├ server idle | 24.7s | 65% | server policy |
| &nbsp;&nbsp;└ SM 空转 | 5.2s | 14% | kernel 优化 |

> 真实单流负载下，**大头(65%)是 server idle**——GPU 被喂不饱。这靠 server 策略解决（多用户、攒批、连续批处理），不是 kernel 问题。

### 场景 B —— 饱和负载（server 喂满，idle≈0），隔离 kernel 层浪费
| 模型 | 单批 GPU 时间 | prefill | decode | **decode 里 SM 空转** |
|---|---|---|---|---|
| LFM2.5 | 2135 ms | 905 (42%) | 1230 (58%) | **824 ms = 占 GPU 39%** |
| Qwen3-30B | 3048 ms | 1254 (41%) | 1794 (59%) | **1399 ms = 占 GPU 46%** |

> **即使把 server 喂满、idle 归零**，GPU 计算时间里仍有 **39%(LFM) / 46%(Qwen3) 是 decode 的 SM 空转**——这块只能靠 kernel 优化回收。这就是"单靠 tuning/server policy 达不到硬件上限"的量化证据。

### 一句话总结
- **SM 空转（decode 里 ~67–78% 的周期，占饱和 GPU 时间 39–46%）→ 只能靠 kernel 优化**（提 occupancy、隐藏内存延迟）。
- **server idle（真实单流下占墙钟 ~65%）→ 只能靠 server policy / 加负载**（攒批、多并发）。
- 两者独立、不可互相替代：kernel 优化不会减少 server idle，加负载也不会减少 SM 空转。

> 数字口径：server idle 是**估算**（墙钟 − 用饱和吞吐折算的 GPU 干活时间）；SM 空转是 NCU **实测**（时间加权 No Eligible）；prefill/decode 分段是 sglang **实测**。
