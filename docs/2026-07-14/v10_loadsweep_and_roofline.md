# v10：Offered-load sweep + 标准 Roofline 重述（为 Ofer/Li 汇报补齐）

**日期**：2026-07-14
**执行**：GPU 4（LFM2.5）+ GPU 5（Qwen3-30B），双卡并行
**目的**：回应 Chendi/Dey 两点——(1) 加更多请求能回收多少 serving idle；(2) 用**标准 roofline** 重述之前的 "TBT headroom"。
**脚本**：`scripts/run_v10_load_sweep.py`、`scripts/compute_v10_roofline.py`
**产出**：`results/consolidated_v10_load_sweep.csv`、`results/2026-07-14_v10_load_sweep/`

---

## Part 1 — Offered-load sweep：加负载能回收多少？

固定最优 config（server `max-running-requests 256`），客户端在真实 toolagent 上扫 `--max-concurrency 8→256`（时间戳压缩 0.02× 使 offered load 成为真正限流）。

### LFM2.5-8B-A1B
| offered | achieved并发 | out tok/s | req/s | TTFT(ms) | **TPOT(ms)** |
|---|---|---|---|---|---|
| 8 | 7.9 | 1663 | 9.0 | 38 | 4.0 |
| 16 | 15.7 | 2817 | 15.3 | 38 | 5.0 |
| 32 | 30.7 | 3732 | 20.2 | 47 | 7.6 |
| 64 | 61.0 | 4684 | 25.4 | 58 | 12.4 |
| 128 | 118.0 | 5857 | 31.8 | 84 | 20.0 |
| 256 | 215.3 | **7309** | 39.6 | 205 | 33.4 |

### Qwen3-30B-A3B
| offered | achieved并发 | out tok/s | req/s | TTFT(ms) | **TPOT(ms)** |
|---|---|---|---|---|---|
| 8 | 8.0 | 739 | 4.0 | 45 | 9.5 |
| 16 | 15.8 | 1040 | 5.6 | 48 | 13.4 |
| 32 | 31.3 | 1425 | 7.7 | 63 | 20.1 |
| 64 | 61.7 | 1859 | 10.1 | 78 | 30.6 |
| 128 | 118.8 | 2219 | 12.0 | 96 | 51.6 |
| 256 | 212.0 | **2416** | 13.1 | 669 | 65.8 |

### 结论（Part 1）
1. **加负载能大幅回收 serving idle 换来的吞吐**：并发 8→256，吞吐 LFM **4.4×**（1663→7309）、Qwen3 **3.3×**（739→2416）。这印证了 serving idle（v9d 实测 ~85%）确实可通过更高负载/更好调度回收。
2. **但有明确的吞吐-延迟权衡**：同一区间 TPOT 恶化 LFM **8.4×**（4→33ms）、Qwen3 **6.9×**（9.5→66ms）。即 TBT（token 间延迟）随并发单调变差。
3. **收益递减**：吞吐增益从每档翻倍逐渐放缓（LFM 128→256 只 +25%），且 TTFT 在 256 急升（LFM 205ms、Qwen3 669ms）。**最优工作点取决于 SLA**：要低延迟就限并发，要高吞吐就加并发，二者不可兼得。
4. **关键**：加负载**只回收 serving idle，不触及 kernel 层 SM 空转**——后者需要 kernel 优化（见 Part 2）。

---

## Part 2 — 标准 Roofline 重述

把之前的 "TBT headroom"（我们自造的启发式）用标准 roofline 语言重新表述（Williams et al. 2009）。

### H200 硬件天花板
- HBM3e 峰值带宽：**4.8 TB/s**
- bf16 Tensor Core 峰值：**~989 TFLOP/s**
- Ridge point（拐点）= 989/4.8 ≈ **206 FLOP/byte**（AI 低于此 → memory-bound）

### 2A. Per-kernel bandwidth roofline（decode b=32，用 v9 NCU 实测 DRAM%）
achieved BW = DRAM% × 4.8 TB/s；距屋顶 = 100/DRAM%。

| 模型 | kernel | DRAM% | achieved BW | 距带宽屋顶 |
|---|---|---|---|---|
| LFM2.5 | flash_attn | 45.1 | 2.17 TB/s | **2.2×** |
| LFM2.5 | nvjet_gemm | 68.1 | 3.27 TB/s | 1.5× |
| Qwen3 | flash_attn | 71.9 | 3.45 TB/s | **1.4×** |
| Qwen3 | fused_moe | 75.3 | 3.61 TB/s | 1.3× |

> decode 热点 kernel 全部 **memory-bound**（DRAM% ≫ SM%），落在 roofline 的带宽斜线区；离屋顶 1.3–2.2×，即 kernel 优化的理论上界。

### 2B. 时间加权的整步 roofline 距离（纯实测，不依赖任何 MoE 假设）
按 duration 加权每个 kernel 的 busiest-roof 距离：
| 模型 | 时间加权 busiest-roof 利用率 | **整步距屋顶** |
|---|---|---|
| LFM2.5 decode | 42.2% | **2.37×** |
| Qwen3-30B decode | 53.5% | **1.87×** |

> 这与之前 "TBT headroom" 数字（2.4× / 1.9×）**完全一致**——现在用标准 roofline 框架表述，可信度更高。

### 2C. 端到端 first-principles floor（decode）
`TBT_floor = 每步必读字节(激活权重+KV) / 4.8 TB/s`。关键洞察在 **batch 依赖**：
| 模型 | batch=1 floor | 实测 TBT | 距 floor |
|---|---|---|---|
| LFM2.5 | 0.42 ms | 5.97 ms | **14×** |
| Qwen3-30B | 1.43 ms | 8.71 ms | **6×** |

> batch=1 时权重每读一次只服务 1 个 token → 带宽严重浪费（14×/6× above floor）。**这个巨大 gap 主要靠加 batch（server policy）收敛**——batch 越大，一次权重读服务越多 token，趋近 per-kernel 的 ~2× 上界。
>
> ⚠️ batch=32 的端到端 floor 对"一个 batch 激活多少 MoE 专家"高度敏感（LFM 32 专家易被激活满，Qwen3 128 专家难），因此 **batch>1 的端到端数字以 2A/2B 的实测 per-kernel roofline 为准**，不用 first-principles 估算。

---

## 汇报主线（给 Ofer/Li）

**我们如何测 performance opportunity gap，以及 gap 有多大：**

1. **方法**：真实 agent 负载（mooncake toolagent）+ 最优 config，用 NCU（kernel 硬件计数器）+ nsys（server 时间线）分层测量。
2. **发现两个正交的 gap**：
   - **Serving idle**（GPU 等请求）：真实单流下占墙钟 ~85%（nsys 实测）。**Part 1 证明**加负载可回收（吞吐 3–4×），但换来更高 TBT，且有收益递减 → **靠 policy/framework**。
   - **Kernel SM 空转**（decode 内存受限、occupancy 低）：占 decode GPU 时间 67–78%。**Part 2 roofline 证明**热点 kernel 距带宽屋顶 1.3–2.4× → **靠 kernel 优化**。
3. **config tuning 已到头**（v8 sweep），触及不到这两个 gap。
4. **两个 lever 正交**：加负载不减 SM 空转，kernel 优化不减 serving idle。

---

## 产物
- `results/consolidated_v10_load_sweep.csv` — 12 行 offered-load 曲线
- `results/2026-07-14_v10_load_sweep/` — 原始 bench 结果
- `scripts/run_v10_load_sweep.py`、`scripts/compute_v10_roofline.py`
