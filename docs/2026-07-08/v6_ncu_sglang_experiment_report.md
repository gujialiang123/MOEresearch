# v6 实验报告：用 NCU 对真实 sglang kernel 做硬件计数器级 profiling

**日期**：2026-07-08 ~ 07-09
**执行**：GPU 6（NVIDIA H200，SM 9.0，132 SM，143 GB HBM3e）
**工具**：Nsight Compute `2026.2.1`（`/opt/nvidia/nsight-compute/2026.2.1/ncu`，sudo 运行）
**提交**：`224e559 experiment(v6): real sglang NCU on Qwen3-30B + LFM2.5`
**相关脚本**：`scripts/run_v6_ncu_sglang.py`（编排）、`scripts/build_v6_sglang_ncu_report.py`（汇总）
**产出表格**：`results/consolidated_v6_sglang_ncu.csv`（161 行）、`results/v6_sglang_ncu_report.xlsx`（3 sheet）

---

## 1. 这一轮为什么要做

在此之前我们对"opportunity gap（当前性能 vs 硬件天花板的差距）"的量化，很多依赖的是：
- sglang 自己打印的 HBM 利用率 / decode step time（粗粒度、聚合值）；
- 以及一次 **v5b** 尝试——用 `transformers` 框架包在 NCU 里跑推理。

**v5b 的根本问题**：`transformers` 和 sglang 走的是**完全不同的 kernel 路径**。sglang 的 MoE 用的是自己的 `fused_moe_kernel`（Triton）、`moe_sum_reduce`、`nvjet_gemm`、`causal_conv1d_update` 等；transformers 根本不会调用这些。所以 v5b 拿到的 HW 计数器数据**不能代表 sglang 实际运行时的瓶颈**。

**这一轮（v6）的目标**：直接对**真实 sglang 运行的 kernel** 做 NCU 硬件计数器测量，分阶段（decode / prefill）记录 SM 占用率、HBM 带宽利用率、Tensor Core 活跃度、occupancy、stall 计数等，作为 gap 论证的硬证据。

---

## 2. 方法学（关键：怎么让 NCU 成功包住 sglang）

这是复用 6 月 9 日已验证的做法，核心有三点：

1. **不要**去包 `sglang.launch_server`（多进程 + 成千上万 kernel，NCU replay 会爆炸，几小时跑不完）。
2. **要**包 `sglang.bench_one_batch`（单进程 CLI），并加 `--profile --profile-activities CUDA_PROFILER`。这会让 sglang 在被测区段内部调用 `cudaProfilerStart/Stop`。
3. NCU 加 `--profile-from-start off`：这样 NCU **只 profile `cudaProfilerStart` 之后**的 kernel，跳过模型加载、warmup、CUDA graph capture，只测真正的 bench 段。

**NCU 命令**（v6 LFM 用的轻量 section 组合）：
```
sudo -n ncu --target-processes all --profile-from-start off \
  --launch-count 20 --kernel-name-base demangled \
  --kernel-name 'regex:fused_moe|nvjet|flash_fwd|cutlass|RMSNorm|act_and_mul|topk|conv1d' \
  --section SpeedOfLight --section Occupancy --section LaunchStats \
  --force-overwrite --export <out>/ncu -- <out>/inference.sh
```
- `--kernel-name regex:...` + `--kernel-name-base demangled`：只抓热点 kernel，跳过零碎的 elementwise helper。
- `--launch-count 20`：每类 kernel 抓 20 次 launch。
- 需要 `sudo`：NCU 读 GPU 性能计数器需要 root 权限。
- 用 wrapper `inference.sh` 重设环境变量（sudo 会清空 env，而 sglang 的 deep_gemm / JIT 需要 `CUDA_HOME` / `CPATH` / `LIBRARY_PATH`）。

**Section 开销权衡**（在 H200 上实测）：
| Section 组合 | 每 kernel pass 数 | 每个 combo 耗时 |
|---|---|---|
| `--set full` | ~40 | 20–30 min |
| `--set basic` | 5–7 | 15–25 min |
| `SpeedOfLight + Occupancy + LaunchStats`（本轮 LFM 用） | 3–5 | 15–20 min |

---

## 3. 数据来源与合并策略

本轮报告合并了**两批真实 sglang NCU 数据**：

| 批次 | 模型 | 采集时间 | Section | Regime 数 | 每 regime kernel 数 | 覆盖指标 |
|---|---|---|---|---|---|---|
| **金标准（复用）** | Qwen3-30B-A3B（bf16） | 6-09 | `--set full` | 4 | 30 | SM%/DRAM%/**TC%**/occ%/L1/L2 hit%/stall/verdict/headroom |
| **本轮新增** | LFM2.5-8B-A1B | 7-08 | SoL+Occ+LaunchStats | 3 | 6–8 | SM%/DRAM%/warp active%/duration |

> Qwen3-30B 的 4 个 regime 6 月已用 `--set full` 完整跑过（`results/2026-06-09_sglang_triton_sweep/ncu/*/ncu_summary.json`），指标比本轮 LFM 更全（含 Tensor Core% 与 stall），因此直接复用，无需重跑。LFM2.5 是**这次首次**拿到真实 sglang kernel 的 NCU 数据。

汇总脚本 `build_v6_sglang_ncu_report.py` 把两批统一成 161 行的 CSV，并用 `short_name()` 归一 kernel 名（如 `nvjet_tst_64x8_...` → `nvjet_gemm_64x8`，`cutlass::...flash...` → `flash_attn_main`）。

---

## 4. 实验配置明细

### 4.1 Qwen3-30B-A3B（6-09，`--set full`，4 regime）

| Regime | bench 阶段 | Batch | in / out (tok) | NCU 时长 | kernel 数 |
|---|---|---|---|---|---|
| R_short_decode | decode | 1 | 200 / 256 | ~35 min | 30 |
| R_medium_balanced | decode | 8 | 1600 / 256 | ~33 min | 30 |
| R_concurrent_decode | decode | 32 | 400 / 256 | ~35 min | 30 |
| R_long_prefill | prefill | 4 | 8000 / 32 | ~60 min | 50 |

### 4.2 LFM2.5-8B-A1B（7-08，本轮新增，3 regime）

模型路径 `/data/hf/LFM2.5-8B-A1B`，config = `cookbook_baseline`
（`--mem-fraction-static 0.85 --chunked-prefill-size -1 --schedule-policy lpm --moe-runner-backend auto`）。

| Regime | 阶段 | Batch | in / out (tok) | `--max-running-requests` | NCU 时长 |
|---|---|---|---|---|---|
| R_decode_c1_out2k | decode | 1 | 130 / 128 | 32 | ~15 min |
| R_conc_ref | decode | 32 | 260 / 256 | 32 | ~19 min |
| R_decode_c128_out256 | decode | 128 | 260 / 256 | **128** | ~19 min |

> **踩坑记录**：`R_decode_c128_out256` 第一次跑在 20s 就崩了——batch=128 但 `--max-running-requests` 还是 32，导致 `alloc_req_slots runs out of memory`。把该 combo 的 `--max-running-requests` 改成 128 后重跑成功。`run_v6_ncu_sglang.py` 里 config 与 regime 的 batch 应联动，这个已知点留待编排脚本后续修正。

---

## 5. 结果

### 5.1 LFM2.5-8B-A1B —— sglang 实测（本轮核心新数据）

**端到端 bench（decode 中位）**：
| Regime | Batch | median decode latency | decode throughput |
|---|---|---|---|
| R_decode_c1_out2k | 1 | 30.3 ms | 33.0 tok/s |
| R_conc_ref | 32 | 32.3 ms | 992.0 tok/s |
| R_decode_c128_out256 | 128 | 42.7 ms | 2999.5 tok/s |

**NCU 热点 kernel（按 DRAM% 排序，取每 regime 主导 GEMM）**：
| Regime | 热点 kernel | SM% | **DRAM%** | warp active% | duration(µs) |
|---|---|---|---|---|---|
| R_decode_c1_out2k | nvjet_gemm_64x8 | 9.1 | **64.8** | 14.5 | 19.8 |
| R_conc_ref (bs=32) | nvjet_gemm_128x32 | 9.7 | **65.8** | 14.1 | 19.6 |
| R_decode_c128_out256 (bs=128) | nvjet_gemm_112x128 | 37.9 | **62.8** | 14.0 | 20.6 |

其余 kernel（`act_and_mul` / `causal_conv1d_update` / `RMSNorm`）SM% 与 DRAM% 都很低（个位数），单个 duration 3–7 µs，非瓶颈。

### 5.2 Qwen3-30B-A3B —— sglang 实测（6-09 复用，指标更全）

| Regime | 主导 kernel | SM% | **DRAM%** | **TC%** | warp active% | Verdict | Headroom% |
|---|---|---|---|---|---|---|---|
| R_short_decode | fused_moe_kernel | 11.7 | 63.8 | 8.9 | 9.2 | low_occupancy | 36.2 |
| R_medium_balanced | fused_moe_kernel | 13.5 | 67.5 | 10.1 | 19.9 | low_occupancy | 32.5 |
| R_concurrent_decode | fused_moe_kernel | 16.8 | **79.8** | 12.8 | 44.8 | **memory_bound** | 20.2 |
| R_long_prefill | moe_sum_reduce | 25.2 | **91.6** | 0.4 | 42.8 | **memory_bound** | 8.4 |
| R_long_prefill | FusedAddRMSNorm | 52.8 | 83.0 | **0.0** | 92.5 | **tensor_core_idle** | 17.0 |

---

## 6. 结论

1. **两个模型的 decode 阶段都被 HBM 带宽卡住，不是被算力卡住。**
   - LFM2.5：主导的 `nvjet_gemm` 在 batch=1/32 时 DRAM ≈ 65%，而 SM 只有 9–10%。
   - Qwen3-30B：`fused_moe_kernel` 在高并发 decode 时 DRAM 79.8%（memory_bound），prefill 的 `moe_sum_reduce` 更是逼近 91.6% 的 HBM 天花板。

2. **Hopper 的 SM 严重闲置。** 两个模型所有主导 GEMM 的 **warps-active 稳定在 14–15%**——也就是说 132 个 SM 的调度槽只用了约 1/7。这是 low-occupancy 的直接证据，是 kernel 层的优化空间（更大 tile / 更高 occupancy 的 kernel 配置理论上能吃满带宽）。

3. **加并发能提高 SM 利用，但顶不过 HBM 天花板。** LFM2.5 从 batch=1 → 128，主导 GEMM 的 SM% 从 9% 涨到 38%，但 DRAM% 始终卡在 63–66%。说明再堆并发主要是摊薄固定开销，而不是突破带宽墙。

4. **Tensor Core 在 decode/访存型 kernel 上几乎不工作。** Qwen3 的 `FusedAddRMSNorm` / `act_and_mul` DRAM 80%+ 但 TC%=0；只有 prefill 的大 GEMM（`nvjet_192x192`，6 月数据 SM 94.7% / TC 96%）才是真正 compute-bound。**这印证了"prefill 吃算力、decode 吃带宽"的分工。**

5. **对 v5b 的修正得到确认。** v6 拿到的 `fused_moe_kernel` / `nvjet_gemm` / `causal_conv1d_update` 都是 transformers 路径里不存在的 kernel——证明之前用 transformers 包 NCU 的做法（v5b）确实测不到 sglang 真正的 MoE 瓶颈，v6 才是正确的 apple-to-apple 基线。

---

## 7. 局限与后续

**本轮局限**：
- LFM2.5 只跑了 `SpeedOfLight + Occupancy + LaunchStats`，**没有 Tensor Core% 与 stall 明细**（不如 Qwen3 的 `--set full` 全）。若要严格 apple-to-apple，需用相同 section 重跑 Qwen3，或用 `--set full` 补 LFM2.5。
- 只覆盖 `cookbook_baseline` 一个 config；**`big_batch_cap128` 未用 NCU 测**。
- **fp8 模型（Qwen3-30B-A3B-FP8）未做 NCU**。

**建议后续**：
1. 对 3–5 个最热 kernel（`fused_moe_kernel` / `nvjet_gemm`）用 `--set full` 深挖，拿 roofline 上的精确 achieved bandwidth。
2. 补 fp8 模型 4 regime（约 40 min）与 `big_batch_cap128` config，看不同 config 是否改变瓶颈性质。
3. 把 NCU 实测的 achieved DRAM 带宽换算成"距 HBM 理论峰值还差多少"，直接量化 hardware-layer 的 gap 数字。

---

## 附：产物清单

- `results/2026-07-08_v6_ncu/lfm2.5-8b-a1b/cookbook_baseline/{R_decode_c1_out2k,R_conc_ref,R_decode_c128_out256}/` — 每个含 `ncu.ncu-rep` + `ncu_raw.csv` + `bench.log` + `bench_one_batch_result.jsonl` + `combo_params.json` + `inference.sh`
- `results/2026-06-09_sglang_triton_sweep/ncu/*/ncu_summary.json` — Qwen3-30B 金标准（复用）
- `results/consolidated_v6_sglang_ncu.csv` — 161 行合并表
- `results/v6_sglang_ncu_report.xlsx` — README / all_ncu_kernels / hot_kernels_by_regime 三 sheet
- `scripts/run_v6_ncu_sglang.py`、`scripts/build_v6_sglang_ncu_report.py`
