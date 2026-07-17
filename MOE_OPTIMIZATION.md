# MoE Optimization — 实验索引

> 本 branch (`moe-optimization`) 专门存放 MoE 推理优化相关的实验、代码与报告，与主线 tuning/profiling 工作分开管理。
> 主题：**为什么 decode MoE 慢（专家搬运受限），以及能怎么优化（减专家 / 动态路由 / move-once-serve-more）。**

模型：`Qwen3-30B-A3B-Instruct-2507`（E=128 专家，原生 top-8，48 层 MoE）| 硬件：单卡 H200 | 数据：agent (toolagent) workload + GSM8K

---

## 一句话主线
Agent workload 是 **decode-bound**（decode 占 wall 88–96%）；decode 里 **FlashAttention + fused_moe 专家 GEMM 占 65–83%**，都是 **DRAM/搬运受限**（搬:算 ≈ 103:1）；最现实的加速是**增大有效 batch**（实测 fused_moe L2 复用随 batch 12%→41.5% 上升），辅以**减专家**（GSM8K top-6 仅 −0.5pp）。

---

## 实验清单

| 版本 | 主题 | 脚本 | 结果 | 报告 |
|---|---|---|---|---|
| v13 | Router 行为分析（agent input 上的专家选择分布） | `scripts/run_v13_router_analysis.py` | `results/2026-07-15_v13_router/` | `docs/2026-07-15/v13_router_analysis.md` |
| v14 | Batch 级专家合并权衡（搬运节省 vs 权重成本） | `scripts/run_v14_consolidation.py`, `run_v14b_consolidation_batch.py` | `results/2026-07-15_v14_consolidation/`, `_v14b_*` | `docs/2026-07-15/v14_consolidation_tradeoff.md` |
| v15 | 减专家的真实 perplexity 成本曲线 | `scripts/run_v15_perplexity.py` | `results/2026-07-15_v15_ppl/` | `docs/2026-07-15/v15_perplexity_tradeoff.md` |
| v16 | Router 分布 / 置信度详细分析（含图） | `scripts/run_v16_router_dist.py`, `plot_v16.py` | `results/2026-07-15_v16_router_dist/` | `docs/2026-07-15/v16_router_distribution.md` |
| **v17** | **GSM8K 固定 top-k 精度×时间曲线** | `scripts/run_v17_gsm8k_topk.py` | `results/2026-07-15_v17_gsm8k_topk/` | `docs/2026-07-15/v17_gsm8k_topk_results.md` |
| **v18** | **动态 top-k（置信度自适应）vs 固定** | `scripts/run_v18_dynamic_topk.py` | `results/2026-07-15_v18_dynamic_topk/` | `docs/2026-07-15/v18_dynamic_topk_results.md` |
| **v19 Part A** | **decode/prefill wall 占比（扫并发）** | `scripts/run_v19_wall_sweep.sh` | `results/2026-07-15_v19_wall_sweep/` | （见 Part C） |
| **v19 Part B** | **NCU decode kernel（11 指标，4 regime）** | `scripts/run_v19b_ncu_decode.py`, `parse_v19b_ncu.py` | `results/2026-07-15_v19b_ncu_decode/` | （见 Part C） |
| **v19 Part C** | **decode 能拿多少 + gap 在哪（综合）** | — | `.../ncu_summary.json` | `docs/2026-07-15/v19_partC_decode_potential.md` |
| **v20** | **Dynamic top-K 正确性修复+验证（P0：物理跳过/去同步/严格解析）** | `scripts/dynamic_topk_utils.py`, `run_v20_dynamic_topk_equivalence.py`, `tests/test_dynamic_topk.py` | `results/2026-07-16_v20_equivalence/` | `docs/2026-07-16/v20_dynamic_topk_validation.md` |
| **v21** | **固定 top-K「K→生成长度」剂量曲线 + L_to/L_post 分解（GSM8K 500）** | `scripts/run_v20_dynamic_topk_free_generation.py`, `analyze_v21_k_vs_length.py` | `results/2026-07-16_v21_k_vs_length/` | `docs/2026-07-16/v21_k_vs_length_results.md` |
| **v22** | **Teacher-forced 终止分析：直接终止效应 vs 轨迹中介（logp(EOS)/margin/KL）** | `scripts/run_v22_teacher_forced_eos.py` | `results/2026-07-16_v22_teacher_forced/` | `docs/2026-07-16/v22_teacher_forced_results.md` |

**背景/调研文档**：
- `docs/2026-07-15/moe_routing_optimization_survey.md` — router 次优性 + route-for-efficiency 文献调研
- `docs/2026-07-15/triton_moe_kernel_analysis.md` — MoE kernel 根因 + 103:1 搬算比
- `docs/2026-07-15/dynamic_topk_and_benchmark_plan.md` — 动态 topk 可行性 + benchmark 计划

**对外沟通（reply 草稿）**：
- `docs/2026-07-15/reply_to_dey_progress_update.md` — 给 Dey 的进度更新
- `docs/2026-07-15/reply_to_dey_decode_kernels.md` — 给 Dey 的 decode kernel 分析（funnel + 13 kernel）
- `docs/2026-07-15/reply_to_chendi_decode_analysis.md` — 给 Chendi 的 decode profiling 交付

---

## 核心数据

### v17 — 固定 top-k（GSM8K 200 题，贪心）
| top-k | 准确率 | 掉分(pp) | 平均生成 tok |
|--:|--:|--:|--:|
| 8 (base) | 83.5% | 0.0 | 244 |
| 7 | 83.0% | −0.5 | 248 |
| 6 | 83.0% | −0.5 | 252 |
| 5 | 80.0% | −3.5 | 260 |
| 4 | 75.0% | −8.5 | 275 |

拐点 top-6→top-5。**top-6 安全档**（省 25% 专家，−0.5pp）。注意生成 tok 数随 k↓ 反而↑ → HF-eager 的“加速”是假象，真实加速需 sglang 侧量。

### v18 — 动态 top-k（置信度自适应，扫 τ）
| τ | 实测 avg_k | 准确率 | wall(s) | tok/s | 平均生成 tok |
|--:|--:|--:|--:|--:|--:|
| 0.60 | 3.85 | 73.0% | 1114 | 49.2 | 274 |
| 0.70 | 4.72 | 80.0% | 1093 | 48.5 | 265 |
| 0.80 | 5.74 | 80.5% | 1040 | 49.3 | 256 |
| 0.88 | 6.72 | 82.5% | 1039 | 48.1 | 250 |
| 0.95 | 7.85 | 81.5% | 1040 | 47.7 | 248 |

**动态 τ=0.7 用 avg_k=4.72 达到 80.0% = 固定 top-5 同精度但平均 k 更低**（省 ~0.3 专家）；中档不占优、收益幅度小。已有 ACL 2024《Harder Tasks Need More Experts》做训练时版本；我们差异点是免训练/推理时。
**时间列注意**：wall/tok-s 基本不随 avg_k 变化（与 v17 同样的“假加速”）——HF-eager 下减专家不省物理搬运，且 avg_k↓ 使生成 tok↑（248→274）抵消。真实加速必须去 sglang `fused_moe` 侧量。

### v19 Part A — decode/prefill wall 占比
| max并发 | prefill TTFT(ms) | decode(ms) | E2E(ms) | decode 占比 |
|--:|--:|--:|--:|--:|
| 1 | 91 | 854 | 945 | 90.4% |
| 4 | 68 | 1400 | 1468 | 95.4% |
| 8 | 81 | 1933 | 2014 | 96.0% |
| 16 | 98 | 2295 | 2393 | 95.9% |
| 32 | 300 | 2184 | 2484 | 87.9% |
| 64 | 177 | 2013 | 2190 | 91.9% |

**decode 占 wall 88–96%，全档主导。**

### v19 Part B — NCU decode kernel（11 指标）
| regime | step(µs) | FlashAttn (DRAM%) | fused_moe (DRAM%/warp%/L2%) | FA+MoE 占比 |
|---|--:|--:|--:|--:|
| decode b32 | 452 | 166 (68%) | 127 (67% / 37.6 / 12.2) | 65% |
| decode b64 | 316 | 102 (74%) | 124 (55% / 23.2 / 25.9) | 71% |
| decode b128 | 402 | 190 (79%) | 145 (57% / 26.5 / **41.5**) | 83% |
| prefill b1 | — | — | 553µs（SM 58% / warp 12%，compute-bound 对照） | — |

- FlashAttn + fused_moe 占 decode step **65–83%**，都是 DRAM-bound。
- **batch 32→128，fused_moe L2 命中 12%→41.5%** = “搬一次服务更多 token” 实测证据（最大杠杆）。
- fused_moe occupancy 被 **warp 数**卡住（warps_active 23–38%）。
- prefill 里同一 kernel 是 compute-bound → 搬运瓶颈是 **decode 特有**。

---

## 下一步（候选）
1. **sglang 端落地固定 top-6**（`custom_routing_function`），测**真实** decode latency，与 v17 的假加速对照。
2. **move-once-serve-more**：spec-decode / 多租户并发 / expert-parallel，直接改善 103:1 搬算比。
3. 补 **LFM2.5-8B-A1B** 的 NCU decode 对照（52:1 搬算比）。
4. 动态 topk 若继续：先做轻量校准，且报告**真实搬运字节**（`dram__bytes_read.sum`）而非仅 avg_k。

---

## 复现环境
- Python env：`/home/t-jialianggu/.conda/envs/sglang`（HF eager 精度实验）、`sglang-dev`（sglang serving / NCU profiling）
- NCU：`/opt/nvidia/nsight-compute/2026.2.1/ncu`（需 sudo）
- HF 数据集缓存需可写目录：`HF_HOME=$PWD/.hf_cache`（`/data/hf/hub` 只读）
- GSM8K 通过 `datasets.load_dataset("gsm8k","main")`
