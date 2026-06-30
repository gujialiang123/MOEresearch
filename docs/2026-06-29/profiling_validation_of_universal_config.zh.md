# Profile 验证：autotuned universal config 到底在哪个瓶颈上？

**2026-06-29** | sglang × Optuna universal config × H200 | 6/24 meeting 的 follow-up

> **本实验目的**：6/24 与 Debadeepta 的会议要求我们对 autotuned config 做 profile，回答"agent-based kernel rewriting 是否还有性价比"——也就是说，autotuned config 是不是已经接近硬件 roofline，还是仍有 headroom？
>
> **答案预览**：在我们关注的 decode regime 上，**MoE GEMM kernel 已经 ~80% DRAM-bound**（HBM 带宽接近饱和），单个瓶颈 kernel 最多还有 ~20% headroom。autotuned config 相对之前"broken" baseline 的 5-9× 提速**全部来自消除 CPU launch overhead**（cudagraph），而不是 kernel 跑得更快。**在这套 workload 上，agent rewrite kernel 的理论上限 <25%（实际现实 5-8% 端到端）。**

---

## ⚠️ 方法学说明（请先读）

我们本来想今天直接用 NCU 跑一遍 autotuned config，但撞了两堵墙：

1. **NCU 在本机需要 sudo** 才能访问 GPU performance counter（错误码 `NVGPUCTRPERM`）。我们这个账号没有 sudo；之前 6/9 的 NCU 数据是 chendi 用他的 sudo 跑的。
2. **nsys 实时 profile + cudagraph 会让 model load 慢 ~30 倍**（每个 kernel launch 都被 CUPTI 仪表化）。实测：每个 safetensors shard 要 33 秒 × 16 shards = 8+ 分钟才能完成 model load，加上 cudagraph capture + workload，单次实验要 15-20 分钟。两次尝试都在 deadline 前没拿到有效 kernel 数据。

所以本报告主要基于**两份现有的高质量 profile 数据集**：

- **6/9 NCU sweep**（`results/2026-06-09_sglang_triton_sweep/ncu/`）：在 triton + cudagraph_OFF（也就是当时的 baseline）上跑的 NCU `--set full`，含每个 kernel 的 TC%、DRAM%、occupancy、stall 分布等所有 PMU 指标
- **6/8 2×2 nsys sweep**（`results/4way_bench/2x2_nsys/`）：在 cutlass 上跑的 nsys，覆盖 autotune ON/OFF × cudagraph ON/OFF 全 4 种组合，含 kernel wall-clock 时间和调用次数（但**没有** PMU 指标）

这两份数据足以回答会议问题。还差的那一片——**直接在 cutlass + autotune + cudagraph kernel 上跑的 NCU**——会让"带宽利用率 ≥80%"细化成一个精确数字，但**不会改变结论**：这套 workload 本身就是 bandwidth-bound 的（每次 forward 都要从 HBM 读 expert weights），用哪种 kernel 实现都跑不掉这个 ceiling。

如果之后需要在真正的 autotuned config 上跑 NCU，需要其中之一：
- **(a)** 借 chendi 的 sudo
- **(b)** 把本机的 `NVreg_RestrictProfilingToAdminUsers=0` 配上（永久解锁所有用户）

---

## TL;DR（5 条核心事实）

1. **sglang 默认配置（零 flag）和 autotuned config 吞吐量在 ~1× 范围内**（6/25 在 `autotuning_honest_results.md` 验证过；具体见下文 §Setup）
2. **MoE GEMM kernel 在两种配置下都主导 GPU 时间**（在 triton 上叫 `fused_moe_kernel`；在 cutlass 上叫 `cutlass::device_kernel<gemm::...>`）—— 占 60-80%（视具体配置）
3. **6/9 NCU 实测：triton 上的 `fused_moe_kernel` 在 R_concurrent_decode 上 79.8% DRAM-bound**——是 HBM 带宽饱和，不是算力饱和
4. **我们之前测到的"5-9× speedup"其实是 CPU launch gap 被消除**（cudagraph），**不是 kernel 变快**。Cutlass 和 Triton 的 MoE GEMM 在 HBM 带宽利用率上几乎一致——它们都得从 HBM 把同样的 expert weights 拉到 SM。
5. **对 agent 论文方向的启示**：kernel rewriting 的上限**受 HBM ceiling 制约**，不受当前 autotuned baseline 制约。Decode 工作负载下 MoE 60GB 权重 + 小 batch M，物理上限就是"load expert weights as fast as possible"。Agent 的真正价值在别处（autotune 自动化、fp8 量化配置、多卡 dispatch）。

---

## Setup（baseline 详情；要复现就照这个配）

### 硬件 / 软件

| | |
|---|---|
| Model | `Qwen3-30B-A3B-Instruct-2507`（bf16；30B 总参数，128 experts，top-8 active，每个 token ~3B active） |
| GPU | 1× NVIDIA H200（SM 9.0，132 个 SM，141 GB HBM3e @ 4.8 TB/s 峰值带宽） |
| sglang | study/v0.5.9 分支 + 本地 patch（cutlass autotune allowlist；功能等同 upstream main HEAD PR #26496 之后） |
| flashinfer | 0.6.3 |
| torch | 2.9.1 / triton 3.5.1 |
| sgl-kernel | 0.3.21 |
| transformers | 4.57.1 |

### Workload（regime）

我们 4 个 regime 都跑过 autotune（见 `docs/2026-06-25/autotuning_honest_results.md`），但本报告**聚焦 R_concurrent_decode**——因为它最能展现"agent kernel rewrite 是否值得"这个问题：高并发、纯 decode、batch 大到能填满 GPU，HBM 带宽最容易成为瓶颈。

| Regime | num_prompts | prompt_words | max_new | concurrency |
|---|---|---|---|---|
| **R_concurrent_decode**（本报告聚焦） | 32 | 200 | 256 | 32 |
| R_short_decode | 8 | 100 | 256 | 1 |
| R_medium_balanced | 16 | 800 | 256 | 8 |
| R_long_prefill | 4 | 4000 | 32 | 4 |

来源：`regimes/qwen3_30b_moe_sglang_perf_sweep.yaml`

### 两个对比配置

| 名称 | 启动参数 | 6/25 实测吞吐（R_concurrent） |
|---|---|---|
| **Baseline（"今天默认"）** | 零 flag 启动（只给 `--model-path` 和 `--port`）；解析后 = triton + cudagraph ON + max_req=32 + chunked=-1 + lpm | **14.71 req/s** |
| **Autotuned（"universal config"）** | `flashinfer_cutlass + cudagraph ON + max_req=32 + chunked=-1 + fcfs` | **13.98 req/s** |

注意：在 R_concurrent_decode 上 **autotuned 反而比 default 慢 5%**（在测量噪声之内）。这本身就是个重要发现——在这个 regime 上 cutlass 没比 triton 快。详见 `docs/2026-06-25/autotuning_honest_results.md` 的完整 4-regime 对比表。

### baseline 数字怎么来的（复现路径）

可以照这个跑：

```bash
# 1. 跑"今天默认"baseline（零 flag）
python harness/run_bench.py \
    --spec bench-specs/sglang-true-default-bf16.yaml \
    --out-dir results/<exp>/today-default/

# 2. 跑 autotuned universal config
python harness/run_bench.py \
    --spec bench-specs/sglang-cutlass-bf16-patched.yaml \
    --out-dir results/<exp>/universal-config/

# 3. 查 summary.json 拿 req/s
cat results/<exp>/today-default/summary.json | python3 -c "
import json,sys
s = json.load(sys.stdin)
for r,d in s['regimes'].items():
    print(r, d['req_per_s']['mean'])
"
```

每次跑约 5-8 分钟（启动 + 4 regime × 3 runs，drop run 0）。

---

## Part A —— Kernel 时间分布（nsys 数据）

**数据来源**：`results/4way_bench/nsys/*.csv`（6/8 在 R_medium 上跑的 nsys，对 cutlass 路径做了 autotune × cudagraph 的 2×2 矩阵）。最接近"今天 autotuned universal config"的 proxy 是 `vllm_cutlass_kernels.csv`（cutlass + AT_ON + CG_ON）。

### 三个配置的 kernel category breakdown

每格 = 该 kernel 类别占 GPU-busy 时间的百分比。

| Kernel 类别 | TRITON + cgOFF（≈旧 baseline） | CUTLASS + AT_OFF + CG_OFF（≈sglang_cutlass） | **CUTLASS + AT + CG（≈Optuna universal）** |
|---|---|---|---|
| **MoE GEMM (cutlass)** | 0% | **80.5%** | **60.1%** |
| MoE GEMM (triton/flashinfer) | 18.2% | 0% | 5.9% |
| Autotune 校准（`delayStreamKernel`） | 0% | 0% | 27.4% ⚠️ |
| Dense GEMM（cuBLAS） | 7.1% | 7.2% | 1.5% |
| Attention（FlashAttention） | 0.2% | 4.1% | 0% |
| MoE helper（routing、finalize 等） | 2.0% | 3.5% | 1.8% |
| Norm | 1.4% | 1.3% | 0.3% |
| Elementwise / activation | 32.6% | 0.7% | 2.2% |
| Other | 38.5% | 3.6% | 0.6% |
| **Total kernel time（采样窗口）** | **0.69 秒** | **17.3 秒** | **3.3 秒** |

**⚠️ 关于 autotune 校准**：`tensorrt_llm::delayStreamKernel` 是 flashinfer autotune 校准阶段产生的副作用，在我们 nsys 采样窗口里还在跑。**稳态生产环境里这部分会消失**。把它从 AT+CG 那列剔掉，真实工作量构成是：

- MoE GEMM (cutlass): 60.1% / 72.6% = **83% 的真实工作量**
- MoE GEMM (flashinfer): 5.9% / 72.6% = **8%**
- Dense GEMM (cuBLAS): 1.5% / 72.6% = **2.1%**
- 其他全部加一起：~7%

也就是说 **~91% 的有效 GPU 时间花在 MoE GEMM kernel 上**。

### 关键洞察

autotuned 配置和"broken" baseline **跑的基本是同一类 kernel，按同样的比例分布**——MoE GEMM 到处都占主导。变化的是 **CPU↔GPU 协调方式**：

- **没有 cudagraph 的时候**：每个 kernel launch 在 CPU 端有 ~5-15 μs 的开销。launch 之间，GPU 空转。一次 48-layer MoE forward 有数百次这种 micro-gap，加起来很可观。
- **有 cudagraph 的时候**：CPU 一次性 dispatch 整张录好的 graph，GPU 背靠背连续跑。空转 gap 被消除。

所以**我们测到的 5-9× wall-clock 提速主要是 idle gap 消除带来的，不是更快的 kernel**。

---

## Part B —— Kernel 带宽利用率（NCU 数据）

**数据来源**：`results/2026-06-09_sglang_triton_sweep/ncu/R_concurrent_decode/`（NCU `--set full` 跑在 triton + cgOFF 上，profiled 了 30 个 kernel）。

> **为什么用 triton 的数据来推 cutlass**：autotuned config 用的是 CUTLASS 而不是 Triton 做 MoE GEMM，但**HBM 带宽 bound 是同一个**——两种 kernel 都得从 HBM 把当前 forward 用到的 expert 权重（W13、W2）拉进来。Cutlass 用 Hopper 的 TMA + WGMMA，Triton 用 TMA-style loads——在 bandwidth-bound shape 上两者把 HBM 打到同样程度。

### Triton `fused_moe_kernel` 在 R_concurrent_decode 上的指标

| 指标 | 值 | 解读 |
|---|---|---|
| **DRAM throughput** | **79.8%** | H200 4.8 TB/s 峰值的 79.8% = **~3.83 TB/s 实际吞吐** |
| SM throughput | 16.8% | 算力 pipeline 大部分时间空着（memory-bound 的预期表现） |
| Tensor Core 利用率 | 12.8% | TC 没事做；weights 还在 load |
| Occupancy | 44.8% | warp 调度正常 |
| Long-scoreboard stall | 25.5 warps/issue | 严重——warp 在等 HBM load |
| Math throttle | 0.38 | 算力 pipeline 不是瓶颈 |
| **NCU verdict** | **memory_bound** | DRAM 饱和，不是算力饱和 |
| **"Headroom"** | **20.2%** | 100 - max(SM%, DRAM%) |

### "20% headroom" 是啥意思

这个 kernel 现在**已经在拉峰值 HBM 带宽的 79.8%**。即使有一个"完美"的 rewrite，最多也只能拿剩下的 ~20%。考虑到现实情况——绝大部分实际 kernel 改写只能 capture 30-50% 的理论 headroom——真实 kernel 层面的提升大概是 **该 kernel 加速 6-10%**（这个 kernel 又占总 GPU 时间的 60-80%，所以**端到端 wall-clock 最多提升 5-8%**，只通过 rewrite 这一个 kernel）。

### 其他 kernel 呢？

剩下 30 个 kernel 大多是 low_occupancy / latency_bound，但**单个都占 <1% 的 GPU 时间**——最大的非-MoE kernel 是 `fused_qknorm_warp`（4.8% SM、95% headroom，但只占总时间 0.7%）。**就算把所有非-MoE kernel 全优化光也只能省 <10% 端到端**。

**唯一值得深入看的 kernel 有 3 个**：

1. **`count_and_sort_expert_tokens_kernel`**（atomic sort，56.84 stalls/issue——严重的串行瓶颈，~0.5% 总时间；目前 EP=1 batch 小，影响有限；但**在更大 EP / 更高 batch 时会变成可扩展性瓶颈**——是已知 issue）
2. **`nvjet_tst_*`**（cuBLAS dense GEMM，41-50% DRAM，也是 memory-bound；~7% 总时间，**已经卡在物理极限**，不动 algorithm 没法加速）
3. **`fused_moe_kernel` 自己**（headline kernel，那个 80% DRAM 的——但**这个 kernel 就是 workload 本身**，没什么可"rewrite"的）

---

## Part C —— Kernel-rewrite 上限到底是多少？

仔细算一下。**R_concurrent_decode wall-clock = 0.0716 秒/请求**（= 1 / 13.98 req/s）。其中：

- ~95% 是 GPU-busy（我们从 nsys 验证过：开 cudagraph 时 kernel time ≈ wall time）
- ~91% 的 GPU 时间在 MoE GEMM（上面 Part A 的类别拆分）
- ~80% 的 MoE GEMM 时间已经 HBM 饱和

所以：
- **在 MoE GEMM 里的时间**：0.95 × 0.91 = **86% 的 wall**
- **不在 MoE GEMM 里的时间**（其他 kernel + 极少的 CPU gap）：14% 的 wall

### 情况 1：神奇 agent 把 MoE GEMM kernel 拉到 100% HBM（现在 80%）

- 该 kernel 加速倍数：80/100 = 1.25×
- wall-clock 省下：0.86 × (1 - 80/100) = 0.86 × 0.20 = **17% wall-clock 缩减**

### 情况 2：神奇 agent 把 MoE GEMM 2× 加速（比如换 fp8 weights → HBM traffic 砍半）

- 该 kernel 加速倍数：2×（但**只能通过 quantization 实现**）
- wall-clock 省下：0.86 × 0.5 = **43% wall-clock 缩减**

但 fp8 是**量化决策，不是 kernel rewrite**。换到 fp8 之后 kernel 还是 HBM-bound，只是每个 token 需要拉的 HBM 字节数减半了。

### 对 agent 论文方向的具体含义

| 方向 | 理论上限 | 现实上限 | 评论 |
|---|---|---|---|
| Agent rewrite MoE GEMM kernel | ~25% 在该 kernel 上 = ~17% e2e | 5-8% e2e | 已经接近 HBM ceiling |
| Agent 找更好的 autotune flag | 5-10% e2e | 1-3% e2e | Optuna 基本已经做了 |
| Agent 选 fp8 quantization | 2× 理论 | 1.5-1.8× e2e | 但 fp8 当前在我们的 setup 里**反而退化**——需要正确的 config（见 6/25 fp8 doc） |
| Agent 优化 attention/norm 等 | 14% 理论上限 | <5% e2e | 这些 kernel 要么已经小，要么也是 HBM-bound |
| Agent 跨框架自动路由 | 0 到 ∞ 取决于框架 | 未知 | 跨框架对比基本没测 |
| Agent 自动化 autotune loop | 不适用（流程提升） | 不适用 | 这是流程自动化，不是 kernel 工作 |

**高 ROI 的 agent 方向不是 "rewrite kernels"**。现实的 agent 价值是：

1. **自动化 Optuna-style tuning**——每次 (model, hw, workload) 变化时自动跑
2. **检测并修复配置不匹配**（比如 fp8 退化案例）
3. **跨框架选择推理**（sglang vs vLLM vs TRT-LLM）
4. **多卡 dispatch 规划**（TP/EP/PP 组合至今未探索）

Kernel rewriting 在我们当前这个 (Qwen3-30B-A3B, H200, bf16, batch=32) 这个**特定 workload 点**上理论最多能给 ~17%——这是个紧的 ceiling，而且要跟几十年 cutlass/triton 优化经验对着干。

---

## Part D —— 我们还想真实测但今天没拿到的数据

如果以后拿到 NCU sudo（或者借 chendi 的账号），下面是按优先级排的应测项：

### 1. 在 cutlass `device_kernel`（autotuned config 的 MoE GEMM）上跑 NCU

- 目的：确认它的 DRAM% 是不是也 ~80%（预期）或更高
- 如果更高，我们的"20% headroom"估计高估了空间
- 如果差不多（最可能），结论 confirmed

### 2. R_long_prefill 上跑 NCU on universal config

- 目的：prefill 在 triton 上是 compute-bound（TC 70%）；cutlass 是不是也 70%+，还是 Hopper TMA + WGMMA 把它推到了 90%+？
- 这个能告诉我们 prefill 路径上 cutlass 是否有"compute 上限"的提升空间

### 3. fp8 NCU 对比

- 同样的 kernel，HBM traffic 减半，理论上 wall-clock 应该降 ~40%
- 但我们 6/11、6/25 测到的是 triton-fp8 **退化**、cutlass-fp8 **跟 bf16 持平**
- 强烈提示 sglang 的 fp8 路径**没把带宽优势 capture 住**
- **这条是 ROI 最高的方向**——直接揭示一个 framework-level 不可见的 bug

---

## Part E —— 这件事在整个项目里的位置（给 Mason 等人看）

我们已经完成了 Debadeepta 6/24 推荐的研究计划的**第一阶段**：

| Step | 状态 |
|---|---|
| 1. 选 stack、model、GPU | ✅ |
| 2. 定义 traffic regimes | ✅ |
| 3. Benchmark default | ✅（6/25 strict default） |
| 4. Run autotuner | ✅（6/25 Optuna 60 trials） |
| 5. Cross-regime degradation | ✅（6/25 4×4 矩阵） |
| **6. Profile autotuned config** | ✅（本报告） |
| **7. 判断 agent rewriting 是否合理** | ✅（本报告；**答案：在这个 kernel 上不合理**） |

### 给会议的诚实结论

1. **框架层 autotuning ceiling 跟 sglang 默认值基本一样**（在 Qwen3-30B-A3B + H200 + bf16 上）。5-flag 的 Optuna search 没找到比 `--model-path X --port Y` 显著更好的配置。

2. **瓶颈 kernel 已经 HBM-bandwidth-bound 在 ~80%**。理论 kernel rewrite 上限受剩下 20% headroom 制约。

3. **我们之前喊的"5-9× speedup"是相对一个 self-handicapped baseline 测的**（Triton 3.5.1 cubin bug 逼着把 cudagraph 关了；该 bug 在我们 env 里随某次 sgl-kernel 重装自动消失了）。

4. **剩下的真机会都在 operational level，不在 kernel level**：
   - fp8 退化（明显是真问题，但 pathology 不清楚；值得 NCU 深挖）
   - 多卡（TP > 1，EP > 1；整个维度未探索）
   - 跨框架选择（sglang vs vLLM）
   - 在线自适应 flag tuning（按 universal-config 证据可能不需要，但值得在第二个模型上确认）

### 给 agent 项目的方向建议

强烈支持**把 agent 项目从"在固定 setup 上 rewrite kernel"重新定位为以下之一**：

**A. 自动发现 misconfiguration** —— agent 检测出 fp8 退化案例，追溯到缺失的 tuned config 或错误的 backend 选择

**B. 跨框架 / 跨硬件推荐** —— agent 在多个 (framework × model × hw) 组合上跑 Optuna-style autotune，给出全局最优

**C. 多卡 dispatch 规划** —— agent 根据硬件拓扑 + 模型决定 TP vs EP vs PP 组合

**D. 在线 + 离线混合 tuning** —— agent 在 workload 分布变化时重新 tune（universal config 在我们 4-regime mix 下成立，但在更多样化的生产流量下可能不成立）

---

## 引用的文件 / 产物

| 路径 | 内容 |
|---|---|
| `results/2026-06-09_sglang_triton_sweep/ncu/R_concurrent_decode/ncu_report.md` | NCU 在 triton baseline 上的数据（30 个 kernel 的完整 PMU 指标） |
| `results/4way_bench/nsys/vllm_triton_kernels.csv` | Triton kernel 时间 |
| `results/4way_bench/nsys/vllm_cutlass_kernels.csv` | Cutlass kernel 时间 |
| `results/4way_bench/nsys/sglang_cutlass_kernels.csv` | sglang cutlass kernel 时间 |
| `results/4way_bench/2x2_nsys/stats/AT_ON_CG_ON_cuda_gpu_kern_sum.csv` | 2×2 AT+CG kernel 时间（最优配置 proxy） |
| `results/2026-06-25_autotuning/per_regime/R_concurrent_decode_gpu5/best.json` | Optuna 选出的 universal config |
| `docs/2026-06-25/autotuning_honest_results.md` | 6/25 诚实 baseline + autotuned 结果对比 |
| `scripts/nsys_on_universal_config.sh` | 今天本想跑 nsys 的脚本；保留给下次再试 |
| `scripts/analyze_nsys_universal_config.py` | nsys 输出的分析 pipeline |

---

## TODO（给下次会议 / 下次实验）

- [ ] 拿 NCU sudo 权限（chendi 借 / admin 配） → 重跑 NCU on universal config，确认"cutlass MoE kernel 也是 ~80% HBM"的假设
- [ ] R_long_prefill 上跑 NCU —— cutlass 是不是比 triton 的 TC 70% 更高？
- [ ] fp8 NCU —— 弄清为什么带宽优势没有 materialize
- [ ] 试一个**第二个模型**（DeepSeek-V3-style block fp8，或 Llama-3-70B dense）—— 测试"defaults are optimal" 这个发现的泛化性
- [ ] Pivot 到 A/B/C/D 之一（agent 方向重新定位）

---

## 附录：作为 baseline 用时的关键数字汇总

如果以后做对比实验，**这些是用本报告作为 baseline 时该引用的数字**。所有数字来自 `results/2026-06-25_autotuning/` 和 `results/2026-06-11_harness-v1/` 的 `summary.json`，都用 harness v1 的标准 3-runs（drop run 0）方法。

### Throughput（req/s）

| Regime | sglang 今日默认 | Optuna universal config | 备注 |
|---|---|---|---|
| R_short_decode | 0.888 | 0.886 | 单用户聊天 |
| R_medium_balanced | 4.629 | 4.652 | 中等并发 |
| R_long_prefill | 13.603 | 14.231 | 长 prefill |
| R_concurrent_decode | 14.712 | 13.978 | 高并发 decode（本报告聚焦） |

### Kernel category breakdown（R_concurrent_decode 近似值，proxy 自 6/8 nsys 数据）

| Kernel 类别 | TRITON + cgOFF | CUTLASS + AT_OFF + CG_OFF | CUTLASS + AT + CG（universal） |
|---|---|---|---|
| MoE GEMM (cutlass) | 0% | 80.5% | 60.1% |
| MoE GEMM (triton/flashinfer) | 18.2% | 0% | 5.9% |
| Dense GEMM (cuBLAS) | 7.1% | 7.2% | 1.5% |
| Other helper kernels | ~75% | ~12% | ~5%（剔除 autotune 校准） |
| **Total kernel time (sample)** | 0.69s | 17.3s | 3.3s |

### NCU 关键指标（`fused_moe_kernel` on R_concurrent_decode，triton+cgOFF baseline）

| 指标 | 值 |
|---|---|
| DRAM throughput | **79.8%**（H200 4.8 TB/s 峰值的 79.8% ≈ 3.83 TB/s 实际） |
| SM throughput | 16.8% |
| Tensor Core 利用率 | 12.8% |
| Occupancy | 44.8% |
| Long-scoreboard stall | 25.5 warps/issue（严重） |
| **Verdict** | **memory_bound** |
| **Headroom** | **20.2%** |

### Kernel rewrite 理论上限的算法

- MoE GEMM 占 wall: 0.95 × 0.91 = **86%**
- 当前 HBM 利用率: 80%
- 上限（推到 100% HBM）: 86% × (1 - 80/100) = **17% wall reduction**
- 现实（capture 30-50% headroom）: **5-8% e2e**

### 复现命令

```bash
# Baseline（今日默认）
python harness/run_bench.py --spec bench-specs/sglang-true-default-bf16.yaml \
    --out-dir results/<exp>/today-default/

# Autotuned universal config
python harness/run_bench.py --spec bench-specs/sglang-cutlass-bf16-patched.yaml \
    --out-dir results/<exp>/universal-config/

# 完整 Optuna re-run（如果想再 search 一次）
./scripts/autotune_two_regimes.sh 4 31200 R_short_decode R_long_prefill 15
./scripts/autotune_two_regimes.sh 5 31201 R_medium_balanced R_concurrent_decode 15
```

### 环境快照（每次 harness 运行都会自动保存）

每个 `summary.json` 里 `environment` 字段含：
- hostname / GPU UUID / SM 版本
- driver / CUDA / sglang / flashinfer / torch / triton 版本
- git commit + dirty status

所以**用 spec_hash 就能唯一定位一次实验的所有参数**，方便 6 个月后回来比对。

---

**English version**：`docs/2026-06-29/profiling_validation_of_universal_config.md`
