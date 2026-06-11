# Ofer 会议草稿：sglang 推理优化项目 —— 近期发现与问题定义

> 时间：2026-06-11 准备稿
> 作者：@gujialiang123（chendi mentor 协助）
> 模型 / 硬件：Qwen3-30B-A3B（128 experts，A=8）on H200 (SM90)
> 目的：汇报近 4–5 天在 sglang vs vLLM 推理路径上的发现。**重点是"我们看到了什么"和"问题是什么"，不在本次会议上提解决方案。**
> 关联文档（细节都在里面，会议上引用）：
> - `docs/2026-06-08/sglang_vs_vllm_flashinfer_cutlass_analysis.md`
> - `docs/2026-06-08/vllm_2x2_autotune_cudagraph_matrix.md`
> - `docs/2026-06-08/nsys_2x2_validation_and_nsys_usage.md`
> - `docs/2026-06-09/cutlass_vs_triton_e2e_investigation.md`（含中文版 `.zh.md`）
> - `docs/2026-06-09/sglang_triton_4regime_profiling.md`（含中文版 `.zh.md`）
> - `docs/2026-06-08/agent_profiling_capability_audit.md`

---

## 0. TL;DR（最想让 Ofer 记住的三件事）

1. **vLLM 比 sglang 在同一台 H200 上跑同一模型快 3.4–4.7×。**  根因不是 kernel 实现，而是 **backend 选择 + autotune 启动逻辑**：vLLM 默认走 flashinfer_cutlass + 启动期 autotune，sglang 默认走 triton；即使把 sglang 强行切到 flashinfer_cutlass，它的 autotune allowlist 也把 cutlass 排除掉了，最终落到 flashinfer 内部的 fallback tactic 0，比 autotuned 配置慢 3–6×（CUTLASS microbench 实测）。
2. **"Triton fused_moe_kernel" 不是一个 kernel，是一族 kernel。**  在 4 个 regime 下，**Block / Grid / regs/thread / num_warps 全都不同**，因此 decode 阶段它被 NCU 判为 *memory_bound*，prefill 阶段同一行源码却是 *compute-leaning (TC 70%)*。"瓶颈类型"会随 batch 和 seqlen 翻转，**单一 kernel-level 的优化建议不可移植**。
3. **autotune 和 cudagraph 必须成对开启**才有大幅收益。2×2 矩阵实测：单开任一个只有 1.0–1.5× 提升，**两个同时开启 → 5.0× 提升**。背后的物理模型是 `latency = max(CPU_work, GPU_work)`：autotune 降 GPU，cudagraph 降 CPU，只降一边就被另一边卡住。这把以前所有"开了 cudagraph 没用 / 开了 autotune 没用"的负面报告全部解释清楚。

---

## 1. 背景：项目原始动机

我们做的是 **end-to-end 自动优化 agent**，目标是给定 (model, hardware) 自动找出最优的 sglang 启动配置。第一步是搞清楚 sglang 在我们关心的场景里**到底慢在哪、为什么慢、和 vLLM 差在哪**。这份报告就是这一步的产物。

工作平台：

- 硬件：1× NVIDIA H200 (SM90, 141GB HBM3e)
- 模型：Qwen3-30B-A3B（fine-grained MoE，128 routed experts，top-8 激活）
- 推理框架：sglang vs vLLM（最近的 main 分支）
- 量化：bf16（暂未引入 fp8/awq 等量化，避免变量过多）

---

## 2. vLLM vs sglang 默认 backend 行为对比

这是会议要回答的第一个问题：**两个框架默认会选什么 backend，差在哪？**

### 2.1 默认行为

| 维度 | vLLM (main) | sglang (main) |
|---|---|---|
| MoE backend（Qwen3-30B-A3B, H200） | **flashinfer_cutlass** | **triton** |
| Attention backend | flashinfer | flashinfer |
| 启动期 autotune | **默认开启**（`kernel_warmup()`，SM90+） | **默认关闭** for cutlass; 仅对 triton/flashinfer_trtllm 开 |
| CUDA Graph | 默认开启 | 默认开启 |
| 静态化策略 | autotune 在 warmup 阶段完成、结果写入内存表 | triton autotune 通过 `@triton.autotune` 装饰器在线进行（首次调用时） |

### 2.2 关键源码 anchor（已读、已验证）

- **flashinfer 的 autotune 闸口**：`flashinfer/autotuner.py:432-451`
  ```python
  if not self.is_tuning_mode:
      return fallback_tactic   # 直接返回 tactic 0
  ```
  → 没进 tuning context 的话，所有 GEMM 都用 fallback，**不查 cache 也不挑形状**。
- **fallback tactic 定义**：`flashinfer/.../flashinfer_cutlass_fused_moe_binding.cu:638`
  ```cpp
  return mAllProfiles.front();   // 第 0 条 profile
  ```
- **load_from_file 只支持少数 SKU**：`flashinfer/autotuner.py:316-332`
  - 会找 `v0_1_trtllm_fused_moe_NVIDIA_<DEVICE>.py`
  - **B200 / GB200 有手工调优表，H200 / H100 没有**
- **vLLM 的 warmup 流程**：`vllm/model_executor/warmup/kernel_warmup.py:55-138`
  ```python
  if device_capability.major >= 9:
      with fi_utils.autotune():
          flashinfer_autotune(...)
  ```
  → vLLM 在 SM90+ 上**显式进入** autotune ctx，跑一遍 dummy forward 把 cache 填满。
- **sglang 的 autotune 排除**：`sglang/.../model_runner.py:1829-1857`，函数 `_should_run_flashinfer_autotune()`
  - allowlist 里**没有** cutlass（line 1841 注释：`# cutlass disabled, TODO: flashinfer compilation errors`）
  - 这意味着即使用户传 `--moe-runner-backend flashinfer_cutlass`，autotune 也不会跑

### 2.3 结果（端到端实测）

CUTLASS microbench（H200, SM90, Qwen3-30B-A3B 真实 shape）：

| batch | fallback (tactic 0) | tuned (best tactic) | 倍数 |
|---|---|---|---|
| 1 | 0.180 ms | 0.054 ms | **3.32×** |
| 8 | 0.855 ms | 0.146 ms | **5.87×** |
| 64 | 1.936 ms | 0.303 ms | **6.39×** |
| 2048 | 2.421 ms | 0.657 ms | **3.69×** |

端到端 throughput（同 regime，同 batch，同 prompt）：

| 配置 | req/s |
|---|---|
| sglang + triton（默认） | baseline = 1.0 |
| vLLM + flashinfer_cutlass + autotune + cudagraph | **3.4–4.7×** |

**结论**：vLLM 快不是因为它的 kernel 写得更好，是因为它 **(a) 默认就走 cutlass**，**(b) 启动期帮你跑完 autotune**，**(c) cudagraph 一起开**。sglang 三件事都没做到，所以慢。

---

## 3. sglang 一次请求的全流程拆解 —— 什么是静态的，什么是 runtime 才变的

会议要回答的第二个问题。下面以 Qwen3-30B-A3B + H200 为例。

### 3.1 在 **server 启动阶段** 决定、之后**不再变**的东西（静态）

| 维度 | 默认值 | 一旦定下来怎么动 |
|---|---|---|
| `moe_runner_backend` | triton | 重启 server |
| Attention backend | flashinfer | 重启 server |
| 量化方式 | bf16 | 重启 server |
| **cudagraph 捕获集合**（哪些 batch size 被 capture） | 一组离散值（如 1,2,4,8,16,...,max） | 重启 server |
| AutoTuner 是否加载离线表 | **否**（H200 上没有 hand-tuned table 文件） | 改 flashinfer 源码 |
| AutoTuner 是否进入 tuning 模式 | **否**（sglang 把 cutlass 从 allowlist 排除了） | 改 sglang 源码 |
| 模型权重在 GPU 上的 layout | 一次性分配 | 重启 |

### 3.2 **每次 forward 都会重新决定**的东西（runtime）

| 维度 | 何时变 | 由谁选 |
|---|---|---|
| 选中的 expert IDs | 每个 token 不同 | top-k gating（依赖输入 hidden state） |
| Triton `fused_moe_kernel` 的 autotune specialization | 第一次见到新 (M, N, K) 时挑一次，之后 LRU cache | `@triton.autotune` 装饰器 + Triton runtime |
| cuBLAS Hopper GEMM 的 tile family（nvjet_*） | 每次 launch 都查 cuBLAS 内部 dispatch | cuBLAS（黑盒）依赖 (M, N, K) + dtype |
| 是否走 CUDA Graph replay | **仅当 batch size 在 capture 集合里**才走 | sglang scheduler |
| 是否进入 AutoTuner tuning ctx | **几乎从不**（只在 warmup 或显式 `with autotune():`） | 框架 |

### 3.3 **prefill ↔ decode 边界上变的东西**（运行时的"质变"）

| 维度 | Prefill | Decode |
|---|---|---|
| Effective M（attention/GEMM 的"行数"） | total tokens（如 8000） | batch_size（如 1–32） |
| Attention kernel 参数 | causal mask 全量计算 | incremental，KV cache 已有 |
| Triton fused_moe_kernel 的 block/grid | Block=256, Grid=12k–17k, regs/thread=194–196, num_warps=8 | Block=128, Grid=192–3.3k, regs/thread=56–64, num_warps=4 |
| 出现的额外 kernel | （较少） | `splitKreduce_kernel`, `FlashAttnFwdCombineKernel`, sampling clamp/topk 等 |
| TC 利用率 | 70%（compute-bound） | 8–13%（memory-bound） |

**关键洞察**：prefill 和 decode 在 sglang 里调的**是同一个 `fused_moe_kernel`（源码上）**，但 Triton 的 autotune 给它们选了**不同的 SASS specialization**，跑到 GPU 上其实是两个完全不同的 kernel（Block/Grid/寄存器/warp 数全不一样）。**所以"优化 fused_moe_kernel" 这句话需要先问"哪个 specialization"**。

### 3.4 一次 sglang 请求的时间线（schematic）

```
client → tokenizer_manager (Python)
     → scheduler (batch + waiting queue)
     → model_executor.forward()
         ├── 如果 batch_size 在 cudagraph capture 集合:
         │       cuda_graph_replay()    ← CPU 几乎不参与，GPU 全速跑
         └── 否则:
                 for each layer:
                     attention (flashinfer)
                     MoE (triton fused_moe_kernel + count_and_sort_expert_tokens + topk)
                     norm/elementwise
         → sampling
     → detokenizer
     → response stream
```

**CPU 工作量瓶颈点**（不开 cudagraph 时）：
- Python 调度 (~每 layer 一次 kernel launch + param packing)
- `count_and_sort_expert_tokens_kernel` 内部用了 atomics 排序 expert，stall reason `long_scoreboard = 3182 warps/issue`（正常 <2），**这是个 sequential bottleneck**，cudagraph 解决不了。

---

## 4. 4-regime 系统画像：sglang triton 后端的真实性能 profile

这是我们这周最完整的实验，已经做完。详细数据在 `docs/2026-06-09/sglang_triton_4regime_profiling.md`，会议上我打算只 show 两张表。

### 4.1 4 个 regime 的覆盖范围

| Regime | 模式 | 输入设定 |
|---|---|---|
| R_short_decode | 极小 batch 解码 | batch=1, 短 prompt |
| R_medium_balanced | 中等并发 | batch=8 |
| R_concurrent_decode | 高并发解码 | batch=32 |
| R_long_prefill | 长 prefill | 4 × 8000-token prompts |

每个 regime 跑了：bench (3 runs + stddev) + nsys (200MB sliced) + ncu (`--set full`，不过滤 kernel name)，共 140 个 unique kernel × full-set NCU。

### 4.2 同一个 fused_moe_kernel，4 个不同的"诊断结论"

| Regime | Block | Grid X | regs/thread | num_warps | SM% | DRAM% | TC% | NCU 判定 |
|---|---|---|---|---|---|---|---|---|
| R_short_decode (B=1) | 128 | 192–256 | 56 | 4 | 12 | 50 | 8 | low_occupancy |
| R_medium_balanced (B=8) | 128 | 1,536 | 64 | 4 | 14 | 67 | 10 | low_occupancy（边界） |
| R_concurrent_decode (B=32) | 128 | 3,288 | 64 | 4 | 17 | 80 | 13 | **memory_bound** |
| R_long_prefill | 256 | 12,768–17,024 | 194–196 | 8 | **70** | 22 | **70** | **compute-leaning** |

> ⚠️ 同一行 Python 源码（`vllm/sglang triton fused_moe_kernel`），autotuner 在不同形状下选了完全不同的 SASS，导致**性能分类完全相反**。

### 4.3 各类 kernel 在不同 regime 占的总时长

| Regime | moe_gemm | dense_gemm | attention | norm | moe_routing | other |
|---|---|---|---|---|---|---|
| R_short_decode | 31.5% | 31.2% | 16.0% | 5.3% | 7.0% | 5.3% |
| R_medium_balanced | 47.7% | 19.8% | 13.9% | 3.6% | 4.7% | 4.1% |
| R_long_prefill | 47.4% | 21.0% | 14.8% | 3.6% | 4.7% | 2.7% |
| R_concurrent_decode | **54.4%** | 11.1% | 12.3% | 3.0% | 3.5% | 4.0% |

**洞察**：
- B=1 解码时 MoE 和 dense GEMM 各占 ~30%，**没有"主瓶颈"**，dense 也得优化
- B=32 并发解码时 MoE 占到 **54%**，是绝对核心
- 长 prefill 看起来 MoE 占比高（47%），但**绝对吞吐其实没问题**（TC=70%），优化空间不大
- norm / elementwise 在所有 regime 都被 NCU 判为 *tensor_core_idle*（TC <2%），在 prefill 时 DRAM 67–92%，**这是 fusion 的候选**

---

## 5. 2×2 (autotune × cudagraph) 矩阵 —— `max(CPU, GPU)` 模型的端到端证据

详细数据在 `docs/2026-06-08/vllm_2x2_autotune_cudagraph_matrix.md`，简要总结：

vLLM CUTLASS MoE on R_medium：

|  | cg_ON | cg_OFF |
|---|---|---|
| **at_ON**  | **4.66** req/s | 1.01 req/s |
| **at_OFF** | 1.36 req/s | 0.93 req/s |

- 两个都关 = 0.93 baseline
- 只开 autotune（at_ON/cg_OFF）= 1.09×
- 只开 cudagraph（at_OFF/cg_ON）= 1.47×
- **两个都开 = 5.03×**（>1.09 × 1.47 = 1.60，说明不是简单乘法，是 `max` 模型解锁后两边同时下降）

物理解释：
```
total_latency ≈ max(CPU_dispatch_time, GPU_compute_time)
- autotune 降 GPU
- cudagraph 降 CPU
- 只降一边的话 max() 卡在另一边，看起来"没效果"
```

我们用 nsys 直接量了 4 个组合的 CPU 空闲 gap 和 GPU SM 占用率，**和理论一致**（细节在 `docs/2026-06-08/nsys_2x2_validation_and_nsys_usage.md`）。

---

## 6. 其他有意思的发现 / 观察

### 6.1 `count_and_sort_expert_tokens_kernel` 的 atomics 瓶颈

- NCU long_scoreboard stall = **3182 warps/issue**（normal <2）
- 这个 kernel 用 atomics 给 expert 排序，**串行写**，warps 全在等
- cudagraph 救不了它（它是 GPU 内部的串行）
- 这是个 MoE 框架共性问题，**vLLM 也有**（暂未验证程度）

### 6.2 cuBLAS nvjet_* dense GEMM 的两副面孔

| 场景 | SM% | TC% |
|---|---|---|
| Prefill 大 GEMM | 94% | 96% |
| Decode B=1 GEMM | 7–8% | 7–8% |

- prefill 阶段 dense GEMM 已经接近**物理峰值**，**优化空间为 0**
- decode B=1 是物理形状决定的（M=1 GEMM），换 backend 也救不回来
- 这告诉我们：**优化 dense GEMM 完全没意义**，问题就在 MoE 和 attention

### 6.3 FlashAttention 的 TC 随 batch 急剧衰减

| 场景 | TC% |
|---|---|
| Prefill | 69% |
| Decode B=32 | 34% |
| Decode B=1 | **~4%** |

- 在 B=1 decode 时 attention 几乎不用 TC
- 这是 attention kernel 在 M=1 时的固有问题，**flash-attention 自己也救不了**
- 是否值得用 paged / persistent attention 重写，是个开放问题

### 6.4 sglang 把 flashinfer_cutlass 从 autotune allowlist 排除

`sglang/.../model_runner.py:1841`:
```python
# moe_runner_backend.CUTLASS,  # TODO: flashinfer compilation errors
```

这条 TODO 注释**至少存在了几个月**（git blame）。我们没有验证它到底还成立不成立。**这是个值得 flag 的工程问题**——可能只要把这行取消注释，sglang flashinfer_cutlass 路径就立刻能用上 autotune，瞬间补齐和 vLLM 的差距。

### 6.5 H200 / H100 没有 flashinfer 的 hand-tuned MoE config 表

- `flashinfer/data/v0_1_trtllm_fused_moe_NVIDIA_*.py`：**只有 B200 / GB200**
- 这意味着即使 sglang 修好了 autotune allowlist，**第一次跑还是要在线 autotune**（约 1–3 分钟）
- 我们可以离线 dump 一份 H200 的表，避免每次启动都重跑

### 6.6 sglang cudagraph + flashinfer_cutlass 的 capture 问题

- 我们试过 `--moe-runner-backend flashinfer_cutlass --enable-cuda-graph`
- capture 阶段直接 crash（细节在 `docs/2026-06-08/sglang_vs_vllm_flashinfer_cutlass_analysis.md`）
- 暂未排查根因。这是另一个会让"开了 cudagraph 没用"的隐藏 bug。

### 6.7 vLLM 的 startup autotune 实际不便宜

- vLLM 启动会多出约 60–90s（H200, Qwen3-30B-A3B）
- 用户感知是"vLLM 启动慢"，但运行起来快 3–4×
- **这是个延迟换吞吐的隐式权衡**，文档里没有强调

---

## 7. 我们能看到什么 / 还看不到什么（工具能力诚实交代）

详细在 `docs/2026-06-08/agent_profiling_capability_audit.md`。

### 能看到
- 每个 kernel 的 launch params (block/grid/regs/warps)
- 每个 kernel 的 SM%, DRAM%, TC% (NCU `--set full`)
- 每个 kernel 的 stall reason 分布
- 每条 stream 的时间线、idle gap (nsys SQL)
- CPU 端的 cudaLaunch / cudaMemcpy 调用次数和位置 (nsys API trace)
- 一次请求的端到端 throughput / TTFT / TPOT 分布 (sglang bench)

### **还看不到**
- cuBLAS / cublasLt 内部到底选了哪条 kernel variant（黑盒）
- flashinfer cutlass profile cache 命中率（没暴露 metric）
- 跨 layer 的 expert 选择稳定性（要自己改 sglang 加 hook）
- multi-GPU 的通信开销（目前实验都是单卡）
- 实际 production trace 上的 batch 分布（regime 是我们手工构造的，未必和真实工作负载一致）

---

## 8. 想请教 Ofer 的开放问题

按优先级排序，希望在会议上听到他的意见：

1. **sglang `# TODO: flashinfer compilation errors` 这条注释**：他知不知道现在还成不成立？是不是 sglang 那边的 P1 修复就能让我们少走半个月弯路？
2. **H200 上离线 dump flashinfer MoE config 表**：这件事到底是 sglang 的活、flashinfer 的活、还是用户的活？我们要不要主动贡献给 flashinfer？
3. **`count_and_sort_expert_tokens` 的 atomics 瓶颈**：vLLM 是不是也有？有没有内部知道的更好方案（segment sort? cub::DeviceRadixSort?）
4. **regime 设计**：我们手工挑的 4 个 regime 是不是合理？production 上是否有更值得关注的 (batch, seqlen) 分布？
5. **profile-driven optimization agent 的形态**：他希望这个 agent 最终是
   - (a) 自动给用户出 "用 vLLM 而不是 sglang" 这种顶层建议
   - (b) 自动 patch sglang 源码补齐缺失能力
   - (c) 只做诊断、把锅丢给框架社区
6. **跨框架 transfer**：我们这套 (4 regime × bench + nsys + ncu + unified) 流水线要不要也对 vLLM / TensorRT-LLM 做一遍，建立同样基线？

---

## 9. 下一步规划（不会议上展开，仅备查）

- 把同样 4 regime 在 vLLM 上跑一遍，直接出对比表
- 试着 patch sglang 把 cutlass 加回 autotune allowlist，看 autotune 是否还 crash
- 离线生成 H200 的 flashinfer MoE config 表
- 把 cross-regime-anomaly skill 跑在这次的 4 regime sweep 上
- 把目前 agent 的 14 个 skill 用一个 README 串起来，让 demo 能跑通

---

## Appendix A：关键源码 anchors 一览（方便会议上 jump-to）

| 主题 | 文件 : 行 |
|---|---|
| flashinfer autotune gate | `flashinfer/autotuner.py:432-451` |
| flashinfer fallback tactic | `flashinfer/.../flashinfer_cutlass_fused_moe_binding.cu:638` |
| flashinfer load_from_file | `flashinfer/autotuner.py:316-332` |
| vLLM 的 kernel_warmup | `vllm/model_executor/warmup/kernel_warmup.py:55-138` |
| sglang autotune allowlist | `sglang/python/sglang/srt/model_executor/model_runner.py:1829-1857`（cutlass 在 1841 被注释） |

## Appendix B：实验产物路径

- `results/2026-06-09_sglang_triton_sweep/` —— 4 regime × bench/nsys/ncu/unified 全部数据
  - `README.md` 是导航索引
  - 4 份 `profile_unified.json` 是规范化的最终产物
  - 4 份 `ncu_report.md` 是 per-regime 人类可读报告
- `regimes/qwen3_30b_moe_sglang_perf_sweep.yaml` —— regime 定义
- `.github/skills/` —— 15 个 skill，主线四件套：e2e-bench-runner / nsys-timeline-sql / ncu-microarch / profile-summary-unified
