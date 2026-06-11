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

## 0. TL;DR（最想让 Ofer 记住的四件事）

1. **按默认配置，sglang 和 vLLM 在这个机器+模型上其实是打平的**（sglang_triton vs vllm_triton ≈ 1.00–1.05×）。**两边的默认 MoE backend 都是 Triton**——sglang 没有别的选项可走（cutlass 被排除在 autotune allowlist 外），vLLM 则是 oracle 主动把 Triton 排在 cutlass 前（`unquantized.py:71`，因为默认开 cudagraph 时 cutlass GPU kernel 反而慢一点）。**"sglang 慢"这个我们一开始以为的故事不成立**。
2. **真正的差距出现在"两边都手动强制走 CUTLASS"时：sglang 比 vLLM 慢 3.4–4.7×。**  根因不是 kernel 实现，是 sglang 的 cutlass 路径同时被 autotune 没跑 + cudagraph capture 失败两个 bug 打：fallback 到 flashinfer 内部 tactic 0，比 autotuned 慢 3–6×（CUTLASS microbench 实测）。**这是一条"可选优化路径被堵死"的问题，不是"默认配置太烂"。**
3. **"Triton fused_moe_kernel" 不是一个 kernel，是一族 kernel。**  4 个 regime 下 **Block / Grid / regs/thread / num_warps 全不同**，decode 阶段被 NCU 判为 *memory_bound*（TC 8%），prefill 阶段同源码却是 *compute-leaning*（TC 70%）。"瓶颈类型"会随 batch 和 seqlen 翻转，**单一 kernel-level 的优化建议不可移植**。
4. **autotune × cudagraph 必须成对开启才有大幅收益**（强制 vLLM CUTLASS 路径上的 2×2 矩阵）：单开任一个 1.0–1.5×，**两个同开 → 5.0×**。背后是 `latency = max(CPU_work, GPU_work)`：autotune 降 GPU，cudagraph 降 CPU，只降一边就被另一边卡住。这把以前所有"开了 cudagraph 没用 / 开了 autotune 没用"的负面报告解释清楚——也解释了为什么 sglang_cutlass 路径上"补一边"补不动。

---

## 1. 背景：项目原始动机 + 这周的认知更正

我们做的是 **end-to-end 自动优化 agent**，目标是给定 (model, hardware) 自动找出最优的 sglang 启动配置。第一步是搞清楚 sglang 在我们关心的场景里**到底慢在哪、为什么慢、和 vLLM 差在哪**。

**这周最重要的认知更正**：项目一开始我们带着"sglang 比 vLLM 慢"的假设进来，几天工作之后发现 —— 在 H200 + Qwen3-30B-A3B + 默认配置下，**sglang ≈ vLLM**（数据见 §2.3）。所谓 3.4–4.7× 的差距是我们**手动把两边都切到 CUTLASS** 才出现的，而那条路径是用户选择 SM100/B200 上更优 backend 时会走到的（在 SM90 上其实不该选）。

所以问题被重新定义为：**"sglang 在用户主动追求 CUTLASS 优化时被一系列工程 bug 堵住"**，而不是"sglang 默认就慢"。

工作平台：

- 硬件：1× NVIDIA H200 (SM90, 141GB HBM3e)
- 模型：Qwen3-30B-A3B（fine-grained MoE，128 routed experts，top-8 激活）
- 推理框架：sglang vs vLLM（最近的 main 分支）
- 量化：bf16（暂未引入 fp8/awq 等量化，避免变量过多）

---

## 2. vLLM vs sglang 默认 backend 行为对比

这是会议要回答的第一个问题：**两个框架默认会选什么 backend，差在哪？**

> ⚠️ **重要更正**：早期分析里我曾说"vLLM 默认 cutlass、sglang 默认 triton"——这是错的。**两边默认都是 Triton**。下面是核对清楚之后的事实。

### 2.1 默认行为

| 维度 | vLLM (main) | sglang (main) |
|---|---|---|
| **MoE backend（Qwen3-30B-A3B, H200，无任何 flag）** | **triton**（oracle `unquantized.py:71` 主动选） | **triton**（cutlass 在 allowlist 外，无法 autotune，实际用不起来） |
| Oracle 为何不选 cutlass | cudagraph 模式下 cutlass 的 GPU kernel 时间 > triton（PR #21872 数据） | 不存在 oracle，直接靠 allowlist 写死 |
| Attention backend | flashinfer | flashinfer |
| 启动期 autotune | **对 cutlass 开启**（`kernel_warmup()`，SM90+），但**默认不会触发，因为默认 backend 是 triton** | **对 cutlass 关闭**（cutlass 不在 allowlist），对 triton 通过 `@triton.autotune` 在线触发 |
| CUDA Graph | 默认开启 | 默认开启 |
| Triton autotune 触发时机 | 首次见到新 (M,N,K) 时通过 `@triton.autotune` 装饰器 | 同左 |

### 2.2 关键源码 anchor（已读、已验证）

- **vLLM oracle 把 Triton 排在 cutlass 前**：`vllm/.../unquantized.py:71` 注释 "FlashInfer is slower than Triton on Hopper"（这条注释在 PR #21872 的 cudagraph 场景下成立）
- **flashinfer 的 autotune 闸口**：`flashinfer/autotuner.py:432-451`
  ```python
  if not self.is_tuning_mode:
      return fallback_tactic   # 直接返回 tactic 0
  ```
  → 没进 tuning context 的话，所有 GEMM 都用 fallback，**不查 cache 也不挑形状**。
- **flashinfer fallback tactic 定义**：`flashinfer/.../flashinfer_cutlass_fused_moe_binding.cu:638`
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
  → vLLM 在 SM90+ 上**显式进入** autotune ctx；但**默认 backend 是 triton 的话，这段也跑，只是缓存出来的 cutlass 配置后续没人用到**。
- **sglang 的 autotune 排除**：`sglang/.../model_runner.py:1829-1857`，函数 `_should_run_flashinfer_autotune()`
  - allowlist 里**没有** cutlass（line 1841 注释：`# cutlass disabled, TODO: flashinfer compilation errors`）
  - 这意味着即使用户传 `--moe-runner-backend flashinfer_cutlass`，autotune 也不会跑

### 2.3 4-way bench 实测（warm-only req/s，3 regime × 3 runs）

来源：`docs/2026-06-08/sglang_vs_vllm_flashinfer_cutlass_analysis.md` §1

| Regime | sglang_triton（默认） | sglang_cutlass（手动强制） | vllm_triton（默认） | vllm_cutlass（手动强制） |
|---|---|---|---|---|
| R_short  | 3.22 | 0.71 | 3.31 | 3.32 |
| R_medium | 4.49 | 1.31 | 4.71 | 4.72 |
| R_long   | 4.50 | 1.33 | 4.44 | 4.52 |

两组相对比：

| 对比维度 | R_short | R_medium | R_long |
|---|---|---|---|
| **默认 vs 默认**（sglang_triton → vllm_triton） | 1.03× | 1.05× | 0.99× |
| **手动强制 cutlass 时**（sglang_cutlass → vllm_cutlass） | **4.70×** | **3.59×** | **3.41×** |
| **vLLM 自己** triton → cutlass | 1.00× | 1.00× | 1.02× |
| **sglang 自己** triton → cutlass | **0.22×**（4.5× 慢） | **0.29×**（3.4× 慢） | **0.29×**（3.4× 慢） |

CUTLASS microbench（H200 SM90，Qwen3-30B-A3B 真实 shape，仅 kernel 层）：

| batch | fallback (tactic 0) | tuned (best tactic) | 倍数 |
|---|---|---|---|
| 1 | 0.180 ms | 0.054 ms | **3.32×** |
| 8 | 0.855 ms | 0.146 ms | **5.87×** |
| 64 | 1.936 ms | 0.303 ms | **6.39×** |
| 2048 | 2.421 ms | 0.657 ms | **3.69×** |

**正确的结论**：

1. **默认配置下 sglang ≈ vLLM**（都走 Triton，几乎打平）。一开始以为的"sglang 比 vLLM 慢一大截"故事是错的。
2. **如果用户主动选 CUTLASS（合理诉求，因为 SM100/B200 上 cutlass 是更快的）**，sglang 上这条路径会比 vLLM 慢 3.4–4.7×。这是个**"可选优化被堵死"的工程 bug**，不是产品默认性能问题。
3. vLLM 自己内部 cutlass ≈ triton，**vLLM oracle 选 triton 是因为 cudagraph 下 cutlass 略慢**——这个判断在 SM90+H200 上是对的，在 SM100/B200 上会反过来。
4. **真正的科学问题**是：在 SM90+H200 上 cutlass 不值得选，那么 sglang 把 cutlass 排除其实和 vLLM oracle 判断一致；问题在于这个排除**是 hardcode 而不是 oracle**，到了 SM100/B200 上 sglang 仍然走不到 cutlass。

---

## 3. Backend 选择决策树 —— sglang 和 vLLM 各画一棵

这是会议上要白板演示的部分。**先看模型架构 → 再看量化 → 再看硬件 SM → 最后看是否被显式覆盖**。两边的代码组织方式完全不同：vLLM 把"按硬件优先级排序"做成了 oracle（每个量化方式一个 oracle 文件），sglang 把"按模型架构 hardcode"和"按量化 hardcode"分散在多处。

### 3.1 sglang 决策树（Qwen3-30B-A3B + H200 + bf16 走的真实路径加粗）

```
启动 sglang.launch_server
└─ ServerArgs.__post_init__ (server_args.py)
   ├─ 解析 hf_config → model_arch ∈ {DeepseekV3, GptOss, Llama4, Qwen3Moe, ...}
   ├─ Step 1: 按模型架构做 attention_backend 默认 (model-specific block)
   │   └─ ★ Qwen3MoeForCausalLM 没有专门 block，落到 generic 默认 (server_args.py:1780)
   │       if not use_mla_backend:                                    # ← MHA 架构
   │           if is_hopper_with_cuda_12_3():
   │               attention_backend = "fa3"          ★ H200 走这条
   │           elif is_sm100_supported():
   │               attention_backend = "trtllm_mha"
   │           elif is_hip():
   │               attention_backend = "aiter"
   │           else:
   │               attention_backend = "flashinfer" or "triton"
   │       else:  # MLA (DeepSeek)
   │           is_hopper → fa3, sm100 → flashinfer, hip → aiter
   │
   ├─ Step 2: 按模型架构做 moe_runner_backend 默认 (server_args.py:1559-1578)
   │   ★ Qwen3MoeForCausalLM block:
   │       if is_sm100_supported() and moe_runner_backend == "auto":
   │           if quantization in {fp8, modelopt_fp4, None}:
   │               moe_runner_backend = "flashinfer_trtllm"
   │   ★ H200 上 is_sm100_supported() == False → 这个 block 不改任何东西
   │   → moe_runner_backend 维持 "auto"
   │
   ├─ Step 3: 量化推断 (get_quantization_config)
   │   ★ Qwen3-30B-A3B-Instruct-2507 没有 quantization_config → quantization = None (bf16)
   │
   └─ initialize_moe_config() → MOE_RUNNER_BACKEND = MoeRunnerBackend.AUTO

模型加载 → 每层创建 FusedMoEMethod
└─ UnquantizedFusedMoEMethod.__init__ (quantization/unquant.py:162)
   └─ self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
                                          # ★ AUTO → False → 不走 cutlass branch
└─ UnquantizedFusedMoEMethod.create_moe_runner (unquant.py:321-330)
   └─ ★ HARDCODE: backend = MoeRunnerBackend.TRITON
      # 注意! 这里完全无视 AUTO，直接选 TRITON
      # 这意味着 bf16 MoE 的 "auto" 在 unquantized path 上 = TRITON
   └─ self.runner = MoeRunner(TRITON, ...)

每次 forward
└─ UnquantizedFusedMoEMethod.forward_cuda
   └─ if self.use_flashinfer_cutlass: ...
      elif use_aiter (only if backend.is_auto()): ...
      else: ★ self.runner.run() → TritonRunnerCore → @triton.autotune fused_moe_kernel
```

**最终路径**：sglang on H200 + Qwen3-30B-A3B + bf16 → **attention=fa3, MoE=triton (硬编码，AUTO 无效)**

> ⚠️ 这里有一个微妙陷阱：sglang `--moe-runner-backend auto` 在**不同量化方式下走不同路径**：
> - bf16 (unquant.py:321) → 直接 hardcode TRITON
> - fp8 (fp8.py:1345-1351) → 先看是否能用 deepgemm，再 fallback TRITON
> - awq (awq.py:828) → 各自不同
> - **"auto" 并不是一个统一的 resolver，而是 N 个 quantization 类各自决定的**

### 3.2 vLLM 决策树（同模型同硬件 bf16 走的真实路径加粗）

vLLM 把这件事做得更"engine"一些：每个量化方式对应一个 `oracle/*.py`，每个 oracle 暴露 `select_*_moe_backend(moe_config)`，返回 (backend, kernel_cls)。

```
启动 vllm serve
└─ ModelConfig.__post_init__
   ├─ 解析 hf_config → architectures = ["Qwen3MoeForCausalLM"]
   ├─ 量化推断 → quant_config = None (bf16)
   └─ 选 kernel_config → moe_backend = "auto" (默认)

模型加载 → FusedMoE layer 构造
└─ FusedMoEConfig(moe_backend="auto", ...)
└─ 进入 quant-specific MoE method
   ★ bf16 → UnquantizedFusedMoEMethod
   └─ select_unquantized_moe_backend(moe_config)  (oracle/unquantized.py:152)

select_unquantized_moe_backend 决策树:
├─ if is_cpu/tpu/oot: return 对应平台 backend
├─ if is_lora_enabled: return TRITON
├─ AVAILABLE_BACKENDS = _get_priority_backends(moe_config)
│   ├─ if rocm: [AITER, TRITON, BATCHED_TRITON]
│   ├─ if cuda:
│   │   ★ 初始 = [FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, TRITON, BATCHED_TRITON]
│   │   ★ if is_device_capability_family(90):     # ← H200/H100/H800
│   │       _move_to_back(FLASHINFER_TRTLLM)
│   │       _move_to_back(FLASHINFER_CUTLASS)
│   │       → [TRITON, BATCHED_TRITON, TRTLLM, CUTLASS]
│   │   ★ if dp_size > 1: _move_to_back(FLASHINFER_CUTLASS)   # Qwen3.5 BF16+DEP crash hack
│   │   (SM100/B200 上保持 TRTLLM > CUTLASS > TRITON 优先级)
│
├─ if runner_backend != "auto":                    # 显式 --kernel-config moe_backend=X
│   ★ map_unquantized_backend(X) → 直接走那个 backend
│
├─ if VLLM_USE_FLASHINFER_MOE_FP16:                # env 显式覆盖
│   ...
├─ if VLLM_ROCM_USE_AITER: ...
│
└─ ★ 默认：遍历 AVAILABLE_BACKENDS，挑第一个 is_supported_config() == True 的
   ★ H200 上第一个就是 TRITON → 选 TRITON

attention backend (略，类似的优先级表，H200 + MHA → FlashAttention v3 / FlashInfer 选其一)
```

**最终路径**：vLLM on H200 + Qwen3-30B-A3B + bf16 → **attention=FlashAttn/FlashInfer, MoE=triton (oracle 主动把 cutlass/trtllm 排后面)**

### 3.3 两边 oracle 的本质差异

| 维度 | sglang | vLLM |
|---|---|---|
| 决策入口 | 散在 `server_args.py` 各 `if model_arch in [...]` 块 + 各 `quantization/*.py` 的 `create_moe_runner` | 集中在 `fused_moe/oracle/*.py`，每个量化方式一个 oracle 文件 |
| AUTO 怎么 resolve | **每个量化类自己 hardcode**（bf16→Triton, fp8→DeepGemm/Triton, ...） | **统一一个 priority list**，按 SM 重排，遍历挑第一个 supported |
| 加新 backend 的代价 | 改 enum + 改每个 `create_moe_runner` + 可能改 `server_args.py` 各 model 块 | 加进 enum + 写 kernel_cls + 加进 oracle 优先级表 |
| 是否支持"在线 oracle"（runtime 切换） | 否（启动时定死） | 否（启动时定死），但 oracle 检查 `is_supported_config()` 可以做形状相关的 fallback |
| 改成"按硬件自动选最优"的难度 | 高（要在 N 个量化类各写一份） | 低（改 `_get_priority_backends` 一处） |

### 3.4 这棵树对我们 agent 的启示

- **agent 不能简单地"看 sglang 启动 log 显示 triton 就以为它有选 cutlass 的余地"**：bf16 路径上 cutlass 是 hardcode 不出现的，要真切到 cutlass 必须显式传 `--moe-runner-backend flashinfer_cutlass`，且会绕过 MoeRunner 直接调 flashinfer。
- **agent 要做硬件 → backend 推荐时，必须知道 (model_arch, quantization, SM) 三元组**，因为决策表是按这个三元组分叉的。
- **vLLM 的 oracle 模式适合自动化**（一处优先级 + 各 backend 自报 supported），sglang 的散落模式不适合自动化。这是一个**结构性观察**，会影响我们 agent 的工程选型——如果做 sglang 优化，需要先建立一份"sglang 真实决策路径表"，而 vLLM 直接读 oracle 文件就能拿到。

---

## 4. 新模型如何被导入 sglang / vLLM —— 三条路径

会议要回答的第三个问题。这一块用户经常误解为"框架自动支持 HuggingFace 上的任何模型"，实际不是。

### 4.1 路径 A：框架内已经手写实现（推荐路径，性能最好）

两边的模型注册机制几乎一样：

**sglang**（`sglang/srt/models/registry.py:124-130`）：
```python
ModelRegistry.register("sglang.srt.models")  # 自动扫描目录

# import_model_classes 会
# 1. pkgutil.iter_modules 遍历 sglang/srt/models/ 下的每个 .py
# 2. importlib.import_module(name) 导入
# 3. 找 module.EntryClass —— 一个 nn.Module 子类
# 4. 按 class.__name__ 注册到 ModelRegistry.models
```

每个模型一个文件，文件末尾导出 `EntryClass`。比如 `qwen3_moe.py:1151`:
```python
EntryClass = Qwen3MoeForCausalLM
```

支持加新模型有两种方式：
1. **进 sglang 主仓**：在 `sglang/srt/models/` 下加一个 `.py`，定义 `XxxForCausalLM(nn.Module)`，导出 `EntryClass = XxxForCausalLM`
2. **不进主仓（external package）**：实现同样接口的 package，启动时 `SGLANG_EXTERNAL_MODEL_PACKAGE=my_pkg sglang.launch_server ...`（`registry.py:132-133`）

**vLLM**（`vllm/model_executor/models/registry.py:71+`）：
```python
_TEXT_GENERATION_MODELS = {
    "AfmoeForCausalLM": ("afmoe", "AfmoeForCausalLM"),
    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
    ...
}
```
这是个**显式映射表**——比 sglang 的 auto-scan 更严格，加新模型必须手动加进这个 dict 才能被识别。vLLM 也支持 plugin 机制（实现 `vllm.plugins` entry point）。

### 4.2 路径 B：框架内**没有**手写，但 HuggingFace transformers 有 → fallback

两边都实现了一个 "TransformersForCausalLM" wrapper 作为 last-resort fallback：

**sglang**（`registry.py:73-76`）：
```python
# normalize_archs:
if len(normalized_arch) != len(architectures):
    normalized_arch.append("TransformersForCausalLM")
# 如果用户给的 arch 在 sglang 自己实现的 models 里找不到，
# 自动追加 TransformersForCausalLM 作为最后一个候选
```
`sglang/srt/models/transformers.py:142` 定义了 `TransformersForCausalLM(nn.Module)`，里面用 HF `AutoModel` 包一层。

**vLLM**（`registry.py:646-660`）：
```python
_TRANSFORMERS_BACKEND_MODELS = {
    "TransformersForCausalLM": ("transformers", "TransformersForCausalLM"),
    "SmolLM3ForCausalLM": ("transformers", "TransformersForCausalLM"),
    ...
}
```
然后在 `inspect_model_cls` 里走 `model_config._get_transformers_backend_cls()`。

**这条 fallback path 的 caveats**：
- 性能差：HF 实现没做 TP/PP optimization，没用 FlashAttention，CUDA Graph 也通常用不上
- backend 选择失灵：上面 §3 那些"按模型架构 hardcode"的 attention/MoE 都不会触发（因为 arch 名字是 `TransformersForCausalLM` 而不是 `Qwen3MoeForCausalLM`）
- 我们做 agent 时**必须检测当前是否走了 fallback path**，如果是的话，"backend 优化"是无意义的——优先建议用户去找/写一个原生实现

### 4.3 路径 C：用户带 `trust_remote_code=True` 的自定义代码

HF 上一些模型用 `auto_map` 指定自己的 `modeling_*.py`（比如 DBRX 早期）。这种情况下：

**vLLM**（`registry.py:1086-1108`）会先尝试 `try_get_class_from_dynamic_module`，从 HF 仓库下载 `modeling_*.py` 并 import；如果失败再 fallback 到 Transformers backend。
**sglang** 在我读到的代码里**没有显式的 dynamic-module path**——它直接走"在 ModelRegistry 找不到 → fallback 到 TransformersForCausalLM"。这是个**差距**，对支持冷门新模型 sglang 落后一步。

### 4.4 我们 agent 视角的 takeaway

| 决策点 | 检查方法 |
|---|---|
| 是否走了原生实现？ | sglang: `ModelRegistry.models` dict 里有 arch 名；vLLM: `_TEXT_GENERATION_MODELS` 里有 |
| 是否走了 transformers fallback？ | 启动 log 里有 `TransformersForCausalLM` 字样，或正在加载 `transformers.py` |
| 如果走 fallback，优化空间 | **大幅缩水**：所有 backend 自动选择都失灵，只能用 HF 默认 attention 实现 |
| Agent 应该做什么 | 优先检测 → 提示用户"换 backend 之前先写/找原生实现" |

---

## 5. sglang 一次请求的全流程拆解 —— 什么是静态的，什么是 runtime 才变的

会议要回答的下一个问题。下面以 Qwen3-30B-A3B + H200 为例。

### 5.1 在 **server 启动阶段** 决定、之后**不再变**的东西（静态）

| 维度 | 默认值（H200 + Qwen3-30B-A3B + bf16） | 一旦定下来怎么动 |
|---|---|---|
| `moe_runner_backend` | triton（unquant.py:321 hardcode，AUTO 失效） | 重启 server |
| Attention backend | **fa3**（FlashAttention v3，Hopper+CUDA12.3+ 默认） | 重启 server |
| 量化方式 | bf16 | 重启 server |
| **cudagraph 捕获集合**（哪些 batch size 被 capture） | 一组离散值（如 1,2,4,8,16,...,max） | 重启 server |
| AutoTuner 是否加载离线表 | **否**（H200 上没有 hand-tuned table 文件） | 改 flashinfer 源码 |
| AutoTuner 是否进入 tuning 模式 | **否**（sglang 把 cutlass 从 allowlist 排除了） | 改 sglang 源码 |
| 模型权重在 GPU 上的 layout | 一次性分配 | 重启 |

### 5.2 **每次 forward 都会重新决定**的东西（runtime）

| 维度 | 何时变 | 由谁选 |
|---|---|---|
| 选中的 expert IDs | 每个 token 不同 | top-k gating（依赖输入 hidden state） |
| Triton `fused_moe_kernel` 的 autotune specialization | 第一次见到新 (M, N, K) 时挑一次，之后 LRU cache | `@triton.autotune` 装饰器 + Triton runtime |
| cuBLAS Hopper GEMM 的 tile family（nvjet_*） | 每次 launch 都查 cuBLAS 内部 dispatch | cuBLAS（黑盒）依赖 (M, N, K) + dtype |
| 是否走 CUDA Graph replay | **仅当 batch size 在 capture 集合里**才走 | sglang scheduler |
| 是否进入 AutoTuner tuning ctx | **几乎从不**（只在 warmup 或显式 `with autotune():`） | 框架 |

### 5.3 **prefill ↔ decode 边界上变的东西**（运行时的"质变"）

| 维度 | Prefill | Decode |
|---|---|---|
| Effective M（attention/GEMM 的"行数"） | total tokens（如 8000） | batch_size（如 1–32） |
| Attention kernel 参数 | causal mask 全量计算 | incremental，KV cache 已有 |
| Triton fused_moe_kernel 的 block/grid | Block=256, Grid=12k–17k, regs/thread=194–196, num_warps=8 | Block=128, Grid=192–3.3k, regs/thread=56–64, num_warps=4 |
| 出现的额外 kernel | （较少） | `splitKreduce_kernel`, `FlashAttnFwdCombineKernel`, sampling clamp/topk 等 |
| TC 利用率 | 70%（compute-bound） | 8–13%（memory-bound） |

**关键洞察**：prefill 和 decode 在 sglang 里调的**是同一个 `fused_moe_kernel`（源码上）**，但 Triton 的 autotune 给它们选了**不同的 SASS specialization**，跑到 GPU 上其实是两个完全不同的 kernel（Block/Grid/寄存器/warp 数全不一样）。**所以"优化 fused_moe_kernel" 这句话需要先问"哪个 specialization"**。

### 5.4 一次 sglang 请求的时间线（schematic）

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

## 6. 4-regime 系统画像：sglang triton 后端的真实性能 profile

这是我们这周最完整的实验，已经做完。详细数据在 `docs/2026-06-09/sglang_triton_4regime_profiling.md`，会议上我打算只 show 两张表。

### 6.1 4 个 regime 的覆盖范围

| Regime | 模式 | 输入设定 |
|---|---|---|
| R_short_decode | 极小 batch 解码 | batch=1, 短 prompt |
| R_medium_balanced | 中等并发 | batch=8 |
| R_concurrent_decode | 高并发解码 | batch=32 |
| R_long_prefill | 长 prefill | 4 × 8000-token prompts |

每个 regime 跑了：bench (3 runs + stddev) + nsys (200MB sliced) + ncu (`--set full`，不过滤 kernel name)，共 140 个 unique kernel × full-set NCU。

### 6.2 同一个 fused_moe_kernel，4 个不同的"诊断结论"

| Regime | Block | Grid X | regs/thread | num_warps | SM% | DRAM% | TC% | NCU 判定 |
|---|---|---|---|---|---|---|---|---|
| R_short_decode (B=1) | 128 | 192–256 | 56 | 4 | 12 | 50 | 8 | low_occupancy |
| R_medium_balanced (B=8) | 128 | 1,536 | 64 | 4 | 14 | 67 | 10 | low_occupancy（边界） |
| R_concurrent_decode (B=32) | 128 | 3,288 | 64 | 4 | 17 | 80 | 13 | **memory_bound** |
| R_long_prefill | 256 | 12,768–17,024 | 194–196 | 8 | **70** | 22 | **70** | **compute-leaning** |

> ⚠️ 同一行 Python 源码（`vllm/sglang triton fused_moe_kernel`），autotuner 在不同形状下选了完全不同的 SASS，导致**性能分类完全相反**。

### 6.3 各类 kernel 在不同 regime 占的总时长

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

## 7. 2×2 (autotune × cudagraph) 矩阵 —— `max(CPU, GPU)` 模型的端到端证据

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

## 8. 其他有意思的发现 / 观察

### 8.1 `count_and_sort_expert_tokens_kernel` 的 atomics 瓶颈

- NCU long_scoreboard stall = **3182 warps/issue**（normal <2）
- 这个 kernel 用 atomics 给 expert 排序，**串行写**，warps 全在等
- cudagraph 救不了它（它是 GPU 内部的串行）
- 这是个 MoE 框架共性问题，**vLLM 也有**（暂未验证程度）

### 8.2 cuBLAS nvjet_* dense GEMM 的两副面孔

| 场景 | SM% | TC% |
|---|---|---|
| Prefill 大 GEMM | 94% | 96% |
| Decode B=1 GEMM | 7–8% | 7–8% |

- prefill 阶段 dense GEMM 已经接近**物理峰值**，**优化空间为 0**
- decode B=1 是物理形状决定的（M=1 GEMM），换 backend 也救不回来
- 这告诉我们：**优化 dense GEMM 完全没意义**，问题就在 MoE 和 attention

### 8.3 FlashAttention 的 TC 随 batch 急剧衰减

| 场景 | TC% |
|---|---|
| Prefill | 69% |
| Decode B=32 | 34% |
| Decode B=1 | **~4%** |

- 在 B=1 decode 时 attention 几乎不用 TC
- 这是 attention kernel 在 M=1 时的固有问题，**flash-attention 自己也救不了**
- 是否值得用 paged / persistent attention 重写，是个开放问题

### 8.4 sglang 把 flashinfer_cutlass 从 autotune allowlist 排除

`sglang/.../model_runner.py:1841`:
```python
# moe_runner_backend.CUTLASS,  # TODO: flashinfer compilation errors
```

这条 TODO 注释**至少存在了几个月**（git blame）。我们没有验证它到底还成立不成立。**这是个值得 flag 的工程问题**——可能只要把这行取消注释，sglang flashinfer_cutlass 路径就立刻能用上 autotune，瞬间补齐和 vLLM 的差距。

### 8.5 H200 / H100 没有 flashinfer 的 hand-tuned MoE config 表

- `flashinfer/data/v0_1_trtllm_fused_moe_NVIDIA_*.py`：**只有 B200 / GB200**
- 这意味着即使 sglang 修好了 autotune allowlist，**第一次跑还是要在线 autotune**（约 1–3 分钟）
- 我们可以离线 dump 一份 H200 的表，避免每次启动都重跑

### 8.6 sglang cudagraph + flashinfer_cutlass 的 capture 问题

- 我们试过 `--moe-runner-backend flashinfer_cutlass --enable-cuda-graph`
- capture 阶段直接 crash（细节在 `docs/2026-06-08/sglang_vs_vllm_flashinfer_cutlass_analysis.md`）
- 暂未排查根因。这是另一个会让"开了 cudagraph 没用"的隐藏 bug。

### 8.7 vLLM 的 startup autotune 实际不便宜（但用户感知不到，因为默认不走 cutlass）

- vLLM SM90+ 启动会跑 `flashinfer_autotune()`，多出约 60–90s（H200, Qwen3-30B-A3B）
- 但是！**默认 backend 是 Triton 的情况下，这段 autotune 出来的 cutlass cache 后续根本没人用**
- 只有当用户显式 `--kernel-config '{"moe_backend":"flashinfer_cutlass"}'` 时才发挥作用
- **这是个"暗收益"**：vLLM 替你提前买好了切到 cutlass 的票，但默认不让你用

---

## 9. 我们能看到什么 / 还看不到什么（工具能力诚实交代）

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

## 10. 想请教 Ofer 的开放问题

按优先级排序，希望在会议上听到他的意见：

0. **认知更正后，项目方向是否要调整**：默认配置下 sglang ≈ vLLM，"sglang 慢"不成立。那这个 end-to-end 优化 agent 的真正价值定位应该是 (a) 在 SM100/B200 这种新硬件上自动选对 backend、(b) 在用户主动选 cutlass 时自动绕开/修补 sglang 的工程 bug、还是 (c) 不局限 sglang，直接做"给定 (model, hw) 推荐最佳框架+backend+flag 组合"？
1. **sglang `# TODO: flashinfer compilation errors` 这条注释**：他知不知道现在还成不成立？是不是 sglang 那边的 P1 修复就能让我们少走半个月弯路？
2. **H200 上离线 dump flashinfer MoE config 表**：这件事到底是 sglang 的活、flashinfer 的活、还是用户的活？我们要不要主动贡献给 flashinfer？
3. **`count_and_sort_expert_tokens` 的 atomics 瓶颈**：vLLM 是不是也有？有没有内部知道的更好方案（segment sort? cub::DeviceRadixSort?）
4. **regime 设计**：我们手工挑的 4 个 regime 是不是合理？production 上是否有更值得关注的 (batch, seqlen) 分布？
5. **跨框架 transfer**：我们这套 (4 regime × bench + nsys + ncu + unified) 流水线要不要也对 vLLM / TensorRT-LLM 做一遍，建立同样基线？
6. **SM100 / B200 access**：能不能给我们一台短期借用？目前所有结论都局限在 SM90，没法验证"cutlass 在 SM100 上反超 triton"这条 vLLM oracle 行为的另一半。

---

## 11. 下一步规划（不会议上展开，仅备查）

- 把同样 4 regime 在 vLLM 上跑一遍，直接出对比表
- 试着 patch sglang 把 cutlass 加回 autotune allowlist，看 autotune 是否还 crash
- 离线生成 H200 的 flashinfer MoE config 表
- 把 cross-regime-anomaly skill 跑在这次的 4 regime sweep 上
- 把目前 agent 的 14 个 skill 用一个 README 串起来，让 demo 能跑通

---

## Appendix A：关键源码 anchors 一览（方便会议上 jump-to）

**Autotune / fallback 相关：**

| 主题 | 文件 : 行 |
|---|---|
| flashinfer autotune gate | `flashinfer/autotuner.py:432-451` |
| flashinfer fallback tactic | `flashinfer/.../flashinfer_cutlass_fused_moe_binding.cu:638` |
| flashinfer load_from_file | `flashinfer/autotuner.py:316-332` |
| vLLM 的 kernel_warmup | `vllm/model_executor/warmup/kernel_warmup.py:55-138` |
| sglang autotune allowlist | `sglang/python/sglang/srt/model_executor/model_runner.py:1829-1857`（cutlass 在 1841 被注释） |

**Backend 选择决策树（§3）相关：**

| 主题 | 文件 : 行 |
|---|---|
| sglang ServerArgs 模型-arch dispatch | `sglang/python/sglang/srt/server_args.py:1290-1700`（每个 `elif model_arch in [...]` 块） |
| sglang Qwen3Moe block | `sglang/.../server_args.py:1559-1578` |
| sglang generic attention default | `sglang/.../server_args.py:1780-1830`（Hopper+CUDA12.3 → fa3） |
| sglang `MoeRunnerBackend` 枚举 | `sglang/.../layers/moe/utils.py:57-100` |
| sglang `initialize_moe_config` | `sglang/.../layers/moe/utils.py:148-180` |
| sglang **unquant create_moe_runner（hardcode TRITON）** | `sglang/.../layers/quantization/unquant.py:321-330` |
| sglang fp8 create_moe_runner | `sglang/.../layers/quantization/fp8.py:1340-1360` |
| vLLM unquantized MoE oracle | `vllm/.../fused_moe/oracle/unquantized.py:152-300` |
| vLLM `_get_priority_backends` | `vllm/.../oracle/unquantized.py:44-87`（SM90 把 cutlass/trtllm 推后） |
| vLLM `map_unquantized_backend` | `vllm/.../oracle/unquantized.py:137-150` |

**模型注册（§4）相关：**

| 主题 | 文件 : 行 |
|---|---|
| sglang `_ModelRegistry` | `sglang/python/sglang/srt/models/registry.py:17-100` |
| sglang auto-scan + EntryClass | `sglang/.../models/registry.py:93-130` |
| sglang Transformers fallback | `sglang/.../models/registry.py:73-76` + `models/transformers.py:142,289` |
| vLLM `_TEXT_GENERATION_MODELS` | `vllm/model_executor/models/registry.py:71-700`（显式 dict） |
| vLLM `_TRANSFORMERS_BACKEND_MODELS` | `vllm/.../models/registry.py:646-660` |
| vLLM `trust_remote_code` dynamic load | `vllm/.../models/registry.py:1086-1128` |

## Appendix B：实验产物路径

- `results/2026-06-09_sglang_triton_sweep/` —— 4 regime × bench/nsys/ncu/unified 全部数据
  - `README.md` 是导航索引
  - 4 份 `profile_unified.json` 是规范化的最终产物
  - 4 份 `ncu_report.md` 是 per-regime 人类可读报告
- `regimes/qwen3_30b_moe_sglang_perf_sweep.yaml` —— regime 定义
- `.github/skills/` —— 15 个 skill，主线四件套：e2e-bench-runner / nsys-timeline-sql / ncu-microarch / profile-summary-unified
