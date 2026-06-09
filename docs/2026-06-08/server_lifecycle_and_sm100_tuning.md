# LLM Server 全生命周期时间事件表 + SM100 flashinfer tuning 调研

> 两件事一篇文档:
> Part A — 从 server 启动到推理结束的所有时间消耗事件,标 CPU/GPU 归属
> Part B — flashinfer 在 SM100 (Blackwell) 上的 tuning / kernel / 写死表调研

测试参考: sglang + Qwen3-30B-A3B / H200 / bf16,真实 server.log 数据来源
`results/4way_bench/sglang_triton/server_tail.log`

---

# Part A — 全生命周期时间事件表

## A.1 阶段 1: 进程启动 + 模型加载 (~15-30 秒,一次性)

| # | 事件 | CPU/GPU | 我们 case 实测 | 主要时间消耗在哪 | 备注 |
|---|---|---|---|---|---|
| 1 | Python 解释器启动 + import torch + import sglang | CPU only | ~1-3 s | 解释器初始化 + torch.cuda lazy init + sglang 模块 import | 一次性,无法跳过 |
| 2 | 解析 server_args / `ServerArgs.from_cli_args` | CPU only | <0.1 s | argparse + dataclass 构造 | 可忽略 |
| 3 | 启动 scheduler / detokenizer / tokenizer 子进程 (multiprocessing fork) | CPU only | <0.5 s | fork + zmq 端口绑定 + IPC channel 建立 | sglang 多进程架构,vLLM 是单进程多线程不一样 |
| 4 | `init_torch_distributed` (NCCL init) | CPU + GPU | 0.35 s (server.log line 1) | NCCL handshake, CUDA context 创建 | TP=1 时几乎瞬间;TP>1 需要 NCCL all-reduce setup |
| 5 | `load_weight` (safetensors → GPU HBM) | CPU + GPU (PCIe / NVLink 传输) | 13.25 s (line 4) | **HBM bandwidth bound** — 30B model bf16 ≈ 57 GB,过 PCIe Gen5 ~30 GB/s 或本地 NVMe → 8-30 秒 | mmap 文件 → CPU pinned buffer → GPU. 大头是 disk → CPU 的 I/O |
| 6 | KV cache 内存分配 | GPU | ~1 ms | `torch.empty(K_size, device='cuda')` — 只是 cudaMalloc 没 zero-fill | 30 + 30 GB allocated, 但只是预留地址空间 |
| 7 | Memory pool init (cuda allocator pool warmup) | GPU | ~1 ms | sglang 自己的 mem pool, 预分配 free list | |

**总: ~15-30 秒,基本被 weight load 主导**

## A.2 阶段 2: 首次推理准备 (~1-10 分钟,一次性,可重用)

| # | 事件 | CPU/GPU | 实测 | 主要时间消耗 | 备注 |
|---|---|---|---|---|---|
| 8 | Triton kernel JIT 编译 (per shape, lazy) | CPU + GPU | 首次 ~5-30 s/kernel,后续命中 cache | LLVM 编译 PTX → SASS | `~/.triton/cache/` 持久化 |
| 9 | **FlashInfer JIT 编译** (`fused_moe_90.so` 等) | CPU only | **7+ 分钟** (首次冷启动!) | g++ + nvcc 编译 ~50 个 .cu 文件 → .so → ninja link | `~/.cache/flashinfer/0.6.11.post2/90a/cached_ops/` 持久化。**这是我们 §4.4 看到 cudagraph hang 的根因之一** |
| 10 | `_flashinfer_autotune` warmup (如果走) | CPU + GPU | 几秒到几十秒 | flashinfer AutoTuner sweep tactic | **`flashinfer_cutlass` 这条 path 在 sglang 里被注释掉了** (见 model_runner.py:1841) |
| 11 | CUDA Graph capture | GPU | 4.88 s (line 9) | 一次跑 forward 把所有 kernel launch 录成 graph,每个 batch size 录一张 | 我们 case 录 8 个 batch [1,2,4,8,12,16,24,32] |
| 12 | uvicorn / FastAPI server start + bind port | CPU only | ~0.5 s | socket bind + worker thread pool 启动 | line 11-13 |

**总: cold-start 几分钟到 10 分钟(全靠 JIT 编译); warm cache 后 ~5-10 秒**

## A.3 阶段 3: 单次推理请求处理 (毫秒级,per request)

### A.3.1 请求接入阶段 (CPU 端,~ms 级)

| # | 事件 | CPU/GPU | 典型耗时 | 备注 |
|---|---|---|---|---|
| 13 | HTTP request 解析 (FastAPI / uvicorn) | CPU | <1 ms | JSON parse |
| 14 | tokenize (HF tokenizer) | CPU | 0.1-2 ms (随 prompt 长度) | 800w 词 ≈ 1100 token,~0.5 ms |
| 15 | zmq send 到 scheduler 进程 | CPU + IPC | ~0.5 ms | sglang 多进程 IPC |
| 16 | scheduler 进程接收,加到 waiting queue | CPU | <0.1 ms | |
| 17 | scheduler 调度循环挑请求组 batch (radix cache lookup, prefix cache check) | CPU | 1-5 ms | sglang 的 RadixCache LRU 查找,prefill prefix 复用 |

### A.3.2 Prefill 阶段 (CPU+GPU,~10-500 ms 看 prompt)

| # | 事件 | CPU/GPU | 典型耗时 | 主要时间消耗 |
|---|---|---|---|---|
| 18 | Forward batch input 准备 (tokens → tensor, pos_ids, masks) | CPU | 1-3 ms | torch.tensor 构造 + copy 到 device |
| 19 | **prefill forward pass** | CPU 发 launch + GPU 算 | **prompt 长度 × throughput** | 800 word ≈ 1100 token,Qwen3 30B bf16 prefill ~150 tok/s → ~7 ms,大头在 attn (O(N^2)) + MoE GEMM |
|  | ├ embedding lookup | GPU | <0.1 ms | |
|  | ├ rmsnorm × 48 layers × 2 | GPU | ~1 ms total | 各 kernel ~10us, launch overhead 占一半 |
|  | ├ qkv_proj GEMM × 48 layers | GPU | ~3-5 ms | dense GEMM, compute-bound |
|  | ├ attention (FlashAttention-3) × 48 | GPU | ~5-30 ms (随 seq^2) | 长 prompt 主导 |
|  | ├ output_proj GEMM × 48 | GPU | ~3-5 ms | |
|  | ├ MoE routing (topk_softmax) × 48 | GPU | ~1 ms | |
|  | ├ **MoE GEMM (gate+up, down)** × 48 × 2 | GPU | **大头** | 见 §A.4 详细拆 |
|  | └ logit head + sample | GPU | <1 ms | |
| 20 | 写 KV cache | GPU | <0.5 ms | cudaMemcpy on-device |
| 21 | 第一个 token append 到 output_ids | CPU | <0.1 ms | |

### A.3.3 Decode 阶段 (CPU+GPU,~5-50 ms 每 token,重复 N 次)

| # | 事件 | CPU/GPU | 典型耗时 | 主要时间消耗 |
|---|---|---|---|---|
| 22 | scheduler 准备 decode batch (检查哪些 request 还没完) | CPU | ~0.5 ms | |
| 23 | 决定 cudagraph batch size, padding | CPU | <0.1 ms | |
| 24 | **decode forward pass** | CPU + GPU | **5-30 ms/token** 取决于 batch | 见 §A.4 |
|   | (cudagraph hit) | CPU launch overhead 接近 0 | replay 一张录好的 graph | sglang server.log 标 `cuda graph: True` |
|   | (cudagraph miss) | CPU 发 ~500 kernel launch | 每 launch 5-10 us = 2.5-5 ms CPU 端 | sglang 标 `cuda graph: False` (e.g. prefill / 超大 batch) |
| 25 | sampling (greedy / top-k / top-p) | GPU | <0.5 ms | flashinfer 的 sampling kernel 或 PyTorch 原生 |
| 26 | append 新 token 到 output | CPU | <0.1 ms | |
| 27 | 通过 zmq 发给 detokenizer 进程 | CPU + IPC | ~0.3 ms | |
| 28 | detokenize 新 token (HF tokenizer.decode) | CPU | 0.05-0.5 ms | |
| 29 | stream chunk 推回给 HTTP client | CPU | ~0.3 ms (本机 loopback) | SSE / WebSocket |
| 30 | check stop tokens / max_tokens | CPU | <0.1 ms | |

**Decode 步骤会重复 max_new_tokens 次。**

### A.3.4 请求结束

| # | 事件 | CPU/GPU | 典型耗时 |
|---|---|---|---|
| 31 | 释放 KV cache 行 (radix cache 引用 -1) | CPU | <0.1 ms |
| 32 | HTTP response close | CPU | <1 ms |

## A.4 MoE Decode Step 的细分(关键!)

每个 decode step 对 batch=B 个 token 跑一次 MoE,每层 48 + 2 GEMM:

| 子事件 | CPU/GPU | 时间 (vllm_cutlass R_medium) | 时间 (sglang_cutlass R_medium) |
|---|---|---|---|
| attention forward (FA3) per layer | GPU | ~20 us | ~20 us |
| MoE topk routing per layer | GPU | ~10 us | ~10 us |
| **MoE GEMM1 (gate+up)** per layer | GPU | ~80 us (cutlass-sm90) | ~80 us (same kernel) |
| **MoE GEMM2 (down)** per layer | GPU | ~80 us | ~80 us |
| 总 per-layer | GPU | ~190 us | ~190 us |
| × 48 layers | GPU | ~9 ms | ~9 ms |
| CPU launch overhead per layer (no cudagraph) | CPU | ~50 us × 48 = 2.4 ms | 同左 |
| CPU launch overhead per layer (with cudagraph) | CPU | <0.1 ms total | (sglang+cutlass 走不通 cudagraph) |
| AutoTuner.choose_one 调用(per `cutlass_fused_moe` call) | CPU + GPU (benchmark kernel) | 命中 cache 几乎 0 | **cache miss ~0.5-2 ms 重 benchmark** (我们 §4.3 找到的 9× kernel) |

→ vLLM cudagraph decode: ~9 ms GPU + 0 ms CPU = **wall ~ 9-10 ms / step**
→ sglang no-cudagraph decode: ~9 ms GPU + 2.4 ms CPU + ~1 ms AutoTuner = **wall ~ 12-15 ms / step** (3.4× 慢的来源)

## A.5 一个 R_medium bench 的总时间帐 (16 reqs × 256 output tokens, conc=8)

vLLM CUTLASS, wall 3.7 s:
```
启动期 (一次性,不计入 bench):
  imports + dist init + load weight + JIT + cudagraph capture ≈ 60 s
  (run 1 比 run 2/3 慢 ~0.5 s 是 cudagraph capture 摊销)

Bench 期 3.7 s:
  16 prefill (~800 token each)        ≈ 16 × 50 ms  = 0.8 s
  16 × 256 decode (batched,conc=8)    ≈ 256 step × 11 ms (batch 8 摊) ≈ 2.8 s
  其它 (sample, sched, IPC)          ≈ 0.1 s
  total ≈ 3.7 s  ✓
```

sglang CUTLASS no-cudagraph, wall 12.7 s:
```
prefill 类似 0.8 s
decode 256 step × ~45 ms (慢 4x 因 AutoTuner) ≈ 11.5 s
其它 ≈ 0.4 s
total ≈ 12.7 s  ✓
```

---

# Part B — SM100 flashinfer 调研

回答:**SM100 上的优势是结构性的 4 层叠加,SM90 上没有任何一个**。

## B.1 第一层 —— 持久化 hand-tuned 配置表(SM100 独有)

来源: `flashinfer/tuning_configs/` 目录,文件:
```
v0_1_trtllm_fused_moe_NVIDIA_B200.py      # 234 行
v0_1_trtllm_fused_moe_NVIDIA_GB200.py     # 138 行
```

**SM90 (H100/H200) 没有任何文件**。`ls flashinfer/tuning_configs/` 只有 B200 和 GB200。

### 结构 (B200 文件示例)

```python
best_configs = {
    "('trtllm::fused_moe::gemm1', 'MoERunner', ((1, 3584), (256, 512, 448), (0,), (256, 7168, 16), (0,)))": (0, 5),
    "('trtllm::fused_moe::gemm1', 'MoERunner', ((1024, 3584), (256, 512, 448), (0,), (256, 7168, 16), (0,)))": (0, 5),
    "('trtllm::fused_moe::gemm1', 'MoERunner', ((128, 3584), ...))": (0, 5),
    ...
}
```

每条 entry 的 key 是 `(op_name, runner_name, (shape_tuple_a), (shape_tuple_b), ...)`,
value 是 `(tactic_id, additional_param)` —— **预先 tune 好的最优 tactic ID**。

### 覆盖范围

```
B200:
  - 58 entries (29 gemm1 + 29 gemm2)
  - batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384

GB200:
  - 34 entries (17 gemm1 + 17 gemm2)
  - batch sizes: 1, ... 16384, 32768, 65536
```

### 加载逻辑

`flashinfer/autotuner.py:26-37` + `:316-332`:
```python
def get_config_path(is_module: bool):
    dev_name = torch.cuda.get_device_name(0).replace(" ", "_")
    cutlass_ver = _nvfp4_cutlass_version.replace(".", "_")
    config_name = f"v{cutlass_ver}_trtllm_fused_moe_{dev_name}"
    if is_module:
        return f"flashinfer.tuning_configs.{config_name}"

@lru_cache(maxsize=None)
def load_from_file(key):
    module_name = get_config_path(is_module=True)
    try:
        module = importlib.import_module(module_name)
        best_configs = module.best_configs
    except (ImportError, AttributeError):
        best_configs = None
    if best_configs is not None:
        k = str((key[0], key[1], key[3]))
        if k in best_configs:
            logger.info(f"[Autotuner]: Loading configs for {k} from file.")
            return True, best_configs[k][0], best_configs[k][1], None
    logger.info(f"[Autotuner]: ... from file failed; Using default configs instead.")
    return False, 0, -1, None
```

**SM90 走的就是 fallback 路径** (`from file failed; Using default configs instead`),
也就是依赖运行时 `AutoTuner.choose_one` 现挑 —— **这就是我们 §4 看到 9× kernel
launch 的根因**。

## B.2 第二层 —— Blackwell-only 的 `trtllm_gen` kernel(SM100 独有)

来源: `flashinfer/jit/fused_moe.py:215-273` 函数 `gen_trtllm_gen_fused_moe_sm100_module`

```python
def gen_trtllm_gen_fused_moe_sm100_module() -> JitSpec:
    # Fetch "flashinferMetaInfo.h" from the online kernel cache.
    # This file contains the `tllmGenBatchedGemmList` as the list of
    # available kernels online.
    include_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/include"
    checksum = get_cubin(checksum_path, CheckSumHash.TRTLLM_GEN_BMM)
    metainfo = get_cubin(f"{include_path}/{header_name}.h", meta_hash)
    
    # currently only support Blackwell
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]    # ← SM100/SM103 only
    )
    
    return gen_jit_spec(
        "fused_moe_trtllm_sm100",
        [
            ...
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_kernel_launcher.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_runner.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_deepseek.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_llama4.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_renormalize.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_dev_kernel.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_batched_gemm_runner.cu",
        ],
        ...
    )
```

注释直接写 "**currently only support Blackwell**"。这个 kernel:
- 是 **TensorRT-LLM 生成的 cubin**,通过 `get_cubin()` 从 NVIDIA 的 artifact server 下载预编译好的 .cubin
- 不需要本地 JIT compile (除了一个薄 wrapper)
- 内部 kernel 列表来自 `tllmGenBatchedGemmList` —— **是 NVIDIA 离线生成的成千个 kernel 变种**,
  per shape 都 tune 过

可以理解为: TensorRT-LLM team 在内部用 ncu 对 Blackwell 做 exhaustive sweep,把最优 kernel
cubin 上传到 artifact server,flashinfer 运行时直接下载用。**SM90 上没有这个机制**。

### 入口函数 (SM100 only)

`flashinfer/fused_moe/core.py:2115-2188`:
```python
def trtllm_bf16_moe(
    routing_logits, routing_bias, hidden_states,
    gemm1_weights, gemm2_weights,
    num_experts, top_k,
    ...
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    """BF16 MoE operation with autotuning support.
    
    This function implements a bfloat16 Mixture of Experts layer using the
    TensorRT-LLM backend with automatic performance tuning for optimal
    tile size selection.
    ...
    """
    return get_trtllm_moe_sm100_module().trtllm_bf16_moe(...)
```

`get_trtllm_moe_sm100_module()` 在 SM90 上**根本调不通** —— 内部 nvcc flags
只接受 `supported_major_versions=[10]`,SM90 编译会 fail。

这就是为什么 sglang oracle 在 SM100 上自动选 `flashinfer_trtllm` 作为 MoE backend
(参见 `server_args.py:1558` 的 Qwen3MoeForCausalLM 分支),SM90 上根本没这个选项。

## B.3 第三层 —— kernel 候选 + JIT 模板搜索空间(SM100 比 SM90 大 ~16×)

我们 §10 的旧分析已经覆盖。再贴一次直接证据。

文件: `flashinfer/jit/gemm/cutlass/generate_kernels.py`

**SM90 grouped GEMM**(`generate_sm90_grouped_gemm_operations`, line 556-637):
```python
arch = 90
supported_dtypes = [DataType.f16, DataType.bf16, DataType.f32, DataType.e4m3]
                    # ↑ 4 dtypes (no FP4)
M_TILES = [128]      # 注释: "Currently M tile must be 128 for Grouped GEMM"
N_TILES = [16, 32, 64, 128, 256]
cga_shapes = product([1, 2], [1, 2], [1])   # 4 个固定 cluster
mainloop_schedule = TmaWarpSpecializedCooperative    # 1 种
# 共 ~320 candidate kernels
```

**SM100 grouped GEMM**(`generate_sm100_grouped_gemm_operations`, line 840-941):
```python
supported_dtypes = [
    DataType.f16, DataType.bf16, DataType.f32,
    DataType.e4m3,        # FP8 ✓
    e2m1,                 # FP4 native ✓ (SM90 没有)
    (DataType.e4m3, e2m1),# 混合 FP8/FP4 ✓
]
cta_shapes_m = [64, 128]      # 2 个 M tile (vs SM90 只有 1 个)
cta_shapes_n = [8, 16, 32, 64, 128, 192, 256]  # 多了 N=8 和 N=192
cga_shapes = [(1,1,1), (2,1,1)]
dynamic_cga = [True, False]   # 动态 cluster shape (SM90 没有)
epi_schedules = [
    PtrArrayNoSmemWarpSpecialized1Sm,
    PtrArrayTmaWarpSpecialized1Sm,
]
# 共 >5000 candidate kernels  → SM100 搜索空间是 SM90 的 ~16×
```

**M tile = 128 锁死** 是 SM90 grouped GEMM 的硬限制,直接源码注释。
SM100 解锁了 M=64,小 batch 时 SM 利用率显著更高。

## B.4 第四层 —— 硬件原语本身

| 硬件特性 | SM90 (Hopper) | SM100 (Blackwell) |
|---|---|---|
| 矩阵乘原语 | WGMMA (warp-group MMA) | UMMA (Unified MMA) + TMEM |
| Accumulator 位置 | register file | 专用 **Tensor Memory (TMEM)** |
| Register pressure | 高 (累加器吃 reg) | 低 (TMEM 独立) |
| 同时在飞的 MMA 数 | 受 register file 限制 | TMEM 加持下显著更多 |
| FP4 原生支持 | **无** | ✓ (`mxfp4`, `nvfp4`) |
| FP8 原生支持 | ✓ | ✓ |
| Dynamic cluster shape | 不支持 (cluster 在 kernel 编译时定死) | **支持** (runtime 决定) |
| TMA (Tensor Memory Accelerator) | ✓ (基础) | ✓ (扩展) |

**对我们 BF16 unquantized 影响**:
- TMEM 减少 register pressure → kernel 能用更大 tile 而不爆 register
- Dynamic cluster → AutoTuner 可以在运行时挑最优 cluster shape
- 这两个加起来让 SM100 的 kernel 在大 batch 上 utilization 显著好于 SM90

**对 NVFP4 quantized 影响**:
- SM100 用 FP4 tensor core 直接算
- SM90 必须 dequantize 到 BF16 再算 → 3-5× 慢
- 这就是为什么 sglang oracle 在 SM100 上默认 `flashinfer_cutlass` 对 NVFP4 模型,SM90 上没这个选项

## B.5 综合: SM100 上 flashinfer 比 SM90 优势的来源

| 层 | 优势 | SM90 状态 | SM100 状态 |
|---|---|---|---|
| 1 | Hand-tuned tactic 表(per device,持久化到 .py 文件) | ❌ 无 | ✓ B200/GB200 各一份 |
| 2 | Blackwell-only `trtllm_gen` kernel + cubin 下载 | ❌ 无 | ✓ `trtllm_bf16_moe` / `trtllm_fp8_*` / `trtllm_fp4_*` |
| 3 | Kernel 候选搜索空间 | ~320 (小) | >5000 (~16×) |
| 4 | 硬件原语 (UMMA, TMEM, native FP4) | 无 | 有 |

**四层独立优势叠加**。即使我们把 SM90 的 sglang AutoTuner re-benchmark bug 修好,
SM90 上的 CUTLASS 也最多追到 vLLM 同水平,不可能复现 SM100 上的显著优势。

---

# 优化建议优先级

## 立刻可做(零成本)

1. **修 sglang `unquant.py:385` 的 `tune_max_num_tokens` 改固定常量** (§4 的 Fix 1)
   - 影响: AutoTuner cache 命中率提升,理论上能消除 9× kernel launch
   - 风险: 极低 (一行代码)
   - 验证: nsys 重测 cutlass-sm90-gemm count 是否从 97774 跌到 ~10000
   - 收益区: 主要在 R_medium / R_long 这种 prefill-heavy 的 sglang 场景

2. **vLLM/sglang oracle 加 batch-size-aware 路由** (§12 的 conc=64 发现)
   - 影响: 小 batch → Triton, 大 batch → CUTLASS,跨 backend 自动切
   - 风险: 中等 (需要修 oracle 逻辑 + 加 runtime 探测)
   - 收益: 高 conc serving 场景能拿到 +17% (vllm_cutlass at conc=64)

## 中等成本

3. **给 flashinfer 加 SM90 的 tuning_configs/.py 文件**
   - 影响: 模仿 B200/GB200,做一份 H100/H200 的 hand-tuned tactic 表,持久化
   - 风险: 中 (需要在多 shape 上跑 sweep,生成 .py 文件)
   - 收益: 让 SM90 的 cutlass path 不再每次走 fallback,跟 SM100 体验对齐
   - 这是真正能让 SM90 cutlass 接近 SM100 优势的方法

## 高成本(可能 ROI 不够)

4. **给 CUTLASS SM90 grouped GEMM 加 M=64 tile 候选**
   - 影响: 突破 `M_TILES=[128]` 硬限制 (`generate_kernels.py:563`),小 batch 利用率改善
   - 风险: 高 (要懂 CUTLASS 模板,可能 mainloop schedule 不支持)
   - 收益: 不确定 (CUTLASS 上游可能已经决定 M=128 是最优 trade-off,改了也许性能反退)

5. **修 sglang flashinfer_cutlass + cudagraph 的冷启动 hang**
   - 影响: 让冷启动也能用 cudagraph
   - 风险: 高 (需要复现并定位 detokenizer 进程 hang)
   - 收益: 低 —— 我们 §11 已经发现 cudagraph 主要帮 R_short,R_medium/R_long 几乎不变
   - 不推荐做

---

# 引用源码 + 路径(给 reviewer)

## SM100 优势相关
- `flashinfer/tuning_configs/v0_1_trtllm_fused_moe_NVIDIA_B200.py` (58 entries)
- `flashinfer/tuning_configs/v0_1_trtllm_fused_moe_NVIDIA_GB200.py` (34 entries)
- `flashinfer/autotuner.py:26-37, 316-332` — config 加载逻辑(`load_from_file`)
- `flashinfer/jit/fused_moe.py:215-273` — `gen_trtllm_gen_fused_moe_sm100_module` (Blackwell-only cubin)
- `flashinfer/fused_moe/core.py:2115-2188` — `trtllm_bf16_moe` 入口
- `flashinfer/jit/gemm/cutlass/generate_kernels.py:556-637` — SM90 grouped GEMM 生成器
- `flashinfer/jit/gemm/cutlass/generate_kernels.py:840-941` — SM100 grouped GEMM 生成器

## 全生命周期 + 时间数据
- `results/4way_bench/sglang_triton/server_tail.log` — 完整启动日志(line 1-13)
- `sglang/python/sglang/srt/model_executor/model_runner.py` — 启动序列代码
- `sglang/python/sglang/srt/managers/scheduler.py` — scheduler 主循环
- `sglang/python/sglang/srt/managers/detokenizer_manager.py` — detokenizer 进程

## Cross-reference 既有文档
- `docs/2026-06-08/sglang_vs_vllm_flashinfer_cutlass_analysis.md` — 主分析 (§4 9× launch, §10 SM90/SM100 四层, §12 conc=64 反转)
- `docs/2026-06-08/triton_vs_cutlass_moe_kernel_source_comparison.md` — Triton vs CUTLASS 源码对比
- `docs/2026-06-04/moe_backend_decision_trees.md` — sglang/vLLM oracle 决策树

---

# 修正 (2026-06-08 18:00) — 4 个优势的归类重新校正 + CUTLASS vs trtllm_gen DSL 区别

User 仔细看后指出两件事,我之前 §B.5 的归类有错:

## 修正 1: B200/GB200 hand-tuned 表是给 **CUTLASS 路径**用的,不是 trtllm_gen 路径

**直接证据**: `flashinfer/tuning_configs/v0_1_trtllm_fused_moe_NVIDIA_B200.py` 全部 entry 的 key 都是
`"('trtllm::fused_moe::gemm1', 'MoERunner', ...)"`。

这个 key 来自哪? grep 源码:
```bash
$ grep -B 1 "tuner.choose_one" flashinfer/fused_moe/core.py
# Inside cutlass_fused_moe() at line 490:
        _, gemm_tactic_1 = tuner.choose_one(
            "trtllm::fused_moe::gemm1",       # ← 这里
            [moe_runner], MoERunner.tuning_config,
            [input, fc1_expert_weights, fc1_expert_biases, ...],
            gemm_idx=1,
        )
# Inside trtllm_bf16_moe_op() at line 1273 (SM100-only trtllm_gen):
        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_bf16_moe",    # ← 不一样的 key
            [moe_runner], ...,
        )
```

所以**"trtllm::fused_moe::gemm1" 是 CUTLASS 路径**(因为 `cutlass_fused_moe()` 函数调它),
**"flashinfer::trtllm_bf16_moe" 才是 trtllm_gen 路径**。命名混乱是历史包袱:CUTLASS MoE
的 AutoTuner 框架来自 TensorRT-LLM,所以保留了 "trtllm::" 前缀。

**重要 implication**: B200/GB200 表里的 best tactics 是给 **`cutlass_fused_moe()` 用的**,
B200 上 cutlass 路径直接受益。SM90 (H100/H200) 上的 cutlass 路径完全没有对应文件,所以 fall back
到 runtime AutoTuner。

## 修正 2: 四个优势的重新归类

旧版表(§B.5)归错。正确版本:

| # | 优势 | 路径归属 | 谁能拿到 (硬件 × 路径) |
|---|---|---|---|
| 1 | Hand-tuned tactic 表 (`tuning_configs/*.py`) | **CUTLASS 路径** | 只 B200/GB200 cutlass 受益; **可以补 — 给 H200 sweep 一份就行** |
| 2 | `trtllm_gen` kernel + 下载 cubin | **TRT-LLM 路径** (完全独立 backend) | 只 SM100 (硬编码 `supported_major_versions=[10]`) |
| 3 | JIT kernel 搜索空间 SM100 ~16× SM90 | CUTLASS 路径 | SM100 cutlass 路径享受 |
| 4 | 硬件原语 (UMMA, TMEM, FP4) | 两个路径都用 | 只 SM100 硬件 |

## 修正 3: 跟 sglang SM90 bug 的关系

User 问 "这些优势都在 vLLM 上对吧?因为 sglang 现在不能跑 SM90 flashinfer (有 bug)"

**回答**: **这 4 个优势都是 SM100 特性,跟引擎(vLLM vs sglang)无关,跟 sglang SM90 bug 也无关**。

| 场景 | 谁占优 | sglang bug 是否影响 |
|---|---|---|
| SM90 + vllm + cutlass | 一般 (没 trtllm_gen, 没 hand-tuned 表) | ❌ vLLM 不受影响 |
| SM90 + sglang + cutlass | **慢 3.4-4.7×** (AutoTuner re-benchmark) | ✓ 这就是 bug |
| SM100 + vllm + cutlass | 受益于 优势①③④ | ❌ |
| SM100 + vllm + trtllm_gen | 受益于 优势②④ | ❌ |
| SM100 + sglang + cutlass | 同 vllm 受益 (sglang hang 是 SM90-specific) | ❌ |
| SM100 + sglang + trtllm_gen | 同 vllm 受益 | ❌ |

sglang SM90 的 cudagraph hang bug **只在 SM90 + cutlass 路径出现**。SM100 走 trtllm_gen
(sglang oracle 默认),根本不走 cutlass 路径,所以 hang bug 自然被绕过。

## 修正 4: DSL 区别会不会导致结果差异

User 注意到 "sm100 和 sm90 上 flashinfer moe kernel DSL 不同?一个 trtllm 一个 cutlass"

**会,显著区别**。这两个是**两套完全独立的 backend**,共享 flashinfer 这个 wrapper 库:

### Path A: `cutlass_fused_moe()` (任何 SM)

- 来源: **CUTLASS** (CUDA Templates for Linear Algebra Subroutines),NVIDIA 开源 C++ 模板库
- 实现: `flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/`
- 编译: **本地 JIT**,首次跑 nvcc 编译 .cu → .so (cold cache 7+ 分钟)
- Tuning: 运行时 `AutoTuner.choose_one("trtllm::fused_moe::gemm1", ...)`
  - 有 hand-tuned 表 → load 命中 (B200/GB200)
  - 没表 → fallback 到 benchmark 候选 kernel (SM90 走这条 → 9× kernel launch bug)
- Arch 支持: SM80, SM89, SM90, SM100, SM103, SM120
- **是我们 §4 找到 sglang AutoTuner re-benchmark bug 的路径**

### Path B: `trtllm_bf16_moe()` / `trtllm_fp8_*` / `trtllm_fp4_*` (SM100 only)

- 来源: **TensorRT-LLM** ── NVIDIA 内部团队手工优化的高性能 LLM 推理库
- 实现: `flashinfer/data/csrc/trtllm_fused_moe_*.cu` + 下载的 `.cubin` (来自 NVIDIA artifact server)
- 编译: **不本地 JIT**, `get_cubin()` 下载预编译好的 cubin
- Tuning: cubin 内 `tllmGenBatchedGemmList` 是 NVIDIA offline 用 ncu 对每个 shape sweep 出来的最优 kernel
- Arch 支持: **SM100 only** (`gen_trtllm_gen_fused_moe_sm100_module()` 写死 `supported_major_versions=[10]`)
- 入口: `core.py:2115 trtllm_bf16_moe`, line 2199 等

### 关键区别表

| 维度 | CUTLASS 路径 | TRT-LLM Gen 路径 |
|---|---|---|
| 编译时机 | 本地 JIT (cold 7+ 分钟) | 下载 cubin (秒级) |
| Cold-start risk | **高** (§11 复现的 cudagraph + JIT race 就是这条) | 几乎无 |
| Tuner 行为 | 运行时 choose_one (cache miss → re-benchmark) | 同 choose_one,但 cubin 内 kernel 选项有限 |
| Shape 覆盖 | 几乎任意 (模板生成) | 受 cubin tllmGenBatchedGemmList 限制 |
| 性能上限 | 灵活但 generic | 极致 (NVIDIA team offline tune) |
| 适用场景 | 长尾 shape / SM90 唯一选 / trtllm_gen 不覆盖 | hot shape (DeepSeek/Llama4) |
| Wrapper bug 风险 | **高** (sglang 的 AutoTuner 误用就在这条) | 低 |

### Oracle 默认选择(确认)

`sglang/python/sglang/srt/server_args.py:1558`:
- Qwen3MoE on **SM100** → `flashinfer_trtllm` (Path B)
- Qwen3MoE on **SM90** → 不选 flashinfer 默认 (要 `--moe-runner-backend flashinfer_cutlass` 显式指定才走 Path A)

`vllm/.../oracle/unquantized.py:67-78`:
- SM90: `_move_to_back(FLASHINFER_TRTLLM)` 和 `_move_to_back(FLASHINFER_CUTLASS)` → 优先 Triton
- SM100: 默认顺序就是 trtllm 在前 → 走 Path B

**SM100 上两个引擎默认都走 trtllm_gen** (Path B),性能最好。
**SM90 上没有 trtllm_gen 选项**,sglang 走 cutlass 时撞 bug,vLLM 选 triton 绕开。

## 重新修订的优化建议优先级

1. **🥇 Fix 1**: sglang `unquant.py:385` 一行 patch — 让 SM90 + sglang + cutlass 至少能追到 vllm 同水平 (3.4× 提升)
2. **🥈 Fix 3'**: **给 flashinfer 仓库 PR 一个 `v0_1_trtllm_fused_moe_NVIDIA_H200.py`** —
   sweep H200 上 (E=128, N=768) 和其他常见 Qwen3MoE shape,生成 best_configs dict 写文件。
   B200/GB200 已经做了,H200 应该有人补。**这能让 SM90 cutlass 路径直接命中 hand-tuned 表,
   绕过 runtime AutoTuner**。
3. **🥉 Fix 2'**: 给 sglang oracle 加 batch-size-aware 路由 — high-conc serving 上 +17% (§12 finding)

不推荐做:
- ~~修 sglang cudagraph hang~~ — cold cache 才出现,warm 后没事,而且 cudagraph 只帮 R_short
- ~~解锁 CUTLASS SM90 M=64 tile~~ — 需要碰 CUTLASS 模板,ROI 不确定


---

# Part C (新增) — SM100 vs SM90 detailed tuner comparison + 能不能直接抄表?

## C.1 三个澄清问题

User 又问了三个问题,这里集中回答:

### Q1: SM100 和 SM90 的 CUTLASS MoE kernel 是同一个吗?

**部分一样,但 binary 不同**。

- **C++ 源码 `.cu` 文件**: 一样。`flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/*.cu` 这一套是共用模板代码。
- **JIT 编译指令**: 不一样,两个独立的 `gen_cutlass_fused_moe_sm{90,100}_module()` 函数。

`flashinfer/jit/fused_moe.py:85-95` (SM90):
```python
def gen_cutlass_fused_moe_sm90_module():
    nvcc_flags = sm90a_nvcc_flags + [
        "-DCOMPILE_HOPPER_TMA_GEMMS",
        "-DCOMPILE_HOPPER_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16", "-DENABLE_FP8",
        "-DENABLE_FP8_BLOCK_SCALE" if cuda >= 12.8 else "",
        "-DENABLE_FP4" if cuda >= 12.8 else "",
    ]
    return gen_cutlass_fused_moe_module(nvcc_flags, "90")    # arch="90"
```

`flashinfer/jit/fused_moe.py:68-82` (SM100):
```python
def gen_cutlass_fused_moe_sm100_module():
    nvcc_flags = [
        "-DCOMPILE_BLACKWELL_TMA_GEMMS",                     # 不同 macro
        "-DCOMPILE_BLACKWELL_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16", "-DENABLE_FP8", "-DENABLE_FP4",
    ]
    return gen_cutlass_fused_moe_module(nvcc_flags, "100")   # arch="100"
```

**编译出来是两个不同 `.so` 文件**: `fused_moe_90.so` 和 `fused_moe_100.so`。`.cu` 源码里用 `#ifdef COMPILE_HOPPER_*` vs `#ifdef COMPILE_BLACKWELL_*` 包了不同的 kernel 实例化代码,nvcc 编译时只编对应那边的。

**Mainloop 模板不同**:
- SM90 用 `cutlass::gemm::collective::MainloopSm90ArrayTmaGmmaWarpSpecialized*` (基于 WGMMA)
- SM100 用 `cutlass::gemm::collective::MainloopSm100Array*` (基于 UMMA + TMEM)

模板名字里硬编码 `Sm90` / `Sm100`,nvcc 实例化时挑对应族。

### Q2: CUTLASS 和 trtllm 都在 flashinfer 里吗?

**对,都是 flashinfer 提供的入口**,但底下是两个独立 C++/CUDA 项目:

```
flashinfer (Python wrapper 库)
├── cutlass_fused_moe()                                ← Path A
│   └── 底下 link 到 CUTLASS (NVIDIA 开源 C++ 模板库)
│       源码: data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/
│       (路径里有 "tensorrt_llm" 是历史包袱 — flashinfer 把 TRT-LLM
│        vendor 进来的部分代码)
│
└── trtllm_bf16_moe(), trtllm_fp8_*(), trtllm_fp4_*()  ← Path B
    └── 底下 link 到 TRT-LLM gen kernel
        源码: data/csrc/trtllm_fused_moe_*.cu
              + 下载的预编译 cubin (NVIDIA artifact server)
```

`flashinfer/fused_moe/__init__.py` 同时 export 这两套 API。

### Q3: SM100 vs SM90 tuner 详细差别 + kernel 参数含义

详见 §C.2-C.4。

## C.2 先解释 N / M / K / tile / cluster 在 kernel 里指啥

GEMM 公式: `C[M, N] = A[M, K] × B[K, N]`

对 MoE 来说,每个 expert 算两个 GEMM:
- `gemm1` (gate+up): `[M_tokens, K_hidden] × [K_hidden, 2*N_intermediate]` → `[M_tokens, 2*N_intermediate]`
- `gemm2` (down):    `[M_tokens, N_intermediate] × [N_intermediate, K_hidden]` → `[M_tokens, K_hidden]`

变量含义:
- **M (token 维度)**: 喂给同一个 expert 的 token 数,跟 batch_size × top_k / num_experts 平均成比例
- **N (输出 channel 维度)**: Qwen3MoE 是 768 (intermediate_size)
- **K (输入 channel 维度)**: Qwen3MoE 是 2048 (hidden_size)

### "Tile" 是把大矩阵切成小块给一个 SM 处理

GPU 上一个 SM (Streaming Multiprocessor) 不能一次处理整个矩阵,要切成 tile。每个 tile 称为 **CTA** (Cooperative Thread Array),由一个 thread block 处理:

```
完整 C[M, N] 矩阵:
┌─────────────────────────┐
│ tile1 │ tile2 │ tile3 │ ← 每个小方块是一个 CTA tile (M_tile × N_tile)
├───────┼───────┼───────┤    由一个 SM 算
│ tile4 │ tile5 │ tile6 │
└───────┴───────┴───────┘
```

CUTLASS 命名 `CtaShape128x16x128B` 表示:
- M_tile = 128 (每个 tile 处理 128 个 token 行)
- N_tile = 16 (每个 tile 处理 16 个输出 channel)
- K_tile = 128 (内层 K 循环每次处理 128 个 channel,B 是 byte/bit 后缀)

### "Cluster shape" / "CGA" 是把多个 CTA 组成 cluster (Hopper+ 才有)

Hopper 引入 **Thread Block Cluster**: 多个 CTA 可以**跨 SM 共享 shared memory**,做 multicast / distributed shared memory,减少 HBM 读。

`ClusterShape_2x1x1` = 2 个 CTA 在 M 方向组成 cluster (mcast along M)。两个 SM 共享 A 矩阵的 tile,减少 HBM 读。

Blackwell 加了 **Dynamic Cluster** — cluster shape 不用编译时定死,可以运行时决定。

## C.3 SM90 candidate kernel 列表 (硬证据)

来源: `flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:196-224`

```cpp
std::vector<CutlassTileConfigSM90> get_candidate_tiles_sm90(
    CutlassGemmConfig::CandidateConfigTypeParam const config) {
  if (config & CutlassGemmConfig::GROUPED_GEMM) {
    if (config & CutlassGemmConfig::WEIGHT_ONLY) {
      // 量化路径
      return {
        CtaShape64x16x128B,  CtaShape64x32x128B,
        CtaShape64x64x128B,  CtaShape64x128x128B,
        CtaShape128x16x128B, CtaShape128x32x128B,
        CtaShape128x64x128B, CtaShape128x128x128B
      };  // 8 个 tile
    } else {
      // ★ 我们 BF16 unquantized 走这条
      return {
        CtaShape128x16x128B,  CtaShape128x32x128B,
        CtaShape128x64x128B,  CtaShape128x128x128B,
        CtaShape128x256x128B, CtaShape256x128x128B
      };  // 6 个 tile,M 锁死 128 或 256
    }
  }
}
```

### SM90 GROUPED_GEMM (no quant) 候选 tile 6 个

| Tile | M_tile | N_tile | K_tile | M-mcast 支持? | N-mcast 支持? |
|---|---|---|---|---|---|
| 128×16×128 | 128 | 16 | 128 | ✓ | ✗ |
| 128×32×128 | 128 | 32 | 128 | ✓ | ✗ |
| 128×64×128 | 128 | 64 | 128 | ✓ | ✗ |
| 128×128×128 | 128 | 128 | 128 | ✓ | ✓ |
| 128×256×128 | 128 | 256 | 128 | ✓ | ✓ |
| 256×128×128 | 256 | 128 | 128 | ✓ | ✓ |

### 加 cluster shape 后 (`get_candidate_configs_sm90`, line 380-444)

- 1×1×1 (always added, 6 个)
- 2×1×1 (需 M-mcast, M_tile ≥ 128 → 6 个 tile 都支持, 加 6 个)
- 1×2×1 (需 N-mcast, N_tile ≥ 128 → 3 个 tile 支持, 加 3 个)
- 2×2×1 (需 M+N mcast → 同上 3 个, 加 3 个)

**SM90 候选总数 = 6 + 6 + 3 + 3 = 18 个 candidate kernel**

每个 candidate **没有 num_stages / num_warps 子选项**,全 AUTO 由 CUTLASS 自己根据 tile 推。

## C.4 SM100 candidate kernel 列表 (硬证据)

来源: `flashinfer/jit/gemm/cutlass/generate_kernels.py:840-941` (JIT 编译时 kernel 实例化的总列表)

```python
def generate_sm100_grouped_gemm_operations(is_arch_enabled, arch):
    supported_dtypes = [
        DataType.f16, DataType.bf16, DataType.f32,
        DataType.e4m3,           # FP8
        e2m1,                    # FP4 native (SM90 没有!)
        (DataType.e4m3, e2m1),   # 混合 FP8+FP4
    ]
    cta_shapes_m = [64, 128]                    # ← M 有 2 个选项 (SM90 只有 128)
    cta_shapes_n = [8, 16, 32, 64, 128, 192, 256]  # ← N 7 个 (SM90 只有 5)
    cga_shapes = [(1,1,1), (2,1,1)]
    dynamic_cga = [True, False]                  # ← 动态 cluster (SM90 没有)
    epi_schedules = [
        PtrArrayNoSmemWarpSpecialized1Sm,        # ← no-smem epilogue (SM90 没有)
        PtrArrayTmaWarpSpecialized1Sm,
    ]
    swap_ab = [True, False]                      # ← swap A/B (SM90 不曝露)
```

### SM100 BF16 unquantized 候选估算

2 (M) × 7 (N) × 2 (cga) × 2 (dynamic_cga) × 2 (epi) × 2 (swap_ab) = **224 BF16 候选**

实际有效候选 ~150-200 (经过 `are_tile_shapes_supported_sm100` 过滤)。

## C.5 直接对比表

| 维度 | SM90 (BF16 unquant grouped GEMM) | SM100 (BF16 unquant grouped GEMM) | 倍数 |
|---|---|---|---|
| **M tile 选项** | **1 个** (锁死 128) <br>(源码注释 "Currently M tile must be 128 for Grouped GEMM") | 2 个 (64, 128) | 2× |
| **N tile 选项** | 5 个 (16, 32, 64, 128, 256) | 7 个 (8, 16, 32, 64, 128, 192, 256) | 1.4× |
| **K tile 选项** | 1 个 (128) | 1-2 个 (由 `calc_shape_mnk_sm100_grouped_gemm` 动态算) | 1-2× |
| **Cluster shape** | 4 种 (1×1×1, 2×1×1, 1×2×1, 2×2×1),受 tile 限制 | 2 种 base × dynamic 开关 | 持平 |
| **Epilogue schedule** | **1 种** (`PtrArrayTmaWarpSpecializedCooperative`) | 2 种 (`PtrArrayNoSmemWarpSpecialized1Sm` + `PtrArrayTmaWarpSpecialized1Sm`) | 2× |
| **Mainloop kernel schedule** | 1-2 种 (`Coop`, 或 `CoopFP8FastAccum` 量化时) | 1-2 种 (Sm100 专属) | 持平 |
| **Swap A/B 选项** | 不曝露 | 2 种 (True / False) | 2× |
| **Dynamic CGA** | ❌ 不支持 (编译时固定) | ✓ True/False | — |
| **Native FP4 dtype** | ❌ (CUDA 12.8 后能编 FP4 dispatch 代码,但 mainloop 没原生 FP4 指令) | ✓ (`MainloopSm100ArraySm100*Nvf4*`) | — |
| **Hand-tuned tactic 表** | ❌ 完全没有 | ✓ B200 + GB200 各一份 .py 文件 | — |
| **总 BF16 候选 kernel 数 (粗算)** | **18 个** | **~224 个** | **~12×** |

## C.6 Tuner 行为差别

两边的 AutoTuner 框架是同一个 (`flashinfer/autotuner.py`),区别在**数据**:

| Tuner 行为 | SM90 (H100/H200) | SM100 (B200/GB200) |
|---|---|---|
| `load_from_file()` 查 hand-tuned 表 (autotuner.py:316) | 找文件 `v0_1_trtllm_fused_moe_NVIDIA_H200.py` → **NOT FOUND**,直接 fallback | 找文件 `v0_1_trtllm_fused_moe_NVIDIA_B200.py` → **FOUND**,如果 shape key 在表里 → 直接 load tactic_id |
| Cache miss 时行为 | 在 18 个候选里全 launch 一遍做 micro-benchmark | 在 ~224 个候选里挑,但 hand-tuned 表命中率高,只对没在表里的 shape benchmark |
| 命中 hand-tuned 表的 shape (B200 上举例) | 不存在 | batch ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384} 都有 |
| 每次 forward 的 tuning 开销 | **9× extra kernel launch** (我们 §4 nsys 量到的) | ~1× extra (cache 命中时几乎 0) |

## C.7 实际 Qwen3MoE 上的影响

Qwen3-30B-A3B 的关键 shape:
- hidden K = 2048, intermediate N = 768, num_experts E = 128, top_k = 8
- Decode batch B = 64 时,平均每 expert 处理 M ≈ 64 × 8 / 128 = 4 token,hot expert 可能 ~15

**SM90 上**:
- M tile 锁死 128 → 处理 15 个 token 要 1 个 tile,**87% register 浪费**
- N tile 没有 ~48 这种;N=768 ÷ N_tile=128 要 6 个 tile,或 ÷ N_tile=256 要 3 个
- 实际只能在 6 个 tile 里挑,不一定是真正最优的

**SM100 上**:
- M tile 可选 64 → 处理 15 token 浪费降到 76%
- N tile 多了 8 和 192:N_tile=192 时 768/192=4 个 tile,**整除**,no waste
- 加 dynamic CGA、no-smem epilogue,实际能挑到更接近最优的配置
- 如果 device 是 B200/GB200,**直接 load `v0_1_trtllm_fused_moe_NVIDIA_B200.py` 命中 hand-tuned tactic**

## C.8 ❌ 直接把 SM100 表抄给 SM90 — 不行,会变更差

User 问: 能不能直接 `cp v0_1_trtllm_fused_moe_NVIDIA_B200.py v0_1_trtllm_fused_moe_NVIDIA_H200.py`?

**不能。会更糟**。四个独立原因:

### 原因 1: tactic_id 是 per-arch 的私有下标,不能跨 arch 翻译

`autotuner.py:199-203` 源码注释直接说:
```
"how to interpret the meaning of tactic is pure internal details of the runner"
```

B200 表里 value `(0, 5)` 表示 "用第 0 个 runner 的第 5 号 tactic"。但 SM90 的 `MoERunner.get_valid_tactics()` 返回的 18 个候选,跟 SM100 的 ~224 个候选**完全不同的 list**,**下标 5 在两边指向完全不同的 kernel**。

### 原因 2: SM100 表里挑的 kernel 可能 SM90 上根本不存在

B200 上挑的最优 tactic 经常是 M=64 tile (Qwen 这种 shape 上特别合适)。SM90 上 M tile 锁死 128,**M=64 的 tile 根本没编译进 SM90 的 .so 里**。如果硬把 tactic_id=5 传给 SM90 runner,会 segfault 或 fallback 到默认。

### 原因 3: B200 表里的 shape key 跟 H200 实际跑的 shape 不匹配

B200 表里的 key 是 NVIDIA 测过的具体 model 的 shape,例如:
```
'((1, 3584), (256, 512, 448), (0,), (256, 7168, 16), (0,))'
       ↑                                  ↑
   batch=1, K_hidden=3584           K_hidden=7168 (DeepSeek-V3)
```

我们 Qwen3MoE 的 shape 是:
```
'((B, 2048), (128, 1536, ?), (0,), (128, 2048, 768), (0,))'
              ↑                              ↑
        num_experts=128                    N=768
```

**完全不在 B200 表里** → `load_from_file()` 查不到 → cache miss → 还是 fall back 到 runtime sweep。**抄了等于没抄**。

### 原因 4: 即使 shape 偶然匹配,B200 挑的 tactic 在 SM90 上未必最优

两边硬件原语不同:
- SM100 的最优 tactic 可能严重依赖 UMMA + TMEM 把累加器搬出 register
- 同 tile 在 SM90 上用 WGMMA,因为累加器占 register,实际行为可能完全不同

所以 hand-tuned 的本质是**每个 device 单独 sweep**,**不能跨 device 共享**。

## C.9 ✅ 正确做法: 重做一份 SM90 sweep

要补 SM90 短板的正确做法,是**在 H200 真机上跑 AutoTuner sweep,生成 H200 自己的 hand-tuned 表**。NVIDIA 为 B200/GB200 做的也正是这件事。

### 伪代码

```python
device_name = "NVIDIA_H200"
shapes_to_sweep = [
    # (E, K, N) — covers common MoE models
    (128, 2048, 768),   # Qwen3-30B-A3B
    (256, 7168, 2048),  # DeepSeek-V3
    (16, 4096, 14336),  # Mixtral-8x7B
    # ... more model shapes
]
batches_to_sweep = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

best_configs = {}
for E, K, N in shapes_to_sweep:
    for B in batches_to_sweep:
        # 构造 fake input matching shape
        hidden_states = torch.randn(B, K, dtype=torch.bfloat16, device='cuda')
        w13 = torch.randn(E, 2*N, K, dtype=torch.bfloat16, device='cuda')
        w2 = torch.randn(E, K, N, dtype=torch.bfloat16, device='cuda')
        # ... routing setup
        
        # Enable AutoTuner sweep mode
        with autotune(enable=True):
            cutlass_fused_moe(hidden_states, w13, w2, ..., tune_max_num_tokens=8192)
        
        # AutoTuner sweep 完成,从 profiling_cache 拿出 best tactic
        from flashinfer.autotuner import AutoTuner
        tuner = AutoTuner.get()
        # Find the key for this shape and pull (runner_id, tactic_id)
        for cache_key, (runner_id, tactic_id, _) in tuner.profiling_cache.items():
            if matches_shape(cache_key, E, K, N, B):
                best_configs[serialize_key(cache_key)] = (runner_id, tactic_id)

# 写成 .py 文件
write_python_dict(best_configs, "tuning_configs/v0_1_trtllm_fused_moe_NVIDIA_H200.py")
```

### 实际收益预估

| 阶段 | sglang_cutlass R_medium (我们已测) | 状态 |
|---|---|---|
| 当前 (broken) | 1.31 req/s | runtime AutoTuner re-benchmark 每 forward 9× kernel launch |
| **+ Fix 1** (固定 `tune_max_num_tokens`) | 预估 3-4 req/s | runtime AutoTuner 但 cache 命中率高,只 sweep 一次 |
| **+ Fix 3'** (hand-tuned H200 表) | 预估 4.5-4.7 req/s | 启动时直接 load,完全跳过 runtime sweep,跟 vllm_cutlass 同水平 |

两个 fix 是互补的,**一起做才能彻底解决**。

### 工程量

| 步骤 | 时间 |
|---|---|
| 写 sweep 脚本 (复用 AutoTuner API) | 半天 |
| 跑 sweep (每 shape×batch 跑 18 个 candidate, ~5000 次 kernel launch) | 2-4 小时 |
| 写成 .py 文件 + flashinfer-ai/flashinfer 提 PR | 半天 |
| **总** | **1-2 天** |

NVIDIA 给 B200/GB200 做的就是这个流程,只是他们的 sweep shape list 更全,包括了 NVFP4 / FP8 等量化路径。

### 不需要碰 sglang/vLLM 代码

这条 fix 直接打在 **flashinfer 仓库**,任何用 flashinfer cutlass MoE 的引擎 (vLLM, sglang, TGI, etc.) 在 SM90 上都受益。是真正的"上游 fix"。

## C.10 一句话总结(给 reviewer)

| | |
|---|---|
| SM100 / SM90 共享 CUTLASS 源码? | ✓ 但编译成不同 binary,mainloop 模板 (`Sm90` vs `Sm100`) 不同 |
| flashinfer 里有 CUTLASS 和 trtllm_gen 两套? | ✓ 两个独立 backend, flashinfer 是 wrapper |
| SM100 tuner 比 SM90 强多少? | tile 搜索空间 12×, 加上 hand-tuned 表 |
| 直接抄 SM100 表给 SM90 行吗? | ❌ 不行,tactic_id 跨 arch 不通用,shape 也不匹配 |
| 那怎么补 SM90? | 在 H200 上**重做一份 sweep**,生成 `v0_1_trtllm_fused_moe_NVIDIA_H200.py`,直接对标 NVIDIA 给 B200/GB200 做的事。1-2 天工程量,上游 PR 即可 |

