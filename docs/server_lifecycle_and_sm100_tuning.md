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
- `docs/sglang_vs_vllm_flashinfer_cutlass_analysis.md` — 主分析 (§4 9× launch, §10 SM90/SM100 四层, §12 conc=64 反转)
- `docs/triton_vs_cutlass_moe_kernel_source_comparison.md` — Triton vs CUTLASS 源码对比
- `docs/moe_backend_decision_trees.md` — sglang/vLLM oracle 决策树
