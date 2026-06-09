# sglang vs vLLM — MoE Kernel 源码对比(SM90 BF16)

> **目的**: 把 Triton MoE kernel 跟 FlashInfer CUTLASS MoE kernel 的关键源码片段
> 放在一起,让你 (人工 reviewer) 看一下两边在 tile/搜索空间/累加器/epilogue 等设计
> 上的具体差别,自己判断是不是 "tuning 不足" 导致 §12 看到的 conc 反转。
>
> 配套数据: `docs/2026-06-08/sglang_vs_vllm_flashinfer_cutlass_analysis.md` §12 (regime sweep)
>
> 测试 case: Qwen3-30B-A3B (E=128, N=768, K=2048), H200 sm_90a, bf16, TP=1

---

## TL;DR

| 维度 | Triton MoE kernel | FlashInfer CUTLASS MoE kernel (SM90) |
|---|---|---|
| Tile M (token 维度 block) | **可调,16/32/64** | **锁死 64 或 128 (二选一)** |
| Tile N (输出 channel 维度) | 64-256 灵活 | 16, 32, 64, 128, 256 (5 个枚举) |
| Tile K (输入 channel 维度) | 64/128/256 (可调) | **锁死 128** (`CtaShape*x*x128B`) |
| Cluster shape (CGA) | 不存在概念 | 1×1, 2×1, 1×2, 2×2 (4 枚举),受 tile shape gating |
| Per-shape autotune database | **18 个 batch size × 多 N 都有 hand-tuned config** | **没有 per-shape 数据库**,运行时 AutoTuner 现挑 |
| Mainloop | 单一 ping-pong loop, num_stages 可调 | `MainloopSm90ArrayTmaGmmaWarpSpecialized*`,coop / pingpong / FP8FastAccum 三选 |
| Epilogue | inline 在 kernel 末尾,可选 bias / RouterWeight / activation | `CollectiveEpilogue` 可选 `ScaledAccPerRowBiasPerColScaleScatter` (融合 bias + router weight + scatter back) |
| 累加器位置 | register (`tl.zeros(BLOCK_SIZE_M, BLOCK_SIZE_N, dtype=tl.float32)`) | register (SM90 WGMMA), SM100 才上 TMEM |
| MoE 分发方式 | "padded token map":每个 (expert, token) pair 排好序后,kernel program 用 expert_id 找权重 | grouped GEMM:一次 launch 包 N 个 expert 的 GEMM,内部分发 |

**核心论点**:
- Triton 端有 `E=128,N=768,device_name=NVIDIA_H200.json` 这种**人工 tune 出来的 18 个 batch size 配置**
- CUTLASS 端**只有 ~10 个候选 tile + 4 个 cluster 总共 ~40 个候选**,运行时 AutoTuner 现挑,**没有跨 batch size 的 hand-tuned 数据库**

这就是为什么 conc=1 时 Triton 赢(有 batch=1 的专属 config: `BLOCK_M=16, N=64, K=64, warps=4, stages=5`),
conc=64 时 CUTLASS 赢(此时 AutoTuner 能在大 batch 下挑到接近最优的 grouped GEMM tile,
Triton 的 batch=64 config 是 `BLOCK_M=16, N=256, K=128` 但仍受限于 per-(expert,token) 排序的开销)。

---

## 1. Triton MoE Kernel 全代码 (vLLM)

来源: `vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:292-553`

### 1.1 函数签名 + meta-parameter

```python
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, b_bias_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N, K, EM, num_valid_tokens,
    # Strides ...
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bse, stride_bsk, stride_bsn,
    stride_bbe, stride_bbn,
    # Block size for block-wise quantization
    group_n: tl.constexpr, group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # ★★ Meta-parameters (tunable) ★★
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
```

### 1.2 Pid → block of C(L2 reuse 优化)

```python
# Map program ids `pid` to the block of C it should compute.
# This is done in a grouped ordering to promote L2 data reuse.
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

### 1.3 sorted_token_ids 索引 (MoE 特殊机制)

```python
# 从 sorted_token_ids 里读这个 pid_m block 对应的 token 索引
offs_token_id = pid_m * BLOCK_SIZE_M + offs
offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
offs_token = offs_token.to(tl.int64)
token_mask = offs_token < num_valid_tokens

# 从 expert_ids 里读这个 pid_m block 对应的 expert
off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
if off_experts == -1:
    # 这个 expert 不在当前 EP rank,写 0 后 return
    write_zeros_to_output(...)
    return
```

**这是 Triton MoE 的核心 trick**: 把 (token, expert) pair 按 expert sort,然后 pad 到
`BLOCK_M` 的倍数,**每个 tile 处理一个 expert 的一批 token**。kernel 通过 `expert_ids[pid_m]`
查询当前 tile 是哪个 expert。

### 1.4 Mainloop (K 维迭代)

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # Load the next block of A and B, generate a mask by checking the K dimension.
    a = tl.load(
        a_ptrs,
        mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
        other=0.0,
    )
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    # We accumulate along the K dimension.
    if use_int8_w8a16:
        accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            # block-wise quant: load scale per K-block
            offs_ks = k * BLOCK_SIZE_K // group_k
            a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        else:
            if use_fp8_w8a8:
                accumulator = tl.dot(a, b, acc=accumulator)  # fp8 fast accum
            else:
                accumulator += tl.dot(a, b)
    else:
        accumulator += tl.dot(a, b)
    # Advance the ptrs to the next K block.
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

注意:**bf16 走的是最后一条 `accumulator += tl.dot(a, b)`,FP32 累加器在 register**。
Triton 编译器会把这个映射到 H200 上的 WGMMA 指令。

### 1.5 Epilogue (bias + router weight + cast + store)

```python
# Dequantization for quantized schemes
if use_int8_w8a16:
    accumulator = accumulator * b_scale
elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
    accumulator = accumulator * a_scale * b_scale

# Bias addition (after dequant)
if HAS_BIAS:
    bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
    bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    accumulator += bias[None, :]

# Router (MoE) weight multiplication — MUST be FP32 for numerical stability
if MUL_ROUTED_WEIGHT:
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
    accumulator *= moe_weight[:, None]

# Cast to compute_type
accumulator = accumulator.to(compute_type)

# Write back the block of the output
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
tl.store(c_ptrs, accumulator, mask=c_mask)
```

整个 kernel 一气呵成: GEMM + dequant + bias + router weight + scatter to output position。

### 1.6 Per-shape autotune database (人工 tune 出来的)

文件: `vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_H200.json`

包含 **18 个 batch size 的 hand-tuned config**:
```json
{
  "1":   {"BLOCK_SIZE_M":16, "BLOCK_SIZE_N":64,  "BLOCK_SIZE_K":64,  "GROUP_SIZE_M":1,  "num_warps":4, "num_stages":5},
  "2":   {"BLOCK_SIZE_M":16, "BLOCK_SIZE_N":64,  "BLOCK_SIZE_K":64,  "GROUP_SIZE_M":1,  "num_warps":4, "num_stages":4},
  "4":   {"BLOCK_SIZE_M":16, "BLOCK_SIZE_N":64,  "BLOCK_SIZE_K":128, "GROUP_SIZE_M":16, "num_warps":4, "num_stages":2},
  "8":   {"BLOCK_SIZE_M":16, "BLOCK_SIZE_N":64,  "BLOCK_SIZE_K":128, "GROUP_SIZE_M":1,  "num_warps":4, "num_stages":3},
  "16":  {"BLOCK_SIZE_M":16, "BLOCK_SIZE_N":64,  "BLOCK_SIZE_K":256, "GROUP_SIZE_M":1,  "num_warps":4, "num_stages":2},
  "24":  {...}, "32":{...}, "48":{...},
  "64":  {"BLOCK_SIZE_M":16, "BLOCK_SIZE_N":256, "BLOCK_SIZE_K":128, "GROUP_SIZE_M":1,  "num_warps":8, "num_stages":2},
  "96":  {...},
  "128": {"BLOCK_SIZE_M":16, "BLOCK_SIZE_N":128, "BLOCK_SIZE_K":128, "GROUP_SIZE_M":1,  "num_warps":4, "num_stages":2},
  "256": {"BLOCK_SIZE_M":32, "BLOCK_SIZE_N":256, "BLOCK_SIZE_K":128, "GROUP_SIZE_M":16, "num_warps":4, "num_stages":2},
  "512": {...}, "1024":{...}, "1536":{...}, "2048":{...}, "3072":{...}, "4096":{...}
}
```

观察:
- batch=1 用 **小 tile** (16×64×64) + **多 stage** (stages=5) 让 SM 充分流水
- batch=64-256 用 **大 N tile** (128 或 256) 让 GEMM 接近 sm 峰值
- batch=4096 (超大) 用 **大 M tile** + **小 num_warps** 让多个 wave 并发
- **每个 batch size 都是手动 tune 出来的最优**

---

## 2. FlashInfer CUTLASS MoE Kernel (SM90)

CUTLASS MoE 的 entry point 是模板代码,Python 看到的是 `cutlass_fused_moe(...)`,
底下展开成 `tma_warp_specialized_generic_moe_gemm_kernelLauncher<...>`。

### 2.1 Kernel 选择: SM90 候选 tile 列表

来源: `flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:196-224`

```cpp
std::vector<CutlassTileConfigSM90> get_candidate_tiles_sm90(
    CutlassGemmConfig::CandidateConfigTypeParam const config) {
#ifdef FAST_BUILD
  // Fast build disables all configs except this one for SM90
  return {CutlassTileConfigSM90::CtaShape128x128x128B};
#else
  if (config & CutlassGemmConfig::GROUPED_GEMM) {
    if (config & CutlassGemmConfig::WEIGHT_ONLY) {
      return {
          CutlassTileConfigSM90::CtaShape64x16x128B,  CutlassTileConfigSM90::CtaShape64x32x128B,
          CutlassTileConfigSM90::CtaShape64x64x128B,  CutlassTileConfigSM90::CtaShape64x128x128B,
          CutlassTileConfigSM90::CtaShape128x16x128B, CutlassTileConfigSM90::CtaShape128x32x128B,
          CutlassTileConfigSM90::CtaShape128x64x128B, CutlassTileConfigSM90::CtaShape128x128x128B};
    } else {
      // ← 我们 BF16 unquantized 走这条
      return {
          CutlassTileConfigSM90::CtaShape128x16x128B,  CutlassTileConfigSM90::CtaShape128x32x128B,
          CutlassTileConfigSM90::CtaShape128x64x128B,  CutlassTileConfigSM90::CtaShape128x128x128B,
          CutlassTileConfigSM90::CtaShape128x256x128B, CutlassTileConfigSM90::CtaShape256x128x128B};
    }
  } else {
    return {
        CutlassTileConfigSM90::CtaShape64x16x128B,   CutlassTileConfigSM90::CtaShape64x32x128B,
        CutlassTileConfigSM90::CtaShape64x64x128B,   CutlassTileConfigSM90::CtaShape64x128x128B,
        CutlassTileConfigSM90::CtaShape64x256x128B,  CutlassTileConfigSM90::CtaShape128x16x128B,
        CutlassTileConfigSM90::CtaShape128x32x128B,  CutlassTileConfigSM90::CtaShape128x64x128B,
        CutlassTileConfigSM90::CtaShape128x128x128B, CutlassTileConfigSM90::CtaShape128x256x128B};
  }
#endif
}
```

**Qwen3-30B-A3B 走的就是 `GROUPED_GEMM && !WEIGHT_ONLY` 这条** —— 只有 **6 个候选 tile**:
```
128×16×128, 128×32×128, 128×64×128, 128×128×128, 128×256×128, 256×128×128
```
M 维度**只有 128 或 256**(对比 Triton 16/32/64 全都有)。

### 2.2 Cluster shape (CGA) 选择

来源: 同文件 line 380-444 (`get_candidate_configs_sm90`)

```cpp
std::vector<CutlassGemmConfig> get_candidate_configs_sm90(
    CutlassGemmConfig::CandidateConfigTypeParam const config) {
  auto tiles = get_candidate_tiles_sm90(config);
  std::vector<CutlassGemmConfig> candidate_configs;
  for (auto const& tile_config : tiles) {
    bool const has_m_mcast = sm90_supports_mcast_along_m(tile_config);
    bool const has_n_mcast = sm90_supports_mcast_along_n(tile_config);
    
    // unquantized 走这一支
    CutlassGemmConfig candidate(tile_config, MainloopScheduleType::AUTO,
                                EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1);
    candidate_configs.push_back(candidate);
    if (has_m_mcast) {
      candidate_configs.push_back(CutlassGemmConfig(tile_config, ...,
                                  ClusterShape::ClusterShape_2x1x1));
    }
    if (has_n_mcast) {
      candidate_configs.push_back(CutlassGemmConfig(tile_config, ...,
                                  ClusterShape::ClusterShape_1x2x1));
    }
    if (has_m_mcast && has_n_mcast) {
      candidate_configs.push_back(CutlassGemmConfig(tile_config, ...,
                                  ClusterShape::ClusterShape_2x2x1));
    }
  }
  return candidate_configs;
}
```

M-mcast 要求 tile M ≥ 128;N-mcast 要求 tile N ≥ 128。所以总候选:
- 6 tile × ~3 cluster (平均) = **~18 个 candidate kernel 给 AutoTuner 挑**
- 对比 Triton 的 18 个 batch size 全 hand-tuned + 多种 (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) 组合

### 2.3 SM90 Mainloop / Epilogue 选择

来源: `flashinfer/.../moe_gemm_tma_ws_launcher.inl:367-470`

```cpp
// EpilogueSchedule for SM90: 只有 1 个选择
using EpilogueScheduleSM90 = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

// KernelSchedule for SM90: 2 个选择 (FP8 / 其它)
using KernelScheduleSM90 = std::conditional_t<
    IsFP8, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
```

对比 SM10x (Blackwell) 有 8+ 个 epilogue/kernel schedule:
```cpp
// SM10x epilogue:
using EpilogueScheduleSM10x = std::conditional_t<
    IsTmaSM10xEpilogue,
    std::conditional_t<Is2SM, cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,
                       cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>,
    std::conditional_t<Is2SM, cutlass::epilogue::PtrArrayNoSmemWarpSpecialized2Sm,
                       cutlass::epilogue::PtrArrayNoSmemWarpSpecialized1Sm>>;

// SM100 kernel schedule: 多变种
using KernelScheduleSM100 = std::conditional_t<
    Is2SM,
    std::conditional_t<IsBlockScaled, KernelSchedule2SmSm100BlockScaled,
                       cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100>,
    std::conditional_t<IsBlockScaled, KernelSchedule1SmSm100BlockScaled,
                       cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100>>;
```

SM90 上**没有 2-SM 选项 / 没有 No-Smem epilogue / 没有 Dynamic CGA**。SM100 这些都有。

### 2.4 Epilogue fusion (有意思的部分)

```cpp
using EpilogueFusionOp = std::conditional_t<
    SwapAB,
    cutlass::epilogue::fusion::ScaledAccPerRowBiasPerColScaleScatter<
        LayoutD, ElementFinalOutput, ElementAccumulator, ElementBias, ElementRouterScales>,
    cutlass::epilogue::fusion::ScaledAccPerColBiasPerRowScaleScatter<
        LayoutD, ElementFinalOutput, ElementAccumulator, ElementBias, ElementRouterScales>>;
```

CUTLASS 可以把 `accumulator * router_weight + bias → scatter to output position` 这一整套
epilogue 融到 GEMM kernel 里 (CUTLASS 术语 "epilogue fusion")。Triton 也做同样的融合,
只是写在 Python 里(§1.5 的最后段)。两边在 fusion 这件事上**等价**。

### 2.5 MoE 分发机制(grouped GEMM vs sorted token map)

CUTLASS Grouped GEMM 把 N 个 expert 的 GEMM **打包成一次 launch**:
```cpp
// 概念上 (CUTLASS template):
GroupedGemm<...>::Arguments args{
    .problem_count = num_experts,
    .problem_shapes = [{M_expert_0, N, K}, {M_expert_1, N, K}, ..., {M_expert_E-1, N, K}],
    .ptr_A = [A_expert_0, A_expert_1, ...],   // 每个 expert 的输入指针
    .ptr_B = [W_expert_0, W_expert_1, ...],   // 每个 expert 的权重指针
    .ptr_D = [out_0, out_1, ...]              // 每个 expert 的输出指针
};
// 内部一次 kernel launch,所有 expert 的 tile 在 SM 间动态调度
```

Triton 走不同路径:
```python
# 先 sort: 把 (token, expert) pair 按 expert 排好,padding 到 BLOCK_M 的倍数
# 然后 launch 一个普通 GEMM kernel,kernel 内部用 expert_ids[pid_m] 查权重
```

两边都是"一次 launch 算所有 expert",但 work distribution 方式不同:
- CUTLASS: per-expert 一个 problem,SM dynamically 抢
- Triton: per-(expert, token batch) 一个 program,固定 launch grid

---

## 3. 关键搜索空间对比表

把 Triton hand-tuned config + CUTLASS SM90 候选放一起:

### Triton @ E=128, N=768, H200 — 18 个 hand-tuned batch size

| batch | BLOCK_M | BLOCK_N | BLOCK_K | GROUP_M | warps | stages | 说明 |
|---|---|---|---|---|---|---|---|
| 1     | 16  | 64  | 64  | 1  | 4 | 5 | 极小 batch, 多 stage 隐藏 latency |
| 2     | 16  | 64  | 64  | 1  | 4 | 4 | |
| 4     | 16  | 64  | 128 | 16 | 4 | 2 | |
| 8     | 16  | 64  | 128 | 1  | 4 | 3 | |
| 16    | 16  | 64  | 256 | 1  | 4 | 2 | |
| 24    | 16  | 64  | 128 | 1  | 4 | 5 | |
| 32    | 16  | 64  | 128 | 1  | 4 | 2 | |
| 48    | 16  | 64  | 128 | 1  | 4 | 4 | |
| 64    | 16  | 256 | 128 | 1  | **8** | 2 | N 跳到 256 |
| 96    | 16  | 64  | 256 | 1  | 4 | 2 | |
| 128   | 16  | 128 | 128 | 1  | 4 | 2 | |
| 256   | **32**  | 256 | 128 | 16 | 4 | 2 | M 也开始变大 |
| 512   | 32  | 256 | 128 | 1  | 4 | 2 | |
| 1024  | 64  | 128 | 128 | 16 | 4 | 2 | |
| 1536  | 64  | 128 | 128 | 1  | 4 | 2 | |
| 2048  | 64  | 128 | 128 | 32 | 4 | 2 | |
| 3072  | 64  | 128 | 128 | 16 | 4 | 2 | |
| 4096  | 64  | 128 | 128 | 32 | 4 | 2 | |

每行都是 Triton-autotune 实际跑出来的最优。

### CUTLASS SM90 grouped-GEMM bf16 — 6 个 tile × ~3 cluster

| Tile (M×N×K) | M-mcast (2×1×1)? | N-mcast (1×2×1)? | (2×2×1)? |
|---|---|---|---|
| 128×16×128 | ✓ | ✗ | ✗ |
| 128×32×128 | ✓ | ✗ | ✗ |
| 128×64×128 | ✓ | ✗ | ✗ |
| 128×128×128 | ✓ | ✓ | ✓ |
| 128×256×128 | ✓ | ✓ | ✓ |
| 256×128×128 | ✓ | ✓ | ✓ |

总共 ~18 个 candidate (1×1×1 + 视支持加 m/n/m+n mcast)。每个 candidate 的 num_stages /
warp 配置由 CUTLASS 自己根据 tile shape **AUTO 推**,**不是 per-(M, N, K, batch) 手 tune**。

### 直接对比

| | Triton | CUTLASS SM90 |
|---|---|---|
| 总候选数 (E=128,N=768,H200) | **18 个 batch-specific tunes** | ~18 个 generic candidates |
| M tile 粒度 | 16, 32, 64 (覆盖 batch=1 到 batch=4096) | 锁死 128 或 256 |
| K tile 粒度 | 64, 128, 256 (随 batch 调) | 锁死 128 |
| num_stages | 2, 3, 4, 5 都有 (per shape 选) | AUTO,不可调 |
| num_warps | 4 或 8 (per shape 选) | 隐含在 mainloop schedule,不可调 |
| 跨 batch size 的差异化 | **巨大**(M, N, K, stages, warps 都变) | **几乎没有**(只 cluster shape 变) |
| 数据库来源 | 人工 autotune 跑出来 commit 到 repo | 运行时 AutoTuner 现挑 |

---

## 4. 这是不是 "tuning 不足" 的回答

**你的直觉对的**: 是 tuning 不足,但 **"不足" 是结构性的,不是简单加几个 config 能补的**。

### 4.1 Tile shape 搜索空间不够

CUTLASS SM90 grouped GEMM 锁死了:
- M tile ∈ {128, 256}
- K tile = 128

**对小 batch (conc=1, M=8 left over after top-k=8 per token) 这是巨大浪费** —— 一个
M=128 的 tile 处理 8 个 token,87.5% 的 register / shared mem 在 idle。

Triton 这里有 BLOCK_M=16 选项,batch=1 时 16 行 register 足够。

### 4.2 Per-(batch_size, shape) hand-tune 缺失

CUTLASS 的 AutoTuner 在每个 forward 现挑 tactic,挑出来的是**当前 shape 下从 18 个候选里最好的**,
但 18 个候选本身没经过 per-shape 优化(只是 generic 的 tile shape × cluster shape 排列组合)。

Triton 的 JSON 里每个 batch size 都是**人工跑过 autotune,把跑出来的最好 config 写进文件**。
这相当于 "搜索 + 离线优化结果保存" vs CUTLASS 的 "运行时搜索每次都重来"。

### 4.3 为什么 conc=64 上 CUTLASS 反而赢?

(对应 §12 的反转)

batch=64 + top_k=8 = 512 个 (token, expert) assignment,均分到 128 expert,平均每 expert ~4 个 token。
但因为 expert 选择有 imbalance,实际 hot expert 可能 ~20+ token。

- **Triton** 的 batch=64 config 用 BLOCK_M=16, N=256, num_warps=8。每个 program 处理 16 个 (token, expert)。
  hot expert 20 个 token 要 2 个 program 处理,有 padding waste。
- **CUTLASS** grouped GEMM 用 M=128 tile,**一次包整个 expert 的 20+ tokens**,加上 grouped GEMM 在
  SM 间做 work stealing,**实际利用率更高**。

也就是说 CUTLASS 的"锁死 M=128"在大 batch 时反而是优势(work granularity 跟 expert 的
natural batch 接近),小 batch 时是劣势(浪费 register)。

### 4.4 怎么 "补 tuning"?三个候选

1. **给 CUTLASS 加更多 SM90 tile 候选**: 在 `cutlass_heuristic.cpp:get_candidate_tiles_sm90`
   里加 `CtaShape64x16x128B`, `CtaShape32x128x128B` 等。需要 CUTLASS 模板能真编出这些 kernel
   (要看 `MainloopSm90ArrayTmaGmmaWarpSpecialized` 对小 M 的支持)。
2. **给 flashinfer 加 per-shape autotune 数据库**(类 Triton 的 JSON 文件): warmup 时 sweep
   (E, N, K, batch),记录最优 tactic,后续直接命中。flashinfer 现在的 AutoTuner cache 是
   per-run 的,不持久化到 disk。
3. **混合策略**: 小 batch (conc<8) 路由到 Triton,大 batch 路由到 CUTLASS。需要在 vLLM/sglang
   的 oracle 加 batch-size-aware dispatch (现在 oracle 只看 dtype / arch)。

### 4.5 一句话总结

CUTLASS SM90 grouped-GEMM kernel **结构性地**只能高效跑 M=128 / 256 tile,对小 batch shape
没有专门的 kernel。Triton 通过 hand-tuned 数据库覆盖了 18 个 batch size 的细分需求。CUTLASS 在
大 batch 上反超,是因为它的 "锁死 M=128" 恰好匹配大 batch + MoE expert 的天然 work granularity。

**不是 Triton 永远赢,也不是 CUTLASS 永远赢。两边在不同 batch size 上各有优势,但 vLLM/sglang
的 oracle 都没有 batch-size-aware 的路由逻辑,所以经常做不到最优。**

---

## 5. 引用的文件路径(给 reviewer)

### Triton MoE kernel
- `/home/t-jialianggu/work/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:292-553`
  —— `fused_moe_kernel`,Triton kernel 全文
- `/home/t-jialianggu/work/vllm/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_H200.json`
  —— hand-tuned config 数据库,18 个 batch size

### CUTLASS MoE kernel (SM90)
- `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:196-224`
  —— `get_candidate_tiles_sm90`,SM90 tile shape 候选列表
- `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:380-444`
  —— `get_candidate_configs_sm90`,完整候选生成 (tile × cluster)
- `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:367-470`
  —— SM90 mainloop / epilogue schedule 选择 (KernelScheduleSM90 = 单一 `KernelPtrArrayTmaWarpSpecializedCooperative`)
- `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/jit/gemm/cutlass/generate_kernels.py:556-637`
  —— `generate_sm90_grouped_gemm_operations`,JIT 编译时 kernel 实例化的总列表

### Cross-reference
- `docs/2026-06-08/sglang_vs_vllm_flashinfer_cutlass_analysis.md` —— 主分析文档 (1027 行)
  - §10: SM90 vs SM100 四层根因
  - §12: regime sweep 实测数据 (conc=1 Triton 赢, conc=64 CUTLASS 赢)
  - §4: sglang 9× kernel launch 根因 (AutoTuner re-benchmark)
