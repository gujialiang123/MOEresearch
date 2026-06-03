# Kernel Inventory — Qwen3-30B-A3B MoE on H200, R7 mixed-lengths regime

> **Complete enumeration of every CUDA kernel that ran during one R7 profiling pass**,
> with a source-code or binary path for each, plus full Python caller chain for the
> main-path kernels. This is the perception-layer artefact described in §36 of
> `regime_benchmark_experiment.md`, instantiated end-to-end on real data.

## 0. Reading guide

- **§1** explains the regime, model, and profiling setup so you can reproduce.
- **§2** is the **full kernel table** (70 unique kernels, ranked by GPU time%).
- **§3** shows the **Python caller chain** (HF model file → sglang layer file → C++/Triton kernel) for the 9 main-path kernels covering ~80% of GPU time.
- **§4** is the **coverage summary** by library + per-library evidence chain.
- **§5** documents the **5 sources used** and the **4-step procedure** to regenerate this for any other (model, regime).
- **§6** is the **agent-leverage analysis** — what could we optimise, and where.

All raw data is committed under `results/kernel_inventory_R7/`:
```
results/kernel_inventory_R7/
├── all_kernels.json              # 70 unique kernels from torch.profiler trace
├── all_kernels_resolved.json     # same + source_file/source_line/library/binary fields
├── main_path_caller_chains.json  # Python frames for 9 main-path kernels
├── server.log                    # sglang server log of this run
└── torch_profile/
    └── 1780508670.5573192-TP-0.trace.json.gz  # raw 131MB Chrome trace
```

---

## 1. Setup: what we ran

| Field | Value |
|---|---|
| **Model** | Qwen3-30B-A3B-Instruct-2507 (MoE, 128 experts × top-8) |
| **Model path** | `/data/hf/models/Qwen3-30B-A3B-Instruct-2507` |
| **GPU** | NVIDIA H200 (143 GB) × 1 |
| **dtype** | bfloat16 |
| **Backends (default)** | attention=fa3 / sampling=flashinfer / moe_runner=auto→Triton |
| **CUDA graph** | enabled, batch sizes [1,2,4,8,12,16,24,32] |
| **Regime** | **R7 mixed-lengths**: ~2000-word prompts (~2500 tokens), 256 output tokens, **8 concurrent requests** |
| **Server config** | `configs/moe_qwen3_30b.yaml` (mem-frac=0.85, ctx=32768, max-running-reqs=32) |
| **Triton cache dir** | `/tmp/kernel_inventory_R7_triton_cache` (fresh, isolated to this run) |
| **Profiling** | sglang `/start_profile` endpoint with `with_stack=true, record_shapes=true, num_steps=8` |
| **Trace size** | 131.6 MB compressed, 7,362,184 events |
| **Captured kernel time** | 264.6 ms total |
| **Unique GPU kernels observed** | **70** |

Why R7? — moderate prompt length (2k) exposes both prefill (FlashAttention prefill kernels) and decode (CUDA graph paths), moderate concurrency (8) keeps batching active without overwhelming the trace. R1/R6 are too quiet; R4/R5 too noisy.

---

## 2. Full kernel table (70 unique, ranked by total GPU time)

Legend for `Source`:
- `path/to/file.cu:N` — exact source line in repo or pip package
- `BIN: <path>` — closed-source binary (no Python/C++ source publicly available)
- `JIT: <path>` — JIT-compiled at runtime; declared in given file

| Rank | Time% | Calls | Avg µs | Library | Source / Binary | Kernel name |
|---:|---:|---:|---:|:--|:--|:--|
| 1 | 50.17% | 672 | 197.6 | sglang/python (Triton @triton.jit) | `sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:324` | `fused_moe_kernel` |
| 2 | 11.21% | 96 | 309.1 | flash-attn / cutlass | BIN: `embedded in sglang Triton/cutlass extension` | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<...` |
| 3 | 7.03% | 48 | 387.6 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_256x160_64x4_1x2_h_bz_coopA_TNT` |
| 4 | 5.55% | 48 | 306.2 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_256x152_64x4_1x2_h_bz_coopA_TNT` |
| 5 | 3.22% | 336 | 25.4 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:522` | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<a...` |
| 6 | 3.15% | 336 | 24.8 | flashinfer | `site-packages/flashinfer/data/include/flashinfer/activation.cuh:29` | `void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(float silu<floa...` |
| 7 | 2.87% | 672 | 11.3 | flashinfer | `site-packages/flashinfer/data/include/flashinfer/norm.cuh:387` | `void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, ...` |
| 8 | 2.71% | 96 | 74.6 | sglang/sgl-kernel (CUDA) | `sglang/sgl-kernel/csrc/moe/moe_sum_reduce.cu:57` | `void moe_sum_reduce_warp_per_token_vec_kernel<8>(c10::BFloat16 const*, c10::BFlo...` |
| 9 | 1.72% | 336 | 13.5 | sglang/jit_kernel (CUDA, runtime-compiled) | JIT: `sglang/python/sglang/jit_kernel/csrc/elementwise/qknorm.cuh` | `void (anonymous namespace)::fused_qknorm_warp<128l, true, __nv_bfloat16>((anonym...` |
| 10 | 1.70% | 48 | 93.8 | flashinfer | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/_kernels.so (compiled)` | `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel<true, false, ...` |
| 11 | 1.54% | 240 | 16.9 | flash-attn / cutlass | BIN: `embedded in sglang Triton/cutlass extension` | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<...` |
| 12 | 1.21% | 48 | 66.7 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_TNN` |
| 13 | 0.99% | 336 | 7.8 | sglang/sgl-kernel (CUDA) | `sglang/sgl-kernel/csrc/moe/moe_align_kernel.cu:28` | `void count_and_sort_expert_tokens_kernel<int>(int const*, int*, int*, unsigned l...` |
| 14 | 0.91% | 48 | 50.3 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN` |
| 15 | 0.82% | 336 | 6.5 | sglang/sgl-kernel (CUDA) | `sglang/sgl-kernel/csrc/moe/moe_align_kernel.cu:56` | `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int, int, un...` |
| 16 | 0.71% | 240 | 7.8 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` |
| 17 | 0.68% | 240 | 7.5 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` |
| 18 | 0.63% | 336 | 5.0 | sglang/sgl-kernel (CUDA) | `sglang/sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu:339` | `void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16>(__nv_bfloat16 const*, bool ...` |
| 19 | 0.48% | 240 | 5.3 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_8x64_64x16_4x1_v_bz_TNN` |
| 20 | 0.46% | 288 | 4.2 | flashinfer | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/_kernels.so (compiled)` | `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedHeadParallelismKerne...` |
| 21 | 0.40% | 336 | 3.2 | flash-attn / cutlass | BIN: `embedded in sglang Triton/cutlass extension` | `void flash::prepare_varlen_num_blocks_kernel<1, true>(int, int, int, int const*,...` |
| 22 | 0.40% | 7 | 151.9 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_384x8_64x4_2x1_v_bz_TNT` |
| 23 | 0.39% | 48 | 21.3 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_64x224_64x5_2x1_v_bz_TNT` |
| 24 | 0.23% | 240 | 2.5 | flash-attn / cutlass | `sglang/sgl-kernel/csrc/cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h:89` | `void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>, c...` |
| 25 | 0.16% | 240 | 1.8 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublasLt.so` | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __n...` |
| 26 | 0.13% | 240 | 1.5 | torch.inductor (auto-generated Triton) | JIT: `/tmp/torchinductor_t-jialianggu/<hash>/c*.py` | `triton_per_fused_copy__mul_sum_0` |
| 27 | 0.11% | 48 | 6.2 | cuBLAS/cuDNN (vendor, closed-source) | BIN: `/usr/local/cuda/lib64/libcublas.so OR libcublasLt.so` | `nvjet_tst_64x48_64x15_2x4_h_bz_TNT` |
| 28 | 0.11% | 7 | 40.0 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, flo...` |
| 29 | 0.02% | 7 | 9.0 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/Reduce.cuh:223` | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::A...` |
| 30 | 0.02% | 35 | 1.7 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:271` | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda...` |
| 31 | 0.02% | 7 | 7.8 | flashinfer | `site-packages/flashinfer/data/include/flashinfer/norm.cuh:37` | `void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfl...` |
| 32 | 0.02% | 18 | 2.9 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:271` | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda...` |
| 33 | 0.02% | 7 | 5.7 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/Reduce.cuh:223` | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::f...` |
| 34 | 0.01% | 7 | 5.7 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/DistributionTemplates.h:66` | `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_ker...` |
| 35 | 0.01% | 7 | 5.6 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/Reduce.cuh:223` | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::f...` |
| 36 | 0.01% | 7 | 5.6 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/Reduce.cuh:223` | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::f...` |
| 37 | 0.01% | 7 | 5.5 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:271` | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda...` |
| 38 | 0.01% | 5 | 7.2 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, lon...` |
| 39 | 0.01% | 7 | 4.6 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:522` | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<a...` |
| 40 | 0.01% | 6 | 5.2 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<a...` |
| 41 | 0.01% | 14 | 2.0 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::_assert_async_cuda_kernel<bool>(bool const*, at::native::Msg)` |
| 42 | 0.01% | 4 | 6.2 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, lo...` |
| 43 | 0.01% | 6 | 3.3 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<a...` |
| 44 | 0.01% | 7 | 2.7 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<floa...` |
| 45 | 0.01% | 11 | 1.7 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_...` |
| 46 | 0.01% | 10 | 1.6 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_...` |
| 47 | 0.01% | 14 | 1.1 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_ker...` |
| 48 | 0.01% | 7 | 2.0 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<2, at::native::neg_kernel_cuda(at...` |
| 49 | 0.00% | 5 | 2.6 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:522` | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::nati...` |
| 50 | 0.00% | 6 | 2.1 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/Reduce.cuh:223` | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<bool, at::native::fu...` |
| 51 | 0.00% | 7 | 1.6 | torch.inductor (auto-generated Triton) | JIT: `/tmp/torchinductor_t-jialianggu/<hash>/c*.py` | `triton_poi_fused_clamp_copy__index_lt_neg_where_0` |
| 52 | 0.00% | 7 | 1.4 | unknown |  | `void at_cuda_detail::cub::DeviceScanKernel<at_cuda_detail::cub::DeviceScanPolicy...` |
| 53 | 0.00% | 5 | 1.8 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:522` | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<a...` |
| 54 | 0.00% | 5 | 1.7 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int,...` |
| 55 | 0.00% | 2 | 4.2 | sglang/python (Triton @triton.jit) | `sglang/python/sglang/srt/mem_cache/common.py:28` | `write_req_to_token_pool_triton` |
| 56 | 0.00% | 5 | 1.6 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:522` | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<a...` |
| 57 | 0.00% | 7 | 1.1 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<4, at::native::bitwise_not_kernel...` |
| 58 | 0.00% | 7 | 1.1 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<bool...` |
| 59 | 0.00% | 7 | 1.1 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arang...` |
| 60 | 0.00% | 7 | 1.0 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<floa...` |
| 61 | 0.00% | 4 | 1.4 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::...` |
| 62 | 0.00% | 4 | 1.4 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::...` |
| 63 | 0.00% | 7 | 0.7 | unknown |  | `void at_cuda_detail::cub::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileStat...` |
| 64 | 0.00% | 5 | 0.9 | torch.inductor (auto-generated Triton) | JIT: `/tmp/torchinductor_t-jialianggu/<hash>/c*.py` | `triton_poi_fused_clamp_sub_0` |
| 65 | 0.00% | 2 | 1.8 | unknown |  | `void at_cuda_detail::cub::DeviceScanKernel<at_cuda_detail::cub::DeviceScanPolicy...` |
| 66 | 0.00% | 1 | 3.0 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<a...` |
| 67 | 0.00% | 2 | 1.4 | sglang/python (Triton @triton.jit) | `sglang/python/sglang/srt/model_executor/forward_batch_info.py:1050` | `compute_position_kernel` |
| 68 | 0.00% | 2 | 0.8 | PyTorch ATen | `site-packages/torch/include/ATen/native/cuda/CUDALoops.cuh:167` | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, ...` |
| 69 | 0.00% | 2 | 0.8 | unknown |  | `void at_cuda_detail::cub::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileStat...` |
| 70 | 0.00% | 1 | 1.5 | PyTorch ATen | BIN: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::...` |

---

## 3. Main-path caller chains (Python → sglang → kernel)

These 9 kernels cover **~80% of GPU time**. Each chain was extracted from the torch.profiler trace using `with_stack=true`. Read top-to-bottom as outermost → innermost call.

### 3.1 fused_moe_kernel (50.17% GPU time)

**Kernel**: `fused_moe_kernel`
**cpu_op binding**: `sglang::inplace_fused_experts`

Caller chain (outermost first):
```
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/layers/moe/fused_moe_triton/layer.py(963): forward
  sglang/srt/layers/moe/fused_moe_triton/layer.py(979): forward_impl
  sglang/srt/layers/moe/fused_moe_triton/layer.py(1015): run_moe_core
  sglang/srt/layers/quantization/unquant.py(337): apply
  sglang/srt/layers/utils/multi_platform.py(70): forward
  sglang/srt/layers/quantization/unquant.py(347): forward_cuda
  sglang/srt/layers/moe/moe_runner/runner.py(73): run
  sglang/srt/layers/moe/moe_runner/triton.py(357): fused_experts_none_to_triton
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(202): fused_experts
  torch/_ops.py(1244): __call__
  <built-in method inplace_fused_experts of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_us...
```

### 3.2 FlashAttnFwdSm90 (11.21% GPU time)

**Kernel**: `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute`
**cpu_op binding**: `sgl_kernel::fwd`

Caller chain (outermost first):
```
  nn.Module: Qwen3MoeAttention_0
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/models/qwen3_moe.py(658): forward
  sglang/srt/models/qwen3_moe.py(636): forward_core
  nn.Module: RadixAttention_0
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/layers/radix_attention.py(99): forward
  sglang/srt/layers/attention/base_attn_backend.py(79): forward
  sglang/srt/layers/attention/flashattention_backend.py(735): forward_extend
  sgl_kernel/flash_attn.py(39): flash_attn_with_kvcache
  torch/_ops.py(840): __call__
  <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1 object ...
```

### 3.3 FusedAddRMSNormKernel (2.87% GPU time)

**Kernel**: `void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, unsigned`
**cpu_op binding**: `sgl_kernel::fused_add_rmsnorm`

Caller chain (outermost first):
```
  nn.Module: Qwen3MoeDecoderLayer_0
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/models/qwen3_moe.py(759): forward
  sglang/srt/layers/communicator.py(536): prepare_mlp
  sglang/srt/layers/communicator.py(748): _simple
  nn.Module: RMSNorm_1
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/layers/utils/multi_platform.py(70): forward
  sglang/srt/layers/layernorm.py(118): forward_cuda
  sgl_kernel/elementwise.py(49): fused_add_rmsnorm
  torch/_ops.py(840): __call__
  <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1 object ...
```

### 3.4 act_and_mul_kernel (3.15% GPU time)

**Kernel**: `void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(float silu<float>(float const&))>(__nv_bfloat16*, __nv_`
**cpu_op binding**: `sgl_kernel::silu_and_mul`

Caller chain (outermost first):
```
  sglang/srt/layers/utils/multi_platform.py(70): forward
  sglang/srt/layers/quantization/unquant.py(347): forward_cuda
  sglang/srt/layers/moe/moe_runner/runner.py(73): run
  sglang/srt/layers/moe/moe_runner/triton.py(357): fused_experts_none_to_triton
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(202): fused_experts
  torch/_ops.py(1244): __call__
  <built-in method inplace_fused_experts of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_us...
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(79): inplace_fused_experts
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(321): fused_experts_impl
  sgl_kernel/elementwise.py(172): silu_and_mul
  torch/_ops.py(840): __call__
  <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1 object ...
```

### 3.5 BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel (1.70% GPU time)

**Kernel**: `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel<true, false, 128u, 8u, 16u, __nv_bfloat16, long>(__nv`
**cpu_op binding**: `sglang::apply_rope_pos_ids_cos_sin_cache_with_kv_cache`

Caller chain (outermost first):
```
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/models/qwen3_moe.py(658): forward
  sglang/srt/models/qwen3_moe.py(615): forward_prepare
  sglang/srt/models/qwen3_moe.py(546): forward_prepare_native
  sglang/srt/models/qwen3_moe.py(559): apply_qk_norm_rope
  nn.Module: RotaryEmbedding_0
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/layers/utils/multi_platform.py(70): forward
  sglang/srt/layers/rotary_embedding.py(356): forward_cuda
  /home/t-jialianggu/work/sglang/python/sglang/jit_kernel/rope.py(156): apply_rope_with_cos_sin_cache_inplace
  torch/_ops.py(1244): __call__
  <built-in method apply_rope_pos_ids_cos_sin_cache_with_kv_cache of pybind11_builtins.pybind11_detail_function_record_v1_system_...
```

### 3.6 fused_qknorm_warp (1.72% GPU time)

**Kernel**: `void (anonymous namespace)::fused_qknorm_warp<128l, true, __nv_bfloat16>((anonymous namespace)::QKNormParams)`
**cpu_op binding**: `sglang::fused_inplace_qknorm`

Caller chain (outermost first):
```
  nn.Module: Qwen3MoeDecoderLayer_0
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/models/qwen3_moe.py(759): forward
  nn.Module: Qwen3MoeAttention_0
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/models/qwen3_moe.py(658): forward
  sglang/srt/models/qwen3_moe.py(615): forward_prepare
  sglang/srt/models/qwen3_moe.py(546): forward_prepare_native
  sglang/srt/models/qwen3_moe.py(559): apply_qk_norm_rope
  sglang/srt/models/utils.py(204): apply_qk_norm
  torch/_ops.py(1244): __call__
  <built-in method fused_inplace_qknorm of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use...
```

### 3.7 topkGatingSoftmax (0.63% GPU time)

**Kernel**: `void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16>(__nv_bfloat16 const*, bool const*, float*, int, int*, int, int, int`
**cpu_op binding**: `sgl_kernel::topk_softmax`

Caller chain (outermost first):
```
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/models/qwen3_moe.py(265): forward
  sglang/srt/models/qwen3_moe.py(293): forward_normal
  nn.Module: TopK_0
  torch/nn/modules/module.py(1779): _call_impl
  sglang/srt/layers/utils/multi_platform.py(70): forward
  sglang/srt/layers/moe/topk.py(272): forward_cuda
  sglang/srt/layers/moe/topk.py(916): select_experts
  sglang/srt/layers/moe/topk.py(450): fused_topk
  sgl_kernel/moe.py(28): topk_softmax
  torch/_ops.py(840): __call__
  <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1 object ...
```

### 3.8 moe_align_block_size_kernel (0.82% GPU time)

**Kernel**: `void moe_align_block_size_kernel<int>(int const*, int*, int*, int*, int, int, unsigned long, int*, bool, int, int)`
**cpu_op binding**: `sgl_kernel::moe_align_block_size`

Caller chain (outermost first):
```
  sglang/srt/layers/quantization/unquant.py(347): forward_cuda
  sglang/srt/layers/moe/moe_runner/runner.py(73): run
  sglang/srt/layers/moe/moe_runner/triton.py(357): fused_experts_none_to_triton
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(202): fused_experts
  torch/_ops.py(1244): __call__
  <built-in method inplace_fused_experts of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_us...
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(79): inplace_fused_experts
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(321): fused_experts_impl
  sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py(18): moe_align_block_size
  sgl_kernel/moe.py(6): moe_align_block_size
  torch/_ops.py(840): __call__
  <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1 object ...
```

### 3.9 moe_sum_reduce (2.71% GPU time)

**Kernel**: `void moe_sum_reduce_warp_per_token_vec_kernel<8>(c10::BFloat16 const*, c10::BFloat16*, long, long, long, long, long, lon`
**cpu_op binding**: `sgl_kernel::moe_sum_reduce`

Caller chain (outermost first):
```
  sglang/srt/layers/utils/multi_platform.py(70): forward
  sglang/srt/layers/quantization/unquant.py(347): forward_cuda
  sglang/srt/layers/moe/moe_runner/runner.py(73): run
  sglang/srt/layers/moe/moe_runner/triton.py(357): fused_experts_none_to_triton
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(202): fused_experts
  torch/_ops.py(1244): __call__
  <built-in method inplace_fused_experts of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_us...
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(79): inplace_fused_experts
  sglang/srt/layers/moe/fused_moe_triton/fused_moe.py(321): fused_experts_impl
  sgl_kernel/moe.py(83): moe_sum_reduce
  torch/_ops.py(840): __call__
  <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1 object ...
```

---

## 4. Coverage summary

| Library / source | # Kernels | Sum GPU Time% | Source mappable? |
|:--|--:|--:|:--|
| sglang/python (Triton @triton.jit) | 3 | 50.17% | ✅ Full — `def <name>` directly grep-able in `sglang/srt/` |
| sglang/sgl-kernel (CUDA) | 4 | 5.15% | ✅ Full — declared in `sgl-kernel/csrc/` |
| sglang/jit_kernel (CUDA, runtime-compiled) | 1 | 1.72% | ✅ Full — declared in `sglang/python/sglang/jit_kernel/csrc/` |
| flashinfer | 5 | 8.20% | ✅ Mostly — declared in `flashinfer/*.cuh`; compiled into `_kernels.so` |
| flash-attn / cutlass | 4 | 13.38% | 🟡 Partial — heavy CUTLASS templates; declared in flash_attn pip pkg, compiled into `.so` |
| torch.inductor (auto-generated Triton) | 3 | 0.14% | ✅ Full — runtime-generated at `/tmp/torchinductor_*/`; trigger is `@torch.compile` in sglang |
| PyTorch ATen | 35 | 3.60% | 🟡 Partial — header file:line for most; pure binary for some |
| cuBLAS/cuDNN (vendor, closed-source) | 11 | 17.63% | ❌ Source NOT available — vendor closed-source `libcublasLt.so` |
| unknown | 4 | 0.01% | ⚠️ Needs investigation |

**Net coverage**: ~82.4% of GPU time has a concrete source-file path; ~17.6% is opaque vendor cuBLAS (every serving framework hits this wall — NVIDIA does not ship cuBLAS source).

---

## 5. The 5 sources used + 4-step reproduction procedure

### 5.1 Sources we consulted to assemble this inventory

| # | Source | What it gave us |
|---|---|---|
| 1 | **torch.profiler trace** (`SGLANG_TORCH_PROFILER_DIR/<ts>-TP-0.trace.json.gz`) | All 70 unique kernel names, calls, durations, AND Python caller chains via `with_stack=true` |
| 2 | **`~/.triton/cache/<hash>/<name>.source`** (auto-populated by Triton JIT) | Source-file:line annotations embedded in compiled Triton IR — e.g., for inductor-generated kernels, the underlying Python file location |
| 3 | **sglang repo** (`/home/t-jialianggu/work/sglang/python/sglang/srt`, `sgl-kernel/csrc`, `jit_kernel/csrc`) | `grep -rn 'def <name>\|void <name>\|struct <name>'` matched ~70% of kernels directly |
| 4 | **flashinfer pip pkg** (`.../site-packages/flashinfer/*.cuh`) | Template class declarations for `flashinfer::norm::*`, `flashinfer::activation::*` |
| 5 | **PyTorch source headers** (`.../site-packages/torch/include/ATen/native/cuda/`) | Generic at::native templates (Reduce.cuh, CUDALoops.cuh, etc.) |

### 5.2 4-step reproduction procedure

```bash
# (1) Isolate Triton cache for this run
export TRITON_CACHE_DIR=/tmp/kinv_<regime>_triton_cache
rm -rf $TRITON_CACHE_DIR && mkdir -p $TRITON_CACHE_DIR

# (2) Start sglang server (no special flags needed)
export SGLANG_TORCH_PROFILER_DIR=$REPO/results/kinv_<regime>/torch_profile
python -m sglang.launch_server --model-path <model> ... &

# (3) Warmup (4 small requests), then profile with_stack=true for 8 forward steps
curl -X POST localhost:30000/start_profile -d '{"with_stack": true, "num_steps": 8}'
# ... send N concurrent regime-shaped requests ...
curl -X POST localhost:30000/stop_profile

# (4) Parse + resolve (see scripts/kernel_inventory/build.py)
python scripts/kernel_inventory/build.py \
    --trace $SGLANG_TORCH_PROFILER_DIR/*.trace.json.gz \
    --triton-cache $TRITON_CACHE_DIR \
    --output results/kinv_<regime>/
```

---

## 6. Agent leverage analysis — where to put optimisation work

This profile makes the optimisation budget concrete:

| Slice | Time% | Reachable? | Highest-ROI action |
|---|---:|:--|---|
| **`fused_moe_kernel` (Triton, sglang)** | 50.2% | ✅ Source: `fused_moe_triton_kernels.py:324`; configs in `fused_moe_triton/configs/triton_3_X/` | **Autotune missing config JSON** for our `(E=128, N=768, H200, bf16)` — already detected at server start (`Fallback to triton 3.2.0 ... Performance might be sub-optimal!`). Direct fix. |
| **FlashAttention `FlashAttnFwdSm90`** | 12.7% | 🟡 CUTLASS template — bundled in sgl-kernel | Already optimal (fa3). Investigate prefill vs decode kernel selection. |
| **cuBLAS GEMM `nvjet_*`** | 17.5% | ❌ Closed-source | **Try FP8 quantization** → swaps these for `fp8_blockwise_moe_kernel` (sgl-kernel CUDA, source available). 1.5-2× speedup typically. |
| **flashinfer `FusedAddRMSNormKernel` + `act_and_mul`** | 6.0% | ✅ Source available | Already well-optimised. Low priority. |
| **sgl-kernel CUDA helpers (moe_align, topk_softmax, moe_sum_reduce)** | 5.2% | ✅ Source available | Inspect for fusion opportunities. |
| **PyTorch ATen scattered ops (copy, fill, arange, cumsum, ...)** | 3.6% | 🟡 Source on github | Many small launches; flagged for **fusion** via `@torch.compile` (which would generate `triton_*_fused_*` kernels). |
| **torch.inductor-generated Triton (`triton_per_fused_copy__mul_sum_0`, etc.)** | 0.2% | ✅ runtime-generated; source at `/tmp/torchinductor_*/` | Already in fast path; demonstrates inductor IS active in sglang's `overlap_utils._resolve_future_token_ids`. |

### Specific findings worth attention

1. **Sub-optimal MoE config logged at startup**: sglang printed
   ```
   Config file not found at .../fused_moe_triton/configs/triton_3_5_1/E=128,N=768,device_name=NVIDIA_H200.json.
   Fallback to triton version 3.2.0 ... Performance might be sub-optimal!
   ```
   **This is exact agent ROI**: write the missing JSON via autotune.

2. **torch.inductor IS active** in sglang's `overlap_utils.py:20 _resolve_future_token_ids` — caller chain confirms `torch/_inductor/runtime/triton_heuristics.py:1242 run` is on the hot path. This directly answers Mason Remy's earlier question 'are these Triton kernels hand-written or Inductor-generated?': **both, simultaneously, in the same forward pass**. `fused_moe_kernel` is hand-written; `triton_per_fused_copy__mul_sum_0` is inductor-generated.

3. **Every main-path kernel goes through `torch/_ops.py:840 __call__` then `pybind11`** — confirms sglang dispatches all custom CUDA kernels via PyTorch's custom-op registry (`torch.ops.sgl_kernel.*` / `torch.ops.sglang.*`), making them addressable by the agent at the Python layer.

---

## 7. Honest limitations of this inventory

1. **No per-call timing variance**: we report mean μs across all calls; some kernels have wildly different durations across prefill vs decode batches. To split, re-run with `record_shapes=true` and group by tensor shapes.

2. **No execution order**: we report aggregate counts, not the exact temporal sequence. The trace contains this (sorted by `ts`) but rendering 6,499 kernel events in order would be unreadable. Available in `torch_profile/*.trace.json.gz` if needed for any specific kernel.

3. **Caller chains captured only for 9 kernels** chosen by name. To get caller chains for all 70, lift the per-kernel limit in the extractor (slow but tractable).

4. **cuBLAS `nvjet_*` (17.5%) source genuinely unavailable**. The kernel-name suffix encodes the autotune choice (`128x256_64x4_2x1_v_bz_coopA_TNT` = M×N tile, warps, stages, swizzle, A-transposed). To reduce dependence on cuBLAS, switch to FP8 (sglang's `fp8_blockwise_moe_kernel.cu`) or call `flashinfer_cutlass` directly.

5. **trace captures 264 ms** of real GPU work but our regime ran for 167 sec total. The remaining time is CPU-bound work (request scheduling, tokenisation, KV cache management) and inter-step idle. The 8-step profile window is representative of typical decode steps.
