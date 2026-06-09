# NCU report — R_medium_balanced

**Workload**: batch=8, in=800w, out=256 — decode (typical workload)
**Kernels profiled**: 30 (NCU `--set full`, no kernel filter)
**Source**: `ncu_summary.json` (parsed from `ncu_raw_full.csv` via `scripts/ncu_csv_wide_to_summary.py`)
**Raw .ncu-rep**: `R_medium_balanced_ncu.ncu-rep` (open with `ncu-ui` for Nsight Compute GUI)

## How to read

- **SM %**: SM throughput as % of peak sustained — high = compute pipeline busy
- **DRAM %**: HBM bandwidth as % of peak — high (≥70%) = memory-bound
- **Occupancy %** (warps active): how full the warp slots are — low (<30%) = grid/block too small
- **TC %** (tensor pipe active): how busy the Tensor Cores are — high on GEMM/MM is good
- **L1/L2 hit %**: cache hit rates
- **Long SB stall**: warps waiting on memory loads — high (>2) = memory-bound signal
- **Math throttle**: warps waiting on math pipe — high (>1.5) = TC saturated
- **Headroom %**: 100 - max(SM%, DRAM%); rough upper bound on improvement
- **Verdict**: derived from rules in `.github/skills/ncu-microarch/SKILL.md`

## Kernels ranked by headroom (highest first = most potential for optimization)

| # | Kernel | Verdict | SM% | DRAM% | Occ% | TC% | L1% | L2% | Long-SB | Math-Th | Headroom% |
|---|--------|---------|-----|-------|------|-----|-----|-----|---------|---------|-----------|
| 1 | `void at::vectorized_elementwise_kernel<4, at::CUDAFunctor...` | low_occupancy | 0.0 | 0.1 | 4.3 | 0.3 | 50.0 | 45.5 | 1.66 | 0.00 | 100.0 |
| 2 | `triton_poi_fused_clamp_sub_0` | low_occupancy | 0.0 | 0.0 | 1.6 | 0.0 | 0.0 | 33.1 | 25.83 | 0.00 | 100.0 |
| 3 | `void <unnamed>::elementwise_kernel_with_index<int, at::ar...` | low_occupancy | 0.0 | 0.0 | 2.8 | 0.0 | 0.0 | 48.6 | 0.00 | 0.00 | 100.0 |
| 4 | `void at_cuda_detail::DeviceScanInitKernel<at_cuda_detail:...` | low_occupancy | 0.0 | 0.0 | 5.5 | 0.6 | 0.0 | 41.1 | 0.09 | 0.00 | 100.0 |
| 5 | `void at::vectorized_elementwise_kernel<4, at::FillFunctor...` | low_occupancy | 0.0 | 0.0 | 6.0 | 0.0 | 0.0 | 38.7 | 0.00 | 0.00 | 100.0 |
| 6 | `void count_and_sort_expert_tokens_kernel<int>(const T1 *,...` | low_occupancy | 0.0 | 0.0 | 5.5 | 0.0 | 16.8 | 36.6 | 17.68 | 0.01 | 100.0 |
| 7 | `void at::vectorized_elementwise_kernel<2, at::CUDAFunctor...` | low_occupancy | 0.0 | 0.1 | 4.6 | 0.0 | 50.0 | 42.6 | 2.10 | 0.00 | 99.9 |
| 8 | `void flash::prepare_varlen_num_blocks_kernel<1, 1>(int, i...` | low_occupancy | 0.0 | 0.1 | 1.6 | 0.0 | 37.5 | 48.8 | 1.53 | 0.00 | 99.9 |
| 9 | `void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16>(cons...` | low_occupancy | 0.1 | 0.1 | 6.2 | 0.0 | 77.5 | 59.9 | 0.59 | 0.00 | 99.9 |
| 10 | `void at_cuda_detail::DeviceScanKernel<at_cuda_detail::Dev...` | low_occupancy | 0.1 | 0.2 | 6.2 | 0.0 | 66.7 | 67.8 | 1.83 | 0.00 | 99.8 |
| 11 | `void moe_align_block_size_kernel<int>(const T1 *, int *, ...` | latency_bound | 0.2 | 0.1 | 41.8 | 0.3 | 0.0 | 62.5 | 0.24 | 0.76 | 99.8 |
| 12 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 5.5 | 0.0 | 0.0 | 69.5 | 1.38 | 0.00 | 99.7 |
| 13 | `void at::index_elementwise_kernel<128, 4, void at::gpu_in...` | low_occupancy | 0.0 | 0.3 | 3.8 | 0.1 | 0.0 | 71.5 | 5.82 | 0.00 | 99.7 |
| 14 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 4.5 | 0.0 | 0.0 | 64.0 | 2.44 | 0.00 | 99.7 |
| 15 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 5.2 | 0.0 | 0.0 | 65.8 | 1.57 | 0.00 | 99.7 |
| 16 | `void norm::RMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2...` | low_occupancy | 0.5 | 0.3 | 12.2 | 0.0 | 25.0 | 52.7 | 6.16 | 0.07 | 99.5 |
| 17 | `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, ...` | low_occupancy | 0.5 | 0.5 | 12.4 | 0.0 | 40.0 | 52.4 | 4.51 | 0.08 | 99.5 |
| 18 | `void at::index_elementwise_kernel<128, 4, void at::gpu_in...` | low_occupancy | 0.8 | 0.4 | 6.2 | 0.2 | 17.7 | 64.2 | 7.45 | 0.00 | 99.2 |
| 19 | `void at::<unnamed>::indexSelectSmallIndex<c10::BFloat16, ...` | low_occupancy | 0.9 | 0.1 | 6.2 | 0.1 | 68.5 | 63.8 | 2.93 | 0.00 | 99.1 |
| 20 | `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhan...` | low_occupancy | 0.9 | 0.5 | 6.1 | 0.0 | 70.1 | 54.6 | 6.31 | 0.00 | 99.1 |
| 21 | `void at::elementwise_kernel<128, 4, void at::gpu_kernel_i...` | low_occupancy | 1.0 | 0.4 | 6.2 | 0.2 | 24.7 | 58.8 | 16.86 | 0.00 | 99.0 |
| 22 | `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(...` | low_occupancy | 1.3 | 0.6 | 6.1 | 0.2 | 58.3 | 50.0 | 6.56 | 0.00 | 98.7 |
| 23 | `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<...` | low_occupancy | 1.1 | 1.8 | 4.5 | 0.0 | 0.0 | 41.7 | 4.91 | 0.00 | 98.2 |
| 24 | `nvjet_tst_8x64_64x16_4x1_v_bz_TNN` | low_occupancy | 2.1 | 1.9 | 14.4 | 5.7 | 0.0 | 56.2 | 4.76 | 0.29 | 97.9 |
| 25 | `void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...` | low_occupancy | 2.8 | 2.3 | 12.2 | 0.1 | 6.4 | 44.4 | 4.44 | 0.15 | 97.2 |
| 26 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | low_occupancy | 3.1 | 1.5 | 15.3 | 0.1 | 0.0 | 42.3 | 10.31 | 0.04 | 96.9 |
| 27 | `void cutlass::device_kernel<flash::enable_sm90_or_later<f...` | low_occupancy | 23.9 | 35.0 | 18.5 | 30.8 | 4.0 | 5.3 | 2.86 | 0.20 | 65.0 |
| 28 | `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | low_occupancy | 8.2 | 42.8 | 14.4 | 3.3 | 0.0 | 6.3 | 8.97 | 0.28 | 57.2 |
| 29 | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` | low_occupancy | 7.9 | 51.1 | 14.3 | 4.3 | 0.0 | 4.2 | 7.94 | 0.28 | 48.9 |
| 30 | `fused_moe_kernel` | low_occupancy | 13.5 | 67.5 | 19.9 | 10.1 | 0.3 | 3.4 | 9.08 | 0.23 | 32.5 |

## Per-kernel notes (auto-derived)

**2. `triton_poi_fused_clamp_sub_0`** (low_occupancy)
  - long_scoreboard stall = 25.83 warps/issue — severe memory-wait

**6. `void count_and_sort_expert_tokens_kernel<int>(const T1 *, int *, int *, unsig...`** (low_occupancy)
  - long_scoreboard stall = 17.68 warps/issue — severe memory-wait

**7. `void at::vectorized_elementwise_kernel<2, at::CUDAFunctorOnSelf_add<long>, st...`** (low_occupancy)
  - long_scoreboard stall = 2.10 warps/issue — severe memory-wait

**11. `void moe_align_block_size_kernel<int>(const T1 *, int *, int *, int *, int, i...`** (latency_bound)
  - Both SM and DRAM under 30% — launch overhead or warp stalls dominate

**13. `void at::index_elementwise_kernel<128, 4, void at::gpu_index_kernel<void at::...`** (low_occupancy)
  - long_scoreboard stall = 5.82 warps/issue — severe memory-wait

**14. `void at::unrolled_elementwise_kernel<at::direct_copy_kernel_cuda(at::TensorIt...`** (low_occupancy)
  - long_scoreboard stall = 2.44 warps/issue — severe memory-wait

**16. `void norm::RMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, un...`** (low_occupancy)
  - long_scoreboard stall = 6.16 warps/issue — severe memory-wait

**17. `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned...`** (low_occupancy)
  - long_scoreboard stall = 4.51 warps/issue — severe memory-wait

**18. `void at::index_elementwise_kernel<128, 4, void at::gpu_index_kernel<void at::...`** (low_occupancy)
  - long_scoreboard stall = 7.45 warps/issue — severe memory-wait

**19. `void at::<unnamed>::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, ...`** (low_occupancy)
  - long_scoreboard stall = 2.93 warps/issue — severe memory-wait

**20. `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedHeadParallelismKe...`** (low_occupancy)
  - long_scoreboard stall = 6.31 warps/issue — severe memory-wait

**21. `void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl_nocast<at::direc...`** (low_occupancy)
  - long_scoreboard stall = 16.86 warps/issue — severe memory-wait

**22. `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(<unnamed>::QKNormPar...`** (low_occupancy)
  - long_scoreboard stall = 6.56 warps/issue — severe memory-wait

**23. `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<float>>(T1 *, const ...`** (low_occupancy)
  - long_scoreboard stall = 4.91 warps/issue — severe memory-wait

**24. `nvjet_tst_8x64_64x16_4x1_v_bz_TNN`** (low_occupancy)
  - long_scoreboard stall = 4.76 warps/issue — severe memory-wait

**25. `void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>...`** (low_occupancy)
  - long_scoreboard stall = 4.44 warps/issue — severe memory-wait

**26. `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, ...`** (low_occupancy)
  - long_scoreboard stall = 10.31 warps/issue — severe memory-wait

**27. `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm...`** (low_occupancy)
  - long_scoreboard stall = 2.86 warps/issue — severe memory-wait

**28. `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT`** (low_occupancy)
  - long_scoreboard stall = 8.97 warps/issue — severe memory-wait

**29. `nvjet_tst_64x8_64x16_4x1_v_bz_TNT`** (low_occupancy)
  - long_scoreboard stall = 7.94 warps/issue — severe memory-wait

**30. `fused_moe_kernel`** (low_occupancy)
  - long_scoreboard stall = 9.08 warps/issue — severe memory-wait
