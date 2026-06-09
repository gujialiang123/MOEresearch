# NCU report — R_short_decode

**Workload**: batch=1, in=100w, out=256 — decode (worst expert utilization)
**Kernels profiled**: 30 (NCU `--set full`, no kernel filter)
**Source**: `ncu_summary.json` (parsed from `ncu_raw_full.csv` via `scripts/ncu_csv_wide_to_summary.py`)
**Raw .ncu-rep**: `R_short_decode_ncu.ncu-rep` (open with `ncu-ui` for Nsight Compute GUI)

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
| 1 | `triton_poi_fused_clamp_sub_0` | low_occupancy | 0.0 | 0.0 | 1.6 | 0.0 | 0.0 | 48.8 | 27.12 | 0.00 | 100.0 |
| 2 | `void <unnamed>::elementwise_kernel_with_index<int, at::ar...` | low_occupancy | 0.0 | 0.0 | 2.8 | 0.0 | 0.0 | 40.4 | 0.00 | 0.00 | 100.0 |
| 3 | `void at_cuda_detail::DeviceScanInitKernel<at_cuda_detail:...` | low_occupancy | 0.0 | 0.0 | 5.2 | 0.8 | 0.0 | 41.8 | 0.09 | 0.00 | 100.0 |
| 4 | `void at::vectorized_elementwise_kernel<4, at::FillFunctor...` | low_occupancy | 0.0 | 0.0 | 5.9 | 0.0 | 0.0 | 37.5 | 0.00 | 0.00 | 100.0 |
| 5 | `void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16>(cons...` | low_occupancy | 0.0 | 0.0 | 1.9 | 0.0 | 75.0 | 59.2 | 0.51 | 0.00 | 100.0 |
| 6 | `void count_and_sort_expert_tokens_kernel<int>(const T1 *,...` | low_occupancy | 0.0 | 0.0 | 4.4 | 0.0 | 0.0 | 49.1 | 8.31 | 0.03 | 100.0 |
| 7 | `void at::vectorized_elementwise_kernel<2, at::CUDAFunctor...` | low_occupancy | 0.0 | 0.1 | 4.6 | 0.0 | 50.0 | 43.8 | 1.05 | 0.00 | 99.9 |
| 8 | `void at::vectorized_elementwise_kernel<4, at::CUDAFunctor...` | low_occupancy | 0.0 | 0.1 | 4.2 | 0.3 | 50.0 | 44.2 | 1.70 | 0.00 | 99.9 |
| 9 | `void norm::RMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2...` | low_occupancy | 0.1 | 0.1 | 11.6 | 0.0 | 25.0 | 50.9 | 6.44 | 0.07 | 99.9 |
| 10 | `void flash::prepare_varlen_num_blocks_kernel<1, 1>(int, i...` | low_occupancy | 0.0 | 0.1 | 1.6 | 0.0 | 57.1 | 50.2 | 1.99 | 0.00 | 99.9 |
| 11 | `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, ...` | low_occupancy | 0.1 | 0.1 | 12.4 | 0.0 | 40.0 | 50.6 | 6.05 | 0.10 | 99.9 |
| 12 | `void at_cuda_detail::DeviceScanKernel<at_cuda_detail::Dev...` | low_occupancy | 0.1 | 0.2 | 6.2 | 0.0 | 66.7 | 58.8 | 2.26 | 0.00 | 99.8 |
| 13 | `void at::index_elementwise_kernel<128, 4, void at::gpu_in...` | low_occupancy | 0.0 | 0.2 | 5.8 | 0.2 | 10.5 | 71.1 | 9.94 | 0.00 | 99.8 |
| 14 | `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(...` | low_occupancy | 0.2 | 0.1 | 6.2 | 0.2 | 58.3 | 55.0 | 4.50 | 0.00 | 99.8 |
| 15 | `void moe_align_block_size_kernel<int>(const T1 *, int *, ...` | latency_bound | 0.2 | 0.1 | 40.6 | 0.3 | 0.0 | 59.3 | 0.09 | 0.76 | 99.8 |
| 16 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 5.8 | 0.0 | 0.0 | 63.6 | 0.59 | 0.00 | 99.7 |
| 17 | `void at::index_elementwise_kernel<128, 4, void at::gpu_in...` | low_occupancy | 0.0 | 0.3 | 3.5 | 0.1 | 0.0 | 66.9 | 7.15 | 0.00 | 99.7 |
| 18 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 5.0 | 0.0 | 0.0 | 63.8 | 2.17 | 0.00 | 99.7 |
| 19 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 4.9 | 0.0 | 0.0 | 64.3 | 1.23 | 0.00 | 99.7 |
| 20 | `void at::<unnamed>::indexSelectSmallIndex<c10::BFloat16, ...` | low_occupancy | 0.3 | 0.1 | 6.2 | 0.1 | 33.1 | 53.9 | 7.02 | 0.00 | 99.7 |
| 21 | `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhan...` | low_occupancy | 0.3 | 0.1 | 2.8 | 0.0 | 53.5 | 48.6 | 6.44 | 0.00 | 99.7 |
| 22 | `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<...` | low_occupancy | 0.2 | 0.3 | 4.6 | 0.0 | 0.0 | 46.0 | 4.56 | 0.00 | 99.7 |
| 23 | `triton_per_fused_copy__mul_sum_0` | low_occupancy | 0.5 | 0.2 | 12.4 | 0.0 | 5.6 | 45.0 | 6.38 | 0.03 | 99.5 |
| 24 | `nvjet_tst_64x8_64x16_2x1_v_bz_TNT` | low_occupancy | 0.3 | 1.9 | 14.4 | 6.3 | 0.0 | 20.7 | 3.93 | 0.30 | 98.1 |
| 25 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | low_occupancy | 2.2 | 0.2 | 8.0 | 0.0 | 0.0 | 46.3 | 1.91 | 0.04 | 97.8 |
| 26 | `void cutlass::device_kernel<flash::enable_sm90_or_later<f...` | low_occupancy | 2.4 | 1.2 | 17.2 | 3.9 | 10.7 | 35.5 | 1.19 | 0.09 | 97.6 |
| 27 | `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | low_occupancy | 7.9 | 43.2 | 14.4 | 3.3 | 0.0 | 5.5 | 8.87 | 0.28 | 56.8 |
| 28 | `fused_moe_kernel` | low_occupancy | 12.1 | 50.5 | 12.0 | 8.0 | 0.7 | 2.8 | 4.06 | 0.06 | 49.5 |
| 29 | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` | low_occupancy | 8.0 | 51.0 | 14.3 | 4.2 | 0.0 | 4.0 | 8.27 | 0.27 | 49.0 |
| 30 | `fused_moe_kernel` | low_occupancy | 11.7 | 63.8 | 9.1 | 8.9 | 0.1 | 1.8 | 3.33 | 0.03 | 36.2 |

## Per-kernel notes (auto-derived)

**1. `triton_poi_fused_clamp_sub_0`** (low_occupancy)
  - long_scoreboard stall = 27.12 warps/issue — severe memory-wait

**6. `void count_and_sort_expert_tokens_kernel<int>(const T1 *, int *, int *, unsig...`** (low_occupancy)
  - long_scoreboard stall = 8.31 warps/issue — severe memory-wait

**9. `void norm::RMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, un...`** (low_occupancy)
  - long_scoreboard stall = 6.44 warps/issue — severe memory-wait

**11. `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned...`** (low_occupancy)
  - long_scoreboard stall = 6.05 warps/issue — severe memory-wait

**12. `void at_cuda_detail::DeviceScanKernel<at_cuda_detail::DeviceScanPolicy<int, s...`** (low_occupancy)
  - long_scoreboard stall = 2.26 warps/issue — severe memory-wait

**13. `void at::index_elementwise_kernel<128, 4, void at::gpu_index_kernel<void at::...`** (low_occupancy)
  - long_scoreboard stall = 9.94 warps/issue — severe memory-wait

**14. `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(<unnamed>::QKNormPar...`** (low_occupancy)
  - long_scoreboard stall = 4.50 warps/issue — severe memory-wait

**15. `void moe_align_block_size_kernel<int>(const T1 *, int *, int *, int *, int, i...`** (latency_bound)
  - Both SM and DRAM under 30% — launch overhead or warp stalls dominate

**17. `void at::index_elementwise_kernel<128, 4, void at::gpu_index_kernel<void at::...`** (low_occupancy)
  - long_scoreboard stall = 7.15 warps/issue — severe memory-wait

**18. `void at::unrolled_elementwise_kernel<at::direct_copy_kernel_cuda(at::TensorIt...`** (low_occupancy)
  - long_scoreboard stall = 2.17 warps/issue — severe memory-wait

**20. `void at::<unnamed>::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, ...`** (low_occupancy)
  - long_scoreboard stall = 7.02 warps/issue — severe memory-wait

**21. `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedHeadParallelismKe...`** (low_occupancy)
  - long_scoreboard stall = 6.44 warps/issue — severe memory-wait

**22. `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<float>>(T1 *, const ...`** (low_occupancy)
  - long_scoreboard stall = 4.56 warps/issue — severe memory-wait

**23. `triton_per_fused_copy__mul_sum_0`** (low_occupancy)
  - long_scoreboard stall = 6.38 warps/issue — severe memory-wait

**24. `nvjet_tst_64x8_64x16_2x1_v_bz_TNT`** (low_occupancy)
  - long_scoreboard stall = 3.93 warps/issue — severe memory-wait

**27. `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT`** (low_occupancy)
  - long_scoreboard stall = 8.87 warps/issue — severe memory-wait

**28. `fused_moe_kernel`** (low_occupancy)
  - long_scoreboard stall = 4.06 warps/issue — severe memory-wait

**29. `nvjet_tst_64x8_64x16_4x1_v_bz_TNT`** (low_occupancy)
  - long_scoreboard stall = 8.27 warps/issue — severe memory-wait

**30. `fused_moe_kernel`** (low_occupancy)
  - long_scoreboard stall = 3.33 warps/issue — severe memory-wait
