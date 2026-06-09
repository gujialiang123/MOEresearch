# NCU report — R_long_prefill

**Workload**: batch=4, in=4000w, out=32 — prefill-dominated
**Kernels profiled**: 50 (NCU `--set full`, no kernel filter)
**Source**: `ncu_summary.json` (parsed from `ncu_raw_full.csv` via `scripts/ncu_csv_wide_to_summary.py`)
**Raw .ncu-rep**: `R_long_prefill_ncu.ncu-rep` (open with `ncu-ui` for Nsight Compute GUI)

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
| 1 | `void at_cuda_detail::DeviceScanInitKernel<at_cuda_detail:...` | low_occupancy | 0.0 | 0.0 | 5.2 | 0.8 | 0.0 | 35.6 | 0.09 | 0.00 | 100.0 |
| 2 | `void at::vectorized_elementwise_kernel<4, at::FillFunctor...` | low_occupancy | 0.0 | 0.0 | 5.9 | 0.0 | 0.0 | 38.1 | 0.00 | 0.00 | 100.0 |
| 3 | `void flash::prepare_varlen_num_blocks_kernel<1, 1>(int, i...` | low_occupancy | 0.0 | 0.1 | 1.6 | 0.0 | 57.1 | 63.6 | 2.54 | 0.00 | 99.9 |
| 4 | `void flash::prepare_varlen_num_blocks_kernel<1, 1>(int, i...` | low_occupancy | 0.0 | 0.1 | 1.6 | 0.0 | 57.1 | 59.9 | 2.06 | 0.00 | 99.9 |
| 5 | `void flash::prepare_varlen_num_blocks_kernel<1, 1>(int, i...` | low_occupancy | 0.0 | 0.1 | 1.6 | 0.0 | 57.1 | 59.8 | 2.07 | 0.00 | 99.9 |
| 6 | `void at_cuda_detail::DeviceScanKernel<at_cuda_detail::Dev...` | low_occupancy | 0.1 | 0.2 | 6.2 | 0.0 | 66.7 | 59.5 | 1.68 | 0.00 | 99.8 |
| 7 | `void moe_align_block_size_kernel<int>(const T1 *, int *, ...` | latency_bound | 0.1 | 0.2 | 49.7 | 0.0 | 0.0 | 53.5 | 58.16 | 0.18 | 99.8 |
| 8 | `void moe_align_block_size_kernel<int>(const T1 *, int *, ...` | latency_bound | 0.1 | 0.2 | 49.8 | 0.0 | 0.0 | 54.1 | 57.95 | 0.19 | 99.8 |
| 9 | `compute_position_kernel` | low_occupancy | 0.3 | 0.0 | 6.2 | 0.0 | 0.6 | 87.3 | 2.03 | 0.00 | 99.7 |
| 10 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 5.4 | 0.0 | 0.0 | 63.3 | 1.34 | 0.00 | 99.7 |
| 11 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 5.4 | 0.0 | 0.0 | 64.4 | 1.33 | 0.00 | 99.7 |
| 12 | `write_req_to_token_pool_triton` | low_occupancy | 0.1 | 0.5 | 6.2 | 0.0 | 69.8 | 43.7 | 12.95 | 0.00 | 99.5 |
| 13 | `void count_and_sort_expert_tokens_kernel<int>(const T1 *,...` | latency_bound | 0.5 | 0.3 | 90.4 | 0.0 | 1.8 | 91.2 | 3182.60 | 0.45 | 99.5 |
| 14 | `void count_and_sort_expert_tokens_kernel<int>(const T1 *,...` | latency_bound | 0.7 | 0.4 | 89.5 | 0.0 | 1.7 | 94.0 | 2348.67 | 0.48 | 99.3 |
| 15 | `void at::vectorized_gather_kernel<16, long>(char *, char ...` | low_occupancy | 0.8 | 1.0 | 11.6 | 0.0 | 4.0 | 54.4 | 15.50 | 0.16 | 99.0 |
| 16 | `fused_moe_kernel` | low_occupancy | 54.2 | 23.8 | 12.4 | 55.0 | 0.8 | 85.4 | 0.58 | 0.99 | 45.8 |
| 17 | `fused_moe_kernel` | low_occupancy | 54.2 | 23.8 | 12.4 | 55.1 | 0.8 | 86.4 | 0.58 | 0.99 | 45.8 |
| 18 | `void at::elementwise_kernel<128, 4, void at::gpu_kernel_i...` | tensor_core_idle | 57.4 | 36.1 | 92.5 | 2.7 | 21.3 | 50.4 | 19.30 | 0.41 | 42.6 |
| 19 | `void at::elementwise_kernel<128, 4, void at::gpu_kernel_i...` | tensor_core_idle | 57.5 | 36.2 | 92.4 | 2.7 | 21.3 | 50.5 | 19.27 | 0.41 | 42.5 |
| 20 | `void at::elementwise_kernel<128, 4, void at::gpu_kernel_i...` | tensor_core_idle | 57.6 | 36.2 | 92.4 | 2.7 | 21.3 | 50.4 | 19.26 | 0.41 | 42.4 |
| 21 | `void at::vectorized_gather_kernel<16, long>(char *, char ...` | balanced | 26.9 | 67.4 | 80.1 | 0.0 | 2.8 | 64.2 | 42.16 | 0.53 | 32.6 |
| 22 | `void cutlass::device_kernel<flash::enable_sm90_or_later<f...` | low_occupancy | 69.1 | 3.6 | 18.7 | 69.2 | 1.6 | 97.1 | 1.11 | 0.09 | 30.9 |
| 23 | `void cutlass::device_kernel<flash::enable_sm90_or_later<f...` | low_occupancy | 69.1 | 3.6 | 18.7 | 69.2 | 1.6 | 96.3 | 1.11 | 0.09 | 30.9 |
| 24 | `void cutlass::device_kernel<flash::enable_sm90_or_later<f...` | low_occupancy | 69.1 | 3.6 | 18.7 | 69.2 | 1.6 | 97.1 | 1.11 | 0.09 | 30.9 |
| 25 | `fused_moe_kernel` | low_occupancy | 69.8 | 22.6 | 12.4 | 70.6 | 0.3 | 79.3 | 0.40 | 1.50 | 30.2 |
| 26 | `fused_moe_kernel` | low_occupancy | 69.9 | 22.5 | 12.4 | 70.6 | 0.3 | 80.4 | 0.40 | 1.50 | 30.1 |
| 27 | `nvjet_tst_128x256_64x4_1x2_h_bz_coopA_TNN` | low_occupancy | 45.4 | 70.5 | 14.1 | 55.3 | 0.0 | 18.9 | 21.36 | 0.30 | 29.5 |
| 28 | `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhan...` | memory_bound | 24.9 | 71.3 | 38.6 | 0.0 | 59.4 | 51.6 | 21.64 | 0.14 | 28.7 |
| 29 | `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhan...` | memory_bound | 24.8 | 71.3 | 38.7 | 0.0 | 59.4 | 51.6 | 21.66 | 0.14 | 28.6 |
| 30 | `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhan...` | memory_bound | 25.0 | 71.4 | 38.5 | 0.0 | 59.4 | 51.7 | 21.66 | 0.14 | 28.6 |
| 31 | `nvjet_tst_128x256_64x4_1x2_h_bz_coopA_TNN` | low_occupancy | 47.3 | 71.5 | 14.1 | 54.9 | 0.0 | 18.9 | 22.01 | 0.30 | 28.5 |
| 32 | `void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16>(cons...` | tensor_core_idle | 71.6 | 6.4 | 77.2 | 0.1 | 72.7 | 72.0 | 1.17 | 5.42 | 28.4 |
| 33 | `void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16>(cons...` | tensor_core_idle | 71.8 | 6.4 | 77.4 | 0.1 | 72.7 | 71.8 | 1.20 | 5.41 | 28.2 |
| 34 | `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(...` | tensor_core_idle | 57.4 | 72.6 | 93.6 | 1.3 | 64.7 | 50.3 | 18.05 | 0.46 | 27.4 |
| 35 | `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(...` | tensor_core_idle | 57.4 | 72.7 | 93.5 | 1.3 | 64.6 | 50.3 | 17.82 | 0.46 | 27.3 |
| 36 | `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(...` | tensor_core_idle | 56.6 | 72.8 | 93.4 | 1.3 | 64.6 | 50.4 | 17.53 | 0.46 | 27.2 |
| 37 | `void norm::RMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2...` | tensor_core_idle | 74.4 | 67.1 | 91.6 | 0.0 | 45.5 | 53.2 | 6.56 | 1.35 | 25.6 |
| 38 | `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<...` | tensor_core_idle | 37.2 | 78.1 | 45.9 | 0.0 | 0.0 | 33.7 | 10.15 | 0.09 | 21.9 |
| 39 | `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<...` | tensor_core_idle | 37.1 | 78.1 | 45.7 | 0.0 | 0.0 | 33.9 | 9.84 | 0.09 | 21.9 |
| 40 | `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, ...` | tensor_core_idle | 52.6 | 82.4 | 92.0 | 0.0 | 45.6 | 50.7 | 14.35 | 0.60 | 17.6 |
| 41 | `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, ...` | tensor_core_idle | 52.4 | 82.8 | 91.9 | 0.0 | 45.4 | 50.6 | 14.43 | 0.59 | 17.2 |
| 42 | `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, ...` | tensor_core_idle | 52.4 | 82.9 | 92.0 | 0.0 | 45.5 | 50.8 | 14.31 | 0.59 | 17.1 |
| 43 | `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, ...` | tensor_core_idle | 52.8 | 83.0 | 92.5 | 0.0 | 45.6 | 50.7 | 14.32 | 0.59 | 17.0 |
| 44 | `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | low_occupancy | 89.6 | 17.5 | 14.8 | 91.7 | 0.0 | 85.1 | 3.28 | 0.27 | 10.4 |
| 45 | `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | low_occupancy | 89.6 | 18.2 | 14.8 | 91.8 | 0.0 | 83.4 | 3.30 | 0.27 | 10.4 |
| 46 | `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | low_occupancy | 89.6 | 17.7 | 14.8 | 91.8 | 0.0 | 85.4 | 3.28 | 0.27 | 10.4 |
| 47 | `void moe_sum_reduce_warp_per_token_vec_kernel<8>(const c1...` | memory_bound | 24.9 | 91.5 | 42.8 | 0.4 | 50.0 | 12.2 | 24.55 | 0.17 | 8.5 |
| 48 | `void moe_sum_reduce_warp_per_token_vec_kernel<8>(const c1...` | memory_bound | 25.2 | 91.6 | 42.8 | 0.4 | 50.0 | 12.3 | 23.87 | 0.18 | 8.4 |
| 49 | `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | low_occupancy | 94.6 | 24.7 | 14.8 | 96.0 | 0.0 | 73.7 | 3.19 | 0.27 | 5.4 |
| 50 | `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | low_occupancy | 94.7 | 24.7 | 14.8 | 96.0 | 0.0 | 72.0 | 3.18 | 0.27 | 5.3 |

## Per-kernel notes (auto-derived)

**3. `void flash::prepare_varlen_num_blocks_kernel<1, 1>(int, int, int, const int *...`** (low_occupancy)
  - long_scoreboard stall = 2.54 warps/issue — severe memory-wait

**4. `void flash::prepare_varlen_num_blocks_kernel<1, 1>(int, int, int, const int *...`** (low_occupancy)
  - long_scoreboard stall = 2.06 warps/issue — severe memory-wait

**5. `void flash::prepare_varlen_num_blocks_kernel<1, 1>(int, int, int, const int *...`** (low_occupancy)
  - long_scoreboard stall = 2.07 warps/issue — severe memory-wait

**7. `void moe_align_block_size_kernel<int>(const T1 *, int *, int *, int *, int, i...`** (latency_bound)
  - Both SM and DRAM under 30% — launch overhead or warp stalls dominate
  - long_scoreboard stall = 58.16 warps/issue — severe memory-wait

**8. `void moe_align_block_size_kernel<int>(const T1 *, int *, int *, int *, int, i...`** (latency_bound)
  - Both SM and DRAM under 30% — launch overhead or warp stalls dominate
  - long_scoreboard stall = 57.95 warps/issue — severe memory-wait

**9. `compute_position_kernel`** (low_occupancy)
  - long_scoreboard stall = 2.03 warps/issue — severe memory-wait

**12. `write_req_to_token_pool_triton`** (low_occupancy)
  - long_scoreboard stall = 12.95 warps/issue — severe memory-wait

**13. `void count_and_sort_expert_tokens_kernel<int>(const T1 *, int *, int *, unsig...`** (latency_bound)
  - Both SM and DRAM under 30% — launch overhead or warp stalls dominate
  - long_scoreboard stall = 3182.60 warps/issue — severe memory-wait

**14. `void count_and_sort_expert_tokens_kernel<int>(const T1 *, int *, int *, unsig...`** (latency_bound)
  - Both SM and DRAM under 30% — launch overhead or warp stalls dominate
  - long_scoreboard stall = 2348.67 warps/issue — severe memory-wait

**15. `void at::vectorized_gather_kernel<16, long>(char *, char *, T2 *, int, long, ...`** (low_occupancy)
  - long_scoreboard stall = 15.50 warps/issue — severe memory-wait

**18. `void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl_nocast<at::direc...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 19.29 warps/issue — severe memory-wait

**19. `void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl_nocast<at::direc...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 19.27 warps/issue — severe memory-wait

**20. `void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl_nocast<at::direc...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 19.26 warps/issue — severe memory-wait

**21. `void at::vectorized_gather_kernel<16, long>(char *, char *, T2 *, int, long, ...`** (balanced)
  - long_scoreboard stall = 42.16 warps/issue — severe memory-wait

**27. `nvjet_tst_128x256_64x4_1x2_h_bz_coopA_TNN`** (low_occupancy)
  - long_scoreboard stall = 21.36 warps/issue — severe memory-wait

**28. `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel<1, 0, 128,...`** (memory_bound)
  - DRAM ≥70% — algorithmic reuse beats tile tuning
  - long_scoreboard stall = 21.64 warps/issue — severe memory-wait

**29. `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel<1, 0, 128,...`** (memory_bound)
  - DRAM ≥70% — algorithmic reuse beats tile tuning
  - long_scoreboard stall = 21.66 warps/issue — severe memory-wait

**30. `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedKernel<1, 0, 128,...`** (memory_bound)
  - DRAM ≥70% — algorithmic reuse beats tile tuning
  - long_scoreboard stall = 21.67 warps/issue — severe memory-wait

**31. `nvjet_tst_128x256_64x4_1x2_h_bz_coopA_TNN`** (low_occupancy)
  - long_scoreboard stall = 22.01 warps/issue — severe memory-wait

**32. `void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16>(const T1 *, const bool *...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - math_pipe_throttle stall = 5.42 — tensor cores saturated

**33. `void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16>(const T1 *, const bool *...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - math_pipe_throttle stall = 5.41 — tensor cores saturated

**34. `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(<unnamed>::QKNormPar...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 18.05 warps/issue — severe memory-wait

**35. `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(<unnamed>::QKNormPar...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 17.82 warps/issue — severe memory-wait

**36. `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(<unnamed>::QKNormPar...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 17.53 warps/issue — severe memory-wait

**37. `void norm::RMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, un...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 6.56 warps/issue — severe memory-wait

**38. `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<float>>(T1 *, const ...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 10.15 warps/issue — severe memory-wait

**39. `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<float>>(T1 *, const ...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 9.84 warps/issue — severe memory-wait

**40. `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 14.35 warps/issue — severe memory-wait

**41. `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 14.43 warps/issue — severe memory-wait

**42. `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 14.31 warps/issue — severe memory-wait

**43. `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned...`** (tensor_core_idle)
  - Tensor Cores firing <10% — likely wrong dtype or non-TC kernel
  - long_scoreboard stall = 14.32 warps/issue — severe memory-wait

**44. `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT`** (low_occupancy)
  - long_scoreboard stall = 3.28 warps/issue — severe memory-wait

**45. `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT`** (low_occupancy)
  - long_scoreboard stall = 3.30 warps/issue — severe memory-wait

**46. `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT`** (low_occupancy)
  - long_scoreboard stall = 3.28 warps/issue — severe memory-wait

**47. `void moe_sum_reduce_warp_per_token_vec_kernel<8>(const c10::BFloat16 *, c10::...`** (memory_bound)
  - DRAM ≥70% — algorithmic reuse beats tile tuning
  - long_scoreboard stall = 24.55 warps/issue — severe memory-wait

**48. `void moe_sum_reduce_warp_per_token_vec_kernel<8>(const c10::BFloat16 *, c10::...`** (memory_bound)
  - DRAM ≥70% — algorithmic reuse beats tile tuning
  - long_scoreboard stall = 23.87 warps/issue — severe memory-wait

**49. `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN`** (low_occupancy)
  - long_scoreboard stall = 3.19 warps/issue — severe memory-wait

**50. `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN`** (low_occupancy)
  - long_scoreboard stall = 3.18 warps/issue — severe memory-wait
