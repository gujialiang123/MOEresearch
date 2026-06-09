# NCU report — R_concurrent_decode

**Workload**: batch=32, in=200w, out=256 — high-concurrency decode
**Kernels profiled**: 30 (NCU `--set full`, no kernel filter)
**Source**: `ncu_summary.json` (parsed from `ncu_raw_full.csv` via `scripts/ncu_csv_wide_to_summary.py`)
**Raw .ncu-rep**: `R_concurrent_decode_ncu.ncu-rep` (open with `ncu-ui` for Nsight Compute GUI)

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
| 1 | `triton_poi_fused_clamp_sub_0` | low_occupancy | 0.0 | 0.0 | 1.6 | 0.0 | 0.0 | 38.3 | 26.72 | 0.00 | 100.0 |
| 2 | `void <unnamed>::elementwise_kernel_with_index<int, at::ar...` | low_occupancy | 0.0 | 0.0 | 3.1 | 0.0 | 0.0 | 39.8 | 0.00 | 0.00 | 100.0 |
| 3 | `void at_cuda_detail::DeviceScanInitKernel<at_cuda_detail:...` | low_occupancy | 0.0 | 0.0 | 5.5 | 0.6 | 0.0 | 39.6 | 0.09 | 0.00 | 100.0 |
| 4 | `void at::vectorized_elementwise_kernel<4, at::FillFunctor...` | low_occupancy | 0.0 | 0.0 | 6.0 | 0.0 | 0.0 | 40.6 | 0.00 | 0.00 | 100.0 |
| 5 | `void flash::prepare_varlen_num_blocks_kernel<2, 1>(int, i...` | low_occupancy | 0.0 | 0.0 | 3.1 | 0.0 | 7.1 | 61.3 | 1.53 | 0.00 | 100.0 |
| 6 | `void count_and_sort_expert_tokens_kernel<int>(const T1 *,...` | low_occupancy | 0.0 | 0.0 | 11.9 | 0.0 | 28.7 | 55.7 | 56.84 | 0.01 | 100.0 |
| 7 | `void at::vectorized_elementwise_kernel<2, at::CUDAFunctor...` | low_occupancy | 0.0 | 0.1 | 4.2 | 0.0 | 50.0 | 41.8 | 1.63 | 0.00 | 99.9 |
| 8 | `void at::vectorized_elementwise_kernel<4, at::CUDAFunctor...` | low_occupancy | 0.0 | 0.1 | 4.0 | 0.3 | 50.0 | 41.2 | 1.65 | 0.00 | 99.9 |
| 9 | `void at_cuda_detail::DeviceScanKernel<at_cuda_detail::Dev...` | low_occupancy | 0.1 | 0.2 | 6.2 | 0.0 | 33.3 | 59.6 | 2.20 | 0.00 | 99.8 |
| 10 | `void moe_align_block_size_kernel<int>(const T1 *, int *, ...` | latency_bound | 0.2 | 0.1 | 40.8 | 0.3 | 0.0 | 62.0 | 0.90 | 0.73 | 99.8 |
| 11 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 5.3 | 0.0 | 0.0 | 67.9 | 1.57 | 0.00 | 99.7 |
| 12 | `void at::index_elementwise_kernel<128, 4, void at::gpu_in...` | low_occupancy | 0.0 | 0.3 | 3.5 | 0.1 | 0.0 | 70.8 | 7.17 | 0.00 | 99.7 |
| 13 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 5.4 | 0.0 | 0.0 | 64.6 | 1.66 | 0.00 | 99.7 |
| 14 | `void at::unrolled_elementwise_kernel<at::direct_copy_kern...` | low_occupancy | 0.0 | 0.3 | 5.2 | 0.0 | 0.0 | 67.2 | 1.66 | 0.00 | 99.7 |
| 15 | `void topkGatingSoftmax<__nv_bfloat16, 8, 128, 4, 16>(cons...` | low_occupancy | 0.4 | 0.1 | 6.2 | 0.0 | 72.8 | 64.2 | 0.94 | 0.00 | 99.6 |
| 16 | `void at::vectorized_gather_kernel<16, long>(char *, char ...` | low_occupancy | 0.8 | 0.2 | 11.7 | 0.0 | 2.7 | 75.3 | 15.96 | 0.17 | 99.2 |
| 17 | `void at::index_elementwise_kernel<128, 4, void at::gpu_in...` | low_occupancy | 0.9 | 0.5 | 6.1 | 0.2 | 17.6 | 68.2 | 8.23 | 0.00 | 99.1 |
| 18 | `void norm::RMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2...` | low_occupancy | 1.7 | 0.9 | 12.3 | 0.0 | 25.0 | 57.2 | 6.34 | 0.06 | 98.3 |
| 19 | `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, ...` | low_occupancy | 2.0 | 1.6 | 12.4 | 0.0 | 40.0 | 56.6 | 4.62 | 0.09 | 98.0 |
| 20 | `nvjet_tst_64x8_64x16_2x4_v_bz_TNT` | low_occupancy | 1.0 | 2.2 | 14.4 | 6.0 | 0.0 | 18.1 | 4.91 | 0.29 | 97.8 |
| 21 | `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhan...` | low_occupancy | 3.2 | 1.9 | 6.4 | 0.0 | 70.5 | 59.1 | 6.50 | 0.01 | 96.8 |
| 22 | `void at::elementwise_kernel<128, 4, void at::gpu_kernel_i...` | low_occupancy | 3.9 | 1.4 | 11.4 | 0.3 | 23.2 | 57.7 | 14.89 | 0.05 | 96.1 |
| 23 | `void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...` | low_occupancy | 4.6 | 0.1 | 12.4 | 0.0 | 87.5 | 68.1 | 6.17 | 0.04 | 95.4 |
| 24 | `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(...` | low_occupancy | 4.8 | 1.9 | 13.0 | 0.4 | 61.9 | 50.6 | 4.93 | 0.08 | 95.2 |
| 25 | `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<...` | low_occupancy | 4.3 | 6.2 | 8.3 | 0.0 | 0.0 | 40.6 | 4.97 | 0.04 | 93.8 |
| 26 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | low_occupancy | 7.5 | 5.2 | 21.5 | 0.2 | 0.0 | 36.0 | 14.07 | 0.11 | 92.5 |
| 27 | `void cutlass::device_kernel<flash::enable_sm90_or_later<f...` | low_occupancy | 28.7 | 38.4 | 18.6 | 34.1 | 2.5 | 3.8 | 2.81 | 0.17 | 61.6 |
| 28 | `nvjet_tst_64x32_64x16_4x1_v_bz_splitK_TNT` | low_occupancy | 7.7 | 41.8 | 14.3 | 13.1 | 0.0 | 13.6 | 8.76 | 0.27 | 58.2 |
| 29 | `nvjet_tst_64x32_64x16_4x1_v_bz_TNT` | low_occupancy | 7.7 | 49.7 | 14.2 | 16.9 | 0.0 | 10.3 | 8.09 | 0.28 | 50.3 |
| 30 | `fused_moe_kernel` | memory_bound | 16.8 | 79.8 | 44.8 | 12.8 | 0.3 | 9.0 | 25.50 | 0.38 | 20.2 |

## Per-kernel notes (auto-derived)

**1. `triton_poi_fused_clamp_sub_0`** (low_occupancy)
  - long_scoreboard stall = 26.72 warps/issue — severe memory-wait

**6. `void count_and_sort_expert_tokens_kernel<int>(const T1 *, int *, int *, unsig...`** (low_occupancy)
  - long_scoreboard stall = 56.84 warps/issue — severe memory-wait

**9. `void at_cuda_detail::DeviceScanKernel<at_cuda_detail::DeviceScanPolicy<int, s...`** (low_occupancy)
  - long_scoreboard stall = 2.20 warps/issue — severe memory-wait

**10. `void moe_align_block_size_kernel<int>(const T1 *, int *, int *, int *, int, i...`** (latency_bound)
  - Both SM and DRAM under 30% — launch overhead or warp stalls dominate

**12. `void at::index_elementwise_kernel<128, 4, void at::gpu_index_kernel<void at::...`** (low_occupancy)
  - long_scoreboard stall = 7.17 warps/issue — severe memory-wait

**16. `void at::vectorized_gather_kernel<16, long>(char *, char *, T2 *, int, long, ...`** (low_occupancy)
  - long_scoreboard stall = 15.96 warps/issue — severe memory-wait

**17. `void at::index_elementwise_kernel<128, 4, void at::gpu_index_kernel<void at::...`** (low_occupancy)
  - long_scoreboard stall = 8.23 warps/issue — severe memory-wait

**18. `void norm::RMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, un...`** (low_occupancy)
  - long_scoreboard stall = 6.34 warps/issue — severe memory-wait

**19. `void norm::FusedAddRMSNormKernel<8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned...`** (low_occupancy)
  - long_scoreboard stall = 4.62 warps/issue — severe memory-wait

**20. `nvjet_tst_64x8_64x16_2x4_v_bz_TNT`** (low_occupancy)
  - long_scoreboard stall = 4.91 warps/issue — severe memory-wait

**21. `void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheEnhancedHeadParallelismKe...`** (low_occupancy)
  - long_scoreboard stall = 6.49 warps/issue — severe memory-wait

**22. `void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl_nocast<at::direc...`** (low_occupancy)
  - long_scoreboard stall = 14.89 warps/issue — severe memory-wait

**23. `void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<8>...`** (low_occupancy)
  - long_scoreboard stall = 6.17 warps/issue — severe memory-wait

**24. `void <unnamed>::fused_qknorm_warp<128, 1, __nv_bfloat16>(<unnamed>::QKNormPar...`** (low_occupancy)
  - long_scoreboard stall = 4.93 warps/issue — severe memory-wait

**25. `void activation::act_and_mul_kernel<__nv_bfloat16, &silu<float>>(T1 *, const ...`** (low_occupancy)
  - long_scoreboard stall = 4.97 warps/issue — severe memory-wait

**26. `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, ...`** (low_occupancy)
  - long_scoreboard stall = 14.07 warps/issue — severe memory-wait

**27. `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm...`** (low_occupancy)
  - long_scoreboard stall = 2.81 warps/issue — severe memory-wait

**28. `nvjet_tst_64x32_64x16_4x1_v_bz_splitK_TNT`** (low_occupancy)
  - long_scoreboard stall = 8.76 warps/issue — severe memory-wait

**29. `nvjet_tst_64x32_64x16_4x1_v_bz_TNT`** (low_occupancy)
  - long_scoreboard stall = 8.09 warps/issue — severe memory-wait

**30. `fused_moe_kernel`** (memory_bound)
  - DRAM ≥70% — algorithmic reuse beats tile tuning
  - long_scoreboard stall = 25.50 warps/issue — severe memory-wait
