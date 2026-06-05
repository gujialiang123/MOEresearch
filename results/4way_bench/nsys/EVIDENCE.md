# NSys evidence — vLLM CUTLASS is genuinely running CUTLASS kernels

**Method**: launched vLLM under `nsys profile -t cuda -s none --capture-range=none`, ran warmup + 1×R_medium bench (16 reqs, conc=8, 800-word prompts), SIGINT-ed nsys to flush profile, extracted GPU kernel time summary with `nsys stats --report cuda_gpu_kern_sum --format csv`.

Two profiles: `vllm_cutlass2.nsys-rep` (with `--kernel-config '{"moe_backend":"flashinfer_cutlass"}'`) and `vllm_triton.nsys-rep` (with `"moe_backend":"triton"`).

## Result: kernels are completely disjoint between the two runs

| Kernel category | vLLM CUTLASS run | vLLM TRITON run |
|---|---|---|
| `cutlass::device_kernel<...sm90...gemm...>` (MoE GEMMs) | **1924 ms / 58.3% / 10802 calls** | **0 ms / 0 calls** |
| `triton_*` (incl. fused MoE) | 16.8 ms / 0.5% / 2918 calls (only inductor utility kernels: rms_norm, embedding) | **259 ms / 37.0% / 9096 calls (includes `triton_red_fused_fused_add_rms_norm_moe_forward_0`)** |
| `fused_moe::run_global` (flashinfer CUDA helper) | 117 ms / 3.5% / 528 calls | 125 ms / 18.2% / 1248 calls |

The CUTLASS sm_90 GEMM kernels (the actual flashinfer cutlass_fused_moe path) are present in 10802 instances when CUTLASS is selected, and **zero** instances when Triton is selected. Conversely, the Triton MoE forward kernel calls are ~10× more frequent in the Triton run.

This is binary proof that `--kernel-config '{"moe_backend":"flashinfer_cutlass"}'` actually dispatches to the CUTLASS kernel, not silently falling back to Triton.

## Sample CUTLASS kernel name (truncated)

```
void cutlass::device_kernel<
  cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<cute::tuple<long, long, long>>,
    cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm90ArrayTmaGmmaWarpSpecialized<(int)12, ...>
```

The `MainloopSm90ArrayTmaGmmaWarpSpecialized` template is the SM90-specific CUTLASS grouped-GEMM mainloop with TMA + WGMMA + warp specialization — i.e., genuine Hopper-tuned CUTLASS, not a generic fallback.

## Raw artifacts

- `vllm_cutlass2.nsys-rep` (50 MB) — full nsys profile, viewable with `nsys-ui`
- `vllm_cutlass_kernels.csv` — full kernel summary
- `vllm_cutlass_kernel_summary.txt` — pre-computed category breakdown
- `vllm_triton.nsys-rep` — same but for Triton backend
- `vllm_triton_kernels.csv` — Triton kernel summary
