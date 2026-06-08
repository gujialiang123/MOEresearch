# nsys kernel-count comparison

R_medium bench (16 reqs × 256 tokens) under each backend, kernel counts and total GPU time:

| Category | vllm_cutlass time/calls | sglang_cutlass time/calls | vllm_triton time/calls |
|---|---|---|---|
| cutlass_gemm_sm90 (MoE) | 1924.4 ms / 10802 calls | 13950.7 ms / 97774 calls | — |
| cutlass_gemm (other) | 88.3 ms / 6464 calls | 342.1 ms / 146661 calls | — |
| cutlass (other) | 231.7 ms / 7464 calls | 1099.7 ms / 219428 calls | 1.3 ms / 196 calls |
| triton (any) | 16.8 ms / 2918 calls | 2.3 ms / 2038 calls | 258.7 ms / 9096 calls |
| fused_moe (flashinfer cuda) | — | — | 125.2 ms / 1248 calls |
| other | 1037.6 ms / 8574 calls | 1924.5 ms / 482327 calls | 302.2 ms / 22119 calls |
