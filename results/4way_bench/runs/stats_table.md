# 4-way MoE Benchmark — 3 runs each, mean ± std

Setup: Qwen3-30B-A3B-Instruct-2507 / H200 / bf16 / TP=1 / single-GPU sequential

## Throughput (req/s, mean ± std over 3 runs)

| Regime | sglang_triton | sglang_cutlass | vllm_triton | vllm_cutlass |
|---|---|---|---|---|
| R_short | 3.23 ± 0.02 | 0.70 ± 0.01 | 3.22 ± 0.15 | 3.23 ± 0.15 |
| R_medium | 4.51 ± 0.22 | 1.30 ± 0.03 | 4.57 ± 0.24 | 4.59 ± 0.23 |
| R_long | 4.38 ± 0.21 | 1.29 ± 0.06 | 4.18 ± 0.46 | 4.26 ± 0.45 |

## Per-run raw req/s (for transparency)

| Regime | sglang_triton | sglang_cutlass | vllm_triton | vllm_cutlass |
|---|---|---|---|---|
| R_short | 3.25, 3.22, 3.22 | 0.69, 0.71, 0.70 | 3.05, 3.31, 3.31 | 3.05, 3.32, 3.32 |
| R_medium | 4.54, 4.27, 4.71 | 1.27, 1.32, 1.31 | 4.29, 4.70, 4.71 | 4.33, 4.71, 4.72 |
| R_long | 4.14, 4.51, 4.48 | 1.22, 1.32, 1.33 | 3.64, 4.42, 4.47 | 3.74, 4.49, 4.55 |

## Throughput **warm only** (runs 2 and 3, excludes cold run 1)

| Regime | sglang_triton | sglang_cutlass | vllm_triton | vllm_cutlass |
|---|---|---|---|---|
| R_short | 3.22 ± 0.00 | 0.71 ± 0.00 | 3.31 ± 0.00 | 3.32 ± 0.00 |
| R_medium | 4.49 ± 0.31 | 1.31 ± 0.00 | 4.71 ± 0.00 | 4.72 ± 0.01 |
| R_long | 4.50 ± 0.02 | 1.33 ± 0.00 | 4.44 ± 0.03 | 4.52 ± 0.04 |

## Warm relative speed (vs sglang_triton warm)

| Regime | sglang Triton→CUTLASS | vLLM Triton→CUTLASS | sglang→vLLM (Triton) | sglang→vLLM (CUTLASS) |
|---|---|---|---|---|
| R_short | 0.22× | 1.00× | 1.03× | 4.70× |
| R_medium | 0.29× | 1.00× | 1.05× | 3.59× |
| R_long | 0.29× | 1.02× | 0.99× | 3.41× |
