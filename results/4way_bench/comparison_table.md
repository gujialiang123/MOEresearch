# 4-way MoE Backend Benchmark — Final Results (3 runs each, mean ± std)

**Setup**: Qwen3-30B-A3B-Instruct-2507 / H200 / bf16 / TP=1 / single-GPU sequential
**Each backend ran 3 times** to estimate noise. vLLM CUTLASS kernel use was verified via nsys profiling — see `nsys/EVIDENCE.md`.

## Throughput (req/s), mean ± std over 3 runs

| Regime | sglang_triton | sglang_cutlass | vllm_triton | vllm_cutlass |
|---|---|---|---|---|
| R_short | 3.23 ± 0.02 | 0.70 ± 0.01 | 3.22 ± 0.15 | 3.23 ± 0.15 |
| R_medium | 4.51 ± 0.22 | 1.30 ± 0.03 | 4.57 ± 0.24 | 4.59 ± 0.23 |
| R_long | 4.38 ± 0.21 | 1.29 ± 0.06 | 4.18 ± 0.46 | 4.26 ± 0.45 |

## Per-run raw req/s (transparency)

| Regime | sglang_triton | sglang_cutlass | vllm_triton | vllm_cutlass |
|---|---|---|---|---|
| R_short | 3.25, 3.22, 3.22 | 0.69, 0.71, 0.70 | 3.05, 3.31, 3.31 | 3.05, 3.32, 3.32 |
| R_medium | 4.54, 4.27, 4.71 | 1.27, 1.32, 1.31 | 4.29, 4.70, 4.71 | 4.33, 4.71, 4.72 |
| R_long | 4.14, 4.51, 4.48 | 1.22, 1.32, 1.33 | 3.64, 4.42, 4.47 | 3.74, 4.49, 4.55 |

## Warm-only throughput (runs 2 + 3, mean) — first run is cold for vLLM

| Regime | sglang_triton | sglang_cutlass | vllm_triton | vllm_cutlass |
|---|---|---|---|---|
| R_short | 3.22 | 0.71 | 3.31 | 3.32 |
| R_medium | 4.49 | 1.31 | 4.71 | 4.72 |
| R_long | 4.50 | 1.33 | 4.44 | 4.52 |

## Warm relative speed

| Regime | sglang Triton→CUTLASS | vLLM Triton→CUTLASS | sglang→vLLM (Triton) | sglang→vLLM (CUTLASS) |
|---|---|---|---|---|
| R_short | 0.22× | 1.00× | 1.03× | 4.70× |
| R_medium | 0.29× | 1.00× | 1.05× | 3.59× |
| R_long | 0.29× | 1.02× | 0.99× | 3.41× |

## Conclusions (warm)

1. **vLLM CUTLASS ≈ vLLM Triton** (1.00-1.02×) — contradicts vllm/.../unquantized.py:71 comment.
2. **sglang CUTLASS = 3.4-4.7× SLOWER than sglang Triton** — sglang wrapper overhead AND mandatory `--disable-cuda-graph` (cudagraph capture hangs detokenizer).
3. **vLLM CUTLASS = 3.4-4.7× faster than sglang CUTLASS** (same `flashinfer.cutlass_fused_moe` kernel).
4. **sglang Triton ≈ vLLM Triton** (-1% to +5%, within noise) — first-version claim of 20-59% gap was a cold-start methodology error.
