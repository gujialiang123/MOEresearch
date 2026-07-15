# Slack reply to Dey — decode is the bottleneck; which kernels to tune (2026-07-15)

> Copy-paste ready. Data sources at the bottom.

---

## ✉️ Reply (copy-paste)

Hi Dey — here's where we've landed. The story is a clean funnel: **tune knobs → best config → profile → decode is the bottleneck → break decode into kernels → target the specific hot kernels.**

### The funnel
**agent-tuned knobs/values → best configuration → profile & find decode is the bottleneck → analyze decode kernels → tune the specific hot kernels.**

### 1. We used the real agent workload and measured prefill vs decode wall
On the real toolagent workload (Qwen3-30B-A3B, H200, input ~2700 / output ~194 tok), under our best-tuned config, sweeping concurrency. **Prefill wall ≈ TTFT; Decode wall ≈ E2E − TTFT.**

| max-conc | achieved conc | prefill TTFT (ms) | decode (ms) | E2E (ms) | decode/prefill | **decode share** | TPOT (ms) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  | 1.0  | 91  | 854  | 945  | 9.4×  | **90.4%** | 4.3 |
| 4  | 4.0  | 68  | 1400 | 1468 | 20.5× | **95.4%** | 7.5 |
| 8  | 7.8  | 81  | 1933 | 2014 | 23.8× | **96.0%** | 9.7 |
| 16 | 12.3 | 98  | 2295 | 2393 | 23.5× | **95.9%** | 12.1 |
| 32 | 12.8 | 300 | 2184 | 2484 | 7.3×  | **87.9%** | 14.4 |
| 64 | 11.3 | 177 | 2013 | 2190 | 11.4× | **91.9%** | 11.4 |

**Across every concurrency level, decode is ~90% of the end-to-end wall** (88–96%). This agent workload is decode-bound — so decode is where the wall-clock lever is.

### 2. What actually runs in a decode step — the kernels
We profiled one decode step with Nsight Compute (Qwen3-30B, batch 32). **13 kernel launches**, each with its role and measured duration:

| # | kernel | what it does | dur (µs) |
|--:|---|---|--:|
| 1 | `norm::RMSNormKernel` | input RMSNorm (pre-attention layernorm) | 3.78 |
| 2 | `nvjet_tst_64x32…` | dense GEMM — QKV projection | 10.21 |
| 3 | `flashinfer::BatchQKApplyRotary` | RoPE rotary position embedding | 4.96 |
| 4 | `flash::prepare_varlen_num_blocks` | FlashAttention varlen scheduling setup | 6.59 |
| 5 | **`cutlass::…FlashAttnFwdSm90`** | **FlashAttention decode (the attention compute)** | **55.71** |
| 6 | `FlashAttnFwdCombine` | FlashAttention split-K partial-result combine | 3.78 |
| 7 | `nvjet_…_splitK` | dense GEMM — output projection (split-K) | 9.12 |
| 8 | `norm::FusedAddRMSNormKernel` | post-attention RMSNorm (fused residual add) | 4.58 |
| 9 | `nvjet_64x8…` | dense GEMM — small projection | 7.23 |
| 10 | `topkGatingSoftmax<…,8,128,…>` | MoE router: softmax + top-8 expert selection | 5.79 |
| 11 | **`fused_moe_kernel`** | **MoE expert GEMM — up/gate projection** | **41.76** |
| 12 | `activation::act_and_mul_kernel` | SiLU gate activation (MoE FFN) | 3.97 |
| 13 | **`fused_moe_kernel`** | **MoE expert GEMM — down projection** | **24.90** |

By family: dense-GEMM (nvjet) ×3, **MoE expert-GEMM (`fused_moe`) ×2**, RMSNorm ×2, FlashAttention ×2, plus RoPE / varlen-setup / router-topk / act_and_mul ×1 each.

### 3. Which kernels dominate → where to focus
Two hot spots eat the decode step:
- **FlashAttention decode — 55.7 µs** (kernel #5). Heavy because the agent context is long (~2700-tok KV to stream per step).
- **MoE expert GEMMs — 41.8 + 24.9 = 66.7 µs** (kernels #11 + #13, the two `fused_moe` calls). **This is the single biggest chunk of the step**, and it's ~80% DRAM-bound: per decode step the expert-weight move:compute ratio is ~103:1 — i.e. it's almost entirely streaming expert weights HBM→SM, computing <1% of the time.

Everything else (norms, RoPE, router, activation, the small dense GEMMs) is small (<11 µs each) and not where the time is.

**So the kernel-level opportunity is concentrated in exactly two places: the `fused_moe` expert GEMMs (biggest, movement-bound) and FlashAttention decode (long-context KV).** That's what we target next — the rest of the decode step isn't worth chasing.

### 4. Where this is going
- **Chendi** — prefill-phase opportunity.
- **Me** — the `fused_moe` movement bottleneck: "move once, serve more tokens" (bigger effective batch: spec-decode / concurrency / expert-parallel — already showed spec-decode +6%/+23% TBT and multi-stream 7.4× throughput), plus "move less weight" (fewer active experts: 8→6 costs ~0.5pp on GSM8K). Roofline says exact-method decode headroom is ~1.8–1.9× (Qwen3) / ~2.2–2.4× (LFM).

All numbers measured (NCU hardware counters + nsys timeline), not estimated — happy to share the per-kernel tables and the CSV.

---

## 📌 Data sources (don't send)

| Claim | Source |
|---|---|
| decode = 88–96% of E2E wall (concurrency 1→64) | `results/2026-07-15_v19_wall_sweep/qwen3-30b-a3b-bf16/wall_proportion.csv` |
| 13 decode kernels + roles + durations | `results/2026-07-15_v19b_ncu_decode/qwen3-30b-a3b-bf16/agent_decode_b32/ncu.ncu-rep` |
| fused_moe 79.8% DRAM-bound; move:compute 103:1 | `docs/2026-07-15/triton_moe_kernel_analysis.md`, `docs/2026-06-29/profiling_validation_of_universal_config.md` |
| roofline headroom 1.8–1.9×/2.2–2.4×; SM idle 67/78% | `docs/2026-07-10/reply_to_dey_tbt_headroom.md` |
| spec-decode +6/+23%; multi-stream 7.4× | `docs/2026-07-15/v11_realize_gap_results.md` |
| 8→6 experts ≈ −0.5pp GSM8K | `docs/2026-07-15/v17_gsm8k_topk_results.md` |
| scripts | `scripts/run_v19_wall_sweep.sh`, `scripts/run_v19b_ncu_decode.py` |
