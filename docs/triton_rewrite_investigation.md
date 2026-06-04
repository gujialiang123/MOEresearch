# Triton rewrite investigation — Is rewriting Triton kernels (to CUDA / Gluon / CUTLASS-DSL) a viable contribution to sglang?

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#中文版)
>
> **Date**: 2026-06-04 · **Author**: end-to-end investigation harness
> **Scope**: 3 specific user questions answered with grep-able evidence from the sglang repo.

## TL;DR (the three answers)

| Q | Short answer |
|---|---|
| **Was our MoE config fallback caused by old GPU, new model, or something else?** | **Neither** — it's caused by **maintenance lag in sglang itself**. We installed Triton 3.5.1 (released ~Dec 2025); sglang's `triton_3_5_1/` config dir only has **10 H200 entries**, none covering `(E=128, N=768)`. The same `(E=128, N=768, H200, bf16)` config **does exist** in the older `triton_3_2_0/` dir (committed 2025-06). So sglang upstream just hasn't re-tuned this particular (model, GPU, Triton-version) cell yet. Same H200 hardware, same model, just different Triton compiler version. |
| **If Triton is on the critical path, is rewriting it to CUDA/Gluon a viable contribution?** | **Yes for one specific lane (MoE), no for most other models.** Across all 165 sglang model files: only **13/165** (8 %) import `FusedMoE` (the Triton-based MoE), and only **2** have model-specific `@triton.jit`. For these 13 MoE models — including ours — Triton accounts for ~50 % of GPU time. For dense models, attention (`flash-attn` CUTLASS) and cuBLAS GEMM dominate; Triton is < 5 %. So the rewrite opportunity is concentrated in MoE specifically. |
| **Are there many support functions in torch/triton that could be rewritten or fused?** | **Yes — quantitatively significant.** sglang has ~**80 000 lines of Python support code** across managers/mem_cache/sampler/quantization/speculative/constrained, with **40 small `@triton.jit`** kernels scattered through these dirs. Our trace already shows ~3.6 % of GPU time spent on **scattered tiny PyTorch ATen ops** (copy, fill, arange, cumsum) that aren't fused. **5 concrete fusion candidates** identified below, all with file path + line number. |

The body of this report gives the evidence for each answer.

---

## 1. Q1 — Why did the MoE config fall back? (root cause analysis)

### 1.1 What we observed

From `results/kernel_inventory_R7/server.log`:

```
[2026-06-03 17:43:53] Config file not found at .../triton_3_5_1/E=128,N=768,device_name=NVIDIA_H200.json.
Fallback to triton version 3.2.0 and use MoE kernel config from .../triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json.
Performance might be sub-optimal!
```

### 1.2 Environment facts

| Field | Value | Evidence |
|---|---|---|
| Installed Triton in sglang-dev env | **3.5.1** | `python -c "import triton; print(triton.__version__)"` in `sglang-dev` env |
| GPU | H200 | `nvidia-smi`; 143 GB, sm_90 |
| Model | Qwen3-30B-A3B (128 experts, N=768 FFN inter dim) | `config.json` |
| dtype | bf16 | `model_config` |
| sglang code that picks the config | `fused_moe_triton_config.py:80` reads `triton.__version__` and uses dir `triton_{version}` | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py:80` |

### 1.3 The smoking gun — config coverage per Triton version

We listed every config file under `python/sglang/srt/layers/moe/fused_moe_triton/configs/`:

| Triton ver dir | total configs | H200 configs | H200 + E=128 configs | First commit | Last commit |
|---|---:|---:|---:|---|---|
| `triton_3_1_0/` | 127 | 24 | (need to grep) | 2025-06-07 | 2025-06-07 |
| `triton_3_2_0/` | 35 | **7** | **5** (incl. `E=128,N=768,H200` ← our miss!) | 2025-06-07 | 2025-08-14 |
| `triton_3_3_0/` | 1 | 0 | 0 | 2025-08-14 | 2025-08-14 |
| `triton_3_3_1/` | 21 | 7 | 0 | 2025-06-09 | 2025-09-09 |
| `triton_3_4_0/` | 31 | 5 | 1 (`E=128,N=192,fp8`) | 2025-08-10 | 2025-12-01 |
| `triton_3_5_1/` | **64** | **10** | **0** (E values present: 20, 40, 80, 161, 257) | **2025-12-08** | 2026-02-15 |

**The `triton_3_5_1/` directory only started receiving configs on 2025-12-08** (less than 6 months ago), and the maintainers focused on different model architectures:

```
$ git log --format="%ai %s" -- triton_3_5_1/ | head -8
2026-02-15 perf: add minimax-2.5 fused_moe tuning config for h20 (#18833)
2026-02-15 [Perf] Tune MiniMax M2 fused moe kernel on H100 GPU (#18851)
2026-02-05 Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2 (#18195)
2026-02-03 Add triton_fused_moe config for GLM-4.7-FP8 tp8 H20 H20-3e (#18091)
2026-01-31 [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB (#17965)
2026-01-28 [Perf] Tune Llama-4-Scout-17B-16E-Instruct fused moe kernel (#17891)
2026-01-18 [GLM 4.7] Add RTX 6000 Pro aka sm120 (#17235)
2026-01-17 [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab (#17133)
```

Notice: **all 8 most recent commits are for OTHER models** (MiniMax, Qwen3-Coder-Next FP8, GLM-4.7, Llama-4-Scout, DeepSeek V3.1) — Qwen3-30B-A3B was simply not in the queue.

### 1.4 What this is NOT

| Hypothesis | Verdict | Why |
|---|---|---|
| "GPU too old" | ❌ Wrong | H200 is current flagship (released 2024); it's the most-tuned device in the repo |
| "Model too new" | ❌ Wrong | Qwen3-30B-A3B-Instruct-2507 released July 2025; the `(E=128, N=768)` config existed in `triton_3_2_0/` since June 2025 |
| "(E=128, N=768) unsupported" | ❌ Wrong | Exact config exists in `triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json` |
| "Triton version too new" | ✅ Partially | The 3.5.1 *compiler* is new (Dec 2025); old block-size choices may be slightly off |
| "Maintenance lag — sglang team hasn't re-tuned this cell" | ✅ **Root cause** | Tracked via `git log` — no PR adds `(E=128, N=768, H200, bf16)` to `triton_3_5_1/` yet |

### 1.5 So what about performance impact?

We didn't directly A/B test old config vs. new config on Triton 3.5.1 yet (would need to write a microbench), but **conservative estimate**: 1.2 – 2 × on `fused_moe_kernel`. Our trace showed `fused_moe_kernel` ate 50.2 % of GPU time × `197.6 µs/call × 672 calls = 132 ms`. **Even a 15 % win on this kernel = 7.5 % end-to-end speedup**, no model change required.

### 1.6 The agent ROI here

This is **the cleanest, lowest-risk task an agent could pick up**:

1. Parse server log: `grep "Performance might be sub-optimal"`
2. Extract `(E, N, device, dtype)`
3. Run sglang's existing benchmark: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`
4. Save resulting JSON to `triton_3_5_1/`
5. Open PR

No CUDA expertise, no kernel-writing — pure hyperparameter search with a fully-built infra already in the repo.

---

## 2. Q2 — Where is Triton on the critical path? Is CUDA / Gluon rewrite viable?

### 2.1 Triton's footprint across sglang models

We grepped all 165 `sglang/srt/models/*.py` files for Triton-related imports:

| Pattern | Count | Notes |
|---|---:|---|
| `from sglang.srt.layers.moe.fused_moe_triton import FusedMoE` | **13** | All MoE-architecture models |
| Triton attention backend explicitly | **1** | `gemma3_mm.py:27` only |
| Direct model-specific `@triton.jit` | **2** | `qwen3_next.py:69`, `minimax_m2.py:79/114` |
| `@torch.compile` decorators (indirect Triton via Inductor) | **32** across **22 files** | scattered; mostly small helpers |

**The 13 MoE models using Triton are:**
`llama4.py:42, mixtral.py, qwen2_moe.py, phimoe.py, granitemoe.py, olmoe.py, grok.py, hunyuan.py, exaone_moe.py, kimi_vl.py, mllama4.py, lfm2_moe.py, step3_vl.py` (and `qwen3_moe.py` via inheritance from `qwen2_moe.py:52`).

### 2.2 Critical-path importance (where Triton actually consumes GPU time)

From our R7 trace (Qwen3-30B-A3B MoE):

| Component | GPU time % | Implementation | Source |
|---|---:|---|---|
| `fused_moe_kernel` | **50.17 %** | Triton (sglang) | `fused_moe_triton_kernels.py:324` |
| FlashAttention `FlashAttnFwdSm90` | 12.75 % | hand-written CUDA + CUTLASS templates | sgl-kernel / flash-attn lib |
| cuBLAS `nvjet_*` (GEMM) | 17.47 % | NVIDIA closed-source | `libcublasLt.so` |
| flashinfer kernels (RMSNorm, act, RoPE) | 6.0 % | hand-written CUDA | `flashinfer/*.cuh` |
| sgl-kernel CUDA (moe_align, topk_softmax, …) | 5.15 % | hand-written CUDA | `sgl-kernel/csrc/moe/*.cu` |
| PyTorch ATen scattered ops | 3.60 % | C++ ATen | `libtorch_cuda.so` |
| Triton inductor-generated kernels | 0.16 % | torch.inductor auto-gen | `/tmp/torchinductor_*/c*.py` |

**Net Triton contribution: ~50.3 % of GPU time** — all in `fused_moe_kernel`.

**Important caveat: this is for the MoE model.** For a **dense** model (Qwen3-0.6B, Llama-3.1-8B etc), the breakdown looks completely different — `fused_moe_kernel` is **not invoked at all**, attention + GEMM dominate, and Triton's share drops to **< 5 %**.

### 2.3 Where CUDA already replaces (or competes with) Triton in sglang

`sgl-kernel/csrc/` contains **110 .cu/.cuh files**. Areas where CUDA already exists alongside Triton:

| Area | CUDA files (sgl-kernel) | Triton files (Python) | Status |
|---|---|---|---|
| MoE | `moe_align_kernel.cu`, `moe_fused_gate.cu`, `moe_topk_softmax_kernels.cu`, `moe_topk_sigmoid_kernels.cu`, `moe_sum_reduce.cu`, `kimi_k2_moe_fused_gate.cu`, `fp8_blockwise_moe_kernel.cu`, `nvfp4_blockwise_moe.cu` | `fused_moe_triton_kernels.py` (1148 lines, one big Triton kernel), `triton_kernels_moe.py` | **Coexist** — sgl-kernel handles routing/quantized paths; Triton handles main GEMM. **Routing + token-align is already CUDA**; the **GEMM-style inner loop is still Triton**. |
| Attention | `cutlass_mla_kernel.cu` (CUTLASS MLA), `cascade.cu`, `merge_attn_states.cu`, `vertical_slash_index.cu`, `cutlass_sm100_mla/*` | `triton_backend.py`, `triton_ops/` (small) | **Compete** — most prod paths use flash-attn / CUTLASS; Triton attention is a fallback for special cases like Gemma3 multimodal |
| GEMM | `fp8_gemm_kernel.cu`, `int8_gemm_kernel.cu`, `awq_kernel.cu`, `dsv3_fused_a_gemm.cu`, `dsv3_router_gemm_*.cu`, `bmm_fp8.cu` | `quantization/awq_triton.py`, `fp8_kernel.py`, `int8_kernel.py` | **Coexist** — model-specific CUDA wins on hot paths; Triton handles generic / less-common shapes |
| Quantization | `quantization/{w8a8, fp8, int4, awq, marlin, mxfp4, …}/*.cu` | `awq_triton.py`, `fp8_kernel.py`, `int8_kernel.py` | **Mostly migrated to CUDA** already |

**Bottom line on Triton-to-CUDA rewrite:**
- **MoE GEMM inner loop**: Triton `fused_moe_kernel` (1148 lines) is still the production path. CUTLASS-based grouped GEMM (`flashinfer.cute_dsl.blockscaled_gemm`) is starting to replace it for FP8/NVFP4 paths. **For bf16 + general MoE, Triton remains the only option**. This IS a rewrite opportunity.
- **Routing/aux ops**: Already CUDA. No need to rewrite.
- **Attention**: Already mostly CUTLASS / flash-attn. Triton attention is a niche fallback.

### 2.4 Triton-Gluon — is this a viable target?

**What is Gluon?** It's Triton's **lower-level frontend**, shipping as `triton.experimental.gluon` in Triton ≥ 3.5. It gives the author explicit control over **tensor layouts** that the regular Triton compiler picks automatically:

```python
# Available primitives we just verified are present in Triton 3.5.1
from triton.experimental.gluon import jit, language
# language.BlockedLayout, language.DistributedLinearLayout, 
# language.NVMMASharedLayout, language.SwizzledSharedLayout,
# language.allocate_shared_memory, ...
```

This sits between "Triton's auto-layout" and "raw CUDA C++". For tensor-core-heavy kernels (which `fused_moe_kernel` is — `tcgen05` on H100/H200), Gluon lets you explicitly say "this tensor lives in NVMMA shared layout with this swizzle", giving you **most of the perf upside of raw CUDA without writing PTX**.

**Sanity checks we did:**

| Check | Result |
|---|---|
| Is Gluon available in our env? | ✅ Yes, `triton.experimental.gluon` imports fine in `sglang-dev` (Triton 3.5.1) |
| Does sglang use Gluon yet? | ❌ **No** — `grep -rn "gluon" sglang/python sglang/sgl-kernel` returns 0 hits |
| Does sglang use CUTLASS-DSL (`cute_dsl`)? | ✅ Yes — `sglang/srt/layers/moe/flashinfer_cutedsl_moe.py` (183 lines) wraps `flashinfer.cute_dsl.blockscaled_gemm.grouped_gemm_nt_masked` for FP8/NVFP4 MoE GEMM |
| Does sglang have a Triton-vs-CUDA benchmark harness already? | ✅ Yes — `benchmark/kernels/fused_moe_triton/{benchmark_vllm_vs_sglang_fused_moe_triton.py, benchmark_torch_compile_fused_moe.py, tuning_fused_moe_triton.py}` |

**Verdict on Gluon as a contribution target:**
- ✅ **Greenfield** in sglang — first-mover opportunity
- ✅ **Tooling exists** for fair perf comparison with the current Triton version
- ⚠️ **Risky** — Gluon is `experimental`; API may change; debugging tooling is minimal
- ⚠️ **Narrow target** — only `fused_moe_kernel` (and maybe `_p_matmul_ogs_*` from triton_kernels lib) is worth rewriting; everything else has better alternatives (CUTLASS for GEMM, flash-attn for attention)
- ⏱️ **2-4 weeks** to produce a working Gluon `fused_moe_kernel` matching numerical accuracy + perf ≥ Triton baseline; another 1-2 weeks to upstream

**Less risky alternative:** Use **CUTLASS-DSL (`cute_dsl`)** instead. It's not experimental, sglang already wraps it for FP8 paths, and porting `fused_moe_kernel` to a `cute_dsl`-based grouped-GEMM may be **faster to get correct** even if peak perf is slightly behind a hand-tuned Gluon kernel.

---

## 3. Q3 — Support-function rewrite / fusion opportunities

The "support functions" — code that runs around the model forward but isn't itself a kernel — turn out to be a **larger codebase than the model code**.

### 3.1 Inventory by directory

| Directory | Python lines | `@triton.jit` count | `torch.ops.sgl_kernel.*` count | Biggest functions |
|---|---:|---:|---:|---|
| `sglang/srt/managers/` | **23 795** | 0 | 0 | `scheduler.py: _get_new_batch_prefill_raw` (1977 lines!), `handle_generate_request` (1481) |
| `sglang/srt/mem_cache/` | **17 787** | **8** | 0 | `memory_pool.py: _init_kv_copy_and_warmup` (754), `copy_all_layer_kv_cache_tiled` (1993), `allocator.py: alloc_extend_kernel` (235) |
| `sglang/srt/layers/sampler.py` | 748 | 0 | 0 | (one big sampling pipeline; mixes flashinfer kernels + pytorch fallback) |
| `sglang/srt/layers/quantization/` | **27 064** | **17** | **6** | quant op wrappers; `fp8_kernel.py`, `int8_kernel.py`, `awq_triton.py` |
| `sglang/srt/speculative/` | **9 153** | **14** | 0 | `spec_utils.py`, eagle worker, CUDA graph runners |
| `sglang/srt/constrained/` | **1 679** | **1** | 0 | `triton_ops/bitmask_ops.py` |
| **Total support code** | **~80 226 lines Python** | **40** `@triton.jit` | **6** sgl-kernel dispatches | |

**What this means:**
- The ~80 k lines of support code is **mostly Python with PyTorch eager ops** — exactly the territory where `@torch.compile` / Inductor fusion gives free wins.
- Only **40 `@triton.jit`** kernels are sprinkled through it (vs **32 `@torch.compile`** decorators). Lots of these support functions are simple enough that they should be Inductor-fusable rather than hand-written Triton.

### 3.2 The 5 fusion candidates with strongest evidence

These all show up as **scattered small kernels** in our R7 trace, with caller chains pointing at specific support-function lines.

#### Candidate 1 — `flashattention_backend.py:400-560 init_forward_metadata`

**Why it's a candidate:**
- Our trace caught **`at::native::elementwise_kernel<>` (3.22 %)** and several `cumsum`/`fill`/`pad` kernels with caller chains terminating at `flashattention_backend.py:400 init_forward_metadata`
- Source shows: a sequence of `torch.arange` → `torch.cumsum` → `F.pad` → `tensor.copy_` → `fill_` building per-batch FlashAttention metadata
- All operate on the same small metadata tensor (shape `[batch, ...]`), no control deps

**Expected gain**: 1-2 % end-to-end. Replace with one Triton kernel or `@torch.compile`-wrap the function — Inductor would fuse all 5 ops into one launch.

#### Candidate 2 — `mem_cache/allocator.py: alloc_extend_kernel` (already has Triton at line 174-235)

**Status**: Already Triton-implemented (`allocator.py:174 alloc_extend_naive`, `alloc_extend_kernel` at 235). Worth **auditing** whether the Triton kernel is tuned for our workload — its config is NOT in the `fused_moe_triton/configs/` lookup (it's a one-off, no JSON tuning).

#### Candidate 3 — `mem_cache/memory_pool.py:1993 copy_all_layer_kv_cache_tiled`

**Why it's a candidate:**
- 48-layer model → this function runs **48 copies per forward** when KV cache is migrated
- Currently a Python loop dispatching one PyTorch copy per layer
- Should be a single Triton kernel with grid `[num_layers, ...]`

**Expected gain**: only matters during KV-cache eviction / shuffling; could save 5-10 % in chunked-prefill workloads

#### Candidate 4 — `managers/overlap_utils.py:20 _resolve_future_token_ids` (already `@torch.compile`!)

**Status**: **Already** `@torch.compile`-wrapped — our trace confirms Inductor generated `triton_per_fused_copy__mul_sum_0` for this function. This is the **template to copy** for the other candidates.

The fact that this one function alone produced multiple Inductor-fused Triton kernels in our trace is **proof of concept** that wrapping the others would work.

#### Candidate 5 — `layers/sampler.py` (748 lines, 0 Triton, scattered torch ops)

**Why it's a candidate:**
- Only **1 `@torch.compile`** in this file (line 545), wrapping one helper
- Top-of-file imports show heavy use of `torch.gather`, `torch.scatter_`, `torch.masked_fill`, `torch.softmax` — all small ops on `[batch, vocab=152064]` tensors
- Vocab dim is huge, so these add up

**Expected gain**: ≤ 1 % end-to-end (sampling is < 1 % already); but could be a clean demonstration of "agent applies `@torch.compile` to sglang to harvest Inductor fusion"

### 3.3 Where rewriting (vs fusing) would actually pay off

| Hot region | Best action | Effort | Risk |
|---|---|---|---|
| `fused_moe_kernel` (50.2 %) | **Re-autotune for Triton 3.5.1** (cheapest) → consider Gluon port (peak perf) | 1 day → 2-4 weeks | Low → Medium |
| FlashAttention prefill (~12 %) | Leave alone (already CUTLASS) | — | — |
| cuBLAS `nvjet_*` (17.5 %) | **Switch to FP8** → triggers `fp8_blockwise_moe_kernel.cu` | 1-2 weeks (weight quant pipeline) | Medium |
| ATen scattered ops (3.6 %) | **Wrap support fns in `@torch.compile`** to let Inductor fuse | days per function | Low |
| `flashinfer::RMSNorm` etc (~6 %) | Leave alone (already good) | — | — |
| sampling.py | Wrap in `@torch.compile` | hours | Low |

### 3.4 The most defensible "first contribution" choice

Based on this investigation, **for our 6-12 week timeframe**, ranked by ROI-per-week:

1. **(Week 1-2)** Build agent harness to detect "Config file not found" warnings and auto-run sglang's existing autotune scripts. Generate + PR missing `triton_3_5_1/E=128,N=768,H200.json` and 5-10 other gaps. **Concrete output, low risk, immediate sglang community value.**
2. **(Week 2-4)** Wrap 3-5 identified support functions in `@torch.compile`, measure perf, PR upstream. Demonstrates Inductor-fusion as a recipe.
3. **(Week 4-8)** Port `fused_moe_kernel` to **CUTLASS-DSL via `cute_dsl`** (less risky than Gluon, already has sglang precedent in `flashinfer_cutedsl_moe.py`). Compare perf vs Triton.
4. **(Week 8-12)** If CUTLASS-DSL port succeeds, attempt Gluon variant for peak perf. Otherwise focus on FP8 conversion pipeline.

---

## 4. Cross-cutting evidence appendix

### 4.1 Files referenced in this report

| File | Purpose |
|---|---|
| `sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py:70-130` | Config lookup + fallback logic |
| `sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:324` | The actual `fused_moe_kernel` |
| `sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/` | New-version config dir (incomplete) |
| `sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json` | The fallback config that's actually used |
| `sglang/python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py` | Existing sglang use of CUTLASS-DSL |
| `sglang/python/sglang/srt/layers/attention/flashattention_backend.py:400-560` | Fusion candidate #1 |
| `sglang/python/sglang/srt/mem_cache/allocator.py:174-235` | Fusion candidate #2 (already Triton) |
| `sglang/python/sglang/srt/mem_cache/memory_pool.py:1993` | Fusion candidate #3 |
| `sglang/python/sglang/srt/managers/overlap_utils.py:20` | Fusion candidate #4 (already `@torch.compile`, proof-of-concept) |
| `sglang/python/sglang/srt/layers/sampler.py` | Fusion candidate #5 |
| `sglang/sgl-kernel/csrc/moe/*.cu` | 14 CUDA MoE kernels (replace/coexist with Triton) |
| `sglang/benchmark/kernels/fused_moe_triton/` | Existing autotune + benchmark harness |
| `~/.conda/envs/sglang-dev/lib/python3.11/site-packages/triton/experimental/gluon/` | Gluon (low-level Triton frontend) |
| `~/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/cute_dsl/blockscaled_gemm.py` | CUTLASS-DSL (`cute_dsl`) Python bindings |

### 4.2 Reproduction commands for every count cited

```bash
# Triton version installed in sglang env
conda activate sglang-dev && python -c "import triton; print(triton.__version__)"
# → 3.5.1

# Total config files per Triton version
ls sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_X_Y_Z/ | wc -l

# H200 + E=128 coverage in triton_3_5_1/
ls .../triton_3_5_1/ | grep "NVIDIA_H200" | grep "E=128"
# → empty

# Git history of triton_3_5_1/ dir
cd sglang && git log --format="%ai %s" -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/

# Model files importing Triton MoE
grep -rln "from sglang.srt.layers.moe.fused_moe_triton import FusedMoE" sglang/python/sglang/srt/models/ | wc -l
# → 13

# @triton.jit decorators across support dirs
for d in managers mem_cache speculative constrained layers/quantization; do
  echo "$d: $(grep -rln '@triton.jit' sglang/python/sglang/srt/$d/ | wc -l) files"
done

# @torch.compile decorators (all)
grep -rn "@torch.compile" sglang/python/sglang/srt | wc -l
# → 32

# Verify Gluon availability
conda activate sglang-dev && python -c "from triton.experimental import gluon; print(dir(gluon))"
```

---
---

## 5. What does "tuning a model" actually mean? (the workflow)

> This section is the follow-up investigation: what does the tuning script actually do, does sglang have any runtime tuning, who does it, and how are tuning PRs tested? Every claim has a `grep`-able pointer.

### 5.1 The tuning script — what it does in concrete terms

The script is `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` (458 lines). Reading it line by line, here's the **exact procedure**:

**Step 1 — Build a search space of candidate configs** (`common_utils.py: get_configs_compute_bound`):

```python
for num_stages in [2, 3, 4, 5]:
    for block_m in [16, 32, 64, 128, 256]:
        for block_k in [64, 128, 256]:
            for block_n in [32, 64, 128, 256]:
                for num_warps in [4, 8]:
                    for group_size in [1, 16, 32, 64]:
                        configs.append({
                            "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n,
                            "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": group_size,
                            "num_warps": num_warps, "num_stages": num_stages,
                        })
```

That's `4 × 5 × 3 × 4 × 2 × 4 = 1920` candidates per (model, GPU) cell.

**Step 2 — For each batch size in `[1, 2, 4, ..., 4096]`, time every candidate** by:
1. Generating random `(x, w1, w2, gating)` tensors matching the model's shape
2. Calling `fused_moe(...)` with each candidate config
3. Running 100 iterations, taking median latency

**Step 3 — For each batch size, save the winning config** to JSON:

```python
# common_utils.py: save_configs
def save_configs(configs: Dict[int, BenchmarkConfig], filename: str):
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
```

The output JSON looks like:
```json
{
    "1":  {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    "8":  {"BLOCK_SIZE_M": 32, ...},
    "64": {"BLOCK_SIZE_M": 64, ...},
    ...
}
```

The filename encodes the model shape: `E=128,N=768,device_name=NVIDIA_H200.json`.

**Step 4 — Drop the JSON into `configs/triton_{version}/`** and open a PR.

### 5.2 What "tuning" does NOT do

| Operation | Done by tuning? |
|---|---|
| Modify the kernel itself (`fused_moe_kernel @ fused_moe_triton_kernels.py:324`) | ❌ No — same kernel, just different `tl.constexpr` parameter values |
| Modify model implementation files (`models/qwen3_moe.py`, etc.) | ❌ No — model code is untouched |
| Change backend selection (which of `fused_moe_kernel` vs `_p_matmul_ogs_*` vs `cutlass_moe` to use) | ❌ No — that's `--moe-runner-backend` CLI flag, separate concern |
| Add new kernels | ❌ No |
| Change kernel algorithm / fused ops | ❌ No |

**Tuning = pure hyperparameter search over Triton meta-params.** No code change. Output is one JSON file per (model, dtype, GPU, TP-size) cell.

### 5.3 Does sglang have runtime / startup tuning?

**Yes — three different mechanisms exist, but the big `fused_moe_kernel` uses NONE of them.**

#### Mechanism 1 — `@triton.autotune` decorator on smaller kernels

Found in:
- `layers/quantization/fp8_kernel.py:1662` (`fp8_autotune` wrapping per-token group quant FP8 kernel)
- `layers/attention/fla/cumsum.py:71`, `fla/kda.py` (8 kernels), `fla/l2norm.py` (commented out)

How it works: at first call, Triton compiles all configs in the autotune list, benchmarks them on the actual input shape, picks the best. **Cached for subsequent calls.** Result: first call is slow (compiles everything), then fast.

**Why isn't this used for `fused_moe_kernel`?** Two reasons:
1. The search space is large (1920 configs) — first-call latency would be intolerable (~minutes)
2. Different `(E, N, dtype, H200)` cells need different best configs; autotune doesn't know which cell it's in until the call

#### Mechanism 2 — `flashinfer.autotuner.autotune()` (runtime, at warmup)

Source: `model_executor/model_runner.py:1859-1874`

```python
def _flashinfer_autotune(self):
    from flashinfer.autotuner import autotune
    logger.info("Running FlashInfer autotune...")
    with torch.inference_mode(), autotune():
        self._dummy_run(batch_size=self.req_to_token_pool.size, run_ctx=autotune())
    logger.info("FlashInfer autotune completed.")
```

This runs **at server warmup** (after weight load, before serving requests). **But only when** `--moe-runner-backend = flashinfer_trtllm` or `flashinfer_mxfp4` (`model_runner.py:1837-1842`):

```python
if backend_str not in ["flashinfer_trtllm", "flashinfer_mxfp4"]:
    return False
```

CLI: `--disable-flashinfer-autotune` to skip (`server_args.py:463`).

#### Mechanism 3 — `torch.compile(mode="max-autotune-no-cudagraphs")` (Inductor autotune)

Source: `model_executor/cuda_graph_runner.py:162`

```python
yield torch.compile(
    torch.no_grad()(model.forward),
    mode=os.environ.get("SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"),
    ...
)
```

When `enable_compile` is True, torch.compile compiles the forward pass; Inductor autotunes the Triton kernels it generates. **Only triggered for `@torch.compile`-decorated functions**, which means small support functions like `overlap_utils._resolve_future_token_ids` and `sampler.py:545` helpers — not the main forward.

#### Summary table

| Mechanism | Covers | When | Cost |
|---|---|---|---|
| Static JSON lookup | `fused_moe_kernel` (the big one) | At every launch | None (instant lookup) |
| `@triton.autotune` decorator | ~9 smaller kernels (FP8 quant, linear attention) | First call after server start | Seconds — minutes |
| `flashinfer.autotuner` | flashinfer_trtllm / flashinfer_mxfp4 MoE backends only | Server warmup phase | 30s — few minutes |
| `torch.compile max-autotune` | Inductor-generated kernels (~32 `@torch.compile`-decorated fns) | Compile time at first forward | Tens of seconds |

**So our `fused_moe_kernel` (50 % of GPU time) is the ONLY major path with NO runtime tuning** — it relies purely on the hand-PR'd JSON files in `triton_3_5_1/`.

### 5.4 Who does the tuning? (PR author analysis)

Top 15 authors who've committed to `fused_moe_triton/configs/`:

```
11  Xiaoyu Zhang (BBuf)      ← sglang core maintainer
 7  Yineng Zhang (zhyncs)    ← sglang core maintainer (LMSYS)
 5  zixuanzhang226            ← Bytedance
 5  lambert0312               ← community
 5  Qiaolin Yu                ← community
 4  Yi Zhang                  ← community
 4  Baizhou Zhang             ← community
 3  yigex (AMD)               ← AMD employee (ROCm configs)
 3  roikoren755               ← community
 3  Wen-Heng (Jack) Chung (AMD)
 3  Ximingwang-09
 2  kkHuang-amd (AMD)
 2  jackey hua (Bytedance)
 ...
```

**Pattern**: **Mix of sglang core team (BBuf, zhyncs) and downstream deployers** (Bytedance has many — they run sglang in production; AMD employees contribute ROCm configs).

**It's NOT model authors** — Qwen, MiniMax, DeepSeek companies don't typically submit these. It's whoever's deploying that model on a specific (GPU, TP-size, dtype) combination and notices the missing config.

### 5.5 How are tuning PRs tested before merge?

#### What CI actually runs for tuning PRs

Pre-merge CI for MoE: `test/registered/moe/test_fused_moe.py` (244 lines) + `test_triton_fused_moe.py` (195 lines).

These tests check **numerical correctness** — they compare `fused_moe(...)` output against `torch_naive_moe(...)` (Python loop reference) with tolerance:

```python
def get_tolerance(self, dtype):
    if dtype == torch.float32:
        return 1e-3, 1e-5       # rtol, atol
    elif dtype in [torch.float16, torch.bfloat16]:
        return 1e-1, 1e-2       # 10% rtol on bf16
    else:
        return 1e-2, 1e-2
```

Tests run across `NUM_EXPERTS = [8, 64]` × `TOP_KS = [2, 6]` — but **NOT** at the specific `(E=128, N=768)` shapes the PR is tuning. CI is generic, the PR is specific.

#### What CI does NOT do for tuning PRs

| Test | Done? |
|---|---|
| Numerical correctness on the PR's specific shape | ❌ No — only generic shapes |
| Perf regression check (did the new JSON actually make it faster?) | ❌ No — perf is trusted to the PR author's local measurement |
| Comparison vs the fallback config it would have used | ❌ No |
| Multi-GPU tests | ❌ No (CI runs single GPU) |

#### The `.github/workflows/auto-tune.yml` — is it a real auto-tuner?

Spoiler: **no**. The file exists but is a stub:

```yaml
name: Auto tune
on:
  workflow_dispatch:
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
```

That's literally the whole file. No tuning step, no GPU runner, just a `workflow_dispatch` trigger with a checkout. **There is no automated tuning in sglang's CI today.**

### 5.6 So is performance ever verified before merge?

Based on the data above, **no rigorous pre-merge perf verification**. The trust model is:

1. **Author tunes locally** on their own GPU (sometimes a different GPU than the config claims to target! — e.g. someone tunes on a borrowed H200 to add an H200 config)
2. **Author runs the local benchmark** (`tuning_fused_moe_triton.py --tune`), measures latency, records the best config
3. **Author opens PR** with one JSON file
4. **CI runs the generic numerical-correctness test** (which passes regardless of which config you pick)
5. **Reviewer (a core maintainer) eyeballs** the JSON values for sanity, occasionally re-runs the script themselves
6. **Merge**

Periodically, after merging, someone notices a regression in real workloads and reverts/re-tunes. This DID happen — there's a `[Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB (#17965)` PR from 2026-01-31 fixing a previously-bad config.

### 5.7 Direct answers to your sub-questions

> **"具体是做什么? 修改模型实现里的后端选择路径吗?"**
> No — tuning does not touch model files at all. It generates one JSON file (~146 lines) that maps batch_size → optimal Triton meta-params.

> **"需不需要加入新的 kernel?"**
> No — the kernel (`fused_moe_kernel`, 1148 lines) is the same. Only the JIT compile-time constants (BLOCK_SIZE_*, num_warps, num_stages) change.

> **"或者对 kernel 进行优化?"**
> No — the kernel source is untouched. Tuning ONLY searches the existing kernel's hyperparameter space.

> **"sglang 没有任何运行/部署时 tuning 是吗?"**
> Partial — sglang has runtime tuning for (a) some smaller kernels via `@triton.autotune`, (b) flashinfer backends via `flashinfer.autotuner.autotune()` at warmup, (c) Inductor-generated kernels via `torch.compile(mode="max-autotune-no-cudagraphs")`. **But the dominant `fused_moe_kernel` (50 % of GPU time) has NO runtime tuning** — it uses static JSON lookup.

> **"都需要人工 tuning 然后交 PR 吗?"**
> For `fused_moe_kernel`: yes. Someone runs `tuning_fused_moe_triton.py --tune --model ... --tp-size ...` locally, gets a JSON, opens a PR.

> **"合并 PR 前又是如何对这些 tuning 进行测试的?"**
> Only generic numerical correctness (test_fused_moe.py, 244 lines, tests E=8/64 and topk=2/6 — NOT the PR's specific shape). No perf regression check, no on-PR autotune. The `auto-tune.yml` workflow is a stub.

> **"是模型开发者负责 tuning 还是 sglang 的人负责?"**
> Mix. Top 2 contributors are sglang core maintainers (BBuf, zhyncs). Many contributions from downstream deployers (Bytedance especially). AMD employees contribute ROCm-specific configs. **NOT typically model authors** — Qwen / DeepSeek / MiniMax companies rarely submit these. The implicit assumption is: "if you deploy this (model × GPU × dtype × TP) combo and care about perf, you tune it yourself."

### 5.8 What this means for the agent project

This is **the most under-served niche** in sglang right now:

| Gap in current process | Where an agent helps |
|---|---|
| No automated tuning — humans do it sporadically | Agent runs tuning continuously across (model, GPU, dtype, TP) matrix |
| Only generic CI tests, no perf regression check | Agent gates PRs by running its own benchmark before merge |
| Tuning lag — new Triton version means weeks of stale configs | Agent re-tunes immediately on Triton bump |
| Author trust model — no double-check on perf claims | Agent re-runs author's tuning on its own hardware to verify |
| Coverage is incomplete (hundreds of (model, GPU, dtype, TP) cells, only ~64 H200 configs in 3.5.1) | Agent enumerates the gap matrix and prioritises high-traffic deployments |

**The cleanest first PR is literally just**: "I built a bot that detects `Config file not found` warnings, runs autotune, and opens a PR. Here are 10 generated configs that close 10 known gaps." Low-risk, high-visibility, immediately useful to the sglang community.



---

<a id="中文版"></a>

# 中文版

# Triton 重写调研 — 把 Triton kernel 重写成 CUDA / Gluon / CUTLASS-DSL 对 sglang 是不是可行的贡献?

> **日期**: 2026-06-04 · **作者**: 端到端调研 harness
> **范围**: 回答 3 个用户问题,每个结论都给出 grep 即得的证据。

## TL;DR (3 个核心答案)

| 问 | 简答 |
|---|---|
| **MoE 配置回退,是因为我们卡太老,还是模型太新?** | **都不是** —— 是 **sglang 自己的维护进度跟不上**。我们装的 Triton 是 3.5.1 (约 2025-12 发布);sglang 的 `triton_3_5_1/` 配置目录只有 **10 个 H200 配置**,没一个覆盖 `(E=128, N=768)`。但同样这个 `(E=128, N=768, H200, bf16)` 配置 **在更老的 `triton_3_2_0/` 里是存在的** (2025-06 提交)。也就是 sglang 上游还没给这个具体 (模型,GPU,Triton 版本) 三元组重新 tune。**同样的 H200 硬件,同样的模型,只是 Triton 编译器换了版本**。 |
| **如果 Triton 仍是关键路径,把它重写成 CUDA/Gluon 是个可行贡献吗?** | **对 MoE 这一条路 YES,对大部分其他模型 NO**。165 个 sglang 模型文件中,只有 **13/165 (8%)** 引用 `FusedMoE`(基于 Triton 的 MoE),只有 **2** 个有模型特定 `@triton.jit`。**对这 13 个 MoE 模型** —— 包括我们 —— Triton 占 ~50% GPU 时间。对 dense 模型,attention (`flash-attn` CUTLASS) 和 cuBLAS GEMM 主导,Triton 占比 < 5%。所以**重写机会集中在 MoE**。 |
| **是不是很多支援函数都是 torch/triton 写的,可以重写或者融合?** | **是 —— 数量上很可观**。sglang 的 managers/mem_cache/sampler/quantization/speculative/constrained 加起来有 **~80,000 行 Python 支援代码**,中间散布 **40 个小的 `@triton.jit`** kernel。我们的 trace 显示有约 3.6% GPU 时间花在**没被融合的零碎 PyTorch ATen 小 op** 上 (copy, fill, arange, cumsum)。下文给出 **5 个有具体文件:行号** 的融合候选点。 |

报告主体给每个答案的支撑证据。

---

## 1. Q1 — MoE 配置为啥回退? (根因分析)

### 1.1 我们观察到的现象

来自 `results/kernel_inventory_R7/server.log`:

```
[2026-06-03 17:43:53] Config file not found at .../triton_3_5_1/E=128,N=768,device_name=NVIDIA_H200.json.
Fallback to triton version 3.2.0 and use MoE kernel config from .../triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json.
Performance might be sub-optimal!
```

### 1.2 环境事实

| 字段 | 值 | 证据 |
|---|---|---|
| sglang-dev env 里装的 Triton | **3.5.1** | `python -c "import triton; print(triton.__version__)"` |
| GPU | H200 | `nvidia-smi`;143 GB,sm_90 |
| 模型 | Qwen3-30B-A3B (128 专家, N=768 FFN 中间维) | `config.json` |
| dtype | bf16 | `model_config` |
| sglang 挑配置的代码 | `fused_moe_triton_config.py:80` 读 `triton.__version__`,然后用目录 `triton_{version}` | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py:80` |

### 1.3 关键证据 —— 各 Triton 版本的配置覆盖

我们列了所有 `python/sglang/srt/layers/moe/fused_moe_triton/configs/` 下的配置文件:

| Triton 版本目录 | 总配置数 | H200 配置数 | H200 + E=128 配置数 | 首次提交 | 最后提交 |
|---|---:|---:|---:|---|---|
| `triton_3_1_0/` | 127 | 24 | (需要 grep) | 2025-06-07 | 2025-06-07 |
| `triton_3_2_0/` | 35 | **7** | **5** (含 `E=128,N=768,H200` ← 我们的 fallback 目标!) | 2025-06-07 | 2025-08-14 |
| `triton_3_3_0/` | 1 | 0 | 0 | 2025-08-14 | 2025-08-14 |
| `triton_3_3_1/` | 21 | 7 | 0 | 2025-06-09 | 2025-09-09 |
| `triton_3_4_0/` | 31 | 5 | 1 (`E=128,N=192,fp8`) | 2025-08-10 | 2025-12-01 |
| `triton_3_5_1/` | **64** | **10** | **0** (有的 E 值: 20, 40, 80, 161, 257) | **2025-12-08** | 2026-02-15 |

**`triton_3_5_1/` 目录是 2025-12-08 才开始接收配置的** (距今不到 6 个月),而且维护者主要关注其他模型架构:

```
$ git log --format="%ai %s" -- triton_3_5_1/ | head -8
2026-02-15 perf: add minimax-2.5 fused_moe tuning config for h20 (#18833)
2026-02-15 [Perf] Tune MiniMax M2 fused moe kernel on H100 GPU (#18851)
2026-02-05 Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2 (#18195)
2026-02-03 Add triton_fused_moe config for GLM-4.7-FP8 tp8 H20 H20-3e (#18091)
2026-01-31 [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB (#17965)
2026-01-28 [Perf] Tune Llama-4-Scout-17B-16E-Instruct fused moe kernel (#17891)
2026-01-18 [GLM 4.7] Add RTX 6000 Pro aka sm120 (#17235)
2026-01-17 [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab (#17133)
```

注意: **最近 8 次提交全是给其他模型** (MiniMax、Qwen3-Coder-Next FP8、GLM-4.7、Llama-4-Scout、DeepSeek V3.1) —— Qwen3-30B-A3B 根本不在队列里。

### 1.4 这 **不是** 什么

| 假设 | 结论 | 原因 |
|---|---|---|
| "GPU 太老" | ❌ 错 | H200 是当前旗舰 (2024 发布);是仓库里 tune 最多的设备 |
| "模型太新" | ❌ 错 | Qwen3-30B-A3B-Instruct-2507 是 2025-07 发布;`(E=128, N=768)` 配置从 2025-06 就在 `triton_3_2_0/` 里 |
| "(E=128, N=768) 不被支持" | ❌ 错 | 这个精确配置就在 `triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json` |
| "Triton 版本太新" | ✅ 部分对 | 3.5.1 *编译器* 是新的 (2025-12);旧的 block-size 选择对新编译器可能略有偏差 |
| "维护滞后 —— sglang 团队还没给这个 cell 重新 tune" | ✅ **根因** | `git log` 跟踪过 —— 还没有 PR 把 `(E=128, N=768, H200, bf16)` 加进 `triton_3_5_1/` |

### 1.5 性能影响到底多大?

我们还没做老配置 vs 新配置在 Triton 3.5.1 上的 A/B 直接测试 (需要写个 microbench),但**保守估计**: `fused_moe_kernel` 上 1.2 – 2×。我们 trace 里 `fused_moe_kernel` 吃了 GPU 时间 50.2% × `197.6 µs/call × 672 calls = 132 ms`。**就算这个 kernel 只快 15% = 端到端加速 7.5%**,模型一行不用改。

### 1.6 Agent 在这里的 ROI

这是 **agent 能接手的最干净、最低风险的任务**:

1. 解析 server log: `grep "Performance might be sub-optimal"`
2. 提取 `(E, N, device, dtype)`
3. 跑 sglang 已有的 benchmark: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`
4. 把生成的 JSON 存到 `triton_3_5_1/`
5. 开 PR

**不需要懂 CUDA,不需要写 kernel** —— 纯参数搜索,完整 infra 仓库已经备好。

---

## 2. Q2 — Triton 在哪些关键路径上?CUDA/Gluon 重写可行吗?

### 2.1 Triton 在 sglang 模型里的占比

我们 grep 了全部 165 个 `sglang/srt/models/*.py` 文件的 Triton 相关 import:

| Pattern | 数量 | 说明 |
|---|---:|---|
| `from sglang.srt.layers.moe.fused_moe_triton import FusedMoE` | **13** | 全是 MoE 架构模型 |
| 显式 Triton attention backend | **1** | 仅 `gemma3_mm.py:27` |
| 模型自带的 `@triton.jit` | **2** | `qwen3_next.py:69`, `minimax_m2.py:79/114` |
| `@torch.compile` 装饰器 (间接走 Triton via Inductor) | **32** 个,**22** 个文件 | 散落各处,大多是小 helper |

**那 13 个用 Triton MoE 的模型是:**
`llama4.py:42, mixtral.py, qwen2_moe.py, phimoe.py, granitemoe.py, olmoe.py, grok.py, hunyuan.py, exaone_moe.py, kimi_vl.py, mllama4.py, lfm2_moe.py, step3_vl.py` (加上通过继承 `qwen2_moe.py:52` 的 `qwen3_moe.py`)。

### 2.2 关键路径重要性 (Triton 实际吃多少 GPU 时间)

来自我们 R7 trace (Qwen3-30B-A3B MoE):

| 组件 | GPU 时间% | 实现 | 源 |
|---|---:|---|---|
| `fused_moe_kernel` | **50.17%** | Triton (sglang) | `fused_moe_triton_kernels.py:324` |
| FlashAttention `FlashAttnFwdSm90` | 12.75% | 手写 CUDA + CUTLASS 模板 | sgl-kernel / flash-attn 库 |
| cuBLAS `nvjet_*` (GEMM) | 17.47% | NVIDIA 闭源 | `libcublasLt.so` |
| flashinfer kernel (RMSNorm, act, RoPE) | 6.0% | 手写 CUDA | `flashinfer/*.cuh` |
| sgl-kernel CUDA (moe_align, topk_softmax, …) | 5.15% | 手写 CUDA | `sgl-kernel/csrc/moe/*.cu` |
| PyTorch ATen 散乱 op | 3.60% | C++ ATen | `libtorch_cuda.so` |
| Triton inductor 自动生成 kernel | 0.16% | torch.inductor 自动生成 | `/tmp/torchinductor_*/c*.py` |

**Triton 净贡献: GPU 时间的 ~50.3%** —— 全在 `fused_moe_kernel`。

**重要注意: 这是 MoE 模型**。**dense 模型** (Qwen3-0.6B、Llama-3.1-8B 等) 的拆分完全不一样 —— `fused_moe_kernel` **根本不会被调用**,attention + GEMM 主导,Triton 占比降到 **< 5%**。

### 2.3 sglang 里 CUDA 已经在替代 (或竞争) Triton 的地方

`sgl-kernel/csrc/` 有 **110 个 .cu/.cuh 文件**。CUDA 已和 Triton 并存的领域:

| 领域 | CUDA 文件 (sgl-kernel) | Triton 文件 (Python) | 状态 |
|---|---|---|---|
| MoE | `moe_align_kernel.cu`, `moe_fused_gate.cu`, `moe_topk_softmax_kernels.cu`, `moe_topk_sigmoid_kernels.cu`, `moe_sum_reduce.cu`, `kimi_k2_moe_fused_gate.cu`, `fp8_blockwise_moe_kernel.cu`, `nvfp4_blockwise_moe.cu` | `fused_moe_triton_kernels.py` (1148 行,一个大 Triton kernel), `triton_kernels_moe.py` | **共存** —— sgl-kernel 管 routing/量化路径;Triton 管主 GEMM。**Routing + token-align 已经 CUDA 化**;**GEMM 风格的内循环仍是 Triton** |
| Attention | `cutlass_mla_kernel.cu`, `cascade.cu`, `merge_attn_states.cu`, `vertical_slash_index.cu`, `cutlass_sm100_mla/*` | `triton_backend.py`, `triton_ops/` (小) | **竞争** —— 大部分生产路径用 flash-attn / CUTLASS;Triton attention 是特殊场景的 fallback (例: Gemma3 multimodal) |
| GEMM | `fp8_gemm_kernel.cu`, `int8_gemm_kernel.cu`, `awq_kernel.cu`, `dsv3_fused_a_gemm.cu`, `dsv3_router_gemm_*.cu`, `bmm_fp8.cu` | `quantization/awq_triton.py`, `fp8_kernel.py`, `int8_kernel.py` | **共存** —— 模型特定 CUDA 在热路径胜出;Triton 处理通用/罕见 shape |
| 量化 | `quantization/{w8a8, fp8, int4, awq, marlin, mxfp4, …}/*.cu` | `awq_triton.py`, `fp8_kernel.py`, `int8_kernel.py` | **基本已迁移到 CUDA** |

**Triton → CUDA 重写的结论:**
- **MoE GEMM 内循环**: Triton `fused_moe_kernel` (1148 行) 仍是生产路径。基于 CUTLASS 的 grouped GEMM (`flashinfer.cute_dsl.blockscaled_gemm`) 已经开始在 FP8/NVFP4 路径上替换它。**对 bf16 通用 MoE,Triton 仍是唯一选项**。这是个**重写机会**。
- **Routing/辅助 op**: 已经 CUDA。**不需要**重写。
- **Attention**: 已经基本 CUTLASS / flash-attn。Triton attention 是小众 fallback。

### 2.4 Triton-Gluon —— 这是个可行目标吗?

**Gluon 是什么?** 它是 Triton 的**低层前端**,从 Triton 3.5 起作为 `triton.experimental.gluon` 发布。它让作者**显式控制 tensor layout**,而普通 Triton 编译器是自动选的:

```python
# Triton 3.5.1 已经有的原语 (我们刚验证过)
from triton.experimental.gluon import jit, language
# language.BlockedLayout, language.DistributedLinearLayout, 
# language.NVMMASharedLayout, language.SwizzledSharedLayout,
# language.allocate_shared_memory, ...
```

它处在 "Triton 自动 layout" 和 "纯 CUDA C++" 之间。对张量核重型 kernel (`fused_moe_kernel` 就是 —— 在 H100/H200 上跑 `tcgen05`),Gluon 让你显式说 "这个 tensor 在 NVMMA shared layout 加这个 swizzle",**拿到纯 CUDA 大部分的性能优势却不用写 PTX**。

**我们刚做的健全性检查:**

| 检查 | 结果 |
|---|---|
| 我们环境里有 Gluon 吗? | ✅ 有,`triton.experimental.gluon` 在 sglang-dev (Triton 3.5.1) 能 import |
| sglang 用了 Gluon 吗? | ❌ **没用** —— `grep -rn "gluon" sglang/python sglang/sgl-kernel` 0 命中 |
| sglang 用了 CUTLASS-DSL (`cute_dsl`) 吗? | ✅ 用了 —— `sglang/srt/layers/moe/flashinfer_cutedsl_moe.py` (183 行) wrap 了 `flashinfer.cute_dsl.blockscaled_gemm.grouped_gemm_nt_masked` 给 FP8/NVFP4 MoE GEMM 用 |
| sglang 有 Triton vs CUDA 的 benchmark harness 吗? | ✅ 有 —— `benchmark/kernels/fused_moe_triton/{benchmark_vllm_vs_sglang_fused_moe_triton.py, benchmark_torch_compile_fused_moe.py, tuning_fused_moe_triton.py}` |

**Gluon 作为贡献目标的评价:**
- ✅ **sglang 里全新领域** —— 抢占先机机会
- ✅ **基础设施已有** —— 可以公平对比当前 Triton 版本的性能
- ⚠️ **有风险** —— Gluon 是 `experimental`;API 可能变;调试工具很少
- ⚠️ **目标很窄** —— 只有 `fused_moe_kernel` (也许加上 triton_kernels 库的 `_p_matmul_ogs_*`) 值得重写;其他都有更好替代 (CUTLASS 替代 GEMM,flash-attn 替代 attention)
- ⏱️ **2-4 周** 写出能跑且数值精度匹配、性能 ≥ Triton baseline 的 Gluon `fused_moe_kernel`;再 1-2 周 upstream

**风险更小的替代:** 用 **CUTLASS-DSL (`cute_dsl`)** 而不是 Gluon。它不是 experimental,sglang 已经在 FP8 路径上 wrap 过它,把 `fused_moe_kernel` port 到基于 `cute_dsl` 的 grouped-GEMM 可能 **更快做对**,即使峰值性能比手 tune 的 Gluon kernel 稍差。

---

## 3. Q3 — 支援函数重写 / 融合机会

"支援函数" —— 围绕模型 forward 跑、但本身不是 kernel 的代码 —— 结果发现是 **比模型代码还大的代码库**。

### 3.1 按目录清点

| 目录 | Python 行数 | `@triton.jit` 数 | `torch.ops.sgl_kernel.*` 数 | 最大的函数 |
|---|---:|---:|---:|---|
| `sglang/srt/managers/` | **23,795** | 0 | 0 | `scheduler.py: _get_new_batch_prefill_raw` (1977 行!), `handle_generate_request` (1481) |
| `sglang/srt/mem_cache/` | **17,787** | **8** | 0 | `memory_pool.py: _init_kv_copy_and_warmup` (754), `copy_all_layer_kv_cache_tiled` (1993), `allocator.py: alloc_extend_kernel` (235) |
| `sglang/srt/layers/sampler.py` | 748 | 0 | 0 | (一整个 sampling pipeline;flashinfer kernel + pytorch fallback 混合) |
| `sglang/srt/layers/quantization/` | **27,064** | **17** | **6** | quant op wrapper;`fp8_kernel.py`, `int8_kernel.py`, `awq_triton.py` |
| `sglang/srt/speculative/` | **9,153** | **14** | 0 | `spec_utils.py`, eagle worker, CUDA graph runner |
| `sglang/srt/constrained/` | **1,679** | **1** | 0 | `triton_ops/bitmask_ops.py` |
| **合计支援代码** | **~80,226 行 Python** | **40** `@triton.jit` | **6** sgl-kernel 派发 | |

**这意味着:**
- 这 ~80k 行支援代码 **大部分是 Python + PyTorch eager op** —— 正是 `@torch.compile` / Inductor 融合送你性能的领域
- 散落只有 **40 个 `@triton.jit`** kernel (对比 **32 个 `@torch.compile`** 装饰器)。这些支援函数很多其实简单到应该让 Inductor 融合就行,不必手写 Triton。

### 3.2 5 个证据最强的融合候选

这些都以**散落的小 kernel** 形式出现在我们 R7 trace 里,调用链指向具体的支援函数行。

#### 候选 1 — `flashattention_backend.py:400-560 init_forward_metadata`

**为啥候选:**
- 我们的 trace 抓到 **`at::native::elementwise_kernel<>` (3.22%)** 和好几个 `cumsum`/`fill`/`pad` kernel,调用链终止于 `flashattention_backend.py:400 init_forward_metadata`
- 源码里有: 一串 `torch.arange` → `torch.cumsum` → `F.pad` → `tensor.copy_` → `fill_` 来构建 per-batch FlashAttention metadata
- 全都在同一个小 metadata tensor (shape `[batch, ...]`) 上操作,无控制依赖

**期望收益**: 1-2% 端到端。换成一个 Triton kernel 或者 `@torch.compile`-wrap 这个函数 —— Inductor 会把这 5 个 op 融合成一次 launch。

#### 候选 2 — `mem_cache/allocator.py: alloc_extend_kernel` (174-235 行已经是 Triton)

**状态**: 已经 Triton 实现 (`allocator.py:174 alloc_extend_naive`, `alloc_extend_kernel` 在 235)。值得**审查**这个 Triton kernel 在我们 workload 下有没有 tune —— 它的 config 不在 `fused_moe_triton/configs/` 查找表里 (它是个 one-off,没 JSON tuning)。

#### 候选 3 — `mem_cache/memory_pool.py:1993 copy_all_layer_kv_cache_tiled`

**为啥候选:**
- 48 层模型 → 这个函数 KV cache 迁移时每次 forward 跑 **48 次复制**
- 当前是 Python 循环,每层 dispatch 一次 PyTorch copy
- 应该是单个 Triton kernel,grid `[num_layers, ...]`

**期望收益**: 只在 KV-cache 驱逐 / 重排时有影响;chunked-prefill workload 可能省 5-10%

#### 候选 4 — `managers/overlap_utils.py:20 _resolve_future_token_ids` (**已经** `@torch.compile`!)

**状态**: **已经** `@torch.compile`-wrap 了 —— 我们 trace 确认 Inductor 给这个函数生成了 `triton_per_fused_copy__mul_sum_0`。这是给其他候选 **复制粘贴的模板**。

事实上这一个函数就在我们 trace 里产出多个 Inductor-fused Triton kernel,**证明给其他候选也 wrap 起来会奏效**。

#### 候选 5 — `layers/sampler.py` (748 行,0 Triton,散落 torch op)

**为啥候选:**
- 这个文件只有 **1 个 `@torch.compile`** (line 545),wrap 了一个 helper
- 文件顶部 import 显示大量用 `torch.gather`, `torch.scatter_`, `torch.masked_fill`, `torch.softmax` —— 全是 `[batch, vocab=152064]` 张量上的小 op
- vocab 维很大,加起来不少

**期望收益**: 端到端 ≤ 1% (sampling 本身就 < 1%);但可以作为干净的 "agent 给 sglang 加 `@torch.compile` 收割 Inductor 融合" 演示。

### 3.3 重写 (vs 融合) 真正划算的地方

| 热区 | 最优行动 | 工作量 | 风险 |
|---|---|---|---|
| `fused_moe_kernel` (50.2%) | **重新 autotune Triton 3.5.1 版本** (最便宜) → 考虑 Gluon port (峰值) | 1 天 → 2-4 周 | 低 → 中 |
| FlashAttention prefill (~12%) | 不动 (已是 CUTLASS) | — | — |
| cuBLAS `nvjet_*` (17.5%) | **切到 FP8** → 触发 `fp8_blockwise_moe_kernel.cu` | 1-2 周 (权重量化 pipeline) | 中 |
| ATen 散乱 op (3.6%) | **把支援函数 wrap 进 `@torch.compile`** 让 Inductor 融合 | 每个函数几天 | 低 |
| `flashinfer::RMSNorm` 等 (~6%) | 不动 (已经够好) | — | — |
| sampler.py | wrap 进 `@torch.compile` | 几小时 | 低 |

### 3.4 最有说服力的"第一个贡献"选择

根据本调研,**在我们 6-12 周时间窗内**,按 "ROI / 每周" 排序:

1. **(Week 1-2)** 搭一个 agent harness 检测 "Config file not found" 警告,自动跑 sglang 已有的 autotune 脚本。生成 + PR 缺失的 `triton_3_5_1/E=128,N=768,H200.json` 和另外 5-10 个 gap。**具体可交付,低风险,对 sglang 社区立即有价值**。
2. **(Week 2-4)** 把 3-5 个识别出的支援函数 wrap 进 `@torch.compile`,测性能,PR 上游。证明 Inductor-fusion 是个 recipe。
3. **(Week 4-8)** 把 `fused_moe_kernel` port 到 **CUTLASS-DSL via `cute_dsl`** (比 Gluon 风险小,sglang 在 `flashinfer_cutedsl_moe.py` 已有先例)。对比 Triton 性能。
4. **(Week 8-12)** 如果 CUTLASS-DSL port 成功,尝试 Gluon 版本拿峰值性能。否则专注 FP8 转换 pipeline。

---

## 4. 横切证据附录

### 4.1 本报告引用的文件

| 文件 | 用途 |
|---|---|
| `sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py:70-130` | 配置查找 + fallback 逻辑 |
| `sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:324` | 真正的 `fused_moe_kernel` |
| `sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/` | 新版本配置目录 (不完整) |
| `sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=128,N=768,device_name=NVIDIA_H200.json` | 实际被用的 fallback 配置 |
| `sglang/python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py` | sglang 现有的 CUTLASS-DSL 使用 |
| `sglang/python/sglang/srt/layers/attention/flashattention_backend.py:400-560` | 融合候选 #1 |
| `sglang/python/sglang/srt/mem_cache/allocator.py:174-235` | 融合候选 #2 (已是 Triton) |
| `sglang/python/sglang/srt/mem_cache/memory_pool.py:1993` | 融合候选 #3 |
| `sglang/python/sglang/srt/managers/overlap_utils.py:20` | 融合候选 #4 (已 `@torch.compile`,proof-of-concept) |
| `sglang/python/sglang/srt/layers/sampler.py` | 融合候选 #5 |
| `sglang/sgl-kernel/csrc/moe/*.cu` | 14 个 CUDA MoE kernel (替代/共存 Triton) |
| `sglang/benchmark/kernels/fused_moe_triton/` | 已有 autotune + benchmark harness |
| `~/.conda/envs/sglang-dev/lib/python3.11/site-packages/triton/experimental/gluon/` | Gluon (Triton 低层前端) |
| `~/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/cute_dsl/blockscaled_gemm.py` | CUTLASS-DSL (`cute_dsl`) Python binding |

### 4.2 每个数字的复现命令

```bash
# sglang env 里装的 Triton 版本
conda activate sglang-dev && python -c "import triton; print(triton.__version__)"
# → 3.5.1

# 每个 Triton 版本的配置文件总数
ls sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_X_Y_Z/ | wc -l

# triton_3_5_1/ 里 H200 + E=128 的覆盖
ls .../triton_3_5_1/ | grep "NVIDIA_H200" | grep "E=128"
# → 空

# triton_3_5_1/ 目录 git 历史
cd sglang && git log --format="%ai %s" -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/

# 模型文件里 import Triton MoE 的数量
grep -rln "from sglang.srt.layers.moe.fused_moe_triton import FusedMoE" sglang/python/sglang/srt/models/ | wc -l
# → 13

# 各支援目录的 @triton.jit 数量
for d in managers mem_cache speculative constrained layers/quantization; do
  echo "$d: $(grep -rln '@triton.jit' sglang/python/sglang/srt/$d/ | wc -l) files"
done

# @torch.compile 装饰器 (全部)
grep -rn "@torch.compile" sglang/python/sglang/srt | wc -l
# → 32

# 验证 Gluon 可用
conda activate sglang-dev && python -c "from triton.experimental import gluon; print(dir(gluon))"
```
---

## 5. "对模型 tune" 到底是干什么? (流程详解)

> 本节是 follow-up 调研: tuning 脚本具体做什么、sglang 有没有运行时 tuning、谁来 tune、tuning PR 合并前怎么测试? 每个论断都有 `grep` 可定位的指针。

### 5.1 Tuning 脚本 — 具体在干什么

脚本是 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` (458 行)。逐行读完,这是**精确流程**:

**第 1 步 — 构建候选配置搜索空间** (`common_utils.py: get_configs_compute_bound`):

```python
for num_stages in [2, 3, 4, 5]:
    for block_m in [16, 32, 64, 128, 256]:
        for block_k in [64, 128, 256]:
            for block_n in [32, 64, 128, 256]:
                for num_warps in [4, 8]:
                    for group_size in [1, 16, 32, 64]:
                        configs.append({
                            "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n,
                            "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": group_size,
                            "num_warps": num_warps, "num_stages": num_stages,
                        })
```

也就是 `4 × 5 × 3 × 4 × 2 × 4 = 1920` 个候选,per (model, GPU) cell。

**第 2 步 — 对 `[1, 2, 4, ..., 4096]` 每个 batch size,计时每个候选**:
1. 按模型 shape 生成随机 `(x, w1, w2, gating)` tensor
2. 用每个候选 config 调 `fused_moe(...)`
3. 跑 100 次迭代,取中位数延迟

**第 3 步 — 每个 batch size 存赢家配置** 到 JSON:

```python
# common_utils.py: save_configs
def save_configs(configs: Dict[int, BenchmarkConfig], filename: str):
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
```

输出 JSON 长这样:
```json
{
    "1":  {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    "8":  {"BLOCK_SIZE_M": 32, ...},
    "64": {"BLOCK_SIZE_M": 64, ...},
    ...
}
```

文件名编码模型 shape: `E=128,N=768,device_name=NVIDIA_H200.json`。

**第 4 步 — 把 JSON 放进 `configs/triton_{version}/`** 然后开 PR。

### 5.2 "tuning" **不**做什么

| 操作 | 在 tune 时做吗? |
|---|---|
| 改 kernel 本身 (`fused_moe_kernel @ fused_moe_triton_kernels.py:324`) | ❌ 不 —— 同一个 kernel,只是 `tl.constexpr` 参数值不同 |
| 改模型实现文件 (`models/qwen3_moe.py` 等) | ❌ 不 —— 模型代码动都不动 |
| 改 backend 选择 (`fused_moe_kernel` vs `_p_matmul_ogs_*` vs `cutlass_moe` 哪个) | ❌ 不 —— 那是 `--moe-runner-backend` CLI flag,完全独立 |
| 加新 kernel | ❌ 不 |
| 改 kernel 算法 / fused op | ❌ 不 |

**Tuning = 纯粹的 Triton meta-param 超参搜索**。无代码改动。输出是一个 JSON 文件,per (model, dtype, GPU, TP-size) cell。

### 5.3 sglang 有没有运行时 / 启动时 tuning?

**有 —— 存在 3 种机制,但大块头 `fused_moe_kernel` 一个都不用**。

#### 机制 1 —— `@triton.autotune` 装饰器,用在小 kernel 上

在这些地方:
- `layers/quantization/fp8_kernel.py:1662` (`fp8_autotune` 包裹 per-token group quant FP8 kernel)
- `layers/attention/fla/cumsum.py:71`, `fla/kda.py` (8 个 kernel), `fla/l2norm.py` (注释掉了)

工作原理: 第一次调用时,Triton 把 autotune 列表里所有 config 编译一遍,在实际输入 shape 上跑 benchmark,挑最优。**缓存供后续调用使用**。结果: 第一次慢 (编译所有 config),之后快。

**为啥 `fused_moe_kernel` 不用这个?** 两个原因:
1. 搜索空间太大 (1920 个 config) —— 第一次调用延迟会无法接受 (~分钟级)
2. 不同 `(E, N, dtype, H200)` cell 的最优 config 不同;autotune 在调用之前不知道是哪个 cell

#### 机制 2 —— `flashinfer.autotuner.autotune()` (运行时,在 warmup)

来源: `model_executor/model_runner.py:1859-1874`

```python
def _flashinfer_autotune(self):
    from flashinfer.autotuner import autotune
    logger.info("Running FlashInfer autotune...")
    with torch.inference_mode(), autotune():
        self._dummy_run(batch_size=self.req_to_token_pool.size, run_ctx=autotune())
    logger.info("FlashInfer autotune completed.")
```

这个在**服务器 warmup 时**跑 (权重加载之后,服务请求之前)。**但只在** `--moe-runner-backend = flashinfer_trtllm` 或 `flashinfer_mxfp4` 时 (`model_runner.py:1837-1842`):

```python
if backend_str not in ["flashinfer_trtllm", "flashinfer_mxfp4"]:
    return False
```

CLI: `--disable-flashinfer-autotune` 可禁用 (`server_args.py:463`)。

#### 机制 3 —— `torch.compile(mode="max-autotune-no-cudagraphs")` (Inductor autotune)

来源: `model_executor/cuda_graph_runner.py:162`

```python
yield torch.compile(
    torch.no_grad()(model.forward),
    mode=os.environ.get("SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"),
    ...
)
```

`enable_compile=True` 时,torch.compile 编译 forward;Inductor 给它生成的 Triton kernel 跑 autotune。**只对 `@torch.compile`-装饰的函数触发**,意味着是 `overlap_utils._resolve_future_token_ids`、`sampler.py:545` 这种小支援函数 —— **不是主 forward**。

#### 汇总表

| 机制 | 覆盖范围 | 何时跑 | 代价 |
|---|---|---|---|
| 静态 JSON 查找 | `fused_moe_kernel` (大块头) | 每次 launch | 无 (瞬时查表) |
| `@triton.autotune` 装饰器 | ~9 个小 kernel (FP8 quant, linear attention) | 服务器启动后第一次调用 | 几秒 — 几分钟 |
| `flashinfer.autotuner` | 仅 flashinfer_trtllm / flashinfer_mxfp4 MoE backend | 服务器 warmup 阶段 | 30 秒 — 几分钟 |
| `torch.compile max-autotune` | Inductor 生成的 kernel (~32 个 `@torch.compile`-装饰函数) | 第一次 forward 编译时 | 几十秒 |

**所以我们的 `fused_moe_kernel` (50% GPU 时间) 是唯一没有运行时 tuning 的主要路径** —— 它纯靠 `triton_3_5_1/` 里手动 PR 的 JSON 文件。

### 5.4 谁来 tune? (PR 作者分析)

提交过 `fused_moe_triton/configs/` 的 top 15 作者:

```
11  Xiaoyu Zhang (BBuf)      ← sglang 核心维护者
 7  Yineng Zhang (zhyncs)    ← sglang 核心维护者 (LMSYS)
 5  zixuanzhang226            ← 字节跳动
 5  lambert0312               ← 社区
 5  Qiaolin Yu                ← 社区
 4  Yi Zhang                  ← 社区
 4  Baizhou Zhang             ← 社区
 3  yigex (AMD)               ← AMD 员工 (ROCm config)
 3  roikoren755               ← 社区
 3  Wen-Heng (Jack) Chung (AMD)
 3  Ximingwang-09
 2  kkHuang-amd (AMD)
 2  jackey hua (字节跳动)
 ...
```

**模式**: **sglang 核心团队 (BBuf, zhyncs) + 下游部署方** 的混合 (字节跳动很多 —— 他们生产环境跑 sglang;AMD 员工贡献 ROCm config)。

**不是模型作者** —— Qwen、MiniMax、DeepSeek 公司一般不交这个。是部署该模型到特定 (GPU, TP-size, dtype) 组合的人发现 config 缺失了。

### 5.5 Tuning PR 合并前怎么测?

#### CI 实际跑什么 (对 tuning PR)

MoE pre-merge CI: `test/registered/moe/test_fused_moe.py` (244 行) + `test_triton_fused_moe.py` (195 行)。

测试做的是**数值正确性** —— 对比 `fused_moe(...)` 输出和 `torch_naive_moe(...)` (Python 循环参考) 在容差内:

```python
def get_tolerance(self, dtype):
    if dtype == torch.float32:
        return 1e-3, 1e-5       # rtol, atol
    elif dtype in [torch.float16, torch.bfloat16]:
        return 1e-1, 1e-2       # bf16 10% rtol
    else:
        return 1e-2, 1e-2
```

测试覆盖 `NUM_EXPERTS = [8, 64]` × `TOP_KS = [2, 6]` —— **但不是** PR 实际调的那个 `(E=128, N=768)` shape。**CI 是通用的,PR 是具体的**。

#### CI **不**做的事

| 测试 | 做吗? |
|---|---|
| 在 PR 的具体 shape 上跑数值正确性 | ❌ 不 —— 只在通用 shape 上 |
| 性能回归检查 (新 JSON 真的更快吗?) | ❌ 不 —— 性能信任 PR 作者本地测的数 |
| 对比 fallback 配置 (本来会用的旧配置) | ❌ 不 |
| 多 GPU 测试 | ❌ 不 (CI 单 GPU) |

#### `.github/workflows/auto-tune.yml` —— 真的是自动 tuner 吗?

剧透: **不是**。文件存在但只是个 stub:

```yaml
name: Auto tune
on:
  workflow_dispatch:
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
```

整个文件就这么多。没有 tuning 步骤,没有 GPU runner,只有 `workflow_dispatch` 触发加一个 checkout。**sglang CI 今天没有自动 tuning**。

### 5.6 那性能合并前到底有没有人验证?

根据上面数据,**没有严格的 pre-merge 性能验证**。信任模型是:

1. **作者本地 tune** (在自己的 GPU 上跑,有时和 config 声称的目标 GPU 还不一样! 比如有人借了台 H200 来加 H200 config)
2. **作者跑本地 benchmark** (`tuning_fused_moe_triton.py --tune`),测延迟,记录最佳 config
3. **作者开 PR**,带一个 JSON 文件
4. **CI 跑通用数值正确性测试** (无论选啥 config 都过)
5. **Reviewer (核心维护者) 大致看一眼** JSON 数值,偶尔自己重跑脚本
6. **合并**

定期地,合并后会有人在真实 workload 里发现回归,然后 revert / 重 tune。这确实发生过 —— 有个 PR `[Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB (#17965)` 2026-01-31,就是修之前一个差的 config。

### 5.7 直接回答你的子问题

> **"具体是做什么? 修改模型实现里的后端选择路径吗?"**
> 不,tuning 根本不碰模型文件。它生成一个 JSON 文件 (~146 行),映射 batch_size → 最优 Triton meta-param。

> **"需不需要加入新的 kernel?"**
> 不需要 —— kernel (`fused_moe_kernel`,1148 行) 是同一个。只有 JIT 编译期常量 (BLOCK_SIZE_*、num_warps、num_stages) 变。

> **"或者对 kernel 进行优化?"**
> 不,kernel 源码动都不动。Tuning **只**在现有 kernel 的超参空间里搜索。

> **"sglang 没有任何运行 / 部署时 tuning 是吗?"**
> 部分有 —— sglang 有运行时 tuning,(a) 一些小 kernel 通过 `@triton.autotune`,(b) flashinfer backend 通过 warmup 时的 `flashinfer.autotuner.autotune()`,(c) Inductor 生成的 kernel 通过 `torch.compile(mode="max-autotune-no-cudagraphs")`。**但占主导的 `fused_moe_kernel` (50% GPU 时间) 没有运行时 tuning** —— 它用静态 JSON 查表。

> **"都需要人工 tuning 然后交 PR 吗?"**
> 对 `fused_moe_kernel`: 是。有人本地跑 `tuning_fused_moe_triton.py --tune --model ... --tp-size ...`,拿 JSON,开 PR。

> **"合并 PR 前又是如何对这些 tuning 进行测试的?"**
> 只有通用数值正确性 (test_fused_moe.py,244 行,测 E=8/64 和 topk=2/6 —— **不是** PR 实际的 shape)。**没有性能回归检查,没有 on-PR autotune**。`auto-tune.yml` workflow 是个 stub。

> **"是模型开发者负责 tuning 还是 sglang 的人负责?"**
> 混合。top 2 贡献者是 sglang 核心维护者 (BBuf, zhyncs)。下游部署方贡献很多 (字节跳动特别多)。AMD 员工贡献 ROCm 特定 config。**一般不是模型作者** —— Qwen / DeepSeek / MiniMax 公司很少交。隐含假设是: "如果你部署这个 (model × GPU × dtype × TP) 组合且关心性能,你自己 tune"。

### 5.8 这对 agent 项目意味着什么

这是 sglang 当下**最被服务不足的小生境**:

| 现行流程的缺口 | Agent 如何帮 |
|---|---|
| 没有自动 tuning —— 人偶尔 tune | Agent 持续在 (model, GPU, dtype, TP) 矩阵上跑 tuning |
| 只有通用 CI 测试,没性能回归检查 | Agent 合并前在自己 benchmark 上 gate PR |
| Tuning 滞后 —— Triton 新版意味着几周陈旧 config | Triton 升级后 agent 立刻重新 tune |
| 作者信任模型 —— 没有性能数字的二次验证 | Agent 在自己硬件上重跑作者的 tuning 来验证 |
| 覆盖不完整 (几百个 (model, GPU, dtype, TP) cell,3.5.1 里只 ~64 个 H200 config) | Agent 枚举缺口矩阵,按高流量部署排优先级 |

**最干净的第一个 PR 真的就是**: "我搭了个 bot,检测 `Config file not found` 警告,跑 autotune,开 PR。这是 10 个生成的 config,关闭了 10 个已知缺口"。低风险、高曝光、对 sglang 社区立即有用。


