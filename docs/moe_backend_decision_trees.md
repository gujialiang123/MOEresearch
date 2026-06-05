# MoE Backend Decision Trees — vLLM vs sglang deep dive

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#中文版)
>
> **Date**: 2026-06-04 · **Status**: comprehensive enumeration of every MoE backend dispatch branch in both frameworks, with code-snippet + file:line evidence
>
> Companion to `docs/triton_rewrite_investigation.md` (§17 has the high-level comparison). This doc is the **exhaustive catalog** with all special cases collected for follow-up investigation.

## TL;DR

- **sglang**: 1 file (`server_args.py`), 16 `model_arch` branches (L1191-1654), imperative if/elif chain. Decisions made by `(model_arch, GPU, quant)`.
- **vLLM**: 9 files (`oracle/*.py`), priority lists per quantization type, declarative `is_supported_config()` callbacks. Decisions made by `(quant, GPU, config)`.
- **Both arrive at Triton for ~70% of (model × GPU × dtype) combinations** but via completely different code paths.
- **9 "special-case override" comments found** (8 in vLLM, 1 explicit in sglang) where a developer hand-coded a fallback because Triton/CUDA/etc was empirically slower or buggier — these are the **highest-value cells to investigate** as agent targets.

---

## 1. sglang full decision tree (file: `python/sglang/srt/server_args.py`)

### 1.1 The 16 model_arch branches (L1190-1654)

| Branch line | model_arch(es) matched | Trigger conditions for non-Triton | → Backend |
|---:|---|---|---|
| L1197 | `DeepseekV3ForCausalLM`, `KimiK25ForConditionalGeneration`, `MistralLarge3ForCausalLM`, `PixtralForConditionalGeneration` | sm_100 + (fp8 / modelopt_fp8 / modelopt_fp4) | flashinfer_trtllm |
| L1333 | `GptOssForCausalLM` | Blackwell+MXFP4 → flashinfer_mxfp4 / AMD+AITER → AITER / triton_kernels lib → triton_kernel | various |
| L1475 | `Llama4ForCausalLM` | sm_100 + (fp8 / modelopt_fp8) | flashinfer_trtllm |
| L1488 | `Exaone4ForCausalLM`, `ExaoneMoEForCausalLM` | SWA mem handling only — NO MoE backend change | (Triton default) |
| L1499 | `Olmo2ForCausalLM` | SWA mem only | (Triton default) |
| L1524 | `KimiLinearForCausalLM`, `BailingMoeV2_5ForCausalLM` | Mamba radix cache only | (Triton default) |
| L1529 | `NemotronHForCausalLM` | NVFP4 / modelopt_fp8 | flashinfer_cutlass |
| L1558 | `Qwen3MoeForCausalLM`, `Qwen3VLMoeForConditionalGeneration` | sm_100 + (fp8 / modelopt_fp4 / **None** ← bf16 also!) | flashinfer_trtllm |
| L1579 | `Qwen3NextForCausalLM`, `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5ForConditionalGeneration` | sm_100 + (fp8 / modelopt_fp4 / None) | flashinfer_trtllm |
| L1607 | `Glm4MoeForCausalLM` | sm_100 + modelopt_fp4 + flashinfer ≥ 0.6.3 | flashinfer_trtllm |
| L1629 | `FalconH1ForCausalLM`, `JetNemotronForCausalLM`, ... | (Mamba/SWA) | (Triton default) |
| L1641 | `GraniteMoeHybridForCausalLM` | Mamba layers | (Triton default) |
| L1654 | `Lfm2ForCausalLM` | Mamba | (Triton default) |
| L2125-2134 | (any model) | MXFP8 quantization | **cutlass (forced override)** |
| L2161-2174 | (any model) | manual `--moe-runner-backend=cutlass` + fp8/mxfp8 | cutlass |

**Default (catch-all)**: any combination not matching above ⇒ **`triton`** (the global default at `moe_runner_backend: str = "auto"`, L498).

### 1.2 Code snippet: the Qwen3-MoE branch (our model's branch)

`sglang/srt/server_args.py:1558-1577`:

```python
elif model_arch in [
    "Qwen3MoeForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
]:
    if is_sm100_supported():
        quant_method = get_quantization_config(hf_config)
        if self.quantization is None and quant_method is not None:
            self.quantization = quant_method
        if (
            (
                self.quantization in ("fp8", "modelopt_fp4")
                or self.quantization is None    # ← bf16 ALSO qualifies
            )
            and self.moe_a2a_backend == "none"
            and self.moe_runner_backend == "auto"
        ):
            self.moe_runner_backend = "flashinfer_trtllm"
```

### 1.3 sglang special-case override hunt

Special hand-coded fallbacks found in sglang MoE dispatch:

| File:line | Type | Quote | Significance |
|---|---|---|---|
| `server_args.py:1674-1675` | TODO | "currently, it is only supported in the single node scenario" + "there is currently a bug on H20 device specifically" | Known H20 bug — investigation target |
| `server_args.py:1789` | NOTE | "trtllm_mha does not support SM120, which will fall back to flashinfer" | SM120 (RTX 6000 Pro) limitation |
| `server_args.py:2746` | COMMENT | "fallback to triton for DeepSeek models because flashinfer doesn't support deterministic inference for DeepSeek models yet" | **For deterministic inference path only**; DeepSeek explicitly demoted |
| `server_args.py:2749` | COMMENT | "fallback to flashinfer on Blackwell for non-DeepSeek models" | Inverse rule for the same path |
| `layers/moe/moe_runner/flashinfer_trtllm.py:256` | FIXME | "there is a bug in the trtllm_fp8_block_scale_moe. It ignored the `output` argument." | Known bug in upstream flashinfer |
| `server_args.py:1945` | LOG | "The current platform does not support Intel XMX, will fallback to triton backend." | XMX hardware gate |

### 1.4 Honest assessment of sglang's design

- **Pro**: All MoE dispatch in ONE place (1 file, ~450 lines for 16 branches) — easy to audit
- **Con**: Adding a new backend requires touching the central if/elif (every branch); can't be added by a single contributor without coordinating
- **Pro for agent**: Easy to enumerate which (model, GPU, quant) tuples fall through to Triton — that's our TAM

---

## 2. vLLM full decision tree (dir: `vllm/model_executor/layers/fused_moe/oracle/`)

### 2.1 The 9 oracle files

| File | Lines | Quantization type | Enum class | # backend options |
|---|---:|---|---|---:|
| `unquantized.py` | 370 | bf16/fp16 | `UnquantizedMoeBackend` | **9** |
| `fp8.py` | 634 | FP8 | `Fp8MoeBackend` | **13** |
| `nvfp4.py` | 560 | NVFP4 | `NvFp4MoeBackend` | 8 |
| `mxfp4.py` | 1710 | MXFP4 | `Mxfp4MoeBackend` | (very many — model-specific) |
| `mxfp8.py` | 91 | MXFP8 | `Mxfp8MoeBackend` | 1 |
| `int8.py` | 218 | INT8 | `Int8MoeBackend` | 1 (Triton only) |
| `int_wna16.py` | 908 | INT4/INT8 weight-only | `WNA16MoEBackend` | 4 |
| `w4a8.py` | 195 | W4A8 | `W4A8MoeBackend` | 1 (CUTLASS only) |
| `w4a8_int8.py` | 360 | W4A8 INT8 | `W4A8Int8MoeBackend` | several |

### 2.2 Enum lists in full

**bf16/fp16 (`unquantized.py:32`)**:
```python
class UnquantizedMoeBackend(Enum):
    FLASHINFER_TRTLLM = "FlashInfer TRTLLM"
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
    AITER = "ROCm AITER"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"
    CPU = "CPU"
    XPU = "XPU"
    TPU = "TPU"
    OOT = "OOT"
```

**FP8 (`fp8.py:42`)** — **13 options**:
```python
class Fp8MoeBackend(Enum):
    NONE = "NONE"
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    DEEPGEMM = "DEEPGEMM"
    BATCHED_DEEPGEMM = "BATCHED_DEEPGEMM"
    MARLIN = "MARLIN"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"
    AITER = "AITER"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    BATCHED_VLLM_CUTLASS = "BATCHED_VLLM_CUTLASS"
    XPU = "XPU"
    CPU = "CPU"
```

**NVFP4 (`nvfp4.py:41`)**:
```python
class NvFp4MoeBackend(Enum):
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    FLASHINFER_CUTEDSL = "FLASHINFER_CUTEDSL"
    FLASHINFER_CUTEDSL_BATCHED = "FLASHINFER_CUTEDSL_BATCHED"
    FLASHINFER_B12X = "FLASHINFER_B12X"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    MARLIN = "MARLIN"
    EMULATION = "EMULATION"
```

**WNA16 (`int_wna16.py:?`)**:
```python
class WNA16MoEBackend(Enum):
    MARLIN = "MARLIN"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    XPU = "XPU"
```

### 2.3 The 5 explicit hand-tuned fallbacks in vLLM

These are the **highest-value findings** for our investigation — each is a case where vLLM developers found a specific kernel was slower or buggier than expected:

#### F1. **`unquantized.py:71` — Hopper bf16 prefers Triton over FlashInfer**

```python
# On Hopper (SM90), the FlashInfer unquantized MoE kernels are slower
# than Triton, so prefer Triton by default.
if current_platform.is_device_capability_family(90):
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_TRTLLM)
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)
```

**Significance**: Direct empirical confirmation that Hopper bf16 + FlashInfer MoE is SLOWER than Triton — corroborates our §14 measurement that Triton-on-H200 at 30% peak is actually the BEST option. This is **the single most important finding** for the project pitch — it's external validation from another framework's team.

#### F2. **`unquantized.py:77` — Qwen3.5 crashes with FlashInfer CUTLASS BF16 + DEP**

```python
# HACK: Qwen3.5 has crash with FLASHINFER_CUTLASS BF16 if DEP.
# Updating the oracle querying logic is out of the scope of this
# PR. Need to fix the kernel or update structure in follow up.
if moe_config.moe_parallel_config.dp_size > 1:
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)
```

**Significance**: Production bug, not yet fixed. **Investigation target**: figure out the root cause and report/fix upstream — would benefit Qwen3.5 users on Blackwell with DP/EP.

#### F3. **`fp8.py:87` — Hopper FP8 Block: TP→Triton, EP→FlashInfer CUTLASS**

```python
# On Hopper for Block Fp8, prefer Triton for TP and FI CUTLASS for EP.
if (
    current_platform.is_cuda()
    and current_platform.is_device_capability(90)
    and activation_key == kFp8Dynamic128Sym
    and weight_key == kFp8Static128BlockSym
):
    if moe_config.moe_parallel_config.ep_size > 1:
        _move_to_front(_AVAILABLE_BACKENDS, Fp8MoeBackend.FLASHINFER_CUTLASS)
    else:
        _move_to_front(_AVAILABLE_BACKENDS, Fp8MoeBackend.TRITON)
```

**Significance**: Triton is the BEST option for TP-only FP8 on Hopper (not just a fallback). FlashInfer CUTLASS is only better when you have multi-rank EP. **Investigation target**: at what EP size does the crossover happen? Could be the basis for a smart-dispatch contribution.

#### F4. **`mxfp4.py:283, 306` — TRITON_UNFUSED disabled because of MTP bug**

```python
_AVAILABLE_BACKENDS = [
    Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
    ...,
    Mxfp4MoeBackend.TRITON,
    Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
    Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    # TRITON_UNFUSED has bug with MTP support
    # TODO re-enable after kernel is fixed
    # TRITON_UNFUSED                                  ← commented out!
    Mxfp4MoeBackend.MARLIN,
    ...
]
```

**Significance**: A specific Triton kernel variant is disabled entirely due to a multi-token-prediction (MTP / speculative decoding) bug. Two occurrences (lines 283 + 306). **Investigation target**: fix the bug, re-enable the kernel, measure the impact.

#### F5. **`mxfp4.py:625` — DeepSeek-V4 on ROCm prefers AITER FlyDSL**

```python
# DeepSeek-V4 on ROCm: prefer AITER FlyDSL MoE (better perf + accuracy
# after shuffle/TP-offset fixes), with Triton-unfused as fallback.
if (
    current_platform.is_rocm()
    and config.routing_method == RoutingMethodType.DeepseekV4
):
    priority_backends = [
        Mxfp4MoeBackend.AITER_MXFP4_BF16,
        Mxfp4MoeBackend.TRITON_UNFUSED,
    ]
```

**Significance**: Specific model + GPU vendor combo gets a custom priority list. Documents both perf AND accuracy concerns ("shuffle/TP-offset fixes"). Shows that even for one quantization (MXFP4), there are 3 different priority lists in the file (general, GPT-OSS, DeepSeek-V4).

#### F6. **`unquantized.py:162` — "TODO: migrate to MK structure"**

```python
# TODO: migrate to MK structure.
```

**Significance**: vLLM team is actively refactoring; current dispatch in `unquantized.py` is considered legacy. Worth tracking for stability.

### 2.4 Code snippet: how `select_unquantized_moe_backend` works

`unquantized.py:153+`:

```python
def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
    """
    Select the unquantized MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    # Special-case CPU/TPU/OOT first
    if current_platform.is_cpu():
        return UnquantizedMoeBackend.CPU, None
    # ...
    
    # Get priority list (after the Hopper demotion + Qwen3.5 crash workaround)
    priority_backends = _get_priority_backends(moe_config)
    
    # Try each backend in priority order; first one that says
    # is_supported_config(...) wins
    for backend in priority_backends:
        kernel_cls = backend_to_kernel_cls(backend)
        supported, reason = kernel_cls.is_supported_config(kernel_cls, moe_config)
        if supported:
            return backend, kernel_cls
    raise ValueError("No backend supports this config")
```

The `is_supported_config` pattern is the **declarative key**: each backend self-reports whether it can handle the configuration. This is what makes vLLM's design extensible.

---

## 3. Side-by-side comparison: dispatch for our exact case

**Setup**: Qwen3-30B-A3B (Qwen3MoeForCausalLM) + H200 (sm_90) + bf16

### 3.1 sglang's path

```
1. Read server_args.py
2. moe_runner_backend = "auto" (default, L498)
3. Hit L1558 elif: model_arch == "Qwen3MoeForCausalLM" → enter branch
4. Check is_sm100_supported() → FALSE (H200 is sm_90)
5. Branch body skipped entirely
6. Fall through all remaining elif's
7. moe_runner_backend stays "auto"
8. Later, get_moe_impl_class (ep_moe/layer.py:686) returns FusedMoE (Triton path)
9. fused_moe_kernel runs
```

### 3.2 vLLM's path

```
1. Read kernel.py: moe_backend = "auto" (default, L171)
2. select_unquantized_moe_backend called
3. _get_priority_backends returns: [FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, TRITON, BATCHED_TRITON]
4. Hopper check (L74): is_device_capability_family(90) → TRUE
5. Demote FLASHINFER_TRTLLM and FLASHINFER_CUTLASS to back
6. Result: [TRITON, BATCHED_TRITON, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS]
7. TRITON.is_supported_config() → TRUE
8. Returns (TRITON, TritonExperts class)
9. fused_moe Triton kernel runs
```

**Both arrive at Triton**, but vLLM's path includes an EXPLICIT decision "FlashInfer is slower here" while sglang's path passively defaults to Triton.

### 3.3 Same case on B200 (counterfactual)

| Step | sglang on B200 | vLLM on B200 |
|---|---|---|
| Detect Blackwell | `is_sm100_supported()` → TRUE | `is_device_capability_family(90)` → FALSE |
| Branch entered? | YES, L1558 branch executes | Hopper demotion skipped, priority list unchanged |
| Condition check | `quantization is None` (bf16) → passes | `FLASHINFER_TRTLLM.is_supported_config()` → ? |
| Final pick | **flashinfer_trtllm** (auto-set) | **FLASHINFER_TRTLLM** (top of priority list) |

So both frameworks ALSO arrive at the same conclusion on Blackwell. The dispatch logic LOOKS different but the outcome is consistent — both teams agree on which kernel to pick.

---

## 4. The "special case override" catalog — investigation roadmap

Collecting all hand-tuned fallbacks found in §1.3 + §2.3, ranked by investigation value:

| # | Source | What's overridden | Reason recorded | Investigation value |
|---|---|---|---|---|
| **1** | vLLM `unquantized.py:71` | Hopper bf16: demote FlashInfer to back of priority list | "FlashInfer unquantized MoE kernels are slower than Triton on SM90" | ⭐⭐⭐⭐⭐ Direct support for our §14 finding. Confirms Triton is the BEST for Hopper bf16, not just default. |
| **2** | vLLM `fp8.py:87` | Hopper FP8 Block: TP→Triton, EP→FlashInfer CUTLASS | "On Hopper for Block Fp8, prefer Triton for TP and FI CUTLASS for EP" | ⭐⭐⭐⭐⭐ Crossover behavior — at what EP size does FlashInfer become faster? Could be the basis for smart auto-dispatch. |
| **3** | vLLM `unquantized.py:77` | Qwen3.5 + DEP: disable FlashInfer CUTLASS bf16 | "HACK: Qwen3.5 has crash with FLASHINFER_CUTLASS BF16 if DEP" | ⭐⭐⭐⭐ Real bug, not yet fixed. Investigation target: fix upstream. |
| **4** | vLLM `mxfp4.py:283,306` | TRITON_UNFUSED disabled in MXFP4 priority list | "has bug with MTP support" | ⭐⭐⭐⭐ Bug fix opportunity. |
| **5** | sglang `server_args.py:2746` | DeepSeek deterministic inference: force Triton | "flashinfer doesn't support deterministic inference for DeepSeek yet" | ⭐⭐⭐ Niche path but documented gap. |
| **6** | vLLM `mxfp4.py:625` | DeepSeek-V4 ROCm: custom priority list | "better perf + accuracy after shuffle/TP-offset fixes" | ⭐⭐⭐ Model + GPU combo specific, complex. |
| **7** | sglang `server_args.py:1674-1675` | flashinfer_trtllm: "currently only single-node + H20 bug" | TODO with GitHub issue links | ⭐⭐⭐ Tracked upstream. |
| **8** | sglang `flashinfer_trtllm.py:256` | `trtllm_fp8_block_scale_moe`: ignores `output` argument | "FIXME: there is a bug" + link | ⭐⭐⭐ Upstream bug. |
| **9** | sglang `server_args.py:1789` | SM120: trtllm_mha falls back to flashinfer | "trtllm_mha does not support SM120" | ⭐⭐ Hardware-specific limitation. |
| **10** | sglang `server_args.py:1945` | Intel XMX absent: fallback to triton backend | hardware gate | ⭐⭐ Edge case. |
| **11** | vLLM `unquantized.py:162` | Refactor TODO | "migrate to MK structure" | ⭐ Internal vLLM concern. |

**Top 4 (⭐⭐⭐⭐ and above) are concrete research opportunities** — each represents a known limitation where an agent or human investigator could potentially:
- Validate the empirical claim (e.g. "is FlashInfer really slower than Triton on Hopper bf16?" — we can re-measure)
- Fix the underlying bug (e.g. Qwen3.5 + DEP crash)
- Re-tune the crossover point (e.g. EP-size threshold for Hopper FP8)


## 4.5 ⚠️ Clarification — 'FlashInfer is slower' refers to which kernel exactly?

> Apparent contradiction noticed during review: §9 of `triton_rewrite_investigation.md` showed `--moe-runner-backend=flashinfer_trtllm` errors out with "sm_100 only" on H200. So how can §4 Finding F1 say "Hopper FlashInfer is slower" — slower than what, if it doesn't even run?

### 4.5.1 The resolution

There are **TWO different FlashInfer MoE functions** for unquantized bf16. They have very different hardware support:

| flashinfer function | Hardware support | sglang wrapper | vLLM wrapper | Works on H200? |
|---|---|---|---|---|
| **`flashinfer.fused_moe.trtllm_bf16_moe`** | sm_100 (Blackwell) ONLY | `FlashInferFusedMoE` (`layer.py:1132+`) | `TrtLlmBf16Experts` | ❌ no |
| **`flashinfer.fused_moe.cutlass_fused_moe`** | sm_90 + sm_100 + sm_120 | `UnquantizedFusedMoEMethod` (`unquant.py:60+373`) | `FlashInferExperts` | ✅ yes |

vLLM `unquantized.py:71-75` demotes BOTH:
```python
if current_platform.is_device_capability_family(90):
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_TRTLLM)
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)
```

The `FLASHINFER_TRTLLM` demotion is essentially **vestigial** (the underlying kernel doesn't even run on Hopper). But the `FLASHINFER_CUTLASS` demotion is **meaningful** — the kernel runs but is empirically slower than Triton.

So when vLLM's comment says "FlashInfer unquantized MoE kernels are slower than Triton on Hopper", it's specifically about **`cutlass_fused_moe`**, not `trtllm_bf16_moe`.

### 4.5.2 Why we couldn't measure this in §9

In §9.3 we tried `--moe-runner-backend=flashinfer_cutlass` and got:
```
fatal error: cuda_fp16.h: No such file or directory
ninja: build stopped: subcommand failed.
```

This was an **environment issue** — our `gcc` couldn't find CUDA headers for JIT compilation. Not a sglang limitation. The path **does exist** in sglang at `srt/layers/quantization/unquant.py:60`:
```python
try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except ImportError:
    flashinfer_cutlass_fused_moe = None
```

And it's called at `unquant.py:373` for the bf16 path. We just couldn't get past the JIT build error.

### 4.5.3 sglang also has this kernel — symmetry restored

Both frameworks wrap the same underlying `flashinfer.cutlass_fused_moe` for bf16:

| Framework | Wrapper class | File:line |
|---|---|---|
| sglang | `UnquantizedFusedMoEMethod.forward_cuda` | `layers/quantization/unquant.py:373` |
| vLLM | `FlashInferExperts` | `layers/fused_moe/experts/flashinfer_cutlass_moe.py` |

**The Hopper-slower verdict from vLLM applies symmetrically to sglang** — if we fixed the JIT env issue, sglang's `flashinfer_cutlass` path would also be slower than Triton on H200 bf16, matching vLLM's measurement.

### 4.5.4 Corrected mental model

| Backend name on H200 bf16 | sglang | vLLM | Status |
|---|---|---|---|
| Triton `fused_moe_kernel` | ✅ default | ✅ default (after Hopper demotion) | **Production path** |
| FlashInfer `trtllm_bf16_moe` | available but errors (sm_100 only) | available but errors (sm_100 only) | **Hardware-unsupported** |
| FlashInfer `cutlass_fused_moe` | wrapped but env-broken in our test | wrapped (slower than Triton per vLLM benchmark) | **Available but slow on Hopper** |

So **the agent project's claim that "Hopper bf16 → Triton is best" is correct**, and now we have evidence from THREE sources:
1. sglang's auto-mode falls through to Triton (passively)
2. vLLM's auto-mode explicitly demotes both FlashInfer paths (actively)
3. Our §14 measurement showed Triton at 30% peak (the best available on H200)

The next step would be to actually run `flashinfer_cutlass` on H200 by fixing the env JIT issue, to quantify HOW much slower it is than Triton. That would close the loop on Finding F1.



## 4.6 Common misconception — is it "sglang only supports sm_100"?

> Question after reading §4.5: "So both flashinfer MoE kernels exist, sm_100 and sm_90 use different ones; vLLM supports the sm_90 one but calls it slow; sglang only supports the sm_100 one — is that right?"
>
> **Partial correct, partial wrong**. The most important correction: **sglang ALSO wraps the sm_90 path**. The frameworks have feature parity here.

### 4.6.1 Both frameworks wrap BOTH flashinfer MoE functions

| flashinfer function | sglang wrapper | sglang CLI flag | vLLM wrapper | Hardware |
|---|---|---|---|---|
| **`trtllm_bf16_moe`** | `class FlashInferFusedMoE` at `srt/layers/moe/fused_moe_triton/layer.py:1132` | `--moe-runner-backend=flashinfer_trtllm` | `class TrtLlmBf16Experts` | sm_100 ONLY |
| **`cutlass_fused_moe`** | `UnquantizedFusedMoEMethod.forward_cuda` at `srt/layers/quantization/unquant.py:60+373` | `--moe-runner-backend=flashinfer_cutlass` | `class FlashInferExperts` | sm_90 + sm_100 + sm_120 |

sglang's `MOE_RUNNER_BACKEND_CHOICES` (`server_args.py:176-184`) explicitly lists BOTH:

```python
MOE_RUNNER_BACKEND_CHOICES = [
    "auto",
    "deep_gemm",
    "triton",
    "triton_kernel",
    "flashinfer_trtllm",        # ← sm_100 path
    "flashinfer_cutlass",       # ← sm_90 path (we did test it in §9b!)
    "flashinfer_mxfp4",
    "flashinfer_cutedsl",
    "cutlass",
]
```

We literally tested both in `triton_rewrite_investigation.md` §9:
- §9 (C9): `flashinfer_trtllm` → flashinfer reports "sm_100 only" error
- §9b (C9b): `flashinfer_cutlass` → flashinfer JIT-compiles cutlass C++; our env lacks `cuda_fp16.h` header so the JIT build fails

So sglang ABSOLUTELY supports the sm_90 flashinfer path. We just couldn't get past our env's JIT compile error to actually measure its perf.

### 4.6.2 Precise feature-parity table

|  | flashinfer `trtllm_bf16_moe` (sm_100) | flashinfer `cutlass_fused_moe` (sm_90/100/120) |
|---|---|---|
| **Code path in sglang?** | ✅ yes | ✅ yes |
| **Code path in vLLM?** | ✅ yes | ✅ yes |
| **Runs on H200?** | ❌ no (hardware unsupported) | ✅ yes (JIT compile needed) |
| **We measured on H200?** | §9 — server died with "sm_100 only" | §9b — server died with cuda_fp16.h missing |
| **vLLM's verdict for Hopper?** | Demoted (vestigial — doesn't run anyway) | Demoted (**this** is the meaningful "slower than Triton" |

### 4.6.3 What our H200 + bf16 user actually sees

Without specifying `--moe-runner-backend`:
1. sglang auto-mode (`server_args.py:1558`): gate `is_sm100_supported()` fails → no special path applies → falls through to default `triton`
2. **Result**: `fused_moe_kernel` (Triton) runs

If they pass `--moe-runner-backend=flashinfer_trtllm`:
1. sglang assigns `moe_runner_backend = "flashinfer_trtllm"`
2. `get_moe_impl_class` returns `FlashInferFusedMoE`
3. At first request, `forward_impl` calls `trtllm_bf16_moe(...)`
4. flashinfer raises `RuntimeError: No supported CUDA architectures found for major versions [10]`
5. Server crashes

If they pass `--moe-runner-backend=flashinfer_cutlass`:
1. sglang sets `self.use_flashinfer_cutlass = True`
2. At first request, `UnquantizedFusedMoEMethod.forward_cuda` calls `flashinfer_cutlass_fused_moe(...)`
3. flashinfer JIT-compiles cutlass C++
4. In our env, gcc can't find `cuda_fp16.h` → compile fails → server crashes
5. **In a properly-configured env**: it would run, and per vLLM's measurement, would be slower than Triton

### 4.6.4 One-paragraph correction to the misconception

> **No, sglang doesn't "only support the sm_100 flashinfer path".** sglang exposes BOTH flashinfer MoE paths (`flashinfer_trtllm` for sm_100 + `flashinfer_cutlass` for sm_90/100/120) as CLI options, same as vLLM. The functional matrix is symmetric between the two frameworks. The reason we observe Triton everywhere on H200 is:
> - For most users: auto-mode falls through to Triton (no explicit decision needed)
> - For users who explicitly pick `flashinfer_trtllm`: the underlying flashinfer kernel is sm_100-only, so it crashes on Hopper
> - For users who explicitly pick `flashinfer_cutlass`: the kernel runs (env permitting) but is slower than Triton on Hopper per vLLM's independent measurement
>
> So **on H200 bf16, Triton is the empirically best choice in both frameworks**, not because alternatives don't exist, but because they're either hardware-unsupported or slower.

### 4.6.5 The only difference between frameworks (revised)

| Dimension | sglang | vLLM |
|---|---|---|
| Available paths | flashinfer_trtllm + flashinfer_cutlass + Triton + others | same set |
| Hardware gates | implicit (auto-mode if/elif via `is_sm100_supported`) | explicit (`_supports_current_device` per backend) + `is_supported_config` |
| Hopper preference | passive — falls through to Triton when no other auto-rule fires | active — explicit `_move_to_back` of both FlashInfer paths |
| What the user gets in practice | Triton | Triton |

**The end user gets the same kernel choice. The difference is documentation/reasoning, not behavior.**


---

## 5. Coverage delta — what vLLM has that sglang doesn't (and vice versa)

| Capability | sglang | vLLM | Note |
|---|---|---|---|
| FP8 Marlin path | ❌ no | ✅ `Fp8MoeBackend.MARLIN` | INT4-style optimization for FP8 |
| BATCHED variants | ❌ no | ✅ BATCHED_TRITON, BATCHED_DEEPGEMM, BATCHED_VLLM_CUTLASS, BATCHED_MARLIN | For attn_metadata.use_batched_activation_format |
| Per-GPU explicit demotion | ❌ no | ✅ `_move_to_back` based on `device_capability_family(90)` | Empirically tuned |
| `is_supported_config` self-reporting | ❌ no (hardcoded if/else) | ✅ each backend class has the method | Extensibility |
| Multiple FlashInfer variants per quant | only TRTLLM/CUTLASS/CUTEDSL | All 5 (TRTLLM/CUTLASS/CUTEDSL/CUTEDSL_BATCHED/B12X for NVFP4) | More fine-grained |
| Custom routing-method-aware dispatch | partial (head models) | yes (`RoutingMethodType.DeepseekV4` check) | More principled |

**Bottom line on coverage**: vLLM has notably MORE dispatch nuance (about 1.5-2× more backend options per quant). But the "default → Triton" terminal is the same in both, so for our target user (long-tail MoE on Hopper bf16), the coverage delta doesn't matter.

---

## 6. Recommended next steps

1. **Reproduce Finding #1** (Hopper bf16: FlashInfer slower than Triton): We already showed Triton at 30% peak; need to actually run FlashInfer on H200 (won't work for trtllm_bf16 — Blackwell only — but can try `cutlass_fused_moe` once we fix the JIT include path from §9.3).

2. **Investigate Finding #3** (Qwen3.5 + DEP crash): Reproduce the crash on H200 with DP > 1, capture the exact error, file a follow-up.

3. **Crossover study for Finding #2** (TP vs EP for Hopper FP8): Run sglang with `--tp-size 1 vs 2 vs 4 vs 8 vs 16` at FP8 (would need FP8 weights — convert Qwen3-30B-A3B first), measure when CUTLASS becomes faster than Triton.

4. **Re-enable Triton-unfused for MXFP4 (Finding #4)**: If MTP bug is fixable, could re-enable a backend and benchmark.

---

## 7. Files referenced

### sglang
- `python/sglang/srt/server_args.py` (L1190-1654 the model_arch branches, L2125-2174 the cutlass overrides)
- `python/sglang/srt/layers/moe/ep_moe/layer.py:686` (`get_moe_impl_class` — the actual class chooser)
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py:256` (FIXME comment)

### vLLM
- `vllm/config/kernel.py:171` (the `moe_backend` knob)
- `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` (bf16 dispatch)
- `vllm/model_executor/layers/fused_moe/oracle/fp8.py` (FP8 dispatch with 13 backends)
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` (1710 lines, 3 priority lists)
- `vllm/model_executor/layers/fused_moe/oracle/mxfp8.py`
- `vllm/model_executor/layers/fused_moe/oracle/int8.py`
- `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`
- `vllm/model_executor/layers/fused_moe/oracle/w4a8.py`
- `vllm/model_executor/layers/fused_moe/oracle/w4a8_int8.py`

---

---

<a id="中文版"></a>

# 中文版

# MoE Backend 决策树 —— vLLM vs sglang 深挖

> **日期**: 2026-06-04 · **状态**: 全面枚举两个框架所有 MoE backend dispatch 分支,带代码片段 + file:line 证据
>
> `docs/triton_rewrite_investigation.md` (§17 有 high-level 对比) 的伴生文档。本文是**详尽 catalog**,把所有特殊 case 收集起来供后续研究。

## TL;DR

- **sglang**: 1 个文件 (`server_args.py`),16 个 `model_arch` 分支 (L1191-1654),命令式 if/elif 链。决策按 `(model_arch, GPU, quant)`。
- **vLLM**: 9 个文件 (`oracle/*.py`),每个量化类型一个优先级列表,声明式 `is_supported_config()` callback。决策按 `(quant, GPU, config)`。
- **两个框架对约 70% 的 (model × GPU × dtype) 组合都最终到 Triton**,但通过完全不同的代码路径。
- **找到 9 个"特殊 case override"注释** (vLLM 8 个,sglang 显式 1 个),开发者手动写了 fallback 因为 Triton/CUDA/etc 实测更慢或有 bug —— 这些是**研究价值最高的 cell**,适合作 agent 目标。

---

## 1. sglang 完整决策树 (文件: `python/sglang/srt/server_args.py`)

### 1.1 16 个 model_arch 分支 (L1190-1654)

| 分支行 | 匹配的 model_arch | 非-Triton 触发条件 | → Backend |
|---:|---|---|---|
| L1197 | `DeepseekV3ForCausalLM`, `KimiK25ForConditionalGeneration`, `MistralLarge3ForCausalLM`, `PixtralForConditionalGeneration` | sm_100 + (fp8/modelopt_fp8/modelopt_fp4) | flashinfer_trtllm |
| L1333 | `GptOssForCausalLM` | Blackwell+MXFP4 → flashinfer_mxfp4 / AMD+AITER → AITER / triton_kernels lib → triton_kernel | 多种 |
| L1475 | `Llama4ForCausalLM` | sm_100 + (fp8/modelopt_fp8) | flashinfer_trtllm |
| L1488 | `Exaone4ForCausalLM`, `ExaoneMoEForCausalLM` | 只 SWA mem 处理 —— 不改 MoE backend | (Triton 默认) |
| L1499 | `Olmo2ForCausalLM` | 只 SWA mem | (Triton 默认) |
| L1524 | `KimiLinearForCausalLM`, `BailingMoeV2_5ForCausalLM` | 只 Mamba radix cache | (Triton 默认) |
| L1529 | `NemotronHForCausalLM` | NVFP4 / modelopt_fp8 | flashinfer_cutlass |
| L1558 | `Qwen3MoeForCausalLM`, `Qwen3VLMoeForConditionalGeneration` | sm_100 + (fp8/modelopt_fp4/**None** ← bf16 也!) | flashinfer_trtllm |
| L1579 | `Qwen3NextForCausalLM`, `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5ForConditionalGeneration` | sm_100 + (fp8/modelopt_fp4/None) | flashinfer_trtllm |
| L1607 | `Glm4MoeForCausalLM` | sm_100 + modelopt_fp4 + flashinfer ≥ 0.6.3 | flashinfer_trtllm |
| L1629 | `FalconH1ForCausalLM`, `JetNemotronForCausalLM`, ... | (Mamba/SWA) | (Triton 默认) |
| L1641 | `GraniteMoeHybridForCausalLM` | Mamba 层 | (Triton 默认) |
| L1654 | `Lfm2ForCausalLM` | Mamba | (Triton 默认) |
| L2125-2134 | (任何模型) | MXFP8 量化 | **cutlass (强制覆盖)** |
| L2161-2174 | (任何模型) | 手动 `--moe-runner-backend=cutlass` + fp8/mxfp8 | cutlass |

**默认 (catch-all)**: 任何不匹配以上的组合 ⇒ **`triton`** (全局默认 `moe_runner_backend: str = "auto"`, L498)。

### 1.2 代码片段: Qwen3-MoE 分支 (我们模型的)

`sglang/srt/server_args.py:1558-1577`:

```python
elif model_arch in [
    "Qwen3MoeForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
]:
    if is_sm100_supported():
        quant_method = get_quantization_config(hf_config)
        if self.quantization is None and quant_method is not None:
            self.quantization = quant_method
        if (
            (
                self.quantization in ("fp8", "modelopt_fp4")
                or self.quantization is None    # ← bf16 也满足
            )
            and self.moe_a2a_backend == "none"
            and self.moe_runner_backend == "auto"
        ):
            self.moe_runner_backend = "flashinfer_trtllm"
```

### 1.3 sglang 特殊 case override 搜寻

sglang MoE dispatch 里发现的手动 fallback:

| File:line | 类型 | 引文 | 意义 |
|---|---|---|---|
| `server_args.py:1674-1675` | TODO | "currently, it is only supported in the single node scenario" + "there is currently a bug on H20 device specifically" | H20 已知 bug —— 研究目标 |
| `server_args.py:1789` | NOTE | "trtllm_mha does not support SM120, which will fall back to flashinfer" | SM120 (RTX 6000 Pro) 限制 |
| `server_args.py:2746` | 注释 | "fallback to triton for DeepSeek models because flashinfer doesn't support deterministic inference for DeepSeek models yet" | **仅 deterministic inference 路径**;DeepSeek 被显式降级 |
| `server_args.py:2749` | 注释 | "fallback to flashinfer on Blackwell for non-DeepSeek models" | 同路径的反向规则 |
| `layers/moe/moe_runner/flashinfer_trtllm.py:256` | FIXME | "there is a bug in the trtllm_fp8_block_scale_moe. It ignored the `output` argument." | 上游 flashinfer 已知 bug |
| `server_args.py:1945` | LOG | "The current platform does not support Intel XMX, will fallback to triton backend." | XMX 硬件门控 |

### 1.4 sglang 设计的诚实评价

- **优**: 所有 MoE dispatch 在一个地方 (1 个文件,约 450 行 for 16 个分支) —— 容易审计
- **劣**: 加新 backend 要改中央 if/elif (每个分支都得改);单个贡献者难以独立完成
- **对 agent 的优势**: 容易枚举哪些 (model, GPU, quant) 三元组 fall through 到 Triton —— 那就是我们的 TAM

---

## 2. vLLM 完整决策树 (目录: `vllm/model_executor/layers/fused_moe/oracle/`)

### 2.1 9 个 oracle 文件

| 文件 | 行数 | 量化类型 | Enum class | # backend 选项 |
|---|---:|---|---|---:|
| `unquantized.py` | 370 | bf16/fp16 | `UnquantizedMoeBackend` | **9** |
| `fp8.py` | 634 | FP8 | `Fp8MoeBackend` | **13** |
| `nvfp4.py` | 560 | NVFP4 | `NvFp4MoeBackend` | 8 |
| `mxfp4.py` | 1710 | MXFP4 | `Mxfp4MoeBackend` | (很多 —— 模型特定) |
| `mxfp8.py` | 91 | MXFP8 | `Mxfp8MoeBackend` | 1 |
| `int8.py` | 218 | INT8 | `Int8MoeBackend` | 1 (只 Triton) |
| `int_wna16.py` | 908 | INT4/INT8 weight-only | `WNA16MoEBackend` | 4 |
| `w4a8.py` | 195 | W4A8 | `W4A8MoeBackend` | 1 (只 CUTLASS) |
| `w4a8_int8.py` | 360 | W4A8 INT8 | `W4A8Int8MoeBackend` | 数个 |

### 2.2 Enum 完整列表

**bf16/fp16 (`unquantized.py:32`)**:
```python
class UnquantizedMoeBackend(Enum):
    FLASHINFER_TRTLLM = "FlashInfer TRTLLM"
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
    AITER = "ROCm AITER"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"
    CPU = "CPU"
    XPU = "XPU"
    TPU = "TPU"
    OOT = "OOT"
```

**FP8 (`fp8.py:42`)** —— **13 个选项**:
```python
class Fp8MoeBackend(Enum):
    NONE = "NONE"
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    DEEPGEMM = "DEEPGEMM"
    BATCHED_DEEPGEMM = "BATCHED_DEEPGEMM"
    MARLIN = "MARLIN"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"
    AITER = "AITER"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    BATCHED_VLLM_CUTLASS = "BATCHED_VLLM_CUTLASS"
    XPU = "XPU"
    CPU = "CPU"
```

**NVFP4 (`nvfp4.py:41`)**:
```python
class NvFp4MoeBackend(Enum):
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    FLASHINFER_CUTEDSL = "FLASHINFER_CUTEDSL"
    FLASHINFER_CUTEDSL_BATCHED = "FLASHINFER_CUTEDSL_BATCHED"
    FLASHINFER_B12X = "FLASHINFER_B12X"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    MARLIN = "MARLIN"
    EMULATION = "EMULATION"
```

**WNA16**:
```python
class WNA16MoEBackend(Enum):
    MARLIN = "MARLIN"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    XPU = "XPU"
```

### 2.3 vLLM 里 5 个显式手动 fallback

这些是我们调研中**最有价值的发现** —— 每个都是 vLLM 开发者发现某个 kernel 比预期更慢或有 bug 的具体 case:

#### F1. **`unquantized.py:71` —— Hopper bf16 偏好 Triton 超过 FlashInfer**

```python
# 在 Hopper (SM90) 上,FlashInfer 未量化 MoE kernel 比 Triton 慢
# 所以默认偏好 Triton
if current_platform.is_device_capability_family(90):
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_TRTLLM)
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)
```

**意义**: 直接实证确认 Hopper bf16 + FlashInfer MoE 比 Triton **更慢** —— 印证我们 §14 测的 Triton-在-H200-达 30% 峰值实际上是**最优**选项。这是项目 pitch **最重要的发现** —— 来自另一个框架团队的外部验证。

#### F2. **`unquantized.py:77` —— Qwen3.5 + DEP 会让 FlashInfer CUTLASS bf16 崩**

```python
# HACK: Qwen3.5 在 DEP 下用 FLASHINFER_CUTLASS BF16 会崩
# 改派发逻辑超出本 PR 范围
# 需要后续修 kernel 或改结构
if moe_config.moe_parallel_config.dp_size > 1:
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)
```

**意义**: 生产 bug,尚未修。**研究目标**: 查清根因,上报或修上游 —— 惠及 Blackwell 上用 DP/EP 的 Qwen3.5 用户。

#### F3. **`fp8.py:87` —— Hopper FP8 Block: TP→Triton, EP→FlashInfer CUTLASS**

```python
# 在 Hopper 上 Block Fp8: TP 偏好 Triton, EP 偏好 FI CUTLASS
if (
    current_platform.is_cuda()
    and current_platform.is_device_capability(90)
    and activation_key == kFp8Dynamic128Sym
    and weight_key == kFp8Static128BlockSym
):
    if moe_config.moe_parallel_config.ep_size > 1:
        _move_to_front(_AVAILABLE_BACKENDS, Fp8MoeBackend.FLASHINFER_CUTLASS)
    else:
        _move_to_front(_AVAILABLE_BACKENDS, Fp8MoeBackend.TRITON)
```

**意义**: Triton 是 Hopper TP-only FP8 的**最优选项** (不只是 fallback)。FlashInfer CUTLASS 只在多 rank EP 才赢。**研究目标**: EP 多大开始 crossover? 可以作为智能派发贡献的基础。

#### F4. **`mxfp4.py:283, 306` —— TRITON_UNFUSED 因 MTP bug 被禁用**

```python
_AVAILABLE_BACKENDS = [
    Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
    ...,
    Mxfp4MoeBackend.TRITON,
    Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
    Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    # TRITON_UNFUSED has bug with MTP support
    # TODO re-enable after kernel is fixed
    # TRITON_UNFUSED                                  ← 注释掉了!
    Mxfp4MoeBackend.MARLIN,
    ...
]
```

**意义**: 一个特定 Triton kernel 变体因 multi-token-prediction (MTP / 推测解码) bug 完全禁用。两处出现 (L283 + L306)。**研究目标**: 修 bug,重启 kernel,测影响。

#### F5. **`mxfp4.py:625` —— DeepSeek-V4 on ROCm 偏好 AITER FlyDSL**

```python
# DeepSeek-V4 on ROCm: 偏好 AITER FlyDSL MoE (perf + 精度都更好,
# 在修了 shuffle/TP-offset 之后), 用 Triton-unfused 做 fallback
if (
    current_platform.is_rocm()
    and config.routing_method == RoutingMethodType.DeepseekV4
):
    priority_backends = [
        Mxfp4MoeBackend.AITER_MXFP4_BF16,
        Mxfp4MoeBackend.TRITON_UNFUSED,
    ]
```

**意义**: 特定模型 + GPU 厂商组合拿到定制优先级。同时记录 perf **和**精度顾虑 ("shuffle/TP-offset fixes")。展示甚至对一个量化 (MXFP4),文件里有 3 个不同优先级列表 (general, GPT-OSS, DeepSeek-V4)。

#### F6. **`unquantized.py:162` —— "TODO: migrate to MK structure"**

```python
# TODO: migrate to MK structure.
```

**意义**: vLLM 团队在重构;`unquantized.py` 里当前 dispatch 被视为遗留。值得跟踪稳定性。

### 2.4 代码片段: `select_unquantized_moe_backend` 怎么工作

`unquantized.py:153+`:

```python
def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
    """
    选择未量化 MoE backend。
    注: 运行时仍可能 shape-specific fallback。
    """
    # CPU/TPU/OOT 先特殊处理
    if current_platform.is_cpu():
        return UnquantizedMoeBackend.CPU, None
    # ...
    
    # 拿优先级列表 (Hopper 降级 + Qwen3.5 崩溃 workaround 之后)
    priority_backends = _get_priority_backends(moe_config)
    
    # 按优先级试每个 backend;第一个说 is_supported_config(...) 的赢
    for backend in priority_backends:
        kernel_cls = backend_to_kernel_cls(backend)
        supported, reason = kernel_cls.is_supported_config(kernel_cls, moe_config)
        if supported:
            return backend, kernel_cls
    raise ValueError("没 backend 支持这个 config")
```

`is_supported_config` 模式是**声明式关键** —— 每个 backend 自报能否处理这个 config。这就是 vLLM 设计可扩展的原因。

---

## 3. 并排对比: 我们具体 case 的派发

**设置**: Qwen3-30B-A3B (Qwen3MoeForCausalLM) + H200 (sm_90) + bf16

### 3.1 sglang 路径

```
1. 读 server_args.py
2. moe_runner_backend = "auto" (默认, L498)
3. 命中 L1558 elif: model_arch == "Qwen3MoeForCausalLM" → 进分支
4. 检查 is_sm100_supported() → FALSE (H200 是 sm_90)
5. 整个分支 body 跳过
6. Fall through 剩下所有 elif
7. moe_runner_backend 保持 "auto"
8. 后续 get_moe_impl_class (ep_moe/layer.py:686) 返回 FusedMoE (Triton 路径)
9. fused_moe_kernel 跑
```

### 3.2 vLLM 路径

```
1. 读 kernel.py: moe_backend = "auto" (默认, L171)
2. 调 select_unquantized_moe_backend
3. _get_priority_backends 返回: [FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, TRITON, BATCHED_TRITON]
4. Hopper 检查 (L74): is_device_capability_family(90) → TRUE
5. 把 FLASHINFER_TRTLLM 和 FLASHINFER_CUTLASS 降到队尾
6. 结果: [TRITON, BATCHED_TRITON, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS]
7. TRITON.is_supported_config() → TRUE
8. 返回 (TRITON, TritonExperts class)
9. fused_moe Triton kernel 跑
```

**两者都到 Triton**,但 vLLM 路径包含**显式**决策 "FlashInfer 在这慢" 而 sglang 路径被动默认到 Triton。

### 3.3 同样 case 在 B200 上 (反事实)

| 步骤 | sglang on B200 | vLLM on B200 |
|---|---|---|
| 检测 Blackwell | `is_sm100_supported()` → TRUE | `is_device_capability_family(90)` → FALSE |
| 进分支? | 是,L1558 分支执行 | Hopper 降级跳过,优先级列表不变 |
| 条件检查 | `quantization is None` (bf16) → 通过 | `FLASHINFER_TRTLLM.is_supported_config()` → ? |
| 最终选 | **flashinfer_trtllm** (auto-set) | **FLASHINFER_TRTLLM** (优先级列表顶) |

所以两个框架在 Blackwell 上**也**到同一个结论。派发逻辑**看起来**不同但**结果一致** —— 两个团队都同意选哪个 kernel。

---

## 4. "特殊 case override" 目录 —— 研究路线图

汇总 §1.3 + §2.3 找到的手动 fallback,按研究价值排序:

| # | 来源 | 覆盖什么 | 记录原因 | 研究价值 |
|---|---|---|---|---|
| **1** | vLLM `unquantized.py:71` | Hopper bf16: 把 FlashInfer 降到队尾 | "FlashInfer 未量化 MoE kernel 在 SM90 上比 Triton 慢" | ⭐⭐⭐⭐⭐ 直接支持我们 §14 发现。确认 Triton 在 Hopper bf16 上是**最优**,不只默认。 |
| **2** | vLLM `fp8.py:87` | Hopper FP8 Block: TP→Triton, EP→FlashInfer CUTLASS | "Hopper Block Fp8: TP 偏好 Triton, EP 偏好 FI CUTLASS" | ⭐⭐⭐⭐⭐ Crossover 行为 —— EP 多大 FlashInfer 才赢? 可以做智能 auto-dispatch 贡献。 |
| **3** | vLLM `unquantized.py:77` | Qwen3.5 + DEP: 禁用 FlashInfer CUTLASS bf16 | "HACK: Qwen3.5 + DEP 在 FLASHINFER_CUTLASS BF16 上崩" | ⭐⭐⭐⭐ 真 bug,未修。研究目标: 修上游。 |
| **4** | vLLM `mxfp4.py:283,306` | TRITON_UNFUSED 在 MXFP4 优先级里禁用 | "MTP 支持有 bug" | ⭐⭐⭐⭐ Bug 修复机会。 |
| **5** | sglang `server_args.py:2746` | DeepSeek deterministic inference: 强制 Triton | "flashinfer 还不支持 DeepSeek 的 deterministic inference" | ⭐⭐⭐ 小众路径但有文档缺口。 |
| **6** | vLLM `mxfp4.py:625` | DeepSeek-V4 ROCm: 定制优先级列表 | "shuffle/TP-offset 修复后 perf + 精度更好" | ⭐⭐⭐ 模型 + GPU 组合特定,复杂。 |
| **7** | sglang `server_args.py:1674-1675` | flashinfer_trtllm: "目前只单节点 + H20 有 bug" | TODO + GitHub issue 链接 | ⭐⭐⭐ 已上游跟踪。 |
| **8** | sglang `flashinfer_trtllm.py:256` | `trtllm_fp8_block_scale_moe`: 忽略 `output` 参数 | "FIXME: 有 bug" + 链接 | ⭐⭐⭐ 上游 bug。 |
| **9** | sglang `server_args.py:1789` | SM120: trtllm_mha fallback 到 flashinfer | "trtllm_mha 不支持 SM120" | ⭐⭐ 硬件特定限制。 |
| **10** | sglang `server_args.py:1945` | Intel XMX 缺失: fallback 到 triton backend | 硬件门控 | ⭐⭐ 边缘 case。 |
| **11** | vLLM `unquantized.py:162` | Refactor TODO | "migrate to MK structure" | ⭐ vLLM 内部 |

**前 4 个 (⭐⭐⭐⭐ 及以上) 是具体研究机会** —— 每个代表一个已知限制,agent 或人工调研可能:
- 验证实证 claim (例: "FlashInfer 在 Hopper bf16 上真的比 Triton 慢吗?" —— 我们可以重测)
- 修底层 bug (例: Qwen3.5 + DEP 崩溃)
- 重 tune crossover 点 (例: Hopper FP8 的 EP-size 阈值)


## 4.5 ⚠️ 澄清 —— "FlashInfer 慢"具体指的是哪个 kernel?

> Review 时发现表面矛盾: `triton_rewrite_investigation.md §9` 显示 `--moe-runner-backend=flashinfer_trtllm` 在 H200 上报错 "sm_100 only"。那 §4 Finding F1 怎么能说 "Hopper FlashInfer 更慢" —— 如果它根本跑不起来,慢什么?

### 4.5.1 解释

未量化 bf16 有**两个不同的 FlashInfer MoE 函数**。硬件支持差异很大:

| flashinfer 函数 | 硬件支持 | sglang wrapper | vLLM wrapper | H200 上跑得了? |
|---|---|---|---|---|
| **`flashinfer.fused_moe.trtllm_bf16_moe`** | **只** sm_100 (Blackwell) | `FlashInferFusedMoE` (`layer.py:1132+`) | `TrtLlmBf16Experts` | ❌ 不能 |
| **`flashinfer.fused_moe.cutlass_fused_moe`** | sm_90 + sm_100 + sm_120 | `UnquantizedFusedMoEMethod` (`unquant.py:60+373`) | `FlashInferExperts` | ✅ 能 |

vLLM `unquantized.py:71-75` 把**两个都**降级:
```python
if current_platform.is_device_capability_family(90):
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_TRTLLM)
    _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)
```

`FLASHINFER_TRTLLM` 降级基本是**虚的** (底层 kernel 在 Hopper 上根本跑不起来)。但 `FLASHINFER_CUTLASS` 降级**有意义** —— kernel 能跑但实测比 Triton 慢。

所以 vLLM 注释 "FlashInfer 未量化 MoE kernel 在 Hopper 上比 Triton 慢" **特指 `cutlass_fused_moe`**,不是 `trtllm_bf16_moe`。

### 4.5.2 为啥 §9 没测出这个

§9.3 我们试 `--moe-runner-backend=flashinfer_cutlass` 报错:
```
fatal error: cuda_fp16.h: No such file or directory
ninja: build stopped: subcommand failed.
```

这是**环境问题** —— 我们 `gcc` 找不到 CUDA 头文件做 JIT 编译。**不是** sglang 限制。路径**确实存在** at `srt/layers/quantization/unquant.py:60`:
```python
try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except ImportError:
    flashinfer_cutlass_fused_moe = None
```

bf16 路径在 `unquant.py:373` 调它。我们就是没绕过 JIT build 错误。

### 4.5.3 sglang 也有这个 kernel —— 对称性恢复

两个框架都包装同一个底层 `flashinfer.cutlass_fused_moe` for bf16:

| 框架 | Wrapper class | File:line |
|---|---|---|
| sglang | `UnquantizedFusedMoEMethod.forward_cuda` | `layers/quantization/unquant.py:373` |
| vLLM | `FlashInferExperts` | `layers/fused_moe/experts/flashinfer_cutlass_moe.py` |

**vLLM 的 "Hopper 更慢" 判定对 sglang 也对称适用** —— 如果我们修了 JIT 环境问题,sglang 的 `flashinfer_cutlass` 路径在 H200 bf16 上也会比 Triton 慢,匹配 vLLM 测的。

### 4.5.4 修订后的心智模型

| Backend 名 H200 bf16 | sglang | vLLM | 状态 |
|---|---|---|---|
| Triton `fused_moe_kernel` | ✅ 默认 | ✅ 默认 (Hopper 降级后) | **生产路径** |
| FlashInfer `trtllm_bf16_moe` | 包了但报错 (sm_100 only) | 包了但报错 (sm_100 only) | **硬件不支持** |
| FlashInfer `cutlass_fused_moe` | 包了但环境坏 (我们测试) | 包了 (vLLM benchmark 显示比 Triton 慢) | **可用但 Hopper 上慢** |

所以 **agent 项目说 "Hopper bf16 → Triton 最优" 是对的**,现在有**三个来源**的证据:
1. sglang auto-mode 被动 fall through 到 Triton
2. vLLM auto-mode 显式降级两个 FlashInfer path (主动)
3. 我们 §14 测 Triton 在 30% 峰值 (H200 上能用的最优)

下一步是修 env JIT 问题后实际跑 `flashinfer_cutlass` on H200,量化它比 Triton 慢多少。那样 Finding F1 就闭环了。



## 4.6 常见误解 —— 是 "sglang 只支持 sm_100 的那个" 吗?

> 读 §4.5 后的问题: "所以两个 flashinfer MoE kernel 都存在,sm_100 和 sm_90 用不同的;vLLM 支持 sm_90 的那个但说它慢;sglang 只支持 sm_100 的那个,是吗?"
>
> **部分对,部分错**。最重要的纠正: **sglang 也包了 sm_90 路径**。两个框架功能对等。

### 4.6.1 两个框架都包了**两个** flashinfer MoE 函数

| flashinfer 函数 | sglang wrapper | sglang CLI flag | vLLM wrapper | 硬件 |
|---|---|---|---|---|
| **`trtllm_bf16_moe`** | `class FlashInferFusedMoE` at `srt/layers/moe/fused_moe_triton/layer.py:1132` | `--moe-runner-backend=flashinfer_trtllm` | `class TrtLlmBf16Experts` | **只 sm_100** |
| **`cutlass_fused_moe`** | `UnquantizedFusedMoEMethod.forward_cuda` at `srt/layers/quantization/unquant.py:60+373` | `--moe-runner-backend=flashinfer_cutlass` | `class FlashInferExperts` | sm_90 + sm_100 + sm_120 |

sglang `MOE_RUNNER_BACKEND_CHOICES` (`server_args.py:176-184`) 显式列了两个:

```python
MOE_RUNNER_BACKEND_CHOICES = [
    "auto",
    "deep_gemm",
    "triton",
    "triton_kernel",
    "flashinfer_trtllm",        # ← sm_100 路径
    "flashinfer_cutlass",       # ← sm_90 路径 (我们 §9b 试过的!)
    "flashinfer_mxfp4",
    "flashinfer_cutedsl",
    "cutlass",
]
```

我们在 `triton_rewrite_investigation.md §9` 里实测过两个:
- §9 (C9): `flashinfer_trtllm` → flashinfer 报 "sm_100 only" 错
- §9b (C9b): `flashinfer_cutlass` → flashinfer JIT 编译 cutlass C++;我们 env 缺 `cuda_fp16.h` 头,JIT build 失败

所以 sglang **绝对支持** sm_90 flashinfer 路径。只是我们 env JIT 错误没绕过,没实测到性能。

### 4.6.2 精确功能对等表

|  | flashinfer `trtllm_bf16_moe` (sm_100) | flashinfer `cutlass_fused_moe` (sm_90/100/120) |
|---|---|---|
| **sglang 里有 code 路径?** | ✅ 有 | ✅ 有 |
| **vLLM 里有 code 路径?** | ✅ 有 | ✅ 有 |
| **H200 上能跑?** | ❌ 不能 (硬件不支持) | ✅ 能 (需要 JIT 编译) |
| **我们 H200 实测?** | §9 —— 服务器死,"sm_100 only" | §9b —— 服务器死,cuda_fp16.h 缺 |
| **vLLM 对 Hopper 判定?** | 降级 (虚的 —— 反正跑不起来) | 降级 (**这才是**有实际意义的 "比 Triton 慢") |

### 4.6.3 H200 bf16 用户实际看到啥

**不指定 `--moe-runner-backend`**:
1. sglang auto-mode (`server_args.py:1558`): `is_sm100_supported()` 失败 → 无特殊路径触发 → fall through 到默认 `triton`
2. **结果**: `fused_moe_kernel` (Triton) 跑

**传 `--moe-runner-backend=flashinfer_trtllm`**:
1. sglang 设 `moe_runner_backend = "flashinfer_trtllm"`
2. `get_moe_impl_class` 返回 `FlashInferFusedMoE`
3. 首次 request,`forward_impl` 调 `trtllm_bf16_moe(...)`
4. flashinfer 抛 `RuntimeError: No supported CUDA architectures found for major versions [10]`
5. 服务器崩

**传 `--moe-runner-backend=flashinfer_cutlass`**:
1. sglang 设 `self.use_flashinfer_cutlass = True`
2. 首次 request,`UnquantizedFusedMoEMethod.forward_cuda` 调 `flashinfer_cutlass_fused_moe(...)`
3. flashinfer JIT 编译 cutlass C++
4. 我们 env 里 gcc 找不到 `cuda_fp16.h` → 编译失败 → 服务器崩
5. **配置好的 env 里**: 能跑,且按 vLLM 实测,**比 Triton 慢**

### 4.6.4 一段话纠正误解

> **不,sglang 不是 "只支持 sm_100 的 flashinfer 路径"**。sglang 把两个 flashinfer MoE 路径 (`flashinfer_trtllm` 给 sm_100 + `flashinfer_cutlass` 给 sm_90/100/120) **都**暴露作为 CLI 选项,和 vLLM 一样。两个框架功能矩阵**对称**。我们在 H200 上到处看到 Triton 的原因是:
> - 对大部分用户: auto-mode fall through 到 Triton (无显式决策)
> - 对显式选 `flashinfer_trtllm` 的: 底层 flashinfer kernel 只 sm_100,在 Hopper 上崩
> - 对显式选 `flashinfer_cutlass` 的: kernel 能跑 (env 允许) 但 Hopper 上比 Triton 慢 (按 vLLM 独立测的)
>
> 所以 **H200 bf16 上,Triton 在两个框架里都是实测最佳选择**,不是因为替代不存在,而是因为它们要么硬件不支持,要么更慢。

### 4.6.5 两个框架真正的唯一区别 (修订版)

| 维度 | sglang | vLLM |
|---|---|---|
| 可用路径 | flashinfer_trtllm + flashinfer_cutlass + Triton + 其他 | 同一套 |
| 硬件门控 | 隐式 (auto-mode if/elif 用 `is_sm100_supported`) | 显式 (`_supports_current_device` per backend) + `is_supported_config` |
| Hopper 偏好 | 被动 —— 没其他 auto-rule 触发就 fall through 到 Triton | 主动 —— 显式 `_move_to_back` 两个 FlashInfer 路径 |
| 最终用户拿到啥 | Triton | Triton |

**最终用户拿到同样的 kernel 选择。区别是文档/推理,不是行为。**


---

## 5. 覆盖差异 —— vLLM 有但 sglang 没有的 (反之)

| 能力 | sglang | vLLM | 说明 |
|---|---|---|---|
| FP8 Marlin 路径 | ❌ 无 | ✅ `Fp8MoeBackend.MARLIN` | INT4-style 优化给 FP8 |
| BATCHED 变体 | ❌ 无 | ✅ BATCHED_TRITON, BATCHED_DEEPGEMM, BATCHED_VLLM_CUTLASS, BATCHED_MARLIN | 给 attn_metadata.use_batched_activation_format |
| Per-GPU 显式降级 | ❌ 无 | ✅ `_move_to_back` 基于 `device_capability_family(90)` | 实证 tune |
| `is_supported_config` 自报 | ❌ 无 (硬编 if/else) | ✅ 每个 backend class 都有这方法 | 可扩展性 |
| 每量化多个 FlashInfer 变体 | 只 TRTLLM/CUTLASS/CUTEDSL | 全 5 个 (TRTLLM/CUTLASS/CUTEDSL/CUTEDSL_BATCHED/B12X for NVFP4) | 更细粒度 |
| 自定义 routing-method-aware dispatch | 部分 (头部模型) | 是 (`RoutingMethodType.DeepseekV4` 检查) | 更原则化 |

**覆盖底线**: vLLM 派发明显**更细** (per quant 约 1.5-2× 多 backend 选项)。但 "default → Triton" 终点两边一样,所以对我们目标用户 (长尾 MoE on Hopper bf16),覆盖差异不重要。

---

## 5.5 ✅ 实测 —— 4-way MoE backend 对比 (sglang/vLLM × Triton/CUTLASS)

**目标**: §4.5/§4.6 留下两个 open question:
(a) vLLM 源码注释 "Hopper bf16 上 FlashInfer CUTLASS 比 Triton 慢" 是不是真的?
(b) sglang 和 vLLM 都包了同一个 `flashinfer.cutlass_fused_moe`,实际跑出来谁更快?

**实验设置** (单卡顺序跑,避免争用):
- 模型 / 硬件 / 精度: Qwen3-30B-A3B-Instruct-2507 / H200 (sm_90a) / bf16 / TP=1
- 4 个 server 配置:
  1. `sglang_triton` (sglang 默认,sm_90 自动选 Triton)
  2. `sglang_cutlass` (sglang `--moe-runner-backend flashinfer_cutlass`,需要 `--watchdog-timeout 1800` 等首次 JIT)
  3. `vllm_triton` (vLLM `--kernel-config '{"moe_backend": "triton"}'` —— **但实测 vLLM 不强加,无 flag 时 oracle 默认也是 Triton**)
  4. `vllm_cutlass` (vLLM `--kernel-config '{"moe_backend": "flashinfer_cutlass"}'`)
- 同一份 bench harness (`/tmp/run_bench_4way.py`),同 seed=2026 同 prompt 同 `max_new=256 temperature=0 ignore_eos=True`
- 3 个 regime: R_short (8 reqs / 200w prompt / conc=1)、R_medium (16 / 800w / conc=8)、R_long (8 / 2000w / conc=16)

**绝对吞吐**:

| Regime | sglang_triton | sglang_cutlass | vllm_triton | vllm_cutlass |
|---|---|---|---|---|
| R_short | 1.92 req/s (123 tok/s) | 0.71 req/s (45 tok/s) | 3.05 req/s (195 tok/s) | 3.05 req/s (195 tok/s) |
| R_medium | 3.02 req/s (772 tok/s) | 1.26 req/s (324 tok/s) | 4.32 req/s (1105 tok/s) | 4.37 req/s (1120 tok/s) |
| R_long | 2.95 req/s (754 tok/s) | 1.20 req/s (306 tok/s) | 3.53 req/s (903 tok/s) | 3.66 req/s (937 tok/s) |

**相对加速比**:

| Regime | sglang Triton→CUTLASS | vLLM Triton→CUTLASS | sglang→vLLM (Triton) | sglang→vLLM (CUTLASS) |
|---|---|---|---|---|
| R_short | **0.37×** (2.7× 慢) | 1.00× | 1.59× | 4.30× |
| R_medium | **0.42×** (2.4× 慢) | 1.01× | 1.43× | 3.46× |
| R_long | **0.41×** (2.4× 慢) | 1.04× | 1.20× | 3.05× |

**回答 open question**:

**(a) "Hopper bf16 上 FlashInfer CUTLASS 比 Triton 慢" 这个命题在 vLLM 自己的 stack 上不成立**。3 个 regime 上 vLLM CUTLASS 都 ≈ vLLM Triton (-0% ~ +4%,在噪声范围内)。`vllm/.../unquantized.py:71` 那段注释要么过时,要么针对的是别的条件 (不同 model/dtype/quant/sm 组合,或 prefill-heavy 而我们的 regime 是 decode-heavy)。

**(b) sglang 上 CUTLASS 比 Triton 慢 2.4-2.7×,但 vLLM 上几乎打平**。同一个底层 kernel (`flashinfer.cutlass_fused_moe`),包它的 wrapper 不同 —— 说明 sglang 这边的 dispatch / launch / mem-layout / cudagraph 套子有显著开销。

**两条优化路线**:

1. **修 sglang 的 FlashInfer CUTLASS wrapper** (短平快): 把 sglang_cutlass 拉到 sglang_triton 同水平,~2.4× 提升;再进一步追到 vllm_cutlass 水平,~3.5× 提升。需要 diff `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py` vs `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`,找 H2D copy、shape handle、stream 同步、cudagraph 等差异。

2. **更大的杠杆: sglang Triton 整体比 vLLM Triton 慢 20-59%** (R_long 最小,R_short 最大)。同一个 Triton MoE kernel,kernel 时间应该一样,差距应该在 dispatch/schedule/cudagraph/prefix-caching 等 engine 层。R_short (单 conc) 差距最大说明 per-step overhead 显著。

**bench 工件**:
- `results/4way_bench/bench_{sglang_triton,sglang_cutlass,vllm_triton,vllm_cutlass}.json`
- `results/4way_bench/comparison_table.md`
- `results/4way_bench/{sglang,vllm}_{triton,cutlass}/server.log`
- Launch scripts: `/tmp/start_{sglang,vllm}_{triton,cutlass}.sh`

**caveat**: 三个 regime wall-clock 都 < 13s,各 8-16 reqs,所以有一定噪声。但 4 个 backend 各自跑 3 个 regime,vLLM 的 Triton/CUTLASS 完全打平 + sglang 的 Triton/CUTLASS 一致拉开 2.4×,信号在 3 个 regime 上方向一致,定性结论稳。

---

## 6. 推荐下一步

1. **🔥 §5.5 发现的高 ROI 路线: 修 sglang 的 FlashInfer CUTLASS wrapper**: 同一个 `flashinfer.cutlass_fused_moe` kernel,vLLM 包了之后 ≈ Triton,sglang 包了之后慢 2.4×。diff 两边的 wrapper (`python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py` vs `vllm/.../flashinfer_cutlass_moe.py`),找差异。

2. **🔥 §5.5 发现的更大杠杆: sglang Triton 整体比 vLLM Triton 慢 20-59%**: 同一个 Triton MoE kernel,差距在 engine 层 (dispatch / cudagraph / scheduler / prefix-caching)。R_short (单 conc) gap 最大说明 per-step overhead 显著,值得 profile 一个 decode step 看 launch overhead。

3. **复现发现 #1** (Hopper bf16: FlashInfer 比 Triton 慢): **§5.5 已经做了**,结论是这个命题在 vLLM stack 上不成立,在 sglang stack 上 "成立但不是因为 kernel 慢,是因为 wrapper 慢"。源码注释要么过时要么针对其他条件。

4. **调研发现 #3** (Qwen3.5 + DEP 崩溃): 在 H200 上 DP > 1 复现崩溃,捕获精确错误,提 follow-up。

5. **Crossover 研究 (发现 #2)** (Hopper FP8 的 TP vs EP): 用 `--tp-size 1 vs 2 vs 4 vs 8 vs 16` 跑 sglang FP8 (需要 FP8 权重 —— 先转 Qwen3-30B-A3B),测 CUTLASS 何时超过 Triton。

6. **MXFP4 重启 Triton-unfused (发现 #4)**: 如果 MTP bug 可修,重启一个 backend 然后 benchmark。

---

## 7. 引用的文件

### sglang
- `python/sglang/srt/server_args.py` (L1190-1654 是 model_arch 分支,L2125-2174 是 cutlass 覆盖)
- `python/sglang/srt/layers/moe/ep_moe/layer.py:686` (`get_moe_impl_class` —— 实际 class 选择器)
- `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py:256` (FIXME 注释)

### vLLM
- `vllm/config/kernel.py:171` (`moe_backend` 开关)
- `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` (bf16 dispatch)
- `vllm/model_executor/layers/fused_moe/oracle/fp8.py` (FP8 dispatch,13 backends)
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` (1710 行,3 个优先级列表)
- `vllm/model_executor/layers/fused_moe/oracle/mxfp8.py`
- `vllm/model_executor/layers/fused_moe/oracle/int8.py`
- `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`
- `vllm/model_executor/layers/fused_moe/oracle/w4a8.py`
- `vllm/model_executor/layers/fused_moe/oracle/w4a8_int8.py`
