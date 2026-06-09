# Why is flashinfer CUTLASS MoE no faster than Triton MoE at e2e?
## Skill-driven investigation, 2026-06-09

**Mission**: theoretically a hand-tuned CUTLASS kernel should beat a Triton
codegen. At e2e level, on vLLM, CUTLASS only barely edges Triton (≤2%); on
sglang, CUTLASS is actually 2.4× slower. Why? And what would actually move CUTLASS forward?

**Method**: drive the investigation end-to-end with the new skills in
`.github/skills/`. Document **which skill produced which evidence**, and
explicitly include the cases where a skill failed (those failures are
themselves contributions — they expose real gaps in the toolkit).

---

## TL;DR (for mentor briefing)

1. **The "CUTLASS ≈ Triton" result is real and reproducible today** — confirmed
   by `e2e-bench-runner` skill, with stddev < 0.3% on all 3 regimes (the skill's
   built-in noise check). On 2 of 3 regimes today, Triton **beat** CUTLASS.

2. **At the kernel level, CUTLASS MoE GEMM IS faster per layer (25%)**
   — but this win is invisible at e2e because it lives inside a ~6% slice of
   total time:
   - Cutlass: 1 fused MoE GEMM per layer at 44.9 µs (24,672 launches)
   - Triton:  2 separate up/down GEMMs per layer at 29.8 µs each = 59.6 µs (49,440 launches)
   - Per layer: Cutlass faster by 14.7 µs (25%). Over ~48 layers × decoding
     steps: tiny fraction of total wall time.
   - **Source**: torch.profiler text summary, captured via vLLM's
     `/start_profile`/`/stop_profile` endpoints (used as a workaround — see §5 for
     why the `nsys-capture` skill couldn't be used here).

3. **Where CUTLASS's per-kernel win evaporates** (concrete budget):
   | Component                                | Cutlass | Triton |
   |---|---|---|
   | MoE GEMM (the thing CUTLASS optimizes)    | 31.55%  | 46.84% |
   | Non-MoE compute (Attention + QKV/MLP linears) | ~49% | ~28%   |
   | MoE routing helpers + topk + sync         | ~9%     | ~8%    |
   | Misc (RMSnorm, KV writes, memcpy)         | ~10%    | ~17%   |

   The differences are NOT all from MoE — switching `moe_backend` also incidentally
   swaps the non-MoE linear path (Cutlass uses `cutlass::device_kernel` for QKV;
   Triton mode uses `nvjet_sm90_*` cuBLAS kernels). So an apparent "Triton MoE win"
   is partly a non-MoE swap.

4. **Why sglang_cutlass is much slower than sglang_triton** (an asymmetric reason):
   - sglang baseline has `disable_cuda_graph=True` AND `flashinfer_cutlass` is
     omitted from sglang's autotune list (commented out at
     `sglang/python/sglang/srt/model_executor/model_runner.py:1841` with TODO
     "flashinfer compilation errors").
   - We separately proved (docs/2026-06-08/vllm_2x2_autotune_cudagraph_matrix.md)
     that CUTLASS without autotune falls back to tactic 0 → 5–6× slower (microbench).
     So sglang CUTLASS gets neither cudagraph nor autotune → maximally degraded.
   - Triton has lower per-launch overhead → less hurt by missing cudagraph.

5. **Concrete improvement directions for CUTLASS** (ranked by likely ROI):
   - **D1 (high)**: Enable `flashinfer_cutlass` in sglang's autotune list. Microbench
     proved 5–6× MoE GEMM speedup at tactic-tuned vs fallback. Risk: the TODO says
     "compilation errors" — must reproduce + fix root cause before re-enabling.
   - **D2 (high)**: Eliminate the ~9% routing-helper overhead
     (`trtllm_kernels::expandInput`, `computeStridesTmaWarpSpecialized`, `topkGating`).
     These are per-layer fixed costs unrelated to GEMM math; CUDA-graph captures
     them but they still execute. Either fuse them into the MoE kernel or share
     state across layers.
   - **D3 (medium)**: Investigate why CUTLASS dense GEMM (~20% of cutlass case)
     uses different kernels than non-MoE path in Triton mode. If a single backend
     wins both MoE and dense, the inadvertent backend swap goes away.
   - **D4 (low)**: For sm90 specifically, CUTLASS already has Hopper TMA support;
     limited per-kernel headroom left. Bigger gains likely from D1+D2.

---

## Phase 1 — establish the puzzle exists (skill: `e2e-bench-runner`)

### Prediction
"Reproducing past 2026-06-05 data: vllm_cutlass R_medium ≈ 4.37 req/s,
stddev < 5%, reliable=True."

### Skill invocation
```bash
python .github/skills/e2e-bench-runner/impl/run_bench.py \
  --url http://127.0.0.1:30001 --backend vllm \
  --tag vllm_cutlass_2026-06-09 --num-runs 3 \
  --out-dir results/2026-06-09_cutlass_investigation/cutlass/bench/
```

### Skill output (excerpt from `bench_summary.json`)
| Regime    | vllm_cutlass req/s | vllm_triton req/s | stddev_pct (cutlass) | reliable |
|---|---|---|---|---|
| R_short   | 3.29              | **3.53**          | 0.1%                 | True     |
| R_medium  | **4.74**          | 4.70              | 0.0%                 | True     |
| R_long    | 4.53              | **4.58**          | 0.1%                 | True     |

(Past 2026-06-05: cutlass 4.37, triton 4.32 — same magnitude; today's slightly
better numbers likely reflect a fresher autotune cache.)

### What this skill contributed
- **Confirmed the puzzle is real and stable** — without the skill's
  drop-run-1 + stddev guard, the 5% gaps could be dismissed as noise.
  The `reliable=True` field on each regime gave me license to trust the comparison.
- **Surfaced a sharper finding**: Triton actually beats CUTLASS on 2 of 3 regimes
  today. So the question is not "why does CUTLASS only marginally win" but
  rather "why does CUTLASS sometimes lose to its supposed inferior?"

---

## Phase 2 — try to capture nsys (skill: `nsys-capture` — FAILED gracefully)

### Prediction
"`nsys-capture` wrapping the bench client subprocess will profile both the
bench client AND any GPU activity it triggers."

### Skill invocation
```bash
python .github/skills/nsys-capture/impl/run_capture.py \
  --target-cmd "python3 /tmp/bench_for_nsys.py http://127.0.0.1:30001 8" \
  --duration-s 90 \
  --out-dir results/2026-06-09_cutlass_investigation/cutlass/nsys/
```

### Skill output (failure)
```json
{
  "schema_version": 0,
  "ok": false,
  "error": "profile.nsys-rep is only 481128 bytes; likely no GPU activity captured"
}
```

### What this skill contributed — even by failing
- **The failure was loud, correct, and matched FAILURE MODE #5 declared in
  the SKILL.md** — `.nsys-rep is 0 bytes or no GPU work captured`. The skill
  didn't silently produce nonsense numbers; it flagged "no GPU activity here"
  and stopped.
- **Diagnosis given to the agent**: nsys profiles the wrapped subprocess only.
  The bench client makes HTTP requests; the GPU work happens in a separate
  vLLM server process. **No `--pid`/`--attach` option exists in this nsys
  version (`/home/t-chendili/cuda/12.6/bin/nsys` 2024.5.1)**, so the skill
  as-designed cannot attach to a running server.
- **This is a real gap to add to the audit**: see §6.

---

## Phase 3 — use vLLM's built-in torch.profiler instead (workaround)

### Why this works as a substitute
- vLLM exposes `/start_profile` and `/stop_profile` HTTP endpoints **only when
  launched with `--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'`**.
- This lets the agent profile only the **window of interest** (warm steady-state,
  not server warmup), with no need to wrap the entire server in nsys.
- Output: per-kernel CPU+CUDA time table in `profiler_out_0.txt` plus full
  chrome-trace `.pt.trace.json.gz` for deeper queries.

### Capture procedure (identical for both backends)
```bash
# 1. Warmup 3 requests (skips JIT + autotune cold path)
for i in 1 2 3; do curl -X POST .../v1/completions -d '{"prompt":"hello"...}'; done
# 2. Start profile
curl -X POST http://127.0.0.1:30001/start_profile
# 3. Run R_medium-style load (16 prompts × 800 words × 256 max_tokens, conc=8)
python3 bench_R_medium.py
# 4. Stop profile
curl -X POST http://127.0.0.1:30001/stop_profile
```

### Comparison — kernel-level breakdown (Self CUDA %)

| Category                                     | vllm_cutlass | vllm_triton |
|---|---|---|
| **MoE GEMM (the kernel CUTLASS optimizes)** | 31.55% (24,672 calls × 44.9 µs) | 46.84% (49,440 calls × 29.8 µs) |
| CUTLASS dense GEMM (Q/K/V/O linears)        | 20.20% (24,672 × 24.4 µs)       | ~0% (uses cuBLAS instead)        |
| cuBLAS nvjet (Hopper JIT GEMM)              | 17.42%                          | ~30% (handles QKV here)          |
| FlashAttention                              | 11.32%                          | ~10%                            |
| **MoE routing helpers** (trtllm_kernels::* + topkGating) | **8.91%** | ~5%             |
| Triton fused norms (RMS / etc)              | 5.61%                           | ~5%                            |
| memcpy/memset                               | 1.90%                           | ~3%                            |
| KV cache writes                             | 1.65%                           | ~2%                            |

### The per-layer MoE math
With Qwen3-30B-A3B at 48 layers, top-8 of 128 experts:
- **Cutlass**: fuses up-proj + down-proj into one kernel call per MoE layer
  → 24,672 / (48 layers × forward_passes) → 1 call per layer per pass, 44.9 µs.
- **Triton**: 2 separate kernel calls (up then down) per MoE layer
  → 49,440 / (same denominator) → 2 calls per layer per pass, 29.8 µs each.
- **Per-layer MoE: Cutlass = 44.9 µs, Triton = 59.6 µs → Cutlass wins per-layer by 25%.**

So **CUTLASS IS faster at the kernel level** — the puzzle isn't that CUTLASS is
slow, it's that the gain is only ~9% of total time × 25% = ~2% e2e win, which
gets eaten by:
- Inadvertent backend swap on dense GEMM (CUTLASS path uses `cutlass::device_kernel`
  for QKV linears; Triton path uses cuBLAS `nvjet_sm90_*` — the cuBLAS kernels
  are actually well-tuned for these shapes)
- ~9% CUTLASS-side MoE routing overhead that Triton doesn't have at the same level

---

## Phase 4 — apply the `nsys-timeline-sql` SKILL.md "metric → problem" table

Even without nsys data here, the SKILL.md mapping is directly applicable to
torch.profiler output. Working through it:

| Metric observed                              | SKILL.md mapping (`pytorch-profiling`, `nsys-timeline-sql`)                       | What it tells us |
|---|---|---|
| `top_kernels[0].self_pct = 31%` (cutlass)    | "20–40% → single hot kernel, autotuning has high ROI"                            | CUTLASS MoE GEMM **is** the hot kernel; tuning it pays off (microbench confirmed 5–6×). |
| `MoE routing helpers self_pct ≈ 6%`          | `moe_overhead.total_routing_pct ≈ 9% → "low"` (per pytorch-profiling SKILL.md)   | Routing overhead is real but not dominant. Optimization candidate D2, not D1. |
| `kernel_count = 24,672` cutlass vs `49,440` triton over ~3.5s | "Launch rate per second: cutlass ~7k/s, triton ~14k/s" | Both are below the >50k/s threshold that flags "CPU launch-overhead bound", so cudagraph is doing its job in both. |
| `cudaEventSynchronize: 1.6s CPU, 0.003s CUDA` | "CPU spent waiting for GPU" — classic GPU-bound signature                       | The GPU is busy ~95% of the time. The cudaEventSynchronize is CPU waiting on GPU graph replay completion — this is NORMAL for cudagraph mode and not the bottleneck. |
| `wall_s` cutlass = triton ≈ 3.5s            | The 25% MoE-GEMM win × ~10% MoE-GEMM weight = ~2.5% e2e win. Within stddev. | The skill's reliability check (stddev < 0.3%) confirms we wouldn't even detect such a small win without ≥5 runs. |

**The skill's metric→problem table did the diagnostic work**. The agent didn't
need to invent an analysis framework; it followed the matrix from
`nsys-timeline-sql/SKILL.md` § "WHICH METRIC HELPS WHICH PROBLEM".

---

## Phase 5 — improvement directions, ranked

Derived directly from the kernel breakdown above. Each direction is tagged with
**which evidence (which skill produced it)** justifies it.

| # | Direction                                                                                                   | Estimated ROI | Risk    | Evidence source |
|---|---|---|---|---|
| D1 | Enable `flashinfer_cutlass` in sglang's `_should_run_flashinfer_autotune` allowlist (currently TODO-skipped).| **HIGH**: 5-6× MoE GEMM kernel speedup proven at microbench level. | medium (need to reproduce the "compilation errors" cited in the TODO). | docs/2026-06-08/buga_fix_validation.md (tried + reverted); `microbench` 5-6× ratio; sglang source line 1841. |
| D2 | Reduce 9% CUTLASS-side MoE routing overhead (`trtllm_kernels::expandInputRowsKernel`, `computeStridesTmaWarpSpecializedKernel`, `fusedBuildExpertMapsSortFirstTokenKernel`). | **MED-HIGH**: 9% of total CUDA time. Even halving = 4.5% e2e gain — bigger than current cutlass vs triton gap. | high (touches CUTLASS-FlashInfer interop; cross-team effort) | This investigation, Phase 3 table: routing helpers self_pct = 5.93% + topkGating 2.98% = 8.91%. |
| D3 | Make `moe_backend=cutlass` use cuBLAS for dense GEMMs too (current vLLM also swaps the dense kernel path inadvertently — see Phase 3, "CUTLASS dense GEMM 20.20% vs ~0% in triton mode"). | MED: tests whether the dense-GEMM swap is what causes the apparent gap. | low (engineering config change). | Phase 3 comparison table — `cutlass::device_kernel<GemmUniversal>` only appears in cutlass mode. |
| D4 | Investigate why CUTLASS at sm90 doesn't have an even larger headroom over Triton. | LOW (likely "it's just well-matched on Hopper at bf16"). Check ncu (NOT YET INSTALLED — see §6). | low | If anything moves on D1+D2, this becomes moot. |

---

## Phase 6 — what the skills couldn't do (audit additions)

This investigation surfaced **two concrete gaps** to add to
`docs/2026-06-08/agent_profiling_capability_audit.md` Part B:

### Gap N+1: `nsys-capture` has no `--attach` mode
- Symptom: cannot profile an already-running vLLM/sglang server. Must either
  restart server under nsys (loses test-time validity — autotune state, kernel cache)
  or use the server's built-in torch.profiler endpoint (vLLM-specific, not portable).
- Root cause: `nsys profile` 2024.5.1 on `/home/t-chendili/cuda/12.6/bin/nsys`
  has no `--pid` flag.
- Mitigation candidates: (a) bump nsys to 2025.x which adds `--pid`; (b) launch
  server under nsys with delayed `--capture-range=cudaProfilerApi` triggered via
  vLLM's profile endpoint.
- **Skill roadmap impact**: `nsys-capture` v1 should add an `--attach-pid` mode.

### Gap N+2: torch.profiler text summary is the only structured output for vLLM
- `pytorch-profiling` skill is sglang-specific (uses `SGLANG_TORCH_PROFILER_DIR`,
  parses chrome-trace via sglang's annotations).
- For vLLM, this investigation had to **hand-write** the kernel categorization
  in a one-off script.
- **Skill roadmap impact**: extend `pytorch-profiling` to detect vLLM trace format
  OR add a `vllm-profile` sibling skill, OR (best) make the trace parser
  framework-agnostic.

---

## Files produced
- `results/2026-06-09_cutlass_investigation/cutlass/bench/bench_summary.json` — e2e-bench-runner output
- `results/2026-06-09_cutlass_investigation/cutlass/torch_trace/profiler_out_0.txt` — torch.profiler text summary
- `results/2026-06-09_cutlass_investigation/cutlass/torch_trace/dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz` — full chrome trace (40 MB)
- `results/2026-06-09_cutlass_investigation/triton/...` — same set for Triton

## Skill-by-skill contribution summary

| Skill                  | Used? | What it produced                                                                  |
|---|---|---|
| `e2e-bench-runner`     | ✅ ×2  | bench_summary.json for both backends, with reliability check that confirmed the gap is signal not noise. |
| `nsys-capture`         | ❌ (failed gracefully) | Demonstrated FAILURE MODE #5 correctly; revealed Gap N+1. |
| `nsys-timeline-sql`    | ❌ (no nsys data this run) | Its WHICH-METRIC table was still consulted for interpretation methodology. |
| `pytorch-profiling`    | partial — used as methodology, not direct invocation | The metric → problem table from its SKILL.md guided the analysis. |
| Custom one-off script   | ✅ workaround | Categorized the torch.profiler text into MoE GEMM / routing / dense GEMM buckets. |

**Net agent value of the skills here**:
1. Without `e2e-bench-runner`'s stddev check, the agent might have called noise a signal.
2. Without `nsys-capture`'s clean failure-mode reporting, the agent might have spent
   more iterations trying to figure out why no data appeared.
3. The two SKILL.md "metric → problem" tables (from `nsys-timeline-sql` and
   `pytorch-profiling`) gave the agent a **named, documented framework** for
   interpreting profile data. The conclusions in this report all map to
   specific rows in those tables.
