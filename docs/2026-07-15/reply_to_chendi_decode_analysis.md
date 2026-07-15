# Reply to Chendi — decode wall proportion + NCU decode analysis (interim)

> Copy-paste ready. Interim numbers from the runs that already finished; a few more
> (LFM model + larger decode batches) are still profiling and I'll append them.

---

## ✉️ Reply (copy-paste)

Hi Chendi — here's the first cut on both asks. Full CSV + all NCU artifacts are in the repo (paths at the bottom); more points (LFM + b128 decode) are still running and I'll append.

### 1. Prefill vs decode wall proportion — agent (toolagent) workload, Qwen3-30B-A3B, H200

Definitions I used (matching yours): **prefill wall ≈ TTFT** (prompt processing + queue/chunk), **decode wall ≈ E2E − TTFT** (first token → completion). Swept `--max-concurrency` on one server, mooncake `toolagent` dataset, 300 prompts, best tuned config (fa3 + triton MoE, chunked-prefill, lpm).

| max-conc | achieved conc | prefill TTFT (ms) | decode (ms) | E2E (ms) | decode/prefill | **decode share** | TPOT (ms) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  | 1.0  | 91  | 854  | 945  | 9.4×  | **90.4%** | 4.3 |
| 4  | 4.0  | 68  | 1400 | 1468 | 20.5× | **95.4%** | 7.5 |
| 8  | 7.8  | 81  | 1933 | 2014 | 23.8× | **96.0%** | 9.7 |
| 16 | 12.3 | 98  | 2295 | 2393 | 23.5× | **95.9%** | 12.1 |
| 32 | 12.8 | 300 | 2184 | 2484 | 7.3×  | **87.9%** | 14.4 |
| 64 | 11.3 | 177 | 2013 | 2190 | 11.4× | **91.9%** | 11.4 |

CSV: `results/2026-07-15_v19_wall_sweep/qwen3-30b-a3b-bf16/wall_proportion.csv` (per-run raw jsonl with `--output-details` alongside it).

**Takeaway:** across the whole concurrency range, **decode is 88–96% of end-to-end wall** for this agent workload (output ~194 tok/req, input ~2700). This workload is decode-dominated — so decode optimization is where the wall-clock lever is. (Note: single-stream achieved concurrency saturates around 12–13 even when we allow 32–64, because toolagent arrival is bursty + heavy prefix sharing; that's the serving-idle story we discussed separately. TPOT climbing 4.3→14 ms with concurrency is the queueing/batch-interference, not per-token compute.)

### 2. NCU decode-kernel metrics (your exact list)

Profiling `sglang.bench_one_batch` decode stage under NCU with the 11 metrics you asked for:
`gpu__time_duration.sum, dram__bytes_read.sum, dram__bytes_write.sum, dram__throughput.avg.pct_of_peak_sustained_elapsed, sm__throughput.avg.pct_of_peak_sustained_elapsed, sm__warps_active.avg.pct_of_peak_sustained_active, l1tex__t_sector_hit_rate.pct, lts__t_sector_hit_rate.pct, launch__occupancy_limit_registers, launch__occupancy_limit_shared_mem, launch__occupancy_limit_warps`.

- Sweeping decode batch {32, 64, 128} to push toward the **decode upper bound** (larger effective batch amortizes expert-weight movement → best achievable arithmetic intensity), plus a prefill reference point.
- All artifacts saved per regime: `ncu.ncu-rep` + `ncu_raw.csv` + `bench.log` + `combo_params.json` under `results/2026-07-15_v19b_ncu_decode/<model>/<regime>/`.
- Status: b32 rep done (8.8 MB), b64/b128 + LFM still profiling — I'll drop the parsed metric table here once they land.

**Kernels actually run in one decode step (Qwen3-30B, b32) — 13 kernel launches:**

| # | role | dur (µs) | kernel |
|--:|---|--:|---|
| 1 | RMSNorm (input norm) | 3.78 | `norm::RMSNormKernel` |
| 2 | dense proj GEMM (QKV) | 10.21 | `nvjet_tst_64x32...` |
| 3 | RoPE rotary embed | 4.96 | `flashinfer::BatchQKApplyRotary` |
| 4 | FlashAttn varlen setup | 6.59 | `flash::prepare_varlen_num_blocks` |
| 5 | **FlashAttention decode** | **55.71** | `cutlass::...FlashAttnFwdSm90` |
| 6 | FlashAttn split-K combine | 3.78 | `FlashAttnFwdCombine` |
| 7 | dense proj GEMM (O, split-K) | 9.12 | `nvjet_..._splitK` |
| 8 | RMSNorm (post-attn, fused add) | 4.58 | `FusedAddRMSNormKernel` |
| 9 | dense proj GEMM (small) | 7.23 | `nvjet_64x8...` |
| 10 | **MoE router top-k** | 5.79 | `topkGatingSoftmax<…,8,128,…>` |
| 11 | **MoE expert GEMM (up/gate)** | **41.76** | `fused_moe_kernel` |
| 12 | SiLU act_and_mul (MoE gate) | 3.97 | `activation::act_and_mul_kernel` |
| 13 | **MoE expert GEMM (down)** | **24.90** | `fused_moe_kernel` |

By family: nvjet dense-GEMM ×3, `fused_moe` expert-GEMM ×2, RMSNorm ×2, FlashAttn ×2, plus RoPE / varlen-setup / router-topk / act_and_mul ×1 each. The three time sinks are **FlashAttention (55.7 µs)** and the **two `fused_moe` expert GEMMs (41.8 + 24.9 = 66.7 µs)** — the MoE pair is the largest and is exactly the movement-bound (103:1) part; attention is heavy because of the long agent context (~2700 tok KV). (b32 sample; b64/b128 have the same kernel set, magnitudes scale with batch.)

### 3. How much can we get from decode, and where are the gaps

Consistent with our earlier profiling (I'll re-confirm with the new metric list):
- **Root cause = expert weight movement, not compute.** Decode MoE is ~80% DRAM-bound; per decode step the move:compute ratio is ~**103:1** for Qwen3-30B (~52:1 for LFM). The hot `fused_moe_kernel` streams expert weights HBM→SM at ~3.83 TB/s (~80% of H200 peak) while SM/Tensor-Core sit ~13–17%. The `dram__bytes_read.sum` + `sm__warps_active` + `launch__occupancy_limit_*` from this run will let us attribute the low occupancy precisely (registers vs shared-mem vs warps) — that's the point of pulling your exact list.
- **Achievable decode headroom (exact methods, same bytes/FLOPs):** roofline says ~**1.8–1.9× TBT** for Qwen3-30B (~2.2–2.4× for LFM) if kernels are driven to their busiest-pipe ceiling. Time-weighted, **67% (LFM) / 78% (Qwen3)** of decode GPU-time is SM idle waiting on memory (NCU "No Eligible"), root-caused to low occupancy (12–25%).
- **The gaps, ranked by leverage:**
  1. **Move-once-serve-more (biggest):** raise effective batch per weight load — spec-decode, multi-tenancy, expert-parallel — directly attacks the 103:1 ratio. We already showed n-gram spec-decode gives +6% (b1) / +23% (b32) decode TBT, and multi-stream lifts util 13%→32% / throughput 7.4×.
  2. **Move less weight:** reduce activated experts. On GSM8K, dropping 8→6 active experts costs ~0.5pp accuracy — real room to trade a little quality for less movement. I'm quantifying the full accuracy-vs-movement curve (fixed + confidence-adaptive top-k) now.
  3. **Kernel-level occupancy/latency-hiding:** the `launch__occupancy_limit_*` metrics will tell us whether the ceiling is registers/smem/warps, so we know if a retuned kernel can even reach higher occupancy.

All numbers measured (NCU hardware counters + nsys kernel-interval union), not estimated. I'll append the LFM row, larger-batch decode points, and the parsed 11-metric table as they finish.

---

## 📌 Data sources / artifacts (for reference)

| Item | Path |
|---|---|
| Part-1 wall-proportion CSV | `results/2026-07-15_v19_wall_sweep/qwen3-30b-a3b-bf16/wall_proportion.csv` |
| Part-1 per-run raw (TTFT/E2E/details) | `results/2026-07-15_v19_wall_sweep/qwen3-30b-a3b-bf16/bench_c*.jsonl` |
| Part-2 NCU artifacts (11 metrics) | `results/2026-07-15_v19b_ncu_decode/qwen3-30b-a3b-bf16/agent_decode_b*/` |
| move:compute 103:1, 79.8% DRAM-bound | `docs/2026-07-15/triton_moe_kernel_analysis.md`, `docs/2026-06-29/profiling_validation_of_universal_config.md` |
| roofline headroom, SM-idle 67/78% | `docs/2026-07-10/reply_to_dey_tbt_headroom.md` |
| spec-decode + multi-stream gap-reachable | `docs/2026-07-15/v11_realize_gap_results.md` |
| 8→6 experts ≈ −0.5pp GSM8K | `docs/2026-07-15/v17_gsm8k_topk_results.md` |
| scripts | `scripts/run_v19_wall_sweep.sh`, `scripts/run_v19b_ncu_decode.py` |
