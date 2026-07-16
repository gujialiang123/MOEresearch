# MoE Dynamic-K Research — Plan & Status

Branch: `moe-optimization`. Two orthogonal lines (kept separate on purpose):
- **系统线 (system)**: does reducing K speed up real kernel/TPOT/E2E? (needs sglang fused — deferred)
- **行为/机理线 (behavior/mechanism)**: WHY does changing K change generation length? (current focus; does NOT depend on real speedup)

## Status

### ✅ Done & verified
- **P0 fixes** (`scripts/dynamic_topk_utils.py`): physical expert skip, sync-free GPU counters,
  3 named policies (top_p_within_topk / min_weight_cutoff / max_dropped_mass), strict `####`
  parser, prefill/decode K split, 3 renorm modes.
- **Unit tests** (`tests/test_dynamic_topk.py`): 9/9 pass (toy call-counter proves dropped
  experts not executed; keep-all equivalence; monotonicity; kmin; no-sync; strict parser).
- **Real-model equivalence** (`run_v20_dynamic_topk_equivalence.py`): keep-all == native at
  EXACTLY 0 error (MoE out + logits), greedy generation token-identical; dyn τ=0.7 → avg_k_decode=4.94.
- **Errata** added to v17 & v18 docs; **validation doc** `docs/2026-07-16/v20_dynamic_topk_validation.md`.

### 🔄 Running
- **v21 K-vs-length dose** (`run_v20_dynamic_topk_free_generation.py`): full GSM8K 1319,
  K∈{4,6,8,10,12} (8=native baseline; 10,12 flagged OOD super-native), phase=all, max_new=512.
  Saves FULL token ids per sample → any metric recomputable offline (no rerun).
  Out: `results/2026-07-16_v21_k_vs_length/`. ~40min/config, ~3.3h total on GPU6.

### ⏳ Next (ready)
- **v21 analysis** (`analyze_v21_k_vs_length.py`, ready): dose curve + L_to_answer/L_post_answer
  decomposition + no_hash trend + repetition + paired Δlen bootstrap CI. Pure log analysis.
- **v22 teacher-forced** (`run_v22_teacher_forced_eos.py`, ready): separate DIRECT termination
  effect vs trajectory-mediated effect. Teacher-force same baseline(K=8) seq under each K, read
  logp(EOS)/margin/KL/ΔNLL in the termination zone (last W tokens before baseline EOS).

## Key question being answered
Extra length from lower K: is it **L_to_answer↑** (reasoning-compute substitution) or
**L_post_answer↑ / no-#### ↑** (termination/format mechanism)? Small run (16q) hinted BOTH.
v21 (dose + decomposition) + v22 (causal teacher-forcing) will resolve it.

## Constraints / notes
- GPU: **only GPU6** usable (7 forbidden). sglang env: `/home/t-jialianggu/.conda/envs/sglang`.
- HF cache must be writable: `HF_HOME=$PWD/.hf_cache` (/data/hf/hub read-only).
- Length study: do NOT reduce max_new (would censor the length being measured).
- K>8 is OUT-OF-DISTRIBUTION (native top-8); reported separately, not on the natural dose curve.
- GPT-suggested papers (TERMINATOR/ESTAR, arXiv 2603/2604.*) unverifiable (future-dated) — treat as leads only.
