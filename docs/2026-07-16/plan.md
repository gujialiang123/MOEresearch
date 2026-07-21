# MoE Dynamic-K Research — Status (2026-07-20 overnight COMPLETE)

Repo: MOEresearch. Behavior/mechanism line (why K changes generation length).

## 2026-07-20 batch 2 — v29–v32 rigorous kill-tests (IN PROGRESS)
Report: docs/2026-07-20/v29_v32_experiment_report.md ; audit: v29_v32_preflight_audit.md
- P0: KPolicy +decode_norm_match/+position_bin_gain; decode_norm_calib.py (same-hidden
  dual-branch DECODE norm); interventions.py; stats KM/RMST/cluster-bootstrap; 21/21 tests.
- v29 decode-norm control: calib n=128 (decode s K4=1.116 > prefill 1.082); realized ratio
  decode 0.998 / prefill 0.96 (prefill under-corrects). Smoke: full_renorm +20.5 ->
  decode_normmatch +8.1 (~60% removed). PILOT n=200 running (GPU3).
- v30 partial-renorm dose: smoke K4 monotone in beta (235->252). PILOT running (GPU4).
- v31 gain controls: smoke fixed_layer_gain +27 ~ full +20.5; shuffled +37.8 (larger);
  clip no reduction => AVERAGE magnitude, not correspondence/tail. PILOT running (GPU6).
- v32 pulse-recovery: full-grid smoke running (GPU2, ~25min/prompt). analyze_v32 handles partial.
Next: finalize report §5-8 with pilot CIs; optional test[0:500] confirmatory for cleanest v29 arm.

---
(previous batch below)

## COMPLETE — all experiments done, committed, pushed to origin/main
Full log: docs/2026-07-20/overnight_experiments.md

- Unified moe_research/k_policy.py (prefill_k/decode_k, cache-state phase, physical
  skip, 4 weight modes) + tests 8/8 + real-model equivalence (0 err, phase routing OK).
- v23 phase factorial (n=500): length effect is DECODE-phase (decode +6.7/K6,+28/K4 sig;
  prefill ~+3 ns; additive). full-test n=1319 confirms (K6 +6.7, K4 +29.2).
- v24 + mode D: HEADLINE — length effect is a renorm PER-TOKEN reweighting artifact,
  not expert count (K4: renorm +28, no_renorm +3.8, calibrated +7.7, fold +144).
- v25 answer-readiness: t_ready +5 vs t_marker +17 (delayed commitment, not more reasoning).
- v26 current-step direct effect: small (KL<=0.06, top1 96-98%) -> trajectory-mediated.
- v28 decode dose (renorm): convex Δlen 2/5/8/13 as K 8->4.
- v28b decode dose (no_renorm, n=500): FLAT 250-255 -> decisive contrast.

## Headline (reframes v21)
"Lower K -> longer generation" is largely a WEIGHT-RENORMALIZATION artifact (renorm's
per-token 1/Σw upscaling), NOT intrinsic to reducing experts, and is DECODE-phase.
= "Scale-Preserving Expert Sparsification" branch of the decision tree.

## Suggested next (morning)
- Tail-restoration probe (α 0->1) as continuous mechanistic confirmation.
- Cross-model (Qwen1.5-MoE, Phi-3.5-MoE) + cross-task (MATH-500, BoolQ) generalization.
- Real sglang fused-kernel latency/TPOT (system line, separate).

## Constraints
- Future runs: use GPU2/GPU3 (per user 2026-07-20). Current v29/v31 finishing on GPU1/4/6. env /home/t-jialianggu/.conda/envs/sglang. HF cache
  HF_HOME=/home/t-jialianggu/work/EndtoEnd-auto-optimization/.hf_cache.
- Length study: never reduce max_new. K>8 is OOD, report separately.
- All raw logs keep full token ids -> any new metric recomputable offline, no rerun.
