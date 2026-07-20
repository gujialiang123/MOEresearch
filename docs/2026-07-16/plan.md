# MoE Dynamic-K Research — Plan & Status (updated 2026-07-20)

Repo: MOEresearch. Behavior/mechanism line (why K changes generation length).

## Done (2026-07-20 overnight)
- Unified `moe_research/k_policy.py` (prefill_k/decode_k, cache-state phase, physical
  skip, 4 weight modes) + tests 8/8 + real-model equivalence (0 err, phase routing OK).
- v23 phase factorial (decode arm done; prefill/both arms finishing).
- v24 weight ablation + mode D (calibrated_norm_match): KEY FINDING — length effect is
  a renorm PER-TOKEN reweighting artifact, not expert count (K4: renorm +28, no_renorm
  +3.8, calibrated +7.7, fold +144).
- v25 answer-readiness: t_ready +5 vs t_marker +17 (delayed commitment, not more reasoning).
- v26 current-step direct effect: small per-step (KL<=0.06, top1 96-98%) -> trajectory-mediated.
- v28 decode dose (renorm): convex Δlen 2/5/8/13 as K 8->4.
- v28b decode dose (no_renorm, n=500): running — expected flat (confirms v24).

## Headline reframing
The "lower K -> longer generation" phenomenon (v21) is largely a **weight-renormalization
artifact** (renorm's per-token 1/Σw upscaling), NOT intrinsic to reducing experts. Under
no_renorm / norm-matched the effect nearly vanishes; under fold it explodes. This is the
"Scale-Preserving Expert Sparsification" branch of the decision tree.

## Next (morning)
- Merge v23 dirs, run full factorial (prefill vs decode effect + interaction).
- Finalize overnight_experiments.md synthesis.
- Consider: no_renorm dose figure vs renorm; tail-restoration probe; sglang real-latency.

## Constraints
- GPUs 4-7 usable. env `/home/t-jialianggu/.conda/envs/sglang`. HF cache
  `HF_HOME=/home/t-jialianggu/work/EndtoEnd-auto-optimization/.hf_cache`.
- Length study: never reduce max_new (censoring). K>8 is OOD, report separately.
