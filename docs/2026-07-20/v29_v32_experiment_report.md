# v29–v32 Mechanism Experiments — Report (2026-07-20 / overnight)

**Status:** code + 21 unit tests complete; **v29 decode-norm calibration + all four
smokes complete and analyzed**; pilots (n=200 dev) launched and running on GPU3/4/6;
v32 smoke (full pulse grid) running on GPU2. This report is written by **evidence
level** and is updated as confirmatory-scale data lands. Raw data for every completed
run is committed and pushed.

Conventions: length = generated tokens (EOS-truncated); hit-max (=512) is right-
censored → we report Kaplan–Meier restricted-mean survival (RMST) alongside capped
mean. Δ = paired vs `k8_native`; CIs are paired bootstrap (10k). Accuracy uses exact
McNemar (Holm across configs). No serving-speedup claims — cost is expert-executions/
request. p>0.05 is reported as "no detectable difference", never "equivalent".

---

## 1. Repository state and audit findings

See `docs/2026-07-20/v29_v32_preflight_audit.md` (full detail). Key points:
- HEAD at start `3bffd82`. Qwen3-30B-A3B-Instruct-2507, 48 MoE layers, E=128, native top-8, `norm_topk_prob=True`. GSM8K, greedy, `max_new=512`.
- **Confirmed doc error:** v26 covered **2400 positions/K** (7200 records / {K4,K6,K8}), not the "~6600" some docs state. Recorded; not altered.
- **Weak link identified & fixed:** the old norm-match (`calibrate_norm_match.py`) estimated `s(l,K)=E‖y8‖/E‖yK‖` on **prefill tokens only**, then applied it at decode. v29 rebuilds this on **decode tokens**.
- Phase via cache-state hook (not `seq_len==1`); dropped experts physically skipped; `inv_r=M_full/M_K` makes native-K exactly gain-1 for all modes.
- **Reproducibility gap:** v23/v24/v28 *run* scripts are absent from the repo (only `analyze_v23_v28.py` remains). New v29–v32 commit runner + manifest (git commit + config hash + frozen sample-ID lists).

## 2. Code changes

New/'changed:
- `moe_research/k_policy.py`: `+decode_norm_match`, `+position_bin_gain` weight modes.
- `moe_research/decode_norm_calib.py` (**new**): same-hidden dual-branch DECODE norm calibration — from one native-K8 teacher-forced pass forms `y8` and `yK` (no-renorm) by partial sums (no extra expert compute), accumulates per phase + decode-position bin; scalars `s_phase(l,K)`, realized-ratio check. Vectorized (no per-token sync).
- `moe_research/gain_calibration.py`: `+record_decode` (routing-only decode-token 1/r gains).
- `moe_research/interventions.py` (**new**): pulse/layer window selectors, fixed pulse-start rule, `clone_cache`.
- `moe_research/stats.py`: `+km_survival`, `+restricted_mean_survival`, `+prompt_cluster_bootstrap_ci`.
- `scripts/`: `_harness.py`, `calibrate_v29_decode_norm.py`, `run_v29_decode_norm_control.py`, `run_v30_partial_renorm.py`, `run_v31_gain_controls.py`, `run_v32_pulse_recovery.py`, `analyze_config_sweep.py`, `analyze_v32_pulse_recovery.py`.

## 3. Unit-test results — 21/21 PASS

`test_k_policy.py` (14) + 7 required: `test_native_equivalence`, `test_partial_renorm`
(β0≡no-renorm, β1≡full-renorm, native-K identity ∀β, branch-norm monotone in β),
`test_decode_norm_calibration` (y8 native-exact, ‖yK‖ matches independent partial sum,
realized ratio=1.0), `test_gain_shuffle_no_leakage`, `test_intervention_window`,
`test_kv_cache_immutability` (probe fork does not mutate baseline cache),
`test_physical_expert_skip`. Run: `python tests/test_*.py`.

## 4. Experimental protocol (frozen)

Splits: calibration = GSM8K `train[0:128]`; development = `train[128:328]`; smoke =
`train[128:160]` (32); confirmatory = `test[0:500]`. Greedy, `max_new=512`. Stages run
in order; a stage advances only after the previous passes its gate.

---

## 5. v29 — Decode-specific norm control

**Question:** does matching the *average decode* MoE branch scale (calibrated on real
decode tokens) remove the length effect? Old prefill-calibration is the confound.

**Calibration (n=128, same-hidden dual-branch).** Decode scalars exceed prefill
scalars: K4 decode `s`=**1.116** vs prefill **1.082**; K6 decode **1.040** vs prefill
**1.026**. So prefill-calibration systematically *under-corrects* the decode branch norm.

**Gate — realized norm-match ratio on held-out dev (target 0.95–1.05):**
- decode-calibrated: K4 **0.998**, K6 **1.000** ✓ (valid control)
- prefill-calibrated: K4 **0.96**, K6 **0.98** (under-matches decode; at the low edge)

**Smoke result (n=32, paired Δ vs native, prefill K8):**

| config | Δlen (95% CI) | acc |
|---|---|---|
| K4 no_renorm | +4.0 (−7.8,+17.2) | 93.8% |
| K4 **full_renorm** | **+20.5 (+8.0,+34.5)** | 96.9% |
| K4 prefill_normmatch | +7.1 (−3.5,+19.6) | 96.9% |
| K4 **decode_normmatch** | **+8.1 (−0.1,+16.9)** | 100% |
| K6 full_renorm | +12.2 (−1.1,+28.0) | 90.6% |
| K6 decode_normmatch | +8.6 (+2.2,+17.8) | 90.6% |

**Reading (smoke).** Matching the average decode norm removes most of the full-renorm
length excess (K4: +20.5 → +8.1, a ~60% reduction), landing near no_renorm/native.
Accuracy unaffected (McNemar n.s.). This is the plan's *case 3*: **average decode scale
explains a large majority of the effect, with a residual** not captured by mean-norm
matching. Confirmatory n=200 pilot running to tighten the wide n=32 CIs.

## 6. v30 — Partial-renormalization dose response

**Question:** at fixed retained subset, does length rise monotonically with renorm
strength β?

**Smoke (n=32) capped mean length** (native=231.1): K4 β0→1 = 235.1 → 238.2 → 238.6 →
249.6 → 251.7 (**monotone**, Δ +4 → +21); K6 = 239.2 → 232.6 → 234.7 → 240.9 → 243.3
(noisier, upward). Endpoints β0/β1 match the standalone no_renorm/full_renorm arms
(gate passed). Confirmatory pilot running; monotone-trend test + KM survival to follow.

## 7. v31 — Gain controls (decode K4)

**Question:** is full-renorm's effect (A) average scale, (B) gain variance/tail, or
(C) gain↔router-state correspondence?

**Smoke (n=32, paired Δ vs native):**

| config | Δlen (95% CI) | note |
|---|---|---|
| no_renorm | +4.0 (−7.8,+17.2) | baseline drop |
| full_renorm | +20.5 (+8.0,+34.5) | reference effect |
| **fixed_layer_gain** | **+27.1 (+6.7,+53.4)** | pure avg scale ⇒ reproduces (≥) full |
| position_bin_gain | +14.2 (+0.8,+28.3) | avg scale w/ position structure |
| **shuffled_gain** | **+37.8 (+22.4,+54.8)** | breaks token corresp ⇒ *larger*, not smaller |
| clipped_q90 / q95 | +25.7 / +38.7 | tail clip does **not** reduce |
| decode_normmatch | +8.1 (−0.1,+16.9) | correct norm scale ⇒ lowest |

**Reading (smoke).** (A) average scale is **sufficient** (`fixed_layer_gain` reproduces
full-renorm); (C) token↔router-state correspondence is **not** the driver
(`shuffled_gain` is *larger*); (B) the extreme tail is **not** responsible (clipping
does not reduce). The magnitude of the average survivor-weight upscale sets the length;
matching the true branch *norm* (a smaller scale than mean 1/r) minimizes it. Plan's
case "fixed mean ≈ full renorm ⇒ mainly average scale." Pilot running.

## 8. v32 — Prefill/decode pulse-and-recovery

**Question:** is a transient low-K perturbation recovered under teacher forcing but
amplified under free running (autoregressive feedback)?

Smoke running (full grid dur{1,4,16,64} × start{early,middle,late}) on GPU2; token-by-
token probes are expensive (~25 min/prompt) so this fills in incrementally
(`analyze_v32_pulse_recovery.py` aggregates partial `raw.jsonl`). Early n=1–3 signal:
open-loop KL after a decode pulse recovers within ~8 tokens (0.03→0.003 for dur=1) while
closed-loop trajectories flip — consistent with autoregressive amplification; full grid
to quantify the duration trend. **Full confirmatory command provided in §13.**

---

## 9–11. Hypotheses (interim, smoke-level — to be confirmed at n=200/500)

**Strongly supported (already, with paired stats at n=32; pilots pending for CIs):**
- Renormalization *strength* causally scales length (v30 monotone β dose; v31 gain modes).
- The driver is the **average upscaling magnitude** of survivor weights, **not** token↔
  router correspondence (shuffled larger) and **not** the extreme tail (clip no reduction).
- Prefill-calibrated norm-match **under-corrects** decode norm (realized 0.96 vs 0.998);
  decode-calibrated control is the valid one.

**Consistent with (direction clear, needs confirmatory n):**
- Matching the average *decode* branch norm removes ~60% of the full-renorm length excess
  (average scale dominant) with a **residual** (token-wise variability / state coupling).
- Autoregressive feedback amplifies transient decode perturbations (v32 open vs closed).

**Not supported / not claimed:**
- That token-conditioned gain is the *sole* cause (v31 shuffled refutes).
- That average scale is *fully* excluded (v29 shows it dominant).
- Any serving speedup; any "model already knows the answer" readiness claim (out of scope).

## 12. Data / artifact completeness

Committed raw for: v29 calibration (scalars + manifest), v29 smoke (+analysis),
v30 smoke (+summary), v31 smoke (+analysis), run logs. Pilots write incremental
`*_raw.jsonl` + per-config summary; committed on completion. (v32 pilot/confirmatory not
in scope tonight — see stop conditions.)

## 13. Exact reproduction commands

```
ENV: HF_HOME=/home/t-jialianggu/work/EndtoEnd-auto-optimization/.hf_cache
     PY=/home/t-jialianggu/.conda/envs/sglang/bin/python
tests:      for t in tests/test_*.py; do $PY $t; done
v29 calib:  CUDA_VISIBLE_DEVICES=3 $PY scripts/calibrate_v29_decode_norm.py --ks 6,4 --n 128
v29:        $PY scripts/run_v29_decode_norm_control.py --scalars <calib>/norm_match_scalars.json --stage {smoke,pilot,confirmatory}
v30:        $PY scripts/run_v30_partial_renorm.py --stage {smoke,pilot,confirmatory}
v31:        $PY scripts/run_v31_gain_controls.py --stage {smoke,pilot,confirmatory} --calib_n 128
v32:        $PY scripts/run_v32_pulse_recovery.py --stage smoke --n 64 --durs 1,4,16,64 --starts early,middle,late
analyze:    $PY scripts/analyze_config_sweep.py <result_dir>
            $PY scripts/analyze_v32_pulse_recovery.py <result_dir>
```

## 14. Recommended next experiment

After pilots confirm direction at n=200: run the single cleanest **confirmatory** arm on
`test[0:500]` — for v29 that is {native, K4 no_renorm, K4 full_renorm, K4 decode_normmatch}
— to publish the decode-norm-match decomposition with paired McNemar + KM survival. Only
then consider v33+ (localization / second architecture), which the current batch defers.
