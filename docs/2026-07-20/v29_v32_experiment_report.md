# v29–v32 Mechanism Experiments — Report (2026-07-20 / overnight)

**Status:** COMPLETE for this batch. Code + 21 unit tests; v29 decode-norm calibration;
**all four experiments run at smoke (n=32) AND pilot (n=200 dev) scale**; v32 full pulse
grid smoke (n=12). All raw data committed and pushed. Confirmatory test[0:500] deferred
(see §11/§14). Numbers below are the **n=200 pilot** unless marked smoke.

GPU note: run on GPU3/4/6 (+GPU2 for the v32 smoke), the free devices at run time;
all shared with other users (util ~40–55%), so wall times are diagnostic only.

**One-paragraph synthesis.** Reducing MoE active experts K makes generation longer, and
this batch pins down the causal chain and rules out the main confounds. The effect is
controlled by the **strength** of survivor renormalization, monotonically (v30). Its
mechanism is the **average magnitude** of the survivor-weight upscale — a per-layer mean
gain reproduces it, breaking the gain↔token correspondence makes it *larger* (while
wrecking accuracy), and clipping the tail does nothing (v31). Rebuilding the norm-match
control properly on **decode** tokens (the old one used prefill tokens and measurably
under-matched, 0.96 vs 0.998) shows that matching the average decode branch scale removes
~73% of the length excess, leaving a ~25–30% token-wise residual (v29). Finally, the
reason a *local* decode perturbation changes the *whole* trajectory length is
**autoregressive feedback**: teacher-forced drift recovers within ~8 tokens, but free-
running flips the trajectory with probability rising monotonically in pulse duration,
0.42→1.0 (v32). Net: this is a **renorm-mediated, scale-driven, autoregressively-amplified
generation-length effect**, not an expert–token semantic-substitution effect.

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

**Smoke→Pilot.** The n=32 smoke (wide CIs) and n=200 pilot agree. Pilot (paired Δ vs
native, prefill K8):

| config | Δlen (95% CI) | acc | McNemar p |
|---|---|---|---|
| K4 no_renorm | +4.3 (−2.6,+11.0) | 87.5% | 0.096 |
| K4 **full_renorm** | **+22.7 (+16.3,+29.5)** | 87.0% | 0.210 |
| K4 prefill_normmatch | +8.8 (+2.4,+15.4) | 89.0% | 0.019 |
| K4 **decode_normmatch** | **+9.2 (+3.6,+14.9)** | 89.5% | 0.008 |
| K6 no_renorm | −1.2 (−6.4,+3.5) | 87.0% | 0.065 |
| K6 full_renorm | +5.6 (+1.5,+10.0) | 87.5% | 0.077 |
| K6 prefill_normmatch | +3.6 (−1.6,+8.6) | 86.5% | 0.109 |
| K6 decode_normmatch | +2.8 (−2.3,+7.3) | 86.0% | 0.180 |

**Gate — realized norm-match ratio on held-out dev (target 0.95–1.05):** decode-
calibrated K4 **0.998** / K6 **1.000** (valid control); prefill-calibrated K4 **0.96** /
K6 **0.98** (under-matches decode). Gate PASSED.

**Reading (pilot).** For the strong K4 arm: matching the average decode branch norm
takes full_renorm's **+22.7 down to +9.2** — removing **≈73%** of the excess over
no_renorm (18.4 → 4.9). So **the average decode branch scale is the dominant driver**.
BUT a residual (**≈+5 over no_renorm**, ~25–30% of the effect) is **not** removed by
average-norm matching → attributable to token-wise variability / state coupling, i.e.
the full token-wise renorm does something average matching cannot. Notably,
**decode_normmatch (+9.2) and prefill_normmatch (+8.8) show no detectable length
difference** despite the realized-ratio gap (0.998 vs 0.96): the extra ~4% average-norm
correction does not translate into length. Accuracy is preserved (~87–90%); the
significant McNemar for the norm-match arms reflects paired label churn, not a net drop.

## 6. v30 — Partial-renormalization dose response

**Question:** at fixed retained subset, does length rise monotonically with renorm
strength β?

**Pilot (n=200) capped-mean length** (native=253.1). K4 β 0→1: 260.4 → 263.8 → 263.1 →
269.9 → 276.1. Paired Δ vs native: **+7.3 → +10.7 → +10.0 → +16.9 → +23.0** — monotone
increasing (one flat step at β=0.5), high-β CIs strictly positive (β=1: +23.0 [+16.8,
+29.6]). K6 β 0→1 Δ: +0.7 → +1.7 → +4.3 → +3.3 → +7.3 (weaker, upward, some non-
monotonicity mid-range). Endpoints match the standalone no_renorm/full_renorm arms
(gate passed). **Renorm strength causally and monotonically scales generation length at
a fixed retained subset** — strong for K4, weaker for K6. hit-max stays ~4–6% (mild
right-censoring; RMST tracks capped mean here).

## 7. v31 — Gain controls (decode K4)

**Question:** is full-renorm's effect (A) average scale, (B) gain variance/tail, or
(C) gain↔router-state correspondence?

**Pilot (n=200, decode K4, paired Δ vs native):**

| config | Δlen (95% CI) | acc | note |
|---|---|---|---|
| no_renorm | +4.3 (−2.6,+11.0) | 87.5% | baseline drop, n.s. |
| full_renorm | +22.7 (+16.3,+29.5) | 87.0% | reference effect |
| **fixed_layer_gain** | **+29.5 (+21.5,+38.1)** | 84.0% | pure avg scale ⇒ reproduces (≥) full |
| position_bin_gain | +28.3 (+19.9,+37.3) | 83.0% | avg scale w/ position structure |
| **shuffled_gain** | **+44.6 (+35.8,+54.1)** | **71.0%** | breaks token corresp ⇒ *larger*, acc collapses (McN p=0.001) |
| clipped_q90 / q95 | +24.4 / +26.4 | 86.5 / 85.5% | tail clip does **not** reduce |
| decode_normmatch | +9.2 (+3.6,+14.9) | 89.5% | correct branch-norm scale ⇒ lowest |

**Reading (pilot).** (A) average scale is **sufficient**: `fixed_layer_gain` (+29.5,
pure per-layer mean gain) reproduces and exceeds `full_renorm` (+22.7). (C) token↔router-
state correspondence is **not** the driver: `shuffled_gain` is **larger** (+44.6), and it
**craters accuracy to 71%** (McNemar p=0.001) — mismatched gains inject harmful noise
without shortening. (B) the extreme tail is **not** responsible: clipping at q90/q95 does
not reduce length. The magnitude of the average survivor-weight upscale sets the length;
the true branch-*norm* scale (smaller than mean 1/r) minimizes it, which is exactly why
`decode_normmatch` (+9.2) is lowest and `fixed_layer_gain` (raw mean gain, larger scalar)
overshoots. **Plan's case "fixed mean ≈/> full renorm ⇒ mainly average scale."**

## 8. v32 — Prefill/decode pulse-and-recovery

**Question:** is a transient low-K perturbation recovered under teacher forcing but
amplified under free running (autoregressive feedback)?

**Smoke — full grid, n=12** (dur{1,4,16,64} × start{early,middle,late}), on GPU2.
Token-by-token probes are expensive (~25 min/prompt); `analyze_v32_pulse_recovery.py`
aggregates the incremental `raw.jsonl`.

**Part A (prefill recovery).** Open-loop teacher-forced KL **decays** 0.090 (first 16
steps) → 0.024 (last 16), median half-life ≈ 0 — a one-shot prefill K4 perturbation is
forgotten once K8 decode resumes. Closed-loop free generation still diverges early
(first-div median ≈ 15) but with small net length change (−1.75).

**Part B (decode pulse) — the decisive duration trend:**

| dur | open-loop KL in-pulse | open-loop post-pulse KL (first 8) | closed-loop flip frac | closed Δlen |
|---|---|---|---|---|
| 1 | 0.167 | **0.002** | 0.417 | −1.5 |
| 4 | 0.136 | 0.004 | 0.639 | +0.1 |
| 16 | 0.113 | 0.017 | 0.806 | +2.1 |
| 64 | 0.129 | 0.034 | **1.000** | +3.9 |

**Reading.** Under teacher forcing the same pulse recovers to near-zero KL within 8
tokens (0.002 for dur=1), yet under free running it flips the trajectory with a
probability that **rises monotonically with pulse duration** (0.42 → 1.0), and final
length grows with duration (−1.5 → +3.9). This is direct evidence that **autoregressive
feedback amplifies transient decode perturbations**: the closed loop, not static
accumulation, converts a recoverable local drift into a lasting length/termination
change. Short pulses recover open-loop and rarely reconverge closed-loop (reconv ≈ 0),
so we do **not** claim irreversible bifurcation — the effect is probabilistic and
duration-graded.

---

## 9–11. Hypotheses by evidence level (n=200 pilot; n=12 for v32)

**Strongly supported** (complete controls, paired stats, independent calibration):
- **Renormalization strength causally and monotonically scales length** (v30: K4 Δ
  +7.3→+23.0 across β; endpoints match no/full renorm).
- **The driver is the average survivor-weight upscale MAGNITUDE**, not token↔router
  correspondence and not the extreme tail (v31: `fixed_layer_gain` +29.5 ≥ `full_renorm`
  +22.7; `shuffled_gain` +44.6 larger with acc 71%; `clipped` no reduction).
- **Prefill-calibrated norm-match under-corrects the decode branch norm** (v29 realized
  ratio 0.96 vs decode-calibrated 0.998); the decode-calibrated control is the valid one.

**Consistent with** (direction clear at pilot; confirmatory test-split still recommended):
- **Average decode branch scale is the dominant mechanism** but not the whole story:
  matching it removes ≈73% of the full-renorm length excess (v29 K4 +22.7 → +9.2), with
  a residual (~+5) requiring full token-wise renorm.
- **Autoregressive feedback amplifies transient decode perturbations** (v32: open-loop
  recovers, closed-loop flip 0.42→1.0 monotone in pulse duration; prefill perturbation
  decays under teacher forcing). n=12 — a larger n would tighten the flip fractions.

**Not supported / explicitly not claimed:**
- Token-conditioned gain as the *sole* cause (v31 `shuffled_gain` refutes).
- Average branch scale *fully excluded* (v29 shows it dominant) — the earlier "per-token
  adaptive gain" framing is corrected to "average magnitude + a token-wise residual".
- Any serving speedup (only expert-executions/request reported); any answer-readiness
  claim (out of scope this batch); irreversible trajectory bifurcation (v32 shows
  probabilistic, duration-graded, often-recoverable divergence).

## 12. Data / artifact completeness

Committed + pushed raw for **every** run: v29 calibration (scalars + manifest),
v29/v30/v31 smoke **and** pilot (`*_raw.jsonl` incl. full token ids, per-config
summaries, `analysis.json`, `manifest.json`), v32 full-grid smoke (`raw.jsonl`,
`summary.json`, `summary_from_raw.json`), and all run logs. Confirmatory test[0:500] not
run tonight (compute; deferred by design — §14). No missing artifacts were fabricated.

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
