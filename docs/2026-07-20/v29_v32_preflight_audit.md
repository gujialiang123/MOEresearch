# v29–v32 Preflight Audit (2026-07-20)

Audit performed against **current code and raw artifacts**, not prior docs. Where
prior docs disagree with code/data, the discrepancy is recorded here and will be
corrected in a separate doc pass after the experiments run. No missing raw results
are fabricated or back-inferred.

## 0. Repository state

- Repo: `/home/t-jialianggu/work/MOEresearch`
- HEAD at audit start: `3bffd82f0aa4a3dbb0db0d1d022c150ddf76ff8d`
- Python: `/home/t-jialianggu/.conda/envs/sglang/bin/python`; HF cache must be
  writable: `HF_HOME=/home/t-jialianggu/work/EndtoEnd-auto-optimization/.hf_cache`
  (`/data/hf` is read-only → `PermissionError` on default cache).
- GPUs at audit: 0/2/5/7 busy (other users), **3/4/6 free**. This batch uses GPU3/GPU4.

## 1. Native-K / router-weight semantics (from `moe_research/k_policy.py`)

- Model: Qwen3-30B-A3B-Instruct-2507; 48 MoE layers; `E=128` experts;
  `num_experts_per_tok = top_k = 8` (native). `norm_topk_prob = True`.
- Router: `routing_weights = softmax(gate(h))` over all 128 experts. The block takes
  native `top_k=8` and (when `norm_topk_prob`) renormalizes those 8 to sum to 1.
  So **native selected mass is normalized to 1 by construction**, but the code does
  NOT assume this — it always divides by the realized pool mass.
- The unified `KPolicy` takes a pool of `max(top_k, eff_k)` and keeps the top
  `eff_k` of it, so keep-all (`eff_k == top_k`) is exactly native.

### Mass definitions actually implemented (k_policy.py lines ~161–199)
- `base = rw / rw.sum(-1, keepdim=True)` (pool renormalized; `rw` = pool top-k weights)
- `M_full = base.sum(-1)` = pool mass (=1 for pool = native top-8)
- `M_K = (base * keep).sum(-1)` = retained mass
- `inv_r = M_full / M_K` = `1/r`, the full-renorm gain. **`inv_r == 1` at keep-all**,
  which is what makes native-K exactly gain-1 for every weight mode.

### Weight-mode formulas (as implemented)
- `renorm_survivors` / `partial_renorm(beta=1)`: `w' = base·keep·inv_r` (full renorm).
- `no_renorm` / `partial_renorm(beta=0)`: `w' = base·keep`.
- `partial_renorm(beta)`: `w' = base·keep·inv_r^beta` → **matches the plan's
  `y = r^{-beta}·Σ_{kept} w_j E_j`** since `inv_r = r^{-1}`. β=0≡no-renorm, β=1≡full
  renorm relative to native-selected mass; K=native ⇒ inv_r=1 ⇒ identity for any β. ✔
- `clipped_gain`: `w' = base·keep·min(inv_r, gain_clip)`.
- `fixed_gain` / `calibrated_norm_match`: `w' = base·keep·s(layer,eff_k)` (frozen scalar).
- `shuffled_gain`: `w' = base·keep·g` with `g` from an external `gain_provider`.
- `fold_mass_to_top1`: dropped mass added to rank-0 weight.

## 2. Dropped experts are PHYSICALLY skipped

`expert_mask = one_hot(selected) & keep`, then only `expert_hit = mask.sum>0`
experts are iterated and their FFN executed (lines ~201–212). Dropped (token,rank)
pairs never enter any expert's input → **no execute-then-multiply-by-zero**. Verified
by `tests/test_k_policy.py::test_physical_skip_call_counter`. ✔

## 3. Phase (prefill vs decode) detection

Decided by **cache state** in a top-level `forward_pre_hook` (`_phase_from_cache`),
NOT by `seq_len==1`: empty/None `past_key_values` (or `cache_position[0]==0`) ⇒
prefill; non-empty ⇒ decode. Robust to chunked prefill / multi-token decode.
The hook also advances a decode-step counter used by `decode_step_selector`.
Verified real-model: policy `(8,8)` gives exactly 0 next-token logit error.

## 4. Old norm-match scalar calibration (the weak link this batch targets)

`scripts/calibrate_norm_match.py`: for each layer `l` and target `K`, measures mean
L2 norm of the MoE branch output under native K8 and under fixed-K `no_renorm`, then
`s(l,K) = E||y8|| / E||yK||`. **Critically, it runs a single `model(ids)` with
`_PHASE='prefill'` on GSM8K-train prompts — i.e. the scalar is estimated on PREFILL
tokens only**, yet it is applied during **decode** in eval. This is exactly the
confound v29 must fix: prefill-token branch-norm statistics need not match decode-token
branch-norm statistics. (The calibration IS same-hidden dual-branch — y8 and yK share
the identical hidden state — but only over prefill positions.)

## 5. v26 position count — DOC ERROR CONFIRMED

`results/2026-07-20_v26_direct_effect/per_position_raw.jsonl` has **7200 records =
2400 positions × {K4, K6, K8}** (verified by counting the `k` field). So the study
covered **2400 positions per K**, not the "~6600" figure that appears in prior docs.
Recorded here; the prior doc will be corrected after this batch. Raw values are NOT
altered.

## 6. Missing / non-standard raw artifacts (no fabrication)

- `2026-07-20_v23_preflight/`: no raw/summary — it was a preflight only (expected).
- `2026-07-20_v25_answer_readiness/`, `..._v26_direct_effect/`: store a single
  `per_step`/`per_position_raw.jsonl` + `summary.json` (not per-config `*_raw.jsonl`).
- `2026-07-20_v31_pulse_recovery/`: raw is `raw.json` (array), not `raw.jsonl`.
- **Run scripts for v23 / v24 / v28 are NOT present in `scripts/`** (only
  `analyze_v23_v28.py` remains). Their raw result dirs exist and are committed, but the
  exact generating scripts are not in the repo → those specific runs are not
  bit-reproducible from source. New v29–v32 scripts fix this by committing the runner,
  manifest (git commit + config hash), and fixed sample-ID lists.

## 7. Frozen protocol decisions for this batch

- **Fixed sample IDs** (committed in each manifest):
  - calibration = GSM8K `train[0:128]`
  - development = GSM8K `train[128:328]` (200, disjoint from calibration)
  - confirmatory = GSM8K `test[0:500]` (matches the `.select(range(500))` convention
    used by the committed v29/v30 test runs, so pairing with v23–v28 test results holds)
  - smoke = `train[128:160]` (first 32 development prompts)
- Generation: greedy, `do_sample=False`, `batch_size` batched only for throughput
  (identical results token-for-token vs bs=1 under greedy), `max_new_tokens=512`
  (NEVER reduced — length is the dependent variable; hit-max = right-censored).
- All runs: deterministic, `--resume`, config hash + git commit in manifest, per-request
  incremental `raw.jsonl`, token/layer traces to parquet.
- `eager wall time` reported only as a diagnostic; NO serving-speedup claim. Cost
  reported as expert-executions/request.

## 8. Execution order (per instructions) & stop conditions

audit → KPolicy+tests → v29 smoke → v29 pilot → v30 smoke → v30 pilot → v31 smoke →
v31 pilot → v32 smoke. Advance only after the previous stage passes. Hard stops:
native-equivalence fail; v29 realized norm-match ratio outside [0.95,1.05] (→ fix
calibration before any mechanism claim); v30 β=0/1 endpoints inconsistent with existing
no/full renorm; v31 shuffled-gain calibration leakage; v32 incoming KV cache mutated.
If compute-limited: prioritize code + tests + smoke and emit exact runnable commands
rather than fabricating pilot/confirmatory numbers.
