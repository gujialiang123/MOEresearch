---
name: cross-regime-anomaly
description: Read a regime_sweep_summary.json matrix and emit an anomaly_report.json that ranks "interesting" findings — winner inversions, regime-dependent gaps, reliability flags, large within-row variance. This is the skill that automates the question "where should I look for an optimization opportunity?"
version: 0
stage: [1, 2, 3]
inputs:
  - sweep_file:  path to regime_sweep_summary.json
  - top_n:       int (default 10; how many anomalies to surface)
  - min_gap_pct: float (default 15.0; only flag config gaps ≥ this)
outputs:
  - anomaly_report.json   (ranked list of findings + suggested next skill per finding)
triggers:
  - "After regime-sweep-runner produces a matrix with ≥2 configs and ≥2 regimes."
  - "When mentor / user asks 'is there an optimization opportunity here?' and you don't yet have a hypothesis to test."
  - "When two new configs (e.g. baseline + patched) are being compared — surfaces whether the patch is consistent or regime-dependent."
depends_on:
  - regime-sweep-runner
---

# cross-regime-anomaly

## WHEN

Concrete conditions:

1. You have a `regime_sweep_summary.json` from `regime-sweep-runner` with at
   least 2 configs × 2 regimes.
2. **You have no specific hypothesis yet** — this skill's job is to surface
   candidate hypotheses for the agent to investigate next.
3. The matrix is sufficiently reliable: at least 60% of cells must have
   `reliable: true` for the anomaly scoring to be meaningful. If less, the
   skill returns `ok: false` with an explanation; the right next step is to
   re-sweep with higher `--num-runs`.

Do NOT call when:
- You only have 1 config or 1 regime — there's nothing to "cross-compare".
- You already know what's wrong; this skill is a finder, not a diagnoser.

## WHY (the failure mode this prevents)

This is the most directly mentor-motivated skill in the toolkit:
**"how can the agent itself find valuable optimization opportunities?"**

Three specific failure patterns it counters:

1. **Confirmation bias on the picked regime** — agent picks R_medium, sees
   "CUTLASS wins by 5%", declares CUTLASS the winner. R_short might show
   CUTLASS losing by 7%. Cross-regime view catches this.

2. **Noise dressed up as a finding** — agent compares two configs that differ
   by 2% on every regime, both with stddev_pct = 3%. That's noise. Without
   automated reliability gating, agents have written entire reports on these
   non-findings (`docs/2026-06-08/fix1_invalidated.md` is one example).

3. **The "winning" config that wins for the wrong reason** — config A wins
   on every regime by exactly the same %, while config B wins on only one
   regime but by 3×. B's win is much more informative — it tells you WHERE the
   bottleneck shifted. This skill ranks B's finding higher than A's.

## HOW

```bash
python .github/skills/cross-regime-anomaly/impl/find.py \
    --sweep-file  results/2026-06-10_sweep/regime_sweep_summary.json \
    --top-n       10 \
    --min-gap-pct 15.0 \
    --out         results/2026-06-10_sweep/anomaly_report.json
```

## OUTPUT CONTRACT — `anomaly_report.json`

```json
{
  "schema_version": 0,
  "ok": true,
  "captured_at": "2026-06-10T05:30:00Z",
  "sweep_file": "results/.../regime_sweep_summary.json",
  "reliability_ratio": 0.83,
  "findings": [
    {
      "rank": 1,
      "kind": "winner_inversion",
      "severity": "high",
      "summary": "vllm_cutlass wins on R_short (+8%) but loses on R_long (-6%)",
      "evidence": {
        "configs": ["vllm_cutlass", "vllm_triton"],
        "regime_wins": {"R_short": "vllm_cutlass", "R_medium": "tie", "R_long": "vllm_triton"},
        "max_gap_pct": 8.0,
        "min_gap_pct": -6.0
      },
      "hypothesis_seed": "MoE backend choice has regime-dependent value — likely tied to a bottleneck that shifts (e.g. launch overhead vs kernel compute) across workload shape.",
      "next_skill": "nsys-capture + nsys-timeline-sql on the inversion regime (R_short)"
    },
    {
      "rank": 2,
      "kind": "large_uniform_gap",
      "severity": "high",
      "summary": "sglang_cutlass is 2.4× slower than sglang_triton on ALL regimes",
      "evidence": {
        "configs": ["sglang_cutlass", "sglang_triton"],
        "gap_pcts_per_regime": {"R_short":-58, "R_medium":-58, "R_long":-59}
      },
      "hypothesis_seed": "A configuration-level issue (autotune disabled? cudagraph off? wrong default tactic?) that affects every workload uniformly. NOT a workload-shape issue.",
      "next_skill": "server-log-mining on sglang server.log to find the config-shaped difference."
    },
    {
      "rank": 3,
      "kind": "reliability_flag",
      "severity": "medium",
      "summary": "vllm_cutlass/R_long has stddev_pct=12.4 — too noisy to compare",
      "evidence": {"config":"vllm_cutlass","regime":"R_long","stddev_pct":12.4,"num_runs":3},
      "hypothesis_seed": "Run-to-run noise dominates — increase num_runs or investigate co-tenant GPU contention.",
      "next_skill": "re-run regime-sweep-runner with --num-runs 5 on just this cell"
    }
  ],
  "warnings": []
}
```

### Finding `kind` enum (`severity` derived from gap magnitude + uniformity):

| `kind`                  | What it means                                              | Severity heuristic |
|---|---|---|
| `winner_inversion`      | Winner changes across regimes (most informative finding!)   | high (large gap) → medium → low |
| `large_uniform_gap`     | Same config wins everywhere by a similar large %             | high if gap > 50% |
| `reliability_flag`      | At least one cell has stddev_pct > 8                        | medium always |
| `failed_cell`           | A cell didn't produce data (server down etc.)               | medium — sweep is incomplete |
| `regime_dependent_gap`  | Gap between same two configs varies > 2× across regimes      | high (bottleneck shifts!) |
| `outlier_regime`        | One regime's req/s deviates > 3× the matrix median for that config | low (often a configuration error) |

## WHICH METRIC HELPS WHICH PROBLEM

The whole **output is** the metric→problem mapping. But within an anomaly,
agents should still read these fields in order:

1. **`hypothesis_seed`** — a one-sentence causal claim the agent should
   *predict-then-verify*, not blindly accept.
2. **`next_skill`** — the recommended downstream skill to confirm/refute the
   hypothesis. Do NOT skip this step.
3. **`evidence`** — the raw data behind the claim. Cross-check it.

If `hypothesis_seed` looks wrong on second read, don't believe it — this
skill is a heuristic flagger, not an oracle.

## METHODOLOGY — predict-then-verify

Agent's predict step (before reading this report):

> "I expect the matrix to show {pattern X}. The most interesting finding will
>  probably be of kind {Y}."

After reading: compare predicted-vs-emitted findings. If the report surfaced
something completely different from what was predicted, **that's the more valuable
signal** — go investigate the unexpected finding first.

## EXTENSION

- The skill is intentionally **rule-based, not ML-based**. Rules are inspectable;
  if a finding looks wrong, the agent reads `find.py` and sees exactly why it
  fired.
- New finding kinds can be added by editing `find.py`'s `KIND_DETECTORS` list
  — each detector is a single function returning a list of findings or [].
- Severity heuristics live in `_severity()`; tunable without affecting detectors.

## FAILURE MODES

| Mode | Detection | Mitigation |
|---|---|---|
| Sweep file unreadable / corrupt | json.load fail | `{"ok": false, "error": "..."}` |
| <60% of cells reliable | reliability_ratio check | `{"ok": false, "error": "reliability_ratio=0.42; re-sweep with --num-runs 5"}` |
| 0 findings produced | post-detection check | `{"ok": true, "findings": [], "warnings": ["no anomalies above threshold — try --min-gap-pct 5"]}` |
| Single-config matrix | input validation | `{"ok": false, "error": "need ≥2 configs"}` |

## ROADMAP

- **v1** — diff two `regime_sweep_summary.json` (before/after a patch): is the
  patch reducing gaps, introducing new ones, or neutral?
- **v1** — incorporate `server-log-mining` features per cell so anomalies tied
  to KV pressure / retract events get a different `kind`.
- **v2** — generate a markdown table alongside the JSON for direct
  paste-into-PR / paste-into-mentor-update.

## REFERENCES

- The 2026-06-04 confirmation-bias incident this skill counters: `docs/2026-06-08/fix1_invalidated.md`
- Sibling: `suspicion-scoring` skill (within-config anomaly); this skill is its **cross-config** complement.
- Mentor framing motivating this skill: "how does the agent find valuable optimization
  opportunities autonomously?" — `docs/2026-06-08/agent_profiling_capability_audit.md` Part C
