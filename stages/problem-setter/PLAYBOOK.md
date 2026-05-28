# Stage 1 — Playbook

> A step-by-step workflow for running Stage 1. Humans, Claude Code sessions,
> and the LLM agent in [`policies/llm_agent.md`](./policies/llm_agent.md)
> all follow this. The deterministic policy
> [`policies/rule_based_explore.py`](./policies/rule_based_explore.py)
> implements the same playbook in Python.

Reads [`AGENT_CONTRACT.md`](./AGENT_CONTRACT.md) (rules) and
[`TOOLS.md`](./TOOLS.md) (CLI reference).

---

## Phase 0 — Verify environment

Before the first run:

1. Activate the conda env: `conda activate sglang-dev`.
2. Check `nvidia-smi` shows the GPU you intend to use.
3. Check `configs/base.yaml`:
   - `model-path` points to a real model directory.
   - `_gpu_id` matches the GPU you want.
   - `mem-fraction-static` is sane for the model (0.7 is safe for small models on H200).

If you change `configs/base.yaml`, this is the **only** moment a Stage 1
agent may touch that file. Once any benchmark has run against it, treat
it as frozen for the rest of the session.

---

## Phase 1 — Noise baseline (recommended once per (model, hardware) tuple)

```bash
python .github/skills/noise-aware-scoring/impl/calibrate_noise.py \
    --config configs/base.yaml \
    --workload regime_scout/candidates/seed_00_smoke.yaml \
    --repeats 5 \
    --out experiments/noise_baseline.json
```

Takes ~10 minutes for a 0.6B model on H200. Produces per-metric mean,
std, CV. Without this, `suspicion-scoring` falls back to hardcoded
thresholds (still works, just less calibrated).

**If you skip this, log the decision** in `regime_scout/outputs/stage1_summary.md`
with the reason (e.g. "first iteration, want fast feedback").

---

## Phase 2 — Wave 0: seed sweep

### Step 2.1 — Materialize seed yamls

```bash
python scripts/generate_seed_suite.py --prune
```

Produces `regime_scout/candidates/seed_00..09_*.yaml` (10 files covering
6 regime types).

### Step 2.2 — Run wave 0

```bash
python scripts/run_regime_suite.py \
    --config configs/base.yaml \
    --workload-dir regime_scout/candidates \
    --out regime_scout/outputs/raw_results.jsonl \
    --mode quick \
    --reset \
    --wall-budget-s 1800
```

This runs each seed end-to-end (~70 s each, ~12 min total on Qwen3-0.6B
+ H200). Each row in `raw_results.jsonl` corresponds to one workload.

### Step 2.3 — Sanity check

```bash
$ wc -l regime_scout/outputs/raw_results.jsonl   # should be 10
$ jq -r 'select(.status=="fail") | .workload_name + " : " + (.error // "?")' \
    regime_scout/outputs/raw_results.jsonl
```

If > 30% of seeds failed, stop and inspect — there's a harness or
environment problem, not a regime discovery problem.

---

## Phase 3 — Score and classify

```bash
python .github/skills/suspicion-scoring/impl/score.py \
    --raw regime_scout/outputs/raw_results.jsonl \
    --noise-baseline experiments/noise_baseline.json \
    --out regime_scout/outputs/suspicious_cases.jsonl \
    --force-mine
```

`--force-mine` regenerates `server_features.json` and `classification.json`
in every run dir. Skip on re-runs if features are already fresh.

After this you have, per row:
- `score` (0–1)
- `classification` (10-state enum)
- `components.{server_log_signal, failure_class, tail_latency_ratio, local_nonlinearity_*}` each with `evidence`.

### Step 3.1 — Glance at the top

```bash
$ jq -r '.workload_name + "\t" + (.score|tostring) + "\t" + .classification' \
    regime_scout/outputs/suspicious_cases.jsonl | head -10
```

---

## Phase 4 — Triage (decide expansions)

### Mode A (rule-based reference)

`rule_based_explore.py` runs these 4 rules in `triage(scored_rows)`:

1. **`concurrency_capped` or `cuda_graph_too_small`** ⇒ bracket
   `max_concurrency` around the suspect.
2. **`at_capacity` or `near_capacity`** ⇒ upward-expand `input_len`.
3. **Lonely cluster** (only member of its `regime_hint`) **AND**
   score ≥ 0.1 ⇒ bracket the hint's natural axis (decode→output_len,
   prefill→input_len, scheduler→max_concurrency, prefix→max_concurrency).
4. Otherwise no expansion.

Dedupe: at most one plan per (workload, axis).

### Mode B (LLM agent)

You may extend rule-based with additional triggers, e.g.:

- If two workloads in the same hint cluster have very different
  `output_throughput` despite similar inputs, expand `output_len` to
  find the cliff.
- If a server log shows `peak_token_usage > 0.5` even though
  `max_concurrency` is small, suspect KV pressure → expand `input_len`.
- If a workload's classification is `near_failure_retract`, generate a
  smaller-size neighbor to see if shrinking recovers the workload.

**Document every plan**: write each triage decision (plan + reason +
cited evidence) into `regime_scout/outputs/triage_log.jsonl` so a
reviewer can audit it.

---

## Phase 5 — Wave 1: expansion

For each triage plan:

```bash
python .github/skills/boundary-expansion/impl/expand.py \
    --parent <plan.parent_workload_path> \
    --axis <plan.axis> \
    --strategy <plan.strategy> \
    --neighbors-out regime_scout/candidates/expanded/ \
    --summary-json regime_scout/outputs/expand_<plan_id>.json
```

Then run all generated neighbors:

```bash
python scripts/run_regime_suite.py \
    --config configs/base.yaml \
    --workload-dir regime_scout/candidates/expanded \
    --out regime_scout/outputs/raw_results.jsonl \
    --mode quick \
    --wall-budget-s 1800
```

**Note**: do NOT pass `--reset` here. Wave 1 appends to wave 0 data.

---

## Phase 6 — Re-score with neighbors present

Same as Phase 3 but neighbors are now in the dataset; `local_nonlinearity`
components will activate for the first time.

```bash
python .github/skills/suspicion-scoring/impl/score.py \
    --raw regime_scout/outputs/raw_results.jsonl \
    --noise-baseline experiments/noise_baseline.json \
    --out regime_scout/outputs/suspicious_cases.jsonl \
    --force-mine
```

Suspicious workloads with confirmed neighbor cliffs should now score
significantly higher than their neighbors (see M1 result: 0.535 → 0.735
after Finding A's neighbors were added).

---

## Phase 7 — Cluster and select

```bash
python scripts/cluster_regimes.py \
    --raw regime_scout/outputs/raw_results.jsonl \
    --suspicious regime_scout/outputs/suspicious_cases.jsonl \
    --server-config configs/base.yaml \
    --out-md regime_scout/outputs/regime_map.md \
    --out-json regime_scout/outputs/regime_map.json

python scripts/select_cases_for_stage2.py \
    --raw regime_scout/outputs/raw_results.jsonl \
    --suspicious regime_scout/outputs/suspicious_cases.jsonl \
    --regime-map regime_scout/outputs/regime_map.json \
    --server-config configs/base.yaml \
    --out regime_scout/outputs/selected_cases.jsonl \
    --threshold 0.30 --max-cases 5
```

---

## Phase 8 — Wrap up

### Mandatory

- Verify all artifacts in §AGENT_CONTRACT.md "Mandatory artifacts" exist.
- Verify per-run dirs are present and non-empty.

### Optional (LLM mode B only)

Write `regime_scout/outputs/stage1_summary.md`:

```markdown
# Stage 1 summary — <model> on <hardware> — <date>

## What I tried
- Wave 0 covered 10 seeds: ...
- Wave 1 expansions: ...

## What I found
- S001: scheduler_overhead_high_concurrency (score=0.735, classification=
  load_shed_concurrency). Evidence: ...

## Where I deviated from rule-based
- I added a custom expansion for X because Y (cite evidence).

## Recommendations for Stage 2
- Start with S001; the cap-on-max_running-requests theory has 4-component
  evidence. Suggested first knob: max-running-requests=64.

## Open questions
- ...
```

---

## Loop termination

End the playbook when **any** of:

- All triage plans executed and all neighbors scored.
- Wall-clock budget exhausted.
- 5 consecutive benchmark failures.
- ≥ `--max-cases` cases produced.

Write a final line to `logs/explore_<ts>.log` stating the termination
reason and the artifact paths.
