# Stage 1 — Extension Guide

> How to add new regimes, new axes, new triage rules, or new skills to
> Stage 1. This is also the contract an LLM agent should follow if it
> wants to propose extensions during a run.

**Rule of thumb**: extensions go through a **proposal directory** that the
agent writes to, and a human (or a follow-up review step) promotes them
into the canonical files. Agents never patch canonical files mid-run.

```
stages/stage1/proposals/        ← create as needed
├── regimes/                    ← new seed regimes
├── axes/                       ← new axes for search_space
├── triage_rules/               ← new triage rules
└── skills/                     ← new skills (full SKILL.md drafts)
```

---

## 1. Adding a new workload regime

A regime is a workload "shape" worth exploring (e.g. `long_context_decode`,
`mixed_arrival_poisson`).

### When to add one

You see a recurring real workload pattern that the existing 6 regimes
don't capture, AND you have at least a rough hypothesis about what
bottleneck it exposes.

### Steps

1. **Write a draft yaml** under `stages/stage1/proposals/regimes/`:

   ```yaml
   # stages/stage1/proposals/regimes/long_context_decode.yaml
   name: "long_context_decode"
   regime_hint: "long_context_decode"
   rationale: |
     Combines long prompt (8k) with long generation (512). Distinct from
     prefill-heavy regime because prefill cost is amortized over many
     decode steps; expected primary bottleneck is KV cache occupancy.
   dataset:
     name: "random"
     random_input_len: 8192
     random_output_len: 512
   traffic:
     max_concurrency: 8
     num_prompts: 40
   cache:
     mode: "cold"
     flush_cache: true
   primary_metric_candidate: "output_throughput"
   axes_likely_to_matter:
     - input_len
     - max_concurrency
   ```

2. **Smoke test it** end-to-end:

   ```bash
   python scripts/run_experiment.py \
       --config configs/base.yaml \
       --workload stages/stage1/proposals/regimes/long_context_decode.yaml \
       --mode quick \
       --out-dir /tmp/regime_smoke_$(date +%s)
   ```

   Confirm: passes, has valid metrics, finishes inside reasonable time
   for `--mode quick` (say ≤ 5 min on the target hardware).

3. **Run with server-log-mining** and check the features look sensible:

   ```bash
   python .github/skills/server-log-mining/impl/parse_server_log.py \
       --server-log /tmp/regime_smoke_*/server.log \
       --out /tmp/regime_smoke_*/server_features.json

   jq '.fields | {peak_running_reqs, peak_token_usage, concurrency_capped,
                  at_capacity, cuda_graph_too_small}' \
       /tmp/regime_smoke_*/server_features.json
   ```

4. **Promote to canonical** (human review):

   Open `regime_scout/seed_suite.yaml` and append the seed (without the
   `rationale` / `primary_metric_candidate` / `axes_likely_to_matter`
   metadata — those stay in the proposal file). Re-run
   `generate_seed_suite.py`.

5. **Document the regime semantics** under
   `docs/stage1/WORKLOAD_REGIMES.md` (one paragraph + the rationale).

### Acceptance criteria for a new regime

- Smoke test passes.
- The regime exposes a measurable signal (`server_features.json` reflects
  the intended bottleneck) OR the rationale explains why baseline looks
  "boring" and an expansion would unlock the signal.
- The `regime_hint` value is unique vs the existing 6.
- The proposal file is preserved (don't delete after promotion).

---

## 2. Adding a new axis

An axis is a workload knob the boundary-expansion skill can sweep
(currently: `max_concurrency`, `num_prompts`, `input_len`, `output_len`).

### When to add one

A recurring triage decision points to an axis that doesn't exist yet
(e.g. `request_rate` for arrival-rate sweep, `prefix_groups` for cache
churn sweep).

### Steps

1. **Write a draft block** under `stages/stage1/proposals/axes/`:

   ```yaml
   # stages/stage1/proposals/axes/request_rate.yaml
   axis_name: request_rate
   workload_yaml_path: traffic.request_rate
   values: [null, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]    # null = closed-loop
   why_this_matters: |
     Open-loop vs closed-loop arrival exposes scheduler tail differently.
     A workload at max_concurrency=16 + request_rate=null queues nothing;
     same workload at request_rate=8 has Poisson bursts.
   neighbor_strategy: bracket
   ```

2. **Verify the field path** in
   `.github/skills/boundary-expansion/impl/expand.py`:

   ```python
   AXIS_TO_FIELD: dict[str, tuple[str, str]] = {
       "max_concurrency": ("traffic", "max_concurrency"),
       ...
       # add:
       "request_rate":    ("traffic", "request_rate"),
   }
   ```

   This is a Python edit, so it goes through the same human-review gate
   as a skill change.

3. **Add the axis to `regime_scout/search_space.yaml`**:

   ```yaml
   axes:
     ...
     request_rate:
       values: [null, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
   ```

4. **Smoke test expansion**:

   ```bash
   python .github/skills/boundary-expansion/impl/expand.py \
       --parent regime_scout/candidates/seed_03_*.yaml \
       --axis request_rate --strategy bracket \
       --neighbors-out /tmp/axis_test/
   ls /tmp/axis_test/        # should show 4 yaml files
   cat /tmp/axis_test/*.yaml | grep request_rate
   ```

### Acceptance criteria

- `AXIS_TO_FIELD` updated.
- `search_space.yaml` updated.
- Smoke test produces valid neighbor yamls that `run_experiment.py`
  accepts.
- A triage rule references this axis (otherwise the axis sits unused).

---

## 3. Adding a new triage rule

The current rule-based triage has 4 rules in
`stages/stage1/policies/rule_based_explore.py::triage()`. Adding a 5th
requires identifying a recurring signal that currently goes ignored.

### When to add one

- Multiple Stage 1 runs surface the same kind of suspect that the
  existing rules don't catch.
- An LLM agent (Mode B) keeps making the same custom decision in its
  `triage_log.jsonl` — turn the pattern into code.

### Steps

1. **Write a proposal** at `stages/stage1/proposals/triage_rules/<name>.md`:

   ```markdown
   # Triage rule: kv_pressure_at_long_inputs

   Trigger: `peak_token_usage >= 0.5` AND `input_len >= 4096`
   Plan: expand input_len downward + max_concurrency downward (find the
         capacity floor).
   Evidence basis: cite the runs that motivated this rule.
   Risk of false positive: low; only fires on actually-heavy KV usage.
   ```

2. **Sketch the Python diff** in the same proposal file:

   ```python
   # In triage(scored_rows):
   if 0.5 <= (contributions.get("near_capacity", 0)
            + contributions.get("at_capacity", 0)):
       plans.append({
           "workload": s["workload_file"],
           "axis": "input_len",
           "strategy": "downward",
           "reason": "KV pressure at long inputs",
           "source_workload_name": s["workload_name"],
       })
   ```

3. **Cite ≥ 2 runs** from past `raw_results.jsonl` where this rule would
   have helped.

4. **Human review and merge** into `rule_based_explore.py`.

### Acceptance criteria

- The rule is **specific** (no generic "if score > X, expand everything").
- It's **bounded** (cap on plans per run; usually 1).
- A reviewer can trace the rule back to ≥ 2 concrete past runs.

---

## 4. Adding a new skill

This is the heaviest extension. Read [`docs/skills/README.md`](../../docs/skills/README.md) for
design principles first.

### When to add one

You realize a **single distinct piece of methodology** is being
re-discovered across multiple ad-hoc places. Wrapping it as a skill makes
it reusable across Stage 2 and Stage 3 too.

### Steps

1. **Draft `SKILL.md`** at `stages/stage1/proposals/skills/<name>/SKILL.md`,
   filling all six required sections (WHEN / WHY / HOW / OUTPUT
   CONTRACT / FAILURE MODES / ROADMAP). Use `.github/skills/_template/SKILL.md`.

2. **Draft `impl/<file>.py`** under the same proposal subdir. Keep it
   stateless: input JSON → output JSON.

3. **Write a 1-page rationale**: which v0.3 failure does this skill
   prevent? Cite a specific run or analysis where the absence of this
   skill caused a missed signal.

4. **Human review and promote** to `.github/skills/<name>/`. Add a row to
   `docs/skills/README.md` §6.

### Acceptance criteria

- All six SKILL.md sections present.
- Implementation never raises on missing input (returns `{"ok": false,
  "error": "..."}`).
- Output schema documented and immutable (only additive changes after
  ship).
- Tests or golden-file examples in `impl/tests/` (planned for v0.4).
- Skill is invokable as a CLI; agent can call it without writing Python.

---

## 5. What an LLM agent should NOT propose

- A score function rewrite (rules stay; LLM proposes weight tweaks at
  most, with cited evidence).
- A new server-log regex that depends on a non-stable sglang log format
  (must cite the sglang version it was tested on).
- A new triage rule that fires on `> 80%` of runs (too general).
- A new skill that wraps a one-line operation (just put it in
  `scripts/utils.py`).
- Anything that requires modifying sglang source code (that belongs in
  Stage 3 kernel-agent territory).

---

## Promotion workflow (proposal → canonical)

```
LLM agent writes proposal           ← during a Stage 1 run
   ↓
stages/stage1/proposals/<kind>/<name>.{md,yaml,py}
   ↓
Human review (or scheduled review job)
   ↓
Promoted to canonical location:
   - new regime → regime_scout/seed_suite.yaml
   - new axis   → regime_scout/search_space.yaml + AXIS_TO_FIELD edit
   - new rule   → rule_based_explore.py::triage()
   - new skill  → .github/skills/<name>/ + row in SKILLS.md
   ↓
Proposal file kept as historical record
```
