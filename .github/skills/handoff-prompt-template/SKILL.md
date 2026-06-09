---
name: handoff-prompt-template
description: Methodology skill (no impl). Defines the contract by which an "analysis agent" (which has free run of the profiling toolkit) passes a structured, falsifiable, minimal-scope task to a "coding agent" (which only edits code and verifies). Prevents narrative loss + scope creep when two agents share information through natural language.
version: 0
stage: [2, 3]
inputs:
  - none (this is a methodology skill — copy the template)
outputs:
  - handoff.md (one file per code-change task, written by the analysis agent)
triggers:
  - "Analysis agent has reached a concrete code change candidate (file:line + proposed diff) and wants to delegate the edit + verification to a coding agent."
  - "Any time you want the next step to be a discrete, falsifiable code change with an acceptance test."
  - "When you would otherwise write a long narrative report and hope the next agent picks up the right thing — DON'T; use this template instead."
depends_on: []
---

# handoff-prompt-template

## WHEN

Concrete conditions:

1. You (analysis agent) have:
   - A specific file + line range identified for the change
   - A concrete proposed patch or pseudocode
   - A mechanical acceptance test (a skill invocation that will pass/fail)
   - At least one falsification path ("if X doesn't change, the hypothesis is wrong")
2. The next actor (coding agent) does NOT need to re-read the original profiles —
   the handoff should be **self-contained** enough that the coding agent can act
   with only the handoff + the cited file paths.

Do **not** use this template when:
- You only have a vague hypothesis ("I think MoE could be faster") — go back
  and finish the analysis.
- The change requires multiple coordinated patches across many files — split
  into multiple handoffs, one per file, each with its own acceptance test.
- The "fix" is purely configuration (env var, launch flag) — those don't need
  a coding agent; just call `e2e-bench-runner` directly on the new config.

## WHY (the failure mode this prevents)

Documented failures:

1. **Narrative loss** (2026-06-08 fix1_invalidated.md). The analysis agent wrote
   a 200-line report identifying "AutoTuner re-benchmarks per forward"; the next
   step proposed a `tune_max_num_tokens=8192` fix; this fix made performance
   WORSE. Root cause: the narrative had a hypothesis but never specified the
   *falsification step*. We "fixed" something that wasn't broken.

2. **Scope creep** (hypothetical but very plausible). A coding agent receiving
   "improve MoE performance" will eventually touch flash-attention, kv-cache,
   tokenization — none of which were measured. The template's "What NOT to do"
   field exists to bound this.

3. **Untested acceptance criteria** (multiple). "Apply the patch and re-bench" —
   but with WHAT bench, on WHAT regimes, with WHAT noise threshold? Coding
   agent guesses; analysis agent later disagrees with the guess. Both waste time.

## HOW

Copy `template/handoff.md` and fill every field. Empty fields are anti-patterns:
either fill them or explain why empty in the same line.

```bash
cp .github/skills/handoff-prompt-template/template/handoff.md \
   experiments/<date-or-ticket>/<short-name>.handoff.md
$EDITOR experiments/<date-or-ticket>/<short-name>.handoff.md
```

Then hand `experiments/<date>/<short-name>.handoff.md` to the coding agent.

## OUTPUT CONTRACT — `handoff.md` (markdown, NOT JSON)

The template (also at `template/handoff.md`) is markdown for readability;
agents that want to programmatically check it should grep for the H2 anchors.
Every required H2 section must exist with non-empty content.

Required sections (anchors):
- `## Problem statement`     — one sentence
- `## Evidence chain`        — bulleted, each `skill_name → key metric → file`
- `## Hypothesis`            — causal chain: measurement + pattern → likely cause
- `## Suggested change`      — file:line, type, diff or pseudocode
- `## Acceptance test`       — exact skill invocation + expected delta + revert criterion
- `## Known risks`           — each `risk → mitigation`
- `## What NOT to do`        — explicit scope bounds (bulleted)

Optional sections:
- `## Cross-references`      — links to docs, source paths, prior handoffs
- `## Predicted outcome`     — analysis agent's prediction for the falsification log

## METHODOLOGY — what each section is for

| Section | What it bounds | What goes wrong without it |
|---|---|---|
| `Problem statement` | one specific thing to fix | "improve perf" → unbounded scope |
| `Evidence chain` | every claim is traceable to a skill output | narrative claims with no data backing |
| `Hypothesis` | a falsifiable causal story | "I think it'll help" — no way to test |
| `Suggested change` | exact file:line + diff | "rewrite the MoE wrapper" — too vague |
| `Acceptance test` | mechanical pass/fail | "re-bench and see" — by what metric? |
| `Known risks` | explicit guard rails | shipping breaks something unrelated |
| `What NOT to do` | scope-creep firewall | coding agent "improves" attention too |

## EXAMPLES

`examples/cutlass_d1_sglang_autotune.handoff.md` — a worked example for the
D1 improvement direction from `docs/2026-06-09/cutlass_vs_triton_e2e_investigation.md`.

## FAILURE MODES

| Failure | How the coding agent should react |
|---|---|
| Required H2 missing | Refuse the handoff; ask analysis agent to fill it. |
| Acceptance test isn't mechanical (e.g. "use your judgement") | Refuse; ask for a concrete skill invocation + threshold. |
| Evidence chain cites no skill names | Refuse; ask for `skill_name → file → metric` per line. |
| `What NOT to do` empty AND `Suggested change` looks broad | Refuse; ask for scope bounds. |
| Acceptance test passes but `Known risks` flagged something the agent now sees | Pause; escalate to analysis agent BEFORE shipping. |

## ROADMAP

- **v1** — add a YAML "frontmatter" block at the top of handoff.md for
  programmatic intake (handoff_id, source_skill, status).
- **v1** — `handoff-validator` impl that lints a handoff.md for missing
  sections / broken file references before passing to coding agent.
- **v2** — when a handoff is closed (acceptance test passed or revert), append
  a short outcome block; this becomes training/eval data for future agents.

## REFERENCES

- The narrative-loss failure this skill counters: `docs/2026-06-08/fix1_invalidated.md`
- The "what to optimize" finder that produces handoff inputs: `cross-regime-anomaly` skill
- The mentor framing that motivated structured handoff:
  `docs/2026-06-08/agent_profiling_capability_audit.md` Part E ("predict-then-verify")
