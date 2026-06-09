# <one-line title — the change you're handing off>

<!-- 
COPY THIS FILE, FILL EVERY H2 SECTION. EMPTY SECTIONS ARE ANTI-PATTERNS.
If a section truly doesn't apply, write "N/A — reason: ..." instead of leaving blank.
-->

## Problem statement

<one sentence — what specifically is suboptimal, and where>

## Evidence chain

<!-- each bullet: skill_name → key metric or finding → file/path -->
- `<skill-name>` → <metric: value> → `<path/to/output_file.json>`
- `<skill-name>` → <metric: value> → `<path/to/output_file.json>`
- source: `<repo/path/to/file.py:LINE-RANGE>`

## Hypothesis

<causal chain: this measurement + this pattern + this code path → likely cause is X>

## Suggested change

- **file**: `<path/to/file.py>`
- **lines**: `<LINE_START-LINE_END>`
- **type**: <one of: source_edit | config_swap | env_var | dep_bump>
- **patch** (diff or pseudocode):

```diff
- <old line>
+ <new line>
```

## Acceptance test

<!-- Must be a mechanical pass/fail. -->

- **call**: `<exact shell command, typically a skill invocation>`
- **expect**: `<exact metric, exact threshold, e.g. R_medium req_per_s_mean ≥ X (improvement ≥ Y%)>`
- **revert if**: `<concrete condition under which the patch must be reverted>`

## Known risks

<!-- bulleted; each risk → mitigation -->

- risk: <what could go wrong>
  - mitigation: <action that bounds the risk>
- risk: ...
  - mitigation: ...

## What NOT to do

<!-- explicit scope bounds — what the coding agent must NOT change beyond `Suggested change` -->

- do NOT modify <other module / other file / other config>
- do NOT bump <other dep / version>
- do NOT touch <related-but-out-of-scope kernel / wrapper>

## Cross-references (optional)

- docs: `docs/<date>/<file>.md`
- prior handoff: `experiments/<date>/<other-name>.handoff.md` (if iterating)

## Predicted outcome (optional)

<analysis agent's prediction. Will be checked against actual measurement post-patch
to confirm the hypothesis was correct OR falsified.>
