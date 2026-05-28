---
name: <skill-name>            # kebab-case
description: <one sentence>
version: 1
stage: [1, 2, 3]               # which stages may consume this
inputs:
  - <name>: <type or path>
outputs:
  - <name>: <type or path>
triggers:
  - "<concrete trigger condition 1>"
  - "<concrete trigger condition 2>"
depends_on: []                 # other skills called inside
---

# <Skill Name>

## WHEN

Concrete trigger conditions. Be specific. "Whenever a benchmark passes" is OK.
"Whenever something interesting happens" is not.

## WHY

The design rationale. Name the specific failure mode this skill prevents.
Reference the v0.2 incident or v0.3 lesson that motivated the skill.

## HOW

Step-by-step procedure or pseudocode. If there is a Python implementation,
reference it: `impl/<file>.py`. Include the exact CLI shape.

## OUTPUT CONTRACT

Exact schema. JSON keys, types, units, expected ranges. Downstream skills
depend on this — treat it as a public API.

## FAILURE MODES

What can go wrong:
- Missing input
- Partial data
- Schema drift in upstream tool

How a caller detects partial output (typically `{"ok": false, "error": "..."}`).

## ROADMAP

Known limitations and what the next version would add.
