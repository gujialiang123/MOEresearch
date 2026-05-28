# Stage 1 — RegimeScout

> **One sentence**: given a model + hardware + a seed set of workloads,
> find which serving regimes expose performance cliffs and hand each
> cliff to Stage 2 as a frozen case.

## Folder layout

```
stages/stage1/
├── README.md                     ← you are here
├── AGENT_CONTRACT.md             ← hard rules: what an agent may and may not do
├── PLAYBOOK.md                   ← step-by-step workflow (humans + LLMs both follow this)
├── TOOLS.md                      ← every CLI tool the agent can call (schemas)
├── EXTENSION_GUIDE.md            ← how to add new regimes / axes / triage rules / skills
├── examples/
│   ├── adding-a-new-regime.md
│   ├── adding-a-new-axis.md
│   └── triage-walkthrough.md
└── policies/
    ├── rule_based_explore.py     ← Headless reference policy (deterministic, CI-friendly)
    └── llm_agent.md              ← System prompt for an LLM agent driving the loop
```

## Two ways to run Stage 1

### Mode A — Headless (rule-based reference policy)

```bash
python stages/stage1/policies/rule_based_explore.py --config configs/base.yaml
```

Deterministic. Uses the 4-rule triage hardcoded in the script. Useful for
CI, regression testing, and as a baseline an LLM agent should at minimum
match.

### Mode B — LLM agent driven

Open a Claude Code (or Copilot CLI) session in this repo. Load
[`policies/llm_agent.md`](./policies/llm_agent.md) as the system prompt.
The agent reads `AGENT_CONTRACT.md` + `PLAYBOOK.md` + `TOOLS.md`, then
drives the loop using the same CLI tools. Smarter triage; more expensive
per run.

**Both modes produce the same artifacts**:
- `regime_scout/outputs/raw_results.jsonl`
- `regime_scout/outputs/suspicious_cases.jsonl`
- `regime_scout/outputs/regime_map.{md,json}`
- `regime_scout/outputs/selected_cases.jsonl`
- `experiments/regimes/cases/SNNN/{case.json, workload.yaml, metrics.json}`

So `Mode A` output and `Mode B` output are interchangeable downstream.

## Quick start (humans)

1. Edit `configs/base.yaml` — at minimum set `model-path`.
2. (Recommended) Calibrate noise:
   ```bash
   python .github/skills/noise-aware-scoring/impl/calibrate_noise.py \
       --config configs/base.yaml \
       --workload regime_scout/candidates/seed_00_smoke.yaml \
       --repeats 5 \
       --out experiments/noise_baseline.json
   ```
3. Run Mode A:
   ```bash
   python stages/stage1/policies/rule_based_explore.py --config configs/base.yaml
   ```
4. Inspect:
   ```bash
   $ cat regime_scout/outputs/regime_map.md
   $ ls experiments/regimes/cases/
   ```

For detail see [`PLAYBOOK.md`](./PLAYBOOK.md).
