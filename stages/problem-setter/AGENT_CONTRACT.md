# Stage 1 — Agent Contract

> This document is **binding for any agent (LLM or human) driving Stage 1**.
> Mode A (`rule_based_explore.py`) implements the same contract automatically.
> Mode B (LLM-driven via `policies/llm_agent.md`) must obey every rule here.

## Mission

Discover SGLang serving regimes that expose performance cliffs for a
specific (model, hardware) pair. Hand each cliff to Stage 2 as a frozen
`case.json` with a machine-readable evidence trail.

## Hard rules (non-negotiable)

1. **Stay headless.** Do not ask the user questions during a Stage 1 run.
   If you would need to ask, write a `STAGE1_QUESTIONS.md` at the end and
   stop; do not block.
2. **Do not modify protected files.** The following are read-only during
   Stage 1:
   - `configs/base.yaml` (the baseline server config — Stage 3 owns
     `configs/best.yaml`, which doesn't exist yet)
   - `scripts/*` (the foundational harness)
   - `.github/skills/*/impl/*` (the skill implementations)
   - `regime_scout/seed_suite.yaml` (the canonical seed set — new
     regimes go through `EXTENSION_GUIDE.md` and only land here after
     human review)
   - `regime_scout/search_space.yaml` (same rule as `seed_suite.yaml`)
3. **Source of truth is JSON, not vibes.** Never claim a workload is
   suspicious without citing a specific field in:
   - `raw_results.jsonl`
   - the run's `server_features.json` (from `server-log-mining`)
   - the run's `classification.json` (from `failure-classification`)
   - the run's `quick_metrics.json`
4. **One workload at a time per benchmark.** Never run two `run_experiment`
   processes in parallel against the same GPU. The harness checks port
   conflicts but you should not push it.
5. **Never delete experiment artifacts.** Failed runs, OOM, timeouts —
   all kept. Stage 2 may want them.
6. **Never silently drop a workload.** If you decide not to expand a seed,
   write the decision and reason into the explore log.
7. **Budget aware.** Respect `--wall-budget-s` (default 5400s = 90min) and
   `--max-waves` (default 2). If you would exceed, stop early and write
   `regime_scout/outputs/budget_exceeded.txt`.
8. **No benchmark gaming.** Forbidden:
   - Reducing `num_prompts` mid-run to make a benchmark "succeed".
   - Changing `flush_cache` to make TTFT look better.
   - Re-running a workload until you get the result you wanted.
   - Reporting only the favourable repeat from a noise calibration.
9. **You may extend, but extensions ship via PR-like files.** If you want
   to add a regime or rule, write the diff into `stages/stage1/proposals/`
   (you may create that directory). Do not patch the canonical files
   directly inside a Stage 1 run.
10. **Stop conditions.** End Stage 1 when any of:
    - All seeds and their selected expansions have been scored.
    - Wall budget exhausted.
    - 5 consecutive benchmark failures (server crash / OOM / timeout).
    - You produced ≥ `--max-cases` frozen cases.

## What you may do

- Read every file under the repo.
- Run any CLI listed in [`TOOLS.md`](./TOOLS.md).
- Generate new workload YAMLs into `regime_scout/candidates/expanded/`.
- Write to `regime_scout/outputs/*`, `experiments/tmp/regime_scout/<ts>/*`,
  `experiments/regimes/cases/SNNN/*`, `logs/*`.
- Append rows to `raw_results.jsonl` (never overwrite — use `--reset`
  only in a fresh wave 0).
- Write a free-form `stage1_summary.md` to
  `regime_scout/outputs/stage1_summary.md` at the end (LLM-only; humans
  read `regime_map.md`).

## What you may NOT do

- Edit configs/base.yaml (or any of the protected paths above).
- Edit `scripts/*.py` or `.github/skills/*/impl/*.py` mid-run.
- Skip server-log-mining for a passed run (it is the only signal for
  Finding-A-class bugs).
- Run Stage 2 or Stage 3 logic. If you believe a knob change would help,
  record the recommendation in `case.json.recommended_stage2` — do not
  apply it.
- Modify sglang source code under `/home/t-jialianggu/work/sglang/`.
- Mutate `model-path` or run a different model than the one in `base.yaml`.

## Handoff contract

When Stage 1 ends successfully, you must produce:

### Mandatory artifacts

- `regime_scout/outputs/raw_results.jsonl` — one row per workload run,
  schema = `run_id, workload_file, workload_name, regime_hint, config_file,
  mode, started_at, finished_at, duration_s, status (pass|fail|skip),
  error, metrics, run_dir`.
- `regime_scout/outputs/suspicious_cases.jsonl` — one row per run, scored.
- `regime_scout/outputs/regime_map.md` — human-readable summary.
- `regime_scout/outputs/regime_map.json` — same, machine-readable.
- `regime_scout/outputs/selected_cases.jsonl` — list of cases handed off.
- `experiments/regimes/cases/SNNN/case.json` — for each selected case;
  schema matches `archive/TWO_STAGE_SUPPLEMENT_v0.2.md` §9 (frozen=true).
- `experiments/regimes/cases/SNNN/workload.yaml` — frozen copy.
- `experiments/regimes/cases/SNNN/metrics.json` — copy of the stage1
  metrics from the relevant run.

### Optional but encouraged (LLM mode B only)

- `regime_scout/outputs/stage1_summary.md` — natural-language synthesis:
  what was tried, what worked, where the LLM's triage decisions differed
  from rule-based and why. Cite file paths.

## Audit trail

Every decision you make must trace back to a JSON file. If a reviewer
later asks "why was S003 selected?" the answer must be a chain of files,
not a paragraph of reasoning.

Good:
> S003 was selected because
> `suspicious_cases.jsonl#run_0004` has `score=0.735` and
> `evidence.components.server_log_signal.evidence` shows
> `concurrency_capped=True` with `peak_queue_reqs=36`.

Bad:
> S003 was selected because it looked like the most interesting case
> when I considered the overall picture.
