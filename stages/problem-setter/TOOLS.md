# Stage 1 — Tools Reference

> Every CLI an agent (or human) may call during Stage 1, with its input
> schema, output schema, exit codes, and a one-line decision rule for
> when to invoke it.

CLI tools are grouped by layer. **Composition direction**: orchestrator
calls suite runner → suite runner calls run_experiment → run_experiment
calls server / wait / bench / parse. Skills produce per-run JSONs that
the orchestrator reads back.

---

## Layer 1 — Foundational harness (`scripts/`)

### `scripts/launch_server.py`
Translate YAML config → sglang launch_server argv, then spawn it.

**You almost never call this directly** in Stage 1. Use `run_experiment.py`
which handles launch+wait+bench+parse+cleanup.

```
--config PATH       sglang server YAML config (required)
--log PATH          where to write server stdout+stderr (required)
--pidfile PATH      optional: write PID for external orchestrator
--conda-env NAME    default: sglang-dev
--dry-run           print argv and exit
```

### `scripts/wait_ready.py`
Poll until sglang server answers HTTP `/health` (or `/v1/models` or TCP).

```
--host HOST         default: 127.0.0.1
--port INT          default: 30000
--timeout INT       default: 300 (seconds)
--interval FLOAT    default: 2.0
```
Exit 0 = ready; 2 = timeout.

### `scripts/run_benchmark.py`
Call `sglang.bench_serving` against an already-running server.

```
--workload PATH       workload YAML
--server-config PATH  server YAML (for model path)
--raw-out PATH        bench_serving --output-file destination
--log PATH            bench stdout+stderr
--timeout INT         seconds, default 900
--conda-env NAME      default: sglang-dev
```
Exit code = bench_serving exit (0 ok, 124 timeout, other = bench error).

### `scripts/parse_metrics.py`
bench_serving JSONL → normalized `metrics.json`.

```
--raw PATH            bench_serving --output-file output (required)
--log PATH            bench log (for scan)
--server-log PATH     sglang server log (for oom/crash detection)
--out PATH            normalized metrics destination (required)
--mode STR            label, e.g. "quick" / "medium" / "repeat"
--expected-requests N optional: compare to completed count
```

Output JSON has stable schema:
```
{
  "schema_version": 1, "mode": "quick", "passed": bool,
  "completed": int, "expected": int|null, "failed_requests": int|null,
  "success_rate": float|null,
  "server_crash": bool, "oom": bool, "timeout": bool, "parse_error": str|null,
  "duration_s": float, "request_throughput": float, "input_throughput": float,
  "output_throughput": float,
  "ttft_mean_ms": float, "ttft_p50_ms": float, "ttft_p95_ms": float,
  "ttft_p99_ms": float, "ttft_std_ms": float,
  "tpot_mean_ms": float, "tpot_p50_ms": float, "tpot_p99_ms": float,
  "itl_p50_ms": float, "itl_p95_ms": float, "itl_p99_ms": float,
  "e2e_p50_ms": float, "e2e_p90_ms": float, "e2e_p99_ms": float, "e2e_mean_ms": float,
  "raw_files": { "raw": str, "benchmark_log": str, "server_log": str }
}
```
Exit 0 if `passed`, 1 otherwise.

### `scripts/run_experiment.py` ★ primary unit of work
**One workload, end-to-end**: launch server → wait → (optional warmup) →
benchmark → parse → cleanup. Call this when you want a single workload's
metrics.

```
--config PATH                   server YAML (required)
--workload PATH                 workload YAML (required)
--mode STR                      quick|medium|full (default: quick)
--out-dir PATH                  destination; default: experiments/tmp/<ts>/
--server-start-timeout INT      default: 300
--benchmark-timeout INT         default: 900
--warmup                        run a tiny warmup before timed bench
--conda-env NAME                default: sglang-dev
```

Produces in `--out-dir`:
- `workload_input.yaml`, `workload_snapshot.yaml`, `config_snapshot.yaml`
- `server.log`, `<mode>_benchmark.log`, `<mode>_raw.jsonl`, `<mode>_metrics.json`
- `orchestrator.log`

Exit 0 = passed, 1 = parse fail, 2 = server didn't start, 3 = bench
timeout.

### `scripts/generate_seed_suite.py`
Expand `regime_scout/seed_suite.yaml` → one yaml per seed in
`regime_scout/candidates/`.

```
--seed PATH               default: regime_scout/seed_suite.yaml
--out-dir PATH            default: regime_scout/candidates
--prune                   delete existing seed_*.yaml first
```

### `scripts/run_regime_suite.py` ★ wave runner
Run every `*.yaml` in a directory (sorted), one workload at a time, append
each run as a row to `raw_results.jsonl`. Continues across failures.

```
--config PATH                 server YAML (required)
--workload-dir PATH           directory of workload YAMLs (default: regime_scout/candidates)
--out PATH                    raw_results.jsonl path (default: regime_scout/outputs/raw_results.jsonl)
--run-root PATH               per-suite parent dir (default: experiments/tmp/regime_scout)
--mode quick|medium|full      default: quick
--server-start-timeout INT    default: 300
--benchmark-timeout INT       default: 600
--wall-budget-s INT           default: 5400 (90 min)
--max-workloads N             default: 999
--reset                       delete existing --out first (use only for wave 0)
--log PATH                    suite-level log
```

### `scripts/cluster_regimes.py`
`raw_results.jsonl` + `suspicious_cases.jsonl` → `regime_map.{md,json}`.

```
--raw PATH
--suspicious PATH
--server-config PATH
--out-md PATH
--out-json PATH
```

### `scripts/select_cases_for_stage2.py`
Top-k scored cases above threshold → frozen `cases/SNNN/{case.json, workload.yaml, metrics.json}`.

```
--raw PATH
--suspicious PATH
--regime-map PATH
--server-config PATH
--out PATH                    selected_cases.jsonl
--cases-root PATH             default: experiments/regimes/cases
--max-cases N                 default: 5
--threshold FLOAT             override search_space's selected_case_score
```

---

## Layer 2 — Skills (`.github/skills/<name>/impl/*.py`)

### `server-log-mining` → `parse_server_log.py`
Parse a sglang `server.log` into 28 structured fields.

```
--server-log PATH   required
--out PATH          required
--max-bytes INT     default 4_000_000
```

Output: `server_features.json` (see skill's SKILL.md).
**Key derived booleans**: `cuda_graph_too_small`, `concurrency_capped`,
`at_capacity`, `near_capacity`, `max_running_above_cuda_graph`.

### `failure-classification` → `classify.py`
Combine metrics + features into one of 10 enum labels.

```
--metrics PATH     optional (may be omitted for total-failure runs)
--features PATH    required (output of server-log-mining)
--out PATH         required
```

Output classification ∈ {clean_pass, load_shed_concurrency,
near_failure_kv, near_failure_retract, partial_success, oom, server_crash,
benchmark_timeout, parse_error, unknown_failure}.

### `noise-aware-scoring` → `calibrate_noise.py`
Run a workload N times, compute CV per metric.

```
--config PATH       sglang server config (required)
--workload PATH     workload to repeat (required)
--repeats N         default 5
--out PATH          noise_baseline.json (required)
--conda-env NAME    default sglang-dev
```

Also exposes `threshold.py`:
```python
from threshold import load_baseline, adjusted_threshold
b = load_baseline("experiments/noise_baseline.json")
t = adjusted_threshold("ttft_p99_ms", base_threshold=3.0, baseline=b, k=2.0)
```

### `boundary-expansion` → `expand.py`
Generate N neighbor workloads along one axis.

```
--parent PATH                 parent workload YAML
--axis NAME                   one of: max_concurrency | num_prompts | input_len | output_len
--strategy NAME               bracket | upward | downward | geometric
--count N                     default 4
--search-space PATH           default regime_scout/search_space.yaml
--neighbors-out DIR           where to write generated YAMLs
--summary-json PATH           optional; writes one-shot summary
```

### `suspicion-scoring` → `score.py`
Compose the four skills above + a local-nonlinearity component into one
score per run with full evidence trail.

```
--raw PATH                    raw_results.jsonl
--noise-baseline PATH         optional
--out PATH                    suspicious_cases.jsonl
--force-mine                  regenerate server_features.json even if present
```

Output: per row, `{run_id, workload_name, score, classification,
components: {server_log_signal, failure_class, tail_latency_ratio,
local_nonlinearity_primary, local_nonlinearity_secondary}}` with each
component's `evidence` block.

---

## Layer 3 — Stage 1 policies (`stages/stage1/policies/`)

### `rule_based_explore.py` — reference orchestrator
End-to-end Stage 1 driver. Encodes the 4-rule triage in
`triage(scored_rows)`. Read its source if you want to imitate it.

```
--config PATH                       required
--seed PATH                         default regime_scout/seed_suite.yaml
--candidates PATH                   default regime_scout/candidates
--expanded-dir PATH                 default regime_scout/candidates/expanded
--raw PATH                          default regime_scout/outputs/raw_results.jsonl
--suspicious PATH                   default regime_scout/outputs/suspicious_cases.jsonl
--regime-md PATH                    default regime_scout/outputs/regime_map.md
--regime-json PATH                  default regime_scout/outputs/regime_map.json
--selected PATH                     default regime_scout/outputs/selected_cases.jsonl
--search-space PATH                 default regime_scout/search_space.yaml
--noise-baseline PATH               default experiments/noise_baseline.json
--max-cases N                       default 5
--wave-budget-s INT                 default 2400 (40 min per wave)
--max-waves N                       default 2 (= seed wave + one expansion)
--max-neighbors-per-plan N          default 3
--reuse-seed-run                    skip wave 0, just re-analyze existing raw_results
--threshold FLOAT                   default 0.30 (final select threshold)
```

---

## Composition recipes

### "Run one workload, score it, classify it"
```bash
python scripts/run_experiment.py \
    --config configs/base.yaml \
    --workload regime_scout/candidates/seed_03_*.yaml \
    --mode quick --out-dir /tmp/single

python .github/skills/server-log-mining/impl/parse_server_log.py \
    --server-log /tmp/single/server.log \
    --out /tmp/single/server_features.json

python .github/skills/failure-classification/impl/classify.py \
    --metrics /tmp/single/quick_metrics.json \
    --features /tmp/single/server_features.json \
    --out /tmp/single/classification.json
```

### "Expand a workload along max_concurrency, run all neighbors"
```bash
python .github/skills/boundary-expansion/impl/expand.py \
    --parent regime_scout/candidates/seed_03_*.yaml \
    --axis max_concurrency --strategy bracket \
    --neighbors-out regime_scout/candidates/expanded/

python scripts/run_regime_suite.py \
    --config configs/base.yaml \
    --workload-dir regime_scout/candidates/expanded \
    --out regime_scout/outputs/raw_results.jsonl \
    --mode quick
```

### "Re-score after adding neighbors"
```bash
python .github/skills/suspicion-scoring/impl/score.py \
    --raw regime_scout/outputs/raw_results.jsonl \
    --noise-baseline experiments/noise_baseline.json \
    --out regime_scout/outputs/suspicious_cases.jsonl \
    --force-mine
```

### "Produce final stage1 artifacts"
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

These four recipes together are exactly what `rule_based_explore.py`
automates.
