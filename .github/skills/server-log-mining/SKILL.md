---
name: server-log-mining
description: Parse a sglang server.log into structured features (cuda graph capture range, max_running_requests, KV pressure, retract events, token usage peak).
version: 1
stage: [1, 2, 3]
inputs:
  - server_log: path to sglang server stdout/stderr (e.g. run_dir/server.log)
outputs:
  - server_features.json
triggers:
  - "after every benchmark run, before scoring"
  - "before opening a Stage 2 diagnosis case (to populate evidence)"
depends_on: []
---

# server-log-mining

## WHEN

Call this skill **immediately after** every benchmark run completes
(`run_experiment.py` finishes), before any scoring or selection logic looks
at the run. Specifically:

- Stage 1: after each workload in `run_regime_suite.py`.
- Stage 2: when building `diagnosis.json` for a case.
- Stage 3: after each fix attempt's benchmark — to verify the symptom moved.

## WHY

The 2026-05-28 Stage 1 run had a clear root-cause signal sitting in
`run_0004_scheduler_overhead_high_concurrency/server.log`:

```
cuda_graph_bs=[1, 2, 4, ..., 256]
Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32]
max_running_requests=32
```

The workload ran at `max_concurrency=64`. CUDA graph captures stopped at 32.
At runtime, batches > 32 fell off the fast path and TTFT p95 jumped from
99 ms to 434 ms.

**Nobody read the server log**, so the v0.2 scorer never saw this. It only
saw the bench_serving aggregate metrics, which can't distinguish
"workload is genuinely hard" from "workload hit a config boundary".

**The skill exists to make that mistake impossible.** Every passed/failed
run now produces a `server_features.json` that any downstream scorer can
use as evidence.

## HOW

Implementation: `impl/parse_server_log.py`.

CLI:

```bash
python .github/skills/server-log-mining/impl/parse_server_log.py \
    --server-log experiments/tmp/.../run_NNNN_xxx/server.log \
    --out        experiments/tmp/.../run_NNNN_xxx/server_features.json
```

Pseudocode:

```
load full text of server.log (truncate to head 2 MB + tail 2 MB for huge logs)
for each regex pattern in PATTERNS:
    scan all matches
    aggregate (first value, peak value, count)
derive booleans (cuda_graph_too_small, at_capacity, ...)
emit JSON
```

## OUTPUT CONTRACT

```json
{
  "schema_version": 1,
  "ok": true,
  "input_log": "experiments/tmp/.../server.log",
  "input_log_bytes": 123456,
  "fields": {
    "model_path": "/data/hf/models/Qwen3-0.6B",
    "kv_cache_size_gb_total": 96.18,
    "max_total_num_tokens": 900480,
    "chunked_prefill_size": -1,
    "max_prefill_tokens": 16384,
    "max_running_requests": 32,
    "context_len": 40960,
    "cuda_graph_bs_configured": [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, ...],
    "cuda_graph_bs_captured":   [1, 2, 4, 8, 12, 16, 24, 32],
    "cuda_graph_capture_seconds": 0.81,
    "peak_token_usage": 0.00,
    "peak_running_reqs": 28,
    "peak_queue_reqs": 28,
    "kv_pool_full_events": 0,
    "retract_events": 0,
    "oom_events": 0,
    "crash_events": 0,
    "server_startup_seconds": 47.1,
    "model_load_seconds": 0.32,
    "cuda_graph_too_small": false,
    "at_capacity": false,
    "near_capacity": false
  },
  "warnings": ["unable to parse model_load_seconds"],
  "errors": []
}
```

`ok=false` only when the log file is missing or unreadable. Partial parsing
returns `ok=true` with warnings listing which fields couldn't be extracted.

## FAILURE MODES

- **Log not yet flushed**: file incomplete because the server was killed
  mid-startup. Returns `ok=true` with most fields null.
- **SGLang version drift**: log format changes between sglang releases. Each
  pattern is best-effort and never raises. Add new patterns; never remove
  old ones (they may still match older runs).
- **Truncation**: default reads head 2 MB + tail 2 MB. For multi-hour Stage
  2 runs increase via `--max-bytes`.

## ROADMAP

- v2: integrate sglang's `/get_server_info` HTTP endpoint (richer than log scraping)
- v2: parse `gen throughput (token/s)` per-batch and compute its CV — would
  catch decode-batch instability that current `tpot_p99/p50` misses
- v3: structured log support if/when sglang adds JSON logging
