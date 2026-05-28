# Hypothesis for scheduler_overhead_high_concurrency
## Symptom
The target workload `scheduler_overhead_high_concurrency` produced `ttft_p95_ms = 2282.48` (lower-is-better). Classification: **load_shed_concurrency**. Suspicion score: **0.735**.
## Why we believe this is a cliff, not noise
- Neighbor `scheduler_overhead_high_concurrency__con_16` (max_concurrency=16) has `ttft_p95_ms = 141.59` — bracket evidence.
- Neighbor `scheduler_overhead_high_concurrency__con_32` (max_concurrency=32) has `ttft_p95_ms = 146.94` — bracket evidence.

## Server-log evidence
- `concurrency_capped` is True
- `peak_running_reqs` = 31
- `peak_queue_reqs` = 39
- `max_running_requests` = 32
- `peak_token_usage` = 0.0

## Suggested mechanism
The server's admission cap (`max_running_requests`) is below the workload's `max_concurrency`; requests queue and observe long TTFT. Raising the cap should reduce queueing without hurting steady-state.

## Status

This hypothesis was **auto-drafted** by `select_problems.py` from rule-based scoring evidence. A setter or solver agent should refine it once they have profile data.
