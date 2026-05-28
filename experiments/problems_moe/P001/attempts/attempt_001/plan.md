# Attempt attempt_001 — config-agent

**Problem**: P001 (R_scheduler_tail)
**Strategy**: S001 — If concurrency_capped, raising admission cap should drain the queue.
**Knob**: `max-running-requests` → `64`
**Expected**: +60% on `ttft_p95_ms`
**Risk**: low for v0.4; verify via controls
**Note**: First attempt: raise max-running-requests from 32 to 64 to match max_concurrency. Expected ≥30% TTFT p95 improvement (acceptance threshold).
