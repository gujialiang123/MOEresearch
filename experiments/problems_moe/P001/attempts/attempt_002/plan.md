# Attempt attempt_002 — config-agent

**Problem**: P001 (R_scheduler_tail)
**Strategy**: S001 — If concurrency_capped, raising admission cap should drain the queue.
**Knob**: `max-running-requests` → `96`
**Expected**: +60% on `ttft_p95_ms`
**Risk**: low for v0.4; verify via controls
**Note**: P2: exhaustive sweep over max-running-requests
