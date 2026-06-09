---
name: e2e-bench-runner
description: Run a 3-regime (short/medium/long) end-to-end benchmark against any OpenAI-compatible or sglang-native server, repeat N times with cold-run dropping, and emit a structured bench_summary.json with throughput + latency percentiles + run-to-run stddev.
version: 0
stage: [1, 2, 3]
inputs:
  - url:        string  (e.g. "http://127.0.0.1:30000")
  - backend:    enum["sglang", "vllm"]   (chooses request schema)
  - tag:        string  (free-form, written into output for cross-run grouping)
  - regimes:    list[regime_id]  default ["R_short", "R_medium", "R_long"]
  - num_runs:   int     default 3   (run 1 always dropped — cold-start)
  - out_dir:    path    (writes bench_summary.json + per_run/*.json)
outputs:
  - bench_summary.json                  (the only file downstream skills should read)
  - per_run/<regime>_run<N>.json        (raw per-run dumps; kept for replay / SQL queries)
triggers:
  - "First-touch comparison of two server configurations (A/B or N-way matrix)."
  - "After ANY non-trivial server config change — never trust a single bench, always re-measure."
  - "Before deciding 'this fix worked' or 'this regressed'."
depends_on: []
---

# e2e-bench-runner

## WHEN

Concrete conditions:

1. **N-way config comparison.** Comparing autotune ON/OFF, cudagraph ON/OFF, backend A vs B,
   sglang vs vllm, two FlashInfer commits, etc. Always run **the same workload** under each.
2. **After applying a fix.** Source-code patch landed, you think it helps. Run this skill
   against the patched server **and** against an unpatched control in the same session
   (GPU thermals + other-user noise differ across hours).
3. **Before declaring victory.** A single run can lie by ±20%. This skill enforces
   `num_runs ≥ 3` and reports stddev; if stddev > 8% of mean, the agent must NOT
   draw a conclusion without more runs.

Do **not** call when:
- You only need to know "does the server respond" — use `curl /health` instead.
- You want per-kernel timing — use `nsys-capture` + `nsys-timeline-sql`.
- The server isn't ready (`/health` not 200). This skill will fail loudly but it's wasted time.

## WHY (the failure mode this prevents)

We made **3 documented mistakes** doing benches by hand:

1. **Run-1 cold-start bias** (2026-06-04, vLLM autotune A/B). First run was 1.4× slower
   than runs 2+3 because PyTorch lazy-init + first cudagraph capture happened mid-bench.
   Mean of 3 looked like "AT_ON gives 1.09× over AT_OFF", but mean of runs 2+3 only
   gave 1.02×. **Skill rule: always drop run 1.**
2. **Single-regime conclusions** (2026-06-01, sglang Triton claim). Looked at R_medium only,
   declared sglang Triton 20% slower than vLLM Triton. R_short showed the opposite. The
   regime affected the conclusion. **Skill rule: always run all 3 regimes unless explicitly disabled.**
3. **No stddev reported** — would compare 4.66 vs 4.45 req/s and call it "a regression"
   when both had run-to-run stddev of 0.3. **Skill rule: stddev_pct in output is mandatory.**

## HOW

```bash
python .github/skills/e2e-bench-runner/impl/run_bench.py \
    --url      http://127.0.0.1:30000 \
    --backend  sglang \
    --tag      "sglang_cutlass_fix3" \
    --num-runs 3 \
    --out-dir  results/<experiment>/<config_name>/
```

Internally:
1. **Generate prompts deterministically** (`random.seed(2026)`, same word list as
   `results/4way_bench/scripts/run_bench_4way.py`). Same seed across runs → identical
   prompts → comparable.
2. For each regime, for each run:
   - Send the prompt set with the configured concurrency (uses ThreadPoolExecutor).
   - Per-request: record wall-clock send→first-token (`ttft`), send→last-token (`e2e`),
     and inter-token-latency derived as `(e2e - ttft) / (output_tokens - 1)`.
   - Per-regime: aggregate wall time, output tokens, completion rate.
3. **Drop run 1**, compute mean/p50/p99/stddev across runs 2..N.
4. Write `bench_summary.json` (see contract below) + `per_run/<regime>_run<N>.json`.

## OUTPUT CONTRACT — `bench_summary.json`

```json
{
  "schema_version": 0,
  "ok": true,
  "captured_at": "2026-06-09T04:00:00Z",
  "tag": "sglang_cutlass_fix3",
  "url": "http://127.0.0.1:30000",
  "backend": "sglang",
  "num_runs_total": 3,
  "num_runs_used":  2,
  "regimes": {
    "R_medium": {
      "num_prompts": 16,
      "prompt_words": 800,
      "max_new": 256,
      "concurrency": 8,
      "req_per_s":      {"mean": 4.66, "stddev": 0.12, "stddev_pct": 2.6, "runs": [4.55, 4.78]},
      "tokens_per_s":   {"mean": 1182, "stddev": 25,   "stddev_pct": 2.1, "runs": [1170, 1195]},
      "ttft_ms":        {"p50": 120,  "p99": 280},
      "itl_ms":         {"p50":  18,  "p99":  45},
      "completion_rate": 1.0,
      "wall_s":         {"mean": 3.46, "stddev": 0.08},
      "reliable": true
    },
    /* R_short, R_long ... */
  },
  "warnings": []
}
```

`reliable: false` is set when `stddev_pct > 8` — agent **must not** quote that number
as a conclusion without rerunning. We picked 8% because in our H200 environment the
floor of run-to-run noise on a stable config is 2–4%; 8% means real instability.

## WHICH METRIC HELPS WHICH PROBLEM

> This is the section the agent reads when "the bench finished but I don't know what to look at".
> Bench-level metrics are coarse — they tell you **whether to dig deeper**, not what's wrong.

| Symptom in `bench_summary.json` | Likely class of problem | Next skill to call |
|---|---|---|
| `req_per_s.mean` differs by >2× across two configs of the same server | Probably config-shaped: cudagraph, max_running_requests, batching. | `server-log-mining` first. If clean → `nsys-capture`. |
| `ttft_ms.p50` regression but `tokens_per_s` healthy | Prefill kernel or scheduler. Cudagraph mostly helps decode → unlikely. | `nsys-capture` focusing on prefill window. |
| `itl_ms` regression but `ttft_ms` healthy | Decode kernel slow OR cudagraph not replaying. | `server-log-mining` for cudagraph capture log. If captured, `nsys-capture` decode-only. |
| `ttft_ms.p99 / p50 > 3` (tail blow-up) | Scheduler stall, KV pressure, retract events. | `server-log-mining` → `failure-classification`. |
| `completion_rate < 1.0` | OOM, KV evict, client timeout. **Stop interpreting throughput numbers**. | `failure-classification` immediately. |
| `stddev_pct > 8` on all regimes | Noisy host (other user, thermals, autotune still running). | Re-run after `sleep 60`; check `nvidia-smi`; do NOT trust comparisons. |
| `R_short` improves but `R_medium`/`R_long` don't | Likely a fix that helps low-batch only (e.g. cudagraph for bs=1 captured, larger bs not captured). | Compare `server.log` `cuda_graph_bs=[...]` ranges. |
| `R_long` improves but `R_short` doesn't | Likely a fix that helps prefill GEMMs (kernels scale with seq-len). | `pytorch-profiling` on `R_long` to confirm. |
| All 3 regimes improve by similar % | Probably a per-launch overhead reduction (CPU side). | `nsys-timeline-sql` — check `cuda_api_sum` `cudaLaunchKernel` count + mean duration. |
| `tokens_per_s` better but `req_per_s` same/worse | Per-token speed up but per-request scheduling regressed. | `server-log-mining` for queue stats. |

**General rule**: if any single regime moves by more than 3× the cross-run stddev,
treat that as "interesting" and dig. If movement < 1× stddev, it's noise.

## METHODOLOGY — predict-then-verify

Before running this skill the agent **must** record (in plan.md or the calling
context) a one-sentence prediction:

> "I expect R_medium req/s to go from X to Y (≥Z% improvement) because <reason>."

After the bench, the agent **must** compare measured vs predicted **before**
drawing any conclusion about the underlying mechanism. The 5 wrong root-cause
errors documented in `docs/agent_profiling_capability_audit.md` Part E all share
the same shape: evidence → narrative → no falsification step. This skill exists
in part to enforce that step.

If measured ≠ predicted, the agent must pick one of:
- (a) The hypothesis is wrong → drop it, restart.
- (b) The bench was unreliable (`stddev_pct > 8`, low completion rate) → rerun.
- (c) The fix didn't land where you thought → check git diff / restart server / verify config.

**Do not** invent a new narrative that explains the observed-but-not-predicted number.
That's how we wrote 4 incorrect docs.

## EXTENSION — adding metrics without forking the skill

The `bench_summary.json` schema is fixed (downstream skills depend on it). But
agents often need ad-hoc views. Three escape hatches:

1. **Replay per_run/*.json** — every individual run's per-request records (TTFT,
   ITL, output_tokens, error code) are dumped to `per_run/`. Agent can read them
   with `json.load` and compute any new aggregate (e.g. 99.9th percentile,
   per-prompt token-distribution).
2. **Add a derived metric to the agent's own notes** — do not modify the schema.
3. **Propose a v1 of the skill** — if the new metric proves useful across ≥3
   investigations, bump `version: 1` and add a top-level field. Old fields stay
   forever (rule §5 #2).

## FAILURE MODES

| Mode | Detection | Mitigation |
|---|---|---|
| Server `/health` returns non-200 before run starts | pre-flight HTTP GET | `{"ok": false, "error": "server not ready at <url>"}` — caller must restart |
| One regime times out (request_timeout 600s default) | per-request exception | record `error_count`, set `completion_rate < 1.0`, continue other regimes |
| All 3 runs fail | post-aggregation check | `{"ok": false, "error": "all runs failed; see per_run/*.json for first traceback"}` |
| `num_runs == 1` requested | input validation | warn loudly but proceed; mark `reliable: false` regardless of stddev |
| Mid-bench the server crashes | connection refused mid-stream | abort, mark `completion_rate < 1.0`, dump partial per_run/*.json |
| Backend mismatch (e.g. `--backend vllm` but URL is sglang) | first request returns 404 | `{"ok": false, "error": "endpoint /v1/completions not found — wrong backend?"}` |

## ROADMAP

- **v1** — accept arbitrary regime YAMLs (not hardcoded R_short/medium/long).
- **v1** — per-request token-arrival timeline dump (for ITL distribution plots).
- **v2** — auto-detect cold start from per-request series (e.g. first 10% of requests
  >2× mean → mark as warmup and exclude, even within a single run).
- **v2** — multi-URL fan-out: run the same workload against N URLs in one invocation,
  emit a comparison table.

## REFERENCES

- Existing harness this skill replaces: `results/4way_bench/scripts/run_bench_4way.py`
- Documented bench-conclusion errors: `docs/agent_profiling_capability_audit.md` Part E
- Run-1 cold-start finding: `docs/vllm_2x2_autotune_cudagraph_matrix.md` "noise" section
