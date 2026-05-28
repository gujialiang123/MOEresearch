# SGLang End-to-End Optimization Agent

> 🇬🇧 English first · 🇨🇳 [跳转中文版 (jump to Chinese)](#-中文版)

**One-line goal**: given a model, a GPU, and SGLang, automatically discover
which serving regimes (input/output length + concurrency + cache patterns)
expose performance cliffs, and hand each cliff off to a downstream agent
that will diagnose and fix it.

**Status (2026-05-28)**: Stage 1 (RegimeScout) end-to-end works. On a real
H200 + Qwen3-0.6B run it auto-discovered a real bug
(`max-running-requests=32` caps `max_concurrency=64`, queue depth peaks at
36, TTFT p95 jumps 99 ms → 434 ms). Stages 2 and 3 are designed but not
yet implemented.

> ⚠️ **Important — what's an "agent" here?** Stage 1 is intentionally
> **rule-based at runtime** (deterministic scripts + a 4-rule triage). No
> LLM is in the loop during a scout run. LLM agents are the planned home
> for **Stage 2 (Diagnose)** and **Stage 3 (Fix)**, where semantic
> reasoning over evidence is required. See §2 for the rationale and §11
> for what "agent" means at each stage.

---

## 1. TL;DR — can I run a single command?

Yes.

```bash
# 0. activate the env once
conda activate sglang-dev

# 1. edit the model path in configs/base.yaml (one-time)
$EDITOR configs/base.yaml          # set model-path to /your/model

# 2. fire it
python scripts/explore_regimes.py --config configs/base.yaml
```

That's it. ~15 minutes later you have:

- `regime_scout/outputs/regime_map.md`           — human-readable summary
- `regime_scout/outputs/selected_cases.jsonl`    — frozen problem cases for downstream stages
- `experiments/regimes/cases/S001/case.json`     — full evidence trail per case
- `logs/explore_<timestamp>.log`                 — orchestrator log
- 12 per-workload run dirs under `experiments/tmp/regime_scout/<ts>/`,
  each containing `server.log`, `quick_raw.jsonl`, `quick_metrics.json`,
  `server_features.json`, `classification.json`

See **§5 Step-by-step what happens inside `explore_regimes.py`** for the
internal walkthrough.

---

## 2. The mental model: two stages

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage A  Problem-Setter (出题人 Agent)                              │
│           "find regimes worth optimizing,                           │
│            prove the cliff is real,                                 │
│            package as a self-contained problem"           ← THIS REPO TODAY
│                                                                     │
│ Stage B  Problem-Solver Fleet (做题人 Agent fleet)                  │
│           "config-agent / scheduler-agent / kernel-agent /          │
│            workload-shape-agent — each solves problems              │
│            within its scope"                              ← designed, not built
└─────────────────────────────────────────────────────────────────────┘
```

**Algorithm-competition metaphor**: the setter prepares problems (with
proof that the cliff is real); the solver fleet picks problems off the
shelf and writes solutions. Each problem package is **self-contained**:
target workload + boundary-expansion neighbors + unrelated controls +
evidence + hypothesis + acceptance criteria + every solver attempt +
the final report — all in one directory `experiments/problems/PNNN/`.
You can `tar` it up and hand it to anyone.

### Why two stages, not three

A previous draft (in [`archive/`](./archive/)) split discovery from
diagnosis. That was wrong: **diagnosis informs discovery**. You can't
tell whether to expand `max_concurrency` or `input_len` until you've
profiled. Cutting them apart broke the feedback loop. The merged setter
can interleave bench → log-mine → score → profile → expand boundary →
re-score in one fluid loop.

Detail and contracts:

- Architecture overview (bilingual): [`docs/architecture/two-stage-overview.md`](./docs/architecture/two-stage-overview.md)
- Problem package schema: [`docs/problem-package/schema.md`](./docs/problem-package/schema.md)
- Idea pool (bidirectional channel): [`docs/idea-pool/schema.md`](./docs/idea-pool/schema.md)

The handoff between setter and solver is a **JSON contract**, so today
you can drive setter with Claude Code and solver with Copilot CLI (or
vice versa); tomorrow we add a single CLI that automates both ends.

---

## 3. Skills — reusable units of methodology

Skills are the project's secret sauce. They are named, single-purpose,
agent-agnostic, stage-agnostic units of knowledge. The same `server-log-mining`
skill is used by Stage 1's scout, Stage 2's diagnoser, and Stage 3's
fixer. Write the methodology once, reuse forever.

See [`SKILLS.md`](./SKILLS.md) for the design principles + how to add a
new skill. Current catalog:

| Skill | Stages | What it does | Status |
|---|---|---|---|
| `server-log-mining` | 1, 2, 3 | Parse a sglang `server.log` into 28 structured fields including derived booleans like `concurrency_capped`, `cuda_graph_too_small`, `at_capacity`. | ✅ implemented |
| `failure-classification` | 1, 2, 3 | Map (metrics + features) → 10-state enum: `clean_pass` / `load_shed_concurrency` / `near_failure_kv` / `oom` / ... | ✅ implemented |
| `noise-aware-scoring` | 1, 3 | Run a baseline workload N times, compute per-metric coefficient of variation, expose `adjusted_threshold()`. | ✅ implemented |
| `boundary-expansion` | 1 | Given one workload + an axis, generate N neighbor workload YAMLs along that axis. 4 strategies: bracket/upward/downward/geometric. | ✅ implemented |
| `suspicion-scoring` | 1 | Combine the four above into one score per run with full evidence trail. | ✅ implemented |
| `minimal-repro-shrink` | 1, 2 | Binary-shrink a workload until the symptom disappears. | 🟨 SKILL.md only; impl deferred |

Each `SKILL.md` follows a fixed structure: **WHEN / WHY / HOW / OUTPUT
CONTRACT / FAILURE MODES / ROADMAP**. The WHY section always names the
specific v0.2 failure mode it prevents — that's the bar for proposing a
new skill.

---

## 4. What "benchmark" actually means here

A single benchmark in this project = **one full closed loop** done by
`scripts/run_experiment.py`. It's not a single command; it's a state machine.

### 4.1 What runs

```
launch_server.py        →  spawn `conda run -n sglang-dev python -m sglang.launch_server
                                 --model-path /data/hf/models/Qwen3-0.6B
                                 --tensor-parallel-size 1
                                 --mem-fraction-static 0.7
                                 --schedule-policy lpm
                                 --max-running-requests 32
                                 --chunked-prefill-size -1
                                 --max-prefill-tokens 16384
                                 --trust-remote-code
                                 --host 127.0.0.1 --port 30000`
                            (env: CUDA_VISIBLE_DEVICES=0, CUDA_HOME=conda env,
                                  HF_HOME=/data/hf/gujialiang123/hf_cache)

wait_ready.py           →  poll /health, /v1/models, then TCP until ready
                            (typically ~50 s on H200 + 0.6B model)

run_benchmark.py        →  spawn `conda run -n sglang-dev python -m sglang.bench_serving
                                 --backend sglang
                                 --host 127.0.0.1 --port 30000
                                 --model /data/hf/models/Qwen3-0.6B
                                 --dataset-name random
                                 --random-input-len 128 --random-output-len 32
                                 --num-prompts 160
                                 --max-concurrency 16
                                 --output-file <run_dir>/quick_raw.jsonl
                                 --output-details --disable-tqdm --seed 1234
                                 --flush-cache`

parse_metrics.py        →  read quick_raw.jsonl (contains per-request ttfts[],
                            itls[], errors[], plus aggregate medians/p99)
                            + scan server.log for oom/crash signals
                          →  emit normalized quick_metrics.json with our
                            stable schema (ttft_p50/p95/p99, tpot_p50/p99,
                            itl_*, e2e_*, output_throughput, success_rate, ...)

kill_process_group()    →  SIGTERM the sglang server group; SIGKILL after 10 s
```

### 4.2 What's preserved per benchmark

Every benchmark leaves a complete forensic trail in its run directory:

```
experiments/tmp/regime_scout/20260528_023213/run_0004_scheduler_overhead_high_concurrency/
├── workload_input.yaml        # original workload yaml (byte-identical copy)
├── workload_snapshot.yaml     # second copy taken by run_experiment.py
├── config_snapshot.yaml       # the server config used
├── server.log                 # full sglang stdout+stderr: ServerArgs banner,
│                              #   model load timing, KV cache size, CUDA
│                              #   graph capture report, per-batch decode logs
├── quick_benchmark.log        # bench_serving stdout+stderr
├── quick_raw.jsonl            # bench_serving raw output (per-request ttfts,
│                              #   itls, errors, plus aggregate metrics)
├── quick_metrics.json         # our normalized schema (see scripts/parse_metrics.py)
├── server_features.json       # ← skill: server-log-mining
├── classification.json        # ← skill: failure-classification
└── orchestrator.log           # run_experiment.py's own stdout
```

### 4.3 Workload schema (what defines "one benchmark")

```yaml
# regime_scout/candidates/seed_03_scheduler_overhead_high_concurrency.yaml
name: scheduler_overhead_high_concurrency
regime_hint: scheduler_tail              # used by triage + scoring
dataset:
  name: random                            # or generated-shared-prefix
  random_input_len: 128
  random_output_len: 16
  random_range_ratio: 0.0
traffic:
  max_concurrency: 64
  num_prompts: 320
cache:
  mode: cold
  flush_cache: true                       # call /flush_cache before benchmark
seed: 1234
notes: "High concurrency stress for scheduler tail latency."
```

10 such files cover the 6 standard regimes
(sanity / scheduler_overhead / prefill / decode / prefix_cache / cache_churn).

### 4.4 Why this is more than a wrapper around bench_serving

`sglang.bench_serving` already produces per-request `ttfts[]` and `itls[]`.
The value this project adds:

1. **Normalized metrics schema** — bench_serving reports `p99_ttft_ms` but
   not `p95_ttft_ms`. We recompute p95 from the raw array and keep both,
   stable across SGLang versions.
2. **Server-log mining** — bench_serving doesn't know if the server hit
   `peak_queue_reqs=36` while `max_running_requests=32`. Our
   `server-log-mining` skill extracts that, which is the only way to
   distinguish "workload was genuinely hard" from "config hit a boundary".
3. **Closed-loop hygiene** — server is always launched fresh, port
   conflict checked, process group cleanly killed even on crash,
   HF_HOME pinned to a writable location, CUDA_HOME set so CUDA graph
   capture works.
4. **Adaptive suite generation** — Stage 1 doesn't just run 10 hand-picked
   workloads; it runs them, sees what's suspicious, and **automatically
   generates 2-3 neighbor workloads along the axis most likely to explain
   the symptom**, then re-scores.

---

## 5. Step-by-step what happens inside `explore_regimes.py`

```
                       ┌──────────────────────────────────┐
                       │  configs/base.yaml               │
                       │  regime_scout/seed_suite.yaml    │
                       │  regime_scout/search_space.yaml  │
                       └─────────────────┬────────────────┘
                                         │
                  ┌──────────────────────┴─────────────────────┐
                  ▼                                            │
   STEP A   generate_seed_suite.py                             │
            seed_suite.yaml (10 seed defs) →                   │
            regime_scout/candidates/seed_*.yaml                │
                  │                                            │
                  ▼                                            │
   STEP B   run_regime_suite.py  (WAVE 0)                      │
            for each candidate yaml:                           │
              run_experiment.py (launch+wait+bench+parse)      │
              append one row to raw_results.jsonl              │
            ~70 s/workload × 10 = ~12 min                      │
                  │                                            │
                  ▼                                            │
   STEP C   suspicion-scoring/impl/score.py  (skill chain)     │
            for each row in raw_results.jsonl:                 │
              call server-log-mining → server_features.json    │
              call failure-classification → classification.json│
              score = w1·local_nonlinearity                    │
                    + w2·tail_latency_ratio (noise-adjusted)   │
                    + w3·server_log_signal                     │
                    + w4·failure_class_score                   │
            write suspicious_cases.jsonl sorted by score       │
                  │                                            │
                  ▼                                            │
   STEP D   triage(scored_rows)  (rule-based, in-process)      │
            for each row:                                      │
              if concurrency_capped or cuda_graph_too_small:   │
                  plan = bracket(max_concurrency)              │
              elif at_capacity or near_capacity:               │
                  plan = upward(input_len)                     │
              elif lonely cluster and score ≥ 0.1:             │
                  plan = bracket(hint's natural axis)          │
            emit plans (max 1 per workload+axis)               │
                  │                                            │
                  ▼                                            │
   STEP E   boundary-expansion/impl/expand.py                  │
            for each plan:                                     │
              read parent workload yaml                        │
              find current value on the chosen axis            │
              pick N neighbor values from search_space.yaml    │
              clone parent yaml, swap that one axis            │
              write to candidates/expanded/                    │
                  │                                            │
                  ▼                                            │
   STEP F   run_regime_suite.py  (WAVE 1)                      │
            same as STEP B but on candidates/expanded/         │
            ~70 s × N neighbors                                │
                  │                                            │
                  ▼                                            │
   STEP G   re-score (STEP C again, now with neighbors)        │
            local_nonlinearity component activates             │
            scores update                                      │
                  │                                            │
                  ▼                                            │
   STEP H   cluster_regimes.py                                 │
            group by regime_hint                               │
            emit regime_map.md + regime_map.json               │
                  │                                            │
                  ▼                                            │
   STEP I   select_cases_for_stage2.py                         │
            top-k passed cases above --threshold (default 0.30)│
            for each: build case.json + workload.yaml in       │
            experiments/regimes/cases/SNNN/                    │
                  │                                            │
                  ▼                                            │
                STOP ──────────────────────────────────────────┘
```

Wave count is controllable via `--max-waves` (default 2). The same script
can be called with `--reuse-seed-run` to skip wave 0 and just re-score
existing data, useful for iterating on the scoring function.

---

## 6. Reproducing the 2026-05-28 result

### Environment we used

| Item | Value |
|---|---|
| Hardware | 8 × NVIDIA H200 (143 GB each); we used GPU 0 only |
| Model | Qwen3-0.6B, bf16, ~1.2 GB on disk |
| Model path | `/data/hf/models/Qwen3-0.6B` |
| SGLang | 0.5.12.post1 (source at `/home/t-jialianggu/work/sglang`) |
| Conda env | `sglang-dev` (provides nvcc, libcudart, sglang, torch 2.x w/ CUDA 12.8) |
| HF cache | `/data/hf/gujialiang123/hf_cache` (user-writable; default `/data/hf/hub` is owned by another user) |

### Exact reproduction command

```bash
cd /home/t-jialianggu/work/EndtoEnd-auto-optimization
conda activate sglang-dev      # the script also re-activates internally via `conda run`
python scripts/explore_regimes.py \
    --config configs/base.yaml \
    --max-waves 2 \
    --threshold 0.30 \
    --max-cases 5
```

### Wave 0 results (10 seed workloads, 11 min 48 s)

| workload                              | mc | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99/p50 | TPOT p50 (ms) | out tps | req tps |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke                                 | 4  | 24   | 99   | 4.06 | 3.42  | 629    | 46  |
| tiny_latency                          | 1  | 12   | 14   | 1.43 | 1.67  | 187    | 66  |
| short_in_short_out                    | 16 | 34   | 99   | 2.90 | 7.25  | 1639   | 99  |
| **scheduler_overhead_high_concurrency** | **64** | **129** | **434** | **3.75** | **16.24** | **1835** | **211** |
| prefill_medium                        | 4  | 33   | 50   | 2.77 | 4.12  | 482    | 55  |
| prefill_long (16k input)              | 2  | 73   | 120  | 1.89 | 2.71  | 174    | 18  |
| decode_medium                         | 16 | 19   | 96   | 5.01 | 2.51  | 5759   | 22  |
| decode_heavy                          | 32 | 23   | 123  | 5.42 | 2.78  | 10141  | 19  |
| prefix_reuse_ideal                    | 16 | 123  | 177  | 1.50 | 3.90  | 3282   | 26  |
| prefix_churn                          | 16 | 65   | 185  | 3.11 | 4.56  | 3126   | 24  |

All 10 passed, 0 OOM, 0 crash, 0 timeout.

### Triage output

The triage rules produced 9 expansion plans. The first (and the one we
report on here) was for `scheduler_overhead_high_concurrency`:

```
PLAN: expand scheduler_overhead_high_concurrency along max_concurrency
      (strategy=bracket) — reason: concurrency_capped per server-log-mining
```

### Wave 1 results (2 neighbors, 2 min 15 s)

| workload (same regime, mc varied) | TTFT p50 (ms) | TTFT p95 (ms) | out tps |
|---|---:|---:|---:|
| `...__con_16` (mc=16)                 | 38  | 96   | 1085 |
| `...__con_32` (mc=32)                 | 41  | 122  | 1622 |
| parent (mc=64)                        | 129 | **434** | 1835 |

The cliff is clearly between mc=32 and mc=64.

### Final scoring (after both waves)

```
scheduler_overhead_high_concurrency        score=0.735  class=load_shed_concurrency
  ├─ server_log_signal:       0.7  (concurrency_capped, peak_queue=36, max_running=32)
  ├─ failure_class:           0.7  (load_shed_concurrency)
  ├─ tail_latency_ratio:      1.0  (ttft p99/p50 = 3.75)
  └─ local_nonlinearity:      1.0  (3.57× worse than mc=32 neighbor)

(2nd place was 0.150 — 5× lower)
```

### Selected case S001

```bash
$ cat experiments/regimes/cases/S001/case.json
{
  "case_id": "S001",
  "regime_id": "R_scheduler_tail",
  "model_path": "/data/hf/models/Qwen3-0.6B",
  "hardware": "H200",
  "symptom": {
    "metric": "ttft_p95_ms",
    "observed_value": 434.23,
    "direction": "lower"
  },
  "evidence": {
    "components": {
      "server_log_signal":  {"score": 0.7, "evidence": {...peak_queue=36, max_running=32...}},
      "failure_class":      {"score": 0.7, "evidence": {"classification": "load_shed_concurrency"}},
      "tail_latency_ratio": {"score": 1.0, "evidence": {"max_ratio": 3.75}},
      "local_nonlinearity": {"score": 1.0, "evidence": {"ratio_worse_than_nb": 3.569}}
    }
  },
  "recommended_stage2": {
    "suggested_first_knobs": [
      "cuda-graph-max-bs",
      "max-running-requests",
      "num-continuous-decode-steps"
    ]
  },
  "frozen": true
}
```

This is what Stage 2 will consume — when we implement it.

---

## 7. Repository layout

```
EndtoEnd-auto-optimization/
│
├── README.md                       ← you are here
├── DESIGN.md                       ← v0.2 detailed spec (27 chapters + Amendments)
├── SKILLS.md                       ← skills design principles + catalog
├── LOGS.md                         ← log architecture
├── TWO_STAGE_AGENT_SUPPLEMENT.md   ← older two-stage design (reference; superseded by §2)
│
├── configs/
│   └── base.yaml                   ← sglang server config (edit model-path here)
│
├── regime_scout/
│   ├── seed_suite.yaml             ← 10 hand-picked seed workloads
│   ├── search_space.yaml           ← axis values + score thresholds
│   ├── candidates/
│   │   ├── seed_00..09_*.yaml      ← generated from seed_suite.yaml
│   │   └── expanded/               ← generated by boundary-expansion skill
│   └── outputs/                    ← final products
│       ├── raw_results.jsonl
│       ├── suspicious_cases.jsonl
│       ├── regime_map.{md,json}
│       └── selected_cases.jsonl
│
├── scripts/                        ← foundational harness
│   ├── utils.py                    ← shared: yaml/json IO, conda-run wrapper,
│   │                                  env builder, log scanners, argv translation
│   ├── logging_setup.py            ← structured file+stdout logger
│   ├── launch_server.py            ← YAML → sglang argv (DESIGN §0.G B1 fix)
│   ├── wait_ready.py               ← poll until /health is OK
│   ├── run_benchmark.py            ← call sglang.bench_serving with workload yaml
│   ├── parse_metrics.py            ← bench_serving jsonl → normalized metrics.json
│   ├── run_experiment.py           ← one workload, end-to-end
│   ├── generate_seed_suite.py      ← seed_suite.yaml → candidates/seed_*.yaml
│   ├── run_regime_suite.py         ← run a directory of workloads, write raw_results.jsonl
│   ├── score_suspicion.py          ← v1 scorer (legacy; fallback)
│   ├── cluster_regimes.py          ← raw + scored → regime_map.md
│   ├── select_cases_for_stage2.py  ← top-k → cases/SNNN/{case.json,workload.yaml}
│   ├── run_stage1.py               ← v0.2 one-shot (legacy)
│   └── explore_regimes.py          ← v0.3 one-shot ← USE THIS
│
├── .github/skills/                 ← reusable methodology units
│   ├── _template/SKILL.md          ← skill author template
│   ├── server-log-mining/          ← parse server.log → 28 fields
│   ├── failure-classification/     ← (metrics, features) → enum
│   ├── noise-aware-scoring/        ← CV calibration + adjusted_threshold()
│   ├── boundary-expansion/         ← yaml + axis → neighbor yamls
│   ├── suspicion-scoring/          ← v2 scorer (composes 4 skills above)
│   └── minimal-repro-shrink/       ← SKILL.md only; impl deferred
│
├── experiments/
│   ├── regimes/
│   │   ├── STAGE1_REPORT_20260528.md   ← human analysis of the first real run
│   │   └── cases/
│   │       └── S001/                    ← frozen case for Stage 2 to consume
│   └── tmp/regime_scout/<timestamp>/   ← per-suite run dirs; never overwritten
│       └── run_NNNN_<name>/             ← full forensic trail per workload
│
└── logs/                            ← suite-level logs (timestamped, never overwritten)
```

---

## 8. Single-skill usage (advanced)

Every skill is also a standalone CLI; useful for debugging.

```bash
# Mine a single server.log
python .github/skills/server-log-mining/impl/parse_server_log.py \
    --server-log experiments/tmp/regime_scout/<ts>/run_0004_*/server.log \
    --out /tmp/features.json

# Classify a single run
python .github/skills/failure-classification/impl/classify.py \
    --metrics experiments/tmp/.../quick_metrics.json \
    --features /tmp/features.json \
    --out /tmp/classification.json

# Expand a workload along one axis
python .github/skills/boundary-expansion/impl/expand.py \
    --parent regime_scout/candidates/seed_03_*.yaml \
    --axis max_concurrency \
    --strategy bracket \
    --neighbors-out regime_scout/candidates/expanded/

# Re-score everything currently in raw_results.jsonl
python .github/skills/suspicion-scoring/impl/score.py \
    --noise-baseline experiments/noise_baseline.json \
    --force-mine

# Calibrate noise baseline (5 repeats of one workload)
python .github/skills/noise-aware-scoring/impl/calibrate_noise.py \
    --config configs/base.yaml \
    --workload regime_scout/candidates/seed_00_smoke.yaml \
    --repeats 5 \
    --out experiments/noise_baseline.json
```

---

## 9. What's NOT in this repo

- **Stage 2 (Diagnoser) and Stage 3 (Fixer)** — designed in
  [`TWO_STAGE_AGENT_SUPPLEMENT.md`](./TWO_STAGE_AGENT_SUPPLEMENT.md) and
  refined into three stages in §2 above. Not implemented yet.
- **Copilot custom agent packaging** — currently you call skills via
  python scripts directly. Wrapping each stage as a Copilot agent (so
  the LLM can drive the loop) is on the v0.4 roadmap.
- **Production deployment** — this is a research harness, not a serving
  framework.
- **Multi-node distributed SGLang**, **kernel-level rewrites**, **model
  quantization** — explicitly out of scope.

---

## 10. Further reading

- [`DESIGN.md`](./DESIGN.md) — full v0.2 spec, 27 chapters
- [`SKILLS.md`](./SKILLS.md) — skills design principles & how to add new ones
- [`LOGS.md`](./LOGS.md) — log layout
- [`experiments/regimes/STAGE1_REPORT_20260528.md`](./experiments/regimes/STAGE1_REPORT_20260528.md) — human analysis of the first real run, including a candid evaluation of where the v0.2 scoring function failed
- Each skill's `SKILL.md` under `.github/skills/<name>/`
- SGLang docs: [server args](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md) · [benchmark](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md)

---

## 11. Where the LLM is (and isn't): rule-based search vs. agent

A reasonable question after reading §5 is: **"this looks like a deterministic
script, not an agent — where's the LLM?"** The honest answer:

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage A  Problem-Setter                                         │
│   Runtime: TWO MODES                                            │  ← TODAY
│                                                                 │
│   Mode A (rule-based reference policy, headless)                │
│     stages/problem-setter/policies/rule_based_explore.py        │
│     Deterministic, CI-friendly, cheap baseline.                 │
│                                                                 │
│   Mode B (LLM agent driven, e.g. Claude Code / Copilot CLI)     │
│     stages/problem-setter/policies/llm_agent.md (system prompt) │
│     LLM picks workloads, triage axes, decides when to L4-profile│
│     and L5-repeat. Calls the same harness tools as Mode A.      │
│                                                                 │
│   Both modes produce the same artifact:                         │
│     experiments/problems/PNNN/  (a frozen problem package)      │
└─────────────────────────────────────────────────────────────────┘
                            ↓  problem package (typed contract)
┌─────────────────────────────────────────────────────────────────┐
│ Stage B  Problem-Solver Fleet  (NOT YET BUILT)                  │
│                                                                 │
│   Sub-agents: config-agent / scheduler-agent / kernel-agent /   │
│               workload-shape-agent                              │
│   Runtime: LLM AGENT per sub-agent + STRICT HARNESS guard       │
│                                                                 │
│   LLM decides:    which knob/patch to try, hypothesis, risk     │
│   Harness enforces: writes only to attempts/, runs full bench   │
│                     suite (target+neighbors+controls), A/B,     │
│                     keep/revert by deterministic rule.          │
└─────────────────────────────────────────────────────────────────┘
```

### Why the Setter has a rule-based fallback on purpose

1. **Reproducibility.** Same `raw_results.jsonl` → same scores, every time.
   An LLM-only scorer would give different scores on different days and we'd
   never know if a score change came from the data or the model.
2. **Speed and cost.** Scoring 10 workloads is milliseconds in Python;
   calling an LLM for each would be seconds and burn tokens.
3. **Auditability.** Every Stage A score traces back to specific bytes in
   specific log files — the P001 evidence trail in §6 has 7 key signals,
   each citing a JSON file. LLM internal reasoning is much harder to audit.
4. **LLMs don't add value at the mechanics layer.** Detecting
   `queue=36 ∧ max_running=32 → concurrency_capped` is one regex match
   plus a comparison. An LLM wouldn't do it better; it would do it slower
   and non-deterministically.

### Where the LLM has actually been so far

LLM (me, in this chat) has done **design work**:
- Drafted the two-stage architecture
- Designed the skills system + wrote every `SKILL.md`
- Wrote all the Python implementations
- Wrote the rule-based reference policy's triage rules

When you run `python stages/problem-setter/policies/rule_based_explore.py`,
**there is no LLM at runtime**. When you drive Mode B via Claude Code or
Copilot CLI, the LLM is the **policy layer** deciding what to do next; it
still calls the same harness tools to actually do the work.

### What "Copilot CLI agent" actually means in v0.4

The v0.4 roadmap item "Copilot agent packaging" is mostly about polishing
Mode B's system prompt (`policies/llm_agent.md`) and exposing the setter
as `@problem-setter`:

```bash
# Today (Mode A, deterministic)
$ python stages/problem-setter/policies/rule_based_explore.py \
    --config configs/base.yaml

# Today (Mode B, manual prompt load)
$ claude code   # then load stages/problem-setter/policies/llm_agent.md as system prompt

# v0.4 (planned, sugar)
$ copilot -p "@problem-setter run a session on configs/moe_qwen3_30b.yaml,
              then summarize the top 3 problems for me"
```

### When does the LLM start doing real reasoning?

**Stage B onward**. That's the design. Concretely:

- **Solver LLM input**: a problem package — `workload.yaml`,
  `baseline_metrics.json`, `server_features.json`, `classification.json`,
  optionally `profile_summary.json`, `hypothesis.md`,
  `suggested_strategies` in `problem.json`, plus neighbors and controls
  for verification.
- **Solver LLM output per attempt**: a single knob (or kernel patch)
  + plan.md + decision.json. The harness then runs the benchmark suite
  and decides keep/revert.

So: **rules for mechanics, LLMs for semantics, contracts in between**.
That separation is the whole point of the two-stage design.

### Will the Setter ever use LLM-only?

Both modes already coexist. What we will *not* do:

- Replace the rule-based score function with an LLM call (sacrifices
  reproducibility).
- Let the LLM directly write to `seed_suite.yaml` / `search_space.yaml`
  / `rule_based_explore.py` mid-session (extensions go via
  `stages/problem-setter/proposals/...` for human review).

**一句话目标**：给一个模型、一台 GPU、一个 SGLang，自动找出哪些
serving 场景（input/output 长度 + 并发 + 缓存模式）会暴露性能悬崖，
并把每个悬崖打包成 case 交给下游 agent 去诊断、修复。

**当前状态（2026-05-28）**：Stage 1 (RegimeScout) 端到端可工作。在
H200 + Qwen3-0.6B 上的第一次真实运行**自动发现了一个真实 bug**
(`max-running-requests=32` 卡住了 `max_concurrency=64`，请求队列堆到 36 深，
TTFT p95 从 99 ms 跳到 434 ms)。Stage 2 / Stage 3 设计完成，未实现。

> ⚠️ **重要 —— 这里说的 "agent" 是啥意思？** Stage 1 在运行时**故意
> 是 rule-based 的**（确定性脚本 + 一组 4 条 triage 规则），整个 scout
> 跑的时候**没有 LLM 在 loop 里**。LLM agent 真正会上场的位置是计划
> 中的 **Stage 2 (诊断)** 和 **Stage 3 (修复)**——这两个地方需要对
> evidence 做语义推理。设计理由见 §2，各阶段 "agent" 的含义见 §11。

---

## 1. TL;DR — 真的能一行命令跑吗？

可以。

```bash
# 0. 激活环境（一次）
conda activate sglang-dev

# 1. 在 configs/base.yaml 里填你的模型路径（一次）
$EDITOR configs/base.yaml          # 把 model-path 改成 /你的/模型

# 2. 启动
python scripts/explore_regimes.py --config configs/base.yaml
```

就这样。大约 15 分钟后你会有：

- `regime_scout/outputs/regime_map.md`           — 人类可读的总览
- `regime_scout/outputs/selected_cases.jsonl`    — 冻结的问题 case（给下游 stage 用）
- `experiments/regimes/cases/S001/case.json`     — 每个 case 的完整证据链
- `logs/explore_<时间戳>.log`                    — 编排日志
- `experiments/tmp/regime_scout/<时间戳>/` 下 12 个 per-workload 目录，
  每个含 `server.log`, `quick_raw.jsonl`, `quick_metrics.json`,
  `server_features.json`, `classification.json`

`explore_regimes.py` 内部具体跑哪些步骤，看 **§5 一步步拆解**。

---

## 2. 心智模型：两阶段

```
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 A  Problem-Setter（出题人 Agent）                              │
│           "找出值得优化的 regime,                                   │
│            证明 cliff 真实存在,                                     │
│            打包成自包含的题目"                            ← 当前仓库实现的部分
│                                                                     │
│ 阶段 B  Problem-Solver Fleet（做题人 Agent 群）                     │
│           "config-agent / scheduler-agent / kernel-agent /          │
│            workload-shape-agent —— 各自在自己的领域解题"  ← 设计完成, 未实现
└─────────────────────────────────────────────────────────────────────┘
```

**算法竞赛比喻**：出题人准备题目（带 cliff 真实性的证据），做题人 fleet
从架上拾题写解答。每个题目包**自包含**：target workload + 边界扩展邻居
+ 无关 controls + 证据 + 假设 + 验收标准 + 做题人所有 attempt + 最终
报告——全在一个目录 `experiments/problems/PNNN/`。可以 `tar` 起来交给
任何人。

### 为什么是两阶段而不是三阶段

之前的草稿（在 [`archive/`](./archive/)）把发现和诊断切开。那是错的：
**诊断本身决定发现的方向**。没 profile 之前你判断不出来该扩
`max_concurrency` 还是 `input_len`。切开两阶段就破坏了反馈环。合并后
的出题人可以自由交错：bench → log-mine → score → profile → 扩边界 →
重新打分。

详情和契约：

- 架构总览（双语）：[`docs/architecture/two-stage-overview.md`](./docs/architecture/two-stage-overview.md)
- 题目包 schema：[`docs/problem-package/schema.md`](./docs/problem-package/schema.md)
- Idea 池（双向通道）：[`docs/idea-pool/schema.md`](./docs/idea-pool/schema.md)

出题人和做题人之间的交接是 **JSON 契约**，所以今天你可以用 Claude Code
驱动出题人、用 Copilot CLI 驱动做题人（或反过来）；明天我们再加一个
统一 CLI 把两端串起来。

---

## 3. Skills — 可复用的方法论单元

Skills 是这个项目的核心价值。每个 skill 是一个命名的、单一职责的、
跨 agent 跨 stage 通用的知识单元。`server-log-mining` 这个 skill 既被
出题人用、也被做题人 fleet 的每个子 agent 用。**方法论写一次，永久
复用**。

设计原则和"如何加新 skill"见 [`SKILLS.md`](./SKILLS.md)。当前清单：

| Skill | Stages | 做什么 | 状态 |
|---|---|---|---|
| `server-log-mining` | 1, 2, 3 | 把 sglang 的 `server.log` 解析成 28 个字段，包括关键的 derived bool 比如 `concurrency_capped`、`cuda_graph_too_small`、`at_capacity`。 | ✅ 已实现 |
| `failure-classification` | 1, 2, 3 | 把 (metrics + features) 映射到 10 个 enum：`clean_pass` / `load_shed_concurrency` / `near_failure_kv` / `oom` / ... | ✅ 已实现 |
| `noise-aware-scoring` | 1, 3 | 把同一个 baseline workload 跑 N 次，算每个指标的 CV，给 `adjusted_threshold()` 用。 | ✅ 已实现 |
| `boundary-expansion` | 1 | 给定一个 workload + 一个 axis，沿该 axis 生成 N 个邻居 workload yaml。4 种策略：bracket / upward / downward / geometric。 | ✅ 已实现 |
| `suspicion-scoring` | 1 | 把上面 4 个 skill 组合成每个 run 一个 score，附完整 evidence trail。 | ✅ 已实现 |
| `minimal-repro-shrink` | 1, 2 | 二分式缩小 workload 直到症状消失。 | 🟨 仅 SKILL.md，实现延后 |

每个 `SKILL.md` 都是固定六节结构：**WHEN（何时调用）/ WHY（为什么存在）
/ HOW（怎么调用）/ OUTPUT CONTRACT（输出契约）/ FAILURE MODES（失败模式）
/ ROADMAP（后续规划）**。WHY 一节必须指名"这个 skill 防止 v0.2 的哪个
具体失败"——这是新 skill 提案的标准。

---

## 4. 这里的 "benchmark" 到底是什么

这个项目里的一次 benchmark = `scripts/run_experiment.py` 跑的**一个
完整闭环**。不是单个命令，是一个状态机。

### 4.1 实际跑的内容

```
launch_server.py    →  起 `conda run -n sglang-dev python -m sglang.launch_server
                            --model-path /data/hf/models/Qwen3-0.6B
                            --tensor-parallel-size 1
                            --mem-fraction-static 0.7
                            --schedule-policy lpm
                            --max-running-requests 32
                            --chunked-prefill-size -1
                            --max-prefill-tokens 16384
                            --trust-remote-code
                            --host 127.0.0.1 --port 30000`
                        (env: CUDA_VISIBLE_DEVICES=0, CUDA_HOME=conda env,
                              HF_HOME=/data/hf/gujialiang123/hf_cache)

wait_ready.py       →  轮询 /health, /v1/models, 最后 TCP，直到 ready
                        (H200 + 0.6B 模型大约 50 秒)

run_benchmark.py    →  起 `conda run -n sglang-dev python -m sglang.bench_serving
                            --backend sglang
                            --host 127.0.0.1 --port 30000
                            --model /data/hf/models/Qwen3-0.6B
                            --dataset-name random
                            --random-input-len 128 --random-output-len 32
                            --num-prompts 160
                            --max-concurrency 16
                            --output-file <run_dir>/quick_raw.jsonl
                            --output-details --disable-tqdm --seed 1234
                            --flush-cache`

parse_metrics.py    →  读 quick_raw.jsonl（含 per-request 的 ttfts[], itls[],
                        errors[] 加上聚合 median/p99）+ 扫 server.log
                        找 oom/crash 信号
                      →  emit 标准化的 quick_metrics.json（schema 稳定：
                        ttft_p50/p95/p99, tpot_p50/p99, itl_*, e2e_*,
                        output_throughput, success_rate, ...）

kill_process_group()→  SIGTERM 整个 sglang 进程组，10 秒后 SIGKILL
```

### 4.2 每次 benchmark 保留什么

每次 benchmark 都在自己的 run_dir 留下完整的取证记录：

```
experiments/tmp/regime_scout/20260528_023213/run_0004_scheduler_overhead_high_concurrency/
├── workload_input.yaml        # 原 workload yaml（字节级副本）
├── workload_snapshot.yaml     # run_experiment.py 自己再写一份（冗余兜底）
├── config_snapshot.yaml       # 用的 server 配置
├── server.log                 # sglang 完整 stdout+stderr：ServerArgs banner、
│                              #   模型加载时间、KV cache 大小、CUDA graph
│                              #   capture 报告、per-batch decode 日志
├── quick_benchmark.log        # bench_serving 的 stdout+stderr
├── quick_raw.jsonl            # bench_serving 原始输出（per-request 的 ttfts,
│                              #   itls, errors, 加上聚合 metrics）
├── quick_metrics.json         # 我们标准化的 schema（见 scripts/parse_metrics.py）
├── server_features.json       # ← skill: server-log-mining 产物
├── classification.json        # ← skill: failure-classification 产物
└── orchestrator.log           # run_experiment.py 自己的 stdout
```

### 4.3 Workload schema（一次 benchmark 由什么定义）

```yaml
# regime_scout/candidates/seed_03_scheduler_overhead_high_concurrency.yaml
name: scheduler_overhead_high_concurrency
regime_hint: scheduler_tail              # triage + scoring 都会读
dataset:
  name: random                            # 或 generated-shared-prefix
  random_input_len: 128
  random_output_len: 16
  random_range_ratio: 0.0
traffic:
  max_concurrency: 64
  num_prompts: 320
cache:
  mode: cold
  flush_cache: true                       # benchmark 前调 /flush_cache
seed: 1234
notes: "High concurrency stress for scheduler tail latency."
```

我们写了 10 个这样的 file，覆盖 6 种标准 regime
(sanity / scheduler_overhead / prefill / decode / prefix_cache / cache_churn)。

### 4.4 为什么这不只是 bench_serving 的薄壳

`sglang.bench_serving` 已经能产 per-request 的 `ttfts[]` 和 `itls[]`。
这个项目额外做的：

1. **标准化的 metrics schema** —— bench_serving 报 `p99_ttft_ms` 但不报
   `p95_ttft_ms`。我们从 raw 数组里自算 p95 并同时保留两者，跨 SGLang
   版本 schema 稳定。
2. **Server log mining** —— bench_serving 不知道 server 是不是出现了
   `peak_queue_reqs=36` 而 `max_running_requests=32` 的情况。我们的
   `server-log-mining` skill 把这个抽出来——这是唯一能区分"workload
   真的难"和"配置撞了边界"的方法。
3. **闭环卫生** —— 每次都新启 server，端口冲突检查、进程组干净
   kill（即使 crash 也保证）、HF_HOME pin 到可写位置、CUDA_HOME 设好
   让 CUDA graph capture 不爆。
4. **自适应 suite 生成** —— Stage 1 不只是跑 10 个手写 workload；它
   跑完看哪些可疑，**自动沿着最可能解释症状的 axis 生成 2-3 个邻居
   workload**，再重新打分。

---

## 5. `explore_regimes.py` 一步步拆解

```
                       ┌──────────────────────────────────┐
                       │  configs/base.yaml               │
                       │  regime_scout/seed_suite.yaml    │
                       │  regime_scout/search_space.yaml  │
                       └─────────────────┬────────────────┘
                                         │
                  ┌──────────────────────┴─────────────────────┐
                  ▼                                            │
   STEP A   generate_seed_suite.py                             │
            seed_suite.yaml (10 个 seed 定义) →                │
            regime_scout/candidates/seed_*.yaml                │
                  │                                            │
                  ▼                                            │
   STEP B   run_regime_suite.py  (WAVE 0)                      │
            对每个 candidate yaml:                             │
              run_experiment.py (launch+wait+bench+parse)      │
              往 raw_results.jsonl 追加一行                    │
            约 70 秒/workload × 10 = 约 12 分钟                │
                  │                                            │
                  ▼                                            │
   STEP C   suspicion-scoring/impl/score.py  (skill 链)        │
            对 raw_results.jsonl 每一行:                       │
              调 server-log-mining → server_features.json      │
              调 failure-classification → classification.json  │
              score = w1·local_nonlinearity                    │
                    + w2·tail_latency_ratio (noise 调整后)     │
                    + w3·server_log_signal                     │
                    + w4·failure_class_score                   │
            按 score 降序写 suspicious_cases.jsonl             │
                  │                                            │
                  ▼                                            │
   STEP D   triage(scored_rows)  (规则化, 在进程内)            │
            对每一行:                                          │
              if concurrency_capped 或 cuda_graph_too_small:   │
                  plan = bracket(max_concurrency)              │
              elif at_capacity 或 near_capacity:               │
                  plan = upward(input_len)                     │
              elif 孤立 cluster 且 score ≥ 0.1:                │
                  plan = bracket(该 hint 的自然 axis)          │
            产 plan（每个 (workload, axis) 最多 1 条）          │
                  │                                            │
                  ▼                                            │
   STEP E   boundary-expansion/impl/expand.py                  │
            对每个 plan:                                       │
              读 parent workload yaml                          │
              在所选 axis 上查当前值                           │
              从 search_space.yaml 挑 N 个邻居取值             │
              clone parent yaml，只换那一个 axis               │
              写到 candidates/expanded/                        │
                  │                                            │
                  ▼                                            │
   STEP F   run_regime_suite.py  (WAVE 1)                      │
            同 STEP B，但跑 candidates/expanded/               │
            约 70 秒 × N 个邻居                                │
                  │                                            │
                  ▼                                            │
   STEP G   重新打分 (重复 STEP C，现在有邻居了)               │
            local_nonlinearity 组件激活                        │
            score 更新                                         │
                  │                                            │
                  ▼                                            │
   STEP H   cluster_regimes.py                                 │
            按 regime_hint 聚类                                │
            emit regime_map.md + regime_map.json               │
                  │                                            │
                  ▼                                            │
   STEP I   select_cases_for_stage2.py                         │
            高于 --threshold (默认 0.30) 的 top-k passed case  │
            为每个 case 在 experiments/regimes/cases/SNNN/     │
            建 case.json + workload.yaml                       │
                  │                                            │
                  ▼                                            │
                STOP ──────────────────────────────────────────┘
```

Wave 数量由 `--max-waves`（默认 2）控制。`--reuse-seed-run` 可以跳过
wave 0 只对已有数据 re-score，方便迭代 scoring 函数。

---

## 6. 复现 2026-05-28 的结果

### 我们用的环境

| 项 | 值 |
|---|---|
| 硬件 | 8 × NVIDIA H200（每张 143 GB），只用了 GPU 0 |
| 模型 | Qwen3-0.6B，bf16，磁盘约 1.2 GB |
| 模型路径 | `/data/hf/models/Qwen3-0.6B` |
| SGLang | 0.5.12.post1（源码在 `/home/t-jialianggu/work/sglang`） |
| Conda env | `sglang-dev`（提供 nvcc / libcudart / sglang / torch 2.x + CUDA 12.8） |
| HF cache | `/data/hf/gujialiang123/hf_cache`（你的写权限路径；默认的 `/data/hf/hub` 是别的 user 的） |

### 精确复现命令

```bash
cd /home/t-jialianggu/work/EndtoEnd-auto-optimization
conda activate sglang-dev      # 脚本内部也会通过 `conda run` 再 activate 一遍
python scripts/explore_regimes.py \
    --config configs/base.yaml \
    --max-waves 2 \
    --threshold 0.30 \
    --max-cases 5
```

### Wave 0 结果（10 个 seed workload，11 min 48 s）

| workload                              | mc | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99/p50 | TPOT p50 (ms) | out tps | req tps |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke                                 | 4  | 24   | 99   | 4.06 | 3.42  | 629    | 46  |
| tiny_latency                          | 1  | 12   | 14   | 1.43 | 1.67  | 187    | 66  |
| short_in_short_out                    | 16 | 34   | 99   | 2.90 | 7.25  | 1639   | 99  |
| **scheduler_overhead_high_concurrency** | **64** | **129** | **434** | **3.75** | **16.24** | **1835** | **211** |
| prefill_medium                        | 4  | 33   | 50   | 2.77 | 4.12  | 482    | 55  |
| prefill_long (16k input)              | 2  | 73   | 120  | 1.89 | 2.71  | 174    | 18  |
| decode_medium                         | 16 | 19   | 96   | 5.01 | 2.51  | 5759   | 22  |
| decode_heavy                          | 32 | 23   | 123  | 5.42 | 2.78  | 10141  | 19  |
| prefix_reuse_ideal                    | 16 | 123  | 177  | 1.50 | 3.90  | 3282   | 26  |
| prefix_churn                          | 16 | 65   | 185  | 3.11 | 4.56  | 3126   | 24  |

10 个全部 pass，0 OOM，0 crash，0 timeout。

### Triage 输出

Triage 规则产了 9 个扩展计划。第一条（也是我们在本文重点报告的那条）
是对 `scheduler_overhead_high_concurrency`：

```
PLAN: 沿 max_concurrency 扩展 scheduler_overhead_high_concurrency
      (strategy=bracket) — 原因: server-log-mining 报 concurrency_capped
```

### Wave 1 结果（2 个邻居，2 min 15 s）

| workload（同 regime，只改 mc） | TTFT p50 (ms) | TTFT p95 (ms) | out tps |
|---|---:|---:|---:|
| `...__con_16` (mc=16)                 | 38  | 96   | 1085 |
| `...__con_32` (mc=32)                 | 41  | 122  | 1622 |
| parent (mc=64)                        | 129 | **434** | 1835 |

悬崖**精确卡在 mc=32 和 mc=64 之间**。

### 最终打分（两轮跑完之后）

```
scheduler_overhead_high_concurrency        score=0.735  class=load_shed_concurrency
  ├─ server_log_signal:       0.7  (concurrency_capped, peak_queue=36, max_running=32)
  ├─ failure_class:           0.7  (load_shed_concurrency)
  ├─ tail_latency_ratio:      1.0  (ttft p99/p50 = 3.75)
  └─ local_nonlinearity:      1.0  (比 mc=32 邻居慢 3.57×)

(第二名 score 才 0.150 —— 差 5 倍)
```

### 被选出的 case S001

```bash
$ cat experiments/regimes/cases/S001/case.json
{
  "case_id": "S001",
  "regime_id": "R_scheduler_tail",
  "model_path": "/data/hf/models/Qwen3-0.6B",
  "hardware": "H200",
  "symptom": {
    "metric": "ttft_p95_ms",
    "observed_value": 434.23,
    "direction": "lower"
  },
  "evidence": {
    "components": {
      "server_log_signal":  {"score": 0.7, "evidence": {...peak_queue=36, max_running=32...}},
      "failure_class":      {"score": 0.7, "evidence": {"classification": "load_shed_concurrency"}},
      "tail_latency_ratio": {"score": 1.0, "evidence": {"max_ratio": 3.75}},
      "local_nonlinearity": {"score": 1.0, "evidence": {"ratio_worse_than_nb": 3.569}}
    }
  },
  "recommended_stage2": {
    "suggested_first_knobs": [
      "cuda-graph-max-bs",
      "max-running-requests",
      "num-continuous-decode-steps"
    ]
  },
  "frozen": true
}
```

这就是 Stage 2 将来要消费的输入——等我们实现它的时候。

---

## 7. 仓库结构

```
EndtoEnd-auto-optimization/
│
├── README.md                       ← 你正在读
├── DESIGN.md                       ← v0.2 详细 spec（27 章 + Amendments）
├── SKILLS.md                       ← skills 设计原则 + catalog
├── LOGS.md                         ← log 架构说明
├── TWO_STAGE_AGENT_SUPPLEMENT.md   ← 旧的两阶段设计（参考用；已被 §2 替代）
│
├── configs/
│   └── base.yaml                   ← sglang server 配置（model-path 在这里改）
│
├── regime_scout/
│   ├── seed_suite.yaml             ← 10 个手写 seed workload
│   ├── search_space.yaml           ← axis 取值 + score 阈值
│   ├── candidates/
│   │   ├── seed_00..09_*.yaml      ← 从 seed_suite.yaml 生成
│   │   └── expanded/               ← boundary-expansion skill 自动生成
│   └── outputs/                    ← 最终产物
│       ├── raw_results.jsonl
│       ├── suspicious_cases.jsonl
│       ├── regime_map.{md,json}
│       └── selected_cases.jsonl
│
├── scripts/                        ← 基础 harness
│   ├── utils.py                    ← 共享：yaml/json IO, conda-run wrapper,
│   │                                  env builder, log scanner, argv 翻译
│   ├── logging_setup.py            ← 结构化 logger（file + stdout）
│   ├── launch_server.py            ← YAML → sglang argv（DESIGN §0.G B1 修复）
│   ├── wait_ready.py               ← 轮询 /health
│   ├── run_benchmark.py            ← 用 workload yaml 调 sglang.bench_serving
│   ├── parse_metrics.py            ← bench_serving jsonl → 标准化 metrics.json
│   ├── run_experiment.py           ← 一个 workload 完整闭环
│   ├── generate_seed_suite.py      ← seed_suite.yaml → candidates/seed_*.yaml
│   ├── run_regime_suite.py         ← 跑一个目录的 workload，写 raw_results.jsonl
│   ├── score_suspicion.py          ← v1 scorer（遗留，保留作 fallback）
│   ├── cluster_regimes.py          ← raw + scored → regime_map.md
│   ├── select_cases_for_stage2.py  ← top-k → cases/SNNN/{case.json,workload.yaml}
│   ├── run_stage1.py               ← v0.2 一键（遗留）
│   └── explore_regimes.py          ← v0.3 一键 ← 用这个
│
├── .github/skills/                 ← 方法论复用单元
│   ├── _template/SKILL.md          ← 新 skill 起手模板
│   ├── server-log-mining/          ← 解析 server.log → 28 个字段
│   ├── failure-classification/     ← (metrics, features) → enum
│   ├── noise-aware-scoring/        ← CV 校准 + adjusted_threshold()
│   ├── boundary-expansion/         ← yaml + axis → 邻居 yaml
│   ├── suspicion-scoring/          ← v2 scorer（组合上面 4 个 skill）
│   └── minimal-repro-shrink/       ← 仅 SKILL.md；实现延后
│
├── experiments/
│   ├── regimes/
│   │   ├── STAGE1_REPORT_20260528.md   ← 第一次跑的工程报告（人工分析）
│   │   └── cases/
│   │       └── S001/                    ← 给 Stage 2 消费的冻结 case
│   └── tmp/regime_scout/<时间戳>/      ← 每个 suite 一个；从不覆盖
│       └── run_NNNN_<name>/             ← 每个 workload 完整取证记录
│
└── logs/                            ← suite 级日志（带时间戳，从不覆盖）
```

---

## 8. 单 skill 用法（进阶）

每个 skill 都是独立的 CLI，调试时单跑很方便。

```bash
# 单独 mine 一份 server.log
python .github/skills/server-log-mining/impl/parse_server_log.py \
    --server-log experiments/tmp/regime_scout/<时间戳>/run_0004_*/server.log \
    --out /tmp/features.json

# 单独分类一个 run
python .github/skills/failure-classification/impl/classify.py \
    --metrics experiments/tmp/.../quick_metrics.json \
    --features /tmp/features.json \
    --out /tmp/classification.json

# 沿一个 axis 扩展一个 workload
python .github/skills/boundary-expansion/impl/expand.py \
    --parent regime_scout/candidates/seed_03_*.yaml \
    --axis max_concurrency \
    --strategy bracket \
    --neighbors-out regime_scout/candidates/expanded/

# 对 raw_results.jsonl 现有数据重新打分
python .github/skills/suspicion-scoring/impl/score.py \
    --noise-baseline experiments/noise_baseline.json \
    --force-mine

# 校准 noise baseline（5 次同 workload 重复）
python .github/skills/noise-aware-scoring/impl/calibrate_noise.py \
    --config configs/base.yaml \
    --workload regime_scout/candidates/seed_00_smoke.yaml \
    --repeats 5 \
    --out experiments/noise_baseline.json
```

---

## 9. 这个仓库**不**包含什么

- **Stage 2 (Diagnoser) 和 Stage 3 (Fixer)** —— 在
  [`TWO_STAGE_AGENT_SUPPLEMENT.md`](./TWO_STAGE_AGENT_SUPPLEMENT.md) 里
  设计过两阶段版本，§2 又细化成三阶段。**未实现**。
- **Copilot custom agent 包装** —— 目前所有 skill 都是人工跑 python
  脚本。把每个 stage 包成 Copilot agent（让 LLM 驱动循环）在 v0.4 roadmap。
- **生产部署** —— 这是研究 harness，不是 serving framework。
- **多机分布式 SGLang**、**kernel 级改写**、**量化** —— 明确 out of scope。

---

## 10. 进一步阅读

- [`DESIGN.md`](./DESIGN.md) — 完整 v0.2 spec，27 章
- [`SKILLS.md`](./SKILLS.md) — skills 设计原则 + 怎么加新 skill
- [`LOGS.md`](./LOGS.md) — log 布局
- [`experiments/regimes/STAGE1_REPORT_20260528.md`](./experiments/regimes/STAGE1_REPORT_20260528.md) — 第一次跑的工程报告，含 v0.2 scoring 函数失败的诚实评估
- 每个 skill 在 `.github/skills/<name>/` 下的 `SKILL.md`
- SGLang 官方文档：[server args](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md) · [benchmark](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md)

---

## 11. LLM 在哪儿，不在哪儿：rule-based search vs. agent

看完 §5 你很合理会问：**"这看起来就是个确定性脚本，agent 在哪？"**
诚实回答：

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 A  Problem-Setter                                          │
│   运行时: 两种模式                                              │  ← 当前
│                                                                 │
│   Mode A (rule-based 参考策略, headless)                        │
│     stages/problem-setter/policies/rule_based_explore.py        │
│     确定性, CI 友好, 廉价基线.                                  │
│                                                                 │
│   Mode B (LLM agent 驱动, 例如 Claude Code / Copilot CLI)       │
│     stages/problem-setter/policies/llm_agent.md (system prompt) │
│     LLM 选 workload、triage 轴、决定何时 L4 profile 和 L5 重复. │
│     调用跟 Mode A 一样的 harness 工具.                          │
│                                                                 │
│   两种模式产同样的产物:                                         │
│     experiments/problems/PNNN/  (冻结的题目包)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓  问题包 (强类型契约)
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 B  Problem-Solver Fleet  (尚未实现)                        │
│                                                                 │
│   子 agent: config-agent / scheduler-agent / kernel-agent /     │
│             workload-shape-agent                                │
│   运行时: 每个子 agent = LLM AGENT + 严格 harness 护栏          │
│                                                                 │
│   LLM 决策:    试哪个 knob / patch, 假设, 风险                  │
│   Harness 强制: 只写 attempts/, 跑完整 bench suite              │
│                  (target + neighbors + controls), A/B,          │
│                  按确定性规则 keep/revert.                      │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么 Setter 有 rule-based 兜底是有意的

1. **可复现性。** 同样的 `raw_results.jsonl` → 每次都给同样的 score。
   LLM-only scorer 不同天给不同 score，永远没法判断 score 变化是因为
   数据还是因为模型。
2. **速度和成本。** 给 10 个 workload 算 score 在 Python 里是毫秒级；
   每个都调 LLM 是秒级 + 烧 token。
3. **可审计。** Stage A 的每个 score 都能追溯到某个 log 文件的具体
   字节——§6 里 P001 的 evidence trail 有 7 个 key signal，每个都引用
   了一个 JSON 文件。LLM 的内部推理很难审计。
4. **机械层 LLM 加不了分。** 检测 `queue=36 ∧ max_running=32 →
   concurrency_capped` 就是一次正则匹配 + 一次比较。LLM 不会做得更
   好，只会更慢 + 非确定。

### LLM 到目前为止做了什么

LLM（我，在这个对话里）做的是**设计工作**：
- 起草两阶段架构
- 设计 skills 体系 + 写每个 `SKILL.md`
- 写所有 Python 实现
- 写 rule-based 参考策略的 triage 规则

当你跑 `python stages/problem-setter/policies/rule_based_explore.py`，
**运行时没有 LLM**。当你通过 Claude Code 或 Copilot CLI 跑 Mode B 时，
LLM 是**策略层**决定下一步做什么；它仍然调用同样的 harness 工具来做
实际工作。

### v0.4 里说的 "Copilot CLI agent" 到底是啥意思

v0.4 roadmap 里"Copilot agent 包装"主要是打磨 Mode B 的 system prompt
(`policies/llm_agent.md`) 并把 setter 暴露为 `@problem-setter`：

```bash
# 今天 (Mode A, 确定性)
$ python stages/problem-setter/policies/rule_based_explore.py \
    --config configs/base.yaml

# 今天 (Mode B, 手动加载 prompt)
$ claude code   # 然后加载 stages/problem-setter/policies/llm_agent.md 作为 system prompt

# v0.4 (规划, 语法糖)
$ copilot -p "@problem-setter 在 configs/moe_qwen3_30b.yaml 上跑一次 session,
              然后给我总结前 3 个 problem"
```

### LLM 啥时候开始做真正的推理？

**阶段 B 起。** 这是设计本身。具体来说：

- **Solver LLM 输入**：一个题目包——`workload.yaml`、
  `baseline_metrics.json`、`server_features.json`、`classification.json`，
  可用的话 `profile_summary.json`、`hypothesis.md`、`problem.json` 里的
  `suggested_strategies`，再加 neighbors 和 controls 用于验证。
- **Solver LLM 每个 attempt 的输出**：一个 knob（或 kernel patch）+
  plan.md + decision.json。harness 然后跑 benchmark 套件决定 keep/revert。

所以：**规则做机械, LLM 做语义, 中间靠契约**。这种切分就是两阶段
设计的全部意义。

### Setter 将来会变成 LLM-only 吗？

两种模式已经共存。我们**不会**做的事：

- 用 LLM 调用替换 rule-based score 函数（牺牲可复现性）。
- session 中让 LLM 直接写 `seed_suite.yaml` / `search_space.yaml`
  / `rule_based_explore.py`（扩展走 `stages/problem-setter/proposals/...`
  给人 review）。
