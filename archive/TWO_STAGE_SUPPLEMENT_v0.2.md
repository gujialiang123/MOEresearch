# TWO_STAGE_AGENT_SUPPLEMENT.md

# SGLang 两阶段自动优化 Agent 补充设计文档

版本：v0.2  
适用项目：`sglang-agent-lab`  
目标硬件：H200 / 单机单卡优先，后续可扩展到多卡  
目标读者：GitHub Copilot CLI / coding agent / 未来维护者  
本文定位：在已有 `AGENTS.md`、benchmark harness、Copilot loop 设计基础上，补充两阶段 Agent 架构。

---

## 0. 总目标

本项目不再把 “SGLang 自动优化” 设计成一个单一 Agent，而是拆成两个阶段：

```text
Stage 1: RegimeScout
  自动寻找 serving performance regimes
  自动构造 benchmark suite
  自动发现可疑性能边界和 minimal reproductions

Stage 2: BottleneckFixer
  针对 Stage 1 输出的 suspicious cases
  做 profiling、诊断、配置搜索、调度/内存策略优化
  必要时再进入 source-level 或 kernel-level 修改
```

核心思想：

```text
先发现值得优化的输入 regime，再优化这些 regime。
不要在少数手写 benchmark 上盲目调参。
不要让优化 Agent 自己随便选择 workload。
```

---

## 1. 为什么要两阶段

之前的设计里，Agent 直接读取 workload，然后尝试调 SGLang 配置。这适合做 MVP，但有一个问题：

```text
如果 workload 本身覆盖不足，Agent 只会优化你手写的几个场景。
它可能错过真正暴露系统弱点的输入分布。
```

因此需要一个专门阶段负责“造数据”和“找 regime”。

类比：

```text
ACM 造数据:
  覆盖算法 corner cases，让错误算法 WA/TLE

RegimeScout:
  覆盖 serving performance corner cases，让系统性能弱点暴露
```

Stage 1 的任务不是修复性能，而是发现：

```text
1. 哪些输入形态触发 TTFT p95 急剧上升
2. 哪些输入形态触发 TPOT / ITL 退化
3. 哪些输入形态触发 throughput plateau
4. 哪些输入形态触发 OOM / KV cache pressure / request retraction
5. 哪些输入形态让 radix cache / prefix cache 特别敏感
6. 哪些输入形态让 scheduler overhead / head-of-line blocking 特别明显
7. 哪些输入形态在 bench_serving 和 offline benchmark 之间出现明显 gap
```

Stage 2 再接收这些已复现的 suspicious cases，并尝试修复。

---

## 2. 基本术语

### 2.1 Workload spec

一个具体 benchmark 输入配置，例如：

```text
input_len=8192
output_len=32
max_concurrency=16
dataset=random
cache_mode=cold
```

它只是输入定义，不等于 regime。

### 2.2 Regime

Regime 是一组表现出相似性能机制的 workload。

例子：

```text
prefill-heavy regime:
  长 input，短 output，TTFT 主导

decode-heavy regime:
  短 input，长 output，TPOT / output throughput 主导

prefix-reuse regime:
  大量共享 prefix，radix cache / prefix cache 主导

KV-pressure regime:
  长上下文 + 高并发，接近 KV cache capacity

scheduler-tail regime:
  混合长短请求导致 p95/p99 latency 爆炸
```

### 2.3 Suspicious case

一个被 Stage 1 判定为值得 Stage 2 进一步诊断的 workload。

例子：

```json
{
  "case_id": "S007",
  "regime_id": "R_prefill_boundary",
  "workload_file": "workloads/discovered/S007.yaml",
  "symptom": "TTFT p95 jumps 3.7x when input_len crosses 8192 at concurrency=16",
  "primary_metric": "ttft_p95_ms",
  "suspicion_score": 0.86,
  "suspected_categories": ["prefill", "chunked_prefill", "memory_pressure"],
  "recommended_stage2_action": "profile_prefill"
}
```

### 2.4 Minimal repro

能稳定触发同一性能问题的最小 workload。

Stage 1 发现一个大 workload 很慢之后，必须尝试 shrink：

```text
降低 num_prompts
降低 concurrency
降低 input_len
降低 output_len
减少 prefix groups
减少 prompt variety
```

如果问题仍然存在，就用更小的 workload 作为 Stage 2 输入。

---

## 3. 新增目录结构

在原有项目结构上新增以下目录和文件。

```text
sglang-agent-lab/

  regime_scout/
    search_space.yaml
    seed_suite.yaml
    diagnostic_toggles.yaml
    README.md

    candidates/
      candidate_*.yaml

    discovered/
      R001.yaml
      R002.yaml

    outputs/
      envelope.json
      raw_results.jsonl
      raw_results.parquet
      regime_map.md
      regime_map.json
      suspicious_cases.jsonl
      selected_cases.jsonl
      shrink_log.jsonl

  prompts/
    discover_regimes_once.md
    evaluate_regimes_once.md
    shrink_case_once.md
    fix_bottleneck_once.md
    two_stage_iteration.md

  .github/
    agents/
      regime-scout.agent.md
      regime-evaluator.agent.md
      regime-shrinker.agent.md
      bottleneck-fixer.agent.md
      two-stage-coordinator.agent.md

  scripts/
    inspect_envelope.py
    generate_seed_suite.py
    generate_candidate_workloads.py
    run_regime_suite.py
    score_suspicion.py
    cluster_regimes.py
    shrink_repro.py
    select_cases_for_stage2.py
    run_stage1.py
    run_stage2.py
    run_two_stage_iteration.py

  experiments/
    regimes/
      stage1_summary.md
      stage2_summary.md
      cases/
        S001/
          case.json
          workload.yaml
          metrics.json
          shrink_result.yaml
          stage2_plan.md
          stage2_result.md
```

---

## 4. 两阶段总流程

### 4.1 Stage 1: RegimeScout

输入：

```text
configs/best.yaml
configs/base.yaml
workloads/current.yaml 可选
regime_scout/search_space.yaml
regime_scout/seed_suite.yaml
benchmark budget
H200 + fixed model
```

输出：

```text
regime_scout/outputs/envelope.json
regime_scout/outputs/raw_results.jsonl
regime_scout/outputs/regime_map.md
regime_scout/outputs/suspicious_cases.jsonl
regime_scout/outputs/selected_cases.jsonl
workloads/discovered/*.yaml
experiments/regimes/cases/S*/case.json
```

流程：

```text
1. Inspect envelope
2. Generate seed suite
3. Run quick benchmark on seeds
4. Run boundary search around suspicious axes
5. Run diagnostic toggles
6. Score suspiciousness
7. Cluster / deduplicate regimes
8. Shrink top suspicious cases
9. Select cases for Stage 2
```

### 4.2 Stage 2: BottleneckFixer

输入：

```text
experiments/regimes/cases/S*/case.json
workloads/discovered/*.yaml
configs/best.yaml
configs/search_space.yaml
benchmark harness
profile harness
```

输出：

```text
experiments/regimes/cases/S*/stage2_plan.md
experiments/regimes/cases/S*/stage2_result.md
experiments/summary.md
experiments/LESSONS.md
configs/best.yaml if accepted
```

流程：

```text
1. Pick one selected suspicious case
2. Read case contract
3. Run baseline benchmark for case
4. Run profile if needed
5. Choose one fix attempt
6. Change exactly one config knob or one allowed code region
7. Run quick / medium / A-B benchmark
8. Decide keep / revert
9. Log result
```

---

## 5. 重要规则

### 5.1 阶段隔离规则

Stage 1 只允许：

```text
生成 workload
运行 benchmark
分析 metrics
打分
聚类
shrink
写 regime map
```

Stage 1 不允许：

```text
修改 configs/best.yaml
修改 SGLang 源码
调参寻找最优配置
宣称修复了性能问题
```

Stage 2 只允许：

```text
读取 Stage 1 输出的 selected cases
针对一个 case 做修复
调一个配置 knob
必要时 profile
必要时修改允许范围内的代码
记录效果
```

Stage 2 不允许：

```text
自己随意更换 workload
为了让结果好看而修改 case
绕过 Stage 1 的 case contract
```

### 5.2 Source of truth

性能结论只来自：

```text
scripts/run_experiment.py
scripts/ab_benchmark.py
scripts/run_regime_suite.py
scripts/profile_server.py
```

不允许 Agent 只凭直觉判断性能。

### 5.3 Workload integrity

一旦 suspicious case 被选中进入 Stage 2，它的 workload 必须冻结：

```text
case.json
workload.yaml
baseline metrics
minimal repro
```

Stage 2 如果需要修改 workload，必须新建 case，不得覆盖原 case。

---

## 6. `AGENTS.md` 需要新增的内容

把下面内容追加到原 `AGENTS.md`。

```md
# Two-Stage SGLang Optimization Protocol

This project has two separate stages.

## Stage 1: RegimeScout

Goal:
Discover performance regimes and suspicious workloads for the fixed model and fixed H200 environment.

Stage 1 may:
- generate workload YAML files
- run benchmark suites
- compute metrics
- score suspicious cases
- shrink reproductions
- write regime_scout/outputs/*
- write experiments/regimes/cases/*

Stage 1 must not:
- modify configs/best.yaml
- modify configs/candidate.yaml except temporary diagnostic runs
- modify SGLang source
- claim a performance fix
- tune for best performance

Stage 1 output contract:
- regime_scout/outputs/regime_map.md
- regime_scout/outputs/suspicious_cases.jsonl
- regime_scout/outputs/selected_cases.jsonl
- experiments/regimes/cases/S*/case.json
- experiments/regimes/cases/S*/workload.yaml

## Stage 2: BottleneckFixer

Goal:
Take one selected suspicious case from Stage 1 and attempt to improve it.

Stage 2 may:
- read one case.json
- run profile
- modify exactly one config knob
- run quick/medium/full/A-B benchmark
- update configs/best.yaml only if accepted
- update experiments/summary.md
- update experiments/LESSONS.md

Stage 2 must not:
- change the selected workload
- change model
- change benchmark seed
- change output length
- change num_prompts
- accept improvement without benchmark evidence
- optimize a workload that was not selected by Stage 1

## Stage boundary

Never mix Stage 1 and Stage 2 in one invocation unless using the explicit two-stage coordinator.

A Stage 1 invocation ends after selected cases are written.
A Stage 2 invocation handles exactly one selected case and stops.
```

---

## 7. RegimeScout search space

创建文件：

```text
regime_scout/search_space.yaml
```

内容：

```yaml
version: 1

model:
  fixed: true
  source: "configs/base.yaml:model-path"

hardware:
  target: "H200"
  fixed_gpu_required: true

budget:
  max_total_benchmark_runs: 80
  max_stage1_wall_seconds: 21600
  max_cases_selected_for_stage2: 5

axes:

  input_len:
    values: [1, 16, 128, 512, 2048, 4096, 8192, 16384, 32768]
    notes: "Covers tiny overhead, normal prompts, RAG, long context, and extreme prefill."

  output_len:
    values: [1, 16, 64, 256, 1024, 2048]
    notes: "Covers TTFT-dominated and decode-heavy regimes."

  max_concurrency:
    values: [1, 2, 4, 8, 16, 32, 64, 128]
    notes: "Used for concurrency knee search."

  request_rate:
    values: [null]
    future_values: ["0.25x_capacity", "0.5x_capacity", "0.8x_capacity", "1.0x_capacity", "1.2x_capacity"]
    notes: "v0.1 can use closed-loop max_concurrency only."

  cache_mode:
    values: ["cold", "warm", "controlled_warmup"]

  dataset_type:
    values: ["random", "generated-shared-prefix", "custom-jsonl"]

  prefix:
    shared_prefix_len: [0, 512, 2048, 4096, 8192]
    groups: [1, 8, 32, 128]
    prompts_per_group: [2, 8, 64]

  mixture:
    enabled: true
    templates:
      - name: "mixed_prefill_80_20"
        components:
          - weight: 0.8
            input_len: 128
            output_len: 64
          - weight: 0.2
            input_len: 8192
            output_len: 64
      - name: "mixed_decode_80_20"
        components:
          - weight: 0.8
            input_len: 128
            output_len: 32
          - weight: 0.2
            input_len: 128
            output_len: 1024

diagnostics:
  enable_toggles: true
  toggles:
    - knob: "disable-radix-cache"
      values: [false, true]
      applies_to: ["prefix_reuse", "prefix_churn"]

    - knob: "disable-cuda-graph"
      values: [false, true]
      applies_to: ["tiny_latency", "short_in_short_out", "decode_heavy"]

    - knob: "schedule-policy"
      values: ["fcfs", "lpm"]
      applies_to: ["prefix_reuse", "mixed_prefill", "scheduler_tail"]

    - knob: "chunked-prefill-size"
      values: [-1, 4096]
      applies_to: ["long_prefill", "prefill_boundary"]

scoring:
  suspicion_weights:
    local_nonlinearity: 0.25
    tail_latency_ratio: 0.20
    diagnostic_sensitivity: 0.20
    stack_gap: 0.15
    failure_nearness: 0.10
    variance: 0.10

  thresholds:
    selected_case_score: 0.65
    high_tail_ratio: 3.0
    large_metric_jump: 2.0
    diagnostic_gap_pct: 20.0
    min_repro_repeats: 3
```

---

## 8. Seed suite

创建文件：

```text
regime_scout/seed_suite.yaml
```

内容：

```yaml
version: 1

seeds:

  - name: "smoke"
    regime_hint: "sanity"
    dataset:
      name: "random"
      random_input_len: 128
      random_output_len: 32
      random_range_ratio: 0.0
    traffic:
      max_concurrency: 4
      num_prompts: 32
    cache:
      mode: "cold"

  - name: "tiny_latency"
    regime_hint: "scheduler_overhead"
    dataset:
      name: "random"
      random_input_len: 1
      random_output_len: 1
      random_range_ratio: 0.0
    traffic:
      max_concurrency: 1
      num_prompts: 32
    cache:
      mode: "cold"

  - name: "short_in_short_out"
    regime_hint: "scheduler_or_cuda_graph"
    dataset:
      name: "random"
      random_input_len: 128
      random_output_len: 32
      random_range_ratio: 0.0
    traffic:
      max_concurrency: 16
      num_prompts: 160
    cache:
      mode: "cold"

  - name: "scheduler_overhead_high_concurrency"
    regime_hint: "scheduler_tail"
    dataset:
      name: "random"
      random_input_len: 128
      random_output_len: 16
      random_range_ratio: 0.0
    traffic:
      max_concurrency: 64
      num_prompts: 320
    cache:
      mode: "cold"

  - name: "prefill_medium"
    regime_hint: "prefill"
    dataset:
      name: "random"
      random_input_len: 4096
      random_output_len: 16
      random_range_ratio: 0.0
    traffic:
      max_concurrency: 4
      num_prompts: 64
    cache:
      mode: "cold"

  - name: "prefill_long"
    regime_hint: "prefill_boundary"
    dataset:
      name: "random"
      random_input_len: 16384
      random_output_len: 16
      random_range_ratio: 0.0
    traffic:
      max_concurrency: 2
      num_prompts: 32
    cache:
      mode: "cold"

  - name: "decode_medium"
    regime_hint: "decode"
    dataset:
      name: "random"
      random_input_len: 128
      random_output_len: 512
      random_range_ratio: 0.0
    traffic:
      max_concurrency: 16
      num_prompts: 160
    cache:
      mode: "cold"

  - name: "decode_heavy"
    regime_hint: "decode_saturation"
    dataset:
      name: "random"
      random_input_len: 128
      random_output_len: 1024
      random_range_ratio: 0.0
    traffic:
      max_concurrency: 32
      num_prompts: 160
    cache:
      mode: "cold"

  - name: "prefix_reuse_ideal"
    regime_hint: "prefix_cache"
    dataset:
      name: "generated-shared-prefix"
      gsp_num_groups: 8
      gsp_prompts_per_group: 32
      gsp_system_prompt_len: 4096
      gsp_question_len: 128
      gsp_output_len: 128
    traffic:
      max_concurrency: 16
      num_prompts: 256
    cache:
      mode: "warm"

  - name: "prefix_churn"
    regime_hint: "cache_churn"
    dataset:
      name: "generated-shared-prefix"
      gsp_num_groups: 128
      gsp_prompts_per_group: 4
      gsp_system_prompt_len: 4096
      gsp_question_len: 128
      gsp_output_len: 128
    traffic:
      max_concurrency: 16
      num_prompts: 512
    cache:
      mode: "warm"

  - name: "mixed_prefill_80_20"
    regime_hint: "head_of_line_prefill"
    dataset:
      name: "custom-jsonl"
      generator: "mixed_prefill_80_20"
    traffic:
      max_concurrency: 32
      num_prompts: 320
    cache:
      mode: "cold"

  - name: "mixed_decode_80_20"
    regime_hint: "head_of_line_decode"
    dataset:
      name: "custom-jsonl"
      generator: "mixed_decode_80_20"
    traffic:
      max_concurrency: 32
      num_prompts: 320
    cache:
      mode: "cold"
```

---

## 9. Suspicious case schema

每个 selected case 必须写成：

```text
experiments/regimes/cases/SXXX/case.json
```

Schema：

```json
{
  "case_id": "S001",
  "regime_id": "R_prefill_boundary",
  "created_at": "YYYY-MM-DD HH:MM:SS",
  "model_path": "...",
  "hardware": "H200",
  "sglang_commit": "...",
  "baseline_config": "configs/best.yaml",
  "workload_file": "experiments/regimes/cases/S001/workload.yaml",

  "workload_summary": {
    "dataset_name": "random",
    "input_len_p50": 8192,
    "input_len_p95": 8192,
    "output_len_p50": 32,
    "output_len_p95": 32,
    "max_concurrency": 16,
    "num_prompts": 160,
    "cache_mode": "cold",
    "prefix_reuse": false
  },

  "symptom": {
    "type": "latency_jump",
    "metric": "ttft_p95_ms",
    "direction": "lower",
    "observed_value": 842.1,
    "neighbor_value": 231.4,
    "multiplier": 3.64,
    "description": "TTFT p95 jumps when input_len crosses 8192 under concurrency 16."
  },

  "evidence": {
    "repeats": 3,
    "coefficient_of_variation_pct": 7.2,
    "success_rate": 1.0,
    "oom": false,
    "timeout": false,
    "server_crash": false,
    "related_runs": [
      "regime_scout/outputs/raw_results.jsonl:line_123"
    ]
  },

  "diagnostics": {
    "sensitive_knobs": [
      {
        "knob": "chunked-prefill-size",
        "best_value_seen": 4096,
        "worst_value_seen": -1,
        "gap_pct": 35.2
      }
    ],
    "suspected_categories": [
      "prefill",
      "chunked_prefill",
      "memory_pressure"
    ]
  },

  "suspicion_score": 0.86,

  "recommended_stage2": {
    "action": "profile_prefill_then_try_one_config",
    "primary_metric": "ttft_p95_ms",
    "primary_direction": "lower",
    "minimum_improvement_pct": 10.0,
    "suggested_first_knobs": [
      "chunked-prefill-size",
      "max-prefill-tokens",
      "schedule-conservativeness"
    ]
  },

  "frozen": true
}
```

Rules:

```text
1. Stage 2 must treat this file as read-only.
2. Stage 2 must use workload_file exactly.
3. If Stage 2 wants a different workload, it must create a new case.
```

---

## 10. Workload YAML schema for discovered cases

Every discovered workload must use this normalized schema.

```yaml
version: 1
name: "S001_prefill_boundary_min_repro"
description: "Minimal repro for TTFT p95 jump around long prefill concurrency boundary."

source:
  generated_by: "RegimeScout"
  parent_seed: "prefill_long"
  case_id: "S001"

model: "/path/to/model"
host: "127.0.0.1"
port: 30000

primary_metric: "ttft_p95_ms"
primary_direction: "lower"

dataset:
  name: "random"
  random_input_len: 8192
  random_output_len: 32
  random_range_ratio: 0.0

traffic:
  max_concurrency: 16
  request_rate: null
  num_prompts:
    quick: 32
    medium: 160
    full: 480

cache:
  mode: "cold"
  flush_cache: true
  controlled_warmup: false

benchmark:
  timeout_seconds:
    server_start: 300
    quick: 600
    medium: 1800
    full: 3600
  repeat:
    quick: 1
    medium: 1
    full: 3

regime_tags:
  - "prefill"
  - "long_context"
  - "concurrency_knee"
  - "ttft_tail"
```

---

## 11. Stage 1 scoring function

Implement `scripts/score_suspicion.py`.

Each benchmark run gets a suspicion score:

```text
score =
  0.25 * local_nonlinearity
+ 0.20 * tail_latency_ratio_score
+ 0.20 * diagnostic_sensitivity
+ 0.15 * stack_gap
+ 0.10 * failure_nearness
+ 0.10 * variance_score
```

### 11.1 local_nonlinearity

Detect nonlinear metric jumps against neighboring workloads.

Examples:

```text
input_len 4096 → 8192:
  expected TTFT maybe ~2x
  observed TTFT p95 5x
  local_nonlinearity high

concurrency 16 → 32:
  throughput +2%
  TTFT p95 +200%
  local_nonlinearity high
```

Implementation hint:

```python
ratio = observed_metric / neighbor_metric
if direction == "lower":
    local_nonlinearity = clamp((ratio - 1.0) / 3.0, 0, 1)
```

### 11.2 tail_latency_ratio_score

Use:

```text
ttft_p95_ms / ttft_p50_ms
tpot_p95_ms / tpot_p50_ms
itl_p95_ms / itl_p50_ms
```

High p95/p50 implies tail regime.

### 11.3 diagnostic_sensitivity

If a diagnostic toggle changes the primary metric by a lot, the workload is mechanism-sensitive.

Examples:

```text
disable-radix-cache false vs true
disable-cuda-graph false vs true
schedule-policy fcfs vs lpm
chunked-prefill-size -1 vs 4096
```

### 11.4 stack_gap

Compare serving-level benchmark to lower-level benchmark when available.

Examples:

```text
bench_serving slow
bench_offline_throughput normal
=> likely scheduler / HTTP / online batching gap
```

For v0.1, if lower-level benchmarks are not implemented, set `stack_gap=0`.

### 11.5 failure_nearness

High if logs contain:

```text
OOM
KV cache pool is full
Retract requests
timeout
failed requests
success rate < 1.0
```

### 11.6 variance_score

Repeat suspicious workloads three times. High coefficient of variation means unstable regime.

---

## 12. Stage 1 scripts

### 12.1 `scripts/inspect_envelope.py`

Purpose:

Read fixed model / fixed H200 / SGLang startup logs and infer safe search envelope.

CLI:

```bash
python scripts/inspect_envelope.py \
  --config configs/base.yaml \
  --out regime_scout/outputs/envelope.json
```

Responsibilities:

```text
1. Launch server with base config.
2. Capture startup log.
3. Extract useful values if present:
   - GPU name
   - available GPU memory
   - model path
   - max context length
   - max_total_num_tokens
   - max_running_requests
   - chunked_prefill_size
   - max_prefill_tokens
4. Kill server.
5. Write envelope.json.
```

If parsing fails, write partial envelope and continue.

### 12.2 `scripts/generate_seed_suite.py`

Purpose:

Convert `regime_scout/seed_suite.yaml` into concrete workload YAML files.

CLI:

```bash
python scripts/generate_seed_suite.py \
  --seed regime_scout/seed_suite.yaml \
  --out-dir regime_scout/candidates
```

Output:

```text
regime_scout/candidates/seed_000_smoke.yaml
regime_scout/candidates/seed_001_tiny_latency.yaml
...
```

### 12.3 `scripts/generate_candidate_workloads.py`

Purpose:

Generate additional candidates from search space and existing results.

CLI:

```bash
python scripts/generate_candidate_workloads.py \
  --search-space regime_scout/search_space.yaml \
  --raw-results regime_scout/outputs/raw_results.jsonl \
  --out-dir regime_scout/candidates \
  --mode boundary
```

Modes:

```text
canonical
boundary
prefix
mixed
near_capacity
```

Rules:

```text
1. Never generate workloads beyond known envelope unless stress mode is explicitly enabled.
2. Prefer logarithmic coverage over dense grid.
3. Prefer boundary refinement around suspicious areas.
4. Do not generate more workloads than budget allows.
```

### 12.4 `scripts/run_regime_suite.py`

Purpose:

Run a list of workload YAML files under baseline config and collect metrics.

CLI:

```bash
python scripts/run_regime_suite.py \
  --config configs/best.yaml \
  --workload-dir regime_scout/candidates \
  --out regime_scout/outputs/raw_results.jsonl \
  --mode quick
```

Behavior:

```text
1. Iterate candidate workloads.
2. For each workload:
   - call scripts/run_experiment.py
   - collect latest metrics
   - append normalized record to raw_results.jsonl
3. Continue after failures.
4. Never overwrite previous raw results unless --reset is passed.
```

Each JSONL row:

```json
{
  "run_id": "run_000123",
  "workload_file": "regime_scout/candidates/seed_004_prefill_medium.yaml",
  "config_file": "configs/best.yaml",
  "mode": "quick",
  "metrics": {},
  "status": "pass",
  "error": null,
  "timestamp": "..."
}
```

### 12.5 `scripts/score_suspicion.py`

Purpose:

Read raw results and compute suspiciousness.

CLI:

```bash
python scripts/score_suspicion.py \
  --raw regime_scout/outputs/raw_results.jsonl \
  --search-space regime_scout/search_space.yaml \
  --out regime_scout/outputs/suspicious_cases.jsonl
```

Output:

```text
regime_scout/outputs/suspicious_cases.jsonl
```

### 12.6 `scripts/cluster_regimes.py`

Purpose:

Deduplicate suspicious cases and group them into regimes.

CLI:

```bash
python scripts/cluster_regimes.py \
  --raw regime_scout/outputs/raw_results.jsonl \
  --suspicious regime_scout/outputs/suspicious_cases.jsonl \
  --out-md regime_scout/outputs/regime_map.md \
  --out-json regime_scout/outputs/regime_map.json
```

v0.1 can be rule-based.

Regime categories:

```text
sanity
scheduler_overhead
prefill
prefill_boundary
decode
decode_saturation
prefix_cache
cache_churn
head_of_line_prefill
head_of_line_decode
kv_pressure
serving_stack_gap
unknown
```

### 12.7 `scripts/shrink_repro.py`

Purpose:

Shrink a suspicious case into a smaller minimal reproduction.

CLI:

```bash
python scripts/shrink_repro.py \
  --case regime_scout/outputs/suspicious_cases.jsonl:S001 \
  --config configs/best.yaml \
  --out-dir experiments/regimes/cases/S001
```

Shrink order:

```text
1. reduce num_prompts
2. reduce max_concurrency
3. reduce input_len
4. reduce output_len
5. reduce prefix group count
6. reduce prompts per group
```

Stop shrinking when symptom disappears.

Symptom is preserved if:

```text
primary metric remains worse than neighbor by threshold
or suspicion_score remains above threshold
or diagnostic sensitivity remains high
```

### 12.8 `scripts/select_cases_for_stage2.py`

Purpose:

Select top cases for Stage 2.

CLI:

```bash
python scripts/select_cases_for_stage2.py \
  --suspicious regime_scout/outputs/suspicious_cases.jsonl \
  --regime-map regime_scout/outputs/regime_map.json \
  --out regime_scout/outputs/selected_cases.jsonl \
  --max-cases 5
```

Selection rules:

```text
1. Prefer high suspicion score.
2. Prefer diversity across regimes.
3. Prefer cases with stable reproduction.
4. Prefer cases with minimal repro.
5. Avoid selecting multiple near-duplicates.
```

### 12.9 `scripts/run_stage1.py`

Purpose:

Run Stage 1 end-to-end.

CLI:

```bash
python scripts/run_stage1.py \
  --config configs/best.yaml \
  --budget 80 \
  --out-dir regime_scout/outputs
```

Pipeline:

```text
1. inspect_envelope
2. generate_seed_suite
3. run_regime_suite on seeds
4. score_suspicion
5. generate boundary candidates
6. run_regime_suite on boundary candidates
7. run diagnostic toggles for likely cases
8. score_suspicion again
9. cluster_regimes
10. shrink top cases
11. select cases for stage2
```

---

## 13. Stage 2 scripts

### 13.1 `scripts/run_stage2.py`

Purpose:

Given one case, run BottleneckFixer flow.

CLI:

```bash
python scripts/run_stage2.py \
  --case experiments/regimes/cases/S001/case.json \
  --config configs/best.yaml \
  --candidate configs/candidate.yaml
```

Behavior:

```text
1. Load case.json.
2. Copy configs/best.yaml to configs/candidate.yaml.
3. Run baseline benchmark on frozen workload.
4. If case recommends profiling, run profile_server.py.
5. Choose one knob from suggested_first_knobs or search_space.
6. Apply exactly one knob change.
7. Run quick benchmark.
8. If quick passes, run medium benchmark.
9. If improvement is <5%, run ab_benchmark.py.
10. If accepted, update configs/best.yaml.
11. Write stage2_result.md.
```

Important:

The Python script may implement mechanical logic, but the actual choice of knob can be delegated to Copilot through `prompts/fix_bottleneck_once.md`.

### 13.2 `scripts/run_two_stage_iteration.py`

Purpose:

Coordinator script for a single two-stage pass.

CLI:

```bash
python scripts/run_two_stage_iteration.py \
  --mode scout_then_fix \
  --max-stage1-runs 80 \
  --max-stage2-cases 1
```

Modes:

```text
stage1_only
stage2_only
scout_then_fix
```

Rules:

```text
stage1_only:
  discover cases and stop

stage2_only:
  consume selected case and stop

scout_then_fix:
  run Stage 1 if no selected cases exist
  then run Stage 2 for exactly one case
  stop
```

---

## 14. Copilot custom agents

### 14.1 `.github/agents/regime-scout.agent.md`

```md
---
name: regime-scout
description: Discovers SGLang serving performance regimes and suspicious benchmark cases.
tools:
  - view
  - edit
  - create
  - glob
  - grep
  - shell
---

You are Stage 1: RegimeScout.

Your job is to discover performance regimes and suspicious workloads.

You may:
- generate workload YAML files
- run scripts/run_stage1.py
- run scripts/run_regime_suite.py
- run scripts/score_suspicion.py
- run scripts/cluster_regimes.py
- run scripts/shrink_repro.py
- write regime_scout/outputs/*
- write experiments/regimes/cases/*

You must not:
- modify configs/best.yaml
- modify SGLang source
- claim a performance fix
- tune for best performance
- change model
- delete experiments
- delete logs

Stop after Stage 1 outputs selected cases.
```

### 14.2 `.github/agents/regime-evaluator.agent.md`

```md
---
name: regime-evaluator
description: Scores, clusters, and selects suspicious SGLang benchmark regimes.
tools:
  - view
  - create
  - shell
---

You evaluate Stage 1 benchmark results.

You may:
- read raw_results.jsonl
- compute suspicion scores
- cluster regimes
- write regime_map.md
- write suspicious_cases.jsonl
- write selected_cases.jsonl

You must not:
- modify configs
- modify workloads except generated case outputs
- run optimization
- modify SGLang source
```

### 14.3 `.github/agents/regime-shrinker.agent.md`

```md
---
name: regime-shrinker
description: Shrinks suspicious SGLang workloads into minimal reproductions.
tools:
  - view
  - create
  - shell
---

You shrink suspicious workloads.

You may:
- run scripts/shrink_repro.py
- create experiments/regimes/cases/S*/workload.yaml
- create experiments/regimes/cases/S*/case.json
- create shrink logs

You must not:
- modify configs/best.yaml
- modify SGLang source
- accept or reject optimization candidates
```

### 14.4 `.github/agents/bottleneck-fixer.agent.md`

```md
---
name: bottleneck-fixer
description: Attempts to improve one selected suspicious SGLang regime using profiling and one-change-at-a-time config optimization.
tools:
  - view
  - edit
  - create
  - glob
  - grep
  - shell
---

You are Stage 2: BottleneckFixer.

You must handle exactly one selected suspicious case.

You may:
- read experiments/regimes/cases/S*/case.json
- run baseline benchmark on frozen workload
- run profile_server.py
- modify exactly one config knob in configs/candidate.yaml
- run quick/medium/full/A-B benchmark
- update configs/best.yaml only if candidate is accepted
- write stage2_result.md
- update experiments/summary.md
- update experiments/LESSONS.md

You must not:
- modify the selected workload
- change model
- change benchmark seed
- change num_prompts
- change output length
- optimize an unselected workload
- modify SGLang source in v0.2 unless explicitly enabled
- change more than one knob
```

### 14.5 `.github/agents/two-stage-coordinator.agent.md`

```md
---
name: two-stage-coordinator
description: Coordinates Stage 1 RegimeScout and Stage 2 BottleneckFixer without mixing their responsibilities.
tools:
  - view
  - create
  - shell
  - task
---

You coordinate the two-stage system.

You may:
- run Stage 1 if no selected cases exist
- select one case for Stage 2
- invoke bottleneck-fixer on that case
- update high-level summaries

You must not:
- directly modify configs
- directly modify SGLang source
- bypass Stage 1 case selection
- run more than one Stage 2 case per invocation

Stop after one coordinated iteration.
```

---

## 15. Prompts

### 15.1 `prompts/discover_regimes_once.md`

```md
Run Stage 1 RegimeScout once.

Follow AGENTS.md and TWO_STAGE_AGENT_SUPPLEMENT.md.

Goal:
Discover SGLang serving performance regimes and selected suspicious cases for Stage 2.

Steps:
1. Inspect:
   - configs/best.yaml
   - configs/base.yaml
   - regime_scout/search_space.yaml
   - regime_scout/seed_suite.yaml
   - experiments/regimes/stage1_summary.md if present

2. Implement missing Stage 1 scripts if necessary:
   - scripts/inspect_envelope.py
   - scripts/generate_seed_suite.py
   - scripts/run_regime_suite.py
   - scripts/score_suspicion.py
   - scripts/cluster_regimes.py
   - scripts/shrink_repro.py
   - scripts/select_cases_for_stage2.py
   - scripts/run_stage1.py

3. Run:
   python scripts/run_stage1.py --config configs/best.yaml --budget 80 --out-dir regime_scout/outputs

4. Verify outputs exist:
   - regime_scout/outputs/raw_results.jsonl
   - regime_scout/outputs/regime_map.md
   - regime_scout/outputs/suspicious_cases.jsonl
   - regime_scout/outputs/selected_cases.jsonl

5. Write or update:
   - experiments/regimes/stage1_summary.md

Do not modify configs/best.yaml.
Do not modify SGLang source.
Do not run Stage 2.
Stop after Stage 1.
```

### 15.2 `prompts/evaluate_regimes_once.md`

```md
Evaluate Stage 1 results.

Read:
- regime_scout/outputs/raw_results.jsonl
- regime_scout/outputs/suspicious_cases.jsonl
- regime_scout/outputs/regime_map.json if present

Run or implement:
- scripts/score_suspicion.py
- scripts/cluster_regimes.py
- scripts/select_cases_for_stage2.py

Write:
- regime_scout/outputs/regime_map.md
- regime_scout/outputs/regime_map.json
- regime_scout/outputs/selected_cases.jsonl
- experiments/regimes/stage1_summary.md

Do not modify configs.
Do not run Stage 2.
```

### 15.3 `prompts/shrink_case_once.md`

```md
Shrink one suspicious case into a minimal reproduction.

Input:
- Use the highest-scoring unshrunk case in regime_scout/outputs/suspicious_cases.jsonl.

Steps:
1. Run scripts/shrink_repro.py for that case.
2. Create experiments/regimes/cases/SXXX/.
3. Write workload.yaml, case.json, metrics.json, shrink_log.jsonl.
4. Ensure case.json has frozen=true.
5. Update regime_scout/outputs/selected_cases.jsonl if the case is stable.

Do not modify configs/best.yaml.
Do not run Stage 2.
Stop after one case.
```

### 15.4 `prompts/fix_bottleneck_once.md`

```md
Run Stage 2 BottleneckFixer once.

Follow AGENTS.md and TWO_STAGE_AGENT_SUPPLEMENT.md.

Goal:
Take exactly one selected suspicious case and attempt one measured improvement.

Steps:
1. Read:
   - regime_scout/outputs/selected_cases.jsonl
   - the first unprocessed experiments/regimes/cases/S*/case.json
   - configs/best.yaml
   - configs/candidate.yaml
   - configs/search_space.yaml
   - experiments/summary.md
   - experiments/LESSONS.md

2. Treat the selected case workload as frozen.
   Do not modify it.

3. Write:
   experiments/regimes/cases/SXXX/stage2_plan.md

4. Copy configs/best.yaml to configs/candidate.yaml.

5. Choose exactly one config knob to change.
   Prefer case.json recommended_stage2.suggested_first_knobs.
   If none are available, choose from configs/search_space.yaml.

6. Run baseline if needed:
   python scripts/run_experiment.py --mode medium --config configs/best.yaml --workload experiments/regimes/cases/SXXX/workload.yaml

7. If the case recommends profiling, run:
   python scripts/profile_server.py --config configs/best.yaml --workload experiments/regimes/cases/SXXX/workload.yaml --out experiments/regimes/cases/SXXX/profile.md

8. Apply exactly one knob change to configs/candidate.yaml.

9. Run quick:
   python scripts/run_experiment.py --mode quick --config configs/candidate.yaml --workload experiments/regimes/cases/SXXX/workload.yaml

10. If quick passes, run medium:
   python scripts/run_experiment.py --mode medium --config configs/candidate.yaml --workload experiments/regimes/cases/SXXX/workload.yaml

11. If improvement is ambiguous or <5%, run A/B:
   python scripts/ab_benchmark.py --a configs/best.yaml --b configs/candidate.yaml --workload experiments/regimes/cases/SXXX/workload.yaml

12. Decide:
   - keep
   - revert
   - needs_profile
   - needs_research

13. Write:
   experiments/regimes/cases/SXXX/stage2_result.md

14. If keep:
   copy configs/candidate.yaml to configs/best.yaml

15. If revert:
   copy configs/best.yaml to configs/candidate.yaml

16. Update:
   - experiments/summary.md
   - experiments/LESSONS.md if durable lesson exists

Stop after this one case.
```

### 15.5 `prompts/two_stage_iteration.md`

```md
Run one two-stage coordinator iteration.

Follow AGENTS.md and TWO_STAGE_AGENT_SUPPLEMENT.md.

Rules:
- Do not ask the user questions.
- Do not change model.
- Do not delete logs or experiments.
- Do not run more than one Stage 2 case.
- Stop after one coordinated iteration.

Flow:
1. If regime_scout/outputs/selected_cases.jsonl is missing or empty:
   invoke Stage 1 RegimeScout by running prompts/discover_regimes_once.md.

2. If selected cases exist:
   pick the first unprocessed case.
   invoke Stage 2 BottleneckFixer using prompts/fix_bottleneck_once.md.

3. Write:
   experiments/regimes/stage2_summary.md

4. Stop.
```

---

## 16. Copilot loop for two-stage mode

新增脚本：

```text
scripts/run_two_stage_copilot_loop.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-50}"

mkdir -p "$ROOT/logs/copilot-loop"
mkdir -p "$ROOT/experiments/regimes"
mkdir -p "$ROOT/regime_scout/outputs"

echo "Starting two-stage SGLang Copilot loop"
echo "ROOT=$ROOT"
echo "INTERVAL_SECONDS=$INTERVAL_SECONDS"

while true; do
  ts="$(date +%Y%m%d_%H%M%S)"
  echo "=== two-stage iteration $ts ===" | tee -a "$ROOT/logs/copilot-loop/two_stage_loop.log"

  if [[ -f "$ROOT/STOP" ]]; then
    echo "STOP file found. Exiting." | tee -a "$ROOT/logs/copilot-loop/two_stage_loop.log"
    exit 0
  fi

  (
    cd "$ROOT"

    copilot \
      --agent two-stage-coordinator \
      --autopilot \
      --max-autopilot-continues 10 \
      -p "$(cat prompts/two_stage_iteration.md)" \
      > "logs/copilot-loop/two_stage_stdout_${ts}.log" \
      2> "logs/copilot-loop/two_stage_stderr_${ts}.log"
  ) || {
    echo "Two-stage iteration $ts failed. See stderr log." \
      | tee -a "$ROOT/logs/copilot-loop/two_stage_loop.log"
  }

  python scripts/check_loop_state.py \
    --root "$ROOT" \
    --max-experiments "$MAX_EXPERIMENTS" \
    || {
      echo "Loop stop condition reached." | tee -a "$ROOT/logs/copilot-loop/two_stage_loop.log"
      exit 0
    }

  sleep "$INTERVAL_SECONDS"
done
```

If the installed Copilot CLI does not support a flag, check `copilot --help` and use equivalent current options.

---

## 17. Diagnostic toggles

Stage 1 may run diagnostic toggles, but these are not optimization attempts.

Create:

```text
regime_scout/diagnostic_toggles.yaml
```

```yaml
version: 1

toggles:

  - name: "radix_cache_sensitivity"
    applies_to_tags:
      - "prefix_cache"
      - "cache_churn"
    knob: "disable-radix-cache"
    values:
      - false
      - true
    metric: "ttft_p95_ms"
    interpretation:
      large_gap: "cache_sensitive_regime"

  - name: "cuda_graph_sensitivity"
    applies_to_tags:
      - "tiny_latency"
      - "scheduler_overhead"
      - "decode"
    knob: "disable-cuda-graph"
    values:
      - false
      - true
    metric: "tpot_p95_ms"
    interpretation:
      large_gap: "cuda_graph_sensitive_regime"

  - name: "schedule_policy_sensitivity"
    applies_to_tags:
      - "prefix_cache"
      - "mixed_prefill"
      - "head_of_line"
    knob: "schedule-policy"
    values:
      - "fcfs"
      - "lpm"
    metric: "ttft_p95_ms"
    interpretation:
      large_gap: "scheduler_sensitive_regime"

  - name: "chunked_prefill_sensitivity"
    applies_to_tags:
      - "prefill"
      - "prefill_boundary"
      - "long_context"
    knob: "chunked-prefill-size"
    values:
      - -1
      - 4096
    metric: "ttft_p95_ms"
    interpretation:
      large_gap: "chunked_prefill_sensitive_regime"
```

Rules:

```text
1. Diagnostic toggles are used to label regimes.
2. Diagnostic toggles must not update configs/best.yaml.
3. Diagnostic toggle results must be written into case.json diagnostics.
```

---

## 18. Regime map output

`regime_scout/outputs/regime_map.md` must be human-readable.

Template:

```md
# Regime Map

Model:
Hardware:
SGLang commit:
Baseline config:
Generated at:

## Overview

Total workloads run:
Successful:
Failed:
Suspicious cases:
Selected for Stage 2:

## Regime clusters

### R001: prefill_boundary

Representative workloads:
- ...

Symptoms:
- TTFT p95 jumps around input_len=...
- concurrency knee at ...

Likely mechanisms:
- prefill
- chunked prefill
- memory pressure

Selected case:
- S001

### R002: prefix_cache

Representative workloads:
- ...

Symptoms:
- radix cache on/off gap ...
- warm cache much faster than cold cache ...

Likely mechanisms:
- prefix cache
- radix cache
- scheduler policy

Selected case:
- S002

## Top suspicious cases

| Case | Regime | Score | Metric | Symptom | Selected |
|---|---|---:|---|---|---|
| S001 | prefill_boundary | 0.86 | ttft_p95_ms | 3.6x jump | yes |
| S002 | prefix_cache | 0.82 | ttft_p95_ms | radix gap 45% | yes |

## Recommended Stage 2 order

1. S001
2. S002
3. S003
```

---

## 19. Stage 2 result format

`experiments/regimes/cases/SXXX/stage2_result.md`:

```md
# Stage 2 Result: SXXX

Case:
Regime:
Workload:
Model:
Hardware:
Baseline config:
Candidate config:

## Frozen workload

Brief summary of case workload.

## Symptom from Stage 1

Metric:
Observed baseline:
Suspicion score:
Suspected categories:

## Plan

Changed knob:
Old value:
New value:
Hypothesis:
Expected improvement:
Risk:

## Profile summary

If profiling was run, summarize key findings.

## Benchmarks

### Baseline

- ttft_p50_ms:
- ttft_p95_ms:
- tpot_p50_ms:
- tpot_p95_ms:
- output_tokens_per_second:
- request_throughput:
- successful_requests:

### Candidate quick

...

### Candidate medium

...

### A/B if any

...

## Decision

keep / revert / needs_profile / needs_research

## Reasoning

Use only measured metrics.

## Follow-up

If this case is not resolved, suggest one next experiment.
```

---

## 20. Implementation order for Copilot

Copilot should implement this supplement in this order.

### Phase 1: Files and schemas

1. Create `regime_scout/` directories.
2. Create `regime_scout/search_space.yaml`.
3. Create `regime_scout/seed_suite.yaml`.
4. Create `regime_scout/diagnostic_toggles.yaml`.
5. Create prompts.
6. Create custom agent files.
7. Append two-stage protocol to `AGENTS.md`.

### Phase 2: Stage 1 minimal implementation

1. Implement `scripts/generate_seed_suite.py`.
2. Implement `scripts/run_regime_suite.py`.
3. Implement `scripts/score_suspicion.py`.
4. Implement `scripts/cluster_regimes.py`.
5. Implement `scripts/select_cases_for_stage2.py`.
6. Implement `scripts/run_stage1.py`.

For MVP, `cluster_regimes.py` may be rule-based.

### Phase 3: Shrink

1. Implement `scripts/shrink_repro.py`.
2. Support random workloads first.
3. Support generated-shared-prefix second.
4. Support custom-jsonl mixed workloads last.

### Phase 4: Stage 2 integration

1. Implement `scripts/run_stage2.py`.
2. Make it consume one `case.json`.
3. Make it call existing benchmark harness.
4. Make it write `stage2_result.md`.
5. Make it update `configs/best.yaml` only on accepted improvement.

### Phase 5: Coordinator

1. Implement `scripts/run_two_stage_iteration.py`.
2. Implement `scripts/run_two_stage_copilot_loop.sh`.
3. Test `stage1_only`.
4. Test `stage2_only`.
5. Test `scout_then_fix`.

---

## 21. MVP acceptance criteria

Stage 1 MVP is complete when:

```text
1. python scripts/run_stage1.py --config configs/best.yaml --budget 20 works.
2. It generates at least 8 workload YAML files.
3. It runs benchmarks for all generated workloads.
4. It writes raw_results.jsonl.
5. It writes regime_map.md.
6. It writes suspicious_cases.jsonl.
7. It selects at least 1 case for Stage 2 if a suspicious case exists.
8. It does not modify configs/best.yaml.
```

Stage 2 MVP is complete when:

```text
1. python scripts/run_stage2.py --case experiments/regimes/cases/S001/case.json works.
2. It uses the frozen workload.
3. It changes exactly one config knob.
4. It runs quick and medium benchmark.
5. It writes stage2_result.md.
6. It accepts or reverts candidate correctly.
7. It updates configs/best.yaml only on keep.
```

Full two-stage MVP is complete when:

```text
1. prompts/two_stage_iteration.md can run via Copilot.
2. The coordinator either runs Stage 1 or consumes one selected case.
3. No invocation handles more than one Stage 2 case.
4. All outputs are reproducible.
5. Failed cases are logged instead of hidden.
```

---

## 22. H200-specific notes

For H200, do not assume the same regimes as H100 or A100.

Likely useful stress directions:

```text
1. Longer input lengths before OOM.
2. Higher max_concurrency before KV pressure.
3. More prefix groups before cache churn.
4. Decode-heavy regimes may require larger output_len to saturate.
5. Small models may expose scheduler and launch overhead more than GPU compute.
```

Stage 1 should therefore include:

```text
input_len up to 32768 if model supports it
output_len up to 2048 for decode stress
max_concurrency up to 128 in quick search
prefix groups up to 128
near-capacity search with safe stop conditions
```

Safe stop conditions:

```text
1. OOM detected
2. server crash
3. benchmark timeout
4. success_rate < 0.99
5. TTFT p95 > 5x previous neighbor
6. throughput no longer increases by more than 5% while latency doubles
```

---

## 23. Non-goals for v0.2

Do not implement these yet unless explicitly asked:

```text
1. Complex Bayesian optimization.
2. RL-based workload generation.
3. Full Nsight Systems automation.
4. Multi-node distributed SGLang.
5. Kernel code generation.
6. Automatic SGLang source modification.
7. Production traffic replay.
8. Web UI dashboard.
```

Focus on:

```text
1. Reliable workload generation.
2. Reliable benchmark execution.
3. Clear suspicious case scoring.
4. Minimal repro generation.
5. Clean handoff from Stage 1 to Stage 2.
```

---

## 24. First manual commands

After implementation, run:

```bash
python scripts/generate_seed_suite.py \
  --seed regime_scout/seed_suite.yaml \
  --out-dir regime_scout/candidates
```

Then:

```bash
python scripts/run_regime_suite.py \
  --config configs/best.yaml \
  --workload-dir regime_scout/candidates \
  --out regime_scout/outputs/raw_results.jsonl \
  --mode quick
```

Then:

```bash
python scripts/score_suspicion.py \
  --raw regime_scout/outputs/raw_results.jsonl \
  --search-space regime_scout/search_space.yaml \
  --out regime_scout/outputs/suspicious_cases.jsonl
```

Then:

```bash
python scripts/cluster_regimes.py \
  --raw regime_scout/outputs/raw_results.jsonl \
  --suspicious regime_scout/outputs/suspicious_cases.jsonl \
  --out-md regime_scout/outputs/regime_map.md \
  --out-json regime_scout/outputs/regime_map.json
```

Then:

```bash
python scripts/select_cases_for_stage2.py \
  --suspicious regime_scout/outputs/suspicious_cases.jsonl \
  --regime-map regime_scout/outputs/regime_map.json \
  --out regime_scout/outputs/selected_cases.jsonl \
  --max-cases 3
```

Then run Stage 2 on one case:

```bash
python scripts/run_stage2.py \
  --case experiments/regimes/cases/S001/case.json \
  --config configs/best.yaml \
  --candidate configs/candidate.yaml
```

Finally, run Copilot two-stage loop:

```bash
bash scripts/run_two_stage_copilot_loop.sh
```

---

## 25. Final instruction to Copilot

When implementing this supplement:

```text
1. Preserve the existing single-stage optimizer.
2. Add the two-stage system without breaking old commands.
3. Implement Stage 1 first.
4. Do not start Stage 2 until Stage 1 produces selected_cases.jsonl.
5. Keep Stage 1 and Stage 2 responsibilities separate.
6. Keep all artifacts human-readable and machine-readable.
7. Prefer simple rule-based logic over complex ML in v0.2.
8. Never hide failed benchmark runs.
9. Never mutate frozen case workloads.
10. Stop after one coordinator iteration.
```

End of supplement.
