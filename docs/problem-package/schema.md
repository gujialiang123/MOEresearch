# Problem Package — Schema

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#-中文版)
>
> **Audience**: anyone (human or agent) who reads or writes a problem
> package. This is the single source of truth for the data contract
> between the Problem-Setter and the Problem-Solver fleet.
>
> Schema version: **1** (will be bumped if breaking changes; until then,
> only additive fields are allowed.)

---

## Directory layout

A problem package is a self-contained directory. **Everything related to
this issue lives inside it**: target workload, evidence, hypothesis,
acceptance criteria, **and** all solver attempts + the final solution
report. Once a problem is closed, the directory is a complete
reproducible experiment record.

```
experiments/problems/PNNN/
├── problem.json                  # The umbrella contract (this schema)
├── workload.yaml                 # The target workload (the cliff)
├── baseline_metrics.json         # L1
├── server_features.json          # L2
├── classification.json           # L3
├── profile_summary.json          # L4 — agent-decided; null if skipped
├── repeats/                      # L5 — agent-decided; folder may be empty
│   ├── rep_01_metrics.json
│   └── ...
├── neighbors/
│   ├── <neighbor_name>.yaml
│   ├── <neighbor_name>_baseline_metrics.json
│   ├── <neighbor_name>_server_features.json
│   └── ...
├── controls/
│   ├── <control_name>.yaml
│   ├── <control_name>_baseline_metrics.json
│   └── ...
├── hypothesis.md                 # Setter's prose: "why this cliff exists"
├── acceptance_criteria.json      # What the solver must achieve
│
├── attempts/                     # Solver writes here. One subdir per attempt.
│   ├── attempt_001/
│   │   ├── plan.md               # what the solver tried + why
│   │   ├── candidate_config.yaml # OR kernel_patch.diff for kernel-agent
│   │   ├── verification/
│   │   │   ├── target_metrics.json
│   │   │   ├── neighbors/<name>_metrics.json
│   │   │   ├── controls/<name>_metrics.json
│   │   │   └── ab_summary.json   # when applicable
│   │   └── decision.json         # {keep | revert | needs_more_evidence, also_solved[]}
│   └── ...
│
├── solution.md                   # Final report (only if accepted). Bilingual for humans.
└── rejection.md                  # If the problem couldn't be solved or was deemed invalid.
```

Every file path inside `problem.json` is **relative to the problem dir**
so the package is relocatable. A reviewer can `tar czf P001.tar.gz
experiments/problems/P001/` and the recipient has the full experiment
record — input, evidence, every fix attempt, and the final report.

---

## `problem.json` schema

```jsonc
{
  "schema_version": 1,
  "problem_id": "P001",
  "regime_id": "R_scheduler_tail",
  "created_at": "2026-05-28 16:00:00",
  "frozen": true,                             // once true, package is read-only

  // ── identity ──────────────────────────────────────────────
  "model_path": "/data/hf/models/Qwen3-0.6B",
  "hardware": "H200",
  "sglang_version": "0.5.12.post1",
  "sglang_commit": null,                      // best-effort
  "baseline_config": "configs/base.yaml",     // setter's frozen server config

  // ── the target (the cliff) ────────────────────────────────
  "target": {
    "workload":            "workload.yaml",
    "baseline_metrics":    "baseline_metrics.json",
    "server_features":     "server_features.json",
    "classification":      "classification.json",
    "profile_summary":     "profile_summary.json",  // null if L4 skipped
    "skip_L4_reason":      null,                    // free-form note when profile_summary is null; agent-decided
    "repeats":             ["repeats/rep_01_metrics.json", "repeats/rep_02_metrics.json", "repeats/rep_03_metrics.json"]
  },

  // ── the cliff itself ──────────────────────────────────────
  "symptom": {
    "metric":         "ttft_p95_ms",
    "direction":      "lower-is-better",            // or "higher-is-better"
    "observed_value": 434.23,                       // from baseline_metrics
    "magnitude_vs_nearest_neighbor_pct": 257,       // 434/122 ≈ 3.57×
    "description":    "TTFT p95 jumps when max_concurrency crosses 32; baseline 122 ms at mc=32, 434 ms at mc=64."
  },

  // ── benchmark suite the solver will run ───────────────────
  "benchmark_suite": {
    "target": {
      "workload":     "workload.yaml",
      "role":         "must_improve",
      "primary_metric": "ttft_p95_ms"
    },
    "neighbors": [
      {
        "name":            "mc_16",
        "workload":        "neighbors/mc_16.yaml",
        "baseline_metrics":"neighbors/mc_16_baseline_metrics.json",
        "role":            "lower_concurrency_neighbor",
        "expected_after_fix": "no_significant_change"
      },
      {
        "name":            "mc_32",
        "workload":        "neighbors/mc_32.yaml",
        "baseline_metrics":"neighbors/mc_32_baseline_metrics.json",
        "role":            "boundary_neighbor",
        "expected_after_fix": "small_improvement_or_no_change"
      }
    ],
    "controls": [
      {
        "name":            "smoke",
        "workload":        "controls/smoke.yaml",
        "baseline_metrics":"controls/smoke_baseline_metrics.json",
        "role":            "no_regression_check"
      }
    ]
  },

  // ── evidence stack ────────────────────────────────────────
  "evidence": {
    "layers_collected": ["L1", "L2", "L3", "L4", "L5"],   // which layers contributed
    "key_signals": [
      {
        "source":  "server_features.json",
        "field":   "concurrency_capped",
        "value":   true,
        "weight":  "high"
      },
      {
        "source":  "server_features.json",
        "field":   "peak_queue_reqs",
        "value":   36,
        "weight":  "high"
      },
      {
        "source":  "server_features.json",
        "field":   "max_running_requests",
        "value":   32,
        "weight":  "high"
      },
      {
        "source":  "classification.json",
        "field":   "classification",
        "value":   "load_shed_concurrency",
        "weight":  "high"
      },
      {
        "source":  "profile_summary.json",
        "field":   "dominant_bottleneck",
        "value":   "scheduler_queueing",
        "weight":  "high"
      }
    ],
    "bias_check": {
      "repeats_n":              3,
      "primary_metric_cv_pct":  4.2,                // CV across repeats
      "metric_stable":          true                // CV below noise baseline threshold
    },
    "ruled_out": [
      {"hypothesis": "KV cache pressure", "reason": "peak_token_usage = 0.00 across all repeats"},
      {"hypothesis": "CUDA graph fallback", "reason": "peak_running_reqs <= max(cuda_graph_bs_captured)"}
    ]
  },

  // ── setter's hypothesis ───────────────────────────────────
  "hypothesis_md": "hypothesis.md",                  // prose lives in its own file

  // ── routing hints for the solver fleet ────────────────────
  "suggested_strategies": [
    {
      "strategy_id":  "S001",
      "kind":         "config",                     // config | scheduler | kernel | workload-shape
      "route_to":     "config-agent",
      "knob":         "max-running-requests",
      "current_value": 32,
      "values_to_try": [48, 64, 96, 128],
      "expected_improvement_pct": 60,
      "rationale":    "Queue depth 36 > admission cap 32 indicates load shed; raising admission cap should let pending requests stream into decode batches.",
      "risk":         "may regress low-concurrency latency due to bigger CUDA-graph batch sizes; check controls.smoke."
    },
    {
      "strategy_id":  "S002",
      "kind":         "config",
      "route_to":     "config-agent",
      "knob":         "cuda-graph-max-bs",
      "current_value": 256,
      "values_to_try": [64, 128],
      "expected_improvement_pct": 15,
      "rationale":    "Smaller CUDA graph cache may reduce padding cost when bs <= 32 dominates.",
      "risk":         "low; complementary to S001"
    }
  ],
  "rejected_strategies": [
    {
      "kind": "config",
      "knob": "chunked-prefill-size",
      "reason": "random_input_len=128 — chunked prefill not relevant at this scale"
    }
  ],

  // ── acceptance criteria ───────────────────────────────────
  // also persisted as acceptance_criteria.json for solver convenience
  "acceptance_criteria_file": "acceptance_criteria.json",

  // ── lineage ───────────────────────────────────────────────
  "lineage": {
    "stage1_run_dirs": [
      "experiments/tmp/regime_scout/20260528_023213/run_0004_scheduler_overhead_high_concurrency",
      "experiments/tmp/regime_scout/20260528_030217/run_0001_scheduler_overhead_high_concurrency__con_16",
      "experiments/tmp/regime_scout/20260528_030217/run_0002_scheduler_overhead_high_concurrency__con_32"
    ],
    "scoring_record":   "regime_scout/outputs/suspicious_cases.jsonl#run_0004",
    "noise_baseline":   "experiments/noise_baseline.json"  // null if not yet calibrated
  }
}
```

---

## `acceptance_criteria.json` schema

```jsonc
{
  "schema_version": 1,
  "problem_id": "P001",

  // primary: the cliff itself
  "primary": {
    "on": "target",
    "metric": "ttft_p95_ms",
    "direction": "lower",
    "required_improvement_pct": 30.0,         // fix must reduce target's TTFT p95 by ≥30%
    "stretch_improvement_pct":  60.0          // setter's expectation (info only)
  },

  // constraints: things that must not regress
  "constraints": [
    {
      "on":     "target",
      "metric": "request_throughput",
      "direction": "higher",
      "max_regression_pct": 10.0              // throughput may drop ≤10%
    },
    {
      "on":     "controls.smoke",
      "metric": "ttft_p95_ms",
      "direction": "lower",
      "max_regression_pct": 5.0
    },
    {
      "on":     "any",
      "metric": "oom",
      "max":    0
    },
    {
      "on":     "any",
      "metric": "server_crash",
      "max":    0
    },
    {
      "on":     "any",
      "metric": "failed_requests",
      "max":    0
    }
  ],

  // statistical verification
  "verification": {
    "always_paired_ab": false,
    "paired_ab_required_when_delta_lt_pct": 5.0,    // ambiguous deltas need A/B
    "min_repeats_per_workload_for_keep": 3
  },

  // honourable mention: what would be even better
  "stretch_goals": [
    "neighbors.mc_32 also improves by ≥10% (bonus, not required)"
  ]
}
```

---

## Validation rules (enforced by `validate_problem.py`, planned)

A problem package is valid iff **all** of:

1. `problem.json.schema_version == 1`.
2. `frozen == true`. (Means: package is **immutable for reproducibility**.
   Solver may freely **read** it and **write to `attempts/` subdirs**
   only; Solver must never **mutate** files at the package root.)
3. Every file path under `target`, `benchmark_suite.*`, `evidence.key_signals[].source` exists relative to the problem dir.
4. `target.profile_summary` is either a path to an existing file OR
   `null` with a free-form `target.skip_L4_reason` note (e.g.
   `"signal already strong without profiling"`, `"profiler unavailable"`).
   L4 is **agent-decided**, not mandatory.
5. `target.repeats` length is 0 or more — L5 repeat count is
   agent-decided based on observed CV vs noise baseline.
6. `benchmark_suite.neighbors` has length ≥ 1 (a problem without a
   neighbor cannot be A/B'd properly).
7. `benchmark_suite.controls` has length ≥ 1.
8. `acceptance_criteria.json.primary.metric` matches `symptom.metric`.
9. `suggested_strategies` is non-empty (setter must commit to at least
   one route hint).
10. `hypothesis.md` exists and is ≥ 100 characters of prose.

A solver may refuse to work on an invalid package; it writes
`rejection.md` inside the package and stops.

---

## Anti-cheating: how it actually works (no hash needed)

There is **no** explicit hash file or integrity check in the package.
Instead, integrity is enforced by **convention + reproducibility**:

- The package root (everything except `attempts/`) is **read-only by
  convention**. The Problem-Setter's `seal_problem.py` (planned) makes
  the files immutable via `chmod -w` after L4/L5 are collected.
- The solver's harness only ever writes under `attempts/attempt_NNN/`.
  Any solver script that attempts to write to `workload.yaml` or
  `neighbors/*.yaml` will fail at the filesystem layer.
- **The ultimate check**: anyone — another solver, the setter on
  re-review, a human reviewer — can re-run the accepted
  `solution.json`'s configuration against the package's **original
  workload yamls** and verify the primary-metric gain reproduces. If it
  doesn't reproduce, the solution is invalidated.

In other words: if a solver "cheats" by quietly tweaking a workload yaml
during its attempt, the cheat is auto-detectable post-hoc because the
package's original yamls are still there for anyone to re-run. No hash
check needed; the immutability of the package + reproducibility check
suffice.

What the solver **may freely do**:

- Read every byte of the package, including L4 profile traces.
- Read other problem packages and the idea pool.
- Spawn extra exploratory benchmarks **outside the package** for its
  own diagnostic purposes — those don't count toward the verification
  decision; only the suite's workloads do.
- **Propose new workloads** (additive, never modifying existing ones)
  by writing `experiments/ideas/from_solver/idea_*.json` with type
  `proposed_workload`. The Setter will consider these in future runs.
- **Report side-solved problems**: if while solving P001 the solver
  notices and incidentally fixes a related issue (e.g. a control
  workload also benefits from the same knob change), document it in
  `decision.json.also_solved[]` and in the final `solution.md`.

---

## What a solver writes back

The solver does NOT modify the package root files. It writes inside the
problem package's `attempts/` subdirectory (so the package is one
self-contained experiment record):

```
experiments/problems/P001/
├── ... (the frozen package contents above)
├── attempts/
│   ├── attempt_001/
│   │   ├── plan.md
│   │   ├── candidate_config.yaml         (config-agent attempts)
│   │   │   OR kernel_patch.diff          (kernel-agent attempts)
│   │   ├── verification/
│   │   │   ├── target_metrics.json
│   │   │   ├── neighbors/mc_16_metrics.json
│   │   │   ├── neighbors/mc_32_metrics.json
│   │   │   ├── controls/smoke_metrics.json
│   │   │   └── ab_summary.json           (when applicable)
│   │   └── decision.json
│   ├── attempt_002/ ...
│   └── ...
├── solution.md             # final accepted solution report (humans + LLMs read)
└── rejection.md            # if no solution found OR problem was invalid
```

### `decision.json` schema (per attempt)

```jsonc
{
  "schema_version": 1,
  "attempt_id": "attempt_003",
  "solver_agent": "config-agent",
  "decision": "keep",                                    // keep | revert | needs_more_evidence
  "reasoning": "Free-form prose citing verification metrics.",
  "primary_delta_pct": 64.2,                             // target metric improvement
  "constraint_violations": [],                           // empty if none
  "also_solved": [                                       // NEW: side-discovered problems this attempt fixed
    {
      "ref": "neighbors.mc_32",
      "metric": "ttft_p95_ms",
      "delta_pct": 12.0,
      "note": "Neighbor also improved by ~12% — same admission cap fix."
    }
  ],
  "new_findings_filed": [                                // ideas written this attempt
    "experiments/ideas/from_solver/idea_007.json"
  ]
}
```

### `solution.md` — the final report (only for accepted solutions)

A bilingual markdown report. Includes:

1. The problem in one paragraph (what was wrong, why it mattered).
2. The fix (which sub-agent applied it, which knob/patch).
3. Quantified gain on target + neighbors + controls.
4. Side-solved problems (from `also_solved` aggregated across attempts).
5. New ideas filed during the work.
6. Reproducibility: exact command to reproduce the gain.

This file IS the experiment record handed to a human reviewer.

---

## Versioning policy

- New fields may be added without bumping `schema_version`. Solvers
  must tolerate unknown fields.
- Removing or renaming a field is a breaking change → bump
  `schema_version`. Both setter and all solvers must update together.
- The `frozen=true` invariant is non-negotiable; never overwrite a
  shipped problem. If reissuing, use a new `problem_id` (`P001_v2`) and
  link `lineage.supersedes`.

---

## Migration from v0.3 `case.json`

A v0.3 `experiments/regimes/cases/SNNN/case.json` is roughly:

```
problem.json (subset)
  with
    target.profile_summary = null
    target.skip_L4_reason  = "v0.3 had no profiling skill"
    benchmark_suite.neighbors = []
    benchmark_suite.controls  = []
    evidence.layers_collected = ["L1", "L2", "L3"]
```

A migration script `migrate_v03_case_to_problem.py` can lift the v0.3
case + the wave-1 neighbor metrics (already captured) into a valid
problem package, with `frozen=false` while neighbors/controls are
fetched and L4 is added.

---
---

# 🇨🇳 中文版

> **读者**：任何读写题目包的人或 agent。本文是出题人和做题人 fleet
> 之间数据契约的唯一真相来源。
>
> Schema 版本：**1**（破坏性改动时升版本号；之前只允许追加字段）。

---

## 目录结构

一个题目包是自包含的目录。**这个 issue 相关的一切都在里面**：target
workload、证据、假设、验收标准、**以及**所有做题人 attempt + 最终
solution 报告。题目关闭时，目录是一份完整可复现的实验记录。

```
experiments/problems/PNNN/
├── problem.json                  # 总契约（本 schema）
├── workload.yaml                 # target workload（悬崖本身）
├── baseline_metrics.json         # L1
├── server_features.json          # L2
├── classification.json           # L3
├── profile_summary.json          # L4 —— agent 自行决定，跳过时为 null
├── repeats/                      # L5 —— agent 自行决定，可空
│   ├── rep_01_metrics.json
│   └── ...
├── neighbors/
│   ├── <neighbor_name>.yaml
│   ├── <neighbor_name>_baseline_metrics.json
│   ├── <neighbor_name>_server_features.json
│   └── ...
├── controls/
│   ├── <control_name>.yaml
│   ├── <control_name>_baseline_metrics.json
│   └── ...
├── hypothesis.md                 # 出题人的散文："为什么有这个悬崖"
├── acceptance_criteria.json      # 做题人要达到什么算解出来
│
├── attempts/                     # 做题人在这里写。每次 attempt 一个子目录。
│   ├── attempt_001/
│   │   ├── plan.md               # 做题人这次试什么 + 为什么
│   │   ├── candidate_config.yaml # config-agent 用；kernel-agent 用 kernel_patch.diff
│   │   ├── verification/
│   │   │   ├── target_metrics.json
│   │   │   ├── neighbors/<name>_metrics.json
│   │   │   ├── controls/<name>_metrics.json
│   │   │   └── ab_summary.json   # 需要时
│   │   └── decision.json         # {keep | revert | needs_more_evidence, also_solved[]}
│   └── ...
│
├── solution.md                   # 最终 accepted 报告（双语，给人看）
└── rejection.md                  # 题目无解或被判无效时
```

`problem.json` 里所有路径都是**相对题目目录的**。reviewer 可以
`tar czf P001.tar.gz experiments/problems/P001/`，收件人就拿到了完整
实验记录——输入、证据、每次修复 attempt、最终报告。

---

## `problem.json` schema

字段含义见前面英文版 `problem.json schema` 一节，下面只重述要点：

- `target` 块下面挂 L1-L5 全部证据文件路径
- `symptom` 块描述悬崖：metric、方向、观测值、与最近邻居的比值
- `benchmark_suite` 把 workload 分成三种角色：
  - `target` （必须改善）
  - `neighbors` （bracket 了悬崖的邻居，验证修复确实推动了 cliff）
  - `controls` （无关 workload，做 negative control）
- `evidence` 块的 `key_signals` 列表每条都带 `source` + `field` +
  `value` + `weight`，做题人能逐条审计
- `suggested_strategies` 是 setter 给 solver 的路由提示，每条有
  `route_to: config-agent` 之类的派遣字段
- `lineage` 指回 stage 1 跑过的具体 run 目录 + scoring 记录 + noise
  baseline

---

## `acceptance_criteria.json` schema

```jsonc
{
  "schema_version": 1,
  "problem_id": "P001",

  // primary：悬崖本身
  "primary": {
    "on": "target",
    "metric": "ttft_p95_ms",
    "direction": "lower",
    "required_improvement_pct": 30.0,         // 必须在 target 上把 TTFT p95 至少降低 30%
    "stretch_improvement_pct":  60.0          // 出题人期望（仅信息性）
  },

  // constraints：必须不回归的东西
  "constraints": [
    {"on": "target",            "metric": "request_throughput", "direction": "higher", "max_regression_pct": 10.0},
    {"on": "controls.smoke",    "metric": "ttft_p95_ms",        "direction": "lower",  "max_regression_pct": 5.0},
    {"on": "any",               "metric": "oom",                "max": 0},
    {"on": "any",               "metric": "server_crash",       "max": 0},
    {"on": "any",               "metric": "failed_requests",    "max": 0}
  ],

  // 统计验证
  "verification": {
    "always_paired_ab": false,
    "paired_ab_required_when_delta_lt_pct": 5.0,    // 5% 以下的 delta 必须跑 paired A/B
    "min_repeats_per_workload_for_keep": 3
  },

  // 加分项：能做到更好就更好
  "stretch_goals": [
    "neighbors.mc_32 也提升至少 10%（加分，不必须）"
  ]
}
```

---

## 验证规则（由 `validate_problem.py` 强制执行，待实现）

题目包**有效当且仅当**满足以下**所有**条件：

1. `problem.json.schema_version == 1`。
2. `frozen == true`。（含义：题目包**为了可复现而不可变**。做题人可以
   随便**读**、**写到 `attempts/` 子目录**；做题人绝不能**改**包根目录
   下的文件。）
3. `target`、`benchmark_suite.*`、`evidence.key_signals[].source` 引用
   的每个相对路径都真实存在。
4. `target.profile_summary` 要么指向真实文件，要么是 `null` 加一条自由
   格式 `target.skip_L4_reason` 备注（例如 `"信号已足够强，无需 profile"`、
   `"profiler 当前不可用"`）。L4 是 **agent 自行决定**的，不是强制。
5. `target.repeats` 长度 0 或更多 —— L5 重复次数由 agent 根据观察到的
   CV 跟 noise baseline 比对后决定。
6. `benchmark_suite.neighbors` 长度 ≥ 1（没邻居就没法做 A/B）。
7. `benchmark_suite.controls` 长度 ≥ 1。
8. `acceptance_criteria.json.primary.metric` 跟 `symptom.metric` 一致。
9. `suggested_strategies` 非空（出题人必须至少给一个路由提示）。
10. `hypothesis.md` 存在且至少 100 字符的散文。

做题人可以拒绝处理无效题目包，在题目包里写一份 `rejection.md` 然后
停。

---

## 反作弊：实际怎么生效（不需要 hash）

题目包**没有**显式 hash 文件或完整性校验。完整性是靠**约定 + 可复现
性**保证的：

- 题目包根目录（除 `attempts/` 之外）**约定为只读**。出题人的
  `seal_problem.py`（待实现）在 L4/L5 收集完之后通过 `chmod -w` 把这些
  文件设为不可写。
- 做题人 harness 只往 `attempts/attempt_NNN/` 下面写。任何尝试改
  `workload.yaml` 或 `neighbors/*.yaml` 的做题人脚本会在文件系统层失败。
- **最终的检查**：任何人——另一个做题人、出题人 re-review、人类
  reviewer——都可以把 accepted `solution.json` 的配置在题目包**原始
  workload yaml** 上重跑，验证 primary metric 的提升能否复现。如果不
  复现，解方案被作废。

换句话说：如果做题人通过"偷偷改 workload yaml"作弊，作弊行为是**事
后可自动检测的**，因为题目包的原始 yaml 还在那里供任何人重跑。不需要
hash，只要包是不可变的 + 重跑能验证就够了。

做题人**可以自由做的事**：

- 读包里的每个字节，含 L4 profile traces。
- 读其他题目包和 idea 池。
- 为自己的诊断目的**在题目包外**额外跑探测 benchmark——这些不算入
  verification 决策；只有 suite 里的 workload 算数。
- **提议新 workload**（增量地，从不修改已有的）：写
  `experiments/ideas/from_solver/idea_*.json`，type 为
  `proposed_workload`。出题人下次会考虑。
- **报告旁征解决的问题**：如果做题人在解 P001 时顺带发现并修复了相关
  issue（例如某个 control workload 也因同一 knob 改动而改善），把它
  记到 `decision.json.also_solved[]` 和最终 `solution.md`。

---

## 做题人写回什么

做题人**不**改题目包根目录的文件。它写到题目包的 `attempts/` 子目录
下（这样题目包是一份自包含的实验记录）：

```
experiments/problems/P001/
├── ...（上面冻结的包内容）
├── attempts/
│   ├── attempt_001/
│   │   ├── plan.md
│   │   ├── candidate_config.yaml         （config-agent attempt）
│   │   │   或 kernel_patch.diff          （kernel-agent attempt）
│   │   ├── verification/
│   │   │   ├── target_metrics.json
│   │   │   ├── neighbors/mc_16_metrics.json
│   │   │   ├── neighbors/mc_32_metrics.json
│   │   │   ├── controls/smoke_metrics.json
│   │   │   └── ab_summary.json           （需要时）
│   │   └── decision.json
│   ├── attempt_002/ ...
│   └── ...
├── solution.md             # 最终接受的解方案报告（人 + LLM 读）
└── rejection.md            # 无解或题目无效时
```

### `decision.json` schema（每个 attempt 一份）

```jsonc
{
  "schema_version": 1,
  "attempt_id": "attempt_003",
  "solver_agent": "config-agent",
  "decision": "keep",                                    // keep | revert | needs_more_evidence
  "reasoning": "自由散文，引用 verification metrics。",
  "primary_delta_pct": 64.2,                             // target 指标改善
  "constraint_violations": [],                           // 没违规时为空
  "also_solved": [                                       // 新增：本 attempt 顺带修复的旁征
    {
      "ref": "neighbors.mc_32",
      "metric": "ttft_p95_ms",
      "delta_pct": 12.0,
      "note": "邻居也改善了约 12% —— 同一 admission cap 修复。"
    }
  ],
  "new_findings_filed": [                                // 本 attempt 写入的 ideas
    "experiments/ideas/from_solver/idea_007.json"
  ]
}
```

### `solution.md` —— 最终报告（只有 accepted 时写）

双语 markdown 报告。包含：

1. 题目一段话总结（出了什么问题、为啥重要）。
2. 修复（哪个子 agent 应用的、哪个 knob/patch）。
3. target + neighbors + controls 上的量化收益。
4. 旁征解决的问题（来自跨所有 attempt 聚合的 `also_solved`）。
5. 工作过程中提交的新 ideas。
6. 可复现性：复现该收益的精确命令。

这份文件**就是**交给 reviewer 的实验记录。

---

## 版本管理

- 新字段可以加，不需要升 `schema_version`。做题人必须容忍未知字段。
- 删字段或改名是破坏性改动 → 升 `schema_version`。出题人和所有做题人
  必须同步升级。
- `frozen=true` 不可协商；绝不覆盖已发布的题目。如果要重发，用新的
  `problem_id`（`P001_v2`）并 link `lineage.supersedes`。

---

## 从 v0.3 `case.json` 迁移

v0.3 的 `experiments/regimes/cases/SNNN/case.json` 大致等价于：

```
problem.json（子集）
  其中
    target.profile_summary = null
    target.skip_L4_reason  = "v0.3 had no profiling skill"
    benchmark_suite.neighbors = []
    benchmark_suite.controls  = []
    evidence.layers_collected = ["L1", "L2", "L3"]
```

迁移脚本 `migrate_v03_case_to_problem.py` 可以把 v0.3 case + wave-1
邻居 metric（已经被记录下来了）提升成有效的题目包，先 `frozen=false`，
等 neighbors/controls 补全 + L4 加上之后再 freeze。
