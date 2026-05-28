# `experiments/` — index of all experiment artefacts

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#中文版)

Every benchmark run, every problem package, every solver attempt, every
idea lives somewhere under this directory. This file is the index. If
you want to know "what experiments have we run?", start here.

## Layout

```
experiments/
├── README.md                     ← you are here
├── problems/                     ← Stage A output for dense models
│   └── P001/                     ← first real problem package
├── problems_moe/                 ← Stage A output for MoE models
│   └── P001/                     ← MoE scheduler tail (fixed by Stage B)
├── ideas/                        ← idea pool (bidirectional channel)
│   ├── README.md
│   ├── from_setter/              ← idea_001.json — R-001 MoE cold-start
│   └── from_solver/              ← (empty so far)
├── regimes/                      ← raw Stage A scout outputs + first-run reports
│   ├── STAGE1_REPORT_20260528.md     ← first real dense run (Qwen3-0.6B)
│   ├── MOE_STAGE_A_REPORT_20260528.md ← first real MoE run (Qwen3-30B-A3B)
│   └── cases/
│       └── S001/                 ← the legacy "case" produced before
│                                   the problem-package refactor (kept
│                                   for traceability; P001 is its v1
│                                   successor)
└── tmp/                          ← per-run server.log / quick_raw.jsonl
                                    (gitignored — regenerable)
```

## What runs have happened, in order

| # | Date | Model | Stage | Output | Headline |
|---|---|---|---|---|---|
| 1 | 2026-05-28 | Qwen3-0.6B | A (rule-based) | `experiments/problems/P001/` + `regimes/STAGE1_REPORT_20260528.md` | Auto-discovered `max-running-requests=32` capping `max_concurrency=64`; TTFT p95 99→434 ms cliff. Suspicion score 0.735. |
| 2 | 2026-05-28 | Qwen3-30B-A3B (MoE) | A (rule-based) | `experiments/problems_moe/P001/` + `regimes/MOE_STAGE_A_REPORT_20260528.md` | Same cap on MoE, **4× sharper cliff** (×15.5 TTFT jump, throughput regression). Filed R-001 (MoE cold-start tail). |
| 3 | 2026-05-28 | Qwen3-30B-A3B (MoE) | B (config-agent) | `experiments/problems_moe/P001/attempts/attempt_001/` | First Stage B win: set `max-running-requests=64` → **TTFT p95 −92.6%**, throughput +147%. |
| 4 | 2026-05-28 | Finding-B probe (MoE smoke n=16/64/256) | A re-investigation | `regime_scout/outputs/moe_finding_b_results.jsonl` | Showed the n=16 "smoke tail" was a measurement artefact (cold-start dominated); **closed** R-001 without filing a problem. |
| 5 | 2026-05-28 | Qwen3-30B-A3B (MoE) | B (config-agent `--exhaustive`) | `experiments/problems_moe/P001/attempts/attempt_002/`, `attempt_003/`, `solution.md` | Swept `max-running-requests ∈ {64, 96, 128}`: all three within ±0.4% (noise). Picker auto-selected 64 (smallest = lowest mem cost). |

The narrative version of all of the above is in
[`docs/reports/2026-05-28-progress.md`](../docs/reports/2026-05-28-progress.md).

## Anatomy of a problem package

```
experiments/problems[_moe]/PNNN/
├── problem.json              ← v1 manifest (schema: docs/problem-package/schema.md)
├── hypothesis.md             ← human-readable problem statement
├── acceptance_criteria.json  ← primary metric + constraints + verification thresholds
├── workload.yaml             ← the target workload
├── baseline_metrics.json     ← target's baseline (BEFORE fix)
├── classification.json       ← L3 evidence (failure classification)
├── server_features.json      ← L2 evidence (mined from server.log)
├── neighbors/                ← workloads that should bracket the cliff
│   └── *.yaml + *.baseline_metrics.json
├── controls/                 ← unrelated regimes that should NOT regress
│   └── *.yaml + *.baseline_metrics.json
├── attempts/                 ← Stage B writes here
│   └── attempt_NNN/
│       ├── plan.md
│       ├── candidate_config.yaml
│       ├── decision.json     ← keep / revert / needs_more_evidence
│       └── verification/{target,neighbors,controls}/...
└── solution.md               ← (Stage B exhaustive only) ranked summary
```

The whole directory is **self-contained**. `tar` it, hand it to anyone,
they can `python scripts/solver/config_agent.py --problem <dir>` and
reproduce.

## Reproducing any of the above

Every experiment has its candidate config + workload snapshot saved.
The canonical re-run shape is:

```bash
python scripts/run_experiment.py \
    --config   <PNNN>/attempts/attempt_NNN/candidate_config.yaml \
    --workload <PNNN>/workload.yaml \
    --mode quick
```

The MoE P001 solution.md prints this exact command.

## Where the raw scout data lives

```
regime_scout/outputs/
├── raw_results.jsonl              ← dense first run: 10 seeds × 1 row
├── suspicious_cases.jsonl         ← rows surviving the suspicion score
├── selected_cases.jsonl           ← legacy stage-1 selection (pre-v1)
├── selected_problems.jsonl        ← v1 problem selection → P001
├── moe_raw_results.jsonl          ← MoE first run: 5 seeds + 2 boundary neighbors
├── moe_suspicious.jsonl
├── moe_selected_problems.jsonl    ← → MoE P001
├── moe_finding_b_results.jsonl    ← Finding-B probe (n=16/64/256)
├── regime_map.md / regime_map.json ← human-readable suite summary
```

---

<a id="中文版"></a>

# 中文版

每一次 benchmark、每一个题目包、每一次 solver 尝试、每一条 idea 都在
本目录下。这份文件是索引。想知道"我们做过哪些实验？"先看这里。

## 目录结构

```
experiments/
├── README.md                     ← 当前文件
├── problems/                     ← Stage A 输出（dense 模型）
│   └── P001/                     ← 第一个真实题目包
├── problems_moe/                 ← Stage A 输出（MoE 模型）
│   └── P001/                     ← MoE 调度尾延（已被 Stage B 修好）
├── ideas/                        ← idea pool（双向通道）
│   ├── README.md
│   ├── from_setter/              ← idea_001.json —— R-001 MoE 冷启动
│   └── from_solver/              ← （目前空）
├── regimes/                      ← 原始 Stage A 扫描输出 + 首跑报告
│   ├── STAGE1_REPORT_20260528.md     ← dense 模型首跑（Qwen3-0.6B）
│   ├── MOE_STAGE_A_REPORT_20260528.md ← MoE 首跑（Qwen3-30B-A3B）
│   └── cases/
│       └── S001/                 ← 题目包重构前留下的旧"case"
│                                   （保留追溯性；P001 是其 v1 后继）
└── tmp/                          ← 单次运行的 server.log / quick_raw.jsonl
                                    （gitignore —— 可重生成）
```

## 已跑过的实验（按时间）

| # | 日期 | 模型 | 阶段 | 输出 | 关键结论 |
|---|---|---|---|---|---|
| 1 | 2026-05-28 | Qwen3-0.6B | A（rule-based）| `experiments/problems/P001/` + `regimes/STAGE1_REPORT_20260528.md` | 自动发现 `max-running-requests=32` 卡住 `max_concurrency=64`；TTFT p95 99→434 ms 出现 cliff。怀疑度 0.735。 |
| 2 | 2026-05-28 | Qwen3-30B-A3B (MoE) | A（rule-based）| `experiments/problems_moe/P001/` + `regimes/MOE_STAGE_A_REPORT_20260528.md` | MoE 上同样的 cap，**cliff 锐 4 倍**（TTFT 飙 15.5 倍，吞吐反而下降）。提了 R-001（MoE 冷启动尾延） idea。 |
| 3 | 2026-05-28 | Qwen3-30B-A3B (MoE) | B（config-agent）| `experiments/problems_moe/P001/attempts/attempt_001/` | Stage B 首胜：把 `max-running-requests` 设为 64 → **TTFT p95 −92.6%**，吞吐 +147%。 |
| 4 | 2026-05-28 | Finding-B 探测（MoE smoke n=16/64/256）| A 复测 | `regime_scout/outputs/moe_finding_b_results.jsonl` | 证明 n=16 的"smoke 尾延"是测量伪影（冷启动主导）；**关闭** R-001，没必要立题。 |
| 5 | 2026-05-28 | Qwen3-30B-A3B (MoE) | B（config-agent `--exhaustive`）| `experiments/problems_moe/P001/attempts/attempt_002/`、`attempt_003/`、`solution.md` | 扫描 `max-running-requests ∈ {64, 96, 128}`：三者差距在 ±0.4% 之内（噪声）。Picker 自动取 64（最小值 = 最低内存代价）。 |

这一段的"叙事版"在
[`docs/reports/2026-05-28-progress.md`](../docs/reports/2026-05-28-progress.md)。

## 一个题目包长啥样

```
experiments/problems[_moe]/PNNN/
├── problem.json              ← v1 清单（schema: docs/problem-package/schema.md）
├── hypothesis.md             ← 给人看的问题陈述
├── acceptance_criteria.json  ← 主指标 + 约束 + 验收阈值
├── workload.yaml             ← 目标 workload
├── baseline_metrics.json     ← 目标的 baseline（修复 *之前*）
├── classification.json       ← L3 证据（失败分类）
├── server_features.json      ← L2 证据（从 server.log 挖出）
├── neighbors/                ← 用来夹住 cliff 的邻居 workload
│   └── *.yaml + *.baseline_metrics.json
├── controls/                 ← 不相关的对照（不应回归）
│   └── *.yaml + *.baseline_metrics.json
├── attempts/                 ← Stage B 写入这里
│   └── attempt_NNN/
│       ├── plan.md
│       ├── candidate_config.yaml
│       ├── decision.json     ← keep / revert / needs_more_evidence
│       └── verification/{target,neighbors,controls}/...
└── solution.md               ← （仅 Stage B exhaustive 产出）排序总结
```

整个目录是 **自包含** 的。`tar` 一下发给别人，他可以直接
`python scripts/solver/config_agent.py --problem <dir>` 复现。

## 复现任何一次实验

每次 attempt 都把 candidate config + workload 快照存了。统一复跑命令：

```bash
python scripts/run_experiment.py \
    --config   <PNNN>/attempts/attempt_NNN/candidate_config.yaml \
    --workload <PNNN>/workload.yaml \
    --mode quick
```

MoE P001 的 solution.md 里印的就是这条命令。

## 原始 scout 数据在哪

```
regime_scout/outputs/
├── raw_results.jsonl              ← dense 首跑：10 seed × 1 行
├── suspicious_cases.jsonl         ← 通过怀疑度评分的行
├── selected_cases.jsonl           ← stage-1 旧版选题
├── selected_problems.jsonl        ← v1 选题 → P001
├── moe_raw_results.jsonl          ← MoE 首跑：5 seed + 2 边界邻居
├── moe_suspicious.jsonl
├── moe_selected_problems.jsonl    ← → MoE P001
├── moe_finding_b_results.jsonl    ← Finding-B 探测（n=16/64/256）
├── regime_map.md / regime_map.json ← 给人看的 suite 总览
```
