# Skill architecture — how the 14 skills fit together (2026-06-09)

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#中文版)

This doc explains the **pipeline** the 14 skills form when an agent investigates
an inference-perf question. It complements `docs/skills/README.md` (which is a
catalog) by showing **flow + division of labour**.

## 1. Two-agent split

The skills are designed for a future split into two agent roles:

| Role | What it does | Skills it owns |
|---|---|---|
| **Analysis agent** (free hands) | reads raw profiles, source, logs; forms hypotheses; produces handoffs | bench, profiling, anomaly-finding, source-reading skills |
| **Coding agent** (bounded scope) | reads ONE handoff at a time; applies a patch; verifies via acceptance test; reverts on fail | only consumes `handoff-prompt-template` artifacts + cited file paths |

Today both roles are played by the same LLM (me, the CLI agent). The handoff
template is the **contract** between them — written today, ready for the day
we actually fork the roles.

## 2. The pipeline (left-to-right data flow)

```
            ┌──────────────────────────────────────────────────────────────┐
            │                  ANALYSIS-AGENT  WORKSPACE                    │
            │                                                              │
  workload                                                                  │
  configs                                                                   │
     │                                                                      │
     ▼                                                                      │
  ┌───────────────────┐    ┌───────────────────┐                            │
  │ regime-sweep-     │───▶│ cross-regime-     │── findings ──┐             │
  │   runner          │    │   anomaly         │              │             │
  │ (calls e2e-bench- │    │                   │              │             │
  │  runner per cell) │    │                   │              │             │
  └───────────────────┘    └───────────────────┘              │             │
     │                                                        ▼             │
     │     If a cell is "interesting" or large gap:        decision         │
     │                                                     "this is        │
     │                                                      worth deep      │
     │                                                      profiling"     │
     │                                                        │             │
     ▼                                                        ▼             │
  ┌───────────────────┐    ┌───────────────────┐                            │
  │ nsys-capture      │───▶│ nsys-timeline-sql │                            │
  │ (record .sqlite)  │    │ (top kernels,     │                            │
  └───────────────────┘    │  idle gaps, etc)  │                            │
                           └───────────────────┘                            │
                                    │                                       │
  ┌───────────────────┐             │                                       │
  │ pytorch-profiling │             │                                       │
  │ (sglang) OR       │             │                                       │
  │ vLLM torch trace  │─────────────┤                                       │
  │ via /start_profile│             │                                       │
  └───────────────────┘             │                                       │
                                    ▼                                       │
                            ┌───────────────────┐                           │
                            │ profile-summary-  │                           │
                            │   unified         │── profile_unified.json ───┤
                            │ (merges all       │                           │
                            │  with evidence_   │                           │
                            │  chain)           │                           │
                            └───────────────────┘                           │
                                    │                                       │
                                    ▼                                       │
                            ┌───────────────────┐                           │
                            │ handoff-prompt-   │                           │
                            │   template        │── handoff.md ─────────────┼──▶  coding agent
                            │ (the contract)    │                           │     receives the
                            └───────────────────┘                           │     handoff.md and
            └──────────────────────────────────────────────────────────────┘     applies the patch
```

## 3. Skill-by-skill role

| Skill | Role in the pipeline | Reads | Writes |
|---|---|---|---|
| `e2e-bench-runner`        | macro signal (req/s, latency, reliability) | URL + regime YAML | `bench_summary.json` |
| `regime-sweep-runner`     | N×M matrix of (config, regime) | `configs.yaml` + `regimes.yaml` | `regime_sweep_summary.json` |
| `cross-regime-anomaly`    | "where to dig" finder (the autonomy skill) | sweep matrix | ranked `anomaly_report.json` |
| `nsys-capture`            | capture per-kernel timeline | shell cmd + duration | `.nsys-rep` + `.sqlite` |
| `nsys-timeline-sql`       | SQL-reduce nsys to numbers | `.sqlite` | `timeline_summary.json` (+ `query` subcmd) |
| `pytorch-profiling`       | sglang-specific per-kernel + phase | sglang server + workload | `profile_summary.json` |
| `ncu-microarch`           | "why is THIS kernel slow?" — SM occupancy, TC utilization, warp stalls | running cmd + kernel regex | `ncu_summary.json` + verdict |
| `server-log-mining`       | config-shaped problems (KV pressure, max_running_requests, etc.) | `server.log` | `server_features.json` |
| `failure-classification`  | what kind of crash / OOM / timeout | per-run result | `failure_class.json` |
| `noise-aware-scoring`     | thresholded "is this signal or noise?" | metric series | scored bools |
| `suspicion-scoring`       | within-config anomaly score | server_features + noise + failure | suspicion per run |
| `boundary-expansion`      | generate neighbour workloads along an axis | seed workload + axis | N workload YAMLs |
| `minimal-repro-shrink`    | shrink failing workload until symptom disappears | failing workload | minimal repro |
| `profile-summary-unified` | merge all profiling outputs → one canonical artifact | bench + timeline + torch trace + **ncu** | `profile_unified.json` + `evidence_chain` |
| `handoff-prompt-template` | analysis → coding contract | unified profile + suggested change | `handoff.md` (markdown, hand-edited) |

## 4. Methodology cross-cuts (not skills, but rules every skill follows)

1. **Reliability gate** (`e2e-bench-runner` stddev_pct ≤ 8% → `reliable=true`).
   No conclusion is allowed from an unreliable cell.
2. **Predict-then-verify**. Every analysis SKILL.md has a METHODOLOGY section
   requiring a 1-sentence prediction before running, compared post-run.
3. **Skill attribution**. `evidence_chain` in `profile_unified.json` and the
   `Evidence chain` section in `handoff.md` together ensure every numeric claim
   in a report traces back to a specific skill + file.
4. **Loud failure**. Every skill emits `{"ok": false, "error": "..."}` rather
   than silently producing zeros. The `cross-regime-anomaly` skill treats a
   `failed_cell` as its own finding kind.

## 5. NCU is now unlocked (2026-06-09 evening)

`kernel_micro` field in `profile_unified.json` is now **populated** when the
agent passes `--ncu-summary` to `profile-summary-unified`.

The new `ncu-microarch` skill wraps `sudo -n ncu` against a specific kernel
regex, outputs SM occupancy / DRAM throughput / L1+L2 hit / Tensor Core
utilization / top warp-stall reason. Verdict enum: `compute_bound /
memory_bound / latency_bound / low_occupancy / tensor_core_idle / balanced`.

Real-world first finding (2026-06-09 evening): CUTLASS MoE GEMM on Qwen3 shape
shows `tensor_pipe_active_pct = 7.7%` — Tensor Cores nearly idle on a bf16 GEMM.
This flipped the improvement-direction priority in the morning's investigation.

**Remaining ncu limitation**: same as nsys-capture — can only profile its own
spawned subprocess, not an already-running vLLM/sglang server. Workaround is
to drive a standalone microbench (or, for vLLM-specific kernels, use vLLM's
own torch.profiler endpoint for time-domain data and ncu only for
microarchitectural counters on standalone reproductions).

## 6. Where to add new skills

Use the existing taxonomy:

- **Workload generation** (regime/boundary/shrink): add under `regime_*` or
  `*_shrink` family.
- **Data capture** (bench/nsys/torch/ncu): add `<source>-capture`.
- **Data reduction** (SQL/CSV/text → JSON summary): add `<source>-summarize`.
- **Cross-source analysis** (anomaly/diff/scoring): add `<verb>-<scope>`.
- **Methodology / template** (handoff/audit): pure SKILL.md, no impl.

Refuse to add a skill when:
- It doesn't have a single, named failure mode in its `WHY` section.
- Its output isn't machine-readable.
- It overlaps an existing skill by >50%; extend the existing one instead.

---

# 中文版

## 1. 两段式 agent 划分

skill 是为**未来分裂出两个 agent 角色**而设计的:

| 角色 | 做什么 | 拥有哪些 skill |
|---|---|---|
| **Analysis agent**(自由读) | 读 raw profile / 源码 / 日志,提假设,产出 handoff | bench、profiling、anomaly-finding、源码阅读这些 skill |
| **Coding agent**(范围有限) | 一次只读一份 handoff,应用 patch,跑 acceptance test,失败就 revert | 只消费 `handoff-prompt-template` 产出 + handoff 里引用的文件路径 |

今天两个角色都是同一个 LLM(就是我,CLI agent)演。handoff 模板是它们俩之间的
**合同** — 今天就写好,等到真正拆开时直接能用。

## 2. 流水线(数据从左往右流)

(图同英文版)

## 3. 每个 skill 的角色

(表同英文版)

## 4. 横切方法论(不是 skill,但每个 skill 都遵守)

1. **可靠性门限** — `e2e-bench-runner` 算 stddev,>8% 标 `reliable=false`,不可
   靠的 cell 不允许下任何结论
2. **predict-then-verify** — 每个 analysis skill 的 SKILL.md 的 METHODOLOGY
   章节都强制要求"跑前写一句预测,跑后对比"
3. **Skill 归因** — `profile_unified.json` 里的 `evidence_chain` + `handoff.md`
   的 `Evidence chain` 章节,确保报告里**每一个数字**都能追溯到具体 skill + 文件
4. **失败要响亮** — 每个 skill 都返回 `{"ok": false, "error": "..."}`,
   不允许默默产生 0 值。`cross-regime-anomaly` 把 `failed_cell` 当成一种 finding kind

## 5. NCU 解锁 (2026-06-09 晚)

`profile_unified.json` 里 `kernel_micro` 字段**现在可以填上了** — 给
`profile-summary-unified` 加 `--ncu-summary` 参数即可。

新 skill `ncu-microarch` 包 `sudo -n ncu`,出 SM 占用率 / DRAM throughput /
L1+L2 hit / Tensor Core 利用率 / top warp stall 原因。Verdict 枚举:
`compute_bound / memory_bound / latency_bound / low_occupancy /
tensor_core_idle / balanced`。

**首个真实发现**(2026-06-09 晚): CUTLASS MoE GEMM 在 Qwen3 shape 上
`tensor_pipe_active_pct = 7.7%` — bf16 GEMM Tensor Core 几乎没用。这翻转了
早上调查报告的改进方向排序。

**NCU 剩余限制**:跟 nsys-capture 一样 — 只能 profile 自己启动的子进程,
不能 attach 已经在跑的 vLLM/sglang server。绕过办法是写独立 microbench
(或者对 vLLM 特定 kernel,用 vLLM 自带的 torch.profiler 端点拿时间数据,
NCU 只对独立 reproduction 测微观计数器)。

## 6. 加新 skill 时怎么命名

按现有分类:

- **Workload 生成**(regime/boundary/shrink): 放 `regime_*` 或 `*_shrink` 家族
- **数据抓取**(bench/nsys/torch/ncu): `<source>-capture`
- **数据归约**(SQL/CSV/text → JSON summary): `<source>-summarize`
- **跨数据源分析**(anomaly/diff/scoring): `<verb>-<scope>`
- **方法论/模板**(handoff/audit): 纯 SKILL.md,无 impl

**不要**加的情况:
- `WHY` 章节说不出**一个具体的、命名过的失败模式**就要防住的
- 产出不是 machine-readable 的
- 跟现有 skill 重合 >50% — 应该扩展现有 skill,而不是新加
