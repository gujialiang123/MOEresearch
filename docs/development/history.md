# Project history — how we got here

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#中文版)
>
> Timeline of the architectural evolution from v0.1 (single-agent fantasy)
> to v0.4 (two-stage Problem-Setter / Problem-Solver). Read this to
> understand **why** the repo looks the way it does today.

## Timeline at a glance

```
v0.1 ──────► v0.2 ──────► v0.3 (skipped name) ──────► v0.4 (current)
2026-05-26   2026-05-27   ─                          2026-05-28

single LLM   three-stage  no separate                two-stage
"end-to-end" Scout +      version cut;               Problem-Setter
optimization Diagnose +   Stage 1 just got           + Problem-Solver
agent design Fix          implemented in v0.2-shape  Fleet
```

## v0.1 — the original sketch (2026-05-26)

**Idea**: one LLM agent gets a model + a GPU and "just figures out the
best config + optimizations". User wrote a design doc.

**Reality check from this codebase**: the doc had concrete bugs:
- it called `sglang --config <yaml>` (no such flag exists)
- it called `copilot --agent <name>` (no such flag exists)
- it assumed the LLM could just "look at numbers and decide", with no
  contract on what counts as evidence
- the loop had no termination criterion

**Outcome**: archived as a learning experiment. The doc is **not** in
`archive/` because we kept iterating on it directly before cutting v0.2.

## v0.2 — three-stage design (2026-05-27)

**Idea**: separate the work into three explicit stages:

```
Stage 1 — RegimeScout  (find perf cliffs)
Stage 2 — Diagnose     (figure out why)
Stage 3 — Fix          (apply changes)
```

**Documents** (preserved in `archive/`):
- [`archive/DESIGN_v0.2.md`](../../archive/DESIGN_v0.2.md) — full system spec, ~2,000 lines
- [`archive/TWO_STAGE_SUPPLEMENT_v0.2.md`](../../archive/TWO_STAGE_SUPPLEMENT_v0.2.md) — later supplement
  that already started hinting at merging stages

**What got built under v0.2 design**:
- The whole Stage 1 harness: `scripts/run_experiment.py`,
  `scripts/run_regime_suite.py`, `scripts/launch_server.py`,
  `scripts/run_benchmark.py`, `scripts/wait_ready.py`,
  `scripts/parse_metrics.py`, `scripts/cluster_regimes.py`,
  `scripts/generate_seed_suite.py`, `scripts/utils.py`.
- The 10 seed regimes under `regime_scout/candidates/`.
- The first 6 skills under `.github/skills/`.

The Stage 2 and Stage 3 directories were created as empty placeholders.

## v0.3 — no, just the "evidence pipeline" rev (early 2026-05-28)

This number was used informally for the **v0.2 implementation as actually
shipped**, after some bugs were fixed during the first real run:
- the `--config` non-flag bug → `yaml_config_to_argv()` translator
- the noise vs. signal problem → noise-aware scoring v2 skill
- the "scorer didn't look at server.log" problem → server-log-mining
  skill (the moment that produced the famous
  `max-running-requests=32` discovery)

No separate doc cut. v0.3 lives in
[`docs/development/restructure-notes.md`](./restructure-notes.md) as
the baseline that v0.4 refactored from.

## v0.4 — two-stage refactor (2026-05-28, afternoon → evening)

**Trigger**: user observed that diagnosis informs discovery. You can't
tell whether to expand `max_concurrency` or `input_len` until you've
profiled. So the v0.2 split between Stage 1 (Scout) and Stage 2
(Diagnose) was wrong — they had to be merged.

**Decision**: collapse Scout + Diagnose into one **Problem-Setter** stage
("出题人"); leave fixing as the **Problem-Solver Fleet** ("做题人"). Two
stages, algorithm-competition metaphor.

**Architecture changes** (all on 2026-05-28):
1. `stages/stage1/` → `stages/problem-setter/`
2. `stages/stage2/` deleted (work absorbed into Setter)
3. `stages/stage3/` → `stages/problem-solver/`
4. Stage A output type changed: ad-hoc `cases/` directories
   → self-contained `experiments/problems/PNNN/` packages
5. Stage A → Stage B hand-off contract written:
   [`docs/problem-package/schema.md`](../problem-package/schema.md)
6. Idea pool added (bidirectional channel between stages):
   [`docs/idea-pool/schema.md`](../idea-pool/schema.md)
7. Integrity model: **convention over hashes** (no SHA lockfile —
   re-running solver against unmodified package YAMLs auto-detects
   cheating)
8. Solver fleet: keep 4 sub-agents (config / scheduler / kernel /
   workload-shape). Build them on demand.

Detailed restructure log:
[`docs/development/restructure-notes.md`](./restructure-notes.md).

## What got implemented under v0.4

| Date | What | Where |
|---|---|---|
| 2026-05-28 morning | Stage 1 (still v0.2-shape) first real run on Qwen3-0.6B → case S001 | `experiments/regimes/cases/S001/` + `experiments/regimes/STAGE1_REPORT_20260528.md` |
| 2026-05-28 afternoon | Restructure to v0.4 two-stage + 6 skills + 3 anchor docs | git commit `52dafb6` |
| 2026-05-28 afternoon | `scripts/select_problems.py` (v1 problem-package producer); S001 → P001 | `experiments/problems/P001/` |
| 2026-05-28 afternoon | MoE Stage A first run | `experiments/problems_moe/P001/` + `experiments/regimes/MOE_STAGE_A_REPORT_20260528.md` |
| 2026-05-28 afternoon | Stage B config-agent v0 (single value) | `scripts/solver/config_agent.py` |
| 2026-05-28 afternoon | **First Stage B win**: MoE P001 attempt_001 → TTFT p95 −92.6%. Git push to GitHub. | commit `ead93ad` |
| 2026-05-28 evening | Finding-B probe → R-001 idea **closed** as measurement artefact | `regime_scout/outputs/moe_finding_b_results.jsonl` |
| 2026-05-28 evening | Progress report for the boss | `docs/reports/2026-05-28-progress.md` |
| 2026-05-28 evening | Regime-search research review (R1-R9 future directions) | `docs/research/regime-search-extensions.md` + commit `5c2da83` |
| 2026-05-28 evening | Stage B `--exhaustive` sweep + pytorch-profiling skill | commit `d1c1321` |
| 2026-05-28 evening | Docs reshuffle: top-level files → `docs/`, new short README | commit `6665145` |

## Git commits

```
6665145  Docs: rewrite README as repo intro + move dev docs to docs/
d1c1321  Stage B: exhaustive config sweep + pytorch-profiling skill
5c2da83  Add regime search extensions research review
ead93ad  Stage B (Solver) first working end-to-end + 92.6% TTFT p95 fix on MoE
52dafb6  Initial commit: end-to-end SGLang optimization agent (v0.4)
```

## Lessons captured

1. **Diagnosis informs discovery** — keep them in one agent (the trigger
   for v0.4).
2. **The benchmark aggregates are not enough**. server.log mining (L2)
   has to happen before scoring, or you'll mistake "hit a config cap"
   for "workload is just hard". This produced the `server-log-mining`
   skill.
3. **Cold-start tail can fake a perf bug**. The Finding-B probe
   (n=16/64/256) saved us from filing a fake problem. Hence the
   `--warmup-requests` discipline now baked into the profiling skill.
4. **Diminishing returns are noise**. The first config-agent attempt was
   so good (+92.6%) that further sweep values (+92.8%, +92.9%) were
   indistinguishable. Picker now prefers smallest value within ±1%.
5. **Self-contained problem packages > "experiment IDs"**. The whole
   package can be `tar`'d, shared, re-run by a stranger. No global
   indexes to keep in sync.

## What v0.5+ probably looks like

Currently open directions (see
[`docs/research/regime-search-extensions.md`](../research/regime-search-extensions.md)
for full menu):

- **R1**: Two-axis interaction expansion (currently all triage is
  single-axis).
- **R6**: MoE-specific axes (expert imbalance, routing entropy, cold
  experts).
- **Bench Stage B kernel-agent on a real problem**. We have the profile
  skill ready; we need a problem to feed it.

---

<a id="中文版"></a>

# 中文版

# 项目演化历史

> 从 v0.1（"一个 LLM 全自动搞定"幻想）到 v0.4（两阶段 Problem-Setter
> / Problem-Solver）的架构演化时间线。读这份你能理解为什么仓库长成
> 今天这个样子。

## 时间线一图流

```
v0.1 ──────► v0.2 ──────► v0.3（跳过命名）──────► v0.4（当前）
2026-05-26   2026-05-27   ─                       2026-05-28

单 LLM       三阶段        没有单独 cut 版本；     两阶段
"端到端"     Scout +       Stage 1 按 v0.2         Problem-Setter
优化 agent   Diagnose +    形态落地实现            + Problem-Solver
设计         Fix                                  Fleet
```

## v0.1 —— 最初草图（2026-05-26）

**想法**：一个 LLM agent 拿到 model + GPU 就"自动搞定最好的 config +
优化"。用户写了一份设计文档。

**翻进 codebase 才发现的 bug**：
- 文档调用 `sglang --config <yaml>`（这个 flag 不存在）
- 文档调用 `copilot --agent <name>`（这个 flag 不存在）
- 假设 LLM"看一眼数就知道怎么办"，对什么算证据没契约
- loop 没有终止条件

**结局**：作为学习经验放下了。原文档没放进 `archive/`，因为我们直接
基于它迭代出了 v0.2。

## v0.2 —— 三阶段设计（2026-05-27）

**想法**：把工作显式拆成三阶段：

```
Stage 1 — RegimeScout（找性能 cliff）
Stage 2 — Diagnose（搞清楚为什么）
Stage 3 — Fix（实施修复）
```

**文档**（保留在 `archive/`）：
- [`archive/DESIGN_v0.2.md`](../../archive/DESIGN_v0.2.md) —— 完整系统 spec，~2000 行
- [`archive/TWO_STAGE_SUPPLEMENT_v0.2.md`](../../archive/TWO_STAGE_SUPPLEMENT_v0.2.md) ——
  后续补充，已经开始暗示阶段合并

**v0.2 设计下落地了什么**：
- 整套 Stage 1 harness：`scripts/run_experiment.py`、
  `scripts/run_regime_suite.py`、`scripts/launch_server.py`、
  `scripts/run_benchmark.py`、`scripts/wait_ready.py`、
  `scripts/parse_metrics.py`、`scripts/cluster_regimes.py`、
  `scripts/generate_seed_suite.py`、`scripts/utils.py`。
- `regime_scout/candidates/` 下的 10 个 seed regime。
- `.github/skills/` 下的前 6 个 skill。

Stage 2、Stage 3 目录建成空 placeholder。

## v0.3 —— 不算独立版本，只是 "evidence pipeline" 修订（2026-05-28 早段）

这个号其实是用来指 **首次真跑后修了一堆 bug 的 v0.2 实际落地版本**：
- `--config` 不存在的 bug → `yaml_config_to_argv()` 翻译器
- 噪声 vs 信号问题 → noise-aware scoring v2 skill
- "scorer 不看 server.log" 问题 → server-log-mining skill（也就是发现著名的
  `max-running-requests=32` 的那一刻）

没切独立文档。v0.3 在
[`docs/development/restructure-notes.md`](./restructure-notes.md)
里作为 v0.4 的重构基线存在。

## v0.4 —— 两阶段重构（2026-05-28 下午→晚上）

**触发**：用户指出 diagnosis 喂 discovery。在 profile 之前根本不知道
该扩 `max_concurrency` 还是 `input_len`。所以 v0.2 里把 Stage 1
（Scout）和 Stage 2（Diagnose）拆开是错的 —— 必须合并。

**决定**：把 Scout + Diagnose 合成一个 **Problem-Setter** 阶段
（"出题人"）；修复留给 **Problem-Solver Fleet**（"做题人"）。两阶段，
算法竞赛隐喻。

**架构变化**（全部于 2026-05-28）：
1. `stages/stage1/` → `stages/problem-setter/`
2. `stages/stage2/` 删除（工作并入 Setter）
3. `stages/stage3/` → `stages/problem-solver/`
4. Stage A 输出类型变化：临时 `cases/` 目录 → 自包含
   `experiments/problems/PNNN/` 题目包
5. Stage A → Stage B 交接契约：
   [`docs/problem-package/schema.md`](../problem-package/schema.md)
6. 新增 idea pool（两阶段双向通道）：
   [`docs/idea-pool/schema.md`](../idea-pool/schema.md)
7. 完整性模型：**约定大于 hash**（没有 SHA 锁文件 —— solver 复跑题目包
   原始 YAML 就能自动检出作弊）
8. Solver fleet：保留 4 个子 agent（config / scheduler / kernel /
   workload-shape）。按需建造。

详细重构日志：
[`docs/development/restructure-notes.md`](./restructure-notes.md)。

## v0.4 下落地了什么

| 日期 | 内容 | 位置 |
|---|---|---|
| 2026-05-28 早 | Stage 1（v0.2 形态）首跑 Qwen3-0.6B → case S001 | `experiments/regimes/cases/S001/` + `experiments/regimes/STAGE1_REPORT_20260528.md` |
| 2026-05-28 午后 | 重构为 v0.4 两阶段 + 6 skill + 3 anchor 文档 | git commit `52dafb6` |
| 2026-05-28 午后 | `scripts/select_problems.py`（v1 题目包生成器）；S001 → P001 | `experiments/problems/P001/` |
| 2026-05-28 午后 | MoE Stage A 首跑 | `experiments/problems_moe/P001/` + `experiments/regimes/MOE_STAGE_A_REPORT_20260528.md` |
| 2026-05-28 午后 | Stage B config-agent v0（单值） | `scripts/solver/config_agent.py` |
| 2026-05-28 午后 | **Stage B 首胜**：MoE P001 attempt_001 → TTFT p95 −92.6%。git push 到 GitHub。 | commit `ead93ad` |
| 2026-05-28 晚 | Finding-B 探测 → R-001 idea **关闭**（属测量伪影） | `regime_scout/outputs/moe_finding_b_results.jsonl` |
| 2026-05-28 晚 | 给老板的进度报告 | `docs/reports/2026-05-28-progress.md` |
| 2026-05-28 晚 | regime-search 研究综述（R1-R9 未来方向） | `docs/research/regime-search-extensions.md` + commit `5c2da83` |
| 2026-05-28 晚 | Stage B `--exhaustive` 扫描 + pytorch-profiling skill | commit `d1c1321` |
| 2026-05-28 晚 | 文档大搬家：顶层文件 → `docs/`，新写短 README | commit `6665145` |

## Git 提交记录

```
6665145  Docs: rewrite README as repo intro + move dev docs to docs/
d1c1321  Stage B: exhaustive config sweep + pytorch-profiling skill
5c2da83  Add regime search extensions research review
ead93ad  Stage B (Solver) first working end-to-end + 92.6% TTFT p95 fix on MoE
52dafb6  Initial commit: end-to-end SGLang optimization agent (v0.4)
```

## 留下来的经验

1. **Diagnosis 喂 discovery** —— 它们必须在一个 agent 里（v0.4 的触发原因）。
2. **bench 聚合数不够**。server.log 挖掘（L2）必须在打分前，否则你会把
   "撞到 config 上限"误判为"workload 就是难"。这条经验催生了
   `server-log-mining` skill。
3. **冷启动尾延会假装性能 bug**。Finding-B 探测（n=16/64/256）让我们没立
   假题。所以现在 profiling skill 默认带 `--warmup-requests` 纪律。
4. **diminishing returns 是噪声**。config-agent 第一次 attempt 就太好（+92.6%），
   后续 sweep 值（+92.8%、+92.9%）根本分不出来。Picker 现在 ±1% 内
   优先取最小值。
5. **自包含题目包 > "实验 ID"**。整包可 `tar` 走、可分享、陌生人也能复现。
   不用维护全局索引。

## v0.5+ 大概长啥样

当前开放方向（完整菜单见
[`docs/research/regime-search-extensions.md`](../research/regime-search-extensions.md)）：

- **R1**：两轴交互扩展（当前 triage 全是单轴）。
- **R6**：MoE 专有 axis（专家失衡、routing entropy、冷专家）。
- **在真实问题上跑 Stage B kernel-agent**。profile skill 已就绪，缺
  喂给它的真问题。
