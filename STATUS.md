# Restructure Status

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#-中文版)
>
> **A working note tracking the in-progress 2-stage refactor.** Read this
> first when resuming work to know what's done, what's pending, and what
> the design has converged on.
>
> Last updated: 2026-05-28 (afternoon session)

---

## TL;DR — where the project is right now

- **Foundational harness is stable** and produces real findings
  (Stage 1 v0.3 ran 10 seeds + 2 boundary neighbors on Qwen3-0.6B + H200
  and automatically discovered the `max-running-requests=32` cap as
  case S001 with a 4-component evidence trail, score 0.735).
- **Architecture has evolved twice**:
  1. v0.2 → 3-stage (Scout / Diagnose / Fix)
  2. **v0.4 (current) → 2-stage (Problem-Setter / Problem-Solver Fleet)**
- The 2-stage anchor docs are written. Downstream stage-specific docs
  still reference the old 3-stage naming and need updating. Code didn't
  change.

---

## Architecture (current, 2-stage)

Algorithm-competition metaphor:

```
Stage A — Problem Setter (出题人 Agent)
   one agent, rich tools (bench, log-mine, classify, profile, expand, repeat)
   produces: experiments/problems/PNNN/  (frozen problem packages)

Stage B — Problem Solver Fleet (做题人 Agent 群)
   per-specialty sub-agents: config / scheduler / kernel / workload-shape
   reads: a problem package + idea pool + cross-problem data (no firewall)
   writes: experiments/solutions/PNNN/attempts/...
           + experiments/ideas/from_solver/idea_*.json

Idea pool (bidirectional)
   experiments/ideas/{from_setter,from_solver}/idea_*.json
   solver observations → setter material for next regime search

Anti-gaming
   benchmark-integrity skill (designed, not impl): hashes workload yamls
   in each problem package; rejects solver attempts that mutate them.
```

The 3-stage design (Stage 1 / Stage 2 / Stage 3) is **superseded** but
its docs are still on disk and referenced from a few places.

---

## What's done in this restructure

### Files moved (and verified working)

| Old | New | Verified |
|---|---|---|
| `stages/stage1/` | `stages/problem-setter/` | ✅ rule_based_explore.py still runs |
| `stages/stage2/` | (deleted, was empty placeholder) | — |
| `stages/stage3/` | `stages/problem-solver/` | ✅ |
| `docs/stage1/` | `docs/problem-setter/` | ✅ |
| `docs/stage2/` | (deleted) | — |
| `docs/stage3/` | `docs/problem-solver/` | ✅ |

### New anchor documents (review these first)

| Path | Purpose | Status |
|---|---|---|
| `docs/architecture/two-stage-overview.md` | The architecture, layered evidence (L1-L5), idea pool data flow, what each side may/may-not do | ✅ written |
| `docs/problem-package/schema.md` | `problem.json` strong-typed contract, including `benchmark_suite` (target + neighbors + controls), `acceptance_criteria.json`, `integrity_hashes.json` (anti-gaming) | ✅ written |
| `docs/idea-pool/schema.md` | Bidirectional notes channel; `idea.json` schema; lifecycle (open → accepted → promoted_to_problem) | ✅ written |
| `experiments/ideas/README.md` | One-page intro pointing to docs/idea-pool/schema.md | ✅ written |

### Code (no changes this round)

- `scripts/*.py` — untouched.
- `.github/skills/*/impl/*.py` — untouched.
- `stages/problem-setter/policies/rule_based_explore.py` — only its
  parent path imports were fixed during the rename. Verified still
  reproduces S001 score=0.735.

---

## What still needs to happen (in priority order)

### Tier 1 — entry points (highest user-visibility)

| Todo | What |
|---|---|
| `update-readme` | Rewrite `README.md` §2 (3-stage diagram → 2-stage), §11 (rule-vs-LLM section needs the Problem-Setter/Solver framing). Both EN and CN. |
| `update-roadmap` | Rewrite `ROADMAP.md` milestones for 2-stage. Current M-numbers are 3-stage-flavored. |

### Tier 2 — Problem-Setter docs (need new-arch rewrite)

| Todo | What |
|---|---|
| `update-stage1-docs` | Five files in `stages/problem-setter/`: `AGENT_CONTRACT.md`, `PLAYBOOK.md`, `TOOLS.md`, `EXTENSION_GUIDE.md`, `policies/llm_agent.md`. Edits: rename "Stage 1" → "Problem-Setter"; add Phase X: deep-profile (L4); Phase Y: bias-removal repeats (L5); Phase Z: package as problem.json with benchmark_suite; add idea-pool consumption (Phase 0) and emission (any phase). |
|  | The `TOOLS.md` list grows by 2 (pytorch-profiling, benchmark-integrity). |

### Tier 3 — Problem-Solver docs (new content)

| Todo | What |
|---|---|
| `update-stage3-docs` | Six new files in `stages/problem-solver/`: `README.md`, `AGENT_FLEET.md` (dispatch table), `config-agent/CONTRACT.md`, `scheduler-agent/CONTRACT.md`, `kernel-agent/CONTRACT.md` (port from `/home/t-jialianggu/work/auto-gpu-kernel/template/CLAUDE.md`), `workload-shape-agent/CONTRACT.md`. |

### Tier 4 — Skill stubs (design only, no impl)

| Todo | What |
|---|---|
| `profile-skill` | `.github/skills/pytorch-profiling/SKILL.md` — the L4 skill. Wraps `sglang.bench_serving --profile` + a trace summarizer. |
| `integrity-skill` | `.github/skills/benchmark-integrity/SKILL.md` — the anti-gaming skill that hashes workload yamls. |

### Tier 5 — Architecture support docs

| Todo | What |
|---|---|
| `docs-arch` | `docs/architecture/agent-vs-harness.md` is referenced from `two-stage-overview.md` but the file isn't created. Either create a stub or remove the link. |
| (cleanup) | `archive/` needs a `README.md` explaining "these are superseded v0.2 docs". |

---

## Key design decisions made this session

1. **Stages 1 and 2 (Scout + Diagnose) merged into one agent (Setter).**
   Reason: diagnosis informs discovery. Cutting them apart broke the
   feedback loop. Profile-informed regime expansion needs an
   interleaving loop that can't cross a stage boundary.
2. **Setter cannot fix; Solver cannot mutate the problem package.**
   But "cannot mutate" is for **reproducibility**, not anti-cheating.
   The two sides are collaborative. Solver may freely read the package,
   cross-problem data, idea pool, etc.
3. **Anti-gaming is narrow and skill-enforced.** Only one rule: solver
   may not modify workload yaml fields that affect benchmark difficulty
   (`num_prompts`, `random_input_len`, `random_output_len`, `seed`,
   `flush_cache`, `max_concurrency`, etc.). Enforced by
   `benchmark-integrity` skill hashing those fields at package time and
   verifying before every solver attempt.
4. **Benchmark suite (not just target).** Every problem package
   includes target + ≥1 neighbor + ≥1 control. Solver runs the whole
   suite; primary metric improvement on target + no regression on
   neighbors/controls = acceptance.
5. **Layered evidence (L1-L5).** L1-L3 cheap, on every run. L4 (deep
   profile) gated by agent decision. L5 (repeats for bias removal)
   required at packaging time.
6. **Idea pool is bidirectional.** Solver writes observations during
   verification; Setter reads them at Phase 0 of next session.

---

## Open questions (answered 2026-05-28 evening)

All four open questions are now resolved. See decisions log below.

### Q1 — integrity_hashes lock fields ✅ DROPPED ENTIRELY

User decision: hash-based integrity check is unnecessary. Integrity is
enforced by **convention + reproducibility**:

- Problem package root is read-only by convention (no hash check at
  runtime).
- The solver harness only writes under `attempts/` subdirs.
- Any cheating (e.g. tweaking workload yaml) is auto-detectable post-hoc:
  anyone can re-run the solution against the package's original yamls
  and check the gain reproduces.

The `benchmark-integrity` skill is **dropped**. The `integrity_hashes.json`
section is removed from `docs/problem-package/schema.md`.

### Q2 — L4 deep profile timing ✅ AGENT-DECIDED

L4 is **not mandatory** at packaging time. The Setter agent decides
whether profile is needed based on signal strength. If skipped, just put
a free-form note in `target.skip_L4_reason` (e.g. `"signal already
strong"`). No structural enforcement.

### Q3 — Solver fleet composition ✅ KEEP 4 FOR NOW

`config-agent / scheduler-agent / kernel-agent / workload-shape-agent`
is fine for v0.4. More specialized sub-agents (`compile-agent`,
`quantization-agent`) can be added later when needed.

### Q4 — Naming ✅ KEEP "Problem-Setter / Problem-Solver"

Directory names stay English; human-facing docs are bilingual.

---

## New decisions made this evening

1. **Solver can propose new workloads.** Via idea pool with
   `kind: "proposed_workload"`. Setter considers them additively. Solver
   can also report a problem is unreasonable via `rejection.md`.
2. **Solver's solution report can include side-solved problems.**
   `decision.json.also_solved[]` and `solution.md` can document fixes
   that incidentally addressed other issues. This is *encouraged*.
3. **Everything related to one issue lives in one problem package.**
   The optimization process, strategies, candidate configs/patches,
   verification metrics, and final reports all live inside
   `experiments/problems/PNNN/attempts/` and `experiments/problems/PNNN/solution.md`.
   No separate `experiments/solutions/` tree. Each problem package = a
   complete reproducible experiment record.
4. **Setter focus discipline: ONE issue per session.** Diverging is
   bad for quality. Large-gap adjacent findings go to
   `experiments/ideas/from_setter/`; the setter does not chase them
   this round.
5. **Solver writes ideas only on LARGE-GAP findings.** Not every
   observation. Noise / expected side effects are not ideas.

---

## Bilingual policy (clarified)

| Doc category | Policy | Examples |
|---|---|---|
| **Human-facing reference** | Full bilingual (EN first, CN appended) | `README.md`, `STATUS.md`, `ROADMAP.md`, `docs/architecture/*`, `docs/problem-package/schema.md`, `docs/idea-pool/schema.md`, `SKILLS.md`, `LOGS.md`, `experiments/ideas/README.md` |
| **Agent-facing operational** | English-only (LLM efficiency; minimize context bloat) | All `SKILL.md` files, `stages/problem-setter/{AGENT_CONTRACT, PLAYBOOK, TOOLS, EXTENSION_GUIDE}.md`, `stages/problem-setter/policies/llm_agent.md` (system prompt), all `stages/problem-solver/**/*.md` once written |

The 8 human-facing reference docs are already bilingual as of this
session. Stage-1/2 agent-facing docs are still in English-only Stage-1
form; they'll be rewritten in English-only when we update them for the
2-stage architecture next session.

---

## How to resume next session

```bash
# 1. Read the three anchor docs first:
cat docs/architecture/two-stage-overview.md
cat docs/problem-package/schema.md
cat docs/idea-pool/schema.md

# 2. Confirm the live pipeline still works:
python stages/problem-setter/policies/rule_based_explore.py \
    --config configs/base.yaml --max-waves 1 --reuse-seed-run
# (should print "S001: scheduler_overhead_high_concurrency (score=0.735)")

# 3. Continue with Tier 1 todos: README.md + ROADMAP.md.

# 4. Then Tier 2: 5 problem-setter docs.

# 5. Then Tier 3: 6 problem-solver docs.
```

In SQL: ready todos are visible via the todo tool. Filter for status
`pending` and look at deps.

---

## Files that haven't moved (intentionally)

- `regime_scout/` (seed_suite.yaml, search_space.yaml, candidates/, outputs/): these are runtime artifacts, not docs. They'll get renamed if/when we adopt `experiments/problems/` as the primary case dir; for now the Stage 1 outputs still land here.
- `experiments/regimes/cases/S001/`: still has the v0.3 case.json schema. Migration to `problem.json` schema needs the L4+L5 evidence to be collected; not done.
- `scripts/select_cases_for_stage2.py`: name is stale ("stage2") but still works. Will be renamed to `select_problems.py` along with the v0.3-case → problem migration.

---

## Summary for the impatient reviewer

- Two-stage architecture is locked in conceptually and documented in
  three anchor files. **Please read those three first.**
- No code changed; verification command still produces the same S001
  result.
- 9 todos remain (`pending`), grouped into 5 tiers. None are urgent.
- The next session should start at Tier 1 (README + ROADMAP) and work down.

---
---

# 🇨🇳 中文版

> **追踪进行中的 2 阶段重构的工作笔记。** 下次接手时先读这份，了解
> 什么已经做了、什么待做、设计收敛到了什么。
>
> 最后更新：2026-05-28（下午 session）

---

## TL;DR — 项目当前在哪

- **基础 harness 稳定**，能产出真实发现（Stage 1 v0.3 在 Qwen3-0.6B
  + H200 上跑了 10 个 seed + 2 个 boundary neighbor，**自动发现了
  `max-running-requests=32` cap 这个 bug 作为 case S001**，含 4 维
  evidence trail，score 0.735）。
- **架构经历了两次演化**：
  1. v0.2 → 三阶段（Scout / Diagnose / Fix）
  2. **v0.4（当前）→ 两阶段（Problem-Setter / Problem-Solver Fleet）**
- 两阶段的 anchor 文档写好了。下游 stage 专属文档仍然引用旧的三阶段
  命名，需要更新。代码没动。

---

## 架构（当前两阶段）

算法竞赛比喻：

```
阶段 A — Problem Setter（出题人 Agent）
   单 agent，工具丰富（bench, log-mine, classify, profile, expand, repeat）
   产出：experiments/problems/PNNN/  （冻结的题目包）

阶段 B — Problem Solver Fleet（做题人 Agent 群）
   按专业分子 agent：config / scheduler / kernel / workload-shape
   读：题目包 + idea 池 + 跨题目数据（无防火墙）
   写：experiments/solutions/PNNN/attempts/...
       + experiments/ideas/from_solver/idea_*.json

Idea 池（双向）
   experiments/ideas/{from_setter,from_solver}/idea_*.json
   做题人观察 → 出题人下一轮 regime 搜索的素材

反作弊
   benchmark-integrity skill（已设计，未实现）：对每个题目包里的
   workload yaml 算 hash；拒绝任何修改它们的做题人 attempt。
```

三阶段设计（Stage 1 / Stage 2 / Stage 3）已被**取代**但 docs 还在
硬盘上，几个地方仍然引用着。

---

## 本次重构已完成

### 文件移动（已验证仍能工作）

| 旧 | 新 | 验证 |
|---|---|---|
| `stages/stage1/` | `stages/problem-setter/` | ✅ rule_based_explore.py 还能跑 |
| `stages/stage2/` | （删了，原本就是空占位） | — |
| `stages/stage3/` | `stages/problem-solver/` | ✅ |
| `docs/stage1/` | `docs/problem-setter/` | ✅ |
| `docs/stage2/` | （删了） | — |
| `docs/stage3/` | `docs/problem-solver/` | ✅ |

### 新写的 anchor 文档（**优先 review 这三份**）

| 路径 | 干啥的 | 状态 |
|---|---|---|
| `docs/architecture/two-stage-overview.md` | 架构图、L1-L5 分层 evidence、idea 池数据流、两边各能 / 不能做什么 | ✅ 双语 |
| `docs/problem-package/schema.md` | `problem.json` 强类型契约（含 `benchmark_suite` = target + neighbors + controls）、`acceptance_criteria.json`、`integrity_hashes.json`（反作弊） | ✅ 双语 |
| `docs/idea-pool/schema.md` | 双向笔记通道、`idea.json` schema、生命周期（open → accepted → promoted_to_problem） | ✅ 双语 |
| `experiments/ideas/README.md` | 一页 intro 指向 docs/idea-pool/schema.md | ✅ 双语 |

### 代码（本轮没动）

- `scripts/*.py` —— 没动。
- `.github/skills/*/impl/*.py` —— 没动。
- `stages/problem-setter/policies/rule_based_explore.py` —— 重命名时
  只改了 parent path 推算。验证仍复现 S001 score=0.735。

---

## 还需要做的（按优先级排序）

### Tier 1 — 入口（用户最先看到）

| Todo | 干啥 |
|---|---|
| `update-readme` | 重写 `README.md` §2（三阶段图 → 二阶段）、§11（rule-vs-LLM 一节要按 Problem-Setter / Solver 框架重写）。中英文都要。 |
| `update-roadmap` | 重写 `ROADMAP.md` 里程碑，按二阶段。当前 M 编号是三阶段味的。 |

### Tier 2 — Problem-Setter 文档（要按新架构重写）

| Todo | 干啥 |
|---|---|
| `update-stage1-docs` | `stages/problem-setter/` 下 5 份文件：`AGENT_CONTRACT.md`, `PLAYBOOK.md`, `TOOLS.md`, `EXTENSION_GUIDE.md`, `policies/llm_agent.md`。编辑：把"Stage 1"换成"Problem-Setter"；加 Phase X（deep-profile, L4）；加 Phase Y（bias 消除重复, L5）；加 Phase Z（按 problem.json + benchmark_suite 打包）；加 idea 池消费（Phase 0）和写入（任何 phase）。 |
|  | `TOOLS.md` 工具表新增两个（pytorch-profiling, benchmark-integrity）。 |

### Tier 3 — Problem-Solver 文档（新内容）

| Todo | 干啥 |
|---|---|
| `update-stage3-docs` | `stages/problem-solver/` 下 6 份新文件：`README.md`, `AGENT_FLEET.md`（调度表）、`config-agent/CONTRACT.md`、`scheduler-agent/CONTRACT.md`、`kernel-agent/CONTRACT.md`（从 `/home/t-jialianggu/work/auto-gpu-kernel/template/CLAUDE.md` 移植）、`workload-shape-agent/CONTRACT.md`。 |

### Tier 4 — Skill 框架（只写设计 SKILL.md，不实现）

| Todo | 干啥 |
|---|---|
| `profile-skill` | `.github/skills/pytorch-profiling/SKILL.md` —— L4 用。包装 `sglang.bench_serving --profile` + trace 摘要器。 |
| `integrity-skill` | `.github/skills/benchmark-integrity/SKILL.md` —— 反作弊 skill，对 workload yaml 算 hash。 |

### Tier 5 — 架构辅助文档

| Todo | 干啥 |
|---|---|
| `docs-arch` | `docs/architecture/agent-vs-harness.md` 被 `two-stage-overview.md` 引用了但文件不存在。要么写个占位，要么删 link。 |
| （清理） | `archive/` 加一份 `README.md` 说明"这些是被取代的 v0.2 文档"。 |

---

## 本 session 做的关键设计决策

1. **Stages 1 和 2（Scout + Diagnose）合并成一个 agent（Setter）。**
   理由：诊断本身决定发现的方向。切开两阶段就破坏了反馈环。Profile
   指导的 regime 扩展需要的是能跨"发现 ↔ profile"自由交错的循环，
   stage 边界做不到。
2. **出题人不能 fix；做题人不能改题目包。** 但"不能改"是为了
   **可复现**，不是反作弊。两边是协作的。做题人可以随便读题目包、
   跨题目数据、idea 池等等。
3. **反作弊很窄、由 skill 强制。** 唯一规则：做题人不能改 workload
   yaml 里影响 benchmark 难度的字段（`num_prompts`、`random_input_len`、
   `random_output_len`、`seed`、`flush_cache`、`max_concurrency` 等）。
   由 `benchmark-integrity` skill 在打包时对这些字段算 hash、每次做题
   attempt 前验证。
4. **benchmark 套件（不只是 target）。** 每个题目包必含 target +
   ≥1 个 neighbor + ≥1 个 control。做题人跑整个套件；target 上 primary
   metric 改善 + neighbors/controls 不回归 = 被接受。
5. **分层 evidence（L1-L5）。** L1-L3 便宜，每个 run 都跑。L4（深度
   profile）由 agent 决定何时跑。L5（重复消 bias）打包时强制。
6. **Idea 池双向。** 做题人在 verification 时写观察；出题人在下次
   session Phase 0 读取。

---

## ⚠️ 待你回答的 4 个 open question 都已解决（本晚 session）

### Q1 —— integrity_hashes 锁字段 ✅ 整段删除

用户决定：hash 完整性校验没必要。完整性靠**约定 + 可复现性**保证：
- 题目包根目录约定为只读（运行时无 hash 校验）。
- 做题人 harness 只往 `attempts/` 子目录写。
- 任何作弊（例如改 workload yaml）**事后可自动检测**：任何人都可以
  把 solution 在题目包原始 yaml 上重跑，看收益还在不在。

`benchmark-integrity` skill **删除**。`integrity_hashes.json` 一节从
`docs/problem-package/schema.md` 移除。

### Q2 —— L4 深度 profile 何时跑 ✅ AGENT 自行决定

打包时 L4 **不强制**。出题人 agent 根据信号强度决定是否需要 profile。
跳过时在 `target.skip_L4_reason` 写自由格式备注（例如`"signal already
strong"`）。无结构性强制。

### Q3 —— 做题人 fleet 组成 ✅ v0.4 暂定 4 个

`config-agent / scheduler-agent / kernel-agent / workload-shape-agent`
v0.4 够用。需要时再加更专门的（`compile-agent`、`quantization-agent`
之类）。

### Q4 —— 命名 ✅ 保留 "Problem-Setter / Problem-Solver"

目录名保留英文；面向人的文档双语。

---

## 本晚做的新决策

1. **做题人可以提议新 workload。** 通过 idea 池 `kind:
   "proposed_workload"`。出题人**增量**考虑。做题人也可以通过
   `rejection.md` 报告题目不合理。
2. **做题人的 solution 报告可以包含旁征解决的问题。**
   `decision.json.also_solved[]` 和 `solution.md` 可以记顺带修复的其它
   issue。这是**鼓励**的。
3. **一个 issue 相关的一切都在一个题目包里。** 优化过程、策略、
   candidate config/patch、verification metrics、最终报告都在
   `experiments/problems/PNNN/attempts/` 和
   `experiments/problems/PNNN/solution.md`。**没有独立的
   `experiments/solutions/` 树。** 每个题目包 = 一份完整可复现的实验
   记录。
4. **出题人 focus 纪律：一次 session 一个 issue。** 发散损害质量。
   大 gap 的旁征发现进 `experiments/ideas/from_setter/`；出题人本轮不
   追。
5. **做题人只在大 gap 发现时写 idea。** 不是每个观察都要写。噪声/
   预期副作用不算 idea。

---

## 双语策略（明确化）

| 文档类别 | 策略 | 例子 |
|---|---|---|
| **人看的参考文档** | 全双语（英文在前，中文追加） | `README.md`、`STATUS.md`、`ROADMAP.md`、`docs/architecture/*`、`docs/problem-package/schema.md`、`docs/idea-pool/schema.md`、`SKILLS.md`、`LOGS.md`、`experiments/ideas/README.md` |
| **Agent 用的操作文档** | 仅英文（LLM 效率，减少 context bloat） | 所有 `SKILL.md`、`stages/problem-setter/{AGENT_CONTRACT, PLAYBOOK, TOOLS, EXTENSION_GUIDE}.md`、`stages/problem-setter/policies/llm_agent.md`（system prompt）、`stages/problem-solver/**/*.md`（写时） |

8 份人看的参考文档已经在本 session 完成双语化。Stage-1 的 agent 操作
文档仍然是英文的（Stage-1 形态），下一 session 重写为二阶段时会保持
英文 only。

---

## 下次 session 怎么接手

```bash
# 1. 先读三份 anchor 文档:
cat docs/architecture/two-stage-overview.md
cat docs/problem-package/schema.md
cat docs/idea-pool/schema.md

# 2. 确认 live pipeline 还活着:
python stages/problem-setter/policies/rule_based_explore.py \
    --config configs/base.yaml --max-waves 1 --reuse-seed-run
# 应输出 "S001: scheduler_overhead_high_concurrency (score=0.735)"

# 3. 从 Tier 1 todo 继续: README.md + ROADMAP.md。
# 4. 然后 Tier 2: 5 份 problem-setter 文档。
# 5. 然后 Tier 3: 6 份 problem-solver 文档。
```

SQL 里看 ready todos：用 todo 工具过滤 `status='pending'`，看依赖。

---

## 没动过的文件（故意的）

- `regime_scout/`（seed_suite.yaml、search_space.yaml、candidates/、
  outputs/）：这些是 runtime 产物，不是 doc。如果 / 当我们采用
  `experiments/problems/` 作为主 case 目录时再考虑重命名；目前 stage 1
  输出还落这里。
- `experiments/regimes/cases/S001/`：还是 v0.3 的 case.json schema。
  迁到 `problem.json` schema 需要先补 L4+L5 evidence；没做。
- `scripts/select_cases_for_stage2.py`：名字过时（含"stage2"）但还能
  工作。等 v0.3 case → problem 迁移时一起改名为 `select_problems.py`。

---

## 给没耐心的 reviewer 的总结

- 两阶段架构概念上锁了，文档化在三份 anchor 文件里。**请先读那三份。**
- 代码没改；验证命令还能复现 S001。
- 还有 9 个 todo（`pending`），分 5 个 tier。都不紧急。
- 下次 session 应该从 Tier 1（README + ROADMAP）开始往下做。
