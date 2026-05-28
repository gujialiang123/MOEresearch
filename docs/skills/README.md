# Skills — design principles & catalog

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#-中文版)

## 1. What a skill is

A **skill** in this project is a *named, reusable unit of procedural knowledge*
that any agent can call. It answers the question **"how do I do X?"** —
*not* "should I do X?" (that's the agent's job).

Concretely a skill is a directory under `.github/skills/<skill-name>/` with:

```
.github/skills/<skill-name>/
  SKILL.md           ← required: front-matter + WHEN / WHY / HOW / OUTPUT-CONTRACT / FAILURE-MODES
  impl/              ← optional: Python scripts that implement the skill
    *.py
  tests/             ← optional: golden-file tests for the skill
```

Skills are **stage-agnostic and agent-agnostic**: the same `server-log-mining`
skill is consumed by Stage 1's scout, Stage 2's diagnoser, and Stage 3's
fixer. This is the whole point — write it once, use it everywhere.

## 2. Why skills (vs. just functions in `scripts/utils.py`)

Three concrete reasons:

1. **Discoverability for agents.** A custom Copilot agent can list available
   skills via `glob .github/skills/*/SKILL.md` and read their front-matter
   to know *what* exists and *when* to use it. Plain Python functions are
   invisible until you grep for them.

2. **Methodology persistence.** When we discover "looking at one metric jump,
   automatically expand along that axis" works well (Finding A in the
   2026-05-28 Stage 1 report), we name it `boundary-expansion`, write down
   the trigger condition, encode it as a skill, and *future agents inherit
   the methodology* without re-discovering it.

3. **Composability.** `suspicion-scoring` calls `server-log-mining` + 
   `noise-aware-scoring` + `failure-classification`. Each one is testable
   in isolation. If we improve `server-log-mining` to detect a new
   sglang event, every downstream skill benefits immediately.

## 3. Anatomy of a `SKILL.md`

Every skill must follow this structure exactly. The front-matter is
machine-readable so agents can filter.

```yaml
---
name: server-log-mining
description: One sentence — agents read this to decide if they want the skill.
version: 1
stage: [1, 2, 3]            # which stages may use this
inputs:
  - server_log: path to sglang server stdout/stderr log
outputs:
  - server_features.json    # what the skill produces
triggers:
  - "after every benchmark run"
  - "before assigning a suspicion score to a passed run"
depends_on: []              # other skills called inside
---
```

Then the body, with these required sections:

- **WHEN** — concrete trigger conditions for using this skill.
- **WHY** — the design rationale; what failure mode does this skill prevent?
- **HOW** — step-by-step procedure or pseudo-code.
- **OUTPUT CONTRACT** — exact schema of what the skill produces. Downstream
  skills depend on this contract.
- **FAILURE MODES** — what can go wrong; how to detect partial output.
- **ROADMAP** — known limitations and next iterations.

Optional sections: **EXAMPLES**, **REFERENCES**.

## 4. Naming convention

- kebab-case
- verb-noun OR noun-noun (`boundary-expansion`, `server-log-mining`)
- never include the model / hardware / stage in the name — skills must be
  general

Bad: `qwen-server-log-parser`, `h200-noise-baseline`, `stage1-shrink-prompts`.
Good: `server-log-mining`, `noise-aware-scoring`, `minimal-repro-shrink`.

## 5. Authoring rules

1. **One concept per skill.** If `WHEN` lists two unrelated triggers, split
   the skill.
2. **Output contract is sacred.** Once a skill ships its schema, downstream
   code depends on it. Add new fields, never remove or rename. Bump
   `version:` in front-matter on breaking changes.
3. **Skills must not modify state outside their declared outputs.** A skill
   that "while I was scanning the server log, also touched configs/best.yaml"
   is broken — that side effect belongs to an agent.
4. **Implementations live under `impl/`** and are callable as scripts.
   Importable from `scripts/skills/<name>.py` thin re-export if needed.
5. **Failure must be loud.** A skill returns structured failure (`{"ok":
   false, "error": "..."}`) rather than silently producing zeros or fake
   data. This is the same rule as `parse_metrics.py` (DESIGN §0).

## 6. Catalog (v0.3)

| Skill | Stages | Description | Status |
|---|---|---|---|
| [`server-log-mining`](.github/skills/server-log-mining/SKILL.md) | 1, 2, 3 | Parse sglang `server.log` into structured features (cuda graph bs captured, max_running_requests, KV pressure events, retract events, token usage peak). | ✅ v1 implemented |
| [`boundary-expansion`](.github/skills/boundary-expansion/SKILL.md) | 1 | Given one workload + a hypothesized axis, generate N neighbor workload YAMLs along that axis. | ✅ v1 implemented |
| [`noise-aware-scoring`](.github/skills/noise-aware-scoring/SKILL.md) | 1, 3 | Compute & apply per-metric coefficient-of-variation thresholds so noise doesn't trigger false positives. | ✅ v1 implemented |
| [`failure-classification`](.github/skills/failure-classification/SKILL.md) | 1, 2, 3 | Classify a failed run into one of: oom / server_crash / benchmark_timeout / kv_pressure / partial_success / parse_error / unknown. | ✅ v1 implemented |
| [`suspicion-scoring`](.github/skills/suspicion-scoring/SKILL.md) | 1 | Combine server-log-mining + noise-aware-scoring + failure-classification + local-nonlinearity into one suspicion score per workload run, with full evidence audit trail. | ✅ v1 implemented |
| [`minimal-repro-shrink`](.github/skills/minimal-repro-shrink/SKILL.md) | 1 (late), 2 | Binary-shrink a workload along (num_prompts, max_concurrency, input_len, output_len) until the symptom disappears. | 🟨 SKILL.md only; impl deferred to v0.4 |

## 7. How to add a new skill

```bash
SKILL=my-new-skill
mkdir -p .github/skills/$SKILL/impl
cp .github/skills/_template/SKILL.md .github/skills/$SKILL/SKILL.md
$EDITOR .github/skills/$SKILL/SKILL.md       # fill in WHEN / WHY / HOW / OUTPUT / FAILURES
# implement under .github/skills/$SKILL/impl/*.py
# add row to §6 catalog above
```

## 8. Where this came from

The methodology behind these skills was distilled from the first real
Stage 1 run on 2026-05-28 (see
`experiments/regimes/STAGE1_REPORT_20260528.md`). That run produced 10
workload measurements but the scoring function failed to flag the
**`max-running-requests=32` bug** that was clearly visible in
`server.log`. The lesson — *"the most useful signals were in places nobody
parsed"* — directly motivated `server-log-mining`. The lesson — *"we have
10 isolated points and zero neighbors, so anomaly detection is impossible"*
— directly motivated `boundary-expansion`. The lesson — *"9 of 10
workloads saturated the tail-ratio threshold because it was hardcoded at
3"* — directly motivated `noise-aware-scoring`.

In other words: **every skill in this catalog corresponds to a specific,
named failure of the v0.2 pipeline that we want to never repeat**. That
is the principle by which new skills should be proposed.

---
---

# 🇨🇳 中文版

## 1. Skill 是什么

项目里的 **skill** 是一个**命名的、可复用的过程性知识单元**，任何
agent 都可以调用。它回答的是"**怎么做 X**"——*不是* "该不该做 X"
（那是 agent 的活）。

具体上，一个 skill 是 `.github/skills/<skill-name>/` 下的一个目录，
含有：

```
.github/skills/<skill-name>/
  SKILL.md           ← 必需：front-matter + WHEN / WHY / HOW / OUTPUT-CONTRACT / FAILURE-MODES
  impl/              ← 可选：实现这个 skill 的 Python 脚本
    *.py
  tests/             ← 可选：golden-file 测试
```

Skill 是 **stage 无关 / agent 无关**的：同一个 `server-log-mining`
skill 会被出题人（之前的 Stage 1）和做题人 fleet（之前的 Stage 2/3）
同时消费。这就是它的全部价值——**写一次，到处用**。

## 2. 为什么用 skill（而不是直接在 `scripts/utils.py` 写函数）

三条具体原因：

1. **对 agent 可发现。** 一个 Copilot custom agent 可以
   `glob .github/skills/*/SKILL.md` 列出可用 skill，读 front-matter
   就知道有什么、什么时候用。纯 Python 函数除非你 grep 才看得见。
2. **方法论可沉淀。** 当我们发现"看到一个 metric 跳变，自动沿那个
   轴扩展" 这种模式 work 好（参见 2026-05-28 Stage 1 报告的 Finding
   A），把它命名为 `boundary-expansion`，写下触发条件，编码成 skill，
   **未来的 agent 自动继承这个方法论**而不需要重新发现。
3. **可组合。** `suspicion-scoring` 调用 `server-log-mining` +
   `noise-aware-scoring` + `failure-classification`。每个都能单独测
   试。如果我们改进了 `server-log-mining` 让它检测一个新的 sglang
   事件，下游 skill 全部立刻受益。

## 3. `SKILL.md` 的结构

每个 skill 必须严格按以下结构。front-matter 是机器可读的，让 agent
能按字段筛选。

```yaml
---
name: server-log-mining
description: 一句话——agent 看这个决定要不要用
version: 1
stage: [1, 2, 3]            # 哪些 stage 可能用（注：v0.4 重组后只有 2 个 stage，但字段保留兼容）
inputs:
  - server_log: sglang server 输出 log 的路径
outputs:
  - server_features.json    # skill 产出什么
triggers:
  - "每次 benchmark 跑完后"
  - "给一个 passed run 分配 suspicion score 之前"
depends_on: []              # 内部调用的其他 skill
---
```

然后正文，必含以下几节：

- **WHEN** —— 使用这个 skill 的具体触发条件。
- **WHY** —— 设计理由；这个 skill 防止的是什么具体失败模式？
- **HOW** —— 一步一步过程或伪码。
- **OUTPUT CONTRACT** —— skill 产出的具体 schema。下游 skill 依赖这
  个契约。
- **FAILURE MODES** —— 可能出错的情况；调用方如何检测部分产出。
- **ROADMAP** —— 已知局限和下一版要加的东西。

可选节：**EXAMPLES**、**REFERENCES**。

## 4. 命名约定

- kebab-case
- 动词-名词 或 名词-名词（`boundary-expansion`、`server-log-mining`）
- 永远不要把模型 / 硬件 / stage 写进 skill 名——skill 必须是通用的

不好：`qwen-server-log-parser`、`h200-noise-baseline`、`stage1-shrink-prompts`。
好：`server-log-mining`、`noise-aware-scoring`、`minimal-repro-shrink`。

## 5. 撰写规则

1. **一个 skill 一个概念。** 如果 `WHEN` 列出两个不相关触发条件，
   拆 skill。
2. **输出契约神圣。** 一旦 skill 发布了 schema，下游代码就依赖它。
   只能加字段，不能删除或重命名。破坏性改动时 front-matter 升
   `version:`。
3. **Skill 不能改自己声明输出之外的状态。** "我在扫 server log 时顺手
   动了 configs/best.yaml" 这种 skill 是坏的——那是副作用，属于 agent。
4. **实现住在 `impl/` 下**，可作脚本调用。可以从 `scripts/skills/<name>.py`
   做一个薄薄的 re-export。
5. **失败要响亮。** skill 返回结构化失败（`{"ok": false, "error":
   "..."}`），而不是默默产出零或假数据。同 `parse_metrics.py` 的规则
   （DESIGN §0）。

## 6. Catalog（v0.3）

| Skill | Stages | 描述 | 状态 |
|---|---|---|---|
| [`server-log-mining`](.github/skills/server-log-mining/SKILL.md) | 1, 2, 3 | 把 sglang `server.log` 解析成结构化 features（cuda graph bs captured, max_running_requests, KV 压力事件, retract 事件, token usage peak）。 | ✅ v1 已实现 |
| [`boundary-expansion`](.github/skills/boundary-expansion/SKILL.md) | 1 | 给一个 workload + 一个假设的轴，沿该轴生成 N 个邻居 workload yaml。 | ✅ v1 已实现 |
| [`noise-aware-scoring`](.github/skills/noise-aware-scoring/SKILL.md) | 1, 3 | 算每 metric 的 coefficient of variation 阈值，避免噪声触发假阳。 | ✅ v1 已实现 |
| [`failure-classification`](.github/skills/failure-classification/SKILL.md) | 1, 2, 3 | 把失败 run 分类到一个 enum：oom / server_crash / benchmark_timeout / kv_pressure / partial_success / parse_error / unknown。 | ✅ v1 已实现 |
| [`suspicion-scoring`](.github/skills/suspicion-scoring/SKILL.md) | 1 | 把 server-log-mining + noise-aware-scoring + failure-classification + local-nonlinearity 组合成每 workload run 一个 suspicion score，带完整 evidence audit trail。 | ✅ v1 已实现 |
| [`minimal-repro-shrink`](.github/skills/minimal-repro-shrink/SKILL.md) | 1 (late), 2 | 按 (num_prompts, max_concurrency, input_len, output_len) 二分缩小 workload 直到症状消失。 | 🟨 仅 SKILL.md；impl 推迟到 v0.4 |

## 7. 怎么加一个新 skill

```bash
SKILL=my-new-skill
mkdir -p .github/skills/$SKILL/impl
cp .github/skills/_template/SKILL.md .github/skills/$SKILL/SKILL.md
$EDITOR .github/skills/$SKILL/SKILL.md       # 填 WHEN / WHY / HOW / OUTPUT / FAILURES
# 在 .github/skills/$SKILL/impl/*.py 实现
# 上面 §6 catalog 加一行
```

## 8. 这套体系从哪来

这些 skill 背后的方法论是从 2026-05-28 第一次真实 Stage 1 跑中提炼
出来的（见 `experiments/regimes/STAGE1_REPORT_20260528.md`）。那次跑
出了 10 个 workload metric 但 scoring 函数没标出 **`max-running-requests=32`
bug**，而那个 bug 明明在 `server.log` 里看得清清楚楚。教训——*"最有用
的信号都在没人解析的地方"*——直接催生了 `server-log-mining`。教训
——*"我们有 10 个孤立点 0 个邻居，所以异常检测不可能"*——直接催生了
`boundary-expansion`。教训——*"9/10 的 workload 都因为 tail-ratio 阈值
hardcoded 在 3 而饱和"*——直接催生了 `noise-aware-scoring`。

换句话说：**这个 catalog 里每一个 skill 都对应 v0.2 pipeline 的一个
具体、有名字的失败**，我们要永不重蹈。新 skill 的提案也应该按这个标
准。
