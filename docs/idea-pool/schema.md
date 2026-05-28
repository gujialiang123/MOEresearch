# Idea Pool — Schema

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#-中文版)
>
> A bidirectional channel between the Problem-Setter and the
> Problem-Solver Fleet. Both sides write observations they noticed but
> didn't (or couldn't) act on; the other side reads them as candidate
> material for the next round.
>
> Schema version: **1** (additive only until bumped).

---

## Why this exists

- The Setter often discovers **adjacent** regimes while exploring (e.g.
  while sweeping `max_concurrency`, it sees suspicious behavior on the
  prefix-cache axis it didn't intend to investigate this round). Today
  that observation evaporates unless it shows up in the final regime
  map.
- The Solver, while verifying a fix, sees **side effects** that may
  reveal new regimes (e.g. when raising `max-running-requests`, a
  control workload's `peak_token_usage` jumps unexpectedly). Today that
  observation evaporates unless someone takes notes.

The idea pool is the simplest possible note-taking system both can write
to, both can read from, with enough structure that LLM and human
reviewers can find the relevant entries.

---

## Directory layout

```
experiments/ideas/
├── README.md          # short usage guide
├── INDEX.md           # auto-aggregated catalog
├── from_setter/
│   ├── idea_001.json
│   └── ...
└── from_solver/
    ├── idea_001.json
    └── ...
```

`idea_NNN.json` files are numbered per side. `INDEX.md`
cross-references them by their unique `idea_id` (e.g. `S-001` for solver,
`R-001` for setter/recon).

---

## Single-idea schema

`kind` field distinguishes idea types:

- `kind: "observation"` (default) — solver or setter spotted a large-gap signal worth investigating.
- `kind: "proposed_workload"` — solver proposes a new workload should become its own problem; carries a sketch.
- `kind: "rejected_problem_feedback"` — solver believes the current problem is invalid; explanation here (solver also writes `rejection.md` in the problem dir).

```jsonc
{
  "schema_version": 1,
  "idea_id":        "S-007",
  "kind":           "observation",       // or "proposed_workload" or "rejected_problem_feedback"
  "created_at":     "2026-05-28 16:00:00",
  "origin": {
    "side":         "solver",
    "agent":        "config-agent",
    "session_id":   "explore_20260528_023213",
    "problem_id":   "P001",
    "attempt_id":   "attempt_003",
    "stage1_run_dir": null
  },

  "observation": "Raising max-running-requests 32→64 on P001 also bumped peak_token_usage on controls.smoke from 0.00 → 0.18. Suspect cuda_graph capture cost growing with max bs.",
  "evidence_files": [
    "experiments/problems/P001/attempts/attempt_003/verification/controls/smoke_server_features.json"
  ],
  "key_metrics_observed": [
    {"metric": "peak_token_usage", "where": "controls.smoke", "value": 0.18, "expected": "~0.00", "gap_pct": "+infinity"}
  ],

  "hypothesis": "Larger captured CUDA-graph batches reserve more KV slots even for short-batch decode, raising idle KV cost.",
  "alternative_explanations": [
    "noise — only one repeat",
    "controls were scheduled with warmer cache this run"
  ],

  "proposed_workload": null,   // for kind=proposed_workload: an embedded workload yaml dict

  "suggested_investigation": "Add regime: input_len=128, max_concurrency=4, max_running_requests∈{32,64,128,256}; track peak_token_usage and cuda_graph_capture_seconds.",
  "suggested_action_for_setter": "Promote to new seed: short_kv_growth_under_admission_cap",

  "priority": "medium",          // low | medium | high | urgent
  "status":   "open",            // open | accepted | rejected | promoted_to_problem | duplicate_of
  "promoted_to_problem_id": null,
  "duplicate_of_idea_id":   null,
  "reviewed_by":            null,
  "reviewed_at":            null
}
```

---

## Lifecycle

```
status=open
   │
   ├─ Setter picks it up next session → reads, considers
   │                                          │
   │                                  ┌───────┴────────┐
   │                                  ▼                ▼
   │                            status=accepted   status=rejected
   │                            (added as seed)   (reason logged)
   │                                  │
   │                                  ▼
   │                            Setter ships problem P012 inspired by it
   │                                  │
   │                                  ▼
   │                            status=promoted_to_problem
   │                            promoted_to_problem_id="P012"
   │
   └─ Found duplicate of earlier one
            │
            ▼
      status=duplicate_of
      duplicate_of_idea_id="S-003"
```

Once an idea reaches a terminal state, **do not delete it**. Keep for
traceability.

---

## Authoring rules

### For the Solver

- **Write an idea only when the observation is a LARGE-GAP finding.**
  Routine metric jitter, expected side effects, or noise are NOT ideas.
  Only write when:
  - A control workload regresses unexpectedly (≥ 20%).
  - A neighbor improves drastically (≥ 30%), suggesting a different
    underlying issue.
  - A profile trace surfaces a brand-new bottleneck.
  - A finding that itself warrants its own problem package.
- **You may also propose new workloads** by writing an idea with
  `kind: "proposed_workload"`. Setter considers them **additively**
  (never modifying existing workloads).
- **Do NOT** suggest changing the **current** problem's workload — that
  is benchmark gaming. If the workload is wrong, write a `rejection.md`
  inside the problem package.
- Always cite at least one `evidence_files` path and one
  `key_metrics_observed` entry. Vibe-only ideas get rejected by review.

### For the Setter

- **Stay focused.** One Setter session = one issue. If you spot an
  adjacent regime while exploring, **do not chase it this session**.
  Write a `from_setter/idea_*.json` entry so it survives to the next
  session.
- Use the pool when an extension proposal is too speculative to land
  directly in `seed_suite.yaml`.

### For reviewers

- Set `reviewed_by` + `reviewed_at` when you triage.
- Reject with concrete reason in the index.

---

## Tools (planned)

- `scripts/idea_pool/new_idea.py --side solver --origin ... --observation ...`
- `scripts/idea_pool/update_idea_index.py`
- `scripts/idea_pool/promote.py --idea-id S-007 --to-problem-id P012`

None required for v0.4. Edit JSON by hand until they exist.

---

## How the Setter consumes the pool

At Phase 0 of each Setter session:

1. Read every `from_solver/idea_*.json` and `from_setter/idea_*.json`
   with `status == "open"`.
2. Sort by `priority` then `created_at`.
3. For each high-priority idea: include as new seed (with proposal),
   accept-for-later, or reject (with reason).
4. Write decisions back into the idea files.

## How the Solver writes to the pool

Anytime during fix verification, if an unexpected signal appears, write
an entry. Always check the `server_features.json` of all
`benchmark_suite.controls` — controls should be flat; anything moving is
information.

---
---

# 🇨🇳 中文版

> 出题人和做题人 fleet 之间的双向通道。两边都可以写下"注意到但没追"
> 的观察；另一边在下一轮把它当下一波 regime / fix 的素材读取。
>
> Schema 版本：**1**（破坏性改动时升版本号；之前只允许追加字段）。

---

## 为什么要这个池

- 出题人在探索时经常发现**邻近的** regime（例如在扫
  `max_concurrency` 时顺便注意到 prefix cache 轴上的可疑行为，但本轮
  没有预算追）。今天这个观察会蒸发，除非碰巧出现在最终 regime map 里。
- 做题人在验证修复时会看到**副作用**——可能揭示新的 regime（例如把
  `max-running-requests` 调高后，一个 control workload 的
  `peak_token_usage` 意外升高）。今天这个观察也会蒸发，除非有人随手
  记下来。

idea 池是最简单的"两边都能写、两边都能读"的笔记系统，结构刚够让 LLM
和人 reviewer 找到相关条目。

---

## 目录结构

```
experiments/ideas/
├── README.md          # 简短的使用指南
├── INDEX.md           # 自动汇总目录（由 `update_idea_index.py` 重新生成）
├── from_setter/
│   ├── idea_001.json
│   └── ...
└── from_solver/
    ├── idea_001.json
    └── ...
```

`idea_NNN.json` 文件按写入方各自编号。`INDEX.md` 用唯一 `idea_id`
（例如做题人的 `S-001`、出题人/recon 的 `R-001`）交叉引用。

---

## 单条 idea schema

参考英文版上面 jsonc 例子。要点：

- `idea_id`：`S-NNN` from solver, `R-NNN` from setter（recon）
- `origin`：哪边写的、agent 名字、session、关联到哪个 problem / attempt
- `observation`：自然语言描述（关键洞察）
- `evidence_files`：至少一个具体文件路径
- `key_metrics_observed`：至少一条 metric 观察
- `hypothesis` + `alternative_explanations`：自己提的假设 + 替代解释
- `suggested_investigation` + `suggested_action_for_setter`：给出题人
  的建议
- `priority`：low / medium / high / urgent
- `status`：open / accepted / rejected / promoted_to_problem / duplicate_of
- 终态字段：`promoted_to_problem_id`, `duplicate_of_idea_id`,
  `reviewed_by`, `reviewed_at`

---

## 生命周期

```
status=open
   │
   ├─ 出题人下次 session 拾起 → 读、考虑
   │                                       │
   │                              ┌────────┴────────┐
   │                              ▼                 ▼
   │                        status=accepted   status=rejected
   │                        （加成 seed）     （在 INDEX 记原因）
   │                              │
   │                              ▼
   │                        出题人跑 explore + 发布 problem P012
   │                              │
   │                              ▼
   │                        status=promoted_to_problem
   │                        promoted_to_problem_id="P012"
   │
   └─ 发现是早先某条的重复
            │
            ▼
      status=duplicate_of
      duplicate_of_idea_id="S-003"
```

到终态（`rejected` / `promoted_to_problem` / `duplicate_of`）的 idea
**不要删**。保留以追溯。`INDEX.md` 按 status 筛选方便 review。

---

## 撰写规则

### 做题人

- **只有大 gap 发现才写 idea。** 例行 metric 抖动、预期副作用、噪声
  都**不算** idea。只在以下情况写：
  - control workload 意外回归（例如 ≥ 20%）。
  - neighbor 大幅改善（例如 ≥ 30%），暗示是另一个底层 issue。
  - profile trace 揭示了完全没人提过的新瓶颈。
  - 一个本身值得打包成新 problem 的发现。
- **可以提议新 workload**：写一个 `kind: "proposed_workload"` 的 idea。
  出题人会**增量**考虑（绝不修改已有 workload）。用这个表达"这个
  regime 应该存在为一个 problem 但还没被打包"。
- **不**许通过 idea 池建议改**当前**题目的 workload——那是作弊。如果
  workload 看起来不对，在题目包里写 `rejection.md`。
- 总是引用至少一个 `evidence_files` 路径和一条 `key_metrics_observed`。
  纯靠 vibe 的 idea 会被 review 时拒掉。

### 出题人

- **保持专注。** 一次出题人 session = 一个 issue。如果探索时发现相邻
  regime，**本轮不要追**。写一个 `from_setter/idea_*.json` 让它跨
  session 存活。
- 当一个 extension proposal 太投机不能直接落 `seed_suite.yaml` 时也用
  idea 池。

### 任何 reviewer（人 + 未来的 LLM）

- triage 时设置 `reviewed_by` 和 `reviewed_at`。
- 拒绝时在 index 里写具体原因。

---

## 工具（待实现）

- `scripts/idea_pool/new_idea.py --side solver --origin ... --observation ...`
- `scripts/idea_pool/update_idea_index.py`
- `scripts/idea_pool/promote.py --idea-id S-007 --to-problem-id P012`

v0.4 之前不需要。手写 JSON、手算 INDEX 也可以。

---

## 出题人怎么消费这个池

每次出题人 session 开始时（PLAYBOOK.md 的 Phase 0）：

1. 读所有 `from_solver/idea_*.json` 和 `from_setter/idea_*.json` 里
   `status == "open"` 的。
2. 按 `priority`（urgent → low）然后 `created_at`（新的先）排序。
3. 对每个高优先级 idea：
   - 要么本轮纳入作新 seed regime（写一份 extension proposal 到
     `stages/problem-setter/proposals/...`，proposal 里引用 idea 的
     `idea_id`）；
   - 要么标 `accepted` 加备注"下一 session 追"；
   - 要么标 `rejected` 加原因。
4. 把决定写回 idea 文件（status, reviewed_by, reviewed_at）。

---

## 做题人怎么往池里写

修复 attempt 的 verification 过程中，发现意外信号就追加。具体来说：
**总是**检查所有 `benchmark_suite.controls` 的 `server_features.json`
里有没有不在预期内的东西——controls 应该是平的，任何变化都是信息。
