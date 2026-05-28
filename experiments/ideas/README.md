# Idea pool

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#-中文版)

This directory is the **bidirectional notes channel** between the
Problem-Setter (出题人 agent) and the Problem-Solver Fleet (做题人 agent).

See [`docs/idea-pool/schema.md`](../../docs/idea-pool/schema.md) for the
full schema.

## Quick rules

- **Both sides write here.** Setter notes go in `from_setter/`, solver
  notes go in `from_solver/`.
- **One JSON file per idea.** Naming: `idea_NNN.json` (numbered per side).
- **Never delete ideas.** Use `status: rejected` / `status: duplicate_of`
  instead.
- **Index is regenerated, not hand-edited.** Use
  `scripts/idea_pool/update_idea_index.py` when it ships; until then,
  read the files directly.

## How an idea becomes a problem

```
solver sees something off → idea_NNN.json (status: open)
       ↓
setter next session reads pool
       ↓
setter decides this is a real regime → status: accepted
       ↓
setter runs explore + ships problem P_M
       ↓
setter sets status: promoted_to_problem, promoted_to_problem_id: P_M
```

## Anti-gaming reminder

Ideas are **suggestions**, not workload changes. The Solver may NOT use
the idea pool as a backdoor to modify benchmark inputs. That rule is
enforced by `benchmark-integrity` skill on every solver attempt.

---
---

# 🇨🇳 中文版

这个目录是出题人 agent（Problem-Setter）和做题人 agent fleet
（Problem-Solver Fleet）之间的**双向笔记通道**。

完整 schema 见 [`docs/idea-pool/schema.md`](../../docs/idea-pool/schema.md)。

## 简要规则

- **两边都往这里写。** 出题人的笔记进 `from_setter/`，做题人的笔记
  进 `from_solver/`。
- **一个 idea 一份 JSON 文件。** 命名：`idea_NNN.json`（每边各自编号）。
- **不要删 idea。** 用 `status: rejected` / `status: duplicate_of`
  代替。
- **Index 是自动生成的，不要手编。** 用
  `scripts/idea_pool/update_idea_index.py`（待实现）；在那之前直接读
  文件。

## 一个 idea 怎么变成题目

```
做题人发现奇怪现象 → idea_NNN.json (status: open)
       ↓
出题人下次 session 读 idea 池
       ↓
出题人判断这是真 regime → status: accepted
       ↓
出题人跑 explore + 发布题目 P_M
       ↓
出题人设置 status: promoted_to_problem, promoted_to_problem_id: P_M
```

## 反作弊提醒

Idea 是**建议**，不是 workload 修改。做题人**不许**把 idea 池当后门
来改 benchmark 输入。这条规则由 `benchmark-integrity` skill 在每次做
题 attempt 上强制执行。
