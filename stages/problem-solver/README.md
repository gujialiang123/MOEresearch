# Stage B — Problem-Solver Fleet / 做题人 agent 群

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#中文版)

This directory holds the **agent-facing** docs for Stage B (the "做题人"
fleet). It is intentionally light today because:

- The only solver that exists is the **config-agent**, which lives at
  [`../../scripts/solver/config_agent.py`](../../scripts/solver/config_agent.py).
  It is small enough to be self-documenting via `--help` and the
  problem-package contract.
- The other three solvers (scheduler / kernel / workload-shape) are
  designed but not yet implemented. When we build them, each will get a
  sub-directory here with the same shape as
  [`../problem-setter/`](../problem-setter/) (AGENT_CONTRACT.md,
  PLAYBOOK.md, TOOLS.md, policies/llm_agent.md).

## What Stage B does today

Reads a frozen problem package under `experiments/problems[_moe]/PNNN/`,
picks **one strategy + one or more values to try**, runs the candidate
config against `target + neighbors + controls`, evaluates against the
package's acceptance criteria, and writes:

- `attempts/attempt_NNN/decision.json` — outcome (`keep` / `revert` /
  `needs_more_evidence`) per attempt
- `solution.md` — problem-level summary ranking all attempts

See:
- [`../../scripts/solver/config_agent.py`](../../scripts/solver/config_agent.py) — code
- [`../../docs/problem-package/schema.md`](../../docs/problem-package/schema.md) — contract
- [`../../docs/architecture/two-stage-overview.md`](../../docs/architecture/two-stage-overview.md) — architecture

## Planned sub-agents

| Agent | Scope | Status |
|---|---|---|
| `config-agent` | Single-knob sglang flags (`max-running-requests`, `cuda-graph-max-bs`, KV-cache fraction, …) | **Built** |
| `scheduler-agent` | Multi-knob scheduling policy (batching, chunk-prefill, retract policy, request admission) | Not built |
| `kernel-agent` | Source-level kernel/attention changes (requires L4 profile evidence — see [`../../.github/skills/pytorch-profiling`](../../.github/skills/pytorch-profiling)) | Not built |
| `workload-shape-agent` | Tokenization, request shape, chat-template choice, batching the caller can change | Not built |

---

<a id="中文版"></a>

# 中文版

本目录存放 Stage B（"做题人"agent 群）的 **agent 视角** 文档。
当前内容很少，原因是：

- 唯一存在的 solver 是 **config-agent**，代码在
  [`../../scripts/solver/config_agent.py`](../../scripts/solver/config_agent.py)。
  它足够小，靠 `--help` 和题目包契约本身就能自解释。
- 另外三个 solver（scheduler / kernel / workload-shape）只是设计好了
  尚未实现。等我们造它们时，每个会按
  [`../problem-setter/`](../problem-setter/) 的样子开一个子目录
  （AGENT_CONTRACT.md、PLAYBOOK.md、TOOLS.md、policies/llm_agent.md）。

## Stage B 当前能做什么

读 `experiments/problems[_moe]/PNNN/` 下的冻结题目包，挑
**一个策略 + 一个或多个候选值**，把 candidate config 跑在
`target + neighbors + controls` 上，用题目包里的 acceptance criteria
判定，输出：

- `attempts/attempt_NNN/decision.json` —— 每次 attempt 的结论
  （`keep` / `revert` / `needs_more_evidence`）
- `solution.md` —— 题目级总结，把所有 attempt 排序

详见：
- [`../../scripts/solver/config_agent.py`](../../scripts/solver/config_agent.py) —— 代码
- [`../../docs/problem-package/schema.md`](../../docs/problem-package/schema.md) —— 契约
- [`../../docs/architecture/two-stage-overview.md`](../../docs/architecture/two-stage-overview.md) —— 架构

## 规划中的子 agent

| Agent | 职责范围 | 状态 |
|---|---|---|
| `config-agent` | 单旋钮 sglang 参数（`max-running-requests`、`cuda-graph-max-bs`、KV-cache fraction……） | **已实现** |
| `scheduler-agent` | 多旋钮调度策略（batching、chunk-prefill、retract、admission） | 未实现 |
| `kernel-agent` | 源码级 kernel/attention 修改（需 L4 profile 证据 —— 见 [`../../.github/skills/pytorch-profiling`](../../.github/skills/pytorch-profiling)） | 未实现 |
| `workload-shape-agent` | 调用方可改的 tokenization / 请求形状 / chat-template / batch 策略 | 未实现 |
