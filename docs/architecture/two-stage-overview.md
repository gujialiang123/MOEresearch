# Two-Stage Architecture

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#-中文版)
>
> The project has **two agent stages**, modeled on algorithm-competition
> problem setting and problem solving. The previous three-stage design
> (Scout + Diagnose + Fix) is superseded by this document. Historical
> docs are in [`archive/`](../../archive/).

---

## The competition metaphor

```
┌──────────────────────────────────────────────────────────────────────┐
│ Stage A — Problem Setter  (出题人 Agent)                              │
│                                                                      │
│   Job: given a model + hardware + sglang, search for serving regimes  │
│        that expose performance cliffs. PROVE they're real cliffs by   │
│        layered evidence collection. Package each into a self-contained │
│        "problem" with a benchmark suite for downstream verification.  │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │  experiments/problems/PNNN/
                                  │  (one frozen problem package per cliff)
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Stage B — Problem Solver Fleet  (做题人 Agent 群)                     │
│                                                                      │
│   Job: read a problem package, propose and validate fixes. Dispatches │
│        to the right specialist sub-agent (config / scheduler /        │
│        kernel / workload-shape). Verification driven by the           │
│        acceptance_criteria the setter declared.                       │
└──────────────────────────────────────────────────────────────────────┘
```

The split is intentionally asymmetric:

- **Setter is one agent** with rich tools (lightweight probes + profilers + boundary expansion). Cannot fix anything; must produce proof.
- **Solver is a fleet** because the right fix for a config-bound regime is different from a kernel-bound regime. Each specialist has narrow surface area + harsh guardrails.

---

## Why the merger from 3 stages

The previous design split discovery (Stage 1) from diagnosis (Stage 2).
That was wrong. **Diagnosis informs discovery**: only after profiling do
you know whether to expand `max_concurrency`, `input_len`, or look at a
completely different axis. Cutting them apart broke the feedback loop.

The merged setter can interleave:

```
seed bench → log mine → score → triage  ┐
                                        ├──→  if signal is strong, profile (L4)
expand axis A → bench → log mine → ...  ┤      then decide next axis
                                        ┘
```

`auto-gpu-kernel`'s template uses a single optimizer agent that both
understands and modifies the kernel. The merger brings us closer to that
proven shape.

---

## Layered evidence collection

A problem cannot ship without enough evidence to be falsifiable by the
solver. The setter collects evidence in layers, cheapest first:

| Layer | What | Tool | Cost per workload | Runs on |
|---|---|---|---|---|
| **L1** | Bench-level metrics | `parse_metrics.py` | 0 (already in bench) | Every workload |
| **L2** | Server-log mining | `server-log-mining` skill | < 1 s | Every workload |
| **L3** | Failure classification | `failure-classification` skill | < 1 s | Every workload |
| **L4** | Deep profile (torch profiler / nsys summary) | `pytorch-profiling` skill (planned) | 10-30 s | **Only when L1-L3 signal is strong**, before packaging |
| **L5** | Bias-removal: N repeats + paired comparisons | `noise-aware-scoring` skill + bench reruns | several minutes | **Only at packaging time**, on the target + ≥1 neighbor |

L4 is gated: the setter only spends profile cost when it has a
hypothesis to test. L5 is the final "I am confident this is real" step
before freezing the problem.

---

## The Problem Package

A problem package is a self-contained, frozen, JSON-typed proof that a
real cliff exists, together with everything a solver needs to verify
its fix.

```
experiments/problems/P001/
├── problem.json              # The umbrella contract (schema-versioned)
├── workload.yaml             # The target workload (the cliff itself)
├── baseline_metrics.json     # L1: bench result for the target
├── server_features.json      # L2: log-mining for the target
├── classification.json       # L3: classifier output for the target
├── profile_summary.json      # L4: deep profile of the target (optional but recommended)
├── repeats/                  # L5: N=3-5 repeats of the target for bias removal
│   ├── rep_01_metrics.json
│   └── ...
├── neighbors/                # The boundary-expansion runs that bracketed the cliff
│   ├── mc_16.yaml
│   ├── mc_16_metrics.json
│   ├── mc_32.yaml
│   └── mc_32_metrics.json
├── controls/                 # Unrelated workloads, used as negative controls
│   ├── smoke.yaml
│   └── smoke_metrics.json
├── hypothesis.md             # Setter's reasoning ("why this cliff exists")
└── acceptance_criteria.json  # What the solver must achieve to claim victory
```

**Why neighbors + controls are mandatory**: the solver needs a way to:

1. **Prove the fix moved the cliff** — re-run the target, check primary metric improved per `acceptance_criteria.primary`.
2. **Prove the fix didn't break neighbors** — re-run neighbors, check they didn't regress (they may even improve, which is bonus evidence).
3. **Prove the fix didn't break unrelated workloads** — re-run controls, check `acceptance_criteria.constraints` (no regression beyond X%).

A solver that only had the target workload could "fix" it by tuning a
global parameter that secretly destroys 95% of workloads. The benchmark
suite makes that gaming impossible.

Schema details: [`docs/problem-package/schema.md`](../problem-package/schema.md).

---

## What the Setter is NOT allowed to do

- Choose a fix. Mention candidate strategies in
  `problem.json.suggested_strategies` (route hints, expected magnitude,
  rationale) — but never apply them.
- Modify `configs/base.yaml` after the first benchmark of a session.
- Skip L1-L3 mining on any run.
- **Diverge mid-session.** One session = one issue. If the Setter
  discovers a large-gap finding unrelated to the current focus, it
  writes an idea to `experiments/ideas/from_setter/` and continues with
  the original target. Avoiding over-exploration is critical for
  Setter quality.
- Modify `sglang/python/sglang/srt/*.py`. Source-level changes are
  Solver-Fleet kernel-agent territory.

## What the Solver Fleet is NOT allowed to do

The Solver–Setter relationship is **collaborative, not adversarial**.
The Solver may read anything in the problem package, anything in
neighbouring problem packages, anything in `experiments/ideas/`, and any
historical metric data. There is no information firewall.

Integrity rules (enforced by **convention + reproducibility**, no
runtime hash check — see [`docs/problem-package/schema.md` §"Anti-cheating: how it actually works"](../problem-package/schema.md)):

- **No benchmark gaming via mutation.** The package root files
  (`workload.yaml`, `neighbors/*.yaml`, `controls/*.yaml`, etc.) are
  read-only by convention; the solver harness only writes to
  `attempts/attempt_NNN/`. If a solver attempts to modify the
  benchmark inputs to look better, the cheat is auto-detectable: anyone
  can re-run the solution's config on the unmodified package yamls and
  see if the gain reproduces.
- **One change per attempt** (per-sub-agent rule; details in each
  sub-agent's `AGENT_CONTRACT.md`).
- **No source modifications outside the protected scope declared by
  each sub-agent** (e.g. `kernel-agent` may touch
  `sglang/python/sglang/srt/layers/attention/*.py` but not
  `scheduler/*`).
- **Always include `also_solved[]` in `decision.json`** when an attempt
  incidentally fixes a control or a neighbor — this is how
  side-discovered fixes get credit and surface to the human reviewer.

What the Solver may freely do:

- Propose new workloads via the idea pool (`type: proposed_workload`).
  The Setter considers them in a future session.
- Report that the problem is unreasonable: write `rejection.md` inside
  the package and stop. (Better than silent failure.)
- Document side-solved problems in `decision.json.also_solved[]` and in
  the final `solution.md`. The setter encourages this — finding bonus
  fixes is valuable.

## Idea pool — the bidirectional channel

```
experiments/ideas/
├── README.md                    # what this is, how to use it
├── INDEX.md                     # auto-aggregated catalog (status: open/accepted/promoted)
├── from_solver/                 # Solver writes here when it spots a LARGE-gap finding
│   ├── idea_001.json
│   └── ...
└── from_setter/                 # Setter writes here for regimes it noticed but didn't pursue
    ├── idea_001.json
    └── ...
```

**When Solver writes** (be selective): only when the observation is a
**large-gap finding** that points at a likely new problem. Routine
metric jitter or expected side effects are NOT ideas — they're noise.
Examples that DO warrant an idea:

- A control workload regressed by 30% under the proposed fix (large gap).
- A neighbor unexpectedly improved by 50% (might be a different problem).
- A profile trace shows an entirely new bottleneck nobody had mentioned.

**Setter writes** for adjacent regimes it noticed mid-exploration but
correctly chose not to chase — to honor the "one issue per session"
discipline.

**Setter consumes** at the start of each new session (Phase 0). Open
ideas with priority ≥ medium become candidate seeds.

**Promotion**: once an idea is investigated and packaged as a problem,
mark the idea `status: "promoted_to_problem"` with
`promoted_to_problem_id: "P012"`. Don't delete the idea.

Schema details: [`docs/idea-pool/schema.md`](../idea-pool/schema.md).

---

## Data flow

```
USER (one-time setup)
  └─ configs/base.yaml  (model + hardware)
  └─ regime_scout/seed_suite.yaml
                      │
                      ▼
        ┌─────────────────────────────────┐
        │ stages/problem-setter/          │ ───── reads ideas at Phase 0:
        │   policies/                     │       experiments/ideas/{from_setter,from_solver}/
        │     rule_based_explore.py       │ ───── uses tools (deterministic, JSON-typed):
        │     llm_agent.md                │       scripts/run_experiment.py
        │                                 │       .github/skills/*/impl/*.py
        │   discipline: ONE issue / session│
        │   large-gap side findings →     │
        │       experiments/ideas/from_setter/
        └─────────────────────────────────┘
                      │
                      ▼
        experiments/problems/PNNN/         (frozen problem packages)
          ├── workload.yaml, evidence files, hypothesis, acceptance
          ├── attempts/                    ←─── solver writes here
          │   ├── attempt_001/{plan, candidate, verification, decision}
          │   └── ...
          ├── solution.md                  ←─── final accepted report (bilingual)
          └── rejection.md                 ←─── if no solution / invalid problem
                      │
                      ▼ each problem package = one complete experiment record
        ┌─────────────────────────────────┐
        │ stages/problem-solver/          │ ───── reads anything (problem package +
        │   policies/dispatch.md          │       other problems + ideas + history)
        │   config-agent/                 │ ───── writes attempts INSIDE the problem dir
        │   scheduler-agent/              │       + new ideas to from_solver/
        │   kernel-agent/                 │ ───── side-solved problems noted in
        │   workload-shape-agent/         │       decision.json.also_solved[]
        └─────────────────────────────────┘
                      │
                      ▼
        experiments/ideas/   ←──────────────── feedback channel both directions
        ├── from_solver/idea_*.json    (LARGE-GAP findings during fixes)
        ├── from_setter/idea_*.json    (adjacent regimes the setter saw but didn't chase)
        └── INDEX.md
```

Integrity is enforced by **convention + reproducibility**, not by hash
checks. The package's frozen files are read-only by convention; any
cheat (e.g. tweaking a workload yaml during an attempt) is
auto-detectable because anyone can re-run the solution against the
unmodified package and see if the gain holds.

---

## How an LLM enters each stage

Same pattern as before: LLM is the policy that decides "what to do
next"; deterministic harness does the work.

| Stage | LLM does | Tools do |
|---|---|---|
| Setter | (a) pick which seeds to start with, (b) read scoring output and decide which workload to expand and along which axis, (c) decide when an L4 profile is warranted, (d) write hypothesis prose, (e) decide acceptance criteria magnitudes | run bench, parse metrics, mine logs, classify, expand, profile, repeat |
| Solver (config) | (a) read problem + diagnosis, (b) choose one knob to try this attempt, (c) write hypothesis prose | edit candidate config, run benchmarks across target+neighbors+controls, A/B, apply keep/revert rule |
| Solver (kernel) | (a) read problem, (b) propose a kernel rewrite, (c) iterate | enforce protected-file boundaries, run benchmarks, A/B, accept/revert |

**Both stages can be driven by a rule-based reference policy** (cheap CI
baseline) **or** by an LLM session. The interface — JSON contracts in
and JSON contracts out — is the same.

---

## Where to read next

- Problem package schema: [`docs/problem-package/schema.md`](../problem-package/schema.md)
- Idea pool schema (bidirectional channel): [`docs/idea-pool/schema.md`](../idea-pool/schema.md)
- Setter contract: [`stages/problem-setter/AGENT_CONTRACT.md`](../../stages/problem-setter/AGENT_CONTRACT.md)
- Setter workflow: [`stages/problem-setter/PLAYBOOK.md`](../../stages/problem-setter/PLAYBOOK.md)
- Solver fleet (skeleton): [`stages/problem-solver/README.md`](../../stages/problem-solver/README.md)
- Why rules vs LLM: [`docs/architecture/agent-vs-harness.md`](./agent-vs-harness.md)
- Auto-GPU-Kernel reference: `/home/t-jialianggu/work/auto-gpu-kernel/template/CLAUDE.md`

---
---

# 🇨🇳 中文版

> 项目使用**两阶段 agent 架构**，借鉴算法竞赛的"出题人 / 做题人"分工。
> 之前的三阶段设计（Scout + Diagnose + Fix）已被本文档取代，历史文档
> 在 [`archive/`](../../archive/)。

---

## 算法竞赛比喻

```
┌──────────────────────────────────────────────────────────────────────┐
│ 阶段 A — Problem Setter（出题人 Agent）                              │
│                                                                      │
│   工作：给定 model + 硬件 + sglang，搜索暴露性能悬崖的 serving       │
│         regime。通过**分层证据收集**证明悬崖真实存在。把每个悬崖     │
│         打包成自包含的"题目"，含一整套用于下游验证的 benchmark 套件。│
└─────────────────────────────────┬────────────────────────────────────┘
                                  │ experiments/problems/PNNN/
                                  │ （每个悬崖一个冻结的题目包）
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 阶段 B — Problem Solver Fleet（做题人 Agent 群）                     │
│                                                                      │
│   工作：读题目包，提出并验证修复方案。按 suggested_strategies 路由   │
│         到正确的专家子 agent（config / scheduler / kernel /          │
│         workload-shape）。验收标准来自出题人声明的                   │
│         acceptance_criteria。                                        │
└──────────────────────────────────────────────────────────────────────┘
```

切分是有意不对称的：

- **出题人是单个 agent**，工具丰富（轻量探针 + profiler + boundary
  expansion）。不能修任何东西，**必须**产生可证伪的证据。
- **做题人是 fleet**，因为 config-bound regime 的修法跟 kernel-bound
  regime 的修法完全不同。每个专家职责窄、约束严。

---

## 为什么从三阶段合并到两阶段

之前的设计把"发现"（Stage 1）和"诊断"（Stage 2）切开。那是错的。
**诊断本身决定发现的方向**：只有 profile 之后才知道该扩展
`max_concurrency`、`input_len` 还是某个完全不同的轴。切开两阶段就破坏
了反馈环。

合并后的出题人可以自由交错：

```
seed bench → log mine → score → triage  ┐
                                        ├──→  若信号强，跑 profile（L4）
expand axis A → bench → log mine → ...  ┤      然后决定下一个轴
                                        ┘
```

`auto-gpu-kernel` 用的就是单 optimizer agent，同时承担"理解 kernel"和
"修 kernel"。合并让我们更接近他们已验证的模式。

---

## 分层证据收集

题目不带充分证据是不能打包的，否则做题人没法对它做证伪。出题人按
**成本由低到高**逐层收证据：

| 层 | 是什么 | 工具 | 单 workload 成本 | 何时跑 |
|---|---|---|---|---|
| **L1** | bench 级 metric | `parse_metrics.py` | 0（bench 自带） | 每个 workload |
| **L2** | server log 信号挖掘 | `server-log-mining` skill | < 1 秒 | 每个 workload |
| **L3** | 失败分类 | `failure-classification` skill | < 1 秒 | 每个 workload |
| **L4** | 深度 profile（torch profiler / nsys 摘要） | `pytorch-profiling` skill（待实现） | 10-30 秒 | **L1-L3 信号够强**时，打包前 |
| **L5** | bias 消除：N 次重复 + 配对比较 | `noise-aware-scoring` skill + bench 重跑 | 数分钟 | **打包时**，target + ≥1 个 neighbor |

L4 有阈门控：出题人只在有假设需要验证时才付 profile 成本。L5 是冻结
题目前的"我确认这是真信号"最终步。

---

## 题目包（Problem Package）

题目包是自包含、冻结、强类型 JSON 的证据——证明真实悬崖存在——加上
做题人验证修复方案需要的一切。

```
experiments/problems/P001/
├── problem.json              # 总契约（带 schema 版本）
├── workload.yaml             # target workload（悬崖本身）
├── baseline_metrics.json     # L1：target 的 bench 结果
├── server_features.json      # L2：target 的 log mining 结果
├── classification.json       # L3：target 的分类
├── profile_summary.json      # L4：target 的深度 profile（可选但推荐）
├── repeats/                  # L5：target 的 N=3-5 次重复，用于 bias 消除
│   ├── rep_01_metrics.json
│   └── ...
├── neighbors/                # 边界扩展产的邻居，bracket 了悬崖
│   ├── mc_16.yaml
│   ├── mc_16_metrics.json
│   ├── mc_32.yaml
│   └── mc_32_metrics.json
├── controls/                 # 无关 workload，做 negative control
│   ├── smoke.yaml
│   └── smoke_metrics.json
├── hypothesis.md             # 出题人的推理（"为什么有这个悬崖"）
└── acceptance_criteria.json  # 做题人要达到什么算解出来
```

**为什么强制要 neighbors + controls**：做题人需要：

1. **证明修复推动了悬崖**——重跑 target，按 `acceptance_criteria.primary`
   检查 primary metric 是否改善。
2. **证明修复没坏邻居**——重跑 neighbors，检查没回归（甚至可能也变
   好，那是额外证据）。
3. **证明修复没破坏无关 workload**——重跑 controls，按
   `acceptance_criteria.constraints` 检查没回归超过阈值。

只有 target workload 的做题人可以靠"调一个偷偷搞砸 95% workload 的
全局参数"来"修好"target。benchmark 套件让这种作弊不可能。

Schema 细节：[`docs/problem-package/schema.md`](../problem-package/schema.md)。

---

## 出题人**不可以**做的事

- 选择修复方案。在 `problem.json.suggested_strategies` 里提示候选策略
  （路由提示、预期幅度、理由）——但不许实施。
- session 第一次 benchmark 之后修改 `configs/base.yaml`。
- 在任何 run 上跳过 L1-L3 mining。
- **本轮发散到不相关的 issue。** 一次 session = 一个 issue。如果出题人
  在探索时碰到了一个大 gap 但跟当前焦点无关的发现，就写一个 idea 到
  `experiments/ideas/from_setter/`，然后**继续追原本的 target**。"不发
  散"对出题人质量至关重要。
- 修改 `sglang/python/sglang/srt/*.py`。源码级改动归做题人 fleet 的
  kernel-agent 管。

## 做题人 fleet **不可以**做的事

出题人和做题人是**协作关系，不是对抗关系**。做题人可以读题目包里的
任何东西、相邻题目包的任何东西、`experiments/ideas/` 里的任何东西、
任何历史 metric 数据。**没有信息防火墙**。

完整性规则（靠**约定 + 可复现性**强制，**不**靠运行时 hash 校验——见
[`docs/problem-package/schema.md` §"反作弊：实际怎么生效"](../problem-package/schema.md)）：

- **不许通过修改 benchmark 输入作弊。** 题目包根目录文件
  (`workload.yaml`、`neighbors/*.yaml`、`controls/*.yaml` 等) 约定为
  只读；做题人 harness 只往 `attempts/attempt_NNN/` 写。如果做题人
  偷偷改 benchmark 输入让数字变好看，作弊**事后可自动检测**：任何人
  把 solution 的配置在未修改的 yaml 上重跑，看收益还在不在。
- **每次 attempt 一次改动**（每个子 agent 自己的规则）。
- **每个子 agent 自己声明的"受保护范围"之外不许改源码**（例如
  `kernel-agent` 可以改 `sglang/python/sglang/srt/layers/attention/*.py`
  但不能改 `scheduler/*`）。
- **`decision.json` 里始终带 `also_solved[]`**：如果一次 attempt 顺带
  修复了某个 control 或 neighbor，记到这个字段——这是旁征修复在 reviewer
  面前显形的方式。

做题人**可以自由做的事**：

- 通过 idea 池提议新 workload（`type: proposed_workload`）。出题人下
  一次 session 会考虑。
- **报告题目不合理**：在题目包里写 `rejection.md` 然后停。比沉默失败
  好。
- 在 `decision.json.also_solved[]` 和最终 `solution.md` 里记**旁征解
  决的问题**。出题人鼓励这样做——发现额外 fix 是有价值的。

## Idea 池——双向通道

```
experiments/ideas/
├── README.md                    # 这是什么、怎么用
├── INDEX.md                     # 自动汇总目录
├── from_solver/                 # 做题人**仅在大 gap 发现**时写在这里
│   ├── idea_001.json
│   └── ...
└── from_setter/                 # 出题人记下"注意到但本轮没追"的 regime
    ├── idea_001.json
    └── ...
```

**做题人何时写**（要节制）：只在观察到**大 gap 发现**时——一个指向
可能的新 problem 的信号。例行的 metric 抖动或预期的副作用**不算**
idea，那是噪声。值得写 idea 的例子：

- 一个 control workload 在提议的修复下回归 30%（大 gap）。
- 一个 neighbor 意外提升了 50%（可能是另一个不同的 problem）。
- profile trace 显示一个完全没人提过的新瓶颈。

**出题人写** —— 探索中注意到的相邻 regime 但**对**地决定不追，为了
守住"一次 session 一个 issue"的纪律。

**出题人消费**（每次新 session 的 Phase 0）。priority ≥ medium 的
open ideas 成为候选 seed。

**升级**：一个 idea 被调查并打包成 problem 后，标记 idea
`status: "promoted_to_problem"` 加 `promoted_to_problem_id: "P012"`。
不要删掉 idea。

Schema 细节：[`docs/idea-pool/schema.md`](../idea-pool/schema.md)。

---

## 数据流

```
USER（一次性 setup）
  └─ configs/base.yaml  （model + 硬件）
  └─ regime_scout/seed_suite.yaml
                      │
                      ▼
        ┌─────────────────────────────────┐
        │ stages/problem-setter/          │ ───── Phase 0 读 ideas:
        │   policies/                     │       experiments/ideas/{from_setter,from_solver}/
        │     rule_based_explore.py       │ ───── 用工具（确定性，JSON 类型化）:
        │     llm_agent.md                │       scripts/run_experiment.py
        │                                 │       .github/skills/*/impl/*.py
        │   纪律: 一次 session 一个 issue │
        │   大 gap 的旁征 →               │
        │       experiments/ideas/from_setter/
        └─────────────────────────────────┘
                      │
                      ▼
        experiments/problems/PNNN/         （冻结的题目包）
          ├── workload.yaml, evidence 文件, hypothesis, acceptance
          ├── attempts/                    ←─── 做题人在这里写
          │   ├── attempt_001/{plan, candidate, verification, decision}
          │   └── ...
          ├── solution.md                  ←─── 最终接受报告（双语）
          └── rejection.md                 ←─── 无解 / 题目无效时
                      │
                      ▼ 每个题目包 = 一份完整实验记录
        ┌─────────────────────────────────┐
        │ stages/problem-solver/          │ ───── 读任何东西（题目包 +
        │   policies/dispatch.md          │       其它题目 + ideas + 历史）
        │   config-agent/                 │ ───── attempt 写在题目包**内**
        │   scheduler-agent/              │       新 ideas 写到 from_solver/
        │   kernel-agent/                 │ ───── 旁征解决的问题记到
        │   workload-shape-agent/         │       decision.json.also_solved[]
        └─────────────────────────────────┘
                      │
                      ▼
        experiments/ideas/   ←──────────────── 双向反馈通道
        ├── from_solver/idea_*.json    （修复时的**大 gap** 发现）
        ├── from_setter/idea_*.json    （出题人看到但没追的相邻 regime）
        └── INDEX.md
```

完整性靠**约定 + 可复现性**保证，不靠 hash 校验。题目包冻结文件约定
为只读；任何作弊（例如 attempt 期间偷改 workload yaml）**事后可检
测**——任何人都可以在未修改的 yaml 上重跑 solution，看收益还在不在。

---

## LLM 在每个阶段做什么

跟之前 README §11 的模式一样：LLM 做"下一步做什么"的策略决定；确定性
harness 做具体工作。

| 阶段 | LLM 做 | 工具做 |
|---|---|---|
| 出题人 | (a) 选先跑哪些 seed, (b) 读 score 输出决定扩展哪个 workload 沿哪个轴, (c) 决定何时跑 L4 profile, (d) 写 hypothesis 散文, (e) 定 acceptance criteria 幅度 | 跑 bench、parse metrics、mine logs、classify、expand、profile、repeat |
| 做题人（config） | (a) 读题目 + 诊断, (b) 选本 attempt 试哪个 knob, (c) 写假设散文 | 改 candidate config, 跑 target+neighbors+controls 的 benchmark、A/B、按 keep/revert 规则定夺 |
| 做题人（kernel） | (a) 读题目, (b) 提议 kernel 重写, (c) 迭代 | 强制受保护文件边界、跑 benchmark、A/B、接受/回滚 |

**两个阶段都可以用 rule-based 参考策略**（廉价 CI 基线）**或** LLM
session 来驱动。接口——JSON 契约进、JSON 契约出——是一样的。

---

## 接下来读哪里

- 题目包 schema：[`docs/problem-package/schema.md`](../problem-package/schema.md)
- Idea 池 schema（双向通道）：[`docs/idea-pool/schema.md`](../idea-pool/schema.md)
- 出题人契约：[`stages/problem-setter/AGENT_CONTRACT.md`](../../stages/problem-setter/AGENT_CONTRACT.md) （注：仍是三阶段叙述，下轮重写）
- 出题人剧本：[`stages/problem-setter/PLAYBOOK.md`](../../stages/problem-setter/PLAYBOOK.md) （同上）
- 做题人 fleet（骨架）：[`stages/problem-solver/README.md`](../../stages/problem-solver/README.md) （待写）
- rule vs LLM 分工：[`docs/architecture/agent-vs-harness.md`](./agent-vs-harness.md) （待写）
- Auto-GPU-Kernel 参考：`/home/t-jialianggu/work/auto-gpu-kernel/template/CLAUDE.md`
