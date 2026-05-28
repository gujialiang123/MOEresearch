# EndtoEnd Optimization Agent for SGLang

> 🇬🇧 English first · 🇨🇳 [跳转中文版 (jump to Chinese)](#中文版)

An end-to-end agent system that, given **a model + a GPU + SGLang**,
automatically:

1. discovers **regimes** where SGLang serves poorly (Stage A — Problem-Setter), and
2. fixes those regimes via specialised solver sub-agents (Stage B — Problem-Solver).

It is built around an **algorithm-competition metaphor**: Stage A is the
problem-setter (出题人), Stage B is the problem-solver (做题人). The
hand-off between them is a frozen, self-contained **problem package** on
disk — so you can drive Stage A with one tool and Stage B with another,
or hand a package to a teammate to solve manually.

**Status (2026-05-28)**: Stage A runs end-to-end and has produced two
validated problem packages (Qwen3-0.6B P001, MoE Qwen3-30B-A3B P001).
Stage B's config-agent fixed MoE P001 with **TTFT p95 −92.6%** in one
attempt. See [`docs/reports/2026-05-28-progress.md`](./docs/reports/2026-05-28-progress.md)
for the latest progress write-up.

---

## 1. Quick start

```bash
# 0. activate the env (one-time)
conda activate sglang-dev

# 1. point the config at your model (one-time)
$EDITOR configs/base.yaml          # set model-path to /path/to/your/model

# 2. Stage A — discover problems for that model
python stages/problem-setter/policies/rule_based_explore.py \
    --config configs/base.yaml
#   → experiments/problems/PNNN/   (frozen problem packages)

# 3. Stage B — solve one of those problems with the config-agent
python scripts/solver/config_agent.py \
    --problem experiments/problems/P001 \
    --strategy S001 --exhaustive
#   → experiments/problems/P001/attempts/attempt_NNN/
#   → experiments/problems/P001/solution.md
```

Total wall-clock on a single H200 + Qwen3-0.6B: Stage A ≈ 15 min,
Stage B config-agent (3 values) ≈ 12 min.

For the full walkthrough (what each script does internally, how seeds
are generated, how triage works, how MoE config differs), read the
[developer guide](./docs/development/developer-guide.md).

---

## 2. Repository layout

```text
EndtoEnd-auto-optimization/
├── README.md                       ← you are here (short intro + quickstart)
├── configs/                        ← per-model server configs
│   ├── base.yaml                   ← default (small dense model)
│   └── moe_qwen3_30b.yaml          ← MoE example
├── regime_scout/                   ← Stage A inputs (seed regimes) + outputs
│   ├── candidates/                 ← seed suite YAMLs (Qwen3 dense)
│   ├── candidates_moe/             ← seed suite YAMLs (MoE)
│   └── outputs/                    ← regime_map.md, selected_cases.jsonl
├── stages/
│   ├── problem-setter/             ← Stage A: the "out-题人" agent
│   │   ├── README.md               ← stage overview + how to invoke
│   │   ├── AGENT_CONTRACT.md       ← what the agent MUST / MUST NOT do
│   │   ├── PLAYBOOK.md             ← step-by-step procedure
│   │   ├── TOOLS.md                ← the harness scripts the agent calls
│   │   ├── EXTENSION_GUIDE.md      ← how to add a new seed / axis / skill
│   │   └── policies/
│   │       ├── rule_based_explore.py    ← Mode A: deterministic harness
│   │       └── llm_agent.md             ← Mode B: LLM system prompt
│   └── problem-solver/             ← Stage B: the "做题人" agent fleet
│       └── README.md               ← (stub — see scripts/solver/ for code)
├── scripts/                        ← the harness (called by both stages)
│   ├── run_experiment.py           ← server + bench wrapper
│   ├── run_regime_suite.py         ← run N workloads back-to-back
│   ├── select_problems.py          ← turn raw runs → problem packages
│   ├── solver/
│   │   └── config_agent.py         ← Stage B: config-agent (--exhaustive)
│   └── utils.py                    ← env / yaml / paths helpers
├── .github/skills/                 ← reusable methodology units
│   ├── server-log-mining/          ← L2 evidence: parse server.log
│   ├── failure-classification/     ← L3 evidence: classify benchmark failures
│   ├── noise-aware-scoring/        ← v2 suspicion score
│   ├── boundary-expansion/         ← grow neighbors along one axis
│   ├── suspicion-scoring/          ← compose evidence → score
│   ├── minimal-repro-shrink/       ← (designed, not implemented)
│   └── pytorch-profiling/          ← L4 evidence: torch profile + reduce
├── experiments/
│   ├── README.md                   ← index of every experiment + how to reproduce
│   ├── problems/                   ← Stage A output (dense models)
│   ├── problems_moe/               ← Stage A output (MoE models)
│   ├── ideas/                      ← idea pool (bidirectional channel)
│   ├── regimes/                    ← raw Stage A scout outputs + first-run reports
│   └── tmp/                        ← per-run scratch (gitignored)
├── docs/                           ← all documentation
│   ├── architecture/
│   │   └── two-stage-overview.md   ← the canonical architecture doc
│   ├── problem-package/
│   │   └── schema.md               ← problem.json v1 contract
│   ├── idea-pool/
│   │   └── schema.md               ← idea.json contract
│   ├── skills/
│   │   └── README.md               ← skills design principles + catalog
│   ├── research/
│   │   └── regime-search-extensions.md  ← R1-R9 future directions
│   ├── reports/
│   │   └── 2026-05-28-progress.md  ← latest progress report
│   └── development/                ← long-form dev docs
│       ├── developer-guide.md      ← full design + mental model
│       ├── history.md              ← project-evolution timeline (v0.1→v0.4)
│       ├── log-layout.md           ← where every log goes
│       └── restructure-notes.md    ← history of the 2-stage refactor
├── archive/                        ← superseded design drafts
└── logs/                           ← orchestrator logs (gitignored payloads)
```

---

## 3. The two stages

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage A · Problem-Setter (出题人 Agent)                                 │
│   input  : model + GPU + SGLang                                         │
│   process: bench → mine logs → classify → expand boundary → re-score    │
│   output : experiments/problems[_moe]/PNNN/  (self-contained packages)  │
│                                                                         │
│ Stage B · Problem-Solver Fleet (做题人 Agent fleet)                     │
│   input  : one problem package                                          │
│   process: per-specialty sub-agent (config / scheduler / kernel /       │
│            workload-shape) reads evidence → proposes & verifies fix     │
│   output : experiments/problems*/PNNN/attempts/attempt_NNN/             │
│            + experiments/problems*/PNNN/solution.md                     │
└─────────────────────────────────────────────────────────────────────────┘
```

A **problem package** is the contract: `problem.json` + workload YAMLs
+ baseline metrics + evidence + suggested strategies + acceptance
criteria + every attempt — all in one tarball-able directory. Schema:
[`docs/problem-package/schema.md`](./docs/problem-package/schema.md).

---

## 4. Stage status

| Stage | What works today | Code |
|---|---|---|
| **A** Problem-Setter | Mode A (rule-based) end-to-end; produces v1 problem packages from seeds → bench → triage → boundary expansion → select. | [`stages/problem-setter/policies/rule_based_explore.py`](./stages/problem-setter/policies/rule_based_explore.py) |
| **B** config-agent | Single-value and `--exhaustive` sweep; reuses prior attempts; writes solution.md; noise-aware tiebreak (smallest value within ±1%). | [`scripts/solver/config_agent.py`](./scripts/solver/config_agent.py) |
| **B** scheduler-agent | Not yet built | — |
| **B** kernel-agent | Not yet built (skill ready: [`.github/skills/pytorch-profiling`](./.github/skills/pytorch-profiling)) | — |
| **B** workload-shape-agent | Not yet built | — |

---

## 5. Where to read next

- **Architecture** → [`docs/architecture/two-stage-overview.md`](./docs/architecture/two-stage-overview.md)
- **How to add a new seed regime / axis / skill** → [`stages/problem-setter/EXTENSION_GUIDE.md`](./stages/problem-setter/EXTENSION_GUIDE.md)
- **Latest results** → [`docs/reports/2026-05-28-progress.md`](./docs/reports/2026-05-28-progress.md)
- **Full dev-side narrative** → [`docs/development/developer-guide.md`](./docs/development/developer-guide.md)
- **Future research directions** → [`docs/research/regime-search-extensions.md`](./docs/research/regime-search-extensions.md)

---

## 6. Where to find experiment records & project history

> The recent docs reshuffle moved only the **top-level `.md` files** into
> `docs/`. Nothing under `experiments/`, `regime_scout/outputs/`,
> `archive/`, or `.github/skills/` was touched.

### Experiment records (what we have run, what we found)

- **Index page** → [`experiments/README.md`](./experiments/README.md)
  — table of every run, anatomy of a problem package, how to reproduce
- Per-run artefacts:
  - `experiments/problems/P001/` — Qwen3-0.6B problem (validated)
  - `experiments/problems_moe/P001/{attempts/, solution.md}` — MoE problem + 3 solver attempts
  - `experiments/ideas/from_setter/idea_001.json` — R-001 (closed, see Finding-B probe)
  - `experiments/regimes/{STAGE1,MOE_STAGE_A}_REPORT_20260528.md` — first-run reports
- Raw scout output: `regime_scout/outputs/` (10 files — `raw_results.jsonl`, `regime_map.md`, MoE counterparts, Finding-B probe results)
- Transient logs / jsonl: `experiments/tmp/` (gitignored — regenerable)

### Project history (how the repo evolved)

- **Timeline doc** → [`docs/development/history.md`](./docs/development/history.md)
  — v0.1 → v0.2 → v0.4 evolution, lessons learned, commit-by-commit
- v0.2 design drafts (superseded): [`archive/DESIGN_v0.2.md`](./archive/DESIGN_v0.2.md), [`archive/TWO_STAGE_SUPPLEMENT_v0.2.md`](./archive/TWO_STAGE_SUPPLEMENT_v0.2.md)
- v0.4 restructure log: [`docs/development/restructure-notes.md`](./docs/development/restructure-notes.md)
- Boss-facing snapshot: [`docs/reports/2026-05-28-progress.md`](./docs/reports/2026-05-28-progress.md)
- Git log: `git log --oneline` (5 commits today; full table in `history.md`)

---

## 7. Environment

- Hardware tested: 8× H200 (143 GB each); single-GPU runs use GPU 0.
- Conda env: `sglang-dev` (sglang 0.5.12.post1, CUDA 12.8).
- Models tested: `/data/hf/models/Qwen3-0.6B`,
  `/data/hf/models/Qwen3-30B-A3B-Instruct-2507` (MoE, 128 experts).
- HF cache pinned to `/data/hf/gujialiang123/hf_cache` (writable).

If you are on a different machine, edit `configs/*.yaml` and
`scripts/utils.py::env_for_config`'s HF-cache path accordingly.

---

## 8. License & attribution

Internal research code. Co-authored with GitHub Copilot CLI.

---

<a id="中文版"></a>

# 中文版

一个端到端的 agent 系统，输入是 **一个模型 + 一台 GPU + SGLang**，自动：

1. 发现 SGLang 在哪些 **regime**（推理场景）上表现不好（Stage A 出题人）；
2. 调动专项子 agent 把这些 regime 修好（Stage B 做题人）。

整个系统借用了 **算法竞赛** 的隐喻：Stage A 是出题人，Stage B 是做题人。
两阶段之间通过一个 **冻结的、自包含的"题目包"** 在磁盘上交接 —— 所以
你可以用一个工具跑出题人、另一个工具跑做题人，或者干脆把题目包发给同事
让 ta 手工解。

**当前状态 (2026-05-28)**：Stage A 已端到端打通，产出了两个经验证的
题目包（Qwen3-0.6B P001、MoE Qwen3-30B-A3B P001）。Stage B 的
config-agent 一发就把 MoE P001 修好了，**TTFT p95 下降 92.6%**。
最新进展见 [`docs/reports/2026-05-28-progress.md`](./docs/reports/2026-05-28-progress.md)。

---

## 1. 快速开始

```bash
# 0. 激活环境（一次）
conda activate sglang-dev

# 1. 让 config 指向你的模型（一次）
$EDITOR configs/base.yaml          # 把 model-path 改成 /你的/模型路径

# 2. Stage A — 自动找问题
python stages/problem-setter/policies/rule_based_explore.py \
    --config configs/base.yaml
#   → experiments/problems/PNNN/   （冻结的题目包）

# 3. Stage B — 用 config-agent 解其中某个题
python scripts/solver/config_agent.py \
    --problem experiments/problems/P001 \
    --strategy S001 --exhaustive
#   → experiments/problems/P001/attempts/attempt_NNN/
#   → experiments/problems/P001/solution.md
```

单 H200 + Qwen3-0.6B 大约耗时：Stage A ≈ 15 分钟，Stage B
config-agent（3 个值）≈ 12 分钟。

每个脚本内部干了什么、seed 是怎么生成的、triage 怎么走、MoE 配置有什么
不一样 —— 完整说明见
[开发文档（developer guide）](./docs/development/developer-guide.md)。

---

## 2. 仓库结构

```text
EndtoEnd-auto-optimization/
├── README.md                       ← 当前文件（项目简介 + 快速开始）
├── configs/                        ← 各模型的 sglang server 配置
│   ├── base.yaml                   ← 默认（小型 dense 模型）
│   └── moe_qwen3_30b.yaml          ← MoE 示例
├── regime_scout/                   ← Stage A 的输入（seed regime）和输出
│   ├── candidates/                 ← seed suite YAML（Qwen3 dense）
│   ├── candidates_moe/             ← seed suite YAML（MoE）
│   └── outputs/                    ← regime_map.md、selected_cases.jsonl
├── stages/
│   ├── problem-setter/             ← Stage A：出题人 agent
│   │   ├── README.md               ← 阶段概览 + 调用方式
│   │   ├── AGENT_CONTRACT.md       ← agent 必须做 / 不能做什么
│   │   ├── PLAYBOOK.md             ← 一步步操作流程
│   │   ├── TOOLS.md                ← agent 可调用的 harness 脚本
│   │   ├── EXTENSION_GUIDE.md      ← 如何加新 seed / 新 axis / 新 skill
│   │   └── policies/
│   │       ├── rule_based_explore.py    ← Mode A：确定性 harness
│   │       └── llm_agent.md             ← Mode B：LLM 系统提示词
│   └── problem-solver/             ← Stage B：做题人 agent 群
│       └── README.md               ← （占位 — 代码在 scripts/solver/）
├── scripts/                        ← harness（两阶段共用）
│   ├── run_experiment.py           ← server + bench 封装
│   ├── run_regime_suite.py         ← 连续跑 N 个 workload
│   ├── select_problems.py          ← 把原始 run → 题目包
│   ├── solver/
│   │   └── config_agent.py         ← Stage B：config-agent（含 --exhaustive）
│   └── utils.py                    ← env / yaml / 路径辅助
├── .github/skills/                 ← 可复用的方法论单元（skill）
│   ├── server-log-mining/          ← L2 证据：解析 server.log
│   ├── failure-classification/     ← L3 证据：bench 失败分类
│   ├── noise-aware-scoring/        ← v2 怀疑度评分
│   ├── boundary-expansion/         ← 沿一个 axis 扩展邻居
│   ├── suspicion-scoring/          ← 把证据组合成 score
│   ├── minimal-repro-shrink/       ← （设计了，未实现）
│   └── pytorch-profiling/          ← L4 证据：torch profile + 归约
├── experiments/
│   ├── README.md                   ← 所有实验的索引 + 复现说明
│   ├── problems/                   ← Stage A 输出（dense 模型）
│   ├── problems_moe/               ← Stage A 输出（MoE 模型）
│   ├── ideas/                      ← idea pool（双向通道）
│   ├── regimes/                    ← 原始 Stage A 扫描输出 + 首跑报告
│   └── tmp/                        ← 单次运行的临时数据（gitignore）
├── docs/                           ← 全部文档
│   ├── architecture/
│   │   └── two-stage-overview.md   ← 架构 canonical 文档
│   ├── problem-package/
│   │   └── schema.md               ← problem.json v1 契约
│   ├── idea-pool/
│   │   └── schema.md               ← idea.json 契约
│   ├── skills/
│   │   └── README.md               ← skills 设计原则 + 清单
│   ├── research/
│   │   └── regime-search-extensions.md  ← R1–R9 未来方向
│   ├── reports/
│   │   └── 2026-05-28-progress.md  ← 最新进展报告
│   └── development/                ← 长篇开发文档
│       ├── developer-guide.md      ← 完整设计 + 心智模型
│       ├── history.md              ← 项目演化时间线（v0.1→v0.4）
│       ├── log-layout.md           ← 所有 log 文件位置
│       └── restructure-notes.md    ← 2-stage 重构历史
├── archive/                        ← 已废弃的设计草案
└── logs/                           ← orchestrator 日志（内容 gitignore）
```

---

## 3. 两个阶段

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage A · 出题人 Agent（Problem-Setter）                                │
│   输入：model + GPU + SGLang                                            │
│   过程：bench → 挖 server log → 失败分类 → 扩展邻居 → 重打分            │
│   输出：experiments/problems[_moe]/PNNN/  （自包含的题目包）            │
│                                                                         │
│ Stage B · 做题人 Agent 群（Problem-Solver Fleet）                       │
│   输入：一个题目包                                                      │
│   过程：按专长分工的子 agent（config / scheduler / kernel /             │
│         workload-shape）读证据 → 提方案 → 验证                          │
│   输出：experiments/problems*/PNNN/attempts/attempt_NNN/                │
│         + experiments/problems*/PNNN/solution.md                        │
└─────────────────────────────────────────────────────────────────────────┘
```

题目包是两阶段之间的 **契约**：`problem.json` + workload YAML + baseline
指标 + 证据 + 建议策略 + 验收标准 + 全部 attempt —— 全在一个可 tar
打包的目录里。Schema 见
[`docs/problem-package/schema.md`](./docs/problem-package/schema.md)。

---

## 4. 各阶段当前状态

| 阶段 | 已能做 | 代码 |
|---|---|---|
| **A** Problem-Setter | Mode A（rule-based）端到端可跑；从 seed → bench → triage → 边界扩展 → 选题 全流程，产出 v1 题目包。 | [`stages/problem-setter/policies/rule_based_explore.py`](./stages/problem-setter/policies/rule_based_explore.py) |
| **B** config-agent | 单值 + `--exhaustive` 扫描；自动复用已有 attempt；写 solution.md；噪声感知 tiebreak（在 ±1% 内取最小值）。 | [`scripts/solver/config_agent.py`](./scripts/solver/config_agent.py) |
| **B** scheduler-agent | 尚未实现 | — |
| **B** kernel-agent | 尚未实现（skill 已就绪：[`.github/skills/pytorch-profiling`](./.github/skills/pytorch-profiling)） | — |
| **B** workload-shape-agent | 尚未实现 | — |

---

## 5. 继续阅读

- **架构**：[`docs/architecture/two-stage-overview.md`](./docs/architecture/two-stage-overview.md)
- **如何加新 seed regime / axis / skill**：[`stages/problem-setter/EXTENSION_GUIDE.md`](./stages/problem-setter/EXTENSION_GUIDE.md)
- **最新结果**：[`docs/reports/2026-05-28-progress.md`](./docs/reports/2026-05-28-progress.md)
- **开发侧长文叙述**：[`docs/development/developer-guide.md`](./docs/development/developer-guide.md)
- **未来研究方向**：[`docs/research/regime-search-extensions.md`](./docs/research/regime-search-extensions.md)

---

## 6. 实验记录 / 项目历史在哪里

> 最近这次文档大搬家只移动了 **顶层 `.md` 文件**，搬到了 `docs/` 下。
> `experiments/`、`regime_scout/outputs/`、`archive/`、`.github/skills/`
> 一个都没动。

### 实验记录（我们跑过什么、发现了什么）

- **索引页** → [`experiments/README.md`](./experiments/README.md)
  —— 所有 run 的总表、题目包结构剖析、如何复现
- 单次 run 产物：
  - `experiments/problems/P001/` —— Qwen3-0.6B 题目（已验证）
  - `experiments/problems_moe/P001/{attempts/, solution.md}` —— MoE 题目 + 3 次 solver attempt
  - `experiments/ideas/from_setter/idea_001.json` —— R-001（已关闭，详见 Finding-B 探测）
  - `experiments/regimes/{STAGE1,MOE_STAGE_A}_REPORT_20260528.md` —— 首跑报告
- 原始 scout 输出：`regime_scout/outputs/`（10 个文件 —— `raw_results.jsonl`、`regime_map.md`、MoE 对应版本、Finding-B 探测结果）
- 临时日志 / jsonl：`experiments/tmp/`（gitignore —— 可重新生成）

### 项目历史（仓库怎么演化到今天的）

- **时间线文档** → [`docs/development/history.md`](./docs/development/history.md)
  —— v0.1 → v0.2 → v0.4 演化、留下的经验、commit-by-commit
- v0.2 设计草案（已被取代）：[`archive/DESIGN_v0.2.md`](./archive/DESIGN_v0.2.md)、[`archive/TWO_STAGE_SUPPLEMENT_v0.2.md`](./archive/TWO_STAGE_SUPPLEMENT_v0.2.md)
- v0.4 重构日志：[`docs/development/restructure-notes.md`](./docs/development/restructure-notes.md)
- 给老板看的进度快照：[`docs/reports/2026-05-28-progress.md`](./docs/reports/2026-05-28-progress.md)
- Git log：`git log --oneline`（今天 5 个 commit；完整表格在 `history.md`）

---

## 7. 环境

- 测试硬件：8× H200（每张 143 GB）；单卡运行用 GPU 0。
- Conda 环境：`sglang-dev`（sglang 0.5.12.post1，CUDA 12.8）。
- 测试模型：`/data/hf/models/Qwen3-0.6B`、
  `/data/hf/models/Qwen3-30B-A3B-Instruct-2507`（MoE，128 专家）。
- HF cache 锁定到 `/data/hf/gujialiang123/hf_cache`（可写）。

如果你在别的机器上跑，请相应修改 `configs/*.yaml` 以及
`scripts/utils.py::env_for_config` 中的 HF-cache 路径。

---

## 8. License & 协作

内部研究代码。Co-authored with GitHub Copilot CLI.
