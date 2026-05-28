# SGLang End-to-End Optimization Agent 设计文档

版本：v0.2
目标读者：GitHub Copilot CLI / coding agent / 未来维护者
项目目标：构建一个 autonomous experiment harness，让 Agent 能够在 SGLang 上针对不同 workload regime 自动搜索、验证、记录 serving 配置优化。

> 📖 **新读者请先读** [`README.md`](./README.md)（工程报告 / 项目总览）。本文档是详细 spec。
> 🆕 **第 0.5 节 (v0.2 Amendments)** 覆盖了 v0.1 的若干错误与缺失，应**优先**于后续章节中被列出的对应小节。

## 0. 背景与核心思想

本项目参考 Auto GPU Kernel 报告里的 autonomous optimization loop 思路：Agent 不应该一次性“自由发挥”去改一堆代码，而应该被限制在一个可重复的实验状态机里：

```text
读取历史状态
→ 分析当前 workload
→ 只提出一个配置改动
→ 应用配置
→ quick benchmark
→ medium benchmark
→ 必要时 paired A/B benchmark
→ 记录实验
→ 决定 keep / revert / profile / research
```

报告中的关键经验是：优化系统的核心不是让 LLM 一次性写出最快 kernel，而是让 LLM 成为严格的实验管理器，依赖 durable artifacts、分层 benchmark、profiler、workload-inspector 和 one-change-per-iteration 规则。

本项目第一版只做 **SGLang config-level optimization**，不让 Agent 修改 SGLang 源码，不写 kernel。

---

## 0.5 v0.2 Amendments（read this first）

本节相对 v0.1 做了以下修订。后续章节（§1 – §27）保持原文不动，但其中以下条目**以本节为准**：

### Critical bug fixes

- **B1 — SGLang launch CLI**：`python -m sglang.launch_server` **不接受 `--config <yaml>` flag**。所有 server 参数都是顶层 CLI flag（`--model-path`、`--tp-size`、`--mem-fraction-static` …）。`scripts/launch_server.py` 必须把 YAML 翻译成 argv list 后再 `subprocess.Popen`。bool 字段语义：`True` → 加入 `--key`，`False` → 完全不传。
- **B2 — Copilot CLI flag**：当前 Copilot CLI **没有 `--agent` 和 `--max-autopilot-continues` flag**。custom agent 通过 `.github/agents/*.agent.md` 注册，运行时由 prompt 内 `@sglang-optimizer` mention 调用；autopilot 通过 `--allow-all-tools` 启用。
- **B3 — `schedule-policy` 取值**：应为 `[lpm, fcfs, random, dfs-weight]`，SGLang 默认是 `lpm`（不是 `fcfs`）。`configs/base.yaml` 与 `search_space.yaml` 都按此修正。
- **B4 — Prompt 注入风险**：`copilot -p "$(cat prompts/optimize_once.md)"` 在 prompt 内包含反引号、`$`、引号时会被 shell 二次展开。改为：让 prompt 文件首行注明 `@sglang-optimizer`，然后在 shell 里用 short 指令 `"Read and execute prompts/optimize_once.md."` 调用。

### New / strengthened concepts

- **N1** — 系统总览架构图（§0.A）
- **N2** — Workload **regime** 的构造与输入策略（§0.B）  ← 用户重点关心的部分
- **N3** — Knob 选择策略（§0.C），覆盖 §3.2 / §15
- **N4** — 统计严谨的接受规则（§0.D），覆盖 §13、§14
- **N5** — Warmup 必要性与具体规则（§0.E），强化 §12.6
- **N6** — 新增 4 个 scripts（§0.F）
- **N7** — 修订对照表（§0.G）

---

### 0.A 系统总览架构图

```
   USER (one-time setup)
     └─ 模型路径 + GPU/机器 + workload yaml (含 regime label)
                                │
                                ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ run_copilot_loop.sh   (outer shell controller, 唯一无限循环者) │
   │  - 每个 tick 调用一次 copilot CLI                              │
   │  - 处理 STOP / PAUSE 文件                                      │
   │  - check_loop_state.py 决定是否退出                            │
   └────────────────────────────────┬───────────────────────────────┘
                                    │  per tick (= one experiment)
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ copilot CLI  →  @sglang-optimizer (single iteration, 跑完即退) │
   │                                                                │
   │   1. READ    AGENTS.md, configs/*, workloads/current.yaml,     │
   │              experiments/summary.md, experiments/LESSONS.md    │
   │   2. PLAN    pick ONE (knob, value)  ── §0.C deterministic     │
   │   3. EDIT    configs/candidate.yaml  = best.yaml ± 1 knob      │
   │   4. RUN    ┌─ launch_server.py  (YAML→argv, see B1)           │
   │     quick   ├─ wait_ready.py                                   │
   │             ├─ warmup  (§0.E)                                  │
   │             ├─ run_benchmark.py                                │
   │             └─ parse_metrics.py  → metrics.json                │
   │   5. RUN    medium (≥3 repeats)                                │
   │   6. RUN    A/B (ABBA×2)  if ambiguous   OR   full if clear    │
   │   7. DECIDE keep / revert / needs_ab / needs_profile /         │
   │             needs_workload_inspection / needs_research  (§0.D) │
   │   8. WRITE  experiments/exp_N/{plan, config, workload,         │
   │             metrics, env, server.log, bench.log, result.md}    │
   │             update summary.md, LESSONS.md, best.yaml           │
   └────────────────────────────────┬───────────────────────────────┘
                                    │  plateau triggers (5 / 10 轮无 keep)
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ Sub-agents (called on plateau, NOT every tick)                 │
   │   @workload-inspector  →  experiments/workload_profile.md      │
   │   @profiler            →  experiments/profile.md               │
   │   @research            →  experiments/research_plan.md         │
   │   (这三个 agent 都只写文档、不动 config)                       │
   └────────────────────────────────────────────────────────────────┘
```

**关键约束**（与 v0.1 一致，但在此重申）：
1. 一次 copilot CLI 调用 = 一个完整 iteration。**跑完即退**，无限循环只在外层 shell。
2. 一个 iteration 只改 **1 个** knob。
3. 任何 "改进" 必须有 `metrics.json` 支撑，不允许 LLM intuition。
4. workloads / scripts / .github/agents 不允许被 optimizer 修改（由 §0.F integrity_check 强制）。

---

### 0.B Workload Regime：构造与输入策略

**Regime 是什么** — 一个表征 workload 主导 compute/memory 模式的标签。它是 agent 决策的核心 prior：决定哪些 knob 该先试、哪些先回避、哪个指标作 primary。

**枚举值**：

| Regime | 典型场景 | 主要瓶颈 | Primary metric 默认 |
|---|---|---|---|
| `short_in_short_out` | 短输入短输出在线服务 | scheduler / HTTP / CUDA-graph overhead | `ttft_p95_ms` |
| `long_in_short_out` | RAG / 摘要 / 长 system prompt | prefill / memory | `ttft_p95_ms` |
| `short_in_long_out` | chat / agent 长生成 | decode throughput / continuous batching | `output_tokens_per_second` |
| `long_in_long_out` | 长上下文生成 | prefill + decode 复合 | `output_tokens_per_second` |
| `prefix_reuse` | 共享 system prompt / few-shot | radix cache / prefix-aware scheduling | `ttft_p95_ms` |
| `mixed_online` | 变长 + Poisson arrival 真实流量 | 多瓶颈共存 | `request_throughput` + constraint |

**Source of truth** — 每个 `workloads/*.yaml` **必须**包含顶层字段 `regime:`，由人手写入或由 `scripts/inspect_workload.py` 自动填入。

```yaml
# workloads/long_doc_qa.yaml
name: "long_doc_qa"
regime: long_in_short_out          # ← MUST be present; "auto" = let inspector classify
description: "Long-document Q&A on 4k context inputs."
primary_metric: "ttft_p95_ms"
primary_direction: "lower"
constraints:                        # ← NEW in v0.2 (see §0.D)
  ttft_p95_ms: { max: 2000 }
  failed_requests: { max: 0 }
  oom: { max: 0 }
model: "/path/to/model"
host: "127.0.0.1"
port: 30000
dataset: { ... }
traffic: { ... }
benchmark:
  warmup_prompts: 16                # ← NEW in v0.2 (see §0.E)
  ...
```

**Rule-based 分类器** (`scripts/inspect_workload.py`)：

```python
def classify_regime(workload: dict) -> str:
    ds = workload["dataset"]
    if ds.get("name") == "generated-shared-prefix" or ds.get("shared_prefix"):
        return "prefix_reuse"

    in_len  = ds.get("random_input_len", 0)
    out_len = ds.get("random_output_len", 0)
    has_rate    = workload["traffic"].get("request_rate") is not None
    range_ratio = ds.get("random_range_ratio", 0.0)

    if has_rate or range_ratio > 0.3:
        return "mixed_online"

    if in_len <= 512  and out_len <= 128:  return "short_in_short_out"
    if in_len >= 2048 and out_len <= 128:  return "long_in_short_out"
    if in_len <= 512  and out_len >= 512:  return "short_in_long_out"
    if in_len >= 1024 and out_len >= 512:  return "long_in_long_out"
    return "unknown"
```

**人写 vs 自动**：
- yaml 显式 `regime: X` → inspector 用规则验算并 WARN 不一致；以 yaml 为准。
- yaml `regime: auto` → inspector 用规则填入并提交到 yaml。
- 规则返回 `unknown` 或人 / 规则不一致且人没解释 → optimizer **拒绝运行**，等人补 label。

**Workload 构造的三种模式**：

```
方式 A：从模板复制（最常用）
  cp workloads/templates/short_in_short_out.yaml workloads/my.yaml
  # 编辑 model / num_prompts / lengths，保留或调整 regime

方式 B：用真实 trace 抽样（v0.3 引入）
  scripts/build_workload_from_trace.py --trace prod.jsonl --out workloads/replay.yaml
  # 自动统计 input/output 长度分布与 arrival rate，inspector 推断 regime

方式 C：launch-time 切换
  ln -sf workloads/long_in_short_out.yaml workloads/current.yaml
  # 下一个 tick 起 optimizer 切到新 workload；它会先重新评估 best.yaml 作为新 baseline
```

**Regime → Agent 的输入路径**：

```
workloads/current.yaml.regime
      │
      ├─→ @sglang-optimizer 的 system prompt 上下文（"current regime = X"）
      ├─→ §15 prior table lookup (try_first / avoid_initially knob 列表)
      ├─→ §0.D primary_metric 默认值（如果 yaml 没写）
      └─→ §0.C knob-selection-policy 的初始优先级
```

---

### 0.C Knob 选择策略（input construction for the optimizer agent）

LLM 每轮挑 knob **不应"自由发挥"**，而应执行下面这条确定性管线。`AGENTS.md` 与 `prompts/optimize_once.md` 都要重申这条规则。

```
INPUT
  S  = configs/search_space.yaml.allowed_knobs
  B  = configs/best.yaml
  R  = workloads/current.yaml.regime
  H  = experiments/summary.md            (all (knob,value,decision) rows)
  L  = experiments/LESSONS.md            (regime-level blacklist)

STEP 1  Eligibility
  candidates = set(S.keys())

STEP 2  Regime priors (from §15)
  try_first = regime_priors[R].try_first        # ordered list
  avoid     = regime_priors[R].avoid_initially  # set

STEP 3  Exclude already-tried for this workload
  tried = { (k, v) for row in H
            if row.workload == current and row.decision in {keep, revert} }

STEP 4  Exclude proven-bad for this regime
  bad   = { (k, v) for entry in L
            if entry.regime == R and entry.proven_bad }

STEP 5  Build ordered queue
  queue = []
  for k in try_first:
      if k in candidates - avoid - {kk for (kk,_) in bad}:
          v = next_unvisited_neighbor(B[k], S[k].values, tried, bad)
          if v is not None:
              queue.append((k, v))
  for k in candidates - set(try_first) - avoid:
      ...同上...

STEP 6  Pick head; empty → plateau trigger
  if queue is empty:
      return PLATEAU                    # §0.C plateau rule below

STEP 7  Within-knob value selection
  numerical: 从 best 当前值出发，按数值邻近顺序挑未访问的；优先朝 LESSONS 暗示的方向
  categorical: 未访问值轮转

STEP 8  Write experiments/pending_plan.md
  {knob, old_value, new_value, regime, hypothesis (from §15 + LESSONS), risk}
```

**Plateau rule**（覆盖 v0.1 §3.2 第 11 步与 §3.1 plateau 描述）：

```
最近 5 个 experiment 全部 revert        → 下一轮先触发 @workload-inspector
最近 10 个 experiment 全部 revert       → 下一轮触发 @profiler + @research
@research 可以提议把 search_space 扩展或允许 knob-group 联合改动
```

**重要：integrity check**（§0.F）会在每轮开始前校验 `workloads/` / `scripts/` / `.github/agents/` 未被 optimizer 修改；若被改则立即 abort 并把变更回滚。这是 agent 作弊的最后一道防线。

---

### 0.D 接受规则（覆盖 §13）

**Repeats**（覆盖 §14 各 stage 的 repeat 数）：

| Stage | Repeats | 用途 |
|---|---:|---|
| `quick`  | 1 | smoke test；不参与 keep 决策 |
| `medium` | **3** | 默认决策依据 |
| `full`   | **5** | 确认已 accept 的 candidate |
| `A/B`    | ABBA×2 (8 runs) | quick / medium 落在 noise 边界时 |

**Noise baseline**（NEW）：

第一次启动 loop 前 **必须**跑：
```bash
python scripts/calibrate_noise.py \
    --config configs/best.yaml --workload workloads/current.yaml --repeats 5
# 输出 experiments/noise_baseline.json
# {
#   "ttft_p95_ms":  { "mean":..., "std":..., "cv_pct": 6.2 },
#   "output_tokens_per_second": { "mean":..., "std":..., "cv_pct": 2.1 },
#   ...
# }
```
每 20 个 experiment 后自动重校（drift check）。

**接受判定**（替换 §13.2）：

```python
def decide(best_runs, cand_runs, workload, noise_baseline, ab_runs=None):
    # Hard fail
    if any(r.crash or r.oom or r.timeout or r.parse_error for r in cand_runs):
        return "revert"
    if violates_constraints(cand_runs, workload.get("constraints", {})):
        return "revert"

    primary   = workload["primary_metric"]
    direction = workload["primary_direction"]
    cv_pct    = noise_baseline[primary]["cv_pct"]

    if ab_runs:
        # paired A/B 已控制 cross-run drift
        delta_pct, p, ci_lo, ci_hi = compare_paired(ab_runs, primary, direction)
        threshold = max(3.0, cv_pct)
    else:
        delta_pct, p, ci_lo, ci_hi = compare_welch(best_runs, cand_runs, primary, direction)
        threshold = max(5.0, 2.0 * cv_pct)   # noise-aware

    if ci_lo > 0 and delta_pct >= threshold and p < 0.05:
        return "keep"
    if ci_hi < 0 and abs(delta_pct) >= threshold and p < 0.05:
        return "revert"
    return "needs_ab"
```

**Constraints**（NEW，§0.B 已示例）：workload yaml 里允许声明硬约束，违反即 revert：
```yaml
constraints:
  ttft_p95_ms: { max: 2000 }
  failed_requests: { max: 0 }
  oom: { max: 0 }
```

---

### 0.E Warmup（强化 §12.6）

每个 `run_experiment.py --mode X` 在开始计时前必须：

1. 跑 `workload.benchmark.warmup_prompts`（默认 16）个 warmup 请求；
2. 等 server log 出现 `cuda graph captured` 或等价信号；
3. 对 `prefix_reuse` 额外把所有 shared prefix 跑一遍填满 radix cache；
4. 等 `nvidia-smi` 连续两次采样 GPU 频率方差 < 5%（GPU 进入热稳态）；
5. **warmup 阶段的所有 metric 全部丢弃**，不写入 `metrics.json`。

---

### 0.F 新增 scripts

```
scripts/calibrate_noise.py              算 noise baseline（§0.D）
scripts/integrity_check.py              校验受保护目录的 SHA（§0.C 最后一段）
scripts/collect_env.py                  采集 SGLang commit / GPU / CUDA / torch 版本
scripts/build_workload_from_trace.py    (v0.3) 从真实 trace 生成 workload
scripts/init_project.sh                 一次性建目录、初始化 summary/LESSONS
```

`run_experiment.py` 在 launch server 前必须依次调用：

```python
integrity_check.main(strict=True)              # 任何 SHA 变化 → abort + rollback
env_meta = collect_env.collect()               # 写入 exp_N/env.json
```

---

### 0.G 修订对照表

| v0.1 章节 | 状态 | 修订内容 |
|---|---|---|
| §3.1 / §3.2 plateau | 覆盖 | 见 §0.C plateau rule |
| §4 copilot 启动命令 | 就地修复 | `--agent` / `--max-autopilot-continues` 移除；改为 `--allow-all-tools` + prompt 内 `@agent` mention |
| §7 prompts/optimize_once.md | 就地修复 | 首行加 `@sglang-optimizer` 调用；增加 §0.C policy 指令 |
| §9.1 / 9.2 / 9.3 base/best/candidate.yaml | 覆盖 | `schedule-policy` 默认 `lpm`（v0.1 写的 `fcfs` 不再是 SGLang 默认） |
| §9.4 search_space.yaml `schedule-policy.values` | 覆盖 | `[lpm, fcfs, random, dfs-weight]` |
| §10 workloads/*.yaml | 覆盖 | 新增必填 `regime:` 与可选 `constraints:`、`benchmark.warmup_prompts:` |
| §12.2 launch_server.py | 就地修复 | YAML → argv 翻译，**不要**用 `--config` flag |
| §12.6 run_experiment.py | 强化 | 增加 warmup 阶段 + integrity_check + collect_env |
| §13 decision logic | 覆盖 | 见 §0.D，含统计检验、constraint、noise-aware 门槛 |
| §14.2 medium repeats | 覆盖 | 1 → **3** |
| §14.3 full repeats | 覆盖 | 3 → **5** |
| §14.4 A/B order | 覆盖 | `ABBA` → `ABBA × 2` (8 runs) |
| §15 regime priors | 保留 | 仍作为 §0.C 的初始优先级表 |
| §17.1 manual launch | 就地修复 | 改用 `python scripts/launch_server.py --config ...` |
| §25 MVP success criteria | 增加 | 加一条 "noise baseline 已校准" |

---

## 1. 项目范围

### 1.1 第一版目标

第一版实现一个可运行的闭环系统：

1. 启动 SGLang server。
2. 根据 workload 配置运行 benchmark。
3. 解析 TTFT、TPOT、ITL、throughput、success rate 等指标。
4. 让 Copilot CLI 每轮只改一个 SGLang 配置参数。
5. 自动执行 quick / medium / full / A/B benchmark。
6. 把每次实验记录到 `experiments/`。
7. 维护 `experiments/summary.md`、`experiments/LESSONS.md`、`configs/best.yaml`。
8. 支持外层 bash loop 每隔一段时间调用一次 Copilot。

### 1.2 非目标

第一版不做：

1. 不修改 SGLang 源码。
2. 不修改模型权重。
3. 不修改 benchmark workload 来“作弊”。
4. 不用 Agent 直接解释性能结果，所有结论必须来自 benchmark 输出。
5. 不做多机分布式。
6. 不做 kernel-level Triton/CUDA 优化。
7. 不做生产服务部署。

### 1.3 后续扩展

第一版稳定后，可以扩展到：

1. regime-aware policy selection；
2. workload classifier；
3. SGLang scheduler 层源码优化；
4. prefix cache / radix cache policy 优化；
5. attention backend dispatch 优化；
6. 与 Claude Code / Codex CLI 做 agent 对照实验。

---

## 2. 总体架构

项目采用如下目录结构：

```text
sglang-agent-lab/
  README.md
  DESIGN.md
  AGENTS.md

  prompts/
    optimize_once.md
    inspect_workload.md
    profile_once.md
    research_once.md

  .github/
    agents/
      sglang-optimizer.agent.md
      workload-inspector.agent.md
      profiler.agent.md
      research.agent.md
    skills/
      sglang-benchmark/
        SKILL.md

  configs/
    base.yaml
    candidate.yaml
    best.yaml
    search_space.yaml

  workloads/
    current.yaml
    short_in_short_out.yaml
    long_in_short_out.yaml
    short_in_long_out.yaml
    prefix_reuse.yaml
    mixed_online.yaml

  scripts/
    run_copilot_loop.sh
    run_experiment.py
    launch_server.py
    wait_ready.py
    run_benchmark.py
    parse_metrics.py
    ab_benchmark.py
    inspect_workload.py
    profile_server.py
    check_loop_state.py
    utils.py

  experiments/
    summary.md
    LESSONS.md
    workload_profile.md
    profile.md
    tmp/
    exp_001/
      plan.md
      config.yaml
      workload.yaml
      quick_metrics.json
      medium_metrics.json
      full_metrics.json
      ab_metrics.json
      result.md
      server.log
      benchmark.log

  logs/
    copilot-loop/
```

---

## 3. Agent 工作方式

### 3.1 外层 loop

外层 loop 由 shell 脚本控制，而不是让 Copilot 在一次 session 里无限循环。

每次 loop 只让 Copilot 做一件事：

```text
运行一次完整 optimization iteration
```

这样可以：

1. 控制每轮实验边界；
2. 限制每次 Copilot 自主执行步数；
3. 保存每轮 Copilot stdout/stderr；
4. 遇到 crash 可以恢复；
5. 用 `STOP` 文件停止 loop；
6. 防止 Agent 自己陷入无限循环。

### 3.2 内层 iteration

每次 Copilot 被调用后，必须执行：

```text
1. 读取 AGENTS.md
2. 读取 experiments/summary.md
3. 读取 experiments/LESSONS.md
4. 读取 configs/search_space.yaml
5. 读取 configs/best.yaml
6. 读取 workloads/current.yaml
7. 选择一个配置 knob
8. 写 experiments/pending_plan.md
9. 修改 configs/candidate.yaml
10. 运行 quick benchmark
11. 如果 quick 通过，运行 medium benchmark
12. 如果提升不明显但可能有效，运行 A/B benchmark
13. 如果明显有效，运行 full benchmark
14. 创建 experiments/exp_N/
15. 写 result.md
16. 更新 summary.md
17. 如果接受改动，更新 best.yaml
18. 如果拒绝改动，candidate.yaml 回滚到 best.yaml
19. 停止
```

---

## 4. Copilot CLI loop 脚本

创建文件：

```text
scripts/run_copilot_loop.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-50}"

mkdir -p "$ROOT/logs/copilot-loop"
mkdir -p "$ROOT/experiments/tmp"

echo "Starting SGLang Copilot optimization loop"
echo "ROOT=$ROOT"
echo "INTERVAL_SECONDS=$INTERVAL_SECONDS"
echo "MAX_EXPERIMENTS=$MAX_EXPERIMENTS"

while true; do
  ts="$(date +%Y%m%d_%H%M%S)"
  echo "=== Copilot optimize iteration $ts ===" | tee -a "$ROOT/logs/copilot-loop/loop.log"

  if [[ -f "$ROOT/STOP" ]]; then
    echo "STOP file found. Exiting loop." | tee -a "$ROOT/logs/copilot-loop/loop.log"
    exit 0
  fi

  (
    cd "$ROOT"

    # NOTE (v0.2 §0.G B2): 原 v0.1 的 --agent / --autopilot / --yolo /
    # --max-autopilot-continues 这些 flag 当前 Copilot CLI 并不支持。
    # custom agent 通过 .github/agents/*.agent.md 注册 + prompt 内 @mention 调用，
    # autopilot 通过 --allow-all-tools 启用。
    copilot \
      --allow-all-tools \
      -p "Read and execute prompts/optimize_once.md as the @sglang-optimizer agent. Stop after one iteration." \
      > "logs/copilot-loop/stdout_${ts}.log" \
      2> "logs/copilot-loop/stderr_${ts}.log"
  ) || {
    echo "Copilot iteration $ts failed. See stderr_${ts}.log" \
      | tee -a "$ROOT/logs/copilot-loop/loop.log"
  }

  python scripts/check_loop_state.py \
    --root "$ROOT" \
    --max-experiments "$MAX_EXPERIMENTS" \
    || {
      echo "Loop stop condition reached." | tee -a "$ROOT/logs/copilot-loop/loop.log"
      exit 0
    }

  sleep "$INTERVAL_SECONDS"
done
```

注意：

1. 开发初期可以把 `--yolo` 去掉，手动批准工具。
2. 真正无人值守时必须在隔离环境里运行，例如单独 conda env、容器、专门实验机。
3. 如果本地 Copilot CLI 不支持某个 flag，先执行 `copilot --help`，替换成当前版本对应选项。

---

## 5. AGENTS.md

创建文件：

```text
AGENTS.md
```

内容：

```md
# SGLang End-to-End Optimization Agent

You are an autonomous experiment manager for SGLang serving performance.

Your job is not to freely rewrite the codebase.
Your job is to run one careful, measurable optimization experiment per invocation.

## Primary objective

Find SGLang serving configurations that improve performance for the current workload.

Primary metrics:
- TTFT p50 / p95
- TPOT p50 / p95
- ITL p50 / p95
- output tokens per second
- request throughput
- successful request count

Secondary metrics:
- server startup success
- OOM count
- crash count
- benchmark timeout count
- GPU memory usage
- GPU utilization if available

## Hard rules

1. Run exactly one experiment per invocation.
2. Change exactly one config knob per experiment.
3. Do not modify workload files unless the human explicitly asks.
4. Do not modify model path, model weights, prompt distribution, output length, request count, or benchmark seed.
5. Do not edit SGLang source code in v0.1.
6. Do not edit benchmark scripts unless the current task is explicitly to implement or fix the harness.
7. Do not claim improvement unless it is measured by scripts/run_experiment.py or scripts/ab_benchmark.py.
8. If improvement is smaller than 5%, run paired A/B before accepting.
9. Always log failed experiments.
10. Always log OOMs, timeouts, server crashes, and benchmark parse failures.
11. If candidate is not accepted, revert configs/candidate.yaml to configs/best.yaml.
12. If candidate is accepted, copy configs/candidate.yaml to configs/best.yaml.
13. Never run git push.
14. Never run sudo.
15. Never delete experiments/.
16. Never delete logs/.
17. Never change more than one knob by hiding changes in multiple files.
18. Stop after one iteration.

## Source of truth

The source of truth for performance is benchmark output.

Do not trust:
- intuition
- single noisy run
- previous chat memory
- incomplete logs
- relative speedups across different GPUs
- runs with different workload files
- runs with different model revisions

Trust:
- metrics.json
- ab_metrics.json
- repeated runs
- same GPU
- same workload
- same model
- same SGLang commit

## Experiment protocol

Each invocation must do:

1. Read:
   - AGENTS.md
   - configs/base.yaml
   - configs/best.yaml
   - configs/candidate.yaml
   - configs/search_space.yaml
   - workloads/current.yaml
   - experiments/summary.md
   - experiments/LESSONS.md

2. Assess:
   - What is the current best config?
   - What workload regime is being optimized?
   - Which metrics are primary?
   - What was tried recently?
   - Which knobs should not be repeated?

3. Plan:
   - Pick exactly one knob from configs/search_space.yaml.
   - Write experiments/pending_plan.md.
   - State hypothesis, expected metric change, and possible risk.

4. Implement:
   - Modify only configs/candidate.yaml.
   - Preserve all other knobs from configs/best.yaml.

5. Validate:
   - Run quick benchmark:
     python scripts/run_experiment.py --mode quick --config configs/candidate.yaml --workload workloads/current.yaml

6. Measure:
   - If quick passes, run medium benchmark:
     python scripts/run_experiment.py --mode medium --config configs/candidate.yaml --workload workloads/current.yaml

7. A/B:
   - If medium improvement is between -5% and +5%, run:
     python scripts/ab_benchmark.py --a configs/best.yaml --b configs/candidate.yaml --workload workloads/current.yaml

8. Full:
   - If medium improvement is clearly positive, run:
     python scripts/run_experiment.py --mode full --config configs/candidate.yaml --workload workloads/current.yaml

9. Log:
   - Create experiments/exp_N/.
   - Copy config, workload, metrics, logs.
   - Write result.md.
   - Update experiments/summary.md.
   - Update experiments/LESSONS.md only if there is a durable lesson.

10. Decision:
   - keep
   - revert
   - needs_ab
   - needs_profile
   - needs_workload_inspection
   - needs_research

11. Stop.

## Acceptance rule

Accept candidate if:
- no server crash
- no OOM
- successful request count equals expected request count
- primary metric improves by at least 5% in medium or A/B
- secondary metrics do not catastrophically regress

For latency metrics, lower is better.
For throughput metrics, higher is better.

If TTFT improves but TPOT regresses, or throughput improves but p95 latency regresses, use the workload's primary_metric field to decide.

## Plateau rule

If the last 5 experiments show no accepted improvement:
- Run workload-inspector next.

If the last 10 experiments show no accepted improvement:
- Run profiler or research next.

## Benchmark gaming rules

Forbidden:
- changing workload to make benchmark easier
- reducing num_prompts
- reducing output length
- changing model
- changing tokenizer
- caching outputs manually
- bypassing correctness
- ignoring failed requests
- reporting only the best trial
- changing benchmark parser to hide regressions

Allowed:
- changing SGLang server config knobs listed in search_space.yaml
- changing benchmark stage from quick to medium to full
- running A/B to reduce noise
- adding more logging
```

---

## 6. Copilot custom agent

创建文件：

```text
.github/agents/sglang-optimizer.agent.md
```

内容：

```md
---
name: sglang-optimizer
description: Runs one measured SGLang serving optimization experiment at a time.
tools:
  - view
  - edit
  - create
  - glob
  - grep
  - shell
  - task
---

You are the main optimizer for the SGLang experiment loop.

Follow AGENTS.md exactly.

You must run exactly one optimization iteration.

You may:
- read configs
- read workloads
- read experiment history
- edit configs/candidate.yaml
- run scripts/run_experiment.py
- run scripts/ab_benchmark.py
- create one experiments/exp_N directory
- update experiments/summary.md
- update experiments/LESSONS.md when there is a durable lesson

You must not:
- edit workloads
- edit model files
- edit SGLang source
- edit benchmark scripts during optimization iterations
- change more than one knob
- claim speedup without measured metrics
- run git push
- run sudo
- delete logs
- delete experiments

If benchmark scripts are missing or broken, stop optimization and implement the missing harness first.
```

---

## 7. Copilot prompt

创建文件：

```text
prompts/optimize_once.md
```

内容：

```md
@sglang-optimizer

Run exactly one SGLang optimization iteration following AGENTS.md and §0.C
Knob-Selection-Policy (DESIGN.md).

Follow AGENTS.md.

Important:
- Do not ask the user questions.
- Make the best reasonable choice from existing files.
- Stop after one complete iteration.
- Change exactly one knob in configs/candidate.yaml.
- Do not modify workloads.
- Do not modify SGLang source.
- Do not modify benchmark scripts unless the harness is missing and this is the first setup run.

Steps:

1. Inspect:
   - AGENTS.md
   - configs/base.yaml
   - configs/best.yaml
   - configs/candidate.yaml
   - configs/search_space.yaml
   - workloads/current.yaml
   - experiments/summary.md
   - experiments/LESSONS.md

2. Choose exactly one candidate config change.
   - The knob must exist in configs/search_space.yaml.
   - The new value must be one of the allowed values.
   - Do not change any other knob.

3. Write experiments/pending_plan.md with:
   - selected knob
   - old value
   - new value
   - hypothesis
   - expected metric improvement
   - possible risk
   - benchmark plan

4. Copy configs/best.yaml to configs/candidate.yaml, then apply the one selected change.

5. Run quick benchmark:
   python scripts/run_experiment.py --mode quick --config configs/candidate.yaml --workload workloads/current.yaml

6. If quick passes, run medium benchmark:
   python scripts/run_experiment.py --mode medium --config configs/candidate.yaml --workload workloads/current.yaml

7. If medium result is ambiguous or improvement is smaller than 5%, run paired A/B:
   python scripts/ab_benchmark.py --a configs/best.yaml --b configs/candidate.yaml --workload workloads/current.yaml

8. If medium result is clearly positive, run full benchmark:
   python scripts/run_experiment.py --mode full --config configs/candidate.yaml --workload workloads/current.yaml

9. Create the next experiments/exp_N directory.

10. Copy into exp_N:
    - experiments/pending_plan.md as plan.md
    - configs/candidate.yaml as config.yaml
    - workloads/current.yaml as workload.yaml
    - generated metric files
    - generated server logs
    - generated benchmark logs

11. Write exp_N/result.md with:
    - changed knob
    - old value
    - new value
    - hypothesis
    - quick result
    - medium result
    - full result if any
    - A/B result if any
    - decision
    - explanation grounded in measured metrics
    - durable lesson if any

12. Update experiments/summary.md.

13. If decision is keep:
    - copy configs/candidate.yaml to configs/best.yaml

14. If decision is revert:
    - copy configs/best.yaml to configs/candidate.yaml

15. Stop.
```

---

## 8. Benchmark skill

创建文件：

```text
.github/skills/sglang-benchmark/SKILL.md
```

内容：

```md
---
name: sglang-benchmark
description: Run SGLang benchmark stages and parse serving metrics.
---

Use this skill whenever benchmarking a candidate SGLang config.

Never hand-write benchmark commands unless scripts/run_experiment.py is missing.

Benchmark stages:

1. quick
   python scripts/run_experiment.py --mode quick --config configs/candidate.yaml --workload workloads/current.yaml

2. medium
   python scripts/run_experiment.py --mode medium --config configs/candidate.yaml --workload workloads/current.yaml

3. full
   python scripts/run_experiment.py --mode full --config configs/candidate.yaml --workload workloads/current.yaml

4. ab
   python scripts/ab_benchmark.py --a configs/best.yaml --b configs/candidate.yaml --workload workloads/current.yaml

After each run, read the generated metrics JSON and summarize:
- successful_requests
- failed_requests
- ttft_p50_ms
- ttft_p95_ms
- tpot_p50_ms
- tpot_p95_ms
- itl_p50_ms
- itl_p95_ms
- output_tokens_per_second
- request_throughput
- total_latency_p50_ms
- total_latency_p95_ms
- server_crash
- oom
- timeout
```

---

## 9. 配置文件格式

### 9.1 `configs/base.yaml`

```yaml
model-path: "/path/to/your/model"
host: "127.0.0.1"
port: 30000

# Keep simple for v0.1.
tensor-parallel-size: 1
enable-metrics: true
log-requests: false

# Baseline knobs.
schedule-policy: "fcfs"
max-running-requests: 16
schedule-conservativeness: 1.0
chunked-prefill-size: -1
max-prefill-tokens: 8192
mem-fraction-static: 0.85
disable-radix-cache: false
disable-cuda-graph: false
cuda-graph-max-bs: 16
num-continuous-decode-steps: 1
```

### 9.2 `configs/best.yaml`

初始时复制 `base.yaml`：

```yaml
model-path: "/path/to/your/model"
host: "127.0.0.1"
port: 30000
tensor-parallel-size: 1
enable-metrics: true
log-requests: false

schedule-policy: "fcfs"
max-running-requests: 16
schedule-conservativeness: 1.0
chunked-prefill-size: -1
max-prefill-tokens: 8192
mem-fraction-static: 0.85
disable-radix-cache: false
disable-cuda-graph: false
cuda-graph-max-bs: 16
num-continuous-decode-steps: 1
```

### 9.3 `configs/candidate.yaml`

初始时也复制 `base.yaml`。Agent 每轮只允许改这个文件中的一个 knob。

### 9.4 `configs/search_space.yaml`

```yaml
allowed_knobs:

  schedule-policy:
    type: categorical
    values:
      - fcfs
      - lpm
      - random
    risk: medium
    notes: "Scheduling policy. lpm may help prefix-heavy workloads."

  max-running-requests:
    type: int
    values:
      - 4
      - 8
      - 16
      - 32
    risk: medium
    notes: "Controls concurrency admitted by server."

  schedule-conservativeness:
    type: float
    values:
      - 0.5
      - 0.8
      - 1.0
      - 1.3
      - 1.5
    risk: medium
    notes: "Lower may admit more requests; higher may avoid KV pressure."

  chunked-prefill-size:
    type: int
    values:
      - -1
      - 2048
      - 4096
      - 8192
    risk: medium
    notes: "-1 disables chunked prefill. Smaller chunks may help long prompt stability."

  max-prefill-tokens:
    type: int
    values:
      - 4096
      - 8192
      - 16384
    risk: medium
    notes: "Controls prefill batch token limit."

  mem-fraction-static:
    type: float
    values:
      - 0.70
      - 0.80
      - 0.85
      - 0.90
    risk: high
    notes: "Too high may OOM; too low may reduce KV capacity."

  disable-radix-cache:
    type: bool
    values:
      - false
      - true
    risk: low
    notes: "Ablation knob. Should usually be false for prefix reuse."

  disable-cuda-graph:
    type: bool
    values:
      - false
      - true
    risk: low
    notes: "Ablation knob. Disabling may hurt throughput but can help debug."

  cuda-graph-max-bs:
    type: int
    values:
      - 8
      - 16
      - 32
    risk: medium
    notes: "Tune CUDA graph max batch size."

  num-continuous-decode-steps:
    type: int
    values:
      - 1
      - 2
      - 4
    risk: medium
    notes: "May reduce scheduling overhead but can increase TTFT."
```

---

## 10. Workload 文件格式

### 10.1 `workloads/current.yaml`

这是当前优化目标。可以是 symlink，也可以手动复制某个 workload。

```yaml
name: "short_in_short_out"
description: "Short prompt, short output, online serving latency regime."

primary_metric: "ttft_p95_ms"
primary_direction: "lower"

model: "/path/to/your/model"
host: "127.0.0.1"
port: 30000

dataset:
  name: "random"
  random_input_len: 128
  random_output_len: 32
  random_range_ratio: 0.0

traffic:
  max_concurrency: 16
  request_rate: null
  num_prompts:
    quick: 32
    medium: 160
    full: 800

benchmark:
  timeout_seconds:
    server_start: 300
    quick: 300
    medium: 900
    full: 1800
  repeat:
    quick: 1
    medium: 1
    full: 3
  flush_cache: true
```

### 10.2 `workloads/short_in_short_out.yaml`

```yaml
name: "short_in_short_out"
description: "Short prompt and short output. Often sensitive to scheduler overhead, launch overhead, and CUDA graph settings."

primary_metric: "ttft_p95_ms"
primary_direction: "lower"

model: "/path/to/your/model"
host: "127.0.0.1"
port: 30000

dataset:
  name: "random"
  random_input_len: 128
  random_output_len: 32
  random_range_ratio: 0.0

traffic:
  max_concurrency: 16
  request_rate: null
  num_prompts:
    quick: 32
    medium: 160
    full: 800

benchmark:
  timeout_seconds:
    server_start: 300
    quick: 300
    medium: 900
    full: 1800
  repeat:
    quick: 1
    medium: 1
    full: 3
  flush_cache: true
```

### 10.3 `workloads/long_in_short_out.yaml`

```yaml
name: "long_in_short_out"
description: "Long prompt and short output. Prefill-heavy workload."

primary_metric: "ttft_p95_ms"
primary_direction: "lower"

model: "/path/to/your/model"
host: "127.0.0.1"
port: 30000

dataset:
  name: "random"
  random_input_len: 4096
  random_output_len: 32
  random_range_ratio: 0.0

traffic:
  max_concurrency: 8
  request_rate: null
  num_prompts:
    quick: 16
    medium: 80
    full: 400

benchmark:
  timeout_seconds:
    server_start: 300
    quick: 600
    medium: 1200
    full: 2400
  repeat:
    quick: 1
    medium: 1
    full: 3
  flush_cache: true
```

### 10.4 `workloads/short_in_long_out.yaml`

```yaml
name: "short_in_long_out"
description: "Short prompt and long output. Decode-heavy throughput workload."

primary_metric: "output_tokens_per_second"
primary_direction: "higher"

model: "/path/to/your/model"
host: "127.0.0.1"
port: 30000

dataset:
  name: "random"
  random_input_len: 128
  random_output_len: 512
  random_range_ratio: 0.0

traffic:
  max_concurrency: 32
  request_rate: null
  num_prompts:
    quick: 32
    medium: 160
    full: 800

benchmark:
  timeout_seconds:
    server_start: 300
    quick: 600
    medium: 1800
    full: 3600
  repeat:
    quick: 1
    medium: 1
    full: 3
  flush_cache: true
```

### 10.5 `workloads/prefix_reuse.yaml`

```yaml
name: "prefix_reuse"
description: "Shared prefix workload. Tests radix cache and prefix-aware scheduling."

primary_metric: "ttft_p95_ms"
primary_direction: "lower"

model: "/path/to/your/model"
host: "127.0.0.1"
port: 30000

dataset:
  name: "generated-shared-prefix"
  gsp_num_groups: 16
  gsp_prompts_per_group: 8
  gsp_system_prompt_len: 2048
  gsp_question_len: 128
  gsp_output_len: 128

traffic:
  max_concurrency: 16
  request_rate: null
  num_prompts:
    quick: 32
    medium: 128
    full: 512

benchmark:
  timeout_seconds:
    server_start: 300
    quick: 600
    medium: 1800
    full: 3600
  repeat:
    quick: 1
    medium: 1
    full: 3
  flush_cache: false
```

---

## 11. 实验输出格式

### 11.1 `experiments/summary.md`

初始化：

```md
# Experiment Summary

Current best config: configs/best.yaml

| Exp | Date | Workload | Changed knob | Old | New | Primary metric | Best | Candidate | Delta | Decision | Notes |
|---:|---|---|---|---|---|---|---:|---:|---:|---|---|
```

### 11.2 `experiments/LESSONS.md`

初始化：

```md
# Durable Lessons

Only add lessons that are likely to remain true across runs.

## General

- Change one knob per experiment.
- For <5% gains, use paired A/B.
- Treat server crash, OOM, timeout, and failed requests as hard failures.

## Workload-specific

No durable workload-specific lessons yet.
```

### 11.3 `experiments/exp_N/result.md`

每次实验写：

```md
# Experiment N

Date:
Workload:
Model:
SGLang commit:
GPU:
Config:

## Change

Changed knob:
Old value:
New value:

## Hypothesis

...

## Benchmark results

### Quick

- passed:
- successful_requests:
- ttft_p50_ms:
- ttft_p95_ms:
- tpot_p50_ms:
- tpot_p95_ms:
- itl_p50_ms:
- itl_p95_ms:
- output_tokens_per_second:
- request_throughput:
- server_crash:
- oom:
- timeout:

### Medium

...

### Full

...

### A/B

...

## Decision

keep / revert / needs_ab / needs_profile / needs_workload_inspection / needs_research

## Reasoning

Use only measured metrics.

## Durable lesson

Only include if there is a durable lesson.
```

---

## 12. Script specifications

Copilot should implement the scripts below.

### 12.1 `scripts/utils.py`

Responsibilities:

1. Load YAML.
2. Save YAML.
3. Run subprocess with timeout.
4. Kill process group.
5. Find next experiment ID.
6. Copy files safely.
7. Detect OOM/crash from logs.
8. Normalize metrics.

Required functions:

```python
def load_yaml(path: str | Path) -> dict: ...
def save_yaml(obj: dict, path: str | Path) -> None: ...
def run_cmd(cmd: list[str], timeout: int | None, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess: ...
def next_exp_dir(root: Path = Path("experiments")) -> Path: ...
def copy_if_exists(src: Path, dst: Path) -> None: ...
def detect_oom(text: str) -> bool: ...
def detect_server_crash(text: str) -> bool: ...
def now_str() -> str: ...
```

### 12.2 `scripts/launch_server.py`

Purpose:

Start SGLang server with a YAML config.

CLI:

```bash
python scripts/launch_server.py --config configs/candidate.yaml --log experiments/tmp/server.log
```

Behavior:

1. Read config.
2. Run:

   ```text
   # NOTE (v0.2 §0.G B1): SGLang 没有 --config flag。launch_server.py 必须把 YAML
   # 翻译成 argv list，例如：
   #   ["python", "-m", "sglang.launch_server",
   #    "--model-path",        config["model-path"],
   #    "--host",               config["host"],
   #    "--port",               str(config["port"]),
   #    "--tp-size",            str(config["tensor-parallel-size"]),
   #    "--mem-fraction-static",str(config["mem-fraction-static"]),
   #    ...]
   # bool 字段：True → 追加 "--key"；False → 完全不传。
   # 字段 -1 → 完全不传（用 server 默认）。
   ```
3. Write stdout/stderr to log file.
4. Print spawned PID.
5. Do not wait forever.
6. Parent script should own termination.

Implementation note:

Use `subprocess.Popen(..., preexec_fn=os.setsid)` on Linux, so the whole process group can be killed.

### 12.3 `scripts/wait_ready.py`

Purpose:

Wait until SGLang server is ready.

CLI:

```bash
python scripts/wait_ready.py --host 127.0.0.1 --port 30000 --timeout 300
```

Behavior:

1. Poll:

   * `http://host:port/health`
   * if unavailable, try a lightweight OpenAI-compatible models endpoint
   * if unavailable, try opening TCP socket
2. Return 0 if server becomes ready.
3. Return nonzero on timeout.

### 12.4 `scripts/run_benchmark.py`

Purpose:

Run one SGLang benchmark stage against an already-running server.

CLI:

```bash
python scripts/run_benchmark.py \
  --mode quick \
  --workload workloads/current.yaml \
  --out experiments/tmp/quick_raw.jsonl \
  --log experiments/tmp/benchmark.log
```

Behavior:

1. Load workload YAML.
2. Convert workload mode to `num_prompts`.
3. Build `python -m sglang.bench_serving` command.
4. Support dataset types:

   * `random`
   * `generated-shared-prefix`
5. Include `--output-file`.
6. Include `--output-details`.
7. Include `--flush-cache` when workload says so.
8. Return nonzero if benchmark command fails.

For random dataset, command shape:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 64 \
  --random-range-ratio 0.0 \
  --num-prompts 80 \
  --max-concurrency 16 \
  --output-file experiments/tmp/quick_raw.jsonl \
  --output-details
```

For generated shared prefix:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 16 \
  --gsp-prompts-per-group 8 \
  --gsp-system-prompt-len 2048 \
  --gsp-question-len 128 \
  --gsp-output-len 128 \
  --num-prompts 128 \
  --max-concurrency 16 \
  --output-file experiments/tmp/medium_raw.jsonl \
  --output-details
```

If the installed SGLang version uses slightly different CLI argument names, implement a compatibility layer by checking `python -m sglang.bench_serving --help`.

### 12.5 `scripts/parse_metrics.py`

Purpose:

Parse SGLang benchmark output into normalized JSON.

CLI:

```bash
python scripts/parse_metrics.py \
  --raw experiments/tmp/quick_raw.jsonl \
  --log experiments/tmp/benchmark.log \
  --server-log experiments/tmp/server.log \
  --out experiments/tmp/quick_metrics.json
```

Output schema:

```json
{
  "passed": true,
  "mode": "quick",
  "successful_requests": 80,
  "failed_requests": 0,
  "ttft_p50_ms": 12.3,
  "ttft_p95_ms": 24.5,
  "tpot_p50_ms": 3.1,
  "tpot_p95_ms": 5.2,
  "itl_p50_ms": 3.0,
  "itl_p95_ms": 5.0,
  "output_tokens_per_second": 1234.5,
  "request_throughput": 12.3,
  "total_latency_p50_ms": 100.0,
  "total_latency_p95_ms": 150.0,
  "server_crash": false,
  "oom": false,
  "timeout": false,
  "parse_error": null,
  "raw_files": {
    "raw": "experiments/tmp/quick_raw.jsonl",
    "benchmark_log": "experiments/tmp/benchmark.log",
    "server_log": "experiments/tmp/server.log"
  }
}
```

Parser requirements:

1. Be robust to JSONL lines containing either request details or summary.
2. If summary metrics are available from stdout, parse them.
3. If only per-request details exist, compute p50/p95 from request details.
4. If parsing fails, set `passed=false` and `parse_error`.
5. Never silently return fake zero metrics.

### 12.6 `scripts/run_experiment.py`

Purpose:

One full benchmark stage from server launch to metrics output.

CLI:

```bash
python scripts/run_experiment.py \
  --mode quick \
  --config configs/candidate.yaml \
  --workload workloads/current.yaml
```

Modes:

```text
quick
medium
full
```

Behavior:

1. Create `experiments/tmp/<timestamp>/`.
2. Kill any old server on target port if needed.
3. Launch server using config.
4. Wait until ready.
5. Run warmup benchmark if necessary.
6. Run benchmark for selected mode.
7. Parse metrics.
8. Kill server process group.
9. Copy final metrics to:

   * `experiments/tmp/latest_quick_metrics.json`
   * `experiments/tmp/latest_medium_metrics.json`
   * `experiments/tmp/latest_full_metrics.json`
10. Print compact summary to stdout.
11. Return nonzero if:

* server failed to start
* benchmark failed
* parse failed
* OOM
* server crash
* successful_requests < expected

High-level pseudocode:

```python
def main():
    args = parse_args()
    tmp = make_tmp_dir(args.mode)

    config = load_yaml(args.config)
    workload = load_yaml(args.workload)

    server_log = tmp / "server.log"
    benchmark_log = tmp / "benchmark.log"
    raw_out = tmp / f"{args.mode}_raw.jsonl"
    metrics_out = tmp / f"{args.mode}_metrics.json"

    proc = launch_server(config_path=args.config, log_path=server_log)

    try:
        wait_ready(host=config["host"], port=config["port"], timeout=...)
        run_benchmark(mode=args.mode, workload=args.workload, raw_out=raw_out, log=benchmark_log)
        parse_metrics(raw=raw_out, log=benchmark_log, server_log=server_log, out=metrics_out)
    finally:
        kill_process_group(proc)

    copy metrics/logs to experiments/tmp/latest_...
    validate metrics
    print summary
```

### 12.7 `scripts/ab_benchmark.py`

Purpose:

Paired A/B benchmark between current best and candidate.

CLI:

```bash
python scripts/ab_benchmark.py \
  --a configs/best.yaml \
  --b configs/candidate.yaml \
  --workload workloads/current.yaml
```

Behavior:

1. Run ABBA order:

   ```text
   A, B, B, A, A, B
   ```
2. For each run:

   * launch server
   * wait ready
   * run medium benchmark
   * parse metrics
   * kill server
3. Compute paired summary:

   * mean A
   * mean B
   * median A
   * median B
   * delta percentage
   * wins by metric
4. Write:

   * `experiments/tmp/latest_ab_metrics.json`

Output schema:

```json
{
  "passed": true,
  "workload": "short_in_short_out",
  "primary_metric": "ttft_p95_ms",
  "primary_direction": "lower",
  "runs": [
    {
      "label": "A",
      "metrics": {}
    },
    {
      "label": "B",
      "metrics": {}
    }
  ],
  "summary": {
    "a_mean_primary": 20.0,
    "b_mean_primary": 18.0,
    "delta_pct": -10.0,
    "candidate_wins": true,
    "decision_hint": "keep"
  }
}
```

Acceptance:

1. If primary direction is lower, B must be lower by at least 5%.
2. If primary direction is higher, B must be higher by at least 5%.
3. Failed requests, OOM, crash, timeout cause candidate rejection.

### 12.8 `scripts/inspect_workload.py`

Purpose:

Analyze workload shape before optimization.

CLI:

```bash
python scripts/inspect_workload.py --workload workloads/current.yaml --out experiments/workload_profile.md
```

Output:

```md
# Workload Profile

Workload:
Dataset:
Primary metric:

## Regime classification

short_in_short_out / long_in_short_out / decode_heavy / prefix_reuse / mixed

## Key properties

- input length:
- output length:
- max concurrency:
- num prompts:
- shared prefix:
- cache flush:
- estimated prefill/decode ratio:

## Likely bottlenecks

...

## Suggested knobs

1.
2.
3.

## Knobs to avoid initially

...
```

For v0.1, this can be rule-based from YAML fields.

### 12.9 `scripts/profile_server.py`

Purpose:

Run profiling only when needed.

CLI:

```bash
python scripts/profile_server.py \
  --config configs/best.yaml \
  --workload workloads/current.yaml \
  --out experiments/profile.md
```

Behavior:

1. Launch server.
2. Run `bench_serving` with profiling enabled if available.
3. Save traces to `experiments/tmp/profile/`.
4. Summarize:

   * prefill-heavy or decode-heavy
   * obvious server idle gaps
   * top-level latency signals
   * suspected next knob

For v0.1, keep this simple. It can parse server logs and benchmark metrics before supporting full PyTorch profiler traces.

### 12.10 `scripts/check_loop_state.py`

Purpose:

Stop loop if conditions are met.

CLI:

```bash
python scripts/check_loop_state.py --root . --max-experiments 50
```

Behavior:

1. Stop if `STOP` file exists.
2. Stop if experiment count >= max.
3. Stop if 5 consecutive crashes/OOMs.
4. Stop if `experiments/summary.md` is missing or corrupted.
5. Return 0 to continue, nonzero to stop.

Implementation:

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-experiments", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.root)

    if (root / "STOP").exists():
        print("STOP file found.")
        return 1

    exp_root = root / "experiments"
    if not exp_root.exists():
        print("experiments/ does not exist.")
        return 1

    exps = sorted(p for p in exp_root.glob("exp_*") if p.is_dir())
    if len(exps) >= args.max_experiments:
        print(f"Reached max experiments: {len(exps)}")
        return 1

    consecutive_bad = 0
    for exp in reversed(exps):
        result = exp / "result.md"
        if not result.exists():
            continue
        text = result.read_text(errors="ignore").lower()
        if "decision: keep" in text:
            break
        if any(x in text for x in ["oom", "server_crash", "timeout", "decision: revert", "failed"]):
            consecutive_bad += 1
        else:
            break

    if consecutive_bad >= 5:
        print("Too many consecutive bad experiments.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 13. Decision logic

Copilot should implement decision logic in Python, not purely in prose.

### 13.1 Metric comparison

Function:

```python
def compare_metric(best: float, candidate: float, direction: str) -> float:
    """
    Return signed improvement percentage.
    Positive means candidate is better.
    """
```

Rules:

```text
direction = lower:
  improvement = (best - candidate) / best * 100

direction = higher:
  improvement = (candidate - best) / best * 100
```

### 13.2 Candidate acceptance

```python
def decide(best_metrics, candidate_metrics, workload, ab_metrics=None) -> str:
    if candidate failed:
        return "revert"

    primary = workload["primary_metric"]
    direction = workload["primary_direction"]

    if ab_metrics exists:
        if ab says candidate improves primary by >= 5%:
            return "keep"
        else:
            return "revert"

    improvement = compare primary metric

    if improvement >= 5:
        return "keep"

    if -5 <= improvement < 5:
        return "needs_ab"

    return "revert"
```

### 13.3 Hard failures

Always reject if:

1. server crash;
2. OOM;
3. timeout;
4. parse error;
5. successful_requests lower than expected;
6. p95 latency is missing;
7. throughput is missing;
8. benchmark file missing.

---

## 14. 多阶段 benchmark 设计

### 14.1 quick

Purpose:

Catch obvious failures quickly.

Properties:

```text
- small num_prompts
- one repeat
- same workload shape
- not used for final decision unless candidate fails
```

Reject on:

```text
- server cannot start
- OOM
- crash
- parse failure
- failed requests
- primary metric > 2x worse than best
```

### 14.2 medium

Purpose:

Default per-iteration measurement.

Properties:

```text
- enough num_prompts for rough stability
- one repeat initially
- used to decide keep / revert / needs_ab
```

### 14.3 full

Purpose:

Confirm accepted improvement.

Properties:

```text
- larger num_prompts
- 3 repeats
- used when candidate looks clearly better
```

### 14.4 paired A/B

Purpose:

Resolve noisy or small deltas.

Properties:

```text
- ABBA order
- same GPU
- same workload
- same model
- same SGLang commit
- restart server between configs
```

A/B is required for:

```text
- improvement between 0% and 5%
- regression between 0% and 5%
- mixed metrics
- suspiciously noisy benchmark
```

---

## 15. 工作负载 regime 与推荐 knobs

Agent 可以用下面的先验，但必须通过 benchmark 验证。

### 15.1 short_in_short_out

Likely bottleneck:

```text
- scheduling overhead
- HTTP overhead
- CUDA graph capture / replay
- small batch inefficiency
```

Try first:

```text
cuda-graph-max-bs
disable-cuda-graph ablation
max-running-requests
num-continuous-decode-steps
```

Avoid initially:

```text
chunked-prefill-size
max-prefill-tokens
```

### 15.2 long_in_short_out

Likely bottleneck:

```text
- prefill
- memory pressure
- chunked prefill
- prefill token batching
```

Try first:

```text
chunked-prefill-size
max-prefill-tokens
mem-fraction-static
schedule-conservativeness
```

Avoid initially:

```text
num-continuous-decode-steps
disable-radix-cache
```

### 15.3 short_in_long_out

Likely bottleneck:

```text
- decode throughput
- continuous batching
- CUDA graph
- KV cache capacity
```

Try first:

```text
num-continuous-decode-steps
max-running-requests
cuda-graph-max-bs
schedule-conservativeness
```

Avoid initially:

```text
chunked-prefill-size
```

### 15.4 prefix_reuse

Likely bottleneck:

```text
- prefix cache reuse
- radix cache
- prefix-aware scheduling
- cache eviction
```

Try first:

```text
schedule-policy: lpm
disable-radix-cache ablation
max-running-requests
schedule-conservativeness
```

Avoid initially:

```text
flush_cache true
```

---

## 16. Initialization commands

Create directories:

```bash
mkdir -p prompts
mkdir -p .github/agents
mkdir -p .github/skills/sglang-benchmark
mkdir -p configs
mkdir -p workloads
mkdir -p scripts
mkdir -p experiments/tmp
mkdir -p logs/copilot-loop
```

Initialize experiment files:

```bash
cat > experiments/summary.md <<'EOF'
# Experiment Summary

Current best config: configs/best.yaml

| Exp | Date | Workload | Changed knob | Old | New | Primary metric | Best | Candidate | Delta | Decision | Notes |
|---:|---|---|---|---|---|---|---:|---:|---:|---|---|
EOF

cat > experiments/LESSONS.md <<'EOF'
# Durable Lessons

Only add lessons that are likely to remain true across runs.

## General

- Change one knob per experiment.
- For <5% gains, use paired A/B.
- Treat server crash, OOM, timeout, and failed requests as hard failures.

## Workload-specific

No durable workload-specific lessons yet.
EOF
```

Copy configs:

```bash
cp configs/base.yaml configs/best.yaml
cp configs/base.yaml configs/candidate.yaml
cp workloads/short_in_short_out.yaml workloads/current.yaml
```

Make scripts executable:

```bash
chmod +x scripts/run_copilot_loop.sh
```

---

## 17. Manual smoke test

Before using Copilot loop, manually test harness.

### 17.1 Launch server

```bash
# v0.2 §0.G B1：用脚本启动，不要直接 `python -m sglang.launch_server --config ...`
python scripts/launch_server.py \
    --config configs/base.yaml \
    --log /tmp/manual_server.log &
python scripts/wait_ready.py --host 127.0.0.1 --port 30000 --timeout 300
```

### 17.2 Run benchmark manually

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model "$(python - <<'PY'
import yaml
print(yaml.safe_load(open('configs/base.yaml'))['model-path'])
PY
)" \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 32 \
  --random-range-ratio 0.0 \
  --num-prompts 32 \
  --max-concurrency 8 \
  --output-file experiments/tmp/manual_smoke.jsonl \
  --output-details
```

### 17.3 Run harness

```bash
python scripts/run_experiment.py \
  --mode quick \
  --config configs/candidate.yaml \
  --workload workloads/current.yaml
```

### 17.4 Run A/B

```bash
python scripts/ab_benchmark.py \
  --a configs/best.yaml \
  --b configs/candidate.yaml \
  --workload workloads/current.yaml
```

Only start Copilot loop after these pass.

---

## 18. Copilot loop run command

Start loop:

```bash
bash scripts/run_copilot_loop.sh
```

Stop loop:

```bash
touch STOP
```

View progress:

```bash
tail -f logs/copilot-loop/loop.log
```

View latest Copilot output:

```bash
ls -t logs/copilot-loop/stdout_*.log | head -1 | xargs tail -n 200
```

View latest experiment:

```bash
ls -td experiments/exp_* | head -1
```

---

## 19. Implementation order for Copilot

Copilot should implement in this order:

### Phase A: skeleton

1. Create directory structure.
2. Create config YAML files.
3. Create workload YAML files.
4. Create `AGENTS.md`.
5. Create prompt files.
6. Create summary and lessons files.

### Phase B: benchmark harness

1. Implement `scripts/utils.py`.
2. Implement `scripts/wait_ready.py`.
3. Implement `scripts/launch_server.py`.
4. Implement `scripts/run_benchmark.py`.
5. Implement `scripts/parse_metrics.py`.
6. Implement `scripts/run_experiment.py`.
7. Test quick benchmark manually.

### Phase C: A/B harness

1. Implement `scripts/ab_benchmark.py`.
2. Implement ABBA order.
3. Implement summary JSON.
4. Test with identical A and B configs first.
5. Test with one changed knob.

### Phase D: loop

1. Implement `scripts/check_loop_state.py`.
2. Implement `scripts/run_copilot_loop.sh`.
3. Run one Copilot iteration manually.
4. Inspect experiment result.
5. Fix harness if needed.

### Phase E: sub-agents

1. Add workload-inspector.
2. Add profiler.
3. Add research agent.
4. Add plateau rule.

---

## 20. Workload-inspector agent

创建文件：

```text
.github/agents/workload-inspector.agent.md
```

内容：

```md
---
name: workload-inspector
description: Analyzes SGLang workload shape and recommends which config knobs are plausible.
tools:
  - view
  - create
  - shell
---

You analyze workload files and benchmark history.

You do not modify configs.
You do not modify workloads.
You do not run optimization.
You only write experiments/workload_profile.md.

Read:
- workloads/current.yaml
- experiments/summary.md
- experiments/LESSONS.md

Write:
- experiments/workload_profile.md

Include:
- workload regime classification
- input length
- output length
- concurrency
- prefix reuse properties
- cache flush behavior
- primary metric
- likely bottleneck
- suggested knobs
- knobs to avoid
```

Prompt:

```text
prompts/inspect_workload.md
```

```md
Run workload inspection.

Read workloads/current.yaml, experiments/summary.md, and experiments/LESSONS.md.

Write experiments/workload_profile.md.

Do not change configs.
Do not change workloads.
Do not run benchmark.
```

---

## 21. Profiler agent

创建文件：

```text
.github/agents/profiler.agent.md
```

内容：

```md
---
name: profiler
description: Runs or summarizes profiling for SGLang serving bottleneck analysis.
tools:
  - view
  - create
  - shell
---

You profile the current best SGLang configuration.

You do not modify configs.
You do not modify workloads.
You do not decide final optimizations.

Read:
- configs/best.yaml
- workloads/current.yaml
- experiments/summary.md
- experiments/LESSONS.md

Run:
- scripts/profile_server.py if available

Write:
- experiments/profile.md

Include:
- whether workload appears prefill-heavy or decode-heavy
- whether scheduler/HTTP overhead appears significant
- whether GPU appears underutilized
- whether OOM or KV pressure appears
- suggested next config knobs
```

Prompt:

```text
prompts/profile_once.md
```

```md
Run one profiling analysis for current best SGLang config.

Read configs/best.yaml and workloads/current.yaml.

Run scripts/profile_server.py if implemented.

Write experiments/profile.md.

Do not modify configs.
Do not modify workloads.
Do not optimize.
```

---

## 22. Research agent

创建文件：

```text
.github/agents/research.agent.md
```

内容：

```md
---
name: research
description: Diagnoses optimization plateaus from clean context and proposes the next experiment.
tools:
  - view
  - create
---

You diagnose optimization plateaus.

You do not modify configs.
You do not run benchmark.
You do not edit source code.

Read:
- AGENTS.md
- configs/search_space.yaml
- configs/best.yaml
- workloads/current.yaml
- experiments/summary.md
- experiments/LESSONS.md
- experiments/workload_profile.md if present
- experiments/profile.md if present

Write:
- experiments/research_plan.md

Diagnose:
- repeated knob loop
- wrong primary metric
- wrong workload regime
- benchmark noise
- too-small workload
- too-large workload
- OOM / KV pressure
- scheduler bottleneck
- prefill bottleneck
- decode bottleneck
- prefix cache issue

Output:
- one recommended next experiment
- knob
- old value
- new value
- hypothesis
- risk
- why previous attempts failed
```

---

## 23. Common failure handling

### 23.1 Server fails to start

Action:

1. Mark experiment failed.
2. Copy server log.
3. Detect OOM if possible.
4. Revert candidate.
5. Update summary.

### 23.2 Benchmark times out

Action:

1. Kill server process group.
2. Mark timeout.
3. Revert candidate.
4. If repeated, reduce workload size only after human approval.

### 23.3 Metrics parser fails

Action:

1. Do not accept candidate.
2. Mark parse failure.
3. Preserve raw logs.
4. Fix parser only if current task is harness implementation, not optimization.

### 23.4 Candidate improves throughput but hurts latency

Action:

1. Use workload primary metric.
2. If primary metric improves and secondary metric regression is not catastrophic, keep.
3. If p95 latency regresses by more than 25%, require A/B or revert.
4. Log tradeoff.

### 23.5 Candidate causes OOM

Action:

1. Reject.
2. Add lesson if knob consistently causes OOM.
3. Prefer lower `mem-fraction-static`, lower `max-running-requests`, or smaller `chunked-prefill-size` in future.

---

## 24. Metrics policy

### 24.1 Required metrics

Every benchmark result must contain:

```text
passed
successful_requests
failed_requests
ttft_p50_ms
ttft_p95_ms
tpot_p50_ms
tpot_p95_ms
itl_p50_ms
itl_p95_ms
output_tokens_per_second
request_throughput
server_crash
oom
timeout
```

### 24.2 Optional metrics

Useful if available:

```text
gpu_memory_used_mb
gpu_utilization_avg
gpu_utilization_max
kv_cache_usage
queue_req
token_usage
server_startup_seconds
```

### 24.3 Metric direction

```yaml
ttft_p50_ms: lower
ttft_p95_ms: lower
tpot_p50_ms: lower
tpot_p95_ms: lower
itl_p50_ms: lower
itl_p95_ms: lower
total_latency_p50_ms: lower
total_latency_p95_ms: lower
output_tokens_per_second: higher
request_throughput: higher
successful_requests: higher
failed_requests: lower
```

---

## 25. First MVP success criteria

The MVP is successful if:

1. `python scripts/run_experiment.py --mode quick ...` works.
2. `python scripts/run_experiment.py --mode medium ...` works.
3. `python scripts/ab_benchmark.py ...` works.
4. Copilot can run one iteration and produce `experiments/exp_001/result.md`.
5. Copilot changes only one knob.
6. Copilot updates `experiments/summary.md`.
7. Candidate is either accepted into `configs/best.yaml` or reverted.
8. No manual intervention is required within one iteration.
9. A `STOP` file stops the loop.
10. All logs are preserved.

---

## 26. Suggested first experiment sequence

Before turning on fully autonomous loop, manually try:

### Experiment 1

Workload:

```text
short_in_short_out
```

Change:

```text
cuda-graph-max-bs: 16 → 8
```

### Experiment 2

Workload:

```text
short_in_short_out
```

Change:

```text
max-running-requests: 16 → 8
```

### Experiment 3

Workload:

```text
long_in_short_out
```

Change:

```text
chunked-prefill-size: -1 → 4096
```

### Experiment 4

Workload:

```text
short_in_long_out
```

Change:

```text
num-continuous-decode-steps: 1 → 2
```

### Experiment 5

Workload:

```text
prefix_reuse
```

Change:

```text
schedule-policy: fcfs → lpm
```

---

## 27. Final instruction to Copilot

When Copilot is asked to implement this project, it should:

```text
1. Implement the harness first.
2. Do not start autonomous optimization until manual quick benchmark passes.
3. Prefer simple robust scripts over clever abstractions.
4. Keep all outputs machine-readable and human-readable.
5. Make every experiment reproducible.
6. Never hide failed runs.
7. Never accept a candidate without measured metrics.
8. Stop after one optimization iteration.
```

End of design document.

[1]: https://docs.github.com/en/copilot/concepts/agents/copilot-cli/autopilot "Allowing GitHub Copilot CLI to work autonomously - GitHub Docs"
[2]: https://docs.github.com/copilot/concepts/agents/about-copilot-cli "About GitHub Copilot CLI - GitHub Docs"
[3]: https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md "sglang/docs/developer_guide/benchmark_and_profiling.md at main · sgl-project/sglang · GitHub"
[4]: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md "sglang/docs/advanced_features/server_arguments.md at main · sgl-project/sglang · GitHub"
