# LLM + Autotuner: 迭代式 sglang 推理配置优化 pipeline

**Design doc / project proposal**
Author: t-jialianggu
Date: 2026-07-02
Status: 提议中，待 mentor review
Related work: v1/v2/v3 experiments on LFM2.5-8B-A1B & Qwen3-30B-A3B (`docs/2026-06-25/`, `2026-06-30/`, `2026-07-02/`)

---

## 1. Motivation

### 1.1 Traditional autotuners（Optuna TPE / BoTorch / random search）的能力边界

我们过去 3 周在 sglang 推理配置调优上做的 3 轮实验揭示：**传统 autotuner 只能优化「给定的 search space × 给定的 workload」**。这个前提本身占了整个 tuning 问题 60%+ 的难度，且**必须由人（或 LLM）来做**：

| 决策 | Autotuner 能做 | 需要人/LLM |
|---|---|---|
| 给 288 个配置组合，选一个最优 | ✓ | |
| 决定哪些 flag 值得放进 search space | | ✓ |
| 决定 workload profile（哪些 regime 要测） | | ✓ |
| 诊断 trial 启动失败（env / model / kernel bug） | | ✓ |
| 从多轮结果里学到「chunked-prefill=-1 是坏默认」的**跨 regime 假设** | | ✓ |
| 决定何时停止搜索 | ✗（跑到预算耗尽） | ✓ |
| 把发现转成可交付给 mentor 的 config 建议 | | ✓ |

### 1.2 我们已有的证据：**LLM 手工介入让每一轮实验都比上一轮好**

| 版本 | 关键 LLM 决策 | Optuna best req/s (R_conc) | vs baseline |
|---|---|---|---|
| v1 | 手写 5 flag search space | 14.0 (Qwen3) | +5-9× (但 baseline 是 broken) |
| v2 | +条件化 search space + 兼容性裁剪（fa3 only） | 22.32 (LFM2.5) | -6% ← Optuna 失败 |
| v2 手工 | 诊断 TPE 漏掉 triton MoE, 手工验证 | 23.53 | -1% |
| v3 | +warm-start (LLM 选先验) + long-context regime + MFU | **23.90** on R_conc, **+46-139%** on 长 regime | free lunch |

**每一次进步不是来自「Optuna 变强」，而是来自「LLM 决定 Optuna 该搜什么」**。

### 1.3 但这个 LLM 介入过程目前是**手工、非结构化、无法复用**

- v2 → v3 花了我一整天，每步都要读日志、grep 源码、拍脑袋写 warm-start
- 结论正确但**过程不可复现**（下一个模型换 Qwen3 时，我要从头再来）
- 中间的决策链**没有被记录**（LLM 输出以对话形式散在 chat log 里）

---

## 2. Proposal

**把过去手工做的 LLM 元决策自动化，做成一个 iterative pipeline：**

```
         ┌────────────────────────────────────┐
         │  Round N: Optuna 跑 N trials       │
         │  （harness/autotune_v3_lfm.py）     │
         └────────────────┬───────────────────┘
                          ▼
         ┌────────────────────────────────────┐
         │  Structured Summary Generator      │
         │  study.db + csv → round_input.yaml │
         │  (500 tokens 摘要)                 │
         └────────────────┬───────────────────┘
                          ▼
         ┌────────────────────────────────────┐
         │  LLM Analysis Agent (Claude/GPT)   │
         │  Input:                            │
         │    - round_input.yaml              │
         │    - sglang server_args reference  │
         │    - 上轮 plan                     │
         │  Output (structured YAML):         │
         │    - new_search_space.yaml         │
         │    - warm_start.yaml               │
         │    - regime_changes.yaml           │
         │    - convergence_verdict           │
         │    - human_review_flags            │
         └────────────────┬───────────────────┘
                          ▼
         ┌────────────────────────────────────┐
         │  User Review (optional)            │
         │  接受 / 修改 / 拒绝                │
         └────────────────┬───────────────────┘
                          ▼
         ┌────────────────────────────────────┐
         │  Round N+1: 用新 plan 再跑 Optuna  │
         └────────────────────────────────────┘
                          │
                          └────► 循环，直到 LLM 说 stop
```

### 2.1 关键设计决策

1. **LLM 只做元决策，不做 sampling**。sampling 是 Optuna 的活。这样：
   - LLM 每轮只被调用 **1 次**（成本可控，1000-5000 tokens）
   - Optuna 数学性质保留（不引入 LLM 采样的非确定性）

2. **喂给 LLM 的是结构化摘要，不是 raw log**：
   ```yaml
   round_input:
     best_all_rounds: 22.32
     baseline: 23.74
     flag_effect:
       moe_runner_backend:
         triton: {n=5, mean=5.5, std=3.2, best=8.4}
         flashinfer_cutlass: {n=20, mean=19.1, std=5.4}
     failed_trials: [{pattern: "cannot find -lcuda", count: 4}]
     workload_coverage:
       tested_prompt_lens_tokens: [130, 260, 1000, 5200]
       model_max_context_tokens: 128000
     hw_ceiling:
       R_conc_MBU: 12.7%
       memory_bound: true
   ```
   这是 500 tokens 而不是 1M tokens 的 log。

3. **LLM 输出是可执行 YAML，不是自然语言**：
   ```yaml
   round_output:
     search_space_changes:
       add:
         - {name: "moe-runner-backend", values: [triton, flashinfer_cutlass, auto]}
       prune:
         - {name: "schedule-policy", reason: "std <1% across trials"}
     warm_start_trials:
       - {name: "triton_good_batching", flags: {...}}
       - {name: "flashinfer_reproduce", flags: {...}}
     regime_changes:
       add:
         - id: R_prompt_16k_c2_out128
           rationale: "model supports 128k, we test max 5k"
     convergence_verdict:
       stop: false
       reason: "6% gap from suspected optimum"
   ```
   这就直接可以 diff / 存档 / 复现。

4. **强制 tool-use 防幻觉**：LLM 建议的每个 flag 名必须在一个白名单里（`grep server_args.py` 生成）。运行时校验，非法直接拒收。

### 2.2 用户体验（提议的 CLI）

```bash
# 起一个 5-round pipeline，每轮 20 trial
python -m harness.iterative_agent \
    --model /data/hf/LFM2.5-8B-A1B \
    --hw configs/hardware/h200.yaml \
    --gpu 6 \
    --max-rounds 5 --trials-per-round 20 \
    --initial-search-space search_spaces/lfm2.5_seed.yaml \
    --llm-model claude-opus-4.7 \
    --auto-approve   # 不需要人 review LLM 输出

# 每轮结束打印：
# Round 3 done. LLM plan: add chunked-prefill knob, prune schedule-policy.
# Best so far: 23.87 req/s (was 22.32). Continuing to Round 4.
```

---

## 3. Experimental Validation Plan

### 3.1 MVP validation：LLM agent 能否自己走到 v3 结论？

**Setup**: 把 v2 (2026-06-30) 的 25 trial 结果作为 round 1 输入。用 LLM agent 生成 round 2 plan。看 LLM 是否会**自己**推荐：
- Warm-start with `triton MoE + good batching`（v2 我们手工加的）
- Add long-context regimes（Dey 建议的）
- 加 `moe-runner-backend=auto` 到候选

**Success criterion**: LLM 的 round 2 plan 覆盖 ≥ 2/3 上述决策。

**Cost**: 半天写代码 + 30 分钟跑

### 3.2 A/B validation：LLM+autotuner vs 纯 autotuner，同样 60 trial 预算

- 组 A：纯 Optuna TPE，一次跑 60 trial（我们 v3 已经有 30 trial，可以扩到 60 复用）
- 组 B：LLM+autotuner，3 轮 × 20 trial，每轮之间 LLM 改 plan

**Metrics**:
- Best req/s 在 target regime
- Best "均衡 config"（所有 regime 加权和）
- 死 trial 数（LLM 应该几乎为 0）
- 找到最优所用的 trial 数（earliest reach time）

**Success criterion**: 组 B best ≥ 组 A best + 至少 3 regime 全面覆盖 vs 组 A 单 regime 优化。

**Cost**: 2 天（60 trial 时间 + 分析）

### 3.3 Transfer validation：跨模型泛化

用 LFM2.5 上训练好的 pipeline，直接扔给 Qwen3-30B-A3B（v1 用过的模型）。看：
- LLM 能否识别不同模型需要不同 search space（比如 Qwen3 有 fp8 版本，可以加 quant knob）
- 是否需要人再手工介入

**Success criterion**: 3 轮之内 LLM 能达到或超越 v1 手工调的 Qwen3 baseline (14 req/s)。

**Cost**: 1 天

---

## 4. Risks & mitigations

| 风险 | 说明 | 缓解 |
|---|---|---|
| **LLM 幻觉 flag 名** | 建议 `--fp8-attention-backend` 但 sglang 没这个 flag | tool-use 白名单校验；invalid 直接拒收 |
| **LLM 震荡** | Round 2 加 chunk, Round 3 又砍 | 强制 LLM 引用「上轮说加 chunk 的理由」并显式解释为什么改主意；保留完整历史 |
| **Token 成本爆炸** | 每轮吃 log/summary | 摘要限 1000 tokens；raw log 只在 debug 阶段按需 grep |
| **LLM 过度自信 stop** | 说收敛了但其实没搜够 | `stop=true` 时强制要求 LLM 提供 3 组独立复跑证据 |
| **非确定性** | 同样输入 LLM 输出不同 | temperature=0 + 输出加时间戳存档；每轮的 prompt+output 全部持久化 |
| **Search space 爆炸** | LLM 每轮都加 knob，最后 15 维 | 强制 LLM 至少 prune 一个才能 add 一个（zero-sum） |
| **LLM 不知道自己不知道** | 建议了没证据的 knob 组合 | 每个 warm-start config 都要 LLM 引用「过去哪个 trial 支持这个选择」 |

---

## 5. Related work

- **AutoML meta-learning** (SMAC, Hyperopt): 用过去实验的统计加速新任务。**用 statistics 不用 semantics**，不能读 sglang 源码或诊断错误。
- **GPT-4 for hyperparameter search** (Zhang et al 2024 等): 用 LLM 直接采样。**取代了 sampler**——反而失去了 TPE 的数学性质，且成本高。
- **CompilerGym / AutoCompiler**: RL/LLM 调编译器 flag。架构类似但目标不同（编译器有明确 IR 特征，我们是黑盒 server）。
- **Vertex AI Model Optimizer / Cortex Tune**: 商业 AutoML pipeline。没有 sglang 特化，也不做 LLM 元决策。

**我们的独特点**：LLM 做元决策（search space / regime / warm-start / stop），不做 sampling。对**系统调优场景**（黑盒 server + 大量 categorical flag + workload profile 关键）是 first mover。

---

## 6. Deliverables

如果 mentor 批：

**Phase 1（1-2 周）**: MVP
- `harness/summarize_round.py`：study.db + csv → structured yaml
- `harness/llm_agent.py`：调 LLM + 校验输出
- 改造 `harness/autotune_v3_lfm.py` 接受 YAML-driven search space
- `harness/iterative_agent.py`：主循环 orchestrator
- 跑 MVP validation (§3.1) + A/B validation (§3.2)

**Phase 2（2-4 周）**: 加固
- Tool-use（LLM 可以 grep sglang 源码验证 flag）
- Human-in-the-loop UI（简单 Streamlit review LLM plan）
- Transfer validation (§3.3)
- 论文 draft

**Phase 3（长期）**: 生态
- 打包成 `pip install sglang-tuner-agent`
- Vertica sglang 团队合作，把发现回馈到 cookbook

---

## 7. Open questions（希望 mentor 意见）

1. **LLM 选型**：Claude 3.5 / GPT-4o / o1 / open-source Llama？成本 vs 质量？
2. **用户信任度**：`--auto-approve` vs 每轮强制 review？前者更实用但风险大。
3. **Metric 选择**：只优化 req/s，还是加 MFU、加成本、加 latency SLO？
4. **和 sglang 团队的关系**：我们发现的问题（e.g. cookbook `chunk=-1` 次优）要不要主动提 issue？
5. **Scope 边界**：只调 sglang，还是也考虑 vLLM / TensorRT-LLM？

---

## Appendix A: 具体的 LLM prompt 示例

```
[SYSTEM]
You are an sglang inference tuning expert with access to:
- Full sglang server_args reference (200+ flags with docstrings)
- Hardware peak flops tables (H100/H200/A100/MI300)
- Common failure patterns database

You produce YAML plans, never natural language answers.

Constraints:
- Every flag name in your output MUST exist in the server_args reference
- Every warm-start config MUST include all 7 required flags
- prompt_words in new regimes MUST satisfy: prompt_words * 1.3 < model.max_context_tokens
- Every "prune" recommendation MUST cite evidence from the input summary

[USER]
Round 2 of tuning LFM2.5-8B-A1B on 1× H200. 
Round 1 finished with 25 trials, TPE only, no warm-start.

Input summary:
```yaml
best_trial: {number: 17, value: 22.32, params: {moe: flashinfer_cutlass, ...}}
baseline_cookbook: 23.74
flag_effect:
  moe_runner_backend:
    triton: {n: 5, mean: 5.5, std: 3.2, best: 8.43, worst: 1.70}
    flashinfer_cutlass: {n: 20, mean: 19.1, std: 5.4, best: 22.32}
  disable_cuda_graph:
    True: {n: 8, mean: 4.5}
    False: {n: 17, mean: 21.6}
    ...
failed_trials:
  count: 4
  patterns:
    - {msg: "cannot find -lcuda", trials: [0,2,3,5], hypothesis: env_issue}
workload_coverage:
  tested_prompt_len_tokens: [130, 260, 1000, 5200]
  model_max_position_embeddings: 128000
hw_ceiling_R_concurrent_decode:
  MBU_pct: 12.7
  MFU_pct: 1.9
  memory_bound: true
```

Analyze and produce next-round plan.
```

**Expected output** (verified against v2→v3 手工决策):

```yaml
analysis:
  key_finding: "TPE marginalized triton MoE as bad, but all 5 triton trials 
                paired with cap=8 or cg-off. Suspect interaction effect masked
                by TPE's per-axis KDE. Recommend stratified warm-start."
  confidence: high
  cross_reference: "trials 1,3,4,9 all triton+bad_batching, best 8.43"

search_space_changes:
  add:
    - {name: "moe-runner-backend", values: [triton, flashinfer_cutlass, auto], 
       rationale: "add sglang's own auto choice as reference"}
  prune:
    - {name: "schedule-policy", reason: "lpm vs fcfs: mean diff <1%, std >5%"}

warm_start_trials:
  - {name: "cookbook_ref", 
     flags: {moe: auto, cap: 32, chunk: -1, sched: lpm, mfs: 0.9, cg: false, attn: fa3}}
  - {name: "triton_fair_test", 
     flags: {moe: triton, cap: 32, chunk: -1, sched: lpm, mfs: 0.9, cg: false, attn: fa3}}
  - {name: "flashinfer_verify", 
     flags: {moe: flashinfer_cutlass, cap: 32, chunk: -1, sched: lpm, mfs: 0.9, cg: false, attn: fa3}}

regime_changes:
  add:
    - id: R_prompt_16k_c2
      rationale: "current max=5200 tokens, model supports 128000 -- 96% blind spot"
      prompt_words: 12500
      max_new: 128
      concurrency: 2

convergence_verdict:
  stop: false
  reason: "Best 6% below suspected true optimum; warm-start should close gap"

human_review_flags:
  - "flashinfer env issue (-lcuda) blocks 3 candidates. Consider filing 
     conda env fix separately; workaround: only sample fa3 for attention."
```

**这就是 v2→v3 我们手工做的全部决策**，一个 LLM prompt+output 就能覆盖。

---

## Appendix B: 与 sglang team 的可能协作

如果这个 pipeline 成熟，可以：

1. 每周对 sglang HEAD 跑一次 pipeline，检测 regression
2. 每次 sglang release 前跑一次 pipeline，验证 cookbook 默认没有 sub-optimal
3. 用户提出 issue「我的模型跑不快」，pipeline 自动跑 3 轮 + 输出可复现 config

这可能是**给 sglang 项目的直接价值**（不只是 mentor 项目内部）。
