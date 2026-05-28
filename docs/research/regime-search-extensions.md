# Regime Search & Input Generation — Strengthening Strategy

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#-中文版)
>
> **Purpose**: a research/engineering review of how the current Stage-A
> input construction works, what it cannot find, and concrete proposals
> for strengthening it. Written for someone deciding which extensions
> are worth investing in (engineering vs research value, cost vs payoff).

**Status**: discussion document, not a spec. Not in execution yet.
**Audience**: project author + collaborators who want to push regime
search beyond the v0.4 baseline.

---

## 1. What the current Stage-A actually searches

### 1.1 Seed workloads (`regime_scout/seed_suite.yaml`)

10 hand-written seed workloads for Qwen3-0.6B / dense models, plus a
trimmed 5-seed subset for MoE:

| Seed | mc | input_len | output_len | num_prompts | dataset | regime_hint |
|---|---:|---:|---:|---:|---|---|
| `smoke` | 4 | 128 | 32 | 32 | random | sanity |
| `tiny_latency` | 1 | 8 | 4 | 32 | random | scheduler_overhead |
| `short_in_short_out` | 16 | 128 | 32 | 160 | random | scheduler_or_cuda_graph |
| `scheduler_overhead_high_concurrency` | 64 | 128 | 16 | 320 | random | scheduler_tail |
| `prefill_medium` | 4 | 4096 | 16 | 64 | random | prefill |
| `prefill_long` | 2 | 16384 | 16 | 16 | random | prefill_boundary |
| `decode_medium` | 16 | 128 | 512 | 160 | random | decode |
| `decode_heavy` | 32 | 128 | 1024 | 160 | random | decode_saturation |
| `prefix_reuse_ideal` | 16 | gsp 4096+128 | 128 | 256 | gsp | prefix_cache |
| `prefix_churn` | 16 | gsp 4096+128 | 128 | 512 | gsp | cache_churn |

Every seed (a) generates fixed-length requests (`random_range_ratio=0.0`),
(b) uses closed-loop infinite arrival (`request_rate=null`), (c) is
single-modal (all requests of one shape), (d) is randomly sampled
synthetic text — no real conversation pattern.

### 1.2 Search axes covered by boundary expansion

`.github/skills/boundary-expansion/impl/expand.py` maps to YAML fields:

```python
AXIS_TO_FIELD = {
    "max_concurrency": ("traffic", "max_concurrency"),
    "num_prompts":     ("traffic", "num_prompts"),
    "input_len":       ("dataset", "random_input_len"),
    "output_len":      ("dataset", "random_output_len"),
}
```

That's **four axes**, all one-dimensional. The four expansion strategies
(`bracket / upward / downward / geometric`) all act on **one axis at a
time** — there is no joint exploration of (input_len × max_concurrency)
or any other pair.

### 1.3 Triage (`rule_based_explore.py::triage`)

The whole adaptive component of regime search is four `if/elif` rules:

```
if  concurrency_capped  or  cuda_graph_too_small  → bracket(max_concurrency)
if  at_capacity         or  near_capacity         → upward(input_len)
if  lonely_cluster      AND score >= 0.1          → bracket(hint_natural_axis)
else                                              → no expansion
```

Note the limits:

- All four rules expand a **single axis** chosen mechanically.
- "Lonely cluster" rule maps `regime_hint → "natural axis"` via a
  hardcoded table (decode→output_len, prefill→input_len, …).
- No use of the `try_first / avoid_initially` priors from DESIGN §15
  (we wrote that table but never wired it into the explore loop).
- No cross-hint reasoning (e.g. "if `prefill_long` is hot, maybe try a
  `prefix_reuse` variant").

### 1.4 What the search loop actually does in a session

```
wave 0   run 10 seeds (one workload each, single-axis, fixed length, closed loop)
score    rule-based composition of 5 components
triage   apply 4 single-axis rules → at most one expansion plan per seed
wave 1   run the expanded neighbors (typically 2–4 per plan)
re-score with neighbors now present (enables local_nonlinearity)
select   threshold + 1-per-cluster → frozen problem packages
stop
```

That's the entirety of v0.4 Stage-A. Effective wall time on H200 +
0.6B: ~14 min. Number of distinct workloads explored: typically
10 + 2-6 = **12-16 per session**.

---

## 2. What this cannot find

### 2.1 Axes / dimensions not covered today

| Missing axis | Why it matters |
|---|---|
| `request_rate` (Poisson arrival) | Open-loop arrival exposes scheduler queueing + admission control under realistic burstiness. Closed-loop never queues idle. |
| `random_range_ratio > 0` (variable lengths) | Within-batch padding waste; non-uniform attention shapes hit different kernel tiles. |
| Mixed workloads (short + long concurrent) | Head-of-line blocking; expert routing imbalance in MoE. |
| Multi-turn conversations / evolving prefix | Radix cache stress, eviction policies, prefill-vs-cache-hit ratio. |
| Generation params (temperature, top-k, top-p variance) | Sampling kernel cost varies non-trivially with top-k. |
| Streaming / partial cancellation | Real production cancels requests; tests cleanup paths. |
| LoRA / multi-model serving | Different load patterns + KV layout. |
| Tokenizer pattern stress (long words, code, multilingual) | Embedding / tokenizer hot path. |

### 2.2 Kernel-level problems we'd systematically miss

Kernel-level performance bugs in serving systems usually appear under
specific combinations:

1. **Attention kernel sub-optimal tile**: needs (head_dim × seq_len × batch_size) sweep.
2. **MoE routing imbalance under skewed token distributions**: needs `random_range_ratio > 0` OR non-uniform expert ID distribution.
3. **CUDA graph fallback edges**: dense sweep around the discrete `cuda_graph_bs` set.
4. **Prefix cache thrash under near-MRU access**: needs multi-turn or trace-driven dataset.
5. **Sampling kernel slowdowns at large top_k**: needs varying generation params.
6. **Tokenizer-bound short requests**: needs request_rate > 0 (so tokenizer becomes the critical path, not GPU compute).

Our 10 seeds with fixed-length / closed-loop / random-text inputs hit
none of these reliably. **That is why every problem we've found so far
is config-level (admission cap)** — those are the cliffs visible from
our axis coverage.

### 2.3 The deeper limitation: missing axes interactions

Even within the 4 axes we DO cover, we only explore them one-at-a-time.
But the genuinely interesting cliffs in production usually require
**axes interactions**:

- *Long input* alone: fine.
- *High concurrency* alone: fine (until admission cap).
- *Long input × high concurrency*: KV cache pressure cliff — invisible from either axis alone.

The same logic applies to any pair. The current `triage()` cannot
construct 2D experiments.

---

## 3. Research directions for a stronger regime search

These are ordered roughly by ascending research value (and engineering
complexity).

### R1. **Axis expansion**: add `request_rate`, `random_range_ratio`, `gsp_num_turns`

Cheap. Pure engineering. Doubles the dimensionality immediately
without changing methodology.

Expected: probably surfaces 1–2 new config-level problems we currently
miss (e.g. arrival-rate-sensitive scheduling). Low research value, high
ROI.

### R2. **Multi-axis boundary expansion**

Generalize `expand.py` to take `--axes input_len,max_concurrency` and
generate a small grid (e.g. log-uniform 3×3) of neighbors. Update
triage to propose 2D expansions when single-axis evidence is ambiguous.

Expected: surfaces axis-interaction cliffs. Some research value (how
do you choose which 2D grid to sample? still mostly engineering).

### R3. **Trace-driven seeds** (cheapest research-relevant step)

`sglang.bench_serving` already supports `--dataset-name mooncake`
(Mooncake conversation / synthetic / toolagent traces) — we just never
used it. Add a seed `mooncake_conversation` that drives the server
with real conversational traffic.

Expected: realistic prefix-reuse + variable length + Poisson-like
arrival patterns. Likely surfaces problems closer to production. Some
research value (how do you ablate a trace-driven cliff to find root
cause?).

### R4. **Compositional / mixed workload generation**

Real services serve mixed traffic. Write a `compose_workload.py`
that interleaves N seeds at chosen mix ratios (e.g. 80%
short_in_short_out + 20% prefill_long, fired concurrently). Likely
exposes head-of-line blocking, expert routing imbalance on MoE,
preemption / retraction logic.

Real research value: there's almost no literature on compositional
regime search for serving systems specifically.

### R5. **Cliff-aware active search (Bayesian Optimization)**

Treat cliff discovery as Bayesian Optimization:
- Surrogate: GP or random forest over `(input_axes, config) → primary_metric`.
- Acquisition function: **expected information gain about cliff
  location** (NOT minimization of metric — we want to *find* cliffs,
  not avoid them).

Practical research contribution: most BO literature is about
minimizing a smooth function. Cliff discovery is the opposite — you
want to find discontinuities. Acquisition function design becomes the
interesting research question.

### R6. **Profile-guided workload generation (closed loop)**

After Stage A finds a cliff, run L4 profile (deferred). Use the top-3
profile kernels to GENERATE the next round of seed workloads designed
to stress those specific kernels harder. Iterate.

Research value: this is "co-evolution of workload and system
understanding". Could lead to a paper showing "adaptive workload
generation finds N% more cliffs than fixed seeds with the same
budget".

### R7. **LLM-driven adversarial workload generation**

Given a model arch + sglang config + known kernel characteristics,
ask an LLM to design a workload most likely to break the system.
Compare against R5 (BO) and random as baselines.

Research value: tests whether LLM domain priors actually help in
performance regression discovery. The auto-gpu-kernel report claimed
LLMs are weak at autotuning kernels themselves but might be strong at
*designing inputs* — this is the experimental verification.

### R8. **Causal cliff attribution**

When Stage A finds a cliff, the v0.4 evidence is correlational ("this
metric is high"). Attribution would automatically ablate each knob in
isolation and report which knob change eliminates the cliff. Connects
the regime search loop directly to the config-space search.

Research value: clean causal-inference framing for serving
performance.

### R9. **Cross-model regime transfer learning**

After running Stage A on enough (model, hardware, regime) tuples,
train a small meta-model: given an unseen model arch + hardware,
predict which 3 seeds are most likely to surface cliffs. Skip the
rest.

Research value: transfer learning for performance regression
discovery. Production relevance: each scout session today is ~10 min
on a small model and could become >> 1 h on production-size 70B
models — meta-prediction is the path to scalability.

---

## 4. What to do first (engineering pragmatics)

To find a kernel-level problem, the **minimum** intervention is probably:

1. **R1** (+`request_rate`, `+random_range_ratio`) — half a day
2. **R3** (+mooncake trace seed) — a few hours
3. **R2** (2D expansion) — half a day

Together: ~1.5 days. After this we have realistic-flavored input
generation AND axis interactions. If config-level problems still
dominate after that, we know it's NOT just an input weakness, and the
case for kernel-agent (or R5/R6/R7) is much stronger.

Beyond engineering, the **research-grade** directions are R5
(Bayesian Optimization for cliff discovery), R6 (profile-guided
generation), and R7 (LLM-driven adversarial). Of these, R6 is the
most natural fit for the existing architecture — it's literally a new
skill that consumes profile output and updates `seed_suite.yaml`.

---

## 5. Open methodological questions

These are research-worthy in their own right:

1. **What's the right metric for "a regime is interesting"?** Today
   we use suspicion score (composite). Are cliffs strictly worse than
   smooth tails? What about regimes where TPOT is fine but
   tail-of-tail (p99.9) explodes?
2. **How do you avoid false-positive cliffs from noise?** Our v0.4
   relies on a single repeat per workload. Noise baseline calibration
   (5 repeats of smoke) is designed but not wired in. Hard threshold
   vs adaptive (e.g. CV-based) is an empirical question.
3. **When is a 2D / 3D regime "worth packaging" vs being a downstream
   solver experiment?** Where exactly is the line between "the setter
   should produce K problem packages" and "the solver should explore
   K knobs"?
4. **Can regime search benefit from cross-session memory?** Today
   each session is independent. A persistent regime index would let
   the setter say "P017 was the worst on Qwen-30B; for the new Llama-70B
   try its closest analogue first".
5. **How do you quantify regime coverage?** Today we have no metric
   for "did we cover the relevant input space?" — only "we ran the 10
   seeds". A coverage metric would let us trade off seed-count vs
   probe-count vs profile-cost.

---

## 6. Concrete asks for the human

When deciding what to invest in, here's the trade-off matrix as I
see it:

| Investment | Cost | Likely outcome | Research weight |
|---|---|---|---|
| R1 axes | ½ day | +1-2 new config-level cliffs | low |
| R2 2D expansion | ½ day | +1 axis-interaction cliff (maybe) | low |
| R3 mooncake trace | ½ day | realistic cliff distribution | medium |
| R4 compositional | 1 day | head-of-line / routing imbalance cliffs | medium |
| R5 BO for cliffs | 1 week | search efficiency claim | high |
| R6 profile-guided | 1 week | self-improving workload generation | high |
| R7 LLM adversarial | 1 week | empirical "does domain LLM help?" | high |
| R8 causal attribution | 1 week | clean causal claim | medium |
| R9 cross-model meta | 2 weeks | scalability claim | high |

For an early-stage project the productive path is probably:

```
R1 + R3 + R2  (~1.5 days, surface a kernel-level problem if one exists)
         ↓
   If kernel-level problem found:
        → P3 (L4 profile skill) + kernel-agent
   If still all config-level:
        → that's interesting too; means kernels are well-tuned for our cases
         ↓
R5 or R6 (research extension)
```

But this is the human's call.

---
---

# 🇨🇳 中文版

> **目的**：对当前 Stage-A 输入构造方式的研究/工程 review —— 它现在
> 在做什么、它**找不到**什么、以及具体能怎么强化。给在考虑"哪些扩展
> 值得投入"的人看（工程价值 vs 研究价值，成本 vs 回报）。

**状态**：讨论文档，不是 spec。还未执行。
**读者**：项目作者 + 想把 regime search 推到 v0.4 baseline 之外的合作者。

---

## 1. 当前 Stage-A 实际搜的是什么

### 1.1 Seed workloads（`regime_scout/seed_suite.yaml`）

10 个手写 seed workload（Qwen3-0.6B / dense 模型用），加 MoE 用的 5 个
精简子集。

每个 seed 都：(a) 生成定长 request (`random_range_ratio=0.0`)，
(b) 用闭环无限到达 (`request_rate=null`)，(c) 单模态（所有 request 一
种 shape），(d) 随机合成文本 —— 没有真实对话 pattern。

详见英文版表格。

### 1.2 Boundary expansion 覆盖的 axes

`.github/skills/boundary-expansion/impl/expand.py`：

```python
AXIS_TO_FIELD = {
    "max_concurrency": ("traffic", "max_concurrency"),
    "num_prompts":     ("traffic", "num_prompts"),
    "input_len":       ("dataset", "random_input_len"),
    "output_len":      ("dataset", "random_output_len"),
}
```

**四个 axis，都一维**。四种扩展策略（bracket / upward / downward /
geometric）全部**单轴**操作 —— 没有 (input_len × max_concurrency) 这样
的联合搜索。

### 1.3 Triage（`rule_based_explore.py::triage`）

整个 regime search 的自适应部分就是 4 条 `if/elif`：

```
if concurrency_capped or cuda_graph_too_small → bracket(max_concurrency)
if at_capacity        or near_capacity        → upward(input_len)
if lonely_cluster     AND score >= 0.1        → bracket(hint_natural_axis)
else                                          → 不扩展
```

局限：

- 四条都**单轴**机械选择
- "lonely cluster" 规则用一个 hardcoded 表把 `regime_hint → "自然 axis"`
  （decode→output_len, prefill→input_len，...）
- DESIGN §15 里写的 `try_first / avoid_initially` 先验没用上
- 没有跨 hint 推理（例如"`prefill_long` 热，要不要试个 `prefix_reuse`
  变种"）

### 1.4 一次 session 实际跑什么

```
wave 0   跑 10 seed（每个一个 workload，单轴、定长、闭环）
score    rule-based 组合 5 个分量
triage   应用 4 条单轴规则 → 每个 seed 最多一个 expand plan
wave 1   跑扩展邻居（每 plan 通常 2-4 个）
重新打分（现在 local_nonlinearity 可激活了）
select   阈值过滤 + 每 cluster 1 个 → 冻结题目包
停
```

v0.4 Stage-A 的全部。H200 + 0.6B 上有效 wall time ~14 min。每 session
探索的 distinct workload 数：通常 10 + 2-6 = **12-16**。

---

## 2. 这套搜不到什么

### 2.1 没覆盖的 axes / dimensions

| 缺的 axis | 为什么重要 |
|---|---|
| `request_rate`（Poisson 到达） | 开环到达暴露 scheduler 队列 + admission control 在真实 burst 下的行为。闭环永远不空闲。 |
| `random_range_ratio > 0`（变长） | batch 内 padding 浪费；非均匀 attention shape 撞不同 kernel tile。 |
| 混合 workload（短 + 长 并发） | head-of-line blocking；MoE expert routing 不均衡。 |
| 多轮对话 / 演化 prefix | radix cache 压力、eviction 策略、prefill vs cache-hit 比 |
| Generation 参数（temperature, top-k, top-p 变化） | sampling kernel cost 跟 top-k 关系非平凡 |
| 流式 / 部分取消 | 真实生产取消请求；测试清理路径 |
| LoRA / 多模型 serving | 不同负载模式 + KV 布局 |
| Tokenizer 模式压力（长词、代码、多语种） | embedding / tokenizer 热路径 |

### 2.2 kernel-level 问题我们系统性看不到

serving 系统的 kernel-level 性能 bug 通常出现在特定组合下：

1. **Attention kernel sub-optimal tile**：需要扫 (head_dim × seq_len × batch_size)
2. **MoE routing 在偏态 token 分布下不均衡**：需要 `random_range_ratio > 0` 或非均匀 expert ID 分布
3. **CUDA graph fallback 边界**：需要在离散 `cuda_graph_bs` 集合附近密集扫
4. **Prefix cache 在近 MRU 访问下 thrash**：需要多轮或 trace 驱动 dataset
5. **Sampling kernel 在大 top_k 时变慢**：需要变化 generation 参数
6. **Tokenizer-bound 短请求**：需要 `request_rate > 0`（让 tokenizer 成为关键路径，不是 GPU compute）

**我们 10 个 seed 全部 fixed-length / closed-loop / random-text，
none of these 能稳定撞到**。这就是为什么我们到目前找到的所有问题都是
**config-level**（admission cap）—— 那是我们 axis 覆盖能看到的所有
cliff。

### 2.3 更深的局限：缺 axes 交互

即使在我们**覆盖的** 4 个 axis 里，我们也只是一个一个扫。但生产里真正
有意思的 cliff 通常需要**axes 交互**：

- *只长 input*：没事
- *只高并发*：没事（直到 admission cap）
- *长 input × 高并发*：KV cache 压力 cliff —— 单看任一个 axis 都看不到

任何 axis pair 同理。当前 `triage()` 没法构造 2D 实验。

---

## 3. 强化 regime search 的研究方向

按研究价值（和工程复杂度）大致升序：

### R1. **Axis 扩展**：加 `request_rate`、`random_range_ratio`、`gsp_num_turns`

便宜。纯工程。立刻把维度翻倍，不改方法论。

预期：很可能surface 1-2 个目前看不到的 config-level 问题（例如 arrival-rate-sensitive 调度）。研究价值低，ROI 高。

### R2. **多轴 boundary expansion**

把 `expand.py` 泛化成接受 `--axes input_len,max_concurrency` 并生成
小 grid（例如 log-uniform 3×3）邻居。triage 在单轴证据不明确时提议
2D expansion。

预期：surface axis-interaction cliff。有一定研究价值（怎么选 2D
grid？仍然主要是工程）。

### R3. **Trace 驱动 seed**（最便宜的 research-relevant 步骤）

`sglang.bench_serving` 已经支持 `--dataset-name mooncake`（Mooncake
对话/合成/toolagent trace）—— 我们就是没用。加一个 `mooncake_conversation`
seed，用真实对话流量驱动服务器。

预期：真实 prefix-reuse + 变长 + 类 Poisson 到达模式。很可能 surface
更接近生产的问题。有研究价值（怎么 ablate trace 驱动的 cliff 找根因？）。

### R4. **Compositional / 混合 workload 生成**

真实服务跑混合流量。写 `compose_workload.py` 按比例交错 N 个 seed
（例如 80% short_in_short_out + 20% prefill_long 并发发出）。很可能
暴露 head-of-line blocking、MoE 上的 expert routing 不均衡、抢占/
retraction 逻辑。

真实研究价值：serving 系统的 compositional regime search 几乎没文献。

### R5. **Cliff-aware active search (Bayesian Optimization)**

把 cliff 发现当 Bayesian Optimization 问题：
- 代理模型：GP 或 random forest 学 `(input_axes, config) → primary_metric`
- Acquisition function：**关于 cliff 位置的 expected information gain**
  （不是最小化 metric —— 我们要**找** cliff，不是避开）

实际研究贡献：大多数 BO 文献是关于最小化平滑函数。Cliff 发现是相反的
—— 你要找的是 discontinuity。Acquisition function 设计本身就是有意思
的研究问题。

### R6. **Profile-guided workload generation（闭环）**

Stage A 找到 cliff 后跑 L4 profile（推迟的）。用 top-3 profile kernel
去**生成**下一轮 seed workload，专门 stress 那些 kernel，迭代。

研究价值："workload 和系统理解的协同进化"。可能产出一篇 paper：
"adaptive workload generation finds N% more cliffs than fixed seeds
with the same budget"。

### R7. **LLM 驱动的 adversarial workload 生成**

给定 model arch + sglang config + 已知 kernel 特征，让 LLM 设计一个
最可能破坏系统的 workload。跟 R5 (BO) 和 random 做 baseline 对比。

研究价值：测试 LLM domain 先验在性能回归发现里是否真的有用。
auto-gpu-kernel 报告声称 LLM 自己 autotune kernel 弱但**设计输入**
可能强 —— 这是实验验证。

### R8. **因果 cliff attribution**

Stage A 找到 cliff 时，v0.4 的证据是**相关的**（"这个 metric 高"）。
Attribution 会自动孤立地 ablate 每个 knob，报告哪个 knob 改动能消除
cliff。把 regime search 循环直接连到 config-space 搜索。

研究价值：serving 性能的 clean causal-inference framing。

### R9. **跨模型 regime transfer learning**

在足够多 (model, hardware, regime) 组合上跑过 Stage A 之后，训一个
小 meta-model：给新模型 + 硬件，预测哪 3 个 seed 最可能 surface
cliff。其它跳过。

研究价值：性能回归发现的迁移学习。生产相关性：每次 scout session 今天
在小模型上 ~10 min，可能在生产规模 70B 模型上 >> 1h —— meta 预测是
scalability 的路径。

---

## 4. 先做什么（工程务实视角）

要找 kernel-level 问题，**最小**干预可能是：

1. **R1**（加 `request_rate`、`random_range_ratio`）—— 半天
2. **R3**（加 mooncake trace seed）—— 几小时
3. **R2**（2D expansion）—— 半天

合计 ~1.5 天。之后我们有"接近真实"的 input 生成 + axis 交互。
如果做完之后 config-level 问题还是主导，那我们就**知道**不是 input 弱，
kernel-agent（或 R5/R6/R7）的 case 就强多了。

工程之外，**研究级**方向是 R5（cliff 发现的 BO）、R6（profile 引导
生成）、R7（LLM 驱动 adversarial）。其中 R6 跟现有架构最自然 —— 字面
上就是一个新 skill 消费 profile 输出更新 `seed_suite.yaml`。

---

## 5. 开放的方法论问题

这些本身就有研究价值：

1. **"一个 regime 有意思"的正确度量是什么？** 今天用 suspicion score
   （复合）。Cliff 一定比平滑长尾更糟糕吗？TPOT 正常但 tail-of-tail
   (p99.9) 爆炸的 regime 算什么？
2. **怎么避免噪声引起的假阳 cliff？** v0.4 每个 workload 只跑一次。
   Noise baseline 校准（smoke 重复 5 次）设计了但没连进去。硬阈值
   vs adaptive（如 CV-based）是经验问题。
3. **2D / 3D regime 什么时候"值得打包"vs 是 downstream solver 实验？**
   "setter 应产 K 个 problem 包" 和 "solver 应探索 K 个 knob" 的界限
   在哪？
4. **regime search 能不能受益于跨 session 记忆？** 今天每 session 独立。
   持久 regime 索引能让 setter 说"P017 是 Qwen-30B 上最差的；新的
   Llama-70B 先试它最近邻"。
5. **怎么量化 regime 覆盖率？** 今天没有"我们覆盖了相关 input 空间吗"
   的 metric —— 只有"我们跑了 10 个 seed"。覆盖度 metric 让我们能权衡
   seed-count vs probe-count vs profile-cost。

---

## 6. 给人的具体决策矩阵

权衡矩阵（如我所见）：

| 投资 | 成本 | 可能结果 | 研究权重 |
|---|---|---|---|
| R1 加 axes | ½ 天 | +1-2 个新 config-level cliff | 低 |
| R2 2D expansion | ½ 天 | +1 个 axis 交互 cliff（可能） | 低 |
| R3 mooncake trace | ½ 天 | 真实 cliff 分布 | 中 |
| R4 compositional | 1 天 | head-of-line / 路由不均衡 cliff | 中 |
| R5 cliff BO | 1 周 | 搜索效率论断 | 高 |
| R6 profile-guided | 1 周 | 自改进 workload 生成 | 高 |
| R7 LLM adversarial | 1 周 | 经验性"domain LLM 有用吗" | 高 |
| R8 因果 attribution | 1 周 | clean causal claim | 中 |
| R9 跨模型 meta | 2 周 | scalability claim | 高 |

早期项目高产出路径可能是：

```
R1 + R3 + R2  (~1.5 天, surface kernel-level 问题——如果存在)
         ↓
   如果找到 kernel-level 问题:
        → P3 (L4 profile skill) + kernel-agent
   如果还都是 config-level:
        → 这本身也有信息: 说明 kernel 对我们的 case 调得好
         ↓
R5 或 R6（研究扩展）
```

最后由人决定。
