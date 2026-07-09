# v7 实验报告：用真实 / agent 负载给 regime 做画像（bench_serving）

**日期**：2026-07-09
**执行**：GPU 1（NVIDIA H200，单卡 tp1）
**目的**：用 sglang `bench_serving` 内置的真实/合成 agent 数据集，给两个模型做**负载画像**，提取真实的 (input, output, 并发) 分布，作为后续回填 NCU（v6 方法学）的代表点。
**脚本**：`scripts/run_v7_agentic_bench.py`
**产出**：`results/2026-07-09_v7_agentic/<model>/<dataset>/`、`results/consolidated_v7_agentic.csv`

---

## 1. 为什么做这一轮

v6 的 3 个 regime 都是人工设的、且 **decode 占 95–98%**，覆盖不到真实 agent 负载。这一轮用两个内置数据集看真实分布：

| 数据集 | 命令 | 模拟什么 |
|---|---|---|
| **toolagent** | `--dataset-name mooncake --mooncake-workload toolagent` | 真实 tool-agent trace（Mooncake FAST'25，kvcache-ai 公开），多轮 + 工具调用 |
| **shared_prefix** | `--dataset-name generated-shared-prefix` | 长 system prompt（2048 tok）+ 短 question（128）→ 典型 RAG/agent |

> 注意：这两个走的是 `bench_serving`（多进程 server 路径），**只收 sglang 自己的 serving 指标，不上 NCU**（NCU 只能包单进程 `bench_one_batch`）。

**踩坑修复**：本地 sglang（`bbe9c7eeb`）的 `bench_serving` 对 mooncake 数据集有 pre-existing bug——多轮检测 (`bench_serving.py:2329`) 和 `calculate_metrics` (`:2122`) 都假设 `DatasetRow` 对象，但 mooncake 传的是 dict，报 `AttributeError: 'dict' object has no attribute 'prompt/prompt_len'`。做了两处最小修复：
1. mooncake 时跳过多轮检测（它有自己的时间戳回放生成器）；
2. `calculate_metrics` 在 `input_requests=None` 时回退用 `outputs[i].prompt_len` 统计输入 token。
（改在 `/home/t-jialianggu/work/sglang/python/sglang/bench_serving.py`）

---

## 2. 配置

- Server（两模型一致，单卡）：`--mem-fraction-static 0.85 --chunked-prefill-size -1 --schedule-policy lpm --max-running-requests 32 --context-length 32768 --trust-remote-code`
- toolagent：`--num-prompts 200`
- shared_prefix：`--gsp-num-groups 8 --gsp-prompts-per-group 16 --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 256`（= 128 请求）
- 模型：`LFM2.5-8B-A1B`、`Qwen3-30B-A3B-Instruct-2507`（bf16）

---

## 3. 结果

| 模型 | 数据集 | 完成 | avg input | avg output | **in:out** | req/s | out tok/s | 并发 | TTFT中位 | TPOT中位 |
|---|---|---|---|---|---|---|---|---|---|---|
| LFM2.5 | shared_prefix | 128 | 2365 | 256 | **9.2:1** | 13.97 | 3577 | 80.8 | 3982 ms | 6.30 ms |
| LFM2.5 | toolagent | 200 | 2667 | 207 | **12.9:1** | 5.27 | 1092 | 6.1 | 357 ms | 3.63 ms |
| Qwen3-30B | shared_prefix | 128 | 2296 | 256 | **9.0:1** | 13.47 | 3448 | 84.5 | 4222 ms | 8.10 ms |
| Qwen3-30B | toolagent | 200 | 2700 | 207 | **13.0:1** | 4.65 | 965 | 25.4 | 815 ms | 20.77 ms |

---

## 4. 结论

1. **真实 agent 负载是 prefill 主导，和 v6 的 decode 主导正好相反。** 两个数据集 input:output 都在 **9–13 : 1**（agent 有大 prompt + 短回答），而 v6 的 3 个 regime 是 out ≫ in、decode 占 95%+。**说明我们之前只测 decode 段，漏掉了真实 agent 场景里占大头的 prefill。**

2. **两个数据集刻画了两种不同压力**：
   - **shared_prefix**：高并发（80+）、长共享前缀 → 压 **prefill 吞吐 + prefix cache 复用**。TTFT 高达 4s（大 prefill 排队），TPOT 很低。
   - **toolagent**：中低并发（6–25）、真实到达时间戳 → 更贴近**在线单请求延迟**。TTFT 低（357–815ms）。

3. **两模型趋势一致，但 Qwen3-30B 的 decode 更贵。** 同样 toolagent 负载，Qwen3 的 TPOT 中位 20.8ms vs LFM2.5 的 3.6ms（≈5.7×）——大模型每步 decode 更重，也更容易在高并发下暴露访存瓶颈（呼应 v6 的 `fused_moe_kernel` memory-bound）。

4. **这批分布可直接回填 NCU。** 见 §5。

---

## 5. 下一步：回填到 bench_one_batch + NCU

从画像里提取 3 个代表点，用**和 v6 完全相同的 NCU 口径**（`bench_one_batch --profile-stage {prefill|decode}` + `--profile-from-start off`）逐个 profile：

| 新 regime | 依据（本轮画像） | in / out | 建议 profile 阶段 | 验证 |
|---|---|---|---|---|
| R_agent_prefill | toolagent avg (in≈2700, out≈207) | 2700 / 207 | **prefill**（首次！） | agent prefill 是否 compute-bound、TC 是否打满 |
| R_rag_shared_prefix | shared_prefix (in≈2300, out≈256, 高并发) | 2304 / 256 | prefill + decode 各一次 | 长共享前缀下 attention/MoE kernel 变化 |
| R_agent_decode | toolagent 的 decode 段 | 2700 / 207 | decode | 与 v6 纯 decode 对比，长 KV 下的带宽压力 |

> 一句话：**本轮用 `bench_serving` 拿到了真实 agent 的 input/output 分布（prefill 主导，9–13:1）；下一步把这几个代表点喂给 v6 的 NCU 流程，就能拿到"真实 agent 负载下 prefill 段的 kernel 级瓶颈"——这是 v6 纯 decode 数据完全没覆盖的象限。**

---

## 附：产物

- `results/2026-07-09_v7_agentic/{lfm2.5-8b-a1b,qwen3-30b-a3b-bf16}/{toolagent,shared_prefix}/bench_serving_result.jsonl` + `bench.log`
- `results/consolidated_v7_agentic.csv` — 4 行汇总
- `scripts/run_v7_agentic_bench.py` — 编排脚本
- sglang bench_serving mooncake 修复（在 sglang 工作副本）
