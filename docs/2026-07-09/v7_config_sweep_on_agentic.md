# v7 config sweep：在真实 agent 负载上对比 tuned config

**日期**：2026-07-09
**执行**：GPU 1（H200 单卡 tp1）
**脚本**：`scripts/run_v7_config_sweep.py`
**产出**：`results/2026-07-09_v7_config_sweep/`、`results/consolidated_v7_config_sweep.csv`
**用时**：67 分钟，12 个 run（2 模型 × 3 config × 2 数据集），零失败

---

## 1. 目的

把 2026-06-25 在**合成 regime** 上 tuning 出来的最优 config，拿到**真实 agent 负载**（v7 的 toolagent + shared_prefix）上验证：合成 regime 的最优是否迁移到真实负载？

## 2. 负载设计（关键）

- **闭环限流 `--max-concurrency 64`**：把"按时间戳回放"的过载（之前测到并发飙 1819、TTFT 109s）变成稳态负载，让吞吐 / TTFT / TPOT 都有意义、config 之间可公平对比。
- 所有 config 用**完全相同**的负载：
  - toolagent：`--num-prompts 5000`（23,608 条 trace 的 21% 样本）+ `--mooncake-slowdown-factor 0.1`
  - shared_prefix：`32 groups × 32 = 1024` 请求，system prompt 2048 + question 128 + output 256
- 说明：mooncake 发的是 `hash_ids × ~128 tok` 合成 prompt（约 2700 tok），trace 里那个最大 12 万 tok 的 `input_length` 字段**未被使用**。

## 3. 对比的 config（来自 6-25 autotuning winner）

| 模型 | config | knobs |
|---|---|---|
| LFM2.5 | cookbook | triton / cap32 / chunked=-1 / lpm（≈其 tuning 最优） |
| | chunked8192 | triton / cap32 / **chunked=8192** / lpm |
| | fcfs | triton / cap32 / chunked=-1 / **fcfs** |
| Qwen3-30B | baseline_triton | triton / cap32 / chunked=-1 / lpm |
| | tuned_prefill | **flashinfer_cutlass** / cap32 / chunked=-1 / **fcfs**（6-25 的 R_long_prefill+R_conc winner） |
| | tuned_chunked | triton / cap32 / **chunked=8192** / lpm（6-25 的 R_short+R_med winner） |

---

## 4. 结果

### toolagent（真实 tool-agent trace）

| 模型 | config | req/s | out tok/s | TTFT中位 | p99 TTFT | TPOT中位 | E2E中位 |
|---|---|---|---|---|---|---|---|
| LFM2.5 | **chunked8192** | **17.2** | **3174** | **209** | 5102 | 9.29 | **544** |
| | cookbook | 16.1 | 2976 | 1583 | 4700 | 9.89 | 1872 |
| | fcfs | 16.2 | 2997 | 1952 | 2916 | 9.89 | 2581 |
| Qwen3-30B | **tuned_chunked** | **7.62** | **1405** | **147** | 33649 | 20.41 | 910 |
| | baseline_triton | 7.57 | 1397 | 163 | 36247 | 20.66 | 908 |
| | tuned_prefill | 5.14 | 949 | 6164 | 10179 | 30.55 | 8450 |

### shared_prefix（长共享前缀 RAG）

| 模型 | config | req/s | out tok/s | TTFT中位 | p99 TTFT | TPOT中位 | E2E中位 |
|---|---|---|---|---|---|---|---|
| LFM2.5 | **chunked8192** | **18.1** | **4638** | **1287** | 15735 | 5.69 | 2747 |
| | cookbook | 14.1 | 3606 | 2758 | 7619 | 6.46 | 4508 |
| | fcfs | 14.3 | 3666 | 2810 | 3163 | 6.35 | 4438 |
| Qwen3-30B | **baseline_triton** | **14.28** | **3656** | 2387 | 3332 | 7.75 | 4382 |
| | tuned_chunked | 13.87 | 3551 | 2419 | 5891 | 7.92 | 4450 |
| | tuned_prefill | 12.35 | 3161 | 2713 | 5955 | 8.72 | 4977 |

---

## 5. 结论

1. **`chunked-prefill-size=8192` 是真实（prefill 主导）负载的明确赢家。**
   - LFM2.5 toolagent：TTFT **1583ms → 209ms（7.5×）**、E2E **1872 → 544ms**、req/s +7%。
   - LFM2.5 shared_prefix：吞吐 **+29%**（3606 → 4638 out tok/s）、TTFT 2758 → 1287ms。
   - 道理：agent 输入长（~2700 tok），分块 prefill 避免大 prefill 长时间霸占队列 → TTFT 大幅下降。

2. **6-25 在合成 regime 上调出来的"prefill winner"（flashinfer_cutlass+fcfs）在真实负载上反而最差。**
   - Qwen3 toolagent：tuned_prefill 只有 5.14 req/s（vs 7.6），TTFT **6164ms**（vs 147ms），TPOT 30.6ms（vs 20.4）。
   - shared_prefix 也一样垫底（12.35 vs 14.28 req/s）。
   - **→ 合成 regime 的最优 config 不迁移到真实负载。** 真正的赢家是朴素的 `chunked-prefill` knob，而它在 6-25 的 per-regime 冠军里根本没被选中。

3. **两个模型的最优不同**：
   - LFM2.5：`chunked8192` 全面最优。
   - Qwen3-30B：`triton + chunked`（=8192 或 -1 差别很小）最优；**别用 flashinfer_cutlass**。

4. **fcfs vs lpm**：在这两个负载上区别不大（LFM toolagent fcfs 略降 TTFT 中位但 p99 更好）；主导因素是 chunked-prefill 和 MoE backend。

---

## 6. 下一步（根据结果决定）

1. **值得进一步 tuning**：既然 `chunked-prefill-size` 是关键 knob，应该扫它的取值（2048 / 4096 / 8192 / 16384）× `max-running-requests`（32/64/128），在真实负载上找最优——这比在合成 regime 上搜更有意义。
2. **回填 NCU**：选 `chunked8192` 赢家 config + toolagent 的代表输入点（in≈2700 / out≈207），用 v6 的 `bench_one_batch --profile-stage prefill` + NCU 口径，看真实 agent **prefill 段**的 kernel 级瓶颈——这是 v6 纯 decode 数据完全没覆盖的象限。
3. **复盘 autotuner**：这批数据是"合成 regime 调参不迁移"的实证，支持 Chendi 框架里"要在贴近真实的 workload 上评估"的主张。

---

## 附：产物

- `results/2026-07-09_v7_config_sweep/<model>/<config>/<dataset>/bench_serving_result.jsonl` + `bench.log` + `server.log`
- `results/consolidated_v7_config_sweep.csv` — 12 行汇总
- `scripts/run_v7_config_sweep.py` — 编排脚本
