# v8 实验报告：真实负载上的 knob tuning（chunked-prefill × max-running-requests）

**日期**：2026-07-09
**执行**：双卡并行 —— GPU 1（LFM2.5）+ GPU 2（Qwen3-30B）
**脚本**：`scripts/run_v8_tuning_sweep.py`
**产出**：`results/2026-07-09_v8_tuning/`、`results/consolidated_v8_tuning.csv`
**用时**：LFM 42 min / Qwen3 68 min（并行），48 个 run（2 模型 × 12 config × 2 数据集），零失败

---

## 1. 目的

v7 发现"合成 regime 调出的 config 不迁移到真实负载"，且 `chunked-prefill-size` 像是关键 knob。本轮**在真实负载上正式 tuning**：网格扫 `chunked-prefill-size × max-running-requests`，找真实最优，并搞清哪个 knob 真正起作用。

## 2. 设置

- **网格**：`chunked-prefill-size ∈ {2048, 4096, 8192, 16384}` × `max-running-requests ∈ {32, 64, 128}` = 12 config/模型。
- MoE backend 固定 `triton`（v7 赢家），schedule 固定 `lpm`。
- **客户端固定提供负载 `--max-concurrency 128`**：所有 config 看到相同的 128 in-flight 提供负载，这样 server 的 `max-running-requests` 才是真正被扫的限流变量。
- 数据集：toolagent（2000 请求）+ shared_prefix（1024 请求）。

---

## 3. 结果（每格 = out tok/s 吞吐；括号内 median TTFT）

### LFM2.5-8B-A1B

**toolagent**（out tok/s）：
| chunked＼cap | 32 | 64 | 128 |
|---|---|---|---|
| 2048 | 2793 | 3424 | 4144 |
| 4096 | 2987 | 3651 | **4601** |
| 8192 | 3012 | 3719 | 4426 |
| 16384 | 2815 | 3479 | 4150 |

**shared_prefix**（out tok/s）：
| chunked＼cap | 32 | 64 | 128 |
|---|---|---|---|
| 2048 | 4770 | 6570 | **9016** |
| 4096 | 4715 | 6497 | 8729 |
| 8192 | 4590 | 6302 | 8349 |
| 16384 | 3551 | 4449 | 5640 |

### Qwen3-30B-A3B (bf16)

**toolagent**（out tok/s）：
| chunked＼cap | 32 | 64 | 128 |
|---|---|---|---|
| 2048 | 1305 | 1699 | 2057 |
| 4096 | 1361 | 1763 | 2130 |
| 8192 | 1405 | 1828 | 2171 |
| 16384 | 1413 | 1832 | **2181** |

**shared_prefix**（out tok/s）：
| chunked＼cap | 32 | 64 | 128 |
|---|---|---|---|
| 2048 | 3554 | 4680 | 5304 |
| 4096 | 3607 | 4782 | 5525 |
| 8192 | 3650 | 4825 | 5540 |
| 16384 | 3661 | 4893 | **5569** |

---

## 4. 结论

1. **主导 knob 是 `max-running-requests`（并发上限），不是 chunked-prefill-size。**
   - cap 32 → 128 带来的吞吐增益：LFM toolagent **+48%**（2987→4601）、LFM shared_prefix **+89%**（4770→9016）、Qwen3 toolagent **+54%**、Qwen3 shared_prefix **+52%**。
   - 而在固定 cap 下，改 chunked-prefill 的影响 **< 10%**。
   - **→ 这修正了 v7 的结论**：v7 说"chunked8192 是赢家"，但那是因为 v7 所有 config 都固定 cap=32；一旦放开 cap，`max-running-requests=128` 才是真正的大杠杆。cookbook/v7 的 cap=32 一直在瓶颈上。

2. **chunked-prefill 是次要 knob，且两模型偏好不同**：
   - LFM2.5 偏好 **2048–4096**（16384 明显变差，尤其 shared_prefix 掉 30%+）。
   - Qwen3-30B 偏好 **8192–16384**（但和 8192 差距很小）。

3. **吞吐 vs 延迟的权衡**：cap=128 吞吐最高，但 median TPOT 也升高（LFM toolagent cap32 ~10ms → cap128 ~27ms；Qwen3 ~21ms → ~53ms），因为更多请求并发 decode。不过 **cap=128 的尾延迟 p99 TTFT 反而大幅改善**（Qwen3 toolagent cap32 的 p99 TTFT 高达 **154s**——大 prefill 头阻塞；cap128 降到 1.7–3.8s）。所以除了 median TPOT，cap=128 在吞吐和尾延迟上全面更好。

4. **真实最优 config**：
   | 模型 | 负载 | 最优 config | out tok/s | req/s | TTFT中位 |
   |---|---|---|---|---|---|
   | LFM2.5 | toolagent | **chunk4096_cap128** | 4601 | 24.75 | 104ms |
   | LFM2.5 | shared_prefix | **chunk2048_cap128** | 9016 | 35.22 | 657ms |
   | Qwen3-30B | toolagent | **chunk16384_cap128** | 2181 | 11.74 | 107ms |
   | Qwen3-30B | shared_prefix | **chunk16384_cap128** | 5569 | 21.76 | 827ms |

   **统一建议**：两个模型都用 `max-running-requests=128`；chunked-prefill LFM 用 2048–4096、Qwen3 用 8192–16384。

---

## 5. cap 天花板验证（cap 128 vs 192 vs 256）

在各自最优 chunked 下把 cap 扩到 192/256（双卡并行，LFM 5.5min / Qwen3 9.7min）：

| 模型 × 负载 | cap128 | cap192 | cap256 | 实际峰值并发 |
|---|---|---|---|---|
| LFM toolagent | 4601 | 4686 | 4676 | ~170（受负载/KV 限，非 cap） |
| LFM shared_prefix | 8729 | 8796 | 8870 | 256 |
| Qwen3 toolagent | 2181 | 2162 | 2100 | ~156 |
| Qwen3 shared_prefix | 5569 | 5630 | 5499 | 256 |

**结论：cap=128 就是拐点。** 128→192→256 吞吐变化只有 ±1-2%，Qwen3 toolagent 在 cap256 反而掉到 2100。原因：toolagent 真实并发被 workload/KV 约束在 ~155-170，设更大的 cap 也吃不满。**→ 最优 config 敲定为 cap=128**（再大无收益、还占更多 KV 显存风险）。

## 6. 下一步

1. **cap 天花板已确认（§5）**：cap=128 是拐点，无需再往上扫。
2. **回填 NCU**：用真实最优 config（cap128 + 各自 chunked）+ toolagent 代表输入，profile prefill 段 kernel 瓶颈（v6 口径）。这次 tuning 已经定了该 profile 哪个 config。
3. **方法论**：v7+v8 连起来是一个完整的"合成 regime 调参会误导、必须在真实负载上 tune、且要扫对 knob（cap 而非 chunked）"的实证链条，支撑 Chendi 框架。

---

## 附：产物
- `results/2026-07-09_v8_tuning/<model>/chunk<N>_cap<M>/<dataset>/` — 48 组原始结果
- `results/consolidated_v8_tuning.csv` — 48 行汇总
- `scripts/run_v8_tuning_sweep.py` — 参数化编排（每卡一个模型，并行）
