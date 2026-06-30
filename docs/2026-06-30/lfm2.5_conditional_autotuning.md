# LFM2.5-8B-A1B 条件化搜索空间自动调参实验报告（2026-06-30）

## 0. 摘要

这是把 Chendi 的「conditional search space」框架（6/30 邮件）落地到一个**新模型**（`LiquidAI/LFM2.5-8B-A1B`）上的实验报告。和 6/25 的 Qwen3-30B-A3B 实验相比，本实验：

1. **换了模型**：从 Qwen3-30B-A3B（dense Transformer + MoE）→ LFM2.5-8B-A1B（hybrid: conv layers + full-attention layers + MoE，1B 活跃参数 / 8B 总参数）。
2. **扩展了搜索空间**：从 6/25 的扁平 5 参数（96 组合）→ 7 参数条件化（理论 864 组合，受设备/模型限制后实际 288）。
3. **保留了每一次 trial 的详细记录**：每个 trial 都有自己的 `flags.json` + `summary.json` + 在 `per_trial_log.csv` 里记录的一行总结，包含 7 个旋钮取值 + 全部 4 个 regime 的 req/s + reliable 标记 + wall time + 启动耗时。

**主要结论**：

1. **三次基线（cookbook-default，3 个独立 server lifetime）**: R_concurrent_decode = **23.74 ± 0.12 req/s（0.5% stddev）**。这是「真正稳定的」基线。
2. **Optuna v2（25 trial，条件化搜索空间）找到的"best"** = 22.32 req/s — **比基线低 ~6%。**
3. **后续手工验证发现**：把 Optuna "best" 里的 `moe-runner-backend` 从 `flashinfer_cutlass` 换成 `triton`（保留其他所有 flag 不变），立即得到 **23.53 req/s**——和基线持平。
4. **这是一个教科书级别的 conditional-search-space failure**：TPE 的 7 个早期 trial 里把 `triton MoE` 和 `cap=8`、`disable-cuda-graph=True` 这些**糟糕的 batching 选项**绑在一起。这些 trial 都 < 10 req/s。TPE 据此把整个 `triton MoE` 子空间判定为"差"，后续 18 个 trial 一次都没有再试 `triton + 好 batching`。这正是 Chendi 6/30 邮件里指出的现象：「The search space is conditional, not flat」——`moe-backend × batching` 维度之间有强交互，TPE 的独立坐标轴假设处理不了。
5. **LFM2.5-8B-A1B 在我们环境里的 attention backend 只有 `fa3` 可用**：`triton` 不支持 hybrid 架构（21 conv + 3 full_attention），`flashinfer` 的 JIT 编译被 conda env 的 libcuda 链接问题挡住。这印证了 Chendi 框架的另一条主张：**autotuner 必须先做 (model, hw, env) 兼容性裁剪**，否则就会浪费 trial 在根本起不来的配置上（Phase 1 我们就吃了 4/7 = 57% 的 trial）。

**结论的可操作含义**：

- 对当前 (LFM2.5-8B-A1B × 1× H200 × bf16) 这个三元组，**sglang 团队的开箱默认就是最优** —— 与 6/25 的 Qwen3-30B-A3B 结论一致。
- TPE + 扁平搜索空间会**漏掉**最优解。下一代 autotuner 必须把 backend 选择和 batching 选择**联合建模**（或者按 Chendi 的方案，把 backend 当作"先选 subspace 再搜"的离散变量）。

> 文档位置：`docs/2026-06-30/lfm2.5_conditional_autotuning.md`
> 原始数据：`results/2026-06-30_lfm2.5/`

---

## 1. 实验环境（与 6/25 不同的关键点）

| 项目 | 值 | 说明 |
|---|---|---|
| 模型 | `LiquidAI/LFM2.5-8B-A1B` | hybrid MoE：24 层中 21 层 `conv`、3 层 `full_attention` |
| 存放路径 | `/data/hf/LFM2.5-8B-A1B/` | 没有放到 `/data/hf/models/`，原因见下 §1.1 |
| 总参数 / 活跃参数 | 8B / ~1B（32 experts top-4） | 比 Qwen3-30B-A3B 小一半（30B / 3B） |
| sglang 版本 | v0.5.9 + 本地 autotune-allowlist 补丁 | 同 6/25 |
| GPU | 1× H200 (140GB, SM90) | 同 6/25 |
| Python env | `sglang-dev` | 同 6/25 |
| 日期 | 2026-06-30 | sglang 仓库当天没有新提交（无需 pull） |
| Optuna | 4.9.0 | TPE sampler, seed=2026（同 6/25） |

### 1.1 存放路径的妥协

用户要求把模型下载到 `/data/hf/models/`，但实际目录权限是 `drwxrwxr-x lizhang4:lizhang4`，当前用户 `t-jialianggu` 无写权。`/data/hf/` 本身是 `drwxrwxrwx`，所以模型实际放在 `/data/hf/LFM2.5-8B-A1B/`。如果要符合规范命名，需要 `sudo mv` 到 `/data/hf/models/LFM2.5-8B-A1B/`。

### 1.2 tokenizer 兼容性 hack（重要！）

LFM2.5 模型卡里的 `tokenizer_config.json` 用了 transformers 5.x 新增的 `tokenizer_class: TokenizersBackend` + `backend` / `is_local` 字段。我们 env 里的 transformers 是 4.57.1（sglang 0.5.9 钉死的版本），不认识。

**解决办法**（已落地，**不可逆**）：

- 原文件备份在 `/data/hf/LFM2.5-8B-A1B/tokenizer_config.json.bak`
- 把 `tokenizer_class` 改成 `PreTrainedTokenizerFast`
- 删掉 `backend` 和 `is_local` 两个字段

底层 `tokenizer.json`（HuggingFace tokenizers 格式）没有改动。smoke 测试 + 4-regime benchmark 都通过了 quality gate，但**仍然是一个 hack**：如果重下模型需要再 patch 一次。

---

## 2. Chendi conditional search space 框架在本实验里的落地

Chendi 6/30 邮件的核心论点：**sglang 的可调参数不是一个扁平的搜索空间，而是按 `subspace` 分组的，每个 subspace 在 (model, hw, workload) 三元组下有不同的「是否激活」状态。autotuner 应先决定哪些 subspace 是 active，再在 active subspace 内搜索。**

对应到本实验（`LFM2.5-8B-A1B` × `1× H200` × `bf16 / 无 spec decode / single-server`）：

### 2.1 Active subspaces（实际搜索）

| Subspace | 旋钮 | 候选值 | 备注 |
|---|---|---|---|
| Memory / KV cache | `mem-fraction-static` | 0.75 / 0.85 / 0.90 | KV 留多少显存；0.90 偶尔 OOM |
| Batching / scheduling | `max-running-requests` | 8 / 16 / 32 / 64 | 并发请求上限 |
|  | `chunked-prefill-size` | -1 / 2048 / 8192 | -1 关闭分块 |
|  | `schedule-policy` | `lpm` / `fcfs` | 长 prefix 共享 vs 先来先服务 |
| Attention backend | `attention-backend` | （理论 `fa3` / `flashinfer` / `triton`） | **实际只有 `fa3` 能用**，见 §3 |
| CUDA graph | `disable-cuda-graph` | True / False | False = 抓 graph |
| MoE | `moe-runner-backend` | `triton` / `flashinfer_cutlass` | 两个 GEMM 实现 |

理论组合数：3 × 4 × 3 × 2 × 3 × 2 × 2 = **864**
应用 §3 的 attention 约束后：3 × 4 × 3 × 2 × 1 × 2 × 2 = **288**

### 2.2 Inactive subspaces（不搜，固定 / 不适用）

| Subspace | 决策 | 原因 |
|---|---|---|
| Parallelism (tp/dp/pp/ep/cp) | 全部 = 1 | 单卡环境 |
| Speculative decoding | 关闭 | 没有 draft model，且不是这次实验目标 |
| PD disaggregation | 关闭 | 单服务器 |
| Quantization | bf16 | LFM2.5-8B-A1B 没有发布 fp8/awq 版本 |
| KV cache dtype | `auto` | 单独的研究方向，本次不冒险 |
| HiCache / offload | 关闭 | KV cache 进不了 host memory，单卡显存足够 |
| LoRA / multimodal / tokenizer backend | 关闭 | 与目标 workload 无关 |

### 2.3 Out-of-scope（v3 再考虑）

- `cuda-graph-bs` / `cuda-graph-max-bs` 显式列表
- `schedule-conservativeness` 浮点
- `max-prefill-tokens`（和 `chunked-prefill-size` 互相纠缠）
- `radix-eviction-policy`

**为什么不在 v2 里加这些？** v2 是从扁平→条件化的第一步演进，先把 7 个对吞吐影响**已知较大**的参数搞清楚；多余的 dimension 会让 25 trials 的 TPE 难以收敛。Chendi 的框架本身允许逐步扩展。

---

## 3. Attention backend 兼容性矩阵（实证）

这是本实验最早发现、且最容易让人忽视的一类问题：**搜索空间里某些候选值对当前 (model, env) 直接不可用。**

### 3.1 实测结果（Phase 1，7 trials）

| trial | attn-backend | moe-backend | 结果 | 错误 |
|---|---|---|---|---|
| 0 | flashinfer | flashinfer_cutlass | ❌ server crash | flashinfer JIT 构建：`ld: cannot find -lcuda` |
| 1 | **fa3** | triton | ✅ 23.68 req/s | — |
| 2 | flashinfer | flashinfer_cutlass | ❌ | 同 trial 0 |
| 3 | flashinfer | flashinfer_cutlass | ❌ | 同 |
| 6 | **triton** | flashinfer_cutlass | ❌ server crash | `ValueError: layer_id=0 not in full attention layers: dict_keys([2, 6, 10, 14, 18, 21])` |
| 7 | **triton** | triton | ❌ | 同 trial 6 |
| 8 | **triton** | triton | ❌ | 同 |

**两条失败规则**：

1. **`triton` attention 不兼容 hybrid 架构**：LFM2.5-8B-A1B 的 layer 0/1/3/4/5/7/8/9/11/12/13/15/16/17/19/20/22/23 是 `conv` 层，只有 [2, 6, 10, 14, 18, 21] 是 `full_attention`。triton attention backend 假设所有层都是 attention，遇到 layer 0 直接抛错。
2. **`flashinfer` attention 被 conda env 挡住**：JIT 编译时 nvcc 需要链接 `-lcuda`（libcuda.so），但 conda env 的 `lib64` / `lib64/stubs` 里没有；ld 报错 `cannot find -lcuda`。这是 env 级问题，不是 flag 级问题——任何 flashinfer attention 候选都会失败。

**对应工程教训**（写进 Chendi 框架）：在搜索之前要做一次「**容性 probe**」——对每个候选 backend 跑 1 个 trivial trial，把不能启动的从候选集合里删除。否则 Optuna 会拿失败 trial 当 `value=0.0` 喂给 TPE，让 TPE 错误地学习到「triton attn 是坏的」，但其实只是和这个模型不兼容。

### 3.2 Phase 2b 行动

Phase 2b 把 `attention-backend` 候选集合收窄到 `["fa3"]`，理论组合数从 864 降到 288。详见 `harness/autotune_v2_lfm.py`（注释里有原因说明）。

---

## 4. 两个 baseline（已运行）

GPU 4，单 GPU，bf16，3 runs/regime，每 regime stddev_pct < 8% 视为 reliable。

### 4.1 baseline-A: true-default（`bench-specs/lfm2.5-8b-a1b-true-default.yaml`）

无任何吞吐调参 flag（即 `configs/lfm2.5_8b_a1b.yaml` 默认值）：`mem-fraction-static 0.85`、`schedule-policy lpm`、`max-running-requests 32`、`chunked-prefill-size -1`、`schedule-conservativeness 1.0`、`max-prefill-tokens 16384`、`disable-radix-cache false`、`disable-cuda-graph false`、`trust-remote-code true`。**不带任何 cookbook 推荐的解析器 flag。**

```
R_short_decode      :   1.692 req/s  (stddev  0.0%, reliable=True)
R_medium_balanced   :   7.279 req/s  (stddev  0.0%, reliable=True)
R_long_prefill      :  13.635 req/s  (stddev  0.7%, reliable=True)
R_concurrent_decode :  21.470 req/s  (stddev 10.8%, reliable=False) ⚠️
```

server 启动 46.1s。`R_concurrent_decode` 的 stddev 10.8% 触发 reliable=False，但其他 3 个 regime 完美稳定。

### 4.2 baseline-B: cookbook-default（`bench-specs/lfm2.5-8b-a1b-cookbook.yaml`）

在 baseline-A 基础上加上 cookbook 推荐的：`--reasoning-parser qwen3 --tool-call-parser lfm2`。这两个 flag 只影响输出解析（思考链/工具调用），**不影响吞吐内核**。

```
R_short_decode      :   1.686 req/s  (stddev  0.0%, reliable=True)
R_medium_balanced   :   7.281 req/s  (stddev  0.1%, reliable=True)
R_long_prefill      :  13.276 req/s  (stddev  1.3%, reliable=True)
R_concurrent_decode :  23.184 req/s  (stddev  1.6%, reliable=True)
```

server 启动 28.1s（更短可能是第二次启动 cache hot）。

### 4.3 对比

| Regime | baseline-A (true-default) | baseline-B (cookbook) | Δ |
|---|---|---|---|
| R_short_decode | 1.692 | 1.686 | -0.4% |
| R_medium_balanced | 7.279 | 7.281 | +0.0% |
| R_long_prefill | 13.635 | 13.276 | -2.6% |
| R_concurrent_decode | 21.470 | 23.184 | +8.0%（在 baseline-A 的 10.8% stddev 范围内） |

**结论**：两个 baseline 在统计噪声内等价。cookbook 的 parser flag 不影响吞吐，符合预期。下面 §5 用 baseline-B 作为对比基准（更稳定）。

### 4.4 LFM2.5 vs Qwen3-30B-A3B（同一基础设施）

R_concurrent_decode 上，LFM2.5-8B-A1B ≈ 23 req/s vs Qwen3-30B-A3B 6/25 baseline ≈ 14 req/s。差异主要来自活跃参数数：1B vs 3B，再加上 LFM2.5 的 hybrid conv 层比 dense attention 便宜。

---

## 5. Optuna v2 自动调参（Phase 2b，目标 regime = R_concurrent_decode）

### 5.1 设定

- Sampler: TPE, seed=2026（与 6/25 一致，便于交叉对比）
- n_trials: 25
- target regime: R_concurrent_decode（吞吐对 flag 最敏感的 regime）
- 每 trial 全 4 regime 都跑（user_attrs 记录），但 TPE 目标只看 R_concurrent_decode
- Per-trial 落盘：`results/2026-06-30_lfm2.5/optuna-v2-R_concurrent_decode/trial_NNNN/{flags.json, summary.json, server.log, trial_spec.yaml}`
- 汇总 CSV: `results/2026-06-30_lfm2.5/optuna-v2-R_concurrent_decode/per_trial_log.csv`
- Optuna RDB: `results/2026-06-30_lfm2.5/optuna-v2-R_concurrent_decode/study.db`

### 5.2 全部 25 个 trial 详表（按 trial 顺序）

| # | R_conc | R_short | R_med | R_long | moe-runner | cg-off | cap | chunk | sched | mem | wall(s) |
|--:|--:|--:|--:|--:|---|---|--:|---|---|--:|--:|
| 0 | **22.17** | 1.65 | 7.08 | 9.25 | flashinfer_cutlass | False | 32 | -1 | fcfs | 0.90 | 54.5 |
| 1 | 8.43 | 1.70 | 7.23 | 15.35 | triton | False | 8 | 8192 | lpm | 0.90 | 60.8 |
| 2 | 5.36 | 0.35 | 2.65 | 7.79 | flashinfer_cutlass | True | 16 | 2048 | fcfs | 0.75 | 136.7 |
| 3 | 1.70 | 0.22 | 1.68 | 5.79 | triton | True | 8 | 2048 | fcfs | 0.75 | 223.5 |
| 4 | 8.50 | 1.69 | 7.49 | 19.83 | triton | False | 8 | 2048 | fcfs | 0.75 | 60.4 |
| 5 | 2.95 | 0.34 | 2.64 | 9.88 | flashinfer_cutlass | True | 8 | 2048 | fcfs | 0.85 | 151.0 |
| 6 | 6.88 | 0.23 | 1.76 | 5.62 | triton | True | 64 | 8192 | fcfs | 0.75 | 174.3 |
| 7 | 9.28 | 0.37 | 2.27 | 6.70 | flashinfer_cutlass | True | 64 | 2048 | fcfs | 0.90 | 128.9 |
| 8 | 7.12 | 0.23 | 1.79 | 6.05 | triton | True | 32 | 2048 | fcfs | 0.85 | 172.8 |
| 9 | 1.87 | 0.23 | 1.84 | 6.28 | triton | True | 8 | 2048 | fcfs | 0.75 | 206.0 |
| 10 | 22.08 | 1.65 | 7.09 | 9.30 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 57.6 |
| 11 | 22.24 | 1.65 | 7.10 | 9.28 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 54.5 |
| 12 | 21.87 | 1.64 | 7.11 | 9.25 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 57.7 |
| 13 | 21.92 | 1.64 | 7.10 | 9.27 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 57.6 |
| 14 | 21.99 | 1.65 | 7.11 | 9.24 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 54.6 |
| 15 | 13.60 | 1.64 | 7.10 | 9.29 | flashinfer_cutlass | False | 16 | -1 | lpm | 0.90 | 57.4 |
| 16 | 22.12 | 1.65 | 7.10 | 9.31 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.85 | 54.6 |
| **17** | **22.32** ⭐ | 1.64 | 7.09 | 9.32 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 54.6 |
| 18 | 22.28 | 1.65 | 7.10 | 9.27 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 54.6 |
| 19 | 21.75 | 1.68 | 7.11 | 11.88 | flashinfer_cutlass | False | 64 | 8192 | lpm | 0.90 | 54.2 |
| 20 | 13.59 | 1.65 | 7.09 | 9.38 | flashinfer_cutlass | False | 16 | -1 | lpm | 0.85 | 57.3 |
| 21 | 22.14 | 1.64 | 7.10 | 9.37 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 54.6 |
| 22 | 21.98 | 1.65 | 7.13 | 9.34 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 54.5 |
| 23 | 21.89 | 1.65 | 7.12 | 9.27 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 54.6 |
| 24 | 22.07 | 1.65 | 7.11 | 9.34 | flashinfer_cutlass | False | 32 | -1 | lpm | 0.90 | 54.5 |

所有 trial 的 `attention-backend` = `fa3`（详见 §3 兼容性裁剪后的 Phase 2b 设定）。

### 5.3 观察

**TPE 收敛模式**：

- Trial 0 一开 就找到一个不错的配置（22.17）：`flashinfer_cutlass + cg-on + cap=32 + chunk=-1 + fcfs + mem=0.9`。
- Trial 1-9 探索了一堆 `cg-off` 和 `cap=8` 的差配置（都 < 10 req/s）。
- Trial 10 起 TPE 锁定了 `flashinfer_cutlass + cg-on + cap=32 + chunk=-1 + lpm + mem=0.90` 这个组合，**反复 sample 了 12 次**（trials 10-14, 17, 18, 21-24），获得的 R_conc req/s 全部在 **21.87-22.32** 之间，均值 **22.07 ± 0.15（0.7% stddev）**。这给我们一个非常精确的「最优 config 的跨 server lifetime 变差」估计。
- TPE 在最后偶尔小幅扰动（trial 15/20 试了 cap=16，掉到 13.6；trial 19 试了 chunk=8192，掉到 21.75；trial 16 试了 mem=0.85，得 22.12 = 不变）。

**关键 flag 的边际效应（基于该数据）**：

| flag 改动 | 效果 |
|---|---|
| `disable-cuda-graph: False → True` | R_conc 从 ~22 → 1.7-9.3（**毁灭性**） |
| `max-running-requests: 32 → 8` | R_conc 从 ~22 → ~8（cap 严重限制并发） |
| `max-running-requests: 32 → 16` | R_conc 从 ~22 → ~13.6（仍然受限） |
| `chunked-prefill-size: -1 → 2048` | 在 cg-on 下基本无影响；R_conc 在 ~22 不变 |
| `schedule-policy: fcfs → lpm` | 无影响（trial 0 vs trial 10：22.17 vs 22.08） |
| `mem-fraction-static: 0.85 → 0.90` | 几乎无影响（trial 16 vs trial 17：22.12 vs 22.32） |
| `moe-runner-backend: flashinfer_cutlass → triton`（在好 batching 下） | **未被 TPE 测试**——见 §6 |

### 5.4 best (trial 17)

```json
{
  "best_trial_number": 17,
  "best_value_req_per_s": 22.32,
  "best_params": {
    "mem_fraction_static": 0.9,
    "max_running_requests": 32,
    "chunked_prefill_size": -1,
    "schedule_policy": "lpm",
    "attention_backend": "fa3",
    "disable_cuda_graph": false,
    "moe_runner_backend": "flashinfer_cutlass"
  },
  "per_regime_req_per_s": {
    "R_short_decode": 1.6449,
    "R_medium_balanced": 7.0923,
    "R_long_prefill": 9.3244,
    "R_concurrent_decode": 22.3194
  }
}
```

### 5.5 工程小坑：port-cleanup race（Phase 2a）

Phase 2a 里 4 个 trial 在 0.7s 就失败，错误是「Port 127.0.0.1:31500 is already in use」。上一个 trial 的 sglang server 进程组 SIGKILL 之后，OS 把端口放入 TIME_WAIT 几秒。

**修复**（已合入 `harness/autotune_v2_lfm.py::_wait_port_free`）：每次 trial 前 poll TCP connect_ex 直到端口空闲（上限 60s）。Phase 2b 25 trial 全部干净，无 port collision。

### 5.6 Phase 1 / Phase 2a 的存档（学到的负面知识）

为完整性，所有失败 trial 都保留：

- `_phase1_archive/`：broad search，发现 attention backend 兼容性问题（见 §3.1 表）。
- `_phase2a_archive/`：窄空间但 port-cleanup race，3 valid + 4 failed-by-bug。

---

## 6. **关键发现**：Optuna 错过了实际最优

### 6.1 三个 baseline 的精确刻画

| Baseline | n runs (server lifetimes) | R_conc mean | stddev | 备注 |
|---|---|---|---|---|
| true-default（无 cookbook flag） | 1 | 21.47 | 10.8%（不可靠）| 单次 `R_concurrent_decode` 噪声大 |
| cookbook-default（+ parser flags） | **3** | **23.74** | **0.5%** | 这是真正稳定的基线 |
| Optuna best (trial 17) | 1 + 11 重复 sample | 22.07 | 0.7% | 收敛到 `flashinfer_cutlass MoE` |

`cookbook-default` 跨 3 个独立 server lifetime 得到 23.617 / 23.850 / 23.756 req/s，均值 23.74 ± 0.12（0.5%）。这个误差比 Optuna "best" 的 0.7% 还小，所以**两者的差 (23.74 - 22.07) / 22.07 = 7.6% 是统计显著的、绝非噪声**。

### 6.2 谜题：为什么 cookbook 默认能赢 Optuna？

观察 baseline-B server log 与 trial 17 server log 的 server_args 差异：

| 参数 | baseline-B (cookbook) | trial 17 (Optuna best) | 注 |
|---|---|---|---|
| `mem_fraction_static` | 0.85 | 0.90 | 差异 |
| `attention_backend` | `'fa3'` (sglang auto-pick) | `'fa3'` (显式) | 等价 |
| `moe_runner_backend` | **`'auto'`** | **`'flashinfer_cutlass'`** | **差异关键** |
| `reasoning_parser` | `'qwen3'` | None | 仅影响输出解析 |
| `tool_call_parser` | `'lfm2'` | None | 仅影响输出解析 |
| 其他 ~200 个 flag | 完全一致 | 完全一致 | — |

baseline-B server log 关键信息：

```
[2026-06-30 21:41:46] Using default MoE kernel config.
Performance might be sub-optimal! Config file not found at
.../fused_moe_triton/configs/triton_3_5_1/E=32,N=1792,device_name=NVIDIA_H200.json
```

→ baseline-B 用了 **fused_moe_triton**（sglang 把 `moe_runner_backend='auto'` 解析成了 triton）。
→ trial 17 强制用了 **flashinfer_cutlass**。

### 6.3 验证实验：手工调一个 trial 的 moe 选择

我们做了一个 explicit-triton-MoE 的验证实验：把 baseline-B 的全部 flag 复用，**只额外加** `--moe-runner-backend triton`：

```yaml
# bench-specs/lfm2.5-8b-a1b-triton-moe.yaml
server:
  config: configs/lfm2.5_8b_a1b.yaml   # 含 mem=0.85, sched=lpm, cap=32, chunk=-1, cg-on
  overrides:
    moe-runner-backend: triton
```

结果：

```
R_short_decode      :  1.691 req/s  (stddev 0.0%, reliable=True)
R_medium_balanced   :  7.303 req/s  (stddev 0.1%, reliable=True)
R_long_prefill      : 13.394 req/s  (stddev 0.3%, reliable=True)
R_concurrent_decode : 23.527 req/s  (stddev 0.6%, reliable=True)  ← 和 cookbook 一致
```

R_concurrent_decode = **23.527 req/s**，与 cookbook baseline 三次平均的 23.74 在 1% 以内，完全一致。

**所以**：实际最优 config 是 **`moe-runner-backend=triton`**（不是 Optuna 选的 `flashinfer_cutlass`），其他 flag 任意（cap=32 + chunk=-1 + cg-on + lpm + mem=0.85 or 0.9）。

### 6.4 TPE 为什么漏掉了这个？

Optuna 的 25 个 trial 中，**只有 5 个 trial 用了 `triton` MoE**：

| trial | moe | cg-off | cap | chunk | sched | mem | R_conc |
|--:|---|---|--:|---|---|--:|--:|
| 1 | triton | False | 8 | 8192 | lpm | 0.90 | 8.43 |
| 3 | triton | True | 8 | 2048 | fcfs | 0.75 | 1.70 |
| 4 | triton | False | 8 | 2048 | fcfs | 0.75 | 8.50 |
| 6 | triton | True | 64 | 8192 | fcfs | 0.75 | 6.88 |
| 8 | triton | True | 32 | 2048 | fcfs | 0.85 | 7.12 |
| 9 | triton | True | 8 | 2048 | fcfs | 0.75 | 1.87 |

每一个 triton trial 都搭配了**至少一个**致命缺陷：

- trial 1, 3, 4, 9: `cap=8`（cap 太小，限制并发，毁掉 R_concurrent_decode）
- trial 3, 6, 8, 9: `cg-off`（cuda graph 关闭，单次 forward 慢 5×）
- trial 6, 8: 没有 cg-off 但 chunk=2048 又 sched=fcfs（小问题）

所以 TPE 看到的"triton MoE 的样本"全部都 < 10 req/s，于是它的概率分布把 `triton MoE = 坏` 学得很死，**剩下 18 个 trial 一次都没再尝试** `triton MoE + 好 batching` 的组合。但这个组合恰好是最优解。

这是 Chendi 6/30 邮件里指出的**flag 之间的非独立性**最典型的表现：

```
moe-backend × max-running-requests × disable-cuda-graph

在 cap=8 或 cg-off 时：triton MoE = 差，flashinfer_cutlass = 中差
在 cap=32 + cg-on 时:  triton MoE = 23.5，flashinfer_cutlass = 22.0  ← 反转！
```

TPE 假设 7 个 hyperparameter 之间独立（它对每一维分别建一个 KDE）。这个假设在「最差区域里 backend 排名」≠「最优区域里 backend 排名」时直接破产。

### 6.5 工程教训

1. **autotuner 必须保证「每个候选 backend × 每个 batching profile」至少有一次被测试**——可以用**分层采样**（stratified sampling）或者**显式 grid 一遍 backend × 一组好 batching**。
2. **`moe-runner-backend=auto` 本身就是 sglang 的"先决」逻辑**——它根据 (model_arch, device_sm, dtype, num_experts, expert_hidden_dim) 做了 lookup。这等于 Chendi 框架里的「Step 1: pick active subspace」。如果 autotuner 不知道这个 lookup，强行去 sweep `triton / flashinfer_cutlass / deep_gemm / ...`，很可能在 25 trial 里搞不清谁好谁坏。
3. **比较时要用「相同 baseline」**：原来的 baseline-B 是 single run（噪声 ~5%），不足以拿来和 Optuna 的 0.7% 收敛做对比。后续所有 baseline 都至少 3 次 server lifetime。

---

## 7. 与 6/25 Qwen3-30B-A3B 实验的对比

| 维度 | 6/25 Qwen3-30B-A3B | 6/30 LFM2.5-8B-A1B |
|---|---|---|
| 模型类型 | 30B 总 / 3B 活跃 MoE（dense + MoE 混合）| 8B 总 / 1B 活跃 MoE（hybrid: conv + attention + MoE）|
| 搜索空间 | flat 5-param × 96 组合 | 条件化 7-param × 288 组合（裁剪后） |
| n_trials | 96（grid）+ 25 Optuna | 25 Optuna |
| baseline R_conc | ~14 req/s | ~23.7 req/s（LFM 更小更快）|
| Optuna best vs baseline | 等价 ± 5% | **比 baseline 慢 7.6%（!）** |
| 主要发现 | "sglang 默认就是最优" | sglang 默认仍然最优；但 TPE 错过了 |
| 失败模式 | 几乎没有；flat 空间 + 小 cardinality 让 TPE 容易覆盖 | TPE 被早期 triton×bad-batching trial 误导，永远没回去试 triton×good-batching |

**两次实验合在一起的元结论**：

- **「sglang 团队的开箱默认对 main-stream MoE 模型已经足够好」**这条结论**复现了两次**——但每次 Optuna 都「没办法证明这点，只能勉强追平/还低一点」。
- **条件化搜索空间**的设计 *(Chendi 6/30 框架)* 不只是更精细的 search space 表达，更是**避免 TPE 在 backend 选择上犯系统性错误**的必要前提。

---

## 8. 下一步

### 8.1 必做（短期）

- [ ] **把 cookbook-default 重新作为 LFM2.5 的官方 baseline**：3-run 平均 23.74 ± 0.5% 已经稳定，写进 master report。
- [ ] **扩展 v2 search space 加入 `moe-runner-backend: auto`**：让 sglang 自己决定。这是最便宜的修复——不改 TPE，只让候选包含 `auto`。
- [ ] **加 stratified sampling**：保证每个 `moe-runner-backend` 候选至少有一个 trial 搭配 `cap=32 + cg-on + chunk=-1`。

### 8.2 中期（Chendi 框架完整落地）

- [ ] 把 Chendi 邮件里 8 大 subspace 实现成一个 declarative YAML（`search_space/lfm2.5_8b_a1b_on_h200_bf16.yaml`）。
- [ ] 写一个 (model, hw, env) → active subspace 的解析器：基于 sglang 的 server_args 默认 + GPU SM 版本 + dtype 自动裁剪。
- [ ] 把 TPE 换成考虑条件依赖的 sampler（候选：`optuna.samplers.NSGAIISampler` 用 grid 起步 + TPE 精搜，或者 BoTorch + 离散变量编码）。

### 8.3 长期（Agent 方向）

- [ ] 把"backend 兼容性 probe"做成 agent 的 Step 0：对当前 (model, hw, env)，先发 N 个 minimal probe trial，把不能起来的候选剔掉。
- [ ] 把"sglang `auto` 的内部决策树"反向工程出来——agent 应该知道 sglang 在什么条件下会选什么默认，避免重复搜索。
- [ ] 把 cookbook 表当作"先验"：对每个 (model, hw) 三元组，cookbook 推荐的 flag 作为 TPE 的 prior，TPE 从这里附近开始搜，而不是从零开始。

### 8.4 跨实验（与 6/25 对接）

- [ ] 把 6/25 Qwen3 的 5-flag autotune 用同样的 6/30 v2 conditional-search-space 重跑一遍，看是否会复现"TPE 漏掉最优 MoE backend"这个失败模式。
- [ ] 跨模型对比：在两个模型上分别 sweep `moe-runner-backend` 单维度（其他 flag 固定为 best）—— 看不同 model 对 backend 偏好是否一致。

