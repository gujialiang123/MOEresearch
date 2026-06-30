# sglang Cookbook 部署 baseline 分析

**2026-06-30** | 输入文件分析报告

> 用户提供了两个从 sglang 官方 cookbook 爬下来的配置表，要求评估其作为 baseline 的可用性，并指出哪些可以"现在跑一份结果出来"。

---

## 输入文件

| 文件 | 行数 | 内容 |
|---|---|---|
| `sglang_cookbook_deployment_baselines.json` | 2135 | 77 个 deployment 配置（机器可读） |
| `sglang_cookbook_deployment_baselines.md` | 1574 | 同样 77 个配置以 markdown 呈现（人类可读） |

两份是**同一份数据的两种格式**。来自 sglang 官方 `docs_new/src/snippets/configs/*.jsx` deployment cells —— **这就是 sglang 团队给出的"已验证最佳启动参数"**。

每个配置含字段：`model / hardware / variant / quantization / strategy / nodes / verified / model_path / env / flags / docker_image / source_file / notes / python_command / docker_command`。

---

## 覆盖矩阵

### 模型 (5 个)

| 模型 | 我们本地有吗 | 备注 |
|---|---|---|
| **GLM-5.2** | ❌ | 29 个配置；H200 上有 FP8 verified=true × 3 strategy |
| **MiniMax-M3** | ❌ | 9 个配置；H200 verified=true，要 tp=8 |
| **LFM2.5** (小，230M-8B) | ❌ | 24 个配置；多数 tp=1 单卡可跑 |
| **Laguna-M.1** | ❌ | 14 个配置；要 tp=8 |
| **Unlimited-OCR** | ❌ | 1 个配置；OCR 模型，方向不对 |
| ~~DeepSeek-V4~~ | ❌ | cookbook 里是个 empty entry，没有可用 cell |

**没有 Qwen 系列**（我们的 Qwen3-30B-A3B 不在 cookbook 里）。

### 硬件

H200 17 / B200 18 / H100 9 / B300 12 / GB200 4 / GB300 13 / MI300X-355X 各 1

### 量化 / 策略 / 节点

- 量化：bf16 42 / fp8 17 / nvfp4 9 / mxfp8 8 / default 1
- 策略：balanced 34 / high-throughput 9 / low-latency 10 / default 24
- 节点：single 68 / multi-2 9

---

## H200 + single-node 全列表（我们这个 setup 能直接用的）

共 **14 条**：

| 模型 | 量化 | variant | strategy | verified | tp | 备注 |
|---|---|---|---|---|---|---|
| GLM-5.2 | fp8 | default | balanced | ✅ | **8** | 要 8 卡 |
| GLM-5.2 | fp8 | default | high-throughput | ✅ | **8** | 要 8 卡 |
| GLM-5.2 | fp8 | default | low-latency | ✅ | **8** | 要 8 卡 |
| MiniMax-M3 | bf16 | default | balanced | ✅ | **8** | 要 8 卡 |
| Laguna-M.1 | bf16 | default | balanced | ✅ | **8** | 要 8 卡 |
| Laguna-M.1 | fp8 | default | balanced | ✅ | **8** | 要 8 卡 |
| **LFM2.5** | bf16 | 230m | default | ✅ | **1** | ★ 单卡可跑 |
| **LFM2.5** | bf16 | 350m | default | ✅ | **1** | ★ 单卡可跑 |
| **LFM2.5** | bf16 | 8b-a1b | default | ✅ | **1** | ★ 单卡可跑（A1B MoE） |
| **LFM2.5** | bf16 | instruct | default | ✅ | **1** | ★ 单卡可跑 |
| **LFM2.5** | bf16 | jp | default | ✅ | **1** | ★ 单卡可跑 |
| **LFM2.5** | bf16 | thinking | default | ✅ | **1** | ★ 单卡可跑 |
| **LFM2.5** | bf16 | vl | default | ✅ | **1** | ★ VL 多模态 |
| **LFM2.5** | bf16 | vl-450m | default | ✅ | **1** | ★ VL 多模态 |

⭐ 8 条 LFM2.5 可以**在我们当前 1× H200 setup 直接跑**。

---

## 重要的元 finding

**Cookbook 里 H200 单卡 + bf16/fp8 MoE 的 verified config = 0 条**（GLM/Laguna/MiniMax 全要 tp=8）。

这本身就是个有用的信号：**sglang 官方认为 30B+ MoE 应该用 8 卡 TP 跑，不应该单卡跑**。我们 6/25 强行单卡跑 Qwen3-30B-A3B 拿到的"defaults are optimal"结论，**可能不能直接外推到 sglang 推荐的部署形态**。值得在以后的报告里 flag。

---

## 三档推荐

### 🥇 第一档：现在 (今天/明天) 能直接跑

**只有 LFM2.5 系列**。8 个 variant 都是 single-GPU bf16。其中最有研究价值的是：

#### LFM2.5-8B-A1B（**最高优先级**）

- **架构**：MoE-A1B（8B total / 1B active）
- **跟 Qwen3-30B-A3B 同家族**（A3B = 3B active；都是稀疏 MoE）
- 文件 ~16 GB，下载 ~10 分钟
- verified=true，sglang 官方推荐配置
- **价值**：验证"defaults are optimal"在另一个 MoE 模型上的泛化性

Cookbook 的推荐 flag：
```bash
sglang serve \
  --model-path LiquidAI/LFM2.5-8B-A1B \
  --trust-remote-code \
  --tp 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser lfm2
```

#### LFM2.5-1.2B-Instruct（**smoke test 候选**）

最快最轻，30 分钟搞完一轮完整 4-regime × autotune 流程。适合先做 smoke 验证我们的 harness 跟新模型 wire 起来没问题，再上 8B-A1B 实验。

### 🥈 第二档：以后能借多卡再跑（H200 verified）

| 配置 | 价值 |
|---|---|
| **GLM-5.2 FP8 三 strategy 对比** (balanced / high-throughput / low-latency) | 唯一一组**官方给出三种 strategy 对比**的 verified config——直接可以验证 "sglang 官方策略选择是不是真的有差异化最优" |
| MiniMax-M3 BF16 balanced | 另一个 MoE 架构 |
| Laguna-M.1 bf16/fp8 | dense model，跟 MoE 形成对比 |

都需要 8× H200，我们目前只能用 1-2 张。**如果以后能借到 8 卡，GLM-5.2 FP8 三 strategy 对比是最高价值实验**。

### 🥉 第三档：参考用（agent 推荐系统的 ground truth）

GLM-5.2 在 b200 / h100 / b300 / gb200 / gb300 都有 verified config。

**有意思的横切**：
- H200 GLM-5.2 用 `--moe-a2a-backend deepep`；B200 某些 strategy 用 `--enable-dp-attention`
- H200 → FP8；GB300 → NVFP4；H100 → BF16
- **这就是 "agent 跨硬件推荐" 的 ground truth dataset** —— agent 推荐的 config 应该 reproduce cookbook 的选择

---

## 我能立刻做的实验（按 ROI 排序）

### Option A: LFM2.5-8B-A1B baseline + autotune on H200（**推荐**）

**步骤**：
1. 下载 `LiquidAI/LFM2.5-8B-A1B`（~16 GB，10 分钟）
2. 写 `bench-specs/lfm2.5-8b-a1b-cookbook-default.yaml`（cookbook 给的 flag 套进我们 harness）
3. 跑 4-regime bench 拿 cookbook baseline 数字
4. 跑 Optuna 15-trial × R_medium 看 cookbook flag 是不是真最优
5. 写报告对比

**预期 timeline**：~1.5 小时

**可能的 3 种结果**：
- (a) cookbook config ≈ Optuna best → **sglang 官方推荐 = 最优**，泛化我们 6/25 结论到不同模型
- (b) Optuna 显著比 cookbook 好 → **官方 baseline 也有 headroom**，agent 价值证据
- (c) cookbook config 给非平凡的 flag (`--reasoning-parser qwen3`, `--tool-call-parser lfm2`) 而这些根本不影响 throughput → **cookbook 跟性能调优是两件事**

任何一种结果都对项目方向有价值。

### Option B: LFM2.5-1.2B-Instruct（smoke + 最快）

更轻量更快，30 分钟搞完。先验证 harness wire-up，再做 A。

### Option C: 把 cookbook → bench-spec 写成自动化 pipeline

把 cookbook JSON 直接喂给一个 generator，自动产出所有 H200 单卡 verified config 的 bench-spec。然后批量跑。**适合做成 demo**，但工程量稍大（~半天）。

---

## 文件位置

- 输入：`sglang_cookbook_deployment_baselines.json` + `.md` (repo 根目录)
- 本分析：`docs/2026-06-30/cookbook_baseline_analysis.md`
- LFM2.5 模型路径 (待下载)：`/data/hf/models/LFM2.5-8B-A1B/`
- 我们已有的可对比 baseline：
  - `docs/2026-06-25/autotuning_honest_results.md` (Qwen3-30B-A3B "defaults are optimal" 结论)
  - `docs/2026-06-29/profiling_validation_of_universal_config.zh.md` (kernel-level roofline 分析)

---

## 待用户决策

1. 跑 Option A、B、还是 C？
2. 用 GPU 4 还是 GPU 5？
3. 要不要顺便下载 LFM2.5 全系列（8 个 variant 共 ~30 GB）做 smoke 矩阵？
