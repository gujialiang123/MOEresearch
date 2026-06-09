# Agent profiling 能力盘点 — 现状 / 缺口 / 需要 mentor 协助的部分

> **2026-06-09 更新**: Part B 6 个 gap 里 **3 个已经修正**(B.1 nsys SQL 路线、B.3 vLLM torch profiler 接通、B.6 跨配置自动对比),**1 个有进展**(B.2 NCU 根因找清楚,已发请求);**2 个仍未解决**(B.4 内存压力、B.5 sglang/vLLM Python 状态)。详见每节标题旁的 ✅/🟨/❌ 状态。
>
> 同时 **5 个新 skill 上线**: `regime-sweep-runner`、`cross-regime-anomaly`、`profile-summary-unified`、`handoff-prompt-template`、`e2e-bench-runner` v1(支持 YAML regime)。新的 skill 流水线图见 `docs/2026-06-09/skill_architecture.md`。

> 目的: 在讨论"如何把 agent 自动化"之前,先理清当前 agent (Copilot CLI with Claude) 在这次 sglang/vLLM/cutlass MoE 调研中实际用了哪些工具、拿到哪些信息、哪些拿不到、哪些将来构建真实 agent 时需要补。
>
> 时间: 2026-06-08 (一系列实验之后的复盘)

---

# Part A — 当前用过的 profiling 工具盘点

按"信息密度从低到高 / agent 调用难度从低到高"排:

## A.1 e2e benchmark (我们自己写的 harness)

**工具**: 一个 Python 脚本 (`results/4way_bench/scripts/run_bench_4way.py`),用 `requests.post` 给 server `/generate` 或 `/v1/completions` 发请求,测 wall time。

**Agent 能拿到的信息**:
- `req/s` (吞吐)
- `tok/s` (生成速度)
- `wall_s` (秒表时间)
- per-request latency (理论上能,我们没记)
- 不同 regime (短 prompt / 长 prompt / 不同 conc) 对比

**信息密度**: 低 (一个数字)。但**是 ground truth**,任何其它优化最终都要回到这里证明。

**Agent 调用代价**: 中等。要先启动 server (~30s)、跑 bench (~3-12s/run)、做 3 runs 取 mean。

**这次实战发现的坑**:
- Cold run 跟 warm run 差很多 (vLLM run1 ~3 req/s vs run 2/3 ~4.7 req/s),必须丢 run 1
- 单次结果有 ±5% 噪声,3 runs 才稳

## A.2 sglang/vLLM 自带的 server.log

**工具**: 用 `grep -E "..."` 读 server.log 文件。

**Agent 能拿到的信息**:
- 启动各阶段时间戳: "Load weight end. elapsed=13.25 s"
- Cudagraph capture: "Capture cuda graph end. Time elapsed: 1.21 s"
- Autotune: `[Autotuner]: Autotuning process starts/ends` (~3s)
- Per-batch decode 时的 throughput: `Decode batch ... gen throughput (token/s): 340.86, cuda graph: True`
- 配置参数 dump (`server_args=ServerArgs(...)` 一长串)
- Error / hang 症状 (我们就是这样发现 Bug A detokenizer heartbeat 停的)

**信息密度**: 中。结构化数据混在 log noise 里。

**Agent 调用代价**: 低。`grep -E` 几次就行。

**这次实战发现的坑**:
- log 太长(单 server 启动 ~30K 行),agent context 装不下,只能 `tail` 或 `grep`
- `server_args=` 那一行 16KB,塞了所有 default 参数,真正改了的只有 5-6 个,需要 diff vs default 才有用
- "cuda graph: True/False" 这个标记很关键,直接告诉我们 prefill 不在 graph 里 — 这就是直接从 log 推论出的核心 insight 之一

## A.3 nsys profile → `nsys stats` CSV

**工具**:
```bash
nsys profile -t cuda -s none -o output.nsys-rep <command>
nsys stats --report cuda_gpu_kern_sum --format csv output.nsys-rep
nsys stats --report cuda_api_sum --format csv output.nsys-rep
```

**Agent 能拿到的信息**:

`cuda_gpu_kern_sum` (GPU 侧 kernel 视图):
- 每个 kernel 模板名 (完整 mangled C++ name, 几 KB 长)
- 总调用次数 (Instances)
- 总时间 (Total Time ns)
- 平均时间 (Avg ns)
- min/max/stddev

`cuda_api_sum` (CPU 侧 CUDA API 视图):
- 每个 CUDA API 名字 (`cudaLaunchKernel`, `cudaMemcpyAsync`, ...)
- 总调用次数
- 总时间 / avg / min / max

**信息密度**: 高,但是是**聚合数据**,没有时间维度。

**Agent 调用代价**: 中等。要把 server launch 包在 nsys 下,跑完 SIGINT flush (~30s),然后 `stats` 命令再 30-60s 处理。

**这次实战拿到的关键数据**:
- vLLM AT_ON_CG_ON: 36k total launches, 3.27s GPU active
- vLLM AT_OFF_CG_OFF: 850k total launches, 17.8s GPU active
- → 直接验证 "cudagraph 把 launch 数压缩 23×"
- → 间接推论 max(CPU, GPU) 模型

**这次实战发现的坑**:
- kernel 模板名特别长 (单个 ~2KB),agent context 一次能装的 row 数有限
- cudagraph 模式下 nsys 看不到 graph 内部的单个 kernel,nsys 算时间会少报 (我们 AT_OFF_CG_ON 那个 GPU active 0.42s 就是这种)
- profile 文件大 (50-150 MB / 30s 录制),不能 commit 到 git
- 录制 overhead 不小,实测 wall 比不开 nsys 慢 1.5-2× (我们最后 bench 也是这种状态)

## A.4 直接读源码 (sglang, vLLM, flashinfer)

**工具**: `view` / `grep` 工具直接读文件。

**Agent 能拿到的信息**:
- 决策逻辑 (`_should_run_flashinfer_autotune`)
- TODO 注释 (sglang `model_runner.py:1841` 那个 "flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.")
- 函数签名 / 调用关系
- 配置 default 值

**信息密度**: 高,而且 actionable。

**Agent 调用代价**: 低。`view`/`grep` 极快。

**这次实战的关键发现**(全是源码读出来的,不是 profiling):
- sglang TODO 直接揭露 Bug A
- `flashinfer/autotuner.py:432` `if not is_tuning_mode: return fallback tactic` → 推翻"runtime sweep" 假设
- `flashinfer/.../flashinfer_cutlass_fused_moe_binding.cu:638` "Fallback tactic is set to be 0" → 知道 fallback 跑哪个 kernel
- `flashinfer/tuning_configs/v0_1_trtllm_fused_moe_NVIDIA_B200.py` 存在,H200 不存在 → 知道 hand-tuned 表的覆盖
- `vllm/.../kernel_warmup.py:132 with autotune():` → 知道 vLLM 显式触发 autotune

**反直觉发现**: 这次大部分关键 insight 来自**读源码**,不是 profiling。Profiling 只能告诉你 "什么慢",源码告诉你 "为什么慢"。

## A.5 Github issue/PR 搜索

**工具**: `gh issue list/view`, `gh pr list/view`, `gh search`

**Agent 能拿到的信息**:
- 同类 bug 是否被报过
- 是否有正在修的 PR
- 维护者讨论 / 历史决策
- benchmark 数据 (有时 PR description 里有)

**这次发现的关键 issue/PR**:
- sglang PR #21872: 同一作者独立测 H100 SM90 FP8,与我们结论一致
- sglang issue #26715: 同症状(SM100 PCG IMA)的反向证据 — workaround 是 cutlass,说明 SM100 上 cutlass + cudagraph 正常
- sglang PR #15565: silent disable cutlass autotune 的 origin commit
- → 这些都不是 profile 出来的,是 github 调研

**Agent 调用代价**: 低 (gh CLI 很快)

## A.6 实际微基准 (Python 直接 call kernel)

**工具**: 我们写了 `cutlass_moe_microbench.py`,绕过 server 直接调 `flashinfer.cutlass_fused_moe()`。

**Agent 能拿到的信息**:
- 单 kernel 时间 (微秒级,排除掉所有 server / scheduling overhead)
- 对比不同条件 (fallback vs tuned, 不同 batch)
- AutoTuner cache 内容 (通过 `tuner.profiling_cache` 直接读)

**这次实战的关键发现**:
- tuned vs fallback 在 SM90 上差 5-6× (微基准证明)
- 不同 batch 选不同 tactic (cache 里看到 tactic IDs: 10, 33, 37, 38, 40, 44, 73, 99, 100, 102, 104, 119)
- AutoTuner 内部 cache key 结构 (按 shape bucket)

**信息密度**: 极高,直接 actionable。

**Agent 调用代价**: 中等(需要写 Python 脚本)。

---

# Part B — 当前拿不到 / 受限的信息

## B.1 nsys GUI 时间线视图 ❌ → ✅ **2026-06-09 部分修正**

**之前以为看不到**:
- Kernel 之间的 idle gap
- CPU thread 和 GPU stream 并排
- cudaMemcpyAsync 和 kernel 的 overlap 程度

**真实情况(修正)**: 用 `nsys export --type sqlite` 能拿到 `CUPTI_ACTIVITY_KIND_KERNEL` 表,**每个 kernel 一行 + start/end 时间戳 + stream + grid/block**。GUI 看到的所有时间线数据都能用 SQL 算出。已经写进新 skill `nsys-timeline-sql`,默认 summary 自带"top 5 idle gaps"字段。

**真实还看不到**(剩余缺口):
- 颜色块时间轴(视觉一秒看出结构)
- 鼠标 hover 看 kernel 全名(我必须 SELECT 查询)
- GUI 的 stream overlap 拓扑视图

**详情**: `docs/2026-06-08/nsys_deep_dive_and_proton.md` 全文修正了这条 gap。

## B.2 ncu (Nsight Compute) — kernel-level metrics ❌ → 🟨 **2026-06-09 进展中**

**仍然没用上**,但**根因找清楚了**:
- NCU 二进制在 `/home/t-chendili/.conda/pkgs/nsight-compute-*` 已经装好
- 撞 `ERR_NVGPUCTRPERM`:`/proc/driver/nvidia/params` 里 `RmProfilingAdminOnly=1`
- t-jialianggu 账号**没有任何 sudo 权限**(已确认 — 不在 `sudo` 或 `wheel` group;没有 `/etc/sudoers.d/` 条目);chendi 有专门的 NOPASSWD 条目
- **已发请求给 chendi**(2026-06-09): 加一条 `t-jialianggu ALL=(ALL) NOPASSWD: <ncu_path>` 或 reload nvidia driver 翻 `RmProfilingAdminOnly=0`

**等解锁后的 skill 计划**: 新增 `ncu-microarch` skill,出 SM occupancy / achieved FLOPs / L2 hit / register spills / top warp-stall reason。`profile_unified.json` 的 `kernel_micro` 字段(目前永远是 `available: false`)就能填上。流水线其他 skill 不需要改。

## B.3 Python op-level 时间分布 ❌ → ✅ **2026-06-09 修正**

**这次的进展**:
- vLLM 路径**已经接通**: server 启动加 `--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'`,运行时通过 `/start_profile` / `/stop_profile` HTTP 端点抓 torch profiler trace
- 实际用过: 2026-06-09 CUTLASS 调查整个 kernel breakdown 就是这么来的(`docs/2026-06-09/cutlass_vs_triton_e2e_investigation.md`)
- 解析逻辑封装进 `profile-summary-unified` skill 的 `_from_torch_profile_text()` adapter,自动分类成 moe_gemm / dense_gemm / moe_routing / attention / norm / kv_cache / memcpy / elementwise

**剩余缺口**:
- sglang 路径已经有 `pytorch-profiling` skill 处理
- proton (Triton 自带 profiler) 评估完了但**没接入**(`docs/2026-06-08/nsys_deep_dive_and_proton.md` Part 2 — 等真要做 in-source 优化时再接入)

## B.4 内存压力 / HBM 带宽利用率 ❌ — 仍未解决

(需要 ncu — 见 B.2 — 等同一个 unlock)

## B.5 sglang/vLLM 内部 Python 状态 ❌ — 仍未解决

(需要 source instrumentation,这是 mentor coordination 级别的事)

## B.6 跨配置自动对比 ❌ → ✅ **2026-06-09 修正**

**完成**: 三个新 skill 一起解决:
- `regime-sweep-runner`: N (config × regime) 矩阵自动跑
- `cross-regime-anomaly`: 自动 rank 5 种 anomaly kind(winner_inversion / large_uniform_gap / regime_dependent_gap / reliability_flag / failed_cell)
- `profile-summary-unified` + `nsys-timeline-sql diff`: profile 层面的 before/after diff

**验证**: 用 2026-06-09 4-config 数据合成 sweep,跑 `cross-regime-anomaly` 正确识别 "sglang_cutlass 在所有 regime 上被 sglang_triton 大幅领先" → 自动推荐 `server-log-mining` 作为下一步。这正是我们之前手动诊断出的结论。

---

# Part C — 构建真实 Agent 时需要的信息(目前 mentor 帮忙补)

## C.1 nsys GUI 截图 / 时间线导出

**Mentor 能帮的**:
- 把 .nsys-rep 下载到本地,用 GUI 打开
- 截图 CPU/GPU 时间线给我看 (PNG / 描述)
- 或者用 nsys 的 `--export json` 导成 JSON 时间线 (我能读 JSON)

**为什么有价值**: 直接验证 max(CPU, GPU) 模型,不用再推算。也能看 NVTX range 内具体哪部分慢。

## C.2 别的硬件实测

**Mentor 能帮的**:
- 安排 B200/GB200 access (我现在只有 H200)
- 在 SM100 上跑同样 2×2 矩阵 (autotune × cudagraph)
- 验证 "SM100 hand-tuned 表的存在 是否真带来收益"

**为什么有价值**: 我们整个分析都在 SM90 一个 case。SM100 可能完全不同的 picture,而且这是 NVIDIA 真正投资 hand-tune 的地方。

## C.3 别的 model / dtype 实测

**Mentor 能帮的(或者下放给 agent 跑)**:
- DeepSeek-V3 (SM90 + 大 E + 大 K)
- Mixtral (SM90 + 小 E + 大 N)
- FP8 / NVFP4 quantized variants
- 跑同样的 4-way bench

**为什么有价值**: 验证我们结论 (sglang_cutlass 比 sglang_triton 慢 3.4×) 是不是 Qwen3-30B-A3B specific。

## C.4 实际请求 trace (生产场景)

**Mentor 能帮的**: 给一个真实 user traffic 的 prompt 分布、conc 分布,不要全是合成数据

**为什么有价值**: 我们 R_short/R_medium/R_long 都是合成 regime。真实 traffic 是 mixed (聊天 + 长文档总结 + code completion + ...),agent 需要在真实分布上做决策,不能只看几个 cherry-picked regime。

---

# Part D — 现状能做 vs 还差什么(给 mentor 看的清单)

## D.1 现状能可靠做到的

| 任务 | 能可靠做 | 备注 |
|---|---|---|
| 启动 sglang / vLLM server with 任意 flag | ✓ | 模板 launcher 脚本可复用 |
| 跑 e2e benchmark, 3 runs, mean ± std | ✓ | 但要小心 cold run |
| 录 nsys profile + 抽 kernel-time CSV | ✓ | 需要 SIGINT 触发 flush |
| 读 server.log 找配置/错误 | ✓ | grep + tail |
| 读 sglang/vLLM/flashinfer 源码定位决策点 | ✓ | view + grep |
| 搜 github issue / PR | ✓ | gh CLI |
| 写 Python 微基准直接调 kernel | ✓ | 需要熟悉 wrapper API |

## D.2 现状能做但不可靠 (容易漏 / 错)

| 任务 | 风险 | 怎么改善 |
|---|---|---|
| 从 nsys 数据推根因 | **高:容易过度解读** | 强制 "predict before look" |
| 跨 system A/B 比较 | **高:confounding variable 容易漏** | 强制画 N×M 矩阵 |
| 自动判断 "实测数字是否反常" | 中 | 需要 baseline / 理论值 |
| 自动决定 "什么时候停止深挖" | **高:容易钻牛角尖** | 需要 ROI 估算 + 时间限制 |

## D.3 现状做不到 (需要工具升级或人工协助)

| 任务 | 为什么做不到 | 谁能补 |
|---|---|---|
| 看 nsys 时间线 | 没 GUI | mentor 帮看 / 帮导 JSON |
| ncu kernel metrics | 没装 ncu | install + 学着用 |
| Python op-level profile | 没用过 torch.profiler | 学着写 (能做但没做) |
| 主动判断"这个数字反常" | 没有理论预期值数据库 | 建一个 "expected perf" reference |
| 主动建议"该比较 A 和 B" | 没有"比较矩阵"的明确触发条件 | 写 prompt 教 agent 看到 X 就比 Y |
| 跨 model / GPU 推广结论 | 没多模型/多 GPU 实测 | mentor 安排资源 |

---

# Part E — 这次 5 次"错误根因 → 实测打脸"的复盘

这是给 agent skill 设计的 negative training data。

| 错误 | 错在哪 | 实测打脸数据 | 教训 |
|---|---|---|---|
| #1 ROOT_CAUSE.md: "AutoTuner re-benchmark 9× kernel launch" | 没读 `is_tuning_mode` if | Fix 1 实测 -6% (反而变慢) + nsys kernel count 几乎不变 | **必须读完代码路径再下结论,看到一个调用就编故事是危险的** |
| #2 fix1_invalidated.md: "差距主要在 cudagraph 覆盖度" | 没控制 autotune 这个变量 | 2×2 矩阵显示 cudagraph 单独只 1.47× | **A/B 比较前列出所有变量** |
| #3 vllm_autotune_e2e_impact.md: "autotune 是主因 3.4×" | 同上,只切了 autotune 没切 cudagraph | 2×2 矩阵显示 autotune 单独只 1.09× | 同上 |
| #4 早期: "sglang Triton 比 vLLM Triton 慢 20-59%" | cold vs warm 没控制 | warm 起来打平 | **必须丢 cold run** |
| #5 早期: "hand-tuned 表是给 trtllm_gen 用的" | 没看 key 字符串 | key 是 `'trtllm::fused_moe::gemm1'` 但属于 cutlass path (历史命名) | **直接读 binding 源码确认归属,别猜** |

**5 次错误的共同模式**: **看到证据 → 编故事 → 没设计 falsification → 故事被打脸**。

修正方向: 任何 "X 导致 Y" 的论断,必须配一个 **specific 数字预测** + **最小代价的实验**,跑完看符不符合预测。

---

# 总结一句

| | 现在 |
|---|---|
| Agent profiling 能力 | L1-L3 (e2e bench, server log grep, kernel-time CSV) 能做。L4-L5 (op-level, kernel metrics) 没碰 |
| 主要 bottleneck | 不是工具,是**推理 discipline** — 没有"假设必须可证伪"的强制流程,导致 5 次推论被打脸 |
| 立刻能补的 | torch.profiler, ncu 学着用; nsys nvtx range; 给 agent 强制 "predict-then-verify" prompt |
| 长期需要 mentor 的 | GUI 时间线截图; B200/SM100 access; 多 model/dtype 实测; 真实 traffic trace |

接下来讨论:**这份盘点你还想加什么 / 删什么?** 然后我们再决定 skill 怎么写。
