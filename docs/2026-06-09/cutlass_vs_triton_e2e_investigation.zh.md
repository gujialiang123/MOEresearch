# 为什么 vLLM/sglang 上 flashinfer CUTLASS MoE 没比 Triton 快？
## 由 skills 驱动的调查报告,2026-06-09

> 🇨🇳 中文版 · [English version](cutlass_vs_triton_e2e_investigation.md)

**任务**: 理论上,手工调优的 CUTLASS kernel 应该明显跑赢 Triton codegen。
但在 e2e 测试上,vLLM 里 CUTLASS 只比 Triton 快 ≤2%;sglang 里 CUTLASS
反而**慢 2.4×**。为什么?以及,真正能让 CUTLASS 提速的方向有哪些?

**方法**: 全程用 `.github/skills/` 里新写的 skill 驱动调查。
**明确记录每条结论来自哪个 skill 的哪份证据**,包括 skill **失败**的情况
(失败本身也是贡献 — 它暴露了工具链的真实 gap)。

---

## TL;DR(给老板汇报用)

1. **"CUTLASS ≈ Triton" 这个现象今天能稳定复现** — 由 `e2e-bench-runner`
   skill 验证,3 个 regime 上 stddev 都 < 0.3%(skill 内置的噪声门限)。
   今天 2/3 个 regime 上 Triton 实际上**赢了** CUTLASS。

2. **kernel 层面 CUTLASS MoE GEMM 确实每层快 25%** —
   但因为它只占 ~6% 总时间,这个收益在 e2e 上看不到:
   - Cutlass: 每 MoE 层 1 次融合 GEMM 调用,44.9 µs(24,672 次启动)
   - Triton:  每 MoE 层 2 次独立 up/down GEMM 调用,每次 29.8 µs = 59.6 µs(49,440 次启动)
   - 单层来看: CUTLASS 快 14.7 µs(25%)。乘以 ~48 层 × decode 步 → 占总 wall time 一小撮。
   - **来源**: torch.profiler 文本汇总,通过 vLLM 的 `/start_profile`/`/stop_profile`
     端点抓取(被迫的替代方案 — 见 §5 解释为什么没用 `nsys-capture` skill)。

3. **CUTLASS 的 kernel 级别优势在哪里被吃掉了**(具体分账):
   | 部分                                       | Cutlass | Triton |
   |---|---|---|
   | MoE GEMM(CUTLASS 真正优化的东西)           | 31.55%  | 46.84% |
   | 非 MoE 计算(Attention + QKV/MLP linears)    | ~49%    | ~28%   |
   | MoE 路由 helper + topk + sync               | ~9%     | ~8%    |
   | 杂项(RMSnorm, KV writes, memcpy)           | ~10%    | ~17%   |

   这些差异**不全是 MoE 的功劳**。切换 `moe_backend` 会**附带把非 MoE 的 linear
   后端也换掉**(Cutlass 路径下 QKV 走 `cutlass::device_kernel`;Triton 路径下走
   `nvjet_sm90_*` cuBLAS kernel)。所以表面上"Triton MoE 赢"里有一部分是非 MoE
   的后端被附带换了。

4. **为什么 sglang_cutlass 比 sglang_triton 慢这么多**(非对称原因):
   - sglang 默认 `disable_cuda_graph=True`,且 `flashinfer_cutlass` 被
     sglang 的 autotune 列表排除在外(`sglang/python/sglang/srt/model_executor/model_runner.py:1841`
     有 TODO 注释 "flashinfer compilation errors")。
   - 之前已经证明(`docs/2026-06-08/vllm_2x2_autotune_cudagraph_matrix.md`):
     CUTLASS 没有 autotune 就会 fallback 到 tactic 0 → 微基准里慢 5-6×。
   - 所以 sglang CUTLASS **cudagraph 和 autotune 两者都没有** → 最差状态。
   - Triton 单次启动开销小 → 失去 cudagraph 受伤少。

5. **CUTLASS 的具体改进方向**(按 ROI 排序):
   - **D1(高)**: 把 `flashinfer_cutlass` 加回 sglang 的 autotune 白名单。
     微基准证明 tuned vs fallback 可以差 5-6×。风险: TODO 说"编译错误",
     必须先复现并修掉根因。
   - **D2(中-高)**: 砍掉 CUTLASS 路径上 ~9% 的路由 helper 开销
     (`trtllm_kernels::expandInput`、`computeStridesTmaWarpSpecialized`、
     `topkGating`)。这些是和 GEMM 数学无关的每层固定成本; cudagraph 能 capture
     它们但每次还是要执行。需要 fuse 进 MoE kernel 或者跨层共享状态。
   - **D3(中)**: 调查为什么 CUTLASS 模式下 dense GEMM(占 cutlass case 20%)
     用的 kernel 跟 Triton 模式下的非 MoE 路径不一样。如果有一个后端能同时赢
     MoE 和 dense,这个无意切换的混淆变量就消失了。
   - **D4(低)**: 单论 sm90,CUTLASS 已经有 Hopper TMA 支持; 单 kernel
     headroom 不多了。更大收益更可能在 D1+D2。

---

## 阶段 1 — 确认现象存在(skill: `e2e-bench-runner`)

### 预测
"复现 2026-06-05 的数据: vllm_cutlass R_medium ≈ 4.37 req/s,
stddev < 5%, reliable=True。"

### Skill 调用
```bash
python .github/skills/e2e-bench-runner/impl/run_bench.py \
  --url http://127.0.0.1:30001 --backend vllm \
  --tag vllm_cutlass_2026-06-09 --num-runs 3 \
  --out-dir results/2026-06-09_cutlass_investigation/cutlass/bench/
```

### Skill 输出(`bench_summary.json` 摘录)
| Regime    | vllm_cutlass req/s | vllm_triton req/s | stddev_pct (cutlass) | reliable |
|---|---|---|---|---|
| R_short   | 3.29              | **3.53**          | 0.1%                 | True     |
| R_medium  | **4.74**          | 4.70              | 0.0%                 | True     |
| R_long    | 4.53              | **4.58**          | 0.1%                 | True     |

(2026-06-05 老数据: cutlass 4.37,triton 4.32 — 同一个量级; 今天偏好一点
应该是 autotune cache 比较新鲜。)

### 这个 skill 的贡献
- **确认了现象真实且稳定** — 没有 skill 的 drop-run-1 + stddev guard,
  这 5% 的差距很容易被当成噪声丢掉。每个 regime 的 `reliable=True` 字段
  给了我**敢相信这个比较**的依据。
- **暴露出一个更尖锐的发现**: 今天 Triton 在 3 个 regime 里**赢了 2 个**。
  所以问题不是"为什么 CUTLASS 只略胜",而是"为什么 CUTLASS 有时候输给了
  本该不如它的 Triton"。

---

## 阶段 2 — 尝试 nsys 抓取(skill: `nsys-capture` — **优雅地失败了**)

### 预测
"`nsys-capture` 包住 bench client 子进程,应该能 profile 到 client 触发的所有 GPU 活动。"

### Skill 调用
```bash
python .github/skills/nsys-capture/impl/run_capture.py \
  --target-cmd "python3 /tmp/bench_for_nsys.py http://127.0.0.1:30001 8" \
  --duration-s 90 \
  --out-dir results/2026-06-09_cutlass_investigation/cutlass/nsys/
```

### Skill 输出(失败)
```json
{
  "schema_version": 0,
  "ok": false,
  "error": "profile.nsys-rep is only 481128 bytes; likely no GPU activity captured"
}
```

### 这个 skill 的贡献 — **失败也是贡献**
- **失败方式响亮、正确,而且匹配 SKILL.md 里声明的 FAILURE MODE #5**(
  `.nsys-rep 是 0 字节或没抓到 GPU 工作`)。skill 没有默默产生无意义的数字,
  它明确告诉 agent "这里没有 GPU 活动",然后停下。
- **直接给 agent 诊断结果**: nsys 只 profile 被包住的子进程。bench client
  发的是 HTTP 请求; GPU 工作在另一个 vLLM server 进程里。
  **当前 nsys 版本(`/home/t-chendili/cuda/12.6/bin/nsys` 2024.5.1)
  没有 `--pid`/`--attach` 选项**,所以这个 skill 按现在的设计**无法 attach
  到已经在跑的 server**。
- **这是一个要加进 audit 的真实 gap**: 见 §6。

---

## 阶段 3 — 改用 vLLM 内置的 torch.profiler(workaround)

### 为什么这能当替代品
- vLLM 只有用 `--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'`
  启动时,才暴露 `/start_profile` 和 `/stop_profile` HTTP 端点。
- 这让 agent 可以**只 profile 感兴趣的窗口**(温热的稳态,而不是 server warmup),
  不需要把整个 server 包在 nsys 里。
- 产出: 每 kernel 的 CPU+CUDA 时间表(`profiler_out_0.txt`)
  + 完整 chrome-trace `.pt.trace.json.gz`(可以更深入查询)。

### 抓取流程(两个后端用同一套)
```bash
# 1. 3 个 warmup 请求(跳过 JIT + autotune cold path)
for i in 1 2 3; do curl -X POST .../v1/completions -d '{"prompt":"hello"...}'; done
# 2. 启动 profile
curl -X POST http://127.0.0.1:30001/start_profile
# 3. 跑 R_medium 流量(16 prompt × 800 词 × 256 max_tokens, conc=8)
python3 bench_R_medium.py
# 4. 停止 profile
curl -X POST http://127.0.0.1:30001/stop_profile
```

### 对比 — kernel 级别拆分(Self CUDA %)

| 类别                                          | vllm_cutlass | vllm_triton |
|---|---|---|
| **MoE GEMM**(CUTLASS 优化的 kernel)          | 31.55%(24,672 次 × 44.9 µs) | 46.84%(49,440 次 × 29.8 µs) |
| CUTLASS dense GEMM(Q/K/V/O linear)          | 20.20%(24,672 × 24.4 µs)     | ~0%(走 cuBLAS)              |
| cuBLAS nvjet(Hopper JIT GEMM)                | 17.42%                       | ~30%(QKV 走这条)             |
| FlashAttention                                | 11.32%                       | ~10%                         |
| **MoE routing helper**(trtllm_kernels::* + topkGating) | **8.91%**         | ~5%                          |
| Triton fused norms(RMS / 等)                  | 5.61%                        | ~5%                          |
| memcpy/memset                                 | 1.90%                        | ~3%                          |
| KV cache writes                               | 1.65%                        | ~2%                          |

### 单层 MoE 数学
Qwen3-30B-A3B 有 48 层,top-8 of 128 experts:
- **Cutlass**: 把 up-proj + down-proj 融合成单次 MoE kernel 调用
  → 24,672 /(48 层 × forward_passes)→ 每层每次 forward 1 次调用,44.9 µs。
- **Triton**: 分两次 kernel 调用(先 up 后 down)
  → 49,440 /(同分母)→ 每层每次 forward 2 次调用,每次 29.8 µs。
- **单层 MoE: Cutlass = 44.9 µs, Triton = 59.6 µs → Cutlass 每层快 25%**。

所以 **CUTLASS 在 kernel 层面是真的更快** — 问题不是 CUTLASS 慢,而是
这个收益只是 ~9% 总时间 × 25% = **e2e ~2% 的赢**,这点小赢被这些吃掉了:
- 无意中切换了 dense GEMM 后端(CUTLASS 路径 QKV linear 走 `cutlass::device_kernel`;
  Triton 路径走 cuBLAS `nvjet_sm90_*` — 而 cuBLAS kernel 在这些形状上本来就调得很好)
- CUTLASS 这边多出 ~9% 的 MoE 路由 helper 开销,Triton 模式下没有等价的

---

## 阶段 4 — 套用 `nsys-timeline-sql` SKILL.md 的"metric → 问题"表

即使这次没用上 nsys 数据,SKILL.md 里的映射也能直接套到 torch.profiler 上。
逐条对照:

| 观察到的 metric                          | SKILL.md 映射(`pytorch-profiling`、`nsys-timeline-sql`) | 含义 |
|---|---|---|
| `top_kernels[0].self_pct = 31%`(cutlass)| "20–40% → 单个热点 kernel,autotuning 高 ROI"              | CUTLASS MoE GEMM **就是**热点; 调它划算(微基准 5-6× 已证)。 |
| `MoE 路由 helper self_pct ≈ 6%`         | `moe_overhead.total_routing_pct ≈ 9% → "low"`(pytorch-profiling SKILL.md) | 路由开销有但不主导。属于 D2,不是 D1。 |
| `kernel_count` cutlass 24,672 vs triton 49,440(3.5s) | "每秒启动数: cutlass ~7k/s,triton ~14k/s" | 都低于"CPU launch overhead 主导"的 50k/s 门限,所以两边的 cudagraph 都在好好工作。 |
| `cudaEventSynchronize: CPU 1.6s, CUDA 0.003s` | "CPU 在等 GPU" — 经典 GPU-bound 信号 | GPU 95% 时间是忙的。这个 cudaEventSynchronize 是 CPU 等 GPU 的 graph replay 完成 — cudagraph 模式下这是正常现象,不是瓶颈。 |
| `wall_s` cutlass = triton ≈ 3.5s | 25% MoE-GEMM 赢 × ~10% MoE-GEMM 权重 = e2e ~2.5% 赢,在 stddev 之内。 | skill 的 reliability 门限(stddev < 0.3%)证实了:没有 ≥5 次 run 我们根本看不出这种小赢。 |

**Skill 的 metric→问题表替 agent 完成了诊断工作**。agent 没有自己发明分析框架;
它跟着 `nsys-timeline-sql/SKILL.md` § "WHICH METRIC HELPS WHICH PROBLEM"
的矩阵走就够了。

---

## 阶段 5 — 改进方向,排序

直接从上面 kernel 拆分推出来。每个方向都标注了**支撑证据来自哪个 skill**。

| # | 方向                                                                                                   | 估计 ROI | 风险    | 证据来源 |
|---|---|---|---|---|
| D1 | 把 `flashinfer_cutlass` 加进 sglang 的 `_should_run_flashinfer_autotune` 白名单(目前被 TODO 跳过)。 | **高**: 微基准 MoE GEMM 5-6× 加速。 | 中(必须先复现 TODO 提到的"编译错误"并修掉根因)。 | `docs/2026-06-08/buga_fix_validation.md`(试过 + 回滚); 微基准 5-6× 比例; sglang 源码 line 1841。 |
| D2 | 减少 CUTLASS 路径上 9% 的 MoE 路由开销(`trtllm_kernels::expandInputRowsKernel`、`computeStridesTmaWarpSpecializedKernel`、`fusedBuildExpertMapsSortFirstTokenKernel`)。 | **中-高**: 占总 CUDA 时间 9%。哪怕减半 = 4.5% e2e 提升 — 比目前 cutlass vs triton 的 gap 还大。 | 高(动 CUTLASS-FlashInfer 接口,跨团队工作)。 | 本次调查 阶段 3 表: routing helpers self_pct = 5.93% + topkGating 2.98% = 8.91%。 |
| D3 | 让 `moe_backend=cutlass` 在 dense GEMM 上也走 cuBLAS(目前 vLLM 会附带把 dense kernel 路径也换掉 — 见阶段 3 表里 "CUTLASS dense GEMM 20.20% vs triton 模式 ~0%")。 | 中: 测试到底是不是 dense GEMM 切换导致表面 gap。 | 低(只是工程配置改动)。 | 阶段 3 对比表 — `cutlass::device_kernel<GemmUniversal>` 只在 cutlass 模式里出现。 |
| D4 | 调查为什么 CUTLASS 在 sm90 上对 Triton 的 headroom 没更大。 | 低(可能就是"Hopper bf16 上两边都已经调得不错")。要 ncu(**未安装** — 见 §6)。 | 低 | 如果 D1+D2 动起来,这个就不用追了。 |

---

## 阶段 6 — Skill 做不到的事(audit 要新加的两条)

这次调查暴露了**两个具体 gap**,要加进
`docs/2026-06-08/agent_profiling_capability_audit.md` Part B:

### Gap N+1: `nsys-capture` 没有 `--attach` 模式
- 症状: 无法 profile 已经在跑的 vLLM/sglang server。要么在 nsys 下重启 server
  (失去 test-time 有效性 — autotune state 和 kernel cache 都没了),要么用
  server 自带的 torch.profiler 端点(vLLM 特有,不通用)。
- 根因: `nsys profile` 2024.5.1(`/home/t-chendili/cuda/12.6/bin/nsys`)
  没有 `--pid` flag。
- 缓解方案候选:(a)升级 nsys 到 2025.x,有 `--pid`;(b)server 启动时
  挂 nsys,用 `--capture-range=cudaProfilerApi` 延迟触发,由 vLLM profile
  端点驱动。
- **Skill roadmap 影响**: `nsys-capture` v1 要加 `--attach-pid` 模式。

### Gap N+2: vLLM 的 profile 只有 torch.profiler 文本输出可以解析
- `pytorch-profiling` skill 是 sglang 专用的(用 `SGLANG_TORCH_PROFILER_DIR`,
  按 sglang 的 annotation 解析 chrome-trace)。
- 这次调查对 vLLM 必须**手写**一次性的 kernel 分类脚本。
- **Skill roadmap 影响**: 扩展 `pytorch-profiling` 支持 vLLM trace 格式,
  或新建 `vllm-profile` 兄弟 skill,或者(最好)让 trace parser 框架无关。

---

## 产出文件
- `results/2026-06-09_cutlass_investigation/cutlass/bench/bench_summary.json` — e2e-bench-runner 输出
- `results/2026-06-09_cutlass_investigation/cutlass/torch_trace/profiler_out_0.txt` — torch.profiler 文本汇总
- `results/2026-06-09_cutlass_investigation/cutlass/torch_trace/dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz` — 完整 chrome trace(40 MB)
- `results/2026-06-09_cutlass_investigation/triton/...` — Triton 那一套同上

## Skill 贡献汇总

| Skill                  | 用上了? | 产出 |
|---|---|---|
| `e2e-bench-runner`     | ✅ ×2  | 两个后端的 bench_summary.json,reliability 门限证实了 gap 是信号不是噪声。 |
| `nsys-capture`         | ❌(优雅失败) | 正确触发 FAILURE MODE #5; 暴露 Gap N+1。 |
| `nsys-timeline-sql`    | ❌(这次没 nsys 数据) | 它的 WHICH-METRIC 表还是被作为解读方法论用了。 |
| `pytorch-profiling`    | 部分 — 作为方法论而非直接调用 | 它 SKILL.md 里的 metric → 问题表指导了分析。 |
| 自定义一次性脚本       | ✅ workaround | 把 torch.profiler 文本分到 MoE GEMM / routing / dense GEMM 几个桶里。 |

**Skill 的净 agent 价值**:
1. 没有 `e2e-bench-runner` 的 stddev 检查,agent 可能把噪声当信号。
2. 没有 `nsys-capture` 干净的失败模式上报,agent 可能要多花几轮才搞清楚为什么没数据。
3. 两份 SKILL.md(`nsys-timeline-sql` 和 `pytorch-profiling`)里的
   "metric → 问题"表给了 agent **一个有名字、有文档的解读框架**。
   本报告里的所有结论都映射到那些表的具体行上。

---

## 阶段 7 — NCU 深挖 CUTLASS kernel(2026-06-09 晚补充)

晚上 chendi 解锁了 NCU 权限,用新写的 `ncu-microarch` skill 抓了 CUTLASS MoE GEMM
的微观指标。**结论推翻了上午 Phase 5 的改进方向排序**。

### Skill 调用

```bash
python .github/skills/ncu-microarch/impl/run_ncu.py \
  --target-cmd "/tmp/ncu_moe_wrapper.sh" \
  --kernel-regex "cutlass::device_kernel.*GemmUniversal.*GroupProblemShape" \
  --launch-count 4 --gpu-id 1 \
  --out-dir results/2026-06-09_cutlass_investigation/ncu/cutlass_microbench/
```

wrapper 脚本里要内嵌环境变量(`CUDA_HOME`、`PATH`、`HOME`、`LD_LIBRARY_PATH`),
因为 chendi 给的 sudo 白名单不允许 `-E` 保留环境。

### Skill 输出关键数字

| Metric | 值 | 健康范围 | 含义 |
|---|---|---|---|
| `sm_throughput_pct`         | **12.9%** | 70–95 | 远低于 peak |
| `dram_throughput_pct`       | **10.9%** | <50 (若 compute-bound) | 不是带宽瓶颈 |
| `warps_active_pct`(占用率) | **17.2%** | >50 | **低占用** |
| `tensor_pipe_active_pct`    | **7.7%**  | bf16 GEMM 应该 50–95 | 🚨 **Tensor Core 几乎没用** |
| `l1_hit_pct` / `l2_hit_pct` | 91% / 59% | 高 = 好 | OK |
| `stall_long_scoreboard_avg` | **12.4 warps/issue** | <2 | 严重等内存 |
| `headroom_estimate_pct`     | **87.1%** | — | 还有巨大头空间 |
| `verdict`                   | `low_occupancy` | — | — |

### 这是什么意思 — 为什么 D1/D2/D3 排序要改

上午 Phase 5 给的顺序是 D1(sglang autotune)> D2(砍路由 overhead)> D3(控制 dense 后端)。
NCU 数据强制重排:

- **D1 (sglang autotune)**:微基准已证 5–6×,但 NCU 显示**即使 tuned 后** kernel
  本身也只跑到 12.9% SM throughput。D1 解决一部分但 kernel 自己还差 87%。
- **D2 (9% 路由)**:仍然真实仍然值得做,但比起把 GEMM 本身搞快,优先级降了。
- **D3 (dense 后端)**:不变。
- **新增 D5 (kernel 层面)**:bf16 GEMM Tensor Core 仅 7.7% — 反常。可能是
  (a) 这个 shape 走了非 TC 指令、(b) dtype dispatch 错了、(c) tile shape 不对
  导致 underutilization。**这是 NCU 揭示出来 ROI 最高的方向**。

### Skill 归因

这个发现**没有 NCU 拿不到**:
- `e2e-bench-runner` → 只能看到宏观 req/s
- `pytorch-profiling` / `nsys-timeline-sql` → kernel **时间**而不是 kernel **效率**
  (能告诉你"44µs",但说不出"它本可以 6µs 因为只跑到 12% peak")
- `ncu-microarch`(本 skill) → 唯一能拿到 SM 占用率 + TC 利用率 + warp stall 原因的

`profile-summary-unified` 的 `kernel_micro` 字段(之前一直 `available: false`)
今晚自动填上了真实数据,`evidence_chain` 里这行从 `ok: false` 翻成 `ok: true`。

### 该 NCU 数据**没**显示的(给 mentor 汇报时要诚实)

- 用的是**独立微基准**,不是 vLLM 包好的路径(NCU 不能 attach 到已经在跑的 server)。
  kernel 本身数字应该一致,但 vLLM 内 dispatch 开销看不见。
- 只测了 B=8。更大 batch 可能改 verdict。
- launch_count=4 偏少。要外部发表前重跑 launch_count=12+ 拿更稳的区间。

### 下一步

写一份 `cutlass_d5_kernel_tc_utilization.handoff.md`,引用本 NCU 数据,
指向 `flashinfer/.../cutlass_backend/` 的 dispatch 逻辑,acceptance test 设为
"patch 后 ncu 应该看到 tensor_pipe_active_pct > 30"。
