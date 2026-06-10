# sglang Triton MoE — 4-regime nsys + ncu profiling sweep
## 2026-06-09 晚 实验

> 🇨🇳 中文版 · [English version](sglang_triton_4regime_profiling.md)

> **状态**: ✅ **完成** — 4 个 regime 全部跑完 nsys (200 MB .nsys-rep,按时间窗切 4 段) + ncu (`--set full`, `--kernel-name regex:.*`,每 regime 30–50 个 unique kernel)。4 份 `profile_unified.json` 全部带完整 `evidence_chain`(每个字段都能追溯到具体 skill)。总耗时 ~3 小时。

---

## 1. 实验配置

| 项 | 值 |
|---|---|
| 框架 | sglang 0.5.9 |
| 模型 | Qwen3-30B-A3B-Instruct-2507 (bf16) |
| 硬件 | NVIDIA H200 (GPU 1), SM 9.0, 132 SMs |
| MoE 后端 | `--moe-runner-backend triton`(默认) |
| cudagraph | **关闭** (`--disable-cuda-graph`),保证每个 kernel 启动都可见 |
| TP / mem / max-seq | 1 / 0.85 / 32 |

## 2. 4 个 regime(已剔除信号弱的)

来自 `regimes/qwen3_30b_moe_sglang_perf_sweep.yaml` — 挑选标准: 覆盖 perf 相关 axis(expert 利用率 × prefill-decode 比 × 并发):

| Regime | num_prompts | prompt_words | max_new | concurrency | 测试什么 |
|---|---|---|---|---|---|
| `R_short_decode`      |  8 |  100 | 256 |  1 | 极低 expert 利用率(batch=1 → 128 个 expert 中 8 个各拿 1 token) |
| `R_medium_balanced`   | 16 |  800 | 256 |  8 | 典型 batch=8 — 大部分 expert 活跃 |
| `R_long_prefill`      |  4 | 4000 |  32 |  4 | prefill 主导,attention kernel 可见 |
| `R_concurrent_decode` | 32 |  200 | 256 | 32 | 高并发 decode — MoE batch 行为 |

**有意丢弃的 regime**: short-input + short-output(跟 R_short_decode 信号重叠);中间值 concurrency sweep(不能区分 kernel)。

---

## 3. e2e-bench-runner (Phase 1)

用 `e2e-bench-runner` skill,`--regimes-file regimes/qwen3_30b_moe_sglang_perf_sweep.yaml`:

| Regime | req/s mean | tokens/s mean | stddev % | reliable? |
|---|---|---|---|---|
| R_short_decode      |  0.11 |   28 |  0.3% | ✅ |
| R_medium_balanced   |  0.80 |  205 |  0.9% | ✅ |
| R_long_prefill      |  2.74 |   88 | 10.3% | ❌ stddev > 8% |
| R_concurrent_decode |  3.20 |  820 |  1.5% | ✅ |

**观察**: `R_long_prefill` 按 skill 的 stddev gate 算**不可靠**。对于 prefill 主导的 regime,kernel 层面数据仍然有意义(一次 forward pass 就主导),但跨 run 的吞吐量对比不能信。

---

## 4. nsys per-regime (Phase 2)

sglang **一次性**包在 nsys 下启动; 4 个 regime workload 背靠背跑;**单个 200 MB .nsys-rep** 按 wall-time 对齐切成 4 段(见 `regime_windows_aligned.json`)。用 nsys-timeline-sql skill 对每段单独跑。

| Regime | gpu_active ms | gpu_util % | top_kernel | top % of active | launch ratio (graph/eager) |
|---|---|---|---|---|---|
| R_short_decode      | 7739 |  8.5% | `fused_moe_kernel` | 31.5% | 0.000(全 eager — cudagraph 关) |
| R_medium_balanced   | 3091 | 11.6% | `fused_moe_kernel` | 47.7% | 0.000 |
| R_long_prefill      |  214 | 12.1% | `fused_moe_kernel` | 47.4% | 0.000 |
| R_concurrent_decode | 2101 | 15.4% | `fused_moe_kernel` | 54.4% | 0.000 |

**观察**:
- `fused_moe_kernel`(Triton 生成的 MoE GEMM)在哪里都是 top kernel — 占 31% 到 54% active 时间。
- GPU util 都很低(8.5% – 15.4%),因为 cudagraph 关了,launch overhead 占主导。
- 并发更高的 regime,MoE 占比更高(每层每次 forward 有更多 token × 更多 expert 活跃)。

每个 per-regime `timeline_summary.json` 有 top-15 kernel、最大 10 个 idle gap、CPU API 计数、memcpy 聚合(见 `results/.../nsys/<regime>/timeline_summary.json`)。

---

## 5. ncu per-regime (Phase 3) — **全部 4 个完成**

每个 regime,sglang 通过 sudo ncu 包住 `sglang.bench_one_batch` 启动,带
`--profile --profile-activities CUDA_PROFILER --profile-stage {prefill|decode}`。
这样:
- ncu 用 `--profile-from-start off`,等 sglang 内部 `cudaProfilerStart` 触发
- 模型加载 + warmup 不在 profile 范围(自动跳过)
- 只采 bench 本身的窗口

NCU 参数(按用户要求):
- `--set full`(每 kernel ~7000 个 metric,每 kernel ~1 min)
- `--kernel-name regex:.*`(不过滤 — 触发窗口内每一个 kernel)
- `--launch-count 50` 给 R_long_prefill,`--launch-count 30` 给其他三个(看到 full set replay 太慢后下调)

**Per-regime 状态(最终)**:

| Regime | bench-one-batch stage | NCU 耗时 | unique kernel 数 |
|---|---|---|---|
| R_long_prefill      | prefill, B=4 in=8000 out=32   | ~60 min | 50 |
| R_concurrent_decode | decode,  B=32 in=400 out=256  | ~35 min | 30 |
| R_medium_balanced   | decode,  B=8  in=1600 out=256 | ~33 min | 30 |
| R_short_decode      | decode,  B=1  in=200  out=256 | ~35 min | 30 |

NCU 总耗时: ~2 h 45 min,覆盖 140 个 unique kernel profile。

### 跨 regime kernel 对比 — 头条发现

同一个 `fused_moe_kernel`(Triton 生成的 MoE GEMM)在不同 regime 行为**完全不同**:

| Regime | 有效 batch | SM% | DRAM% | Occupancy% | TC% | Headroom% | Verdict |
|---|---|---|---|---|---|---|---|
| **R_short_decode**      | 1   | 12.1 | 50.5 | 12.0 |  8.0 | 49.5 | low_occupancy |
| **R_medium_balanced**   | 8   | 13.5 | **67.5** | 19.9 | 10.1 | 32.5 | low_occupancy(濒临 memory_bound) |
| **R_concurrent_decode** | 32  | 16.8 | **79.8** | 44.8 | 12.8 | 20.2 | **memory_bound** |
| **R_long_prefill**      | 4 prefill (8000 tok) | **69.9** | 22.5 | 12.4 | **70.6** | 30.2 | low_occupancy(偏 compute,TC 在跑) |

**解读**:
- **decode regime**(batch=1, 8, 32),MoE kernel 是 **memory-bound** 或趋近 memory-bound。expert 权重加载占主导;tile shape 被压成 GEMV;TC 几乎不亮(8-13%)。
- **prefill regime**(8000 token × 4 prompt),同一个 kernel 变 **compute-bound**,TC 70.6%,SM 69.9%。每个 expert 工作量足、GEMM 形状合适。
- **含义**: 帮 decode 的 MoE-kernel 优化(memory 布局、权重预取、persistent kernel)和帮 prefill 的优化(tile 搜索、TC 调度)是**正交的**。要么都做,要么按 workload 分发。

### "同一个 kernel" — 真的是同一个吗?(Triton autotune specialization 证据)

`fused_moe_kernel` 在 sglang 源码里只是一个 `@triton.jit` 函数,但 Triton 会**autotune** — 运行时按 call-site 形状挑 (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) 组合。不同组合编译成不同 SASS、不同寄存器数量、不同 shared mem 布局 — 本质上**同名不同 kernel**。

我们用 NCU 的 `Block Size` / `Grid Size` / `registers/thread` 列直接验证:

| Regime | Block Size | Grid Size (X) | Registers/thread | 推断 num_warps |
|---|---|---|---|---|
| R_short_decode (B=1)       | (128, 1, 1) | 192–256       |  56     | 4 |
| R_medium_balanced (B=8)    | (128, 1, 1) | 1,536         |  64     | 4 |
| R_concurrent_decode (B=32) | (128, 1, 1) | 3,288         |  64     | 4 |
| **R_long_prefill**         | **(256, 1, 1)** | **12,768–17,024** | **194–196** | **8** |

关键观察:
- **Block 256 (prefill) vs 128 (decode)** — `num_warps=8` vs `num_warps=4`,完全不同 autotune specialization。
- **Registers/thread 194-196 (prefill) vs 56-64 (decode)** — prefill 用 ~3× 寄存器。强烈暗示 wgmma/TMA 软件流水线 + 大 tile(BLOCK_K 可能 64+)。196 是 H200 上限 255 的 77%。
- **Grid 17,024 (prefill) vs 192 (decode)** — prefill 比 decode 多 88× thread block,远超 132 个 SM。decode 192 个 block 几乎填不满 SM(H200 上,这个寄存器数下 one wave 能跑 ~528 个 block)。

**所以"同一个 Triton kernel"实际上是两个不同的 kernel 实现,共享源文件但运行时按 shape specialization 不同**。prefill specialization 因为有寄存器 + grid + tile shape 而能用满 TC。decode specialization 牺牲 TC 利用率换更低寄存器压力 + 更多并发 decode batch。

**对优化方向的精化**:
- "优化 fused_moe_kernel" 必须指定 WHICH specialization。改 prefill 256-block 196-reg 那个不会帮 decode 128-block 64-reg 那个。
- decode 真正的优化目标可能**不**在 Triton kernel 本身,而是:(a) expert 权重预取 / on-chip 驻留策略,(b) batch 组合(通过 prefill chunking 让单次 forward 多 token),(c) 换 MoE 后端(比如 flashinfer cutlass 配合 autotune 找 decode-tuned tactic — 早上那次调查走的路径)。

### 其他 kernel 的横切

**cuBLAS Hopper GEMM (`nvjet_*`)** — 这些是非 MoE 的 QKV/MLP linear:

| Regime | 最佳 nvjet SM% | TC% | Verdict |
|---|---|---|---|
| R_long_prefill      | 94.7 | **96.0** | 接近 peak |
| R_concurrent_decode | 7.7  | 13-17   | low_occupancy(batch 太小) |
| R_medium_balanced   | 8.2  |  3-5    | low_occupancy(batch 太小) |
| R_short_decode      | 8.0  |  3-6    | low_occupancy(batch=1 → GEMV) |

prefill 把 cuBLAS 跑满;decode batch 都太小,cuBLAS 跑不到 peak。这是**根本性物理限制**,不是优化目标。

**FlashAttention** (`cutlass::device_kernel<flash::*>`):

| Regime | SM% | DRAM% | TC% |
|---|---|---|---|
| R_long_prefill      | 69.1 |  3.6 | 69.2 |
| R_concurrent_decode | 28.7 | 38.4 | 34.1 |
| R_medium_balanced   | 23.9 | 35.0 | 30.8 |
| R_short_decode      |  2.4 |  1.2 |  3.9 |

prefill attention 是 compute-bound on TC;decode attention 因为 seq 短所以利用率低。Single-batch decode (R_short_decode) 基本是空闲的。

**RMSNorm / activation / rotary**(elementwise + 预期 memory-bound):
- R_long_prefill: DRAM 67-92%, TC <2% → tensor_core_idle(elementwise op 本就该如此)
- R_concurrent_decode: DRAM 1-6%(workload 太小,饱和不了) — 这些 kernel 在小 batch 不是瓶颈

### 通用观察

**所有 regime 里没有任何 kernel 跑到 Tensor Core peak**,除了:
- prefill 的 nvjet GEMM(96%)
- prefill 的 MoE(70%)
- prefill 的 flash-attn(69%)

其他每一个 kernel × regime 组合都低于 50% TC 利用率。headroom 来源有两个:
- (a) 小 batch 下天然 underutilization(decode),这是根本性的;
- (b) launch warp 太少填不满 SM(occupancy < 20%),这个可以通过 persistent kernel 或更大 grid config 解决。

---

## 6. profile-summary-unified per regime (Phase 4) — **完成**

跑 `scripts/unify_sweep.py`,产出
`results/2026-06-09_sglang_triton_sweep/unified/<regime>/profile_unified.json`
每 regime 一份。每个 unified JSON 合并:
- `subject` + `workload`: 框架/模型/regime 元数据
- `e2e`: 来自 `bench_summary.json`(req/s + reliability)
- `gpu_macro` + `kernel_breakdown`: 来自 `timeline_summary.json`(nsys)
- `kernel_micro`: 来自 `ncu_summary.json`(全 metric set,全 kernel)
- `evidence_chain`: 机器可读的 skill 归因,**首次** 4 字段全 ok=true 跑通

4 份 unified JSON 是**唯一权威产物**;下游消费(handoff 草稿、对比表、cross-regime-anomaly skill)都读这个,不读源文件。

## 7. Per-regime 横向对比

### 按 category 聚合(从 nsys;同样的数据流入 unified)

| Regime | moe_gemm | dense_gemm | attention | norm | moe_routing | other |
|---|---|---|---|---|---|---|
| R_short_decode       | 31.48% | 31.15% | 15.98% | 5.34% | 7.02% | 5.32% |
| R_medium_balanced    | 47.67% | 19.84% | 13.90% | 3.61% | 4.66% | 4.08% |
| R_long_prefill       | 47.43% | 21.04% | 14.81% | 3.56% | 4.68% | 2.65% |
| R_concurrent_decode  | 54.41% | 11.09% | 12.29% | 2.95% | 3.46% | 4.00% |

趋势: 随着有效 batch 增加(R_short → R_concurrent),MoE 占比增长(31 → 54%),dense GEMM 占比下降。原因: 单次 forward token 多了,MoE GEMM 每次启动有更多工作,dense linear GEMM 则被分摊。

### sglang triton 在各 regime 表现好坏

| 维度 | R_short_decode | R_medium_balanced | R_long_prefill | R_concurrent_decode |
|---|---|---|---|---|
| e2e req/s                | 0.11   | 0.80  | 2.74(噪声大) | 3.20 |
| GPU util %               | 8.5    | 11.6  | 12.1   | 15.4 |
| 总 launch(nsys)          | 1.68M  | 477k  | 34k    | 226k |
| Hot kernel               | fused_moe_kernel (31%) | fused_moe_kernel (48%) | fused_moe_kernel (47%) | fused_moe_kernel (54%) |
| MoE kernel verdict (ncu) | low_occupancy | low_occupancy | low_occupancy(偏 compute) | **memory_bound** |
| MoE TC 利用率            | 8% | 10% | **70%** | 13% |
| MoE headroom 估计        | 50% | 33% | 30% | 20% |

### 各 regime 改进方向候选

从 `profile_unified.json` 推出:

- **R_short_decode** (B=1, 低 expert 利用): kernel 层面优化有根本性限制(每个 expert 顶多见到 1 个 token)。更好的目标: 调度/打包(多个请求合并成一次 forward)。
- **R_medium_balanced** (B=8): MoE kernel TC 10%, occupancy 20%。可能受益于 batch=8 下用好 TC 的 CUTLASS 重写。verdict 跟早上 CUTLASS 调查发现一致。
- **R_long_prefill** (prefill 主导): kernel 已经 70% on TC。headroom 来自 elementwise fusion(rmsnorm + activation + rotary)。persistent kernel 也能减少 launch overhead(214ms 内 34k launch = 159k launches/sec)。
- **R_concurrent_decode** (B=32 decode): MoE memory-bound(DRAM 80%)。优化方向: 预取 expert 权重、persistent kernel 做稳态 decode、或换内存布局更好的 MoE 后端。

---

## 8. Skill 归因

| 阶段 | 用了哪些 skill | 产出 |
|---|---|---|
| Setup | (配置 + `regimes/` 的自定义 YAML) | regime 定义 |
| Phase 1 | `e2e-bench-runner` v1 (--regimes-file) | bench_summary.json 带 stddev gate |
| Phase 2 | `nsys-capture` 风格 + `nsys-timeline-sql` × 4 窗口 | 4 份 timeline_summary.json |
| Phase 3 | `ncu-microarch` 风格 包 `sglang.bench_one_batch`(自定义 adapter) | 4 份 ncu_summary.json |
| Phase 4 | `profile-summary-unified` × 4 | 4 份 profile_unified.json,带 evidence_chain |
| 报告 | 本文档 | sglang_triton_4regime_profiling.md |

---

## 9. 文件位置

```
results/2026-06-09_sglang_triton_sweep/
├── README.md                       # 文件索引(从这看起)
├── bench/                          # e2e-bench-runner 输出
│   ├── bench_summary.json
│   └── per_run/<regime>_runN.json
├── nsys/
│   ├── sglang_all4regimes.nsys-rep  (200 MB — gitignored)
│   ├── sglang_all4regimes.sqlite    (470 MB — gitignored)
│   ├── regime_windows.json
│   ├── regime_windows_aligned.json
│   └── <regime>/timeline_summary.json
├── ncu/<regime>/
│   ├── bench.log                    # ncu 进度
│   ├── <regime>_ncu.ncu-rep         (~MB — gitignored,需要 ncu-ui 打开)
│   ├── ncu_raw_full.csv             (wide format, ~7000 metric × N kernel)
│   ├── ncu_summary.json             # 由 scripts/ncu_csv_wide_to_summary.py 生成
│   └── ncu_report.md                # 人读 markdown(在这看 kernel 详情)
└── unified/<regime>/
    └── profile_unified.json         # 权威产物
```

脚本:
- `scripts/bench_ncu_one_regime.sh` — 单 regime NCU runner
- `scripts/bench_ncu_all_regimes.sh` — 批跑(串行)
- `scripts/ncu_csv_wide_to_summary.py` — CSV → ncu_summary.json adapter
- `scripts/unify_sweep.py` — 调 profile-summary-unified per regime
- `scripts/generate_ncu_reports.py` — 生成 per-regime ncu_report.md(人读版)
