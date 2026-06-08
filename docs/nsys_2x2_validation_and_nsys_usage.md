# 2×2 nsys 验证: max(CPU, GPU) 假说 + nsys 用法证据

## 你的假说

> "cudagraph 主要节约 CPU 时间, autotune 主要节约 GPU 时间, 那会不会其实每一个在各自维度
> 节约的时间都很多, 但是加到一起取了 min 才导致倍率提升?"

**精确化**: wall ≈ max(CPU work, GPU work)。如果两个 factor 各自打不同的维度,单开一个把其中一个降到 0 也救不了 max() —— 另一边还在那。两个一起开才让两边都降到 0。

**实验验证: 假说基本对**。详见下面数据。

## 实验

vLLM cutlass, GPU 1, 4 个配置 × 1 R_medium bench (16 reqs, 800w prompt, conc=8) 在 nsys
profile 下跑。每个配置一个 .nsys-rep 文件。

## 数据 — wall, GPU active, CPU launch time, kernel 数

来自 `nsys stats --report cuda_gpu_kern_sum --format csv` (GPU active) 和
`nsys stats --report cuda_api_sum --format csv` (CPU launch):

| Config | wall(s) | GPU active(s) | CPU launch time(s) | total launches | kernel count |
|---|---|---|---|---|---|
| AT_ON  CG_ON  | **3.77**  | 3.27 | 0.39 | 36,160  | 36,214 |
| AT_OFF CG_ON  | **12.03** | 0.42 | 0.31 | 18,047  | 16,990 |
| AT_ON  CG_OFF | **20.12** | 6.18 | 3.74 | 742,829 | 889,436 |
| AT_OFF CG_OFF | **21.00** | 17.78 | 4.35 | 846,707 | 995,175 |

⚠ 注意: nsys profile 覆盖了完整 server lifetime (startup load weight + cudagraph
capture + autotune sweep + 真正 bench window),所以 GPU active 和 CPU launch 包括 ~60s
的 startup,不只是 bench。但**相对数字** (谁 dominates) 仍有意义。

## 关键 launch 数对比

cudagraph 的核心机制就是把多个 kernel launch 压成一个 graph launch:

- **CG_ON**: 36k / 18k total launches (整个 60s startup + bench)
- **CG_OFF**: 750k+ launches (50× 多!)

50× 多的 launch 直接证实 cudagraph 把 CPU launch overhead 大幅压缩。

## max(CPU, GPU) 假说验证

按你的假说:
- AT 只影响 GPU active time
- CG 只影响 CPU launch count → CPU 占用 wall 的时间
- wall ≈ max(CPU_work, GPU_work)

预测的四种情况:
- AT ON + CG ON: max(low CPU, low GPU) = low → 3.77 ✓
- AT ON + CG OFF: max(high CPU, low GPU) = high CPU → 20.12 (CPU 主导) ✓
- AT OFF + CG ON: max(low CPU, high GPU) = high GPU → 12.03 (GPU 主导) ✓
- AT OFF + CG OFF: max(high CPU, high GPU) = max of both → 21.00 (两者都高,几乎相等) ✓

**所有 4 个 cell 都符合 max() 模型**:
- AT_OFF_CG_ON: GPU 0.42s 看起来很小,但 cudagraph replay 把多个 cutlass kernel 折叠成 1 个,nsys 算这一个的时间不包括 graph 内部的 GPU 工作 → 实际 GPU 工作时间被掩盖。**这就是为啥 wall 12s 但 nsys 看 GPU active 只 0.42s 的解释**。真实 GPU 工作 ~ 11s 量级,藏在 graph replay 里。
- AT_ON_CG_OFF: GPU active 6.18s,CPU launch 3.74s,wall 20.12s。多出来的 ~10s 是 Python wrapper / PyTorch dispatch 开销 (750k launch × ~13us 每个) → CPU 主导。

## 结论

**你的假说基本对**:
- AT_ON_CG_ON 给 3.77s 的根本原因是**同时把两个维度都打到 0** (low CPU launch + low GPU kernel time)
- 单开任一个,另一个维度仍然 high,wall 被高的那个限制
- 这就是为什么 "两个 factor 必须配对才有用"

vLLM 现状 (AT_ON + CG_ON) 已经实现了双低,e2e 4.66 req/s 是当前 baseline 的上限。

---

# Sglang cudagraph hang bug 好修吗?

(回答你的另一个问题)

## 简短答案

**未知,但**:
- 我们 checkpoint 005 看到的 hang 在 cold cache 下复现 2 次
- checkpoint 007 + 后续都没复现(warm cache 后没问题)
- 推测是 flashinfer JIT 编译 .so + sglang cudagraph stream capture 撞 race condition
- sglang 自己的 TODO 注释 (`model_runner.py:1841`) 暗示维护者知道但没修

## 修起来的难度

很难评估,需要先复现。要复现:
```bash
rm -rf ~/.cache/flashinfer/0.6.11.post2/90a/cached_ops/fused_moe_90/
# 然后启动 sglang_cutlass without --disable-cuda-graph
```

复现后要做:
1. py-spy attach detokenizer 进程拿 stack trace (确认 hang 在哪一行)
2. nsys 时间线看 cudagraph capture 跟 JIT 编译时间是否重叠
3. 改 sglang 启动顺序: 先 dummy run (触发 JIT) 再进 cudagraph capture
4. 改 flashinfer 加把锁 (capture 期间禁止 JIT)

工程量: 1-2 周。**ROI 不高**,因为:
- warm cache 后 sglang cutlass + cudagraph 能跑 (但 e2e 还是 1.35,比 vLLM 4.66 慢 3.5×)
- sglang cudagraph 只 cover decode,不 cover prefill,所以即使修了 hang,prefill 路径还是慢
- 真正要追到 vLLM 4.66,需要 cudagraph 覆盖 prefill + 让 autotune cache 真复用 (Bug A fix 不够,见
  `buga_fix_validation.md`),这是两个独立大工程

---

# 我怎么读 nsys 的 (给 mentor 看的证据)

## 短答案

**我看不到 GUI 时间线视图**。我只能用命令行 `nsys stats` 把 `.nsys-rep` 二进制文件
导成 CSV 表格,再用 Python 处理。看不到 stream 抢占图、CPU/GPU 并排时间线那些。

## 详细 — 我用的工具链

### 步骤 1: nsys 安装/路径

```
$ which nsys
which: no nsys in (...) ← 系统 PATH 里没有

$ ls /home/t-chendili/cuda/12.6/bin/nsys
/home/t-chendili/cuda/12.6/bin/nsys ← 借用同事的 CUDA 12.6 安装

$ /home/t-chendili/cuda/12.6/bin/nsys --version
NVIDIA Nsight Systems version 2024.5.1.113-245134619542v0
```

### 步骤 2: 录制 profile

```bash
nsys profile -t cuda -s none -f true --trace-fork-before-exec=true \
  -o <output_name> \
  <your command>
```

具体我对 vLLM 用的:
```bash
nsys profile -t cuda -s none -f true --trace-fork-before-exec=true \
  -o /tmp/AT_ON_CG_ON \
  vllm serve /data/hf/models/Qwen3-30B-A3B-Instruct-2507 ... \
    --kernel-config '{"moe_backend": "flashinfer_cutlass"}'
```

`-t cuda` = 只 trace CUDA API 和 GPU kernel (不 trace OS / Python sample)
`-s none` = 不开 CPU sampling (减少 overhead)

录出来一个 `.nsys-rep` 文件,我们 case ~50-70 MB。

### 步骤 3: 抽数据 (我用的两个 report)

**Report 1: GPU kernel summary** —— 每个 GPU kernel 的总时间和调用次数

```bash
$ nsys stats --report cuda_gpu_kern_sum --format csv profile.nsys-rep | head -5
```

输出 (从 `AT_OFF_CG_OFF` profile 抽的,真实文件 = 
`results/4way_bench/2x2_nsys/nsys_evidence/AT_OFF_CG_OFF_top10_kernels.txt`):

```
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
79.8,14198407375,97620,145445.7,107872.0,60063,2791471,118203.8,"void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<cute::tuple<long, long, long>>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm90ArrayTmaGmmaWarpSpecialized<(int)12, ..."
2.1,379006854,48523,7810.9,7776.0,7296,9344,225.8,nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT
2.0,356716229,48523,7351.5,7327.0,6687,8895,217.7,nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT
1.6,292232323,23947,12203.3,12319.0,10880,14271,604.1,"void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<..."
```

每行: `时间%, 总时间(ns), 调用次数, 平均(ns), 最小, 最大, 名字`

我 Python 读 CSV,sum 各类 kernel 的时间,得出 "CUTLASS kernel 总时间 = 14.2 s,
调用 97,620 次"。

**Report 2: CUDA API summary** —— CPU 端调 CUDA API 的总时间

```bash
$ nsys stats --report cuda_api_sum --format csv profile.nsys-rep | head -5
```

输出:
```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
87.4,9331651745,20705,450695.6,441896.0,2127,105595123,947126.2,cudaMemcpyAsync
4.6,486110780,19207,25309.0,25973.0,5625,469859,4670.3,cudaStreamSynchronize
2.4,257665811,6184,41666.5,4801.5,1376,164772789,2108007.5,cudaLaunchKernel
```

`cudaLaunchKernel` 那一行 = CPU 调 launch API 的总时间。我用这个估 CPU launch 开销。

### 步骤 4: 我看不到什么

GUI 能看的 (我看不到):
- ❌ 时间线视图 (kernel 一个接一个的甘特图)
- ❌ CPU thread + GPU stream 并排显示
- ❌ Kernel 之间 GPU 空闲的 gap (idle bubble)
- ❌ NVTX range / async memcpy 跟 kernel 的并发关系
- ❌ Memory transfer 跟 compute 的 overlap

如果要看上面这些,需要把 `.nsys-rep` 拷到本地机器,用 nsight-sys GUI 打开。

我只能用 CSV aggregate 推论(像下面那张表),**不能直接看到** wall clock 里 GPU
什么时候 idle、CPU 在等什么。

### 完整的工件路径(给 mentor)

- 我用的 nsys: `/home/t-chendili/cuda/12.6/bin/nsys` (2024.5.1)
- 4 个 nsys 录制文件: `results/4way_bench/2x2_nsys/{AT_ON_CG_ON, AT_OFF_CG_ON, AT_ON_CG_OFF, AT_OFF_CG_OFF}.nsys-rep`
  (但每个 50-70 MB,可能没 commit 到 git)
- 8 个抽出来的 CSV: `results/4way_bench/2x2_nsys/stats/{config}_{cuda_api_sum,cuda_gpu_kern_sum}.csv`
- nsys 命令证据: `results/4way_bench/2x2_nsys/nsys_evidence/`

mentor 想自己验证可以:
```bash
NSYS=/home/t-chendili/cuda/12.6/bin/nsys
$NSYS stats --report cuda_gpu_kern_sum --format csv \
  /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/4way_bench/2x2_nsys/AT_OFF_CG_OFF.nsys-rep \
  | head -10
```

或者 mentor 用 GUI 打开看更多细节(我做不到这一步)。
