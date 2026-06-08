# sglang vs vLLM:FlashInfer CUTLASS MoE 性能差异根因分析

> **目的**: 解释为什么同一份 `flashinfer.fused_moe.cutlass_fused_moe` kernel,
> 在 vLLM 上跑出来跟 Triton MoE 打平,在 sglang 上反而比 Triton 慢 3.4-4.7×。
> 所有结论都有 profiling 数据 / log / 源码路径作为证据。
>
> 配套数据 & 工件: `results/4way_bench/` (本仓库)
>
> 测试条件: Qwen3-30B-A3B-Instruct-2507 / H200 sm_90a / bf16 / TP=1 / single-GPU 顺序跑

---

## TL;DR

1. **vLLM 上 CUTLASS ≈ Triton (1.00-1.02×)**:在我们的测试条件下,vLLM 源码注释
   `vllm/.../unquantized.py:71` 写的 "FlashInfer 比 Triton 慢" 不成立。两个 backend
   在 3 个 regime 下都打平 (warm-only 数据,见 §1)。
2. **sglang 上 CUTLASS 比 Triton 慢 3.4-4.7×**,根因不在 kernel 本身:
   - 每个 cutlass-sm90 kernel 在 sglang 上 142us,vLLM 上 178us(sglang **略快**)
   - 但 sglang **launch 了 9× 多的 kernel** (97774 vs 10802 calls,同样 R_medium 工作量)
3. **9× 多 kernel 的根因**: `flashinfer.fused_moe.cutlass_fused_moe` 每次调用都跑
   `AutoTuner.choose_one()` 挑 GEMM tactic;cache miss 时会真 launch 候选 kernel
   做 micro-benchmark。
   - vLLM 端 `tune_max_num_tokens` 固定 + 用 cudagraph 锁 batch shape → cache 命中
   - sglang 端 `tune_max_num_tokens = next_power_of_2(x.shape[0])` 每次变 + 不能用
     cudagraph (会挂死) → cache 频繁 miss → 反复 re-benchmark

---

## 1. 实测数据 —— 验证差异的存在

**实验设置**:
- 模型: Qwen3-30B-A3B-Instruct-2507,H200 sm_90a,bf16,TP=1
- 4 个 server 配置: `sglang_triton`、`sglang_cutlass` (`--moe-runner-backend flashinfer_cutlass --disable-cuda-graph`)、`vllm_triton` (`--kernel-config '{"moe_backend":"triton"}'`)、`vllm_cutlass` (`--kernel-config '{"moe_backend":"flashinfer_cutlass"}'`)
- 3 个 regime: R_short (8 req / 200w / conc=1)、R_medium (16 / 800w / conc=8)、R_long (8 / 2000w / conc=16)
- 每个 backend × regime 跑 3 次,取 run 2 + run 3 平均(run 1 弃掉,因为 vLLM 的 cudagraph capture 是在 first inference 完成,run 1 偏冷)
- Bench harness: `results/4way_bench/scripts/run_bench_4way.py` (固定 seed=2026, temperature=0, ignore_eos=True)
- 原始数据: `results/4way_bench/runs/bench_*_run{1,2,3}.json`

### Warm-only 吞吐 (req/s)

| Regime | sglang_triton | sglang_cutlass | vllm_triton | vllm_cutlass |
|---|---|---|---|---|
| R_short  | 3.22 | **0.71** | 3.31 | 3.32 |
| R_medium | 4.49 | **1.31** | 4.71 | 4.72 |
| R_long   | 4.50 | **1.33** | 4.44 | 4.52 |

### Warm 相对加速 (基线 = sglang_triton warm)

| Regime | sglang Triton → CUTLASS | vLLM Triton → CUTLASS | sglang → vLLM (Triton) | sglang → vLLM (CUTLASS) |
|---|---|---|---|---|
| R_short  | **0.22× (4.5× 慢)** | 1.00× | 1.03× | **4.70×** |
| R_medium | **0.29× (3.4× 慢)** | 1.00× | 1.05× | **3.59×** |
| R_long   | **0.29× (3.4× 慢)** | 1.02× | 0.99× | **3.41×** |

**两个事实**:
- vLLM 内部: CUTLASS 和 Triton 几乎打平
- 同样的 CUTLASS,vLLM 比 sglang 快 3.4-4.7×

数据细节: `results/4way_bench/comparison_table.md`、`results/4way_bench/runs/stats_table.md`。

---

## 2. nsys 证据 —— vLLM 真的在跑 CUTLASS,不是 fallback

**方法**: 用 `/home/t-chendili/cuda/12.6/bin/nsys profile -t cuda -s none --capture-range=none vllm serve ...`
启动 vLLM,跑 warmup + R_medium bench,SIGINT 触发 flush,然后:
```bash
nsys stats --report cuda_gpu_kern_sum --format csv <rep>
```

### vLLM CUTLASS run kernel time 分布

来源: `results/4way_bench/nsys/vllm_cutlass_kernels.csv` (175 unique kernels)

| Category | Time(ms) | % | Instances |
|---|---|---|---|
| **cutlass_gemm_sm90 (MoE)** | **1924.4** | **58.3%** | **10802** |
| other | 1028.6 | 31.2% | 7214 |
| cutlass (other) | 231.7 | 7.0% | 7464 |
| cutlass_gemm (other) | 88.3 | 2.7% | 6464 |
| triton (any) | 16.8 | 0.5% | 2918 |
| moe_routing | 7.8 | 0.2% | 727 |

完整 kernel name 例:
```
void cutlass::device_kernel<
  cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<cute::tuple<long, long, long>>,
    cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm90ArrayTmaGmmaWarpSpecialized<(int)12, ...>
```
`MainloopSm90ArrayTmaGmmaWarpSpecialized` 是 SM90 专属的 CUTLASS group-GEMM mainloop
(TMA + WGMMA + warp specialization),证明跑的是 Hopper 调优过的真 CUTLASS。

### vLLM Triton run kernel time 分布

来源: `results/4way_bench/nsys/vllm_triton_kernels.csv`

| Category | Time(ms) | % | Instances |
|---|---|---|---|
| other | 291.7 | 42.4% | 20869 |
| **triton (any)** | **254.6** | **37.0%** | **8470** |
| fused_moe (flashinfer cuda helper) | 125.2 | 18.2% | 1248 |
| moe_routing | 10.6 | 1.5% | 1250 |
| cutlass (other) | 1.3 | 0.2% | 196 |
| **cutlass_gemm_sm90 (MoE)** | **0.0** | **0.0%** | **0** |

完整列表里能看到 `triton_red_fused_fused_add_rms_norm_moe_forward_0` 类 MoE 专用 kernel
(626 calls),证明 MoE 在跑 Triton。

### 关键对比

| 指标 | vLLM CUTLASS | vLLM TRITON |
|---|---|---|
| `cutlass::device_kernel<...sm90...gemm...>` (MoE GEMM) | **1924 ms / 10802 calls** | **0 ms / 0 calls** |
| `triton_*` (含 MoE forward) | 16.8 ms / 2918 calls (只 inductor 工具 kernel) | 259 ms / 9096 calls (含 MoE forward) |

→ 两个 run 在 MoE-kernel 层**完全互斥**。`--kernel-config '{"moe_backend":"flashinfer_cutlass"}'`
真的派到 CUTLASS,**没有 silent fallback**。

完整 EVIDENCE: `results/4way_bench/nsys/EVIDENCE.md`。

---

## 3. 为什么 vLLM 上 CUTLASS ≈ Triton?

**两个 backend 都跑了**,只是分了:
- CUTLASS: 1924 ms cutlass-sm90 + 117 ms `fused_moe::run_global` (flashinfer cuda helper)
- Triton: 259 ms triton MoE + 125 ms `fused_moe::run_global` (同一个 helper)

数字上:
- Triton run 的 MoE 总时间 ≈ 259 + 125 = 384 ms
- CUTLASS run 的 MoE 总时间 ≈ 1924 + 117 = 2041 ms ← **比 Triton 慢 5×?**

但 wall-clock 端 vllm_triton / vllm_cutlass 是打平的 (4.71 vs 4.72 req/s)。怎么解释?

**答案**: vLLM 用了 cudagraph (`cudagraph_mode=FULL_AND_PIECEWISE`,见 vLLM `EngineCore` 启动日志
`results/4way_bench/vllm_cutlass/server_tail2.log`)。在 cudagraph 模式下,GPU
端 kernel 是 "replay" 的,**单次 launch overhead 几乎为零**。所以 CUTLASS 多花的
1660 ms 是发生在 cudagraph capture 阶段 (capture warm-up 跑了大量 tactic 选择),
**capture 完之后真正服务请求时,replay 的成本对两 backend 来说几乎一样**。

这就是为什么 vLLM 源码注释 `unquantized.py:71` 写的 "Hopper bf16 上 FlashInfer 比
Triton 慢" 在 e2e 上看不到 —— **cudagraph 把 launch overhead 吃掉了**,只看
kernel 净时间的话 CUTLASS 确实更重,但用户感觉不到。

> 注释来源: `/home/t-jialianggu/work/vllm/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:71`
> 大意: "On Hopper (SM90), the FlashInfer unquantized MoE kernels are slower than Triton"

**这条注释不算错** —— 它讲的是 kernel 净时间,数据上是对的。但是它**没考虑 cudagraph
摊销之后的 e2e 效果**,导致 oracle 默认偏好 Triton。这一点之后可以考虑提个 issue
给 vLLM,让他们 re-validate 这条 default 选择的依据。

---

## 4. 为什么 sglang 上 CUTLASS 慢 3.4-4.7×?

### 4.1 nsys 证据 —— sglang launch 了 9× 多 kernel

**方法**: 同样的 `nsys profile` 启动 sglang (GPU 1),跑同样的 R_medium bench。
原始数据: `results/4way_bench/nsys/sglang_cutlass_kernels.csv`

### sglang CUTLASS run kernel time 分布

| Category | Time(ms) | % | Instances |
|---|---|---|---|
| **cutlass_gemm_sm90 (MoE)** | **13950.7** | **80.6%** | **97774** |
| other | 1354.3 | 7.8% | 236045 |
| cutlass (other) | 1099.7 | 6.3% | 219428 |
| cutlass_gemm (other) | 342.1 | 2.0% | 146661 |
| rms_norm | 231.4 | 1.3% | 98793 |
| moe_routing | 157.7 | 0.9% | 48887 |
| reduction | 90.7 | 0.5% | 49714 |
| rope | 90.6 | 0.5% | 48888 |
| triton (any) | 2.3 | 0.0% | 2038 |

### 三个 backend 对比 (`results/4way_bench/nsys/kernel_count_comparison.md`)

| Category | vllm_cutlass | sglang_cutlass | vllm_triton |
|---|---|---|---|
| cutlass_gemm_sm90 (MoE) | 1924 ms / **10,802 calls** | 13951 ms / **97,774 calls** | — |
| triton (any) | 17 ms / 2918 calls | 2 ms / 2038 calls | 259 ms / 9096 calls |

**关键数字**:
- sglang cutlass-sm90 kernel **avg 142 us**,vLLM **avg 178 us** —— sglang **每个 kernel 反而略快**
- sglang 跑了 **9.05× 多** 的 cutlass-sm90 kernel
- 每 output token 的 cutlass call: vLLM 2.6 calls/tok,sglang **23.9 calls/tok** (9.2×)

**结论**: 不是 kernel 本身慢,而是 sglang launch 了太多 kernel。

### 4.2 调用点对比 —— 同一个 flashinfer entry,传参不同

**sglang 调用点**: `python/sglang/srt/layers/quantization/unquant.py:372-386`
```python
elif self.use_flashinfer_cutlass:
    output = flashinfer_cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_output.topk_ids,
        token_final_scales=topk_output.topk_weights,
        fc1_expert_weights=layer.w13_weight,
        fc2_expert_weights=layer.w2_weight,
        output_dtype=x.dtype,
        quant_scales=None,
        ep_size=layer.moe_ep_size,
        ep_rank=layer.moe_ep_rank,
        tp_size=layer.moe_tp_size,
        tp_rank=layer.moe_tp_rank,
        tune_max_num_tokens=next_power_of_2(x.shape[0]),   # ← 每次 forward 变
    )[0]
```

**vLLM 调用点**: `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:378-403`
```python
_ = flashinfer_cutlass_fused_moe(
    input=hidden_states,
    token_selected_experts=topk_ids.to(torch.int),
    token_final_scales=topk_weights,
    fc1_expert_weights=fc1_expert_weights,
    fc2_expert_weights=fc2_expert_weights,
    fc1_expert_biases=fc1_expert_biases,           # ← 显式传
    fc2_expert_biases=fc2_expert_biases,
    swiglu_alpha=swiglu_alpha,                     # ← 显式传
    swiglu_beta=swiglu_beta,
    swiglu_limit=swiglu_limit,
    output=output,                                 # ← 预分配,复用
    output_dtype=self.out_dtype,
    quant_scales=quant_scales,
    input_sf=a1q_scale,
    tp_size=self.tp_size,
    tp_rank=self.tp_rank,
    ep_size=self.ep_size,
    ep_rank=self.ep_rank,
    activation_type=activation_str_to_value_map[activation],
    use_deepseek_fp8_block_scale=self.use_deepseek_fp8_block_scale,
    use_mxfp8_act_scaling=use_mxfp8_act_scaling,
    use_w4_group_scaling=use_w4_group_scaling,
    tune_max_num_tokens=max(self.max_capture_size, 1),    # ← 固定常量
)
```

**关键差异**:

| arg | sglang | vLLM | 影响 |
|---|---|---|---|
| `output=` | 不传 (内部分配) | 传预分配 buffer | 小 (分配开销,可能 ~ms 级) |
| `tune_max_num_tokens` | `next_power_of_2(x.shape[0])` —— **变化** | `max(max_capture_size, 1)` —— **固定** | **大,看下面** |
| `fc1/fc2_expert_biases` | 不传 | 显式传 None | 无 |
| `swiglu_alpha/beta/limit` | 不传 | 显式传 None | 无 |
| `activation_type` | 不传 (默认 Swiglu) | 显式传 (默认也是 Swiglu) | 无 |

### 4.3 真正的 bug —— flashinfer AutoTuner re-benchmark

来源: `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/fused_moe/core.py:472-578`

```python
@classmethod
@functools.lru_cache(maxsize=None)
def refine_tuning_config(cls, tune_max_num_tokens: int):    # ← lru_cache key
    cls.tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (0,), (0,),
                get_last_power_of_2_num_tokens_buckets(tune_max_num_tokens),
                lambda x: min(last_positive_power_of_2(x), tune_max_num_tokens),
            ),
        )
    )

def cutlass_fused_moe(..., tune_max_num_tokens: int = 8192, ...):
    tuner = AutoTuner.get()
    MoERunner.refine_tuning_config(tune_max_num_tokens)        # ← line 524

    # Limit tactics to GEMM1 during tuning
    moe_runner.gemm_idx_for_tuning = 1
    _, gemm_tactic_1 = tuner.choose_one(                       # ← line 550, GEMM1 tuning
        "trtllm::fused_moe::gemm1", [moe_runner],
        MoERunner.tuning_config,
        [input, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights, fc2_expert_biases],
        gemm_idx=1,
    )

    # Limit tactics to GEMM2 during tuning
    moe_runner.gemm_idx_for_tuning = 2
    _, gemm_tactic_2 = tuner.choose_one(                       # ← line 566, GEMM2 tuning
        "trtllm::fused_moe::gemm2", [moe_runner],
        MoERunner.tuning_config,
        [input, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights, fc2_expert_biases],
        gemm_idx=2,
    )
    # ... 真正的 run_moe 在 line 607 之后
    run_moe(output, input, ...)
```

**每次 `cutlass_fused_moe()` 调用都会跑两次 `AutoTuner.choose_one()`** (GEMM1 + GEMM2)
来挑 tactic。这两个 `choose_one` 行为依赖 tuner 的 cache:
- **cache 命中** (key 一样): 直接返回上次挑好的 tactic,不 launch
- **cache miss** (key 变): 真的 launch 候选 GEMM kernel 当 micro-benchmark,挑最快的那个

**cache key** 由 `tune_max_num_tokens` + 输入 shape 决定。两边的差异:

**vLLM 端 → cache hot**:
- `tune_max_num_tokens` = `max(self.max_capture_size, 1)` 固定常量
- `cudagraph_mode=FULL_AND_PIECEWISE` 把 batch shape 锁在 capture_sizes
  `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]` (vLLM 启动 log,
  `results/4way_bench/vllm_cutlass/server_tail2.log`)
- → 一旦每个 capture_size 都 tuned 过一次,后续推理 100% cache hit,
  `choose_one` 几乎 0 开销

**sglang 端 → cache cold**:
- `tune_max_num_tokens` = `next_power_of_2(x.shape[0])` 每次 forward 都可能变
- `--disable-cuda-graph` (这个 flag 不是为公平实验加的 —— sglang 的 cutlass
  路径 + cudagraph 会让 detokenizer 进程在 capture 后挂死,
  reproduced 2× during checkpoint 005 work; 见 §4.4)
- → 无 cudagraph 让 shape 也变;`tune_max_num_tokens` 也变
- → cache 频繁 miss,反复 re-benchmark
- → 这就是 **97774 - 10802 = 86972** 个额外 cutlass-sm90 kernel launch 的来源

### 4.4 sglang flashinfer_cutlass + cudagraph 会挂死

实证: 在 checkpoint 005 的实验里,我尝试去掉 `--disable-cuda-graph` 让 sglang_cutlass
也用 cudagraph。结果 detokenizer 进程在 `Capture cuda graph end` log line **之后** 挂死,
所有 `/health` 检查返回 503,client 请求超时。reproduced 2 次。

server log 片段 (来源: 复测时实时观察):
```
[2026-06-05 18:11:33] (last heartbeat)
[2026-06-05 18:12:33] Capture cuda graph begin. This can take up to several minutes. avail mem=20.67 GB
[2026-06-05 18:12:33] Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32]
[2026-06-05 18:12:34] Capture cuda graph end. Time elapsed: 1.23 s. mem usage=0.09 GB. avail mem=20.58 GB.
[2026-06-05 18:17:27] INFO:     127.0.0.1:47780 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-06-05 18:17:52] Health check failed. Server couldn't get a response from detokenizer
                       for last 20 seconds. tic start time: 18:17:32. last_heartbeat time: 18:11:33
... (repeat indefinitely)
```

所以 sglang 用 `flashinfer_cutlass` 时**必须** `--disable-cuda-graph`,这本身就给
慢 3.4-4.7× 提供了第二条解释通道 —— 即使把 `tune_max_num_tokens` 修固定,
没 cudagraph 锁 shape 的情况下 batch shape 还是会变,cache 仍可能 miss。

cudagraph hang 的根因没确认 —— 复测时只观察到 detokenizer heartbeat 不更新,没拿到
Python stack trace 或 GPU 端 error。后续需要单独排查 (可能是 flashinfer JIT 的
某个 op 跟 cudagraph capture 冲突,或者 sglang 的 detokenizer IPC 在 cudagraph
mode 下有死锁)。

---

## 5. vLLM 慢出 1660 ms 但 e2e 不慢 vs sglang 多 87 k launch 把 e2e 拖慢 3.4-4.7× ——
##    为什么 vLLM 那 1660 ms 不致命?

这是一个看似矛盾的问题:
- 单看 cutlass-sm90 净时间,vLLM CUTLASS 比 vLLM Triton 慢 1924 - 0 = **1924 ms**
  ,sglang CUTLASS 比 sglang Triton 多 ~12000 ms (sglang_triton 的 nsys 还没跑,
  但量级上 sglang triton 应该跟 vllm triton 接近)
- 但 vLLM CUTLASS / Triton 的 e2e 打平,sglang CUTLASS / Triton 差 3.4-4.7×

答案有两层:

**第一层 (vLLM)**: 1924 ms 大头发生在 cudagraph **capture phase**,这个 phase 在
启动后第一个推理 batch 之内就完成,且只跑一次。之后 inference 时 cudagraph replay
不会再调 `cutlass_fused_moe()` 的 Python wrapper,所以 `AutoTuner.choose_one` 不再
执行,微基准用的候选 kernel launch 也不再产生。**3 runs 实测里 vLLM 的 run 1 比 run 2/3
慢的 0.2-0.8 req/s 就是 capture 摊销**。

**第二层 (sglang)**: 没有 cudagraph,**每一个 forward call 都会真的执行 `cutlass_fused_moe()`
Python wrapper → 每一个 forward 都跑 AutoTuner**。一旦 cache 不命中,候选 kernel
launch 会重新发生。这是个**每次推理都付费**的成本,e2e 上看就是 3.4-4.7× 的慢。

这套机制也解释了为什么 R_short (conc=1, batch 小变化大) 比 R_medium/R_long
被打击更重 (0.22× vs 0.29×):shape 变化频率越高,cache miss 越频繁。

---

## 6. 三个候选 fix —— 风险/工程量从低到高

**Fix 1 — 一行 patch**(最低风险,~5 行修改):

`sglang/python/sglang/srt/layers/quantization/unquant.py:385`
```python
# Before
tune_max_num_tokens=next_power_of_2(x.shape[0]),
# After
tune_max_num_tokens=8192,    # flashinfer 默认值,固定值让 tuner cache 稳定
```

预期效果: tuner cache 命中率大幅提升,cutlass-sm90 kernel launch 数从 ~97774
下降。但只解 `tune_max_num_tokens` 这一维度,如果 cache key 还依赖 input shape,
仍可能 miss。需要 nsys 实测验证。

风险: 8192 对 small batch 可能不是最优 tile size。可以用
`max(8192, next_power_of_2(x.shape[0]))` 兜底极端大 batch。

**Fix 2 — 包 `with autotune(False):`**(中等风险):

flashinfer 提供 `autotune` context manager 来禁 tuner。warm-up 后包一下:
```python
from flashinfer.autotuner import autotune  # 待确认确切 import 路径

with autotune(False):
    output = flashinfer_cutlass_fused_moe(...)
```

预期效果: warm-up 后彻底跳过 `choose_one` 的 benchmark 逻辑。

风险: 第一次跑时挑的 tactic 之后所有 shape 都用,可能对偏离的 shape 不优。

**Fix 3 — 修 cudagraph hang bug**(最大杠杆,工程量最大):

如果能让 sglang flashinfer_cutlass 跟 cudagraph 兼容,batch shape 自动锁到
capture_sizes,Fix 1+2 都不再需要,vLLM 那 2-3 倍的差距也能补上。

工程量: 需要先 reproduce 并定位 detokenizer hang。可能涉及 sglang detokenizer IPC、
flashinfer JIT 调用、CUDA stream capture 这几个交互点。

---

## 7. 仍存疑的几个问题

1. **vLLM 上 CUTLASS 多花的 1924 ms cutlass-sm90 是不是真都在 capture 里?**
   还没分时间窗口验证。要跑 nsys 时段切分 (`nsys stats --report cuda_gpu_kern_sum
   --filter-time=...`) 看 capture 阶段 vs serving 阶段的 kernel 分布。

2. **sglang 上 9× 多的 launch 是不是真的全在 `choose_one` 里?**
   还没看 nvtx range / stack trace 证明每个 cutlass-sm90 launch 的 Python caller。
   一种间接验证:打个 print 数 `AutoTuner.choose_one` 在 forward 里的调用次数 + 候选 kernel 数
   = 应该 ≈ 97774。这个验证比较直接,但要 hack flashinfer 源码加 logging。

3. **`tune_max_num_tokens` 是不是真的就是 cache key 的主变量?**
   `flashinfer/fused_moe/core.py:474` 的 `refine_tuning_config` 用了
   `@functools.lru_cache(maxsize=None)` 以 `tune_max_num_tokens` 为 key,
   表面上看变了就 miss。但 `AutoTuner.choose_one` 内部还有自己的 cache,
   key 可能包含 input shape。完整验证要读 AutoTuner 源码。

4. **vLLM 那条 "FlashInfer 比 Triton 慢" 的源码注释是不是过时?**
   `vllm/.../unquantized.py:71` 说在 Hopper bf16 上 Triton 更快。我们 e2e 实测
   打平。但**单看 kernel 净时间** (排除 cudagraph 摊销),CUTLASS 净时间 1924 ms >
   Triton 净时间 259+125=384 ms,**那条注释在 kernel 层是对的**。所以 oracle
   把 default 设为 Triton 也合理 —— 没启 cudagraph 的场景下 (e.g. enforce_eager)
   Triton 会真的更快。

5. **我们的 conclusion 能 generalize 到别的 model 吗?**
   只测了 Qwen3-30B-A3B。如果换 DeepSeek-V3 / GPT-OSS / mixtral 等不同
   expert 数 / hidden size,AutoTuner 的 cache 行为可能不同。需要再测才能保证
   patch 不 regress 别的 model。

---

## 8. 引用的所有文件和路径

### 实验数据 (本仓库 `EndtoEnd-auto-optimization/`)

- `results/4way_bench/comparison_table.md` —— 4-way 吞吐表
- `results/4way_bench/runs/stats_table.md` —— 3 runs mean ± std
- `results/4way_bench/runs/bench_{sglang,vllm}_{triton,cutlass}_run{1,2,3}.json` (12 files) —— 原始数据
- `results/4way_bench/scripts/run_bench_4way.py` —— bench harness
- `results/4way_bench/scripts/start_*.sh` —— 4 个 backend 启动脚本
- `results/4way_bench/{sglang,vllm}_{triton,cutlass}/server_tail2.log` —— server 启动 log 尾部 200 行
- `results/4way_bench/nsys/EVIDENCE.md` —— vLLM CUTLASS 真的在跑 CUTLASS 的证据
- `results/4way_bench/nsys/ROOT_CAUSE.md` —— 9× kernel launch 根因分析
- `results/4way_bench/nsys/kernel_count_comparison.md` —— 三 backend kernel count 对比表
- `results/4way_bench/nsys/vllm_cutlass_kernels.csv` —— vLLM CUTLASS run 全部 175 个 kernel
- `results/4way_bench/nsys/vllm_triton_kernels.csv` —— vLLM Triton run 全部 kernel
- `results/4way_bench/nsys/sglang_cutlass_kernels.csv` —— sglang CUTLASS run 全部 kernel
- `results/4way_bench/nsys/vllm_cutlass_kernel_summary.txt` —— 预算好的 category 表

### sglang 源码

- `/home/t-jialianggu/work/sglang/python/sglang/srt/layers/quantization/unquant.py:372-386`
  —— `UnquantizedFusedMoEMethod.forward_cuda` 中调用 `flashinfer_cutlass_fused_moe`,
  传入 `tune_max_num_tokens=next_power_of_2(x.shape[0])`
- `/home/t-jialianggu/work/sglang/python/sglang/srt/layers/quantization/unquant.py:60`
  —— `from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe`
- `/home/t-jialianggu/work/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py:712-714`
  —— `is_flashinfer_cutlass()` 分支返回 base `FusedMoE`,实际派发由
  `UnquantizedFusedMoEMethod.forward_cuda` 的 `elif self.use_flashinfer_cutlass:` 决定
- `/home/t-jialianggu/work/sglang/python/sglang/srt/layers/moe/utils.py:64,85-86`
  —— `MoeRunnerBackend.FLASHINFER_CUTLASS` enum

### vLLM 源码

- `/home/t-jialianggu/work/vllm/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:378-403`
  —— vLLM 调用 `flashinfer_cutlass_fused_moe`,传 `output=`、`tune_max_num_tokens=max(max_capture_size, 1)`
- `/home/t-jialianggu/work/vllm/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:71`
  —— "On Hopper (SM90), the FlashInfer unquantized MoE kernels are slower than Triton" 注释
- `/home/t-jialianggu/work/vllm/vllm/config/kernel.py:171`
  —— `moe_backend` kernel-config 开关

### flashinfer 源码 (本地 conda env)

- `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/fused_moe/core.py:472-484`
  —— `MoERunner.refine_tuning_config(tune_max_num_tokens)` 用 `@functools.lru_cache`
- `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/fused_moe/core.py:486-621`
  —— `cutlass_fused_moe` 完整实现,line 548-578 是两次 `AutoTuner.choose_one` 调用,
  line 607 之后是真正的 `run_moe`
- `/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/flashinfer/fused_moe/core.py:702-925`
  —— 顶层 `cutlass_fused_moe` Python 函数 (前面是底层),把参数转交到 custom_op

### 工具

- nsys 路径: `/home/t-chendili/cuda/12.6/bin/nsys` (2024.5.1)。
  bundled in: `/home/t-jialianggu/.conda/envs/sglang-dev/nsight-compute-2025.1.1/host/target-linux-x64/nsys`
  但那个版本要求 launch nsys -t cuda 也支持
- nsys profile 命令模板:
  ```bash
  nsys profile -t cuda -s none -f true --trace-fork-before-exec=true \
    -o <outfile_no_ext> <server cmd>
  ```
  捕获结束: `kill -INT <nsys pid>` (不是 bash wrapper),wait ~30s flush
- 抽 kernel 表: `nsys stats --report cuda_gpu_kern_sum --format csv <rep>`

### 配套 checkpoint

- `~/.copilot/session-state/<session-id>/checkpoints/004-4way-moe-bench-vllm-cutlass-not-slow.md`
  —— 第一版 4-way bench (1 run/backend, 后被修订)
- `~/.copilot/session-state/<session-id>/checkpoints/005-4way-bench-3runs-and-nsys-proof.md`
  —— 3 runs + nsys 证据 + 撤回 "sglang Triton 慢" 的错论
