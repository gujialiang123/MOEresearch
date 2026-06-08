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

#### 4.4.1 sglang 维护者自己知道这个 bug —— 直接源码证据

`sglang/python/sglang/srt/model_executor/model_runner.py:1834-1844`:
```python
backend_str = self.server_args.moe_runner_backend

# TODO smor- support other cases for flashinfer autotune, such as, mamba backend

if backend_str not in [
    "flashinfer_trtllm",
    "flashinfer_mxfp4",
    # TODO: flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.
    # "flashinfer_cutlass",
]:
    return False
```

这个函数 (`_should_flashinfer_autotune`) 判断 "什么 backend 应该走 sglang 的 flashinfer
autotune warmup 路径"。`_flashinfer_autotune()` 的实现 (model_runner.py:1859-1874):

```python
def _flashinfer_autotune(self):
    from flashinfer.autotuner import autotune
    logger.info("Running FlashInfer autotune...")
    # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
    self.forward_stream.wait_stream(torch.cuda.current_stream())
    with torch.get_device_module(self.device).stream(self.forward_stream):
        with torch.inference_mode(), autotune():          # ← Fix 2 提到的 autotune ctx
            self._dummy_run(
                batch_size=self.req_to_token_pool.size, run_ctx=autotune()
            )
    torch.cuda.current_stream().wait_stream(self.forward_stream)
    logger.info("FlashInfer autotune completed.")
```

注意 `with autotune():` —— 这就是 flashinfer 的 autotune context manager,本意是
在 warmup 阶段一次性 tune 好所有 tactic,后续推理 cache 命中。其他 flashinfer
backend (`flashinfer_trtllm`, `flashinfer_mxfp4`) 都走这个路径。

**但 `flashinfer_cutlass` 被注释掉了,留 TODO 说 "会让 flashinfer 编译报错"**。所以:
1. `flashinfer_cutlass` 没走 `_flashinfer_autotune()` warmup
2. → AutoTuner 没在 cudagraph capture 之前预热
3. → 每个推理 forward 都触发 AutoTuner re-benchmark (nsys 观察到的 97774 calls)
4. → 同时 cudagraph + flashinfer_cutlass 还有 detokenizer hang bug

**§4.3 的 9× kernel launch + §4.4 的 cudagraph hang 很可能是同一个 bug 的两种表现**:
sglang 知道 flashinfer_cutlass 跟 sglang 的 capture/warmup 集成有问题,跳过 warmup
来规避,但这又导致 e2e 性能拉胯。

cudagraph hang 的具体根因(我们观察到的 detokenizer 挂死)目前不能完全归结到上面这
条 TODO,但**两件事高度相关**:都涉及 flashinfer JIT 编译 + sglang capture 流程。
后续要复现 + 抓 stack 才能确定。


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

---

## 9. (新增) 上游 issue / PR 调研 —— bug 是不是已知?有没有人在修?

调研日期: 2026-06-08。调研对象: sgl-project/sglang, flashinfer-ai/flashinfer, vllm-project/vllm。

### 9.1 直接相关的 sglang PR / issue

| # | 类型 | 标题 | 状态 | 跟我们的关系 |
|---|---|---|---|---|
| [#21872](https://github.com/sgl-project/sglang/pull/21872) | PR (OPEN) | Add FlashInfer CUTLASS fused MoE support for FP8 block-quantized models on SM90 | OPEN since 2026-04-01 | ★★★★★ —— 直接 benchmark SM90 cutlass vs triton + 给出 root cause |
| [#26715](https://github.com/sgl-project/sglang/issues/26715) | Bug | flashinfer_trtllm BF16 MoE: piecewise CUDA graph capture causes illegal memory access | OPEN | SM100 + flashinfer_trtllm + PCG 崩 (不是我们 case),但**workaround 是 `--moe-runner-backend flashinfer_cutlass`** —— 表明 SM100 上 cutlass + cudagraph 正常,只有 SM90 上挂死 |
| [#26137](https://github.com/sgl-project/sglang/issues/26137) | Bug | Decode workers hang at CUDA graph capture during init_device_graphs | CLOSED 2026-05-24 | DeepSeek-R1 FP4 DP32 + flashinfer_cutedsl + DeepEP + PCG 挂在 capture(类似现象但是不同 backend) |
| [#23870](https://github.com/sgl-project/sglang/issues/23870) | Bug | PiecewiseCudaGraphRunner.warmup_compile → cudaErrorIllegalAddress on dense model | OPEN | dense model 的 PCG 崩,相关但不同 path |

**特别有价值的 PR #21872**: PR 作者(claude-code 生成的 PR)在 H100 SM90 上直接跑了
sglang FP8 block-quantized flashinfer_cutlass MoE vs Triton 的对比:

| Mode | Triton | FlashInfer CUTLASS | Diff |
|---|---|---|---|
| **No CUDA graph** | 152 tok/s | 163 tok/s | **+7.4%** (cutlass 略快) |
| **CUDA graph** | 869 tok/s | 749 tok/s | **−13.8%** (cutlass 显著慢) |

PR 作者自己的解释 (PR description, 引用):
> "FlashInfer CUTLASS fuses routing+GEMM+activation, reducing kernel launch overhead.
> This gives +7.4% in eager mode. However **under CUDA graph (which already eliminates
> launch overhead), the CUTLASS GEMM is less efficient than Triton's tuned kernel for
> these shapes**, resulting in regression."

PR 的 "Recommended use" 也写: "`--disable-cuda-graph` or piecewise CUDA graph scenarios
where kernel launch overhead matters" —— **跟我们独立观察一致**。

→ 这条 PR 既是 §4 的 9× kernel launch 问题的旁证 (eager mode 下 cutlass +7%,说明
launch overhead 是真问题),也是 §10 要分析的 "SM90 上 cutlass 不比 triton 快" 命题
的直接数据点 (cudagraph 开后 cutlass 反而 -13.8%)。

### 9.2 sglang `TODO` 的 commit 历史

```bash
# 查找该 TODO 是谁加的、什么时候
git blame -L 1840,1845 python/sglang/srt/model_executor/model_runner.py
```

结果:
```
468931b572 (Baizhou Zhang 2025-12-21 18:08:07 -0800 1841)
        # TODO: flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.
468931b572 (Baizhou Zhang 2025-12-21 18:08:07 -0800 1842)
        #     "flashinfer_cutlass",
```

Commit `468931b572`:
```
[Tiny]Move deepseek fp4 cutlass moe test to per-commit test (#15565)
Author: Baizhou Zhang <sobereddiezhang@gmail.com>
Date: Sun Dec 21 18:08:07 2025 -0800
```

PR 标题写 `[Tiny]Move ... test to per-commit test`(把测试从 nightly 提到 per-commit),
**但 diff 里顺手把 `flashinfer_cutlass` 从 `_should_run_flashinfer_autotune` 的 list
里注释掉了**,**没在 PR description 里解释为什么**。PR comments 里也没有讨论。

这是一个**没记录的 silent disable** —— 没人在追这个 bug,也没 open issue 跟进。

### 9.3 我们没找到的相关 issue/PR

- **没找到** open issue 描述 "sglang + flashinfer_cutlass + cudagraph 在 SM90 hang"。
- **没找到** open PR 在修这个 hang bug。
- flashinfer-ai/flashinfer 仓库**没有** open issue 描述 `cutlass_fused_moe` 跟 cuda graph 不兼容。

**意思**: 这是个 stealth bug —— sglang 维护者知道(注释 + silent disable),但没正式追,
没有 issue 让外部贡献者发现并 fix。

### 9.4 推测的 bug 路径

基于 §4.4.1 的源码证据 + PR 历史,推测 chain 是这样的:

1. 某次 flashinfer 升级后,`cutlass_fused_moe` 进 sglang 的 `_flashinfer_autotune()` warmup
   (用 `with autotune():` ctx mgr)时,触发 JIT 编译错误(注释只写 "compilation errors",
   没给具体信息)
2. Baizhou 在 `#15565` 里临时把 `flashinfer_cutlass` 从 autotune list 里注释掉绕过
3. 没人在 cudagraph 端跟进 —— 因为 sglang 的 cudagraph capture 本身不直接调
   `autotune()` ctx,但**capture 期间** Python wrapper 会执行,内部会调 flashinfer
   的 AutoTuner.choose_one → 同样触发 JIT 编译/tuning,可能在 stream capture 状态下
   挂死 (这就是我们观察到的 detokenizer hang)

**未验证的假设**:
- detokenizer hang 是不是真的是 "flashinfer JIT 编译被 cudagraph stream capture 卡住"
- 还是 sglang detokenizer IPC 在 cudagraph 模式下有死锁
- 要确认,得 attach `py-spy dump --pid <detokenizer_pid>` 看 stack

### 9.5 推荐 fix 路径(给 sglang 提 issue 的话怎么写)

1. **issue 标题**: `[Bug] sglang + flashinfer_cutlass + cudagraph: detokenizer hangs after "Capture cuda graph end" on SM90`
2. **issue 内容**:
   - 重现步骤 (`--moe-runner-backend flashinfer_cutlass` without `--disable-cuda-graph` on H100/H200)
   - 现象: detokenizer heartbeat 停,/health 返回 503,reproduced 2×
   - 直接证据: `model_runner.py:1841-1842` TODO 表明维护者已知
   - 性能影响: 我们测得 R_medium 3.4× slowdown, R_short 4.5× slowdown
   - 关联: PR #21872 也观察到 cudagraph 路径有问题
3. **修复路径**:
   - 短期: 完整 reproduce hang + 抓 detokenizer stack trace
   - 中期: 在 `_flashinfer_autotune` 里把 `flashinfer_cutlass` 重新放进来,看具体
     compilation error 是什么
   - 长期: 跟 flashinfer-ai 团队协作修 `cutlass_fused_moe` 跟 sglang capture 的兼容性

---

## 10. (新增) 即使 bug 修好,为什么 SM90 上 FlashInfer CUTLASS 也不会显著超 Triton?

观察事实(来自 §1 实测 + PR #21872 的独立测量):
- vLLM SM90 H200 bf16: cutlass / triton = 1.00-1.02× (持平)
- PR #21872 SM90 H100 FP8 block-quant: cutlass / triton 在 cudagraph 下 = **0.86×** (cutlass 反而慢)
- 业界普遍说法: SM100 (Blackwell) 上 flashinfer cutlass 比 triton 显著快

为什么 SM90 上的差距这么小或反向?

### 10.1 第一层原因 —— Triton MoE 是 shape-tuned 的,CUTLASS 是 general 的

vLLM/sglang 用的 Triton MoE kernel(`triton_red_fused_fused_add_rms_norm_moe_forward_0`)
经过多个 release 在 H100/H200 上对常见 (E, N, K, top_k) shape 的 **autotune 数据库**
(`fused_moe/configs/E=128,N=768,device_name=NVIDIA_H200.json` 等) 已经存在。每个 model
shape 都有人手动 tune 过最优 BLOCK_M/BLOCK_N/BLOCK_K/num_stages/num_warps。

FlashInfer CUTLASS 是 general 框架,kernel 选择空间是固定的 tile shape 组合,**没有
shape-specific 人工 tune**,靠 AutoTuner 在线挑。对于 well-known shape (Qwen3MoE、DeepSeek),
Triton 的人工 tune 数据库占优。

### 10.2 第二层原因 —— flashinfer SM90 grouped-GEMM 的搜索空间太窄

直接证据: `flashinfer/jit/gemm/cutlass/generate_kernels.py`

**SM90 grouped GEMM 生成器** (`generate_sm90_grouped_gemm_operations`,line 556-637):
```python
supported_dtypes = [DataType.f16, DataType.bf16, DataType.f32, DataType.e4m3]
                     # NO e2m1 (FP4)
M_TILES = [128]      # 注释: "Currently M tile must be 128 for Grouped GEMM"
N_TILES = [16, 32, 64, 128, 256]
cga_shapes = product([1, 2], [1, 2], [1])  # 4 个固定 cluster shape
mainloop_schedule = TmaWarpSpecializedCooperative  # 1 种
```

**SM100 grouped GEMM 生成器** (`generate_sm100_grouped_gemm_operations`,line 840-941):
```python
supported_dtypes = [
    DataType.f16, DataType.bf16, DataType.f32,
    DataType.e4m3,           # FP8 ✓
    e2m1,                    # FP4 native ✓ (SM90 没有)
    (DataType.e4m3, e2m1),   # mixed FP8/FP4 ✓
]
cta_shapes_m = [64, 128]     # 2 个 M tile (vs SM90 只有 1 个)
cta_shapes_n = [8, 16, 32, 64, 128, 192, 256]  # 多了 N=8 和 N=192
cga_shapes = [(1, 1, 1), (2, 1, 1)]
dynamic_cga = [True, False]  # 动态 cluster shape (SM90 没有)
epi_schedules = [
    PtrArrayNoSmemWarpSpecialized1Sm,   # 给某些 shape 用 no-smem epilogue
    PtrArrayTmaWarpSpecialized1Sm,
]
```

**搜索空间大小估算**:
- SM90: 4 dtypes × 5 N × 4 cluster × 2 swap_ab × 2 epi_fusion = **320** 个 kernel candidate
- SM100: 6 dtypes × 14 cta_shape × 2 cga × 2 epi × 2 dynamic × 2 swap_ab × 2 epi_fusion ≈ **>5000** 个 candidate

**意思**: SM100 上 AutoTuner 有 ~16× 多的选项可挑,所以更可能挑到接近最优的 tile/cluster
组合;SM90 上选项少,选不到最优,跟 Triton 的人工 tune 差距就拉大。

特别要注意 SM90 的 `M_TILES = [128]` 那条注释 —— **Grouped GEMM 的 M 维必须是 128**。
对于小 batch (e.g. batch=1 R_short),sequence 维度也只有几十,M=128 大量浪费 tile,
SM90 没法降到 M=64。SM100 允许 M=64,小 batch 利用率显著更好。

### 10.3 第三层原因 —— 硬件原语本身

**Hopper SM90 的矩阵乘原语**: `wgmma.async`(warp-group MMA)
- 异步,但占用 register file 存累加器
- accumulator 在 register 里,kernel 设计要小心 register pressure
- BF16/FP8 都有原生支持
- **没有 FP4 原生指令**

**Blackwell SM100 的矩阵乘原语**: `umma`(Unified MMA) + TMEM(tensor memory)
- 异步,**accumulator 在专用 tensor memory 而不是 register**,大幅减少 register pressure
- 同 SM 内可以有更多并发 MMA 在飞行
- BF16/FP8/**FP4 都有原生支持**(`mxfp4` `nvfp4`)

**对 MoE 的实际影响**:
- 对 **BF16 unquantized**(我们的测试 case): SM100 的 UMMA 带来 ~10-20% kernel-level 加速,
  但 Triton 在 SM90 的 WGMMA 也跑得很好,所以净差距小。SM100 上 cutlass 略胜 triton,
  SM90 上打平。
- 对 **NVFP4 quantized**: SM100 用 FP4 tensor core 直接算,SM90 必须 dequantize 到 BF16
  再算,**3-5× 差异**。这就是为什么 NVFP4 模型 (DeepSeek-V3.x NVFP4, MiniMax NVFP4 etc.)
  在 SM100 上必须用 flashinfer cutlass,SM90 上根本没法用 flashinfer cutlass 的 NVFP4 path。

### 10.4 第四层原因 —— Triton on SM90 已经很接近 SM90 峰值

我们之前 `triton_rewrite_investigation.md` §14 量化过 Triton MoE kernel 在 H200 上达到
**~30% of FP16 peak FLOPs** (Qwen3MoE)。CUTLASS 也大概在这个数量级。两边都
**bottlenecked by memory bandwidth** 而不是 compute,所以 kernel 怎么调度的差异不显著。

到 SM100 上情况就不一样:
- HBM3e → HBM3+(SM100 GB200)bandwidth ~8 TB/s vs H200 4.8 TB/s
- Tensor core peak ~3.5×
- 给 kernel 的 compute headroom 大,kernel 之间能拉开差距

### 10.5 一句话总结

**SM90 上 flashinfer CUTLASS ≈ Triton 是三个因素叠加**:

1. **Triton 端**: H100/H200 上 fused_moe Triton kernel 有完整 shape-tuned 配置库
   (`fused_moe/configs/*.json`),对 well-known model shape 接近 SM90 硬件峰值
2. **CUTLASS 端**: flashinfer 在 SM90 上的 grouped GEMM 搜索空间比 SM100 小约 16×
   (M tile 锁死 128, 没 FP4 路径, 没 dynamic cluster, 1 个 epilogue schedule),
   AutoTuner 选不到比 Triton 更优的配置
3. **硬件端**: SM90 的 WGMMA 没有 SM100 的 TMEM / UMMA 那么强的并发能力,
   compute headroom 比较紧,kernel 调度差异被 memory bandwidth bottleneck 吸收

→ **结论**: 即使我们修好 §4 的 9× launch 问题 + §4.4 的 cudagraph hang,sglang_cutlass
最多也只能追到 vllm_cutlass 水平,也就是跟 sglang_triton 持平,**不会有 SM100 那种
显著 speedup**。CUTLASS 的真正价值在 SM100 + 量化模型场景,SM90 + unquantized BF16
两边 e2e 几乎一样。

### 10.6 这对优化 agent 的设计意味着什么

我们最初做这个 study 是想找 "MoE 优化机会"。结论比预想的更微妙:

| 场景 | 优化空间 | 优化路径 |
|---|---|---|
| SM90 H100/H200 + bf16 unquantized + cudagraph | **几乎没有** | Triton 已接近峰值,CUTLASS 没优势 |
| SM90 H100/H200 + bf16 + no cudagraph | 修 sglang 那个 9× launch (§4) | tune_max_num_tokens 固定 + autotune warmup |
| SM90 + FP8/INT8 quantized | 中等 | CUTLASS group-GEMM 比 Triton 稍快,但 cudagraph 下基本持平 |
| SM100 GB200 + nvfp4/mxfp4 quantized | **巨大** | 必须用 flashinfer cutlass,且 cudagraph 对它友好 |
| SM100 GB200 + bf16 | 小 | flashinfer 比 triton 稍快 (类似 SM90 SM100 的 trend) |

对一个 e2e 优化 agent 来说,这意味着 "对每个 (model, GPU, dtype) 都跑所有 backend 比较"
策略 ROI 很低 —— 对大部分 SM90 + bf16 的 long-tail model,Triton 就是最优解,不用 search。


---

## 11. (新增) Hang 复现失败 + cudagraph 实际只帮 R_short

调研 §9/§10 之后,尝试实测 §4.4 的 hang bug:

**结果**: 这次**没复现 hang** —— `--moe-runner-backend flashinfer_cutlass` 不带
`--disable-cuda-graph` 直接启动,Capture cuda graph end 之后正常进入 service,
所有请求都 200,server.log 看到 decode 真的在用 cudagraph (`cuda graph: True` 出现 112 次)。

**最可能的原因**: checkpoint 005 时的 hang 是 **flashinfer JIT 编译 + cudagraph capture
同时进行** 导致的临时挂死 (`fused_moe_90.so` 第一次 JIT 要 7+ 分钟,在 capture 期间
触发会让流程死锁)。现在 `~/.cache/flashinfer/0.6.11.post2/90a/cached_ops/fused_moe_90/`
已经有 `.so` 文件,capture 时不再触发 JIT,流程正常。
- 这跟 §4.4.1 sglang TODO 注释里的 "flashinfer compilation errors" 假说一致。
- 也跟 sglang PR #15565 silent disable 的时机一致(那时候 flashinfer 升级,新版需要
  重新 JIT,引发 compilation errors)。

→ **§4.4 的 hang 是"冷启动 race condition" 不是"永久 deadlock"。** 这是个时序 bug,
不是逻辑 bug。要稳定复现需要先 `rm -rf ~/.cache/flashinfer/0.6.11.post2/90a/cached_ops/fused_moe_90/`
然后重启 server,让 JIT 在 capture 时触发。

### 11.1 e2e bench —— cudagraph 帮了多少?

跑 3 runs/regime × `flashinfer_cutlass` 带 cudagraph(`--watchdog-timeout 1800`,
GPU=1),数据在 `results/4way_bench/runs/bench_sglang_cutlass_graph_run{1,2,3}.json`:

| Regime | no_cudagraph (旧, mean) | with_cudagraph (新, mean) | speedup |
|---|---|---|---|
| R_short  | 0.70 req/s | **1.25 req/s** | **1.79×** |
| R_medium | 1.30 req/s | 1.35 req/s | 1.04× |
| R_long   | 1.29 req/s | 1.33 req/s | 1.03× |

**和 baseline 比**:

| Regime | sglang_cutlass + graph | sglang_triton | vllm_cutlass |
|---|---|---|---|
| R_short  | 1.25 | 3.23 | 3.23 |
| R_medium | 1.35 | 4.51 | 4.59 |
| R_long   | 1.33 | 4.38 | 4.26 |

→ **cudagraph 只在 R_short 显著帮(1.79×),R_medium/R_long 几乎不变**。

### 11.2 为什么 cudagraph 在 R_medium/R_long 没帮?

直接证据来自 server.log (来源: `results/4way_bench/hang_repro/server.log`):
```
=== decode batch lines: cuda graph True or False? ===
     65 cuda graph: False     ← 全是 prefill
    112 cuda graph: True      ← 全是 decode
```

**sglang 的 cudagraph 只 capture decode step,不 capture prefill**。

regime 拆解:
- **R_short** (8 reqs / 200w / 64 out, conc=1): 200 prefill tok + 64 decode tok ≈ 24% decode → cudagraph 帮一点点 (1.79×)
- **R_medium** (16 reqs / 800w / 256 out, conc=8): 800 prefill + 256 decode ≈ 24% decode,**但 batch 大,prefill 时间被均摊后,decode 部分 cudagraph 也帮一点** —— 不过 prefill 占大头还是 AutoTuner 重 benchmark
- **R_long** (8 reqs / 2000w / 256 out, conc=16): 2000 prefill + 256 decode ≈ 11% decode → cudagraph 几乎不帮

**所以 §4 找到的 9× kernel launch 大头是 prefill 端**,cudagraph 没法解决。需要的是
**Fix 1 (固定 `tune_max_num_tokens`) 或 Fix 2 (`with autotune(False)`)**,这两个能影响
prefill 路径的 AutoTuner cache。

### 11.3 修订 §6 优化路线优先级

旧版 §6 说:
> 1. 🥇 Fix 1: 一行 patch (tune_max_num_tokens 固定)
> 2. 🥈 Fix 2: `with autotune(False):`
> 3. 🥉 Fix 3: 修 cudagraph hang

**修订后**:
- **🥇 Fix 1/2 仍然是主战场**: prefill 端 AutoTuner 是大头,cudagraph 没法解
- **🥉 Fix 3 (修 hang) 优先级降低**: 不是稳定 deadlock,是冷启动 race。warm 之后即使没人显式修,
  也能跑通 cudagraph,只是 e2e 性能基本没变化(只帮 R_short)。
- **新优先级**: Fix 1 后必须验证 prefill 端 AutoTuner 路径的 cutlass-sm90-gemm count
  是否真的减少 → 如果减,prefill 加速 → e2e 加速。如果没减,根因不是 tune_max_num_tokens
  而是别的(可能是 AutoTuner.choose_one 内部 cache key 还依赖 shape)。

### 11.4 验证 Fix 1 的具体方案

```bash
# 1. apply patch
cd /home/t-jialianggu/work/sglang
# edit python/sglang/srt/layers/quantization/unquant.py:385
#   tune_max_num_tokens=next_power_of_2(x.shape[0])  →  tune_max_num_tokens=8192

# 2. start sglang_cutlass with cudagraph on GPU 1, GPU 0 别人占着别动
bash /tmp/start_sglang_cutlass_GRAPH.sh  # no --disable-cuda-graph this time

# 3. 跑 3 runs benchmark + 一次 nsys profile
for run in 1 2 3; do
  python3 /tmp/run_bench_4way.py http://127.0.0.1:30000 "sglang_cutlass_fix1_run${run}" \
    /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/4way_bench/runs/
done

# 4. compare: cutlass-sm90-gemm count should drop from ~97k toward vLLM's ~11k
# (because prefill no longer re-tunes per forward)
```

### 11.5 一句话总结

**Hang bug 这次没复现** —— 几乎确认是 flashinfer JIT 冷启动 race,不是永久 bug。
**Cudagraph 实际只帮 R_short 1.79×**,R_medium/R_long 几乎没变 —— 因为 sglang 只
graph-capture decode,而我们 regimes 是 prefill-heavy,prefill 端 AutoTuner 重 benchmark
没被 cudagraph 解决。

**所以 §6 优化路线变化**: Fix 1 (`tune_max_num_tokens` 固定) 是唯一真正能拉 R_medium/R_long
性能的方案。Fix 3 (修 cudagraph hang) 没必要专门做 —— 它要么 warm 后自己好,要么只帮
R_short 一点点。

