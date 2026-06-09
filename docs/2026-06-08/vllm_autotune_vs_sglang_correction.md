# vLLM 的 flashinfer autotune 机制 — 跟 sglang 的关键差异

> **2026-06-08 新发现**: vLLM 启动时显式跑 `with autotune(): _dummy_run()`,
> sglang 不跑。这解释了为什么 vLLM 在 SM90 上 cutlass 跟 triton 打平,而 sglang
> cutlass 慢 3.4-4.7×。

## 实证 — vLLM server log 直接证据

`results/4way_bench/vllm_cutlass/server_tail2.log`:
```
(EngineCore pid=...) [Autotuner]: Autotuning process starts ...
(EngineCore pid=...) [Autotuner]: Autotuning process ends
```
~4 秒。

`results/4way_bench/sglang_cutlass/server.log`: **完全没有这两行**。

## vLLM 源码 — `vllm/model_executor/warmup/kernel_warmup.py:55-188`

```python
def kernel_warmup(worker: "Worker"):
    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)    # ← SM90 也会跑

def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.
    """
    import vllm.utils.flashinfer as fi_utils

    if not _FLASHINFER_USE_PERSISTENT_CACHE:    # ← currently True (持久化关掉了)
        with torch.inference_mode(), fi_utils.autotune():    # ★ 进入 tuning mode
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        get_world_group().barrier()
        return

    # 下面是 persistent cache 路径 (目前 disable):
    # 第一次跑: 'with fi_utils.autotune(tune_mode=True, cache=cache_path)'
    #   把结果存进 ~/.cache/vllm/flashinfer_autotune_cache/autotune_configs.json
    # 第二次跑: AutoTuner.get().load_configs(str(cache_path))  ← 直接 load 不 sweep
```

**关键点**:
1. vLLM 在 SM90 上**也跑** autotune (line 73 `has_device_capability(90)` 包括 SM90)
2. 用的是 `with fi_utils.autotune():` ctx mgr,**这就是模式 2 (startup warmup)**
3. 当前 `_FLASHINFER_USE_PERSISTENT_CACHE=False` (line 112),所以每次启动都重 sweep
4. 注释里写: "Without autotuning, FlashInfer will rely on heuristics, which **may be significantly slower**"

## sglang 对比

`sglang/python/sglang/srt/model_executor/model_runner.py:1834-1844`:
```python
def _should_run_flashinfer_autotune(...):
    ...
    if backend_str not in [
        "flashinfer_trtllm",
        "flashinfer_mxfp4",
        # TODO: flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.
        # "flashinfer_cutlass",    # ← 注释掉了!
    ]:
        return False
```

sglang 给 `flashinfer_trtllm` / `flashinfer_mxfp4` 走 autotune warmup,**但 `flashinfer_cutlass` 被排除**。

`sglang/.../model_runner.py:1859-1874` 的 `_flashinfer_autotune()` 跟 vLLM 的 `flashinfer_autotune()` 几乎是 1:1 对应实现:
```python
def _flashinfer_autotune(self):
    from flashinfer.autotuner import autotune
    ...
    with torch.get_device_module(self.device).stream(self.forward_stream):
        with torch.inference_mode(), autotune():    # ★ 同样用 ctx mgr
            self._dummy_run(batch_size=..., run_ctx=autotune())
```

**所以两边都有 warmup 框架,只是 sglang 给 cutlass 关掉了**。

## 这是为什么 sglang cutlass 慢的真正根因

之前 ROOT_CAUSE 我说 "AutoTuner re-benchmark 每 forward 触发" — 错的。

**真正机制**:
- vLLM 启动时显式 `with autotune(): _dummy_run()` → AutoTuner.is_tuning_mode=True → sweep 18 个候选 → **best tactic 存进进程内 cache**
- 推理时 `cutlass_fused_moe()` 调 `AutoTuner.choose_one()` → cache hit → 返回 tuned tactic → run_moe 用最优 kernel
- sglang 启动时**不跑** warmup → cache 永远空
- 推理时同样调 `choose_one()` → cache miss → 返回 fallback tactic (-1) → run_moe 用**默认 (heuristic) kernel**,可能比最优慢

但 §F1 (Fix 1 验证) 也证明: **改 `tune_max_num_tokens` 不能让 sglang 进入 tuning mode**,因为 sglang 根本没显式包 `with autotune():` ctx。

## Fix 3' (给 SM90 写 hand-tuned `.py` 表) 重新评估

**自动加载流程** (`flashinfer/autotuner.py:316-332 load_from_file`):
```python
@lru_cache(maxsize=None)
def load_from_file(key):
    module_name = get_config_path(is_module=True)
    # e.g. "flashinfer.tuning_configs.v0_1_trtllm_fused_moe_NVIDIA_H200"
    try:
        module = importlib.import_module(module_name)
        best_configs = module.best_configs
    except (ImportError, AttributeError):
        best_configs = None
    if best_configs is not None:
        k = str((key[0], key[1], key[3]))
        if k in best_configs:
            return True, best_configs[k][0], best_configs[k][1], None
    return False, 0, -1, None
```

`load_from_file()` 在哪被调? 看 `AutoTuner._search_cache` 链路: **只有 `is_tuning_mode=True` 时 cache miss 才查 file**。

**所以 Fix 3' 的受益情况**:

| 场景 | 是否走 `is_tuning_mode=True` | Fix 3' 受益? |
|---|---|---|
| vLLM SM90 cutlass | ✓ (启动时 `flashinfer_autotune()`) | ✓ **能省 first launch sweep 时间 (~4s)**,后续推理同样快 |
| vLLM SM100 cutlass | ✓ + load `v0_1_..._B200.py` 已经存在 | 已经在用 |
| sglang SM90 cutlass | ✗ (TODO 注释掉了) | **不受益,除非先修 Bug A** |
| sglang SM100 trtllm_gen | ✓ (flashinfer_trtllm path 没被注释) | 走 trtllm_gen 不是 cutlass,无关 |

## 修正结论

**之前以为**: Fix 3 (给 SM90 写表) 是 sglang 也受益的"上游 fix"。

**修正**:
- vLLM 上: 已经在 startup 跑 autotune (模式 2),Fix 3 只省 sweep 时间 (~4 秒),不影响 e2e
- sglang 上: **必须先修 Bug A 让 cutlass 也走 autotune warmup 路径,Fix 3 才能起效**

## 另一个 caveat — vLLM 实测 cutlass = triton 持平,说明 tuned tactic 跟 fallback 差异小

我们 §1 实测 `vllm_cutlass` 跟 `vllm_triton` 在 R_medium 上几乎打平。vLLM 已经在跑 autotune 了,理论上拿到了"最优 tactic",但结果仍跟 triton 打平。

**意味着**: 在 SM90 + bf16 unquantized + Qwen3MoE 这个 case,**最优 tactic 跟 default heuristic fallback 性能差异很小**。SM90 上 candidate kernel 候选小 (18 个) + M tile 锁死 128 限制了 tuning 上限。

所以即使我们给 sglang 修了 Bug A + 给 flashinfer ship SM90 表,**sglang cutlass on R_medium 也未必能跟 vllm_cutlass 一样**(因为差距主要在 **cudagraph 覆盖度**,不在 tactic)。

## 综合: 真正能改善 sglang cutlass on SM90 的 fix path

**短期 (容易)**:
- 不改

**中期 (中等工程)**:
- 修 sglang Bug A (让 cutlass 走 `_flashinfer_autotune()` warmup),期待小提升
- 这跟 vLLM 1:1 对齐 startup tuning

**长期 (大工程)**:
- 让 sglang 把 cudagraph 覆盖扩展到 prefill (vLLM piecewise CUDA graph 思路)
- 这是真正消除 sglang vs vllm 3-4× 差距的方法

**不该做**:
- ~~Fix 1 (改 `tune_max_num_tokens`)~~ — 已经验证无效
- ~~只做 Fix 3 不修 Bug A~~ — sglang 仍不会用到表

## 留下的 open question

1. 如果在 vLLM 上把 `_FLASHINFER_USE_PERSISTENT_CACHE` 改成 True,vLLM 第二次启动会快多少?
2. vLLM 现在 dummy_run 是用 `max_num_batched_tokens` 这一个 size 触发 sweep — 我们 bench batch=8 这种小 size 真的命中了 cache 吗?如果没有,实际 vLLM 推理时用的也是 fallback tactic,意味着我们看到的 e2e 性能不是 hand-tuned 的功劳
3. 如果给 sglang 修 Bug A + 加 startup warmup,实测 sglang_cutlass 能涨多少?
