# Triton MoE Kernel 分析：为什么 SM 利用率低 + 优化空间

**目的**：研究 sglang 现在用的 triton `fused_moe_kernel` 实现，结合我们 NCU 实测（decode fused_moe SM 16%、DRAM 75%、No-Eligible 78%），找出提升 SM 利用率的具体优化空间。
**代码**：`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`（kernel）、`fused_moe_triton_config.py`（配置）、`moe_align_block_size.py`（token 分组）

---

## 1. 这个 kernel 长什么样（结构）

`fused_moe_kernel`（`fused_moe_triton_kernels.py:324`）是一个**分组 GEMM**（grouped GEMM）：
- 把每个 token 路由到的 topk 个专家，按专家分组；
- 每个 program（CUDA block）负责算 `[BLOCK_SIZE_M × BLOCK_SIZE_N]` 的一块输出，沿 K 维循环累加：
  ```python
  for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
      a = tl.load(a_ptrs, ...)     # 读激活 [BLOCK_M, BLOCK_K]
      b = tl.load(b_ptrs, ...)     # 读专家权重 [BLOCK_K, BLOCK_N]
      accumulator = tl.dot(a, b, acc=accumulator)   # Tensor Core MMA
      a_ptrs += BLOCK_SIZE_K * stride_ak
      b_ptrs += BLOCK_SIZE_K * stride_bk
  ```
- 这是标准的 Triton tiled GEMM，性能**完全由 tile 尺寸（BLOCK_M/N/K）、num_warps、num_stages 决定**。

---

## 2. ★ 根因：decode 时走了"小 tile"分支

配置在 `get_default_config`（`fused_moe_triton_config.py:138`）。**关键逻辑（bf16 路径）**：
```python
config = {BLOCK_SIZE_M:64, BLOCK_SIZE_N:64, BLOCK_SIZE_K:32, GROUP_SIZE_M:8}
if M <= E:                    # ← decode 时 M 很小，命中这个分支！
    config = {BLOCK_SIZE_M:16, BLOCK_SIZE_N:32, BLOCK_SIZE_K:64, GROUP_SIZE_M:1}
```
其中 `M` = 每个专家平均分到的 token 数，`E` = 专家总数。

**实测我们两个模型 decode 时的 M（每专家 token 数）**：
| 模型 | E | topk | batch=1 | batch=32 | batch=128 |
|---|---|---|---|---|---|
| LFM2.5 | 32 | 4 | M≈0 | M≈4 | M≈16 |
| Qwen3-30B | 128 | 8 | M≈0 | M≈2 | M≈8 |

**→ decode 时 M 永远 ≪ E，100% 命中小 tile 分支 `BLOCK_SIZE_M=16`。**

**这个小 tile 就是 SM 利用率低的直接原因**：
1. **BLOCK_SIZE_M=16 但实际每专家只有 2-4 个 token** → tile 里 16 行只有 2-4 行是真数据，**其余 75-87% 是 padding**（`moe_align_block_size` 把每专家 token 数 pad 到 BLOCK_SIZE_M 的倍数）。**Tensor Core 在算大量 padding 的零。**
2. **BLOCK_SIZE_M=16 太小** → 每个 tile 的算术强度低，MMA 指令少，warp 很快就卡在等下一块权重（memory-bound），occupancy 上不去 → 这正是 NCU 测到的 **No-Eligible 78%**。
3. **没指定 num_warps/num_stages** → 用 Triton 默认（通常 num_warps=4, num_stages=2-3），**流水线深度不足**，无法用足够的 in-flight 内存请求掩盖 HBM 延迟。

---

## 3. NCU 数据与 kernel 结构的对应（证据闭环）

| NCU 实测（decode fused_moe） | kernel 层面的原因 |
|---|---|
| SM 16% / DRAM 75% → memory-bound | 小 tile 算术强度低，权重读取主导 |
| Occupancy 12-37% | BLOCK_M=16 + 浅流水线 → 驻留 warp 少 |
| No-Eligible 78%（SM 空转） | 每个 warp 发几条 MMA 就卡等权重，num_stages 浅无法掩盖 |
| TC%（Tensor Core）低 | tile 里大量 padding 零参与 MMA，有效算力被浪费 |

**核心矛盾**：MoE decode 是"**很多专家、每个专家只有几个 token**"的**极瘦 GEMM**（M=2-4, N/K=数千）。Triton 的 tiled GEMM 是为"方阵/大 M"设计的，在这种极瘦形状下：tile 填不满、流水线掩不住延迟、Tensor Core 算 padding。

---

## 4. 优化空间（按可行性 / 收益排序）

### 机会 A（最直接）：调 tile 配置 / 补 tuned config
- sglang 支持从 JSON 加载 per-(E,N,device) 的 tuned config（`E=32,N=1792,device=H200.json`）。我们之前日志里看到 **"Using default MoE kernel config. Performance might be sub-optimal! Config file not found"** —— **说明现在跑的就是上面那个 default 小 tile，根本没 tuning 过！**
- **行动**：用 sglang 的 `benchmark/kernels/fused_moe_triton` 对我们的 (E, N, decode M 范围) 做一次 tile 搜索，生成 tuned JSON。可能仅调 num_warps/num_stages/GROUP_SIZE_M 就能提 occupancy。
- **预期**：中等收益，零 kernel 改写，当天可做。**这是最该先做的。**

### 机会 B（中等）：针对 MoE-decode 的专用 kernel 形状
- 小 M 的极瘦 GEMM，标准 tiled GEMM 不是最优。可以：
  - 用 **persistent kernel + 更深 num_stages**（更多 in-flight 权重读取掩盖 HBM 延迟）；
  - 或 **swap A/B**（kernel 里已有 `swap_ab` 分支，line 518）让瘦维度在 N 上，减少 padding。
- **预期**：较高收益，需 kernel 调整 + 验证正确性。

### 机会 C（最激进，最高收益）：换非 GEMM 的 MoE 范式
- decode 的 MoE 本质是"每个 token 选 topk 专家做 GEMV(向量×矩阵)"，不是 GEMM。可以用 **专门的 grouped-GEMV / SpMM kernel**，或 **fused 的 gather-GEMV**，避免把 GEMV 硬塞进 GEMM tile 导致的 padding。
- 这也是为什么工业界有专门的 MoE decode kernel（如 vLLM 的 Marlin-MoE、cutlass grouped GEMM、DeepGEMM）。
- **预期**：最高收益（直接消除 padding 浪费），但工作量大。

### 机会 D（正交，已知）：spec decoding 增大有效 M
- spec decoding 一次验证多 token → 增大每专家的 M → 小 tile 的 padding 比例下降。这解释了为什么 A1b spec 在 batch 大时收益更明显。但这是"喂更多活给同一个次优 kernel"，不是修 kernel 本身。

---

## 5. 结论

1. **SM 利用率低有明确的、可定位的 kernel 层根因**：decode 时 M≪E → 走 `BLOCK_SIZE_M=16` 小 tile → 大量 padding + 浅流水线 → occupancy 低、SM 空转 78%。**不是玄学，是具体代码路径。**
2. **最容易的优化（机会 A）现在就漏掉了**：日志显示我们跑的是 default config，**从没 tuning 过 MoE tile**。补一个 tuned JSON 是零风险、当天可做的第一步。
3. **真正的大头（机会 C）**：MoE decode 是极瘦 GEMV，用 GEMM tile 天然浪费；专用 kernel 是接近硬件上限的正解，但工程量大。
4. 这条链把"SM 空转 78%"从一个**现象**变成了一个**可操作的 kernel 工程问题**——正是之前一路排除（scheduler、spec、backend 都碰不到）后剩下的那块硬骨头。

---

## 附：关键代码索引
| 内容 | 文件:行 |
|---|---|
| fused_moe_kernel 主体 | `fused_moe_triton_kernels.py:324` |
| GEMM 主循环（tl.dot） | `:513-308` |
| swap_ab 分支 | `:518` |
| **小 tile 分支（M<=E）** | `fused_moe_triton_config.py:193-199` |
| tuned config 加载 | `fused_moe_triton_config.py:203 try_get_optimal_moe_config` |
| token padding 到 BLOCK_M | `moe_align_block_size.py:38-55` |
