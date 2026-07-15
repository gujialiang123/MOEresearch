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

---

## 6. 一步 MoE 的执行流程模拟（decode，单卡）

以 **Qwen3-30B-A3B、单卡 H200、decode 一步、batch=32** 为例。
前提：58GB 专家权重**常驻显存**（H200 143GB）。"搬运"= **HBM(显存) → SM(计算核心)** 的数据流，**不是跨卡搬移**。

**流程**：
1. **32 个 token 到达本层 MoE**（各 2048 维）。
2. **路由（gating）**：每 token 选 8 个专家 → 32×8 = **256 个 (token,专家) 配对**，摊到 128 个专家上，**平均每专家 2 个 token**。
3. **按专家分组**（moe_align_block_size）：去同一专家的 token 排一起；每专家 token 数**pad 到 BLOCK_SIZE_M=16 的倍数**（2 个真 token → 补 16 行，14 行是零）。
4. **★核心计算（"搬运 experts"）**：逐个专家——
   - 从 HBM 把【该专家权重 9.4MB】搬到 SM；
   - 用这 9.4MB 只算 2 个真 token（其余 14 行乘 padding 零）；
   - 用完丢弃，搬下一个专家的 9.4MB……重复至覆盖所有激活专家（一层 ~1.21GB）。
5. **加权合并**：每 token 从 8 个专家的输出按 gating 权重求和。
6. **下一层**：重复 48 层。

**一句话**：为了给 32 个 token 各算一步，GPU 要把 **58GB 专家权重整个从 HBM 读一遍**，每个权重只被 ~2 个 token 用一下就扔。

---

## 7. ★ 搬运 vs 计算的时间占比（回答"多少时间在通信 vs 计算"）

### 重要概念：两者是**重叠**的，总时间 ≈ max(搬运, 计算)，不是相加
GPU 流水线一边搬下一块权重、一边算当前块。所以由**更慢的那个**决定总时间。

### 视角一：第一性原理（各自单独打满硬件的耗时）
| 模型 (decode 1步, b=32) | 要搬的权重 | 有效计算 | **纯搬运** | **纯计算** | **搬:算** |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 58.0 GB | 0.12 TFLOP | **12.08 ms** | **0.117 ms** | **103 : 1** |
| LFM2.5-8B-A1B | 15.5 GB | 0.062 TFLOP | 3.23 ms | 0.063 ms | **52 : 1** |

（搬运 = 权重字节 / 4.8TB/s；计算 = 有效 FLOP / 989 TFLOP/s）

**→ 理想完美重叠下，~99% 的时间该花在搬运（HBM→SM），<1% 花在计算。** 计算完全被藏在搬运的影子里——算力再强也用不上，因为在等数据。

### 视角二：NCU 实测（实际硬件利用率）
| Qwen3 fused_moe (decode) | 实测 |
|---|---|
| DRAM 带宽利用（搬运 pipe 有多忙） | **75%** |
| SM 算力利用（计算 pipe 有多忙） | **16%** |
| No-Eligible（SM 完全空转，在等搬运） | **80%** |

搬运 pipe 75% 一直在忙（主导瓶颈）；计算 pipe 只 16% 忙；SM 80% 周期在等权重搬过来。

### 结论
**decode MoE 的时间几乎 100% 花在"从 HBM 搬专家权重"，计算量小到可忽略（Qwen3 103:1，LFM 52:1）。** 这从第一性原理和 NCU 两个角度一致证明：**这是彻底的 memory-bound（搬运受限），SM 空转是本质而非缺陷。**

---

## 8. ★ 修正优化目标（重要，避免误导汇报）

上面的 103:1 决定了：
- ❌ **"提升 SM 利用率"是错误目标** —— 计算只需要 <1% 的时间，把 SM 填满没有意义（没那么多活给它算）。SM 16%、空转 80% 是 memory-bound workload 的**正常物理结果**。
- ✅ **正确目标是"搬得更快 / 搬得更少 / 搬一次服务更多 token"**：
  1. **搬更快**：DRAM 75% → ~90%（打满带宽），约 1.2×。
  2. **搬更少 / 不浪费**：消除 padding（现在搬进来的权重要去乘一堆零）；这也是 §4 机会 B/C 的核心。
  3. **搬一次服务更多 token（最大杠杆）**：增大有效 batch —— spec decoding、多租户并发、expert 并行。直接改变 103:1 这个比例（batch 越大，分母"有效计算"越多，每字节权重服务的 token 越多）。

**注意 H200 硬件限制（用户指出）**：bf16 WGMMA 最小 tile M=64，而 decode 每专家只有 2-4 个 token。所以：
- 小 tile（M=16）：用不上最快的 WGMMA；
- 大 tile（M=64）：93% 是 padding。
- **两条路在极小 M 下都不好** → 单纯 tile tuning（机会 A）天花板低；真正的解是让有效 M 变大（增大 batch）或换 GEMV 专用 kernel（机会 C）。

**真实 kernel headroom ≈ 把带宽打满 + 消除 padding ≈ roofline 的 1.5–1.9×（v10）**，而**不是**"把闲置 SM 全用起来"。decode 的 SM 本来就该闲，因为它在搬数据。
