# v11：干预实验——证明 opportunity gap "摸得着"（首批结果）

**日期**：2026-07-15
**执行**：GPU 4 + GPU 5
**目的**：把 gap 从"看得见（potential）"升级到"摸得着（可回收）"——用 config 以外的手段，实测某指标真的朝预测方向移动。
**脚本**：`scripts/run_v11b2_multistream.sh`（B2）、`scripts/run_v9_ncu_realworkload.py`（A2 复用+backend 覆盖）

---

## 实验 B2（serving 层）：多路并发流 → GPU 利用率曲线 ★强证据

**逻辑**：v9d 测到真实单流 GPU idle 86%（利用率 14%）。如果 idle 真是"负载不足"造成的，那么同时跑 N 条独立真实到达流（模拟多租户），GPU 利用率应随 N 单调上升。
**做法**：一个 server（最优 config，max-running 256），N=1/2/4/8 条并发 toolagent 流（真实到达 slowdown 1.0，各 200 请求，不同 seed），nsys 测 GPU busy（内核+memcpy 并集）。

### 结果（LFM2.5，GPU 利用率 = busy / 时间跨度）
| 并发流数 | GPU 利用率 | GPU busy | 合计吞吐 |
|---|---|---|---|
| 1 | **13%** | 5.4s | 1087 tok/s |
| 2 | 18% | 7.3s | 2163 tok/s |
| 4 | 25% | 10.3s | 4217 tok/s |
| 8 | **32%** | 13.7s | 8062 tok/s |

### 结论（B2）
- **利用率单调上升 13%→32%（2.5×），吞吐 1087→8062 tok/s（7.4×）**，1→8 流。
- streams=1 的 13% 完美复现 v9d 的 14%（方法自洽）。
- **→ 直接证明 serving idle 是"负载不足"造成的、可被多租户/多路流回收**。这是"serving idle 摸得着"的硬证据（不是理论，是实测利用率真的涨了）。
- 注：8 流仍只到 32%，说明单条 toolagent 流非常稀疏；填满一张 H200 需要更多路并发（趋势明确，未到饱和）。
**出处**：`results/v11b2_multistream_util.csv`、`results/2026-07-15_v11b2_multistream/`

---

## 实验 A2（kernel 层）：attention backend 对照（Qwen3 decode）

**逻辑**：同一 attention 计算，换 kernel 实现（fa3/flashinfer/triton），若 TBT 改变则证明"kernel 实现可影响性能、存在 kernel 层空间"。
**做法**：Qwen3 最优 config，decode 点（b32/in2700/out32），只换 `--attention-backend`，测 decode TBT。

### 结果
| attention backend | decode TBT | 状态 |
|---|---|---|
| **fa3**（基线） | **8.71 ms** | ✅ 最优 |
| triton | 10.27 ms | ✅ 慢 18% |
| flashinfer | — | ❌ JIT 链接失败（`ld: cannot find -lcuda`，env 问题，与 v3 同） |

### 结论（A2）——**证据力度有限，需诚实标注**
- 换 backend **确实改变 TBT**（fa3→triton 差 18%）→ 证明"kernel 实现会影响性能"。
- **但**：基线用的 fa3 **已经是现有三种实现里最快的**，triton 更差、flashinfer 起不来。所以 A2 **没有**证明"能比基线更快"——只证明了"kernel 实现之间有差异、且我们已选到最优的那个"。
- **→ 用现成 backend 对照，无法证明 kernel 层还有可回收空间**（因为已用最优实现）。要真正证明 kernel 层 space 摸得着，需要 **A1（speculative decoding）**——它不是"换一个现成 kernel"，而是"用 exact 方法改变 decode 的计算结构，把闲置 SM 算力用起来"。
**出处**：`results/2026-07-15_v11a2_backend/`

---

## 阶段小结

| 战线 | 实验 | 结果 | 是否证明"摸得着" |
|---|---|---|---|
| **serving idle** | B2 多流 | 利用率 13%→32%，吞吐 7.4× | ✅ **是** |
| **kernel SM 空转** | A2 backend | fa3 已最优，换 backend 更差 | ⚠️ 部分（证明有差异，未证明能超基线） |

**关键洞察**：
1. **serving idle 那条战线已经有硬证据**（B2）——多路流真的把利用率吃上去了。
2. **kernel 那条战线，用现成 backend 不够**——需要 spec decoding（A1）才能证明。这本身也是有价值的结论：现有 attention kernel（fa3）已被 sglang 选到最优，kernel 层的进一步空间需要**算法级手段**（spec decoding）而非换实现。

---

## 下一步：A1（Speculative Decoding，最强 kernel 层证据）

**为什么 A1 才是 kernel 层的关键证据**：decode 的 SM 空转（67-78% No-Eligible）根因是"每步只算 1 个 token，权重读一遍只服务一个 token"。spec decoding（EAGLE，exact）让一次前向验证多个 token → 把闲置 SM 算力用起来、提高 arithmetic intensity → decode 的 SM 利用率应上升、TBT 应下降。

**待解决**：现成 draft（`/data/hf/spec_decode/qwen3_32b_redhat_eagle3`）是 Qwen3-**32B** 稠密版，与主线 Qwen3-**30B-A3B** 不配对。选项：
1. 找 Qwen3-30B-A3B 对应的 EAGLE3 draft（问 t-vinkapoor，他在做 EAGLE3）。
2. 先在 Qwen3-32B（有 draft）上做**概念验证**——证明 spec decoding 能提 SM 利用率/降 TBT，虽非主线模型但足以证明"kernel 层 space 摸得着"。

**建议**：先做选项 2 的概念验证（Qwen3-32B + EAGLE3），拿到"spec decoding 把 decode SM 利用率吃上去"的实测，作为 kernel 层"摸得着"的证据。

---

## 产物
- `results/v11b2_multistream_util.csv` — 多流利用率曲线
- `results/2026-07-15_v11b2_multistream/` — nsys 时间线（4 个流数）
- `results/2026-07-15_v11a2_backend/` — backend TBT 对照
- `scripts/run_v11b2_multistream.sh`
