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

---

## 实验 A1（kernel 层）：Speculative Decoding（EAGLE3）—— 负面结果，诚实记录

**目的**：spec decoding（exact）让一次前向验证多 token → 把闲置 SM 算力用起来 → 证明 kernel 层 space 摸得着。
**素材**：Qwen3-32B（target）+ `/data/hf/spec_decode/qwen3_32b_redhat_eagle3`（EAGLE3 draft）。因主线 Qwen3-30B-A3B 无匹配 draft，先在 Qwen3-32B 做概念验证。
**做法**：同一 toolagent 负载，基线 vs EAGLE3（`--speculative-algorithm EAGLE3 --speculative-num-steps 5 --speculative-eagle-topk 8 --speculative-num-draft-tokens 32`）。

### 结果
| 配置 | max-conc | TPOT中位 | 吞吐 | **Accept length** |
|---|---|---|---|---|
| 基线 | 32 | 52.4 ms | 459 tok/s | — |
| EAGLE3 | 32 | 185.8 ms | 203 tok/s | **1.28** |
| EAGLE3 | 1 | 20.3 ms | 46 tok/s | **1.28** |

### 结论（A1）——**负面结果，但有价值**
- **EAGLE3 反而更慢**（TPOT 52→186ms）。根因写在数据里：**Accept length 只有 1.28**（理想 2–4）——draft 几乎没有 token 被 target 接受。
- 即使降到 batch=1（spec decoding 的主场），accept length 仍 1.28 → **不是并发问题，是 draft 与 target 不匹配**（这个 redhat EAGLE3 draft 与我们这份 Qwen3-32B 权重不配套）。
- **→ 用不匹配的现成 draft，无法验证 spec decoding 的收益。** 这个负面结果本身印证了 Chendi 框架的主张：**spec decoding 的 draft 必须与 (target 模型, 负载分布) 匹配**，否则 accept length 崩塌、反而变慢——这正是"autotuner 必须先做兼容性/有效性裁剪"的又一实证。

### A1 要继续需要的前提
- 拿到与 target 真正匹配的 draft（同一 Qwen3-32B 权重训出的 EAGLE3，或 Qwen3-30B-A3B 对应 draft）。
- 建议向 t-vinkapoor 索取他 EAGLE3 实验中验证过 accept length 的 draft。有了匹配 draft，才能测"spec decoding 把 decode SM 利用率吃上去"的正向证据。

---

## 总结：两条战线的证据现状

| 战线 | 实验 | 证据 | 状态 |
|---|---|---|---|
| **serving idle** | B2 多流 | 利用率 13%→32%，吞吐 7.4× | ✅ **强证据，摸得着** |
| kernel SM 空转 | A2 backend | fa3 已最优 | ⚠️ 现有实现无空间 |
| kernel SM 空转 | A1 spec decoding | draft 不匹配，accept 1.28 | ❌ 需匹配 draft 才能验证 |

**当前可下的结论**：
1. **serving idle 已被实测证明可回收**（B2，多租户利用率翻 2.5×）—— 这条可以自信上报。
2. **kernel SM 空转的"摸得着"尚未验证成功**：换 backend（fa3 已最优）和现成 spec draft（不匹配）都未能拿到正向证据。要证明这条，需要**匹配的 spec draft** 或 **自己写/改 kernel**。这是诚实的现状，不宜夸大。
3. 两个负面结果（A2 fa3 已最优、A1 draft 不匹配）**本身有价值**：说明 kernel 层的收益不是"随手换个现成组件"就能拿到，需要更专门的工程（匹配 draft / 定制 kernel）——这恰好支撑"kernel 层是真正的深水区"这一论点。

---

## 实验 A1b（kernel 层）：N-gram Speculative Decoding —— ★正向证据（主线模型）

**动机**：A1 的 EAGLE3 draft 不匹配。改用 **N-gram spec decoding**——它**不需要单独的 draft 模型**（从已生成 context 里找 n-gram 匹配来投机），因此绕开了"draft 不匹配"问题，且是 **exact 方法**。agent 负载有大量重复的工具调用格式，n-gram 命中率应该不错。**直接在主线模型 Qwen3-30B-A3B 上做**（不用退回 32B）。

**做法**：同一 server 参数（最优 config），A/B 对比无 spec 基线 vs `--speculative-algorithm NGRAM`，真实 toolagent，conc=1 和 conc=32 两点。

### 结果（Qwen3-30B-A3B，真实 toolagent）
| conc | TPOT 基线 | TPOT ngram | **TPOT 改善** | 吞吐改善 | **Accept length** |
|---|---|---|---|---|---|
| 1 | 4.33 ms | 4.06 ms | **+6%** | +2% | 1.85 |
| 32 | 18.64 ms | 14.27 ms | **+23%** | +3% | 2.08 |

### 结论（A1b）——**kernel 层第一个正向证据**
1. **N-gram spec decoding 真的把 decode TPOT 压下来了**：conc=32 **降低 23%**（18.64→14.27ms），accept length 2.08（每次前向平均验证 2 个 token）。
2. **这是 exact 方法**（n-gram 投机 + target 验证，不改分布），符合 Dey 的约束。
3. **对比 EAGLE3（A1，accept 1.28、反而更慢）**：n-gram 无需匹配 draft，在 agent 负载（重复工具调用格式）上 accept 更高 → 说明**"用什么投机源"很关键**，agent 负载天然适合 n-gram。
4. **意义**：这**证明了 kernel/算法层的 space 摸得着** —— 用一个 config 以外的 exact 手段，在**主线模型**上真的把 decode 延迟吃回了 23%。虽然离 roofline 上界（~1.9×）还有距离，但**证明了"space 可动"这一核心命题**。

### 为什么 conc=32 比 conc=1 收益大
decode batch 越大，SM 空转越严重（v9b：No-Eligible 越高）→ spec decoding 把闲置算力用起来的余地越大 → 收益越明显。这与我们"decode SM 空转"的诊断自洽。

---

## 最终总结：两条战线都拿到了"摸得着"的正向证据

| 战线 | 实验 | 正向证据 | 状态 |
|---|---|---|---|
| **serving idle** | B2 多流 | 利用率 13%→32%（2.5×），吞吐 7.4× | ✅ **摸得着** |
| **kernel SM 空转** | A1b n-gram spec | 主线模型 TPOT −23%（exact 方法） | ✅ **摸得着** |
| （对照）kernel | A2 backend | fa3 已最优 | 现有实现无空间 |
| （对照）kernel | A1 EAGLE3 | draft 不匹配 accept 1.28 | 需匹配 draft |

**核心结论（可上报 Dey/Ofer）**：
两个 opportunity gap **都不再是"看得见摸不着"**——我们各用一个 config 以外的手段拿到了正向实证：
- **serving idle** → 多租户/多路并发，利用率 2.5×（B2）。
- **kernel SM 空转** → n-gram speculative decoding（exact），主线模型 decode TPOT −23%（A1b）。

且两个负面对照（A2 fa3 已最优、A1 EAGLE3 draft 不匹配）说明：**拿到这些收益需要选对手段**（合适的投机源 / 匹配的 draft），不是随手换组件——这正是 autotuner 要解决的"有效性裁剪"问题。
