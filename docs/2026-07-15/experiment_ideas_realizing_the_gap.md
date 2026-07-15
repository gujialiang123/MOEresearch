# 实验设计：证明 opportunity gap "摸得着"（非 config 手段真能吃回一部分）

**背景**：前面（v6–v10）已经证明 gap **存在且看得见**（config 到头、硬件没吃满、roofline 距屋顶 1.3–2.4×、server idle 85%）。但那是"上界/潜力"，是**看得见**。
**本轮目标**：做**干预实验（intervention）**——换一个 **config 以外**的手段（kernel 实现 / 执行策略 / 调度 / spec decoding），在同一负载上测出利用率 / TBT / idle **真的改变了**，把 gap 变成**摸得着**（可回收的实证）。

**方法论原则**：每个实验都是 A/B 对照——基线（最优 config）vs 干预，**只改一个非 config 变量**，用同样的 NCU/nsys/bench 口径测量，看目标指标是否朝预测方向移动、移动多少。

---

## 分两条战线（对应两个 gap）

| gap | 干预手段（非 config） | 预期能证明 |
|---|---|---|
| **kernel SM 空转** | 换 kernel 实现 / torch.compile / spec decoding | 同负载下 TBT↓ 或 SM 利用率↑ |
| **serving idle** | 换调度/批处理策略、连续批处理、chunked 交错 | 同到达流下 GPU 利用率↑、idle↓ |

---

## 战线 A：证明 kernel 层的 space 摸得着

### 实验 A1（★最高优先，最有说服力）：Speculative Decoding（EAGLE3）
**逻辑**：decode 是 memory-bound + SM 空转（No-Eligible 67-78%）。spec decoding 让**一次前向验证多个 token**，等于把"读一遍权重"摊到多个 token 上——直接提高 decode 的 arithmetic intensity，把闲置的 SM 算力用起来。**这是 exact 方法**（EAGLE 不改目标分布，符合 Dey 约束）。
**素材**：`/data/hf/spec_decode/qwen3_32b_redhat_eagle3`（现成 EAGLE3 draft，注意是 Qwen3-32B 不是 30B-A3B，需确认兼容或找对应 draft）。
**做法**：
- 基线：Qwen3 最优 config，无 spec，测 toolagent 的 TBT + NCU 的 SM%/No-Eligible。
- 干预：开 `--speculative-algorithm EAGLE3 --speculative-draft-model-path ...`，同负载同口径。
**预期证据**：TBT 下降 X%（accept length 决定）、decode 的 SM 利用率上升、No-Eligible 下降。**这直接证明"闲置的 SM 空转能被 exact 方法吃回来"。**
**成本**：~2-3 小时（含调 draft 兼容）。**风险**：draft 是 32B 稠密版，未必匹配 30B-A3B；可能要换 14B 或先在 Qwen3-32B 上验证概念。

### 实验 A2：不同 attention kernel 实现对照（Qwen3）
**逻辑**：同一个 attention 计算，不同 kernel 实现（fa3 vs flashinfer vs triton）的 occupancy / 访存 pattern 不同 → 如果换实现能改变 SM 利用率/TBT，就证明"当前 kernel 不是最优，存在 kernel 层空间"。
**做法**：Qwen3（标准 transformer，三种 backend 都支持），decode 段 NCU 测 flash_attn kernel 的 SM%/DRAM%/occupancy/duration，三种 backend 对照。
**预期证据**：至少一种 backend 的 attention kernel 比基线快/占用高 → 证明"换 kernel 就能动"。
**成本**：~2 小时（3 × NCU decode）。**注意**：LFM 只有 fa3，此实验只能在 Qwen3 上做。

### 实验 A3：torch.compile / CUDA graph 执行层对照
**逻辑**：`--enable-torch-compile` 会做算子融合，减少 kernel launch 数和访存往返 → 如果能降 TBT 或提利用率，证明执行层（非 config）有空间。
**做法**：基线 vs `--enable-torch-compile`，测 TBT + kernel 数量 + SM 利用率。
**成本**：~1 小时。**风险**：compile 首次慢、可能对 MoE 支持不全。

---

## 战线 B：证明 serving 层的 space 摸得着

### 实验 B1（★高优先，直接可做）：真实到达 vs 攒批调度的 idle 对比
**逻辑**：v9d 测到真实单流 GPU idle 86%。如果换一个**攒批/连续批处理**策略（把稀疏到达的请求攒一攒再一起跑），同样的请求流，GPU idle 应该下降。**这直接证明 idle 可被调度策略回收，不只是理论。**
**做法**：同一 toolagent 真实到达流（slowdown 1.0），对比：
- 基线：默认调度，nsys 测 GPU idle（已有 = 86%）。
- 干预：调 `--schedule-conservativeness` / 更激进的连续批处理 / 或人为攒批，nsys 测 GPU idle。
**预期证据**：idle 从 86% 降到 Y% → 证明"调度策略真能吃回 idle"。
**成本**：~2 小时（含 nsys）。

---

## 战线 B（续）

### 实验 B2：多路并发流叠加（模拟多租户）
**逻辑**：单条 toolagent 流喂不饱（并发 6-20）。如果**同时跑 N 条独立的真实到达流**（模拟多用户），GPU 利用率应该随 N 上升直到饱和 → 证明 idle 是"负载不足"造成的、多租户能回收。
**做法**：1/2/4/8 条 toolagent 流并发打同一 server，nsys 测每种情况的 GPU 利用率。
**预期证据**：利用率随流数上升的曲线 → 量化"多少路流能填满这张 H200"。
**成本**：~2 小时。

---

## 推荐执行顺序（按"证据力度 / 成本"排序）

1. **B1（攒批 idle 对比）** —— 最快、最直接证明 serving idle 摸得着，纯策略无需新模型。
2. **A2（attention backend 对照）** —— 中等成本，直接证明 kernel 换实现能动 SM 利用率。
3. **A1（spec decoding）** —— 最有说服力（exact 方法吃回 decode 空转），但要先解决 draft 模型兼容。
4. **B2 / A3** —— 补充证据。

---

## 每个实验产出的"摸得着"证据形态

| 实验 | 基线数字（已有/待测） | 干预后预期 | 证明什么 |
|---|---|---|---|
| A1 spec | Qwen3 TBT ~20ms, SM 16%, No-Elig 80% | TBT↓, SM↑, No-Elig↓ | exact 方法吃回 decode 空转 |
| A2 backend | fa3 flash_attn SM 46% | 另一 backend 更高/更快 | 换 kernel 能动 |
| A3 compile | 基线 kernel 数 + TBT | 融合后更少 kernel / 更低 TBT | 执行层有空间 |
| B1 攒批 | GPU idle 86% (nsys) | idle 降到 Y% | 调度回收 idle |
| B2 多流 | 单流利用率 14% | N 流利用率曲线 | 多租户填满 |

---

## 关键说明（和 Chendi 讨论时强调）

1. 这些实验把结论从 **"gap 存在（potential，看得见）"** 升级到 **"gap 可回收（realized，摸得着）"**——一个具体手段真的把某个指标朝预测方向推动了。
2. **A1（spec decoding）是把 kernel 层 gap 变现最有力的证据**，且是 exact 方法，直接回应 Dey 之前"不引入近似"的约束。
3. **B1（攒批）是把 serving idle 变现最快的证据**，纯策略、当天可出。
4. 即使某个干预只吃回一部分（比如 TBT 只降 20% 而非 roofline 的 2×），那也**足够证明"摸得着"**——因为我们要证的是"space 真实存在且可动"，不是"一次到位打满"。

---

## 待确认（开跑前）
- A1：`/data/hf/spec_decode` 里的 EAGLE3 draft 是 Qwen3-**32B**，我们主线是 Qwen3-**30B-A3B**。需确认能否配对，或改用 Qwen3-32B 做概念验证，或找 30B-A3B 对应 draft。（注：该目录属 t-vinkapoor，他在做 EAGLE3，可以问他要对应 draft。）
- GPU：目前只能用 4/5（0-3、7 是别人的）。多数实验单卡即可，B2 多流可能需要更多显存但单卡够。
