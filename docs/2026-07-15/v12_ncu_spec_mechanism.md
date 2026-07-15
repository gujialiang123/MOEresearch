# v12：NCU 测 spec decoding 的 SM 空转 —— 机制发现（反直觉但重要）

**日期**：2026-07-15
**执行**：GPU 4（baseline）+ GPU 5（ngram），NCU WarpStateStats+SchedulerStats
**目的**：回答"spec decoding 具体减少了多少 SM 空转空间"——用 NCU 直接测 No-Eligible（SM 空转率）在 ngram vs 无 spec 下的变化。

---

## 结果：per-kernel 的 SM 空转几乎没变

| 变体 | 时间加权 No-Eligible（SM 空转） | SM 利用 |
|---|---|---|
| baseline（无 spec） | 77.5% | 21.1% |
| ngram spec | 78.0% | 19.8% |

**→ spec decoding 并没有降低单个 decode kernel 的 SM 空转率**（77.5%→78.0%，基本不变）。

---

## 这不是失败，而是揭示了 spec decoding 的真实机制（重要洞察）

之前 A1b 实测 spec decoding 让 server 端 decode TPOT **降 23%**。但这里 NCU 显示单 kernel 空转率没变。两者**不矛盾**，合起来揭示了一个反直觉的机制：

**spec decoding 降 TPOT 不是靠"让每个 kernel 更满"，而是靠"减少每个 token 需要的前向次数"。**

- 无 spec：生成 N 个 token 需要 **N 次**前向，每次前向的 kernel 都是 78% 空转（memory-bound，读一遍权重只出 1 个 token）。
- ngram spec：一次前向验证 ~2 个 token（accept 2.08），生成 N 个 token 只需 **~N/2 次**前向。每次前向的 kernel **仍然 78% 空转**，但**前向次数少了一半** → 每 token 的墙钟时间下降。

**类比**：spec decoding 不是"让厨师(SM)不发呆"，而是"让厨师一次多做几道菜"——单次操作的空转率没变，但完成同样的菜需要的操作次数少了。

---

## 对"gap 摸得着"论证的修正（诚实）

这个发现**修正**了我之前的因果表述：

**之前（不够准确）**：spec decoding 把闲置的 SM 算力用起来 → 降低 SM 空转。
**修正后（实测支持）**：spec decoding **不降低 SM 空转率**，而是**减少前向次数**，从而降低每 token 延迟（TPOT −23%，A1b 实测）。SM 空转（78%）依然存在——它是 decode memory-bound 的根本属性，需要**别的手段**（更高 occupancy 的 kernel）才能真正吃掉。

**所以两个 gap 的手段要重新精确表述**：
| gap | 手段 | 机制（修正后） |
|---|---|---|
| serving idle | 多流/多租户 | 填满 GPU（真的减少 idle） |
| **每 token 延迟（TPOT）** | spec decoding | **减少前向次数**（不改 SM 空转率） |
| **kernel SM 空转（78%）** | ？（仍未攻克） | 需要更高 occupancy 的 decode kernel —— **这才是纯 kernel 层的硬骨头** |

---

## 关键结论（给讨论）

1. **spec decoding 确实"摸得着"地降了 TPOT 23%**（A1b server 实测），这是真实收益、exact 方法。
2. **但它绕过了 SM 空转，而不是消除了 SM 空转**。decode 的 78% SM 空转是 memory-bound 的本质，spec decoding 用"少跑几次"规避了它，没有真正"填满 SM"。
3. **真正的纯 kernel 层 gap（让 decode kernel 的 SM 空转从 78% 降下来）仍未被任何手段攻克**——换 backend（fa3 已最优）、spec decoding（绕过而非消除）都没做到。这需要**定制更高 occupancy 的 decode kernel**，是更深的工程。
4. **这个诚实的机制区分很重要**：向 Ofer/Li 汇报时，spec decoding 的 23% 应表述为"减少前向次数的算法收益"，而不是"填满了 SM 空转"——否则会被 NCU 数据打脸。

> 注：本次用 bench_one_batch + NCU，decode latency 绝对值失真（NCU 干扰 + 固定 batch 不体现 spec 的多 token 优势）；spec 的真实 TPOT 收益以 A1b 的 server 端实测（−23%）为准。NCU 在此仅用于测 per-kernel 的 SM 空转率（这个不受 batch 口径影响）。

---

## 产物
- `results/2026-07-15_v12_ncu_spec/{baseline,ngram}/ncu_raw.csv` — per-kernel No-Eligible/SM/occupancy
- `scripts/run_v12_ncu_spec.sh`
