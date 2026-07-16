# v20：Dynamic Top-K 正确性修复与验证（P0）

**日期**：2026-07-16
**目的**：修复 v18 的实现/测量问题（物理跳过、同步污染、解析、prefill/decode 混合），并在 toy 模型与真实 Qwen3-30B 上验证修复正确。这是"K→输出长度"机理研究可信的前提。

**代码**：
- `scripts/dynamic_topk_utils.py` — 共享 utils（policies / 权重聚合 / 物理跳过 forward / strict 解析 / 无同步计数）
- `tests/test_dynamic_topk.py` — 单元测试（9/9 通过）
- `scripts/run_v20_dynamic_topk_equivalence.py` — 真实模型 equivalence 检查
- `scripts/run_v20_dynamic_topk_free_generation.py` — 带丰富日志的自由生成
**结果**：`results/2026-07-16_v20_equivalence/equivalence_report.json`

---

## 修复清单（对应 GPT review 的 P0）

| 问题 | v18 旧行为 | 修复 |
|---|---|---|
| **P0.1 物理跳过** | `rw = rw * keep` 只置零；`expert_mask` 仍用完整 `selected_experts` → 被丢弃 assignment 仍执行 expert FFN，结果乘零。`avg_k` 是逻辑 K | `expert_mask &= keep`，被丢弃 (token,rank,expert) **不进入** expert_hit / gather / FFN。`avg_k` 是**实际执行 K** |
| **P0.2 去同步** | 每层 `int(keep.sum().item())` → 每 step D2H 同步污染 timing | 计数器为 GPU 长整型 tensor，forward 内**无** `.item()/.cpu()`；整轮结束后 `stats()` 只读一次 |
| **P0.4 policy 语义** | 只有 top-p-within-topk，方向易混 | 三种命名 policy：`top_p_within_topk`(τ↑⇒K↑)、`min_weight_cutoff`(cutoff↑⇒K↓)、`max_dropped_mass`(β↑⇒K↓)，均支持 kmin/kmax，单调性有测试 |
| **P0.5 严格解析** | `last_number()` 取整段最后一个数字 → 低 K 输出长时伪错 | `parse_strict()` 只取最后一个 `####` 后数字；无 `####` 记 `parse_failure`，不静默退化 |
| **P0.8 phase 分离** | K 同时作用 prefill+decode，`avg_k` 混合 | `--phase {decode_only,prefill_only,all}`；prefill/decode K 分开统计 |
| **权重聚合消融** | 只有 renorm | 三模式：`renorm_survivors` / `no_renorm` / `fold_mass_to_top1` |

---

## 验证结果

### 1. 单元测试（toy call-counter，9/9 通过）
- `test_physical_skip_drops_experts`：被丢弃的 expert **不被调用**（call counter 证明），处理 token 数 == 保留 assignment 数。
- `test_fully_dropped_expert_not_called`：某 expert 全部 assignment 被丢 → 完全不执行。
- `test_kept_output_matches_zero_weight_reference`：物理跳过输出 == "全算再置零" 参考实现（数学一致，err<1e-5）。
- `test_equivalence_keep_all`：keep-all 复现 native（3 种 renorm × 2 种 norm_topk）。
- `test_monotonicity`：三 policy 的 threshold→K 方向正确。
- `test_kmin_respected` / `test_prefill_decode_split` / `test_strict_parser` / `test_no_sync_in_forward_source` 全通过。

### 2. 真实 Qwen3-30B equivalence（ACCEPTANCE_PASS = True）
| 检查 | 结果 |
|---|---|
| keep-all vs native：MoE 输出 max_abs err | **0.00e+00**（bf16 精确一致，3 种 renorm 都是 0） |
| keep-all vs native：router logit err | **0.00e+00** |
| 贪心生成 token-for-token（2 prompt） | **完全一致** |
| 动态 τ=0.7 decode_only | `avg_k_prefill=8.0`（prefill 未裁剪）、`avg_k_decode=4.94`（物理跳过生效） |

**结论**：修复后的 monkeypatch 在不裁剪时严格复现原模型 → 任何长度/质量变化都来自 K 本身，而非 patch 数值误差（直接排除 GPT 提出的"替代解释 4"）。

---

## 小验证初见（16 题，字符级线索，噪声大）

| 配置 | avg_k_dec | tok 长度 | L_pre(到####) | L_post(####后) | no_hash |
|---|--:|--:|--:|--:|--:|
| k8 (base) | 8.0 | 267 | 665 ch | 89 ch | 1/16 |
| fixed_k5 | 5.0 | 292↑ | 795↑ | 9↓ | 3/16 |
| dyn_τ0.7 | 4.85 | 301↑ | 723↑ | 8↓ | 6/16↑ |

初步线索（**待全量确认**）：多出的长度在 **L_pre（答案前）**，且 **no_hash（没输出####）随 K 降上升** → 可能同时涉及"推理变长"与"终止/格式受影响"。这正是 v21（全量剂量曲线）+ v22（teacher-forced 因果分解）要回答的。

---

## 下一步
- **v21**：全量 GSM8K 固定 K∈{4,6,8,10,12} 剂量曲线，token 级 L_to_answer/L_post_answer 分解，保存完整 token ids（离线可重算任意指标）。K>8 标注为 OOD（super-native）。
- **v22**：teacher-forced EOS/margin/KL —— 在相同 baseline prefix 上比不同 K，分离"直接终止效应"与"轨迹中介效应"。
