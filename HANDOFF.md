# HANDOFF / 续作交接文档

> 目的:让**另一台机器上全新的 Copilot 会话**能立刻接手这条 MoE 研究线,无需回看历史对话。
> 更新时间:2026-07-17 · 最新 commit 见 `git log`。

---

## 0. 一句话现状

MoE 的「专家数 K → 自由生成长度」**机理研究线**。v13–v22 已完成并有结论;下一步是按
`docs/2026-07-17/v23_v28_mechanism_experiment_plan.md` 做 **v23–v28** 的因果分离实验。
**当前任务 = Phase 0 基础重构 + 实验 1/2 优先**。

---

## 1. 这个仓库是什么 / 来源

- 本仓库(`MOEresearch`)是从原 monorepo `EndtoEnd-auto-optimization` 的 `moe-optimization`
  分支**整体 clone 拆分**出来的独立项目(保留完整 165 条 git 历史)。原仓库保留作备份。
- 拆分时:脚本里所有硬编码的老路径 `/home/.../EndtoEnd-auto-optimization` 已改为
  `/home/.../MOEresearch`。**在新机器上 clone 后,这些绝对路径需要再次适配**(见 §2 注意事项)。
- 模型:`Qwen3-30B-A3B-Instruct-2507`(E=128 专家,原生 top-8,48 层 MoE)。硬件:单卡 H200 级。
- 数据:GSM8K(通过 `datasets.load_dataset("gsm8k","main")`)。

## 2. 环境 / 复现(⚠️ 新机器需适配)

原机器环境(仅供参考,新机器多半不同):
- Python env:`/home/t-jialianggu/.conda/envs/sglang`
- 版本:torch 2.9.1+cu128 · transformers 4.57.1 · CUDA 12.8
- 模型路径:`/data/hf/models/Qwen3-30B-A3B-Instruct-2507`
- HF 缓存需可写:`export HF_HOME=$PWD/.hf_cache`(原机器 `/data/hf/hub` 只读)

**新机器 clone 后必做**:
1. 建等价 conda env(同版本 torch/transformers),或按新机器现况调整。
2. 确认模型可访问(路径可能不同 → 改脚本顶部 `MODEL=` 常量,或改成环境变量)。
3. 脚本里仍有绝对路径默认值(`--out`/`--dir` 指向 `/home/t-jialianggu/work/MOEresearch/...`)。
   在新机器用 `--out`/`--dir` 显式传参,或全局 sed 成新路径。
4. `.hf_cache/`、`context.md`(旧对话 dump)**未纳入 git**,不会被 clone 带来;无需担心。

## 3. 已完成工作(v13–v22)与结论

索引见 `MOE_OPTIMIZATION.md`。核心报告在 `docs/2026-07-15/` 和 `docs/2026-07-16/`。

**最重要的三个(机理线基石):**
- **v20**(`docs/2026-07-16/v20_dynamic_topk_validation.md`)— 正确性修复+验证:
  物理跳过 dropped experts、sync-free 计数、strict parser、prefill/decode 分离、3 种 renorm。
  **K=8 keep-all == 原生模型 0 误差**(MoE 输出/router logits/greedy 生成全一致)。这是后续一切的前提。
- **v21**(`docs/2026-07-16/v21_k_vs_length_results.md`)— GSM8K 500 题固定 K∈{4,6,8,10,12} 剂量曲线:
  **K↓ → 生成变长(单调、显著)**,额外长度 **82–97% 落在 L_to_answer(答案前推理段)**;
  no-#### 仅在 k4 从 5%→19% 才显著;**k6 是安全档**(省 25% 专家、精度不降);K>8(OOD)更差更短。
- **v22**(`docs/2026-07-16/v22_teacher_forced_results.md`)— teacher-forced 终止分析(100 题):
  相同 K=8 前缀下 **logp(EOS) 几乎不随 K 变**,只在 k4 收窄 EOS **margin**(15.6→10.2);KL 随 K 降单调升。
  ⚠️ v22 的局限:固定的是**整条序列**的 token,低 K 从头累积计算,**不能严格隔离"只改当前 step 的 K"**
  —— 这正是新计划里 **实验4(v26)** 要改进的(fixed-KV direct-effect)。

**保守表述纪律**(计划明确要求,务必遵守):在 answer-readiness(v25)证实前,**不得**写
"低 K 让模型主动多思考";在 current-step direct probe(v26)证实前,**不得**写"低 K 直接压低 EOS"。
目前只能说:**降低 K 改变了自回归轨迹和内生输出长度**。

## 4. 代码基础设施现状与局限

核心:`scripts/dynamic_topk_utils.py`(`DynamicKController` + `make_dynamic_forward`)。

**已具备**(新计划可直接复用):
- ✅ 物理跳过、0 误差 keep-all、sync-free 计数
- ✅ 3 种 weight mode:`renorm_survivors / no_renorm / fold_mass_to_top1`(= 计划的 A/B/C)
- ✅ `--phase {decode_only,prefill_only,all}`,prefill/decode 的 avg_k 分开统计
- ✅ strict parser、完整 token ids 落盘、**增量写入 + 配置级 resume**(见
  `run_v20_dynamic_topk_free_generation.py`:每 batch append+flush+fsync;`{cfg}_summary.json` 作完成标记)

**局限(新计划要重构的点)**:
- ❌ **phase 靠 `seqlen>1` 猜**(`make_dynamic_forward` 第~102 行 `is_prefill = seqlen>1`)。
  chunked prefill / multi-token decode 会判错 → 计划要求改成**顶层显式 phase context**(contextvar/thread-local)。
- ❌ **不能同时独立设 prefill_k ≠ decode_k**(现在激活时只用一个 kmin/kmax)。
  实验1(factorial)需要 (prefill_k, decode_k) 任意组合 → 需扩展 controller 持两套 K。
- ❌ **无 layer index / decode step 追踪** → 实验5(layer×time)需要新增。
- ❌ 无 tolerant parser、无 `calibrated_norm_match`(weight mode D)、无 answer-readiness/direct-effect probe。

## 5. 下一步计划(v23–v28)

**完整计划**:`docs/2026-07-17/v23_v28_mechanism_experiment_plan.md`(940 行,务必通读)。

建议目录结构(计划 §三):
```
moe_research/{k_policy,generation_metrics,answer_parsing,stats}.py
scripts/run_v23_phase_factorial.py ... run_v28_k_dose_response.py, analyze_v23_v28.py
```

执行顺序(计划 §十五,按信息增益):
```
0. 基础重构 + 正确性测试   ← 先做这个(地基)
1. v23 Prefill K × Decode K factorial
2. v24 weight renorm / residual-scale ablation
3. v25 answer-readiness probe
4. v26 current-step fixed-KV direct-effect(改进 v22)
5. v27 layer × time intervention
6. v28 K=4..8 dose-response / change-point
```
计划 §929:**先完成 1–4,就能定性**现象属于 prompt-repr / trajectory-accum / answer-formation-delay /
verbosity-format-delay / termination-instability / residual-scale-artifact 中的哪一类。

## 6. 可行性评估(上个会话给出的结论)

**方案可行且质量高**,是在已验证代码上模块化扩展,不是从零。**唯一硬约束 = 计算成本**:
- 计划要求 **bs=1 + 完整 GSM8K test(1319)**。单卡 eager 30B 下,bs=1 单配置粗估 **3–4 小时**;
  实验1(7 配置)≈ 一天,全部实验走完整 test → **数百 GPU 小时**,单卡数周。
- **务实分层**(计划本身留了 smoke/dev/main 三档):
  1. 主机理**先做 v23/v24**(信息增益最高、成本可控);
  2. **只有最关键配置**(v23 的 7 个、v28 的 5 个 K)用完整 1319;其余一律 **dev 200 题 + paired bootstrap CI**
     (长度是低方差量,v21 已证 200–500 题 CI 足够窄);
  3. v27 的 layer×time 组合多 → 全部限 200 题,敏感 block 再细化;
  4. 纯测"长度/正确性"的配置可用中等 batch + 存 censoring flag;只有 v25/v26 这类逐位置 probe 才严格 bs=1。

## 7. 关键约定 / 坑(别踩)

- **长度研究不要降 `max_new_tokens`**(会截断被测的长度)。命中 max 的样本要存 **censoring flag**,不能当正常长度。
- **不要推翻已验证的物理跳过/等价实现**;在其上模块化扩展。
- **benchmark/forward 关键路径禁止逐层 `.item()/.cpu()/.tolist()`**(会引入同步、污染 timing)。
- **所有配置必须作用于完全相同的样本顺序**;已完成的 `(sample_id, config_id)` 不重复计算(resume)。
- **保存逐样本 JSONL + 完整环境元数据**(git commit、模型 revision、torch/tf/cuda、GPU),便于离线重算任意指标
  (这是本项目/用户的一贯偏好:先记录丰富原始 log,新指标事后可算,不用重跑)。
- K>8 是 **OOD probe**,必须单独 section,不能混入正常 dose-response。
- 报告结尾用**三段式**:证据支持什么 / 不支持什么 / 下一步应用。

## 8. 给新 Copilot 的第一步建议

1. 通读 `docs/2026-07-17/v23_v28_mechanism_experiment_plan.md` + 本文件 + v20/v21/v22 三份报告。
2. 适配环境(§2),先跑 `scripts/analyze_v21_k_vs_length.py`(纯日志分析,无需 GPU)确认数据/代码可用。
3. 若模型可用,跑一个 smoke(16 题、`run_v20_dynamic_topk_free_generation.py --limit 16`)确认前向链路。
4. **动手 Phase 0**:建 `moe_research/` 模块,实现独立 prefill/decode K + 显式 phase context +
   计划 §三要求的单元测试(K8/K8 等价、phase routing、physical skip call-counter、no-sync)。
5. 再进入 v23(factorial),dev 200 题先出定性结论。

---

*本文件由上个会话(仓库拆分 + 计划评估)生成,作为跨机器交接锚点。后续可持续更新。*
