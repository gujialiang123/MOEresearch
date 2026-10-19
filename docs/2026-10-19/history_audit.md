# GSM8K 数据来源与历史使用审计

数据固定为 `openai/gsm8k` / `main` / `740312add88f781978c0658806c59bc2815b9866`。题目哈希使用原始 question 的 UTF-8 字节，不做空白归一化。

## 固定划分

|集合|train 半开区间|样本数|
|---|---|---:|
|C_fit|[0,128)|128|
|C_diag|[128,256)|128|
|D_select|[256,512)|256|
|D_validate|[512,1024)|512|

上述集合对本次拟合互斥；C_diag/D_select 不能宣称历史未使用。

## 原始文件与校验

- train: 7473 条；parquet SHA256 `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d`；JSONL SHA256 `5e10fad46ec942b079dde00cad28bfb1061d2b13fde2587f0d7591b37f178abf`。
  [固定 revision 原始文件](https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet)；本地 `/scratch/jgu7/moe-state-study/data/train.jsonl`。
- test: 1319 条；parquet SHA256 `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`；JSONL SHA256 `dfade68a3c738b4de4c0bbc108235867181c51182119d173cb3967566935a605`。
  [固定 revision 原始文件](https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet)；本地 `/scratch/jgu7/moe-state-study/data/test.jsonl`。

## 已保存的历史证据

审计当前 checkout 中 876 份 JSON/JSONL，包含逐样本生成、逐位置 probe、分组 raw、汇总和 manifest。逐文件 SHA256、schema、ID、来源判定与错误见 history_audit.json。

|历史目录|观测 ID（按保存的 split/源码语义映射）|声明范围|原始样本记录|
|---|---|---|---:|
|2026-07-15_v17_gsm8k_topk|未保存|test: 200 / [[0, 200]]|0|
|2026-07-15_v18_dynamic_topk|未保存|test: 200 / [[0, 200]]|0|
|2026-07-16_v20_free_gen|test: 16 / [[0, 16]]|test: 16 / [[0, 16]]|48|
|2026-07-16_v21_k_vs_length|test: 500 / [[0, 500]]|test: 500 / [[0, 500]]|2500|
|2026-07-16_v22_teacher_forced|test: 100 / [[0, 100]]|test: 100 / [[0, 100]]|500|
|2026-07-20_v23_phase_factorial|test: 500 / [[0, 500]]|test: 0 / []|3500|
|2026-07-20_v23_phase_factorial_gpu4|test: 500 / [[0, 500]]|test: 500 / [[0, 500]]|1500|
|2026-07-20_v23_phase_factorial_gpu7|test: 500 / [[0, 500]]|test: 500 / [[0, 500]]|1500|
|2026-07-20_v23_preflight|未保存|test: 0 / []|0|
|2026-07-20_v24_weight_ablation|test: 200 / [[0, 200]]|test: 200 / [[0, 200]]|1800|
|2026-07-20_v24b_calibrated_norm|test: 200 / [[0, 200]]|test: 200 / [[0, 200]]|600|
|2026-07-20_v25_answer_readiness|test: 80 / [[0, 80]]|test: 0 / []|240|
|2026-07-20_v26_direct_effect|test: 60 / [[0, 60]]|test: 0 / []|7200|
|2026-07-20_v28_dose_response|test: 500 / [[0, 500]]|test: 500 / [[0, 500]]|2500|
|2026-07-20_v28_fulltest_k4|test: 1319 / [[0, 1319]]|test: 1319 / [[0, 1319]]|2638|
|2026-07-20_v28_fulltest_k6|test: 1319 / [[0, 1319]]|test: 1319 / [[0, 1319]]|2638|
|2026-07-20_v28b_dose_no_renorm|test: 500 / [[0, 500]]|test: 500 / [[0, 500]]|2500|
|2026-07-20_v28b_dose_no_renorm_gpu4|test: 500 / [[0, 500]]|test: 500 / [[0, 500]]|1500|
|2026-07-20_v29_decode_norm|未保存|train: 128 / [[0, 128]]|0|
|2026-07-20_v29_decode_norm_pilot|train: 200 / [[128, 328]]|train: 200 / [[128, 328]]|1800|
|2026-07-20_v29_decode_norm_smoke|train: 32 / [[128, 160]]|train: 32 / [[128, 160]]|288|
|2026-07-20_v29_partial_renorm|test: 500 / [[0, 500]]|test: 500 / [[0, 500]]|5500|
|2026-07-20_v30_gain_controls|test: 200 / [[0, 200]]|test: 200 / [[0, 200]]; train: 64 / [[0, 64]]|1400|
|2026-07-20_v30_partial_renorm_pilot|train: 200 / [[128, 328]]|train: 200 / [[128, 328]]|2200|
|2026-07-20_v30_partial_renorm_smoke|train: 32 / [[128, 160]]|train: 32 / [[128, 160]]|352|
|2026-07-20_v31_gain_controls_pilot|train: 200 / [[128, 328]]|train: 328 / [[0, 328]]|1800|
|2026-07-20_v31_gain_controls_smoke|train: 32 / [[128, 160]]|train: 96 / [[0, 64], [128, 160]]|288|
|2026-07-20_v31_pulse_recovery|test: 40 / [[0, 40]]|test: 40 / [[0, 40]]|40|
|2026-07-20_v32_pulse_recovery_smoke|train: 12 / [[128, 140]]|train: 12 / [[128, 140]]|12|

记录数包含不同方法及同一题的多位置记录，不能当独立样本量。声明范围与实际观测分列；声明不是成功运行证明。旧数据缺少 revision/题目哈希，因此这些映射属于有来源依据的 ID 推断。gold 一致也不能单独确认题目身份。

## 确认候选封存与不确定性

已封存 train[1024,2048) 的 1024 个候选题目哈希；排除 0 个已观测/已声明/与核心划分重题的候选。候选列表哈希 `0a363ea41c03457fbd3ae0386e5b55f60d13280f28ac86f29a5c4b9da855a279`。

**INDEPENDENCE_UNVERIFIED**：这只证明对当前可见记录的排除和选配置前的候选固定，不能证明历史从未使用。旧校准参数部分未保存拟合 ID/count；未恢复所有 Git 历史、外部或已删除日志。完整 test 已有历史使用，不可重新称为全新确认集。

候选池不是正式 P6 的最终样本量。解封前还须冻结比较、主要指标、质量约束与功效/精度计划。不得根据候选结果换题。

未解决记录：
- `2026-07-20_v24_weight_ablation/norm_match_scalars.json`: Calibration exists, but original fitting IDs/count/revision were not saved; current script default n=64 is not execution evidence.
- `2026-07-15_v17_gsm8k_topk`: Aggregate-only GSM8K results save limit but no per-question IDs or hashes; declaration is inferred using the current checked source's test-prefix convention.
- `2026-07-15_v18_dynamic_topk`: Aggregate-only GSM8K results save limit but no per-question IDs or hashes; declaration is inferred using the current checked source's test-prefix convention.
- JSON 解析错误：0。

## 重现命令

```bash
/home/jgu7/work/e2e-runtime/venv/bin/python scripts/audit_state_calibration_data.py \
  --repo /home/jgu7/work/MOEresearch --out /scratch/jgu7/moe-state-study/data
```

脚本不加载模型；下载、parquet、导出、manifest 和封存文件均在 scratch。已有不同内容的导出或封存会报错，不能静默覆盖。
