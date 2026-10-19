# 全512自身状态验收与最终报告

这是对原计划9.4覆盖缺口的独立补足。原runner在D_validate为入围方法各生成512题，但更新后自身状态只采集前128题；两个范围必须分别披露。原run、18份模型测量源码、5份质量分析源码/方案和已封存quality draft/final均保持原档案。本补充不把128题冒充512题，也不靠512条自由生成证明512题状态已收集。

补充使用原正式 `formal/validate/generations/M_NT.jsonl` 与 `M_ST.jsonl` 的全部 `[512,1024)` trace，冻结参数直接自身回放；不拟合新参数、不重复自由生成。全部512重放包含原128子集的重复诊断，其额外离线时间及shadow成本须单列。原128报告继续保留，作为范围明确的子集证据。

源码归档分两份：新的科学plan绑定stats/report两个脚本和三份补充说明的实际副本、manifest及SHA；调度的supplement_ops、launcher、continuation三文件及focused tests另存实际副本/manifest/SHA，与phase launch及运行metadata绑定。纯调度修复不静默改科学plan，也不因其变化使既有科学证据自动失效。原18+5源码档案保持原身份。

## CPU验收接口

`scripts/state_study_validation512_report.py` 提供：

- `read_base_evidence(run_dir, verify_hashes=True)`：复用锁定原reader得到真实未改的base/raw/evidence，不调用旧128正向分类。
- `check_validation512(run_dir, plan_path=None, verify_hashes=True)`：只读验收，不运行模型、拟合或任一种bootstrap。返回 `complete`、`not_required`、`coverage_complete`、`analysis_complete`、`analysis_estimable`、`positive_state_gate`/`own_state_improves`，这些含义分开。
- `evaluate_validation512_result(base, raw, evidence, state512, quality_plan_path, cache_dir=None, allow_statistics=True)`：逐项构建全部正向门槛；不改base/raw、不伪造128字段、不做monkeypatch，也不取 `old_positive AND full512`。
- `build_report(...)`：生成完整九问中文报告。默认不运行新的下游统计；调用者显式允许后，也仅在全部结构/状态先决条件满足时复用或建立原quality/cost缓存。

默认补充plan为 `reports/<run>/validation512/plan.json`，数据在 `reports/<run>/validation512/<plan_hash>/`。验收调用补充模块的 `load_plan`、`output_dir`、`source_snapshot`、`audit_archive`；不反调会触发验收的status，避免递归。

必须核验：前瞻plan与真实源码副本；当前原始生成/校准source snapshot；两个臂各512个唯一ID；1024条整轨迹argmax replay checks；raw_completion与completion自哈希及相互绑定；每个archive/token提交SHA、物理行数、48层、真实 `trace[:-1]` 的T−1位置数组、mask、driver与自己calibration hash；立即EOS的empty身份；保存摘要与completion的文件SHA、完整expected IDs、逐prompt/层覆盖及bootstrap契约。复用补充模块的archive审计还检查moment数组存储的shape/dtype/finiteness；本模块独立复查metadata与position，不重新估计moment数组或bootstrap，不能称为独立数值重算。

缺plan/缺P4/缺数据不是NOT_REQUIRED。只有正式P3/P4及其完整身份明确记录更新分支未触发，才返回NOT_REQUIRED；它表示该分支不适用，绝不称全512已完成，也不凭此宣称科学负结果。损坏、伪造或已发布receipt/summary矛盾为BLOCKED；未完成为INCONCLUSIVE。

## 完成、可估与科学方向分开

COMPLETE表示全部计划工作及来源/归档/保存分析核验通过，不以正负方向或区间可估性决定。真实立即EOS、零有效状态或undefined bootstrap必须如实完成交付，不重排重放追求显著性。只接受冻结分析实际产生的有限区间，或有 `undefined_bootstrap_draw`、`finite_draws` 与空区间等明确记录；空字典、缺CI、未知placeholder不能算已分析。缺失path的不可估理由只有在真实完整zero-observation覆盖支持时才接受。

全512主状态改善条件沿用既有数学口径：token加权的 `E_moment(M_ST own)-E_moment(M_NT own)` 保存95%区间上界<0。bootstrap固定expected `[512,1024)`、配对prompt、10000次、seed20261019、冻结参数不重拟合、empty/inactive仍在全集。prompt等权敏感性完整报告，不临时增加方向否决。更新模型的自身文本/mask可以不同；不能要求它们共享生成token位置。

## 原predicate到新predicate

以下条件完全保留，直接重用锁定纯函数：P0与原protocol身份、全部生成错误检查、`validation_contract`、八臂完整D_select、冻结入围臂完整512请求、`p3_numerical_gate`、C_diag128的`own_state_improves`与bootstrap/coverage、`calibration_provenance`、原 `generation_identities(base, raw, calibrations)`、C_diag `diagnostic_identities`。原generation checker直接接真实未改数据，因此旧validation_state128的生成身份没有被删掉。

旧128诊断archive继续接受 `diagnostic_identities` 的commit/hash/driver/calibration/实际位置核验，但**不要求旧128 own CI有限、改善或显著**。原验证状态的主expected IDs/coverage、bootstrap契约、底层512身份及own改善条件替换为新512证据。旧128正链不单独代表原9.4完成；同样也不能让旧128的不确定或反向区间否决全512主证据。

4096触发、所有finalists固定128敏感性完整性和原前瞻补充的无逆向点估计规则保持不变。质量/成本直接重用原 `downstream_statistics`：四项同时保守exact-binomial质量下界>0，且对NT/strongest的两个paired ratio名义同时区间下界均>=5%。不修改数学规则、seed、迭代数、controls或输入scope；缓存继续以原quality plan hash及真实配对score/length输入绑定，避免因状态范围扩展重复真实成本bootstrap。完整映射也作为结构化 `predicate_mapping` 随报告保存。

最终POSITIVE_EXPLORATORY必须全部条件成立且全512主状态条件成立；原NI-only条件与None仍保持未满足，不声称NI。质量界是保守二项精确界，成本区间是percentile bootstrap近似覆盖。COMPLETE不保证正向；不显著不等于等价；任何结论都不自动解封P6/P7，不消除历史暴露或校准重拟合不确定性。

## 命令与输出边界

只读验收命令不生成新统计：

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/jgu7/envs/moe-state/bin/python \
  scripts/state_study_validation512_report.py \
  --run-dir /scratch/jgu7/moe-state-study/runs/20261019_bf16_eager_v4 \
  --check-only
```

完整交付命令（新统计仅在全部先决条件满足后执行，或复用相同输入缓存）：

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/jgu7/envs/moe-state/bin/python \
  scripts/state_study_validation512_report.py \
  --run-dir /scratch/jgu7/moe-state-study/runs/20261019_bf16_eager_v4 \
  --output-dir /scratch/jgu7/moe-state-study/reports/20261019_bf16_eager_v4/validation512/final_report \
  --allow-downstream-statistics
```

输出 `RUN_REPORT.md`、同正文 `VALIDATION512_REPORT.md`、`validation512_summary.json`。不会写原run、原128报告或Home，不运行模型、不提交Slurm作业。初次离线报告统计不并入原无observer生成计时。

自动主链调用唯一CPU writer `ensure_delivery(run, plan_path=None)`，在 `<plan_hash>/delivery/` 发布以上三文件和 `delivery_receipt.json`。实际报告字节还保留于不可覆盖的 `delivery/bundles/<bundle_hash>/`，receipt另存 `delivery/receipts/`；当前receipt最后原子发布，防止把半写文件当交付。`verify_delivery(..., work_proof=None)`逐项核验工作与报告实际SHA，成功必须同时有 `verified=True`、`delivery_complete=True`。整体COMPLETE还要求这份交付证明；只有raw/analysis完成而报告缺失时仍为RAW_COMPLETE，CPU重入补报告、复用缓存，不重跑状态bootstrap。NOT_REQUIRED也须交付完整九问，并明确没有完成updated512状态；没有quality final时可交付诚实INCONCLUSIVE，不能伪造正向。
