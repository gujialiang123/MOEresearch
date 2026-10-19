# 前瞻质量优越性分析补充

这份规则补齐独立报告的正向分类，不改变冻结的 18 个模型测量源码。原 `protocol.config.quality_constraint_for_internal_validation` 的确采用 NI-only 文字，`validation_freeze.json` 的 NI margin 仍是 `None`；原 NI-only 条件并未满足。本补充依据原用户计划 11.4 允许“明确质量约束”的优先要求，在任何正式验证结果出现前，声明一条质量优越性路径。若将来成立，`POSITIVE_EXPLORATORY` 依据此追加分析计划，而非伪称原 protocol 已包含这些条件。

保持模型、精度、K、校准形式、提示词、样本范围、cap、原主状态阈值和原 validation_freeze 不变。先前 `INCONCLUSIVE` 是缺少完整数据及可执行质量细则，不表示假设被否定；本补充不得按 D_validate 的结果倒推门槛。

## 两步、只写 reports 的前瞻封存

在正式 D_select 尚无输出时声明数学规则与角色草案：

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/jgu7/envs/moe-state/bin/python \
  scripts/freeze_state_study_quality_plan.py --draft-only \
  --run-dir /scratch/jgu7/moe-state-study/runs/20261019_bf16_eager_v4
```

产生独占不可覆盖的 `reports/20261019_bf16_eager_v4/quality_plan.draft.json`，明确 `DRAFT_NOT_FROZEN`，没有选定最终强基线。冻结器同时保存四个独立 Python 分析脚本和本方案文本的实际副本到 `quality_analysis_sources/<analysis_source_hash>/`，含 hash manifest。它们与 run/source_code.tar.gz 中的冻结模型测量代码分开；重建分析环境时同时保留两套来源。

正式六臂 D_select 全部固定 256 题完成后、任何 D_validate 或 validation-state 文件产生前，执行：

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/jgu7/envs/moe-state/bin/python \
  scripts/freeze_state_study_quality_plan.py \
  --run-dir /scratch/jgu7/moe-state-study/runs/20261019_bf16_eager_v4
```

最终 `quality_plan.json` 绑定实际 source/protocol hash、完整六个原始文件 SHA、ID/提示/解析器/结果身份、strongest/reference 选择、草案 hash、创建时间与验证输出不存在的检查。选择规则与 runner 相同：已有非 N 基线按 max(min(strict,tolerant))，再短平均 token、再固定顺序；reference 只在 F/R8/R16 中按该规则选择，S0 始终 L_N。P2 未完成时返回 `PENDING`（exit 2），不写 final plan。出现验证数据后拒绝新冻结。已有 final 只验证，不覆盖、不倒填时间。

草案、最终计划和实际源码副本都不可覆盖。改变分析代码/方案后原 hash 不再匹配，不能静默换规则。没有可核验草案/最终封存，正向分类仍为 INCONCLUSIVE。

冻结器在发布前再次检查验证输出不存在；发布草案前也重查正式选择仍无非空输出。它不暂停正在运行的 auto，协调者必须在六基线选择完成后、P3/更新等步骤仍提供充足时间的窗口发布 final，不能在验证启动边缘操作。验证时另作保守 mtime 核验：现有验证文件若当前 mtime 早于 final 时间则拒绝；append 会改变 mtime，因此该检查仅是附加的 fail-closed 条件，不能证明第一次写入的时序或替代前述协调和 absence 检查。

## 固定统计规则

主下游门控只使用冻结后的 D_validate 512 个配对 prompt；完整八臂 D_select 用于探索表及冻结选择，不再对选择集作确认性筛选。

质量比较为 strict/tolerant 两评分分别对 M_ST−M_NT 和 M_ST−预选 strongest，共四项。每个 prompt 的配对 wins/losses 分别为 M_ST 对、control 错和反向。准确率差为 `(wins-losses)/n`。计算 win 概率的 Clopper–Pearson 下界减 loss 概率的 Clopper–Pearson 上界；八个单尾概率界各用 `alpha/(2*4)=0.00625`，Bonferroni 联合覆盖至少 95%，不要求比较彼此独立。四个差值下界必须全部严格大于 0；零 discordance 不产生虚假的 `[0,0]` 等价证明。这是明确优越性，不是允许下降 2pp 的 NI。

成本端点固定为观测完整请求生成 token 相对减少：`R=1-sum(T_M_ST)/sum(T_control)`。分别对 NT、strongest 比较；每次抽同一组配对 prompt，重新计算 ratio，10000 次、seed=20261019。两项比较用 family alpha=.05、每尾 .0125 的 percentile 区间（单项名义97.5%、两项名义同时95%）；这是 bootstrap 近似覆盖，不是有限样本 exact 95%。任一零分母或非有限 draw 都使区间不可估，不丢 draw。两项下界都须 >=.05。它不是绝对 token 差除以固定均值的近似。

新的下游统计只在完整先决证据成立后运行一次，结果由 plan/source/样本ID/结局 hash 缓存。既有状态 bootstrap 不重跑，首次离线计算时间也不并入无 observer 的生成时间。

## 完整证据链与截断边界

还须满足：P0 原协议通过；正式 P3 数值而非单一 status 证明 native 留出匹配、student 部署失配及同文本 C−D 额外作用；B−A 可估；NT/ST 真正同公式、同 C_fit student trace mask、同 token/逐层逐题计数、正确 source driver 与参数 hash；两者在 C_diag 及固定验证状态子集各自新轨迹的 E_moment 差 M_ST−M_NT 的保存95%区间上界都<0。验证状态只含 own NT/ST 时，其总状态 INCONCLUSIVE 不能替代对应 own 比较。

正式生成需完整固定 ID、相同 prompt/parser/model/protocol、真实 arm/calibration 身份。诊断需全底层 NPZ/token commit SHA 与行数复核、逐 archive 的 driver/observed-calibration/实际位置数组和 mask/count 核验、48层及128题完整覆盖；立即 EOS 的空观察标记保留在抽样全集，不虚构状态。核验只读 metadata/position 数组，不重新估计 moment 数组。

2048 行政截断仍是删失的观测成本，不能推出 cap 之后的自然长度。若任何参评选择臂真行政截断>1%，所有入围方法固定前128题、cap4096 必须完成，不能混合两个 cap。**本补充另行预声明**敏感性一致性要求：对 NT 或 strongest 若出现相对 token 收益点估计<0，或 strict/tolerant 任一准确率差点估计<0，则暂缓 POSITIVE。这个逐项方向否决是新增补充条件，不冒充原计划已有要求。残余截断继续披露，只限定观测 cap 内成本的作用域，不用单个成功样本延长 cap。

全部通过可标 `POSITIVE_EXPLORATORY`，仍不证明完全中介、NI、全球新颖性、部署系统加速或历史独立确认。任何缺失/未定义证据继续 INCONCLUSIVE，明确负状态证据仍按原预声明实用界限报告。P6 比较、独立性与功效/样本量计划仍需另审计，绝不自动解封；P7 亦未获自动授权。

派生报告同时写 `SCIENTIFIC_REPORT.md` 与完整九问 `RUN_REPORT.md` 到独立输出目录，注明分析补充与原始 run；不改 live run 的基础 RUN_REPORT。
