# Observer 成本验收与交付

本模块补足原执行计划8.4:288“observer 的额外时间与显存单列”。它是独立工程成本附页，保留原18份模型源码、原5份质量分析源码/方案、全512补充5份科学源码/文档及已封存计划。统计门槛、模型参数、质量判据、原raw和P6状态均不改变。

固定C_diag题号128..131；A/B/C/D以及真实P4触发后的M_NT/M_ST各自轨迹。复用真实原生成trace、prompt、driver与被观察参数，完整T−1、48层和原六个位置段；A-D保留原common分组。立即EOS保留显式empty身份。只回放，不拟合、不自由生成、不bootstrap，不把512条自由生成冒充512条状态。

每unit两mode前进行一次observer-on `trace[:min(8,T)]` 热身，独立落盘、不计入测量。on/off先后由`(sample_id+scenario_index)%2`平衡；各mode在GC、empty_cache、GPU同步和peak重置后测量完整trace。同步elapsed包含observer、原有限logits检查的共有开销，以及benchmark callback时钟/counter的host instrumentation；没有额外逐回调同步。callback_host_seconds包括隐式同步，不是纯GPU kernel时间。`archive_seconds`单列归档与验收秒，含finish_prompt写盘及随后的读/hash/header核验，不能称为纯写盘时间。报告baseline、绝对peak与peak-minus-baseline allocated/reserved bytes，on−off允许负值且保留，不能截零。

这只是四题×路径、每unit一次配对的工程边际测量。描述性mean/median不是置信区间；不外推全run累计、原run当时实际开销、未测prompt/arm、拟合observer或生产服务速度。正常自由生成计时与显存仍来自原无observer的记录。

`scripts/state_study_observer_cost_report.py` 的纯CPU接口：

- `check_observer_cost(run, plan_path=None)`：返回`complete`、`data_complete`、`not_required=False`、`requirements/errors/status`。A-D总要测，P4未触发只省略更新路径，不使整个成本阶段NOT_REQUIRED。完成与delta正负分离。
- `build_report(run, plan_path=None, *, work_proof=None)`：返回机器结果、完整原九问报告加成本附页、单独成本附页。不执行或调用任何统计bootstrap；原科学delivery必须已验证。
- `ensure_delivery(run, plan_path=None)`：唯一成本交付writer，可重入。原科学report/summary/receipt只读，保持其科学分类。
- `verify_delivery(run, plan_path=None, *, work_proof=None)`：只读。成功时`verified is True`且`delivery_complete is True`，否则不能因非空dict停止续跑。

计划默认`reports/<run.name>/observer_cost/plan.json`，产物`reports/<run.name>/observer_cost/<plan_hash>/`。底层`raw_completion.json`绑定actual source snapshot和全部pair receipts；`completion.json`绑定raw和描述性summary实际SHA。报告独立复核完整角色/固定题号/顺序、每mode plan与self hash、全trace mask/T−1/48层callback数、有限非负elapsed与内存、peak减baseline、两mode argmax相等，A/C/own再核source argmax；B/D teacher-forced argmax不要求等于输入trace。warmup/intent另行核验并以实际SHA进入交付work identity。collector验收读取实际commit、header、位置数组、计数、driver/calibration与字节SHA；不声称重算/全扫描moment值。

描述性summary必须逐项重现实际pair mode及mean/median；单纯存在空字典、标签或receipt不算完成。缺pair/mode/来源为未完成，矛盾/损坏为BLOCKED；model work与delivery分开，前者完而报告缺失使用RAW_COMPLETE，只补CPU报告、不追求成本差方向重跑。

独立最终目录`<plan_hash>/delivery/`包含：

- `RUN_REPORT.md`：原已验证全512/NOT_REQUIRED九问报告原文，追加成本附页。
- `OBSERVER_COST_REPORT.md`：单独成本附页。
- `observer_cost_summary.json`：实际工程成本、原9.3分项、源/校验映射。
- `delivery_receipt.json`：绑定当前work、原科学delivery receipt/report/summary SHA、当前三个文件和immutable bundle；旧`bundles/<hash>/`与`receipts/<hash>.json`保留。

原9.3只读展示native `calibration_cost.json`、updated attempts JSONL、校准参数metadata中的fit token/rollout/shadow计数及推理公式。JSONL以二进制逐物理LF读取，包括记录中可能出现的Unicode行分隔符；末条即使是有效JSON但没有物理LF也拒绝计为已提交attempt，保留原字节不修复。共享S0 rollout相同记录只列一次，不把重复字段相加。native replay计时在finalize/save前截止，updated timer含finalize.save；完整离线墙钟保留null，未记录失败/热身/I/O/序列化等明确未知；同拟合token不推出同总预算，也不新增拟合重跑。

新工程科学计划只绑定cost/report两个script和两份新doc的实际源码档案；调度ops另有actual source归档/launch metadata。当前文档不意味着已实际封存、运行成本模型或提交作业。P6/P7仍属于原用户条件目标，满足原触发条件和前瞻设计/独立性/功效要求后继续；本阶段不自动解封，也不新增用户审批流。
