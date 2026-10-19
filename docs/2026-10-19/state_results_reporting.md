# 独立科学汇总

`scripts/summarize_state_study_results.py` 只读一个 run 目录，不导入实验模块、不加载模型、不安装依赖。它补充 runner 的阶段状态及各 scope 的生成表，按原计划第 15 节回答九个问题。原始结果不存在时写 `PENDING`，不会把未运行当作负结果。最终归档身份核验按需使用已安装的 NumPy 读取 metadata/position 数组，不重新估计 moment 数组。

使用当前解释器或标准 Python 3.10+：

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/jgu7/envs/moe-state/bin/python \
  scripts/summarize_state_study_results.py \
  --run-dir /scratch/jgu7/moe-state-study/runs/RUN_ID \
  --output-dir /scratch/jgu7/moe-state-study/reports/RUN_ID
```

输出 `SCIENTIFIC_REPORT.md`、同正文的完整九问 `RUN_REPORT.md` 与 `scientific_summary.json`。报告目录必须在 run 之外且不在 Home，避免修改实验原证据；不提供 `--output-dir` 时只向 stdout 输出 Markdown。不改 live run 的基础 RUN_REPORT。

最终复核可添加 `--verify-diagnostic-hashes`，通过 CPU/I/O 读取全部已提交诊断 NPZ/token 文件并检查 commit 中的 SHA256；这可能需要较多磁盘读取。默认仅核对诊断清单、文件大小和提交文件，并明确披露未做完整 hash 复核。

脚本按物理 LF 行读取 JSONL，保留问题中的 Unicode 行分隔符。运行中仅读取开始时的文件长度，未写完的末行不计入；缺失配对、重复 ID、问题/协议身份不一致、EOS 后仍有 token、真实 cap 与计数冲突均显式报告。它不修复或覆盖输入。

具体进程状态从 `launch.json` 核对 `hostname`、`boot_id`、`pid` 和 `proc_start_ticks`。仅在同主机、当前 boot ID 相同、`/proc/PID/stat` 的 PID/启动 ticks 相同、`cmdline` 的 `--run-dir` 与当前证据目录相同且命令符合启动记录、进程非 zombie/dead，并再次读取 stat 仍符合身份时，记录 `VERIFIED_LIVE`。核验只代表读取时刻。`study.lock` 是否存在或被持有只是辅助信息，不能替代该证明。

跨主机、缺少启动身份、读不到 `/proc`、boot/ticks/命令不符或 zombie 状态均记录 `UNKNOWN` 及具体原因，不直接宣布整个实验已 terminal，也不据此建议重复启动。报告可给出对应 `scontrol show job JOB_ID` 只读补证据命令；脚本本身不执行调度命令，作业分配状态也不单独证明具体 PID 存活。

生成描述统计从逐请求保存字段重算，不重跑解析器。Strict/tolerant、长度均值/中位数/p90、EOS/cap、失败、重复/格式、无 observer 计时、显存与标记明确的专家简化预算并列。失败尝试与最终请求分开，不能用重试次数作为独立样本。最多导出 20 个正式解析分歧例子。

置信区间引用已有摘要；引用前核对对应原始请求身份、ID、数量和点估计，不重新做 bootstrap。旧生成摘要没有对全部逐样本结局的密码学绑定，因此脚本不声称独立重算了区间。正式状态分析另外对账完整 trace 的 prompt、48 层、decode 位置和 token mask；共同前缀不替代长后缀。

不从不显著推出质量保持，不从短输出推出收益。原实现 NI-only 条件与 margin=None 保留；新 [前瞻质量分析补充](state_quality_plan.md) 允许在验证前冻结明确优越性规则。只有完整状态/同文本重估/新自身轨迹/全部原始身份与诊断 hash 证据，再加四项同时质量下界>0和两项相对成本区间下界≥5%，才标 `POSITIVE_EXPLORATORY`；它依据显式追加计划，不冒充原 NI 条件满足。没有有效计划或缺任一证据时仍 `INCONCLUSIVE`。状态假设的有限作用域 `NEGATIVE` 仍要求正式及冻结后复核满足原预声明实用界限、完整对账和底层 hash 复核。实现/证据身份/运行错误为 `BLOCKED`。

新质量界与相对 token ratio bootstrap 不重复已有生成/状态分析。只有前置链完成才懒执行，按 plan/source/ID/实际结局 hash 缓存；缓存命中不再计算。新统计属于离线分析时间，不能并入无 observer 的生成成本。P6 不自动解封，仍需独立性和前瞻功效计划。

只使用指定 run 的结果。历史已暴露数据、确认池独立性未核实、固定参数 bootstrap 不包含校准重拟合不确定性，以及 Python 研究后端的计时外推边界始终保留。P6/P7 未运行不会被写成完成。
