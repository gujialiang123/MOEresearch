# 冻结实验的 Slurm 续跑

本包装为现有 `/scratch/jgu7/moe-state-study/runs/` 下的 run 接续原 P0–P5、另行前瞻封存的全512 validation own-state 补充，以及独立 observer 成本工程测量和各自真实报告交付。它不修改模型策略、样本数、cap、runtime/config 或原始逐样本结果；不安装环境、不下载模型、不创建 SQLite。资源仍为一张 B200、8 CPU、128G、一天。合法的新 batch 交接完成后，独立 ops helper 保持同 PID exec 原 auto 或固定补充命令，并只预排一个 `afterany:当前job` 后继；不会从仍存活的旧科学进程另启模型，也不建立常驻 daemon。

当前 v4 的旧作业是 `2652067`。提交前只读检查：

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/jgu7/envs/moe-state/bin/python \
  scripts/launch_state_study_resume.py --check-only \
  --run-dir /scratch/jgu7/moe-state-study/runs/20261019_bf16_eager_v4 \
  --previous-job-id 2652067
```

旧 PID 仍存活或旧作业非终态时，检查返回 `REFUSED_NO_MODEL_LOAD`、退出码 3；这是预期的拒绝重复启动，不修改 `launch.json`。`--check-only` 永不写启动历史或执行模型。P5 complete、P6_P7 gated_pending 和非空 RUN_REPORT 只表示 core 完成，随后转入独立补充状态检查，不能据此宣布所有工作结束。全512补充及交付核验通过（或正式未触发且 CPU 交付核验通过）后，还必须完成独立 observer 成本测量及交付，才返回 `ALL_REQUIRED_PHASES_COMPLETE_NO_MODEL_LOAD`。

下例记录已经提交的唯一依赖续跑命令；当前已有 `2655921`，不得重复执行：

```sh
mkdir -p /scratch/jgu7/moe-state-study/slurm
sbatch --dependency=afterany:2652067 \
  scripts/state_study_resume.sbatch \
  /scratch/jgu7/moe-state-study/runs/20261019_bf16_eager_v4 2652067
```

不要给同一旧启动排多个后继，也不要在交接期间手动 requeue 旧作业。core 的有效前任是原 run 的最新 `launch.json`；进入补充后依次使用报告目录下独立 `validation512/ops/launch.json`、`observer_cost/ops/launch.json`，以最近实际注册的阶段为准。后续合法 allocation 按该实际前任的 job ID 自动预排一跳；人工提交前必须先查看下述 ops intent/commit 及实际调度器，不能并行补排。helper 会拒绝参数与最新有效启动记录不符的过期后继。Slurm 的 `afterany` 在旧作业终止后释放依赖；满足的 dependency 会从作业记录移除，因此启动时的 `Dependency=(null)` 不能证明提交时没设依赖。[官方 sbatch 文档](https://slurm.schedmd.com/sbatch.html#OPT_dependency)

helper 的启动条件：

1. 当前 18 个冻结源码文件逐个匹配 `source_hashes.json`，清单、protocol source hash/self hash、manifest 身份都一致；增加/改变冻结模块也拒绝。
   当前 manifest、任一 stage 或当前 P0 文件若明确为 `error`/`FAILED_IMPLEMENTATION`/`FAILED`/`FAIL`，在任何 handoff claim、排队或模型加载前拒绝。NEGATIVE、INCONCLUSIVE 和 `scientifically_not_triggered` 本身不是运行错误，不代替剩余 P5 工作。
2. 核验旧 launch 的同主机 boot/PID/start ticks/命令行，具体旧 PID 仍 `VERIFIED_LIVE` 则拒绝。旧锁文件或锁观测不能证明进程终止。
3. 用只读 `scontrol -o show job OLD` 核验精确旧 job 的明确终态及有效 EndTime；RUNNING/PENDING/CONFIGURING/COMPLETING、附加 `+`、查询不明均拒绝。仅当控制器明确返回“Invalid job id specified”才查询 `sacct -j OLD --noheader --parsable2 --format=JobIDRaw,State%64,NodeList%256,End%64`，严格只取 exact JobID，排除 `.batch/.extern` 等 steps；需要唯一、终态白名单、有效 End。记录冲突、截断或未知不自动放行。列宽显式放大，避免截断合法的 OUT_OF_MEMORY/CANCELLED 状态。
4. 实际执行必须在不同的新 Slurm batch allocation 内，核对 RUNNING、当前用户和 `/proc/self/cgroup` 中的 job ID，并在 claim 前核验同一 B200/8CPU/128G/24h、partition/account/QOS、Requeue=0；只读检查不要求已取得新 allocation。
5. 用有效旧 launch 内容的 SHA256 为文件名，将原始字节独占原子发布到对应 `launch_history/`；然后原子更新新 hostname、boot ID、PID/start ticks、cgroup、job ID、固定命令、日志路径和调度证明。core 原有交接照旧更新 run/launch，最后 `os.execvpe` 原 auto；补充交接只写独立 ops 元数据，最后 exec 固定 validation512 run。两者 PID/start ticks 均保持一致。

实际交接还会在旧 job 已确认终态、新 auto 尚未开始时，读取旧 manifest 原字节，并逐物理 LF 行验证所有 `**/generations/*.jsonl`。为每个文件记录字节数、物理行/记录数、最后 sample_id、文件/最后记录 SHA256 和文件身份；保留 Unicode 行分隔符，不使用 `str.splitlines()` 拆记录。未写完末行、JSON 解析异常、缺少/重复 ID、捕获期间文件或清单变化都会明确拒绝，不修 raw。

不可覆盖证据位于 `launch_history/boundaries/<旧launch内容hash>/`，含旧 manifest 原文、`boundary.json` 与最后写入的 hash commit；新 launch 的 `predecessor_snapshot` 指向它们。这样即使下一次 `runner.status` 覆盖 manifest.runtime，也能保留旧硬件/软件记录，并用相邻交接的物理行/字节边界将追加段归属不同 allocation。边界之前的全部记录仅表示“此前已存在”，不把它们一律归给最后一个旧 job，也不虚构逐行时间戳。热身、回放拟合及未落盘失败耗时仍不能从生成行推算。

`launch_history/<sha>.json` 同时阻止两个后继覆盖同一旧启动，不依赖 NFS advisory lock，也不是通用锁服务。若在独占发布旧历史后、边界证据发布/更新 launch/exec 前崩溃，后续同一交接会保守拒绝；必须人工对照旧/新 launch、history/boundaries、Slurm 与具体 PID 再处理，不能盲删历史 claim。已发布历史和边界始终保留。exec 失败另写 `launch_exec_failure.JOB.json`。

普通返回后，sbatch 保留科学主命令退出码，自动运行原独立汇总 `--verify-diagnostic-hashes`，输出到 `/scratch/jgu7/moe-state-study/reports/<run.name>/`；再调用已有绘图脚本，缺少实际摘要时只记录 skipped，不合成结果。保存各自的 `report-JOB.log`、`plots-JOB.log` 和 `batch-exit-codes-JOB.json`。后处理失败会单独报错，不改变科学主命令退出码，也不把失败标为报告成功。这个旧 batch 尾部仍是原128诊断的派生报告，不能代替新512补充的验收与交付；新交付位于 `<plan_hash>/delivery/`，不会被旧尾部报告覆盖。

Slurm 墙钟到时若整个 batch shell 被终止，不能保证执行末尾后处理；下一次续跑仍由原 runner 的持久化记录恢复。本包装不使用信号临时缩短 cap、修补 raw、重拟合不同协议或变更科学门槛。未知/损坏数据仍按原 runner 拒绝并留下明确错误。完成 P0–P5 也不自动解封 P6 或执行 P7。

## 本次实际提交记录

已提交唯一作业 `2655921`，依赖 `afterany:2652067`，提交/核验记录在 v4 run 的 `slurm_continuation.json`。2026-10-19 08:04 EDT 只读控制器核验仍为 PENDING、依赖未满足，资源为1 B200/8 CPU/128G/24h，不与当前模型并行。控制器已确认 `Requeue=0`，当前脚本亦加入 `--no-requeue`：本 helper 的 predecessor 绑定属于明确的单次交接，同一 batch 自动 requeue 时旧参数会过期。因此后续抢占后应以最新有效 launch 的 job ID 建立一个新的 afterany 后继，保留原始数据；不复用旧参数自动循环重启。core 提前完成不能作为取消该后继的理由，它可能仍需进入512补充或 CPU 交付。

## 只预排一跳的自动延伸

`scripts/state_study_continuation.py` 在原 handoff 的旧主机/boot/PID、终态、当前 allocation、18源码、物理行边界与独占 history claim 全部成功后，由 launcher 同 PID 调用，随后仍 `exec` 原冻结 auto。它不进入冻结的18源码清单，不修改原始数据。已提交 `2655921` 的 batch body 是旧拷贝，但它调用共享路径上的最新 Python launcher，因此未来合法启动时可接入此 ops 动作；此修改本身未提交任何新 job，也不改变当前 `2652067`/`2655921`。

每个当前 job 的证据目录为 `/scratch/jgu7/moe-state-study/reports/<run.name>/continuation/<current_job>/`：

- `intent.json`：fsync 后独占发布，绑定 run/protocol/source/launch hash、唯一 job name、owner、完整 sbatch 参数、资源、依赖和 ops 脚本 hash。已有 intent 永不覆盖。
- `submission.json`：实际 `sbatch --parsable` 返回码、stdout/stderr；不会预填一个作业 ID。唯一 job name 由 run/protocol 与 parent job 构成。
- `committed.json`：仅在 `scontrol` 精确核实实际 ID、owner、PENDING、唯一 name、`afterany:当前job`、1 B200/8 CPU/128G/24h、partition/account/QOS、Requeue=0、batch 路径和工作目录后发布。后继参数中的 predecessor 也是当前 job。
- `outcome-*.json`：每次实际调用的独立状态记录，历史不覆盖。只有 `QUEUED`/`ALREADY_QUEUED` 表示已核实排队；`REFUSED`/`SUBMISSION_UNKNOWN` 不能写成排队成功。

每次先查看当前用户的活动 job，识别同 run 的唯一名称前缀及原已知后继；另有 pending/unknown 后继则不再提交。网络 timeout、返回 ID 不明确、核验失败或“intent 已有但没有精确活动后继”均保留 UNKNOWN，不盲重提。再次调用时，只有唯一 name+owner+精确依赖及全部资源匹配的实际 controller 记录才能 adopt；没有确定证据就等待人工核对。提交失败不杀原科学 auto，但意味着未来自动覆盖未获保证，必须如实报告。

core 完成仍按 `P5 complete + P6_P7 gated_pending + 非空 RUN_REPORT` 判断，但现在进入补充 dispatch，不再作为整体停止条件。全512补充 `READY` 进行固定512×2收集；`RAW_COMPLETE` 只补缺失的分析或真实交付，已提交分析不重做状态10k bootstrap。真实完整验收且交付 receipt 核验通过的 `COMPLETE`，或正式 P3/P4 数值与身份验证证明不触发更新、并已完成 CPU 交付的 `NOT_REQUIRED`，只使 resolver 转向 observer_cost；不停止整条链。observer_cost 的 `COMPLETE` 还须 `delivery_complete is True` 才整体停止且不再排队。任何阶段缺 plan、坏 plan、WAITING/PENDING 均不等于补齐。所有 required phases 完成后，已预排的一跳可在下一次合法启动时无模型核验收尾；操作者也只能在核验整体完成后取消仍 pending 的确定 ID。科学结论 NEGATIVE/INCONCLUSIVE 不阻止固定范围完成，也不触发为了追求正结果而追加实验。

显式科学错误不重试。core 任意非0 `scientific_exit_code`、任一阶段 error3、OOM、CANCELLED、UNKNOWN、损坏记录或来源不符均保守拒绝。非core的实际注册阶段只有下述已完成工作、纯交付Pending的CLI2例外；不能泛化忽略FAILED。普通科学采集的TIMEOUT/PREEMPTED/NODE_FAIL仍可在原检查通过后恢复。提交 RPC 与 claim/exec 之间崩溃留下的 UNKNOWN/claim 都保留，不删除来“恢复”自动循环。

### 已完成工作后的有限 CPU 交付重入

冻结 full512 与 observer_cost CLI 在真实工作及分析已经完成、报告交付仍Pending时返回2。旧batch会忠实保存精确job的 `scientific_exit_code=2` 并以2结束，Slurm通常因此为FAILED。ops只对**实际注册的非core阶段**核验如下例外：当前状态严格为RAW_COMPLETE；full512的`completion_proof.complete`或cost的`work_proof.complete`必须是bool True；delivery必须真实待完成（full512允许MISSING/PENDING，cost只允许PENDING），不能是INVALID；实际plan、科学来源、原始工作、manifest、固定命令、launch、claim/history、job与完整batch三字段记录均须通过。raw-only尚无完整分析证明不能使用此例外。

每个(run,phase,plan,completion文件SHA)只允许一次自动CPU交付重入，独占证据在`<phase>/ops/delivery_retries/<identity>/intent.json`。check-only只读证明资格，不消费或写intent；实际allocation和独占handoff完成后才发布intent，绑定当前真实job/PID/boot/ticks及原失败启动。注册run复核同一已消费intent须对同job/PID幂等通过，不能误把自身合法重入挡住。

retry job仍最多预排一个后继，以便交付成功后继续cost或整体核验。如果这次delivery-only retry仍未真实交付，下一helper无模型停止为`STOP_FOR_DELIVERY_REVIEW`，不再排后继；预算涵盖该retry allocation的所有终态，包括TIMEOUT/PREEMPTED/NODE_FAIL或batch尾记录未落，不能借基础设施状态重新开report循环。这个预算不影响没有delivery-retry标记的正常core/采集/尚无analysis阶段，也不改变科学样本或方法。

root/operator可核验原因、修复I/O并由唯一CPU writer完成真实交付；这属于ops复核，不是新增用户审批。当前来源、plan、工作及实际receipt再次验证为COMPLETE+delivery True（或full512已验证NOT_REQUIRED交付）后，历史注册CLI2可记为`resolved_delivery`，继续后续cost或整体收尾；记录实际receipt身份，保留全部旧batch历史与retry intent，不重开预算、不删claim、不永久毒化已修好的工作。科学错误、INVALID与非CLI2失败仍不适用。

此机制可跨第三及更多 allocation 持续恢复既定 core、已封存512补充、observer工程测量、验收和交付，前提是调度器允许提交/提供资源且未出现需人工处理的错误；它不保证排队等待时间或完成日期，不缩小固定范围，不自动进入 P6/P7。必要检查使用 fake Slurm 和 CPU 文件夹：完成/科学错误跳过、活跃 PID 拒绝、同 job 幂等、未知提交不重提/精确 adopt、错误来源/资源/依赖拒绝、真实返回 ID 的 durable commit，以及 queue failure 后仍保持原 PID/history/boundary/exec 合约。

## 已封存的全512补充与独立交接

2026-10-19 **08:04:42.123951 EDT（12:04:42.123951Z）**，root 在正式 validation 文件仍为空时实际封存 `reports/20261019_bf16_eager_v4/validation512/plan.json`。`plan_hash` 为 `c49f4f18696e032ca3b75960cfbbf4787137c8da513af49b10c50c7791bb9685`，文件 SHA256 为 `ca90bceba14ac723be1e343701eaeeb171762d2784dd23535a31860b9e584c5c`。这是原9.4范围的前瞻补足，不是已完成512的声明，也不替代等待正式 D_select 的 final quality plan。

该 plan 绑定原 protocol、旧18+5来源和既有 quality draft，另将新 collector/report 两份脚本及三份科学说明实际归档到 `validation512/source_archives/0f318658fcbc2c80279ce1bd20ee584c1e54eef152a690a97c528e2af54bf4af/`。补充固定读取原 `formal/validate` 的 M_NT/M_ST、IDs512–1023，48层全部有效 `trace[:-1]` decode、原六位置 bins、FP64/ddof0、token与prompt统计、seed20261019/10000重采样；不新生成、不重拟合、不复用前128归档冒充512。只有原正式 P3/P4 更新门控实际触发，且 core 已完成并终止后，才允许注册的 GPU 收集。

封存后独立只读 `load_plan` 与 ops CPU 解析为 `WAITING_P4`、`core_complete=false`；未 import torch。原主 PID2581314 仍为同 host/boot/start ticks/argv 的 VERIFIED_LIVE，原 launch 不变，独立 ops launch 尚不存在。实际封存证据是 `validation512_actual_plan_seal.json`；它不是模型运行或补充完成证据。

未来合法补充 handoff 的元数据在 `reports/<run.name>/validation512/ops/`，不会覆盖原 run/launch：

- `launch.json` 绑定 task/phase、plan/protocol/source、同 PID/boot/start ticks/argv/cgroup、当前 allocation、原 core launch SHA、实际前任 launch SHA、旧 Slurm 与原 core 终态证明、checkpoint 边界和 exec 合约。
- `claims/<previous_launch_sha256>.json` 由共享 scratch 独占发布，绑定 run/plan/current job/PID/owner/host/boot/ticks 和前任 hash。跨节点互斥不能只靠本机 `/raid` flock；collector 在加载模型前重新验证共享 claim、独占 history 和实际 allocation。
- 第一轮前任是原 core launch；后续前任是最新独立 supplement launch。两者都须拒绝 VERIFIED_LIVE 旧 PID 和 UNKNOWN/非终态 Slurm；独立阶段保留原 core 的终态证明与原 launch 身份。
- `ops_source_archive` 与 `ops_source_sha256` 保存实际运行的 ops 源码及测试副本/清单。这些可维护 ops 不混入科学 plan；最终115项 CPU/fake 联接证据及 actual source archive 见 `validation512_final_joint_cpu_verification.json` 与 `validation512_ops_integration_review.json`。

明确 `NOT_REQUIRED` 也必须由唯一 CPU writer `ensure_delivery` 生成完整适用/不适用报告，并经 `verify_delivery` 返回 `verified is True` 且 `delivery_complete is True`，才完成全512阶段交付并进入 observer_cost；该 CPU 交付本身不加载模型、不新增队列。若512原始收集/状态分析已经完成但交付缺失，状态保留 `RAW_COMPLETE`，固定 run 重入只补真实报告。交付目录为 `validation512/<plan_hash>/delivery/`，含九问报告、summary 与绑定实际文件 SHA/工作身份的 receipt。

2026-10-19 08:04 EDT 还只读提取并核验了**已排队的旧2655921 batch body**：它 `cd` 到本仓库，再调用共享的相对 `scripts/launch_state_study_resume.py`。因此它将执行当前共享 dispatcher；不依赖修改已排队脚本尾部。若届时 core 未完成，仍同 PID exec 原 auto 并预排一跳；若 core 已完成，则同一 helper PID 可转入已封存补充。不能在仍存活的旧 core 旁启动补充，也不声称原冻结 auto 会在同一 PID 返回后自行追加新功能。

## Observer 额外成本的独立工程阶段

原8.4要求 observer 额外时间与显存单列。该阶段采用预先固定的 C_diag IDs128–131，各题 A/B/C/D 完整保存轨迹；若原 P4 更新门控实际触发，再包含两种更新策略各自的 C_diag own-trace。同一固定 trace、driver/calibration、48层和全部 `T−1` 有效 decode，以真实 frozen driver/collector 做 observer on/off 配对。它是固定子集的工程边际 benchmark，不能写成全正式轨迹累计开销、原 run 当时实际成本、自由生成方法速度或拟合总成本。原8.4未指定成本测量N；这里没有缩短所选轨迹，也不由成本结果更改科学门控或解封 P6。

独立计划路径为 `reports/<run.name>/observer_cost/plan.json`，输出为其 `<plan_hash>/`；计划、实际采集与交付均须以独立 artifact 为证。本说明不证明 observer_cost 计划已经实际封存、模型已经运行或测量已经完成。原18+5与全512新5来源、full512 plan 及其 gate 保持不变。

只有 core 和全512真实工作/交付（或正式 verified NOT_REQUIRED 交付）均通过后，resolver 才允许 observer_cost `READY`/`RAW_COMPLETE`；该阶段独立 `COMPLETE` 必须同时 `delivery_complete is True`。部分 on/off 配对、未完成末行、缺计划、无效来源、未知 Slurm 状态或缺报告均不能当作整体完成。

独立元数据位于 `observer_cost/ops/`。launch schema 为 `state_observer_cost_launch.v1`，task/phase 为 `observer_cost`，绑定 `plan_hash`/`observer_cost_plan_hash`、原 core 与全512计划/启动身份、当前真实 allocation、最新有效前任与实际 ops 源码档案。claim schema 为 `state_observer_cost_handoff_claim.v1`，仍独占发布在共享 scratch。加载任何 GPU 模块前，成本模块调用纯CPU `registered_cost_launch(run,plan)`，重新核验固定 `sys.executable -u <abs>/state_study_observer_cost.py run --run-dir <RUN>`、同PID/boot/ticks/cgroup、当前资源、共享claim/history、实际源码副本，以及原core、实际full512与有效前任确已终止。NOT_REQUIRED分支可没有full512模型launch，但不能省略原A-D工程测量。

三个阶段共用同一 run/protocol/current-job 后继命名规则和每job唯一 durable intent/commit；intent 绑定实际 phase 与相应计划，不另建一条并行调度链。已 exec 的冻结 full512 run 不会返回 helper 以追加新功能；最小接续由它启动前预排的下一 allocation 调用共享 launcher 再进入 observer_cost。旧2655921 body依然支持这一调用路径，不依赖改动已排队body尾部。任何明确科学错误停止自动链；RPC UNKNOWN 和claim后崩溃仍保留证据、等待人工复核，不盲重投或删除claim。

这里的 `ALL_REQUIRED_PHASES_COMPLETE` 只指当前 core/full512/observer_cost 机器链及真实交付。P6/P7仍属于原用户已授权的条件性目标范围；当前不自动解封，表示仍须满足原科学门控和阶段条件，不是新增用户审批要求或永久排除后续工作。不能仅凭这条机器链结束就把所有原条件目标标为完成；后续是否适用与完成仍按原门控及实际证据判断。

## 正式选择后的质量判据自动冻结

`scripts/state_study_quality_watch.py` 是绑定一次实际 launch 的CPU监控，复用已锁定的 freezer；它不加载模型或提交Slurm作业。当前运行可从仓库目录执行：

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/jgu7/envs/moe-state/bin/python -u \
  scripts/state_study_quality_watch.py start \
  --run-dir /scratch/jgu7/moe-state-study/runs/20261019_bf16_eager_v4
```

先核验已有实际draft、自哈希、五份分析源码及归档、18份测量源码，再绑定主机/boot/PID/start ticks、完整命令行、当前launch及同一Slurm cgroup。本机`/raid/jgu7/state-study-quality-watch/`保存继承flock和PID元数据；不可覆盖事件、启动证据与日志在scratch报告的`quality_watch/`下。锁被持有但监管PID身份不能证明时明确报UNKNOWN，不重复启动。相对helper命令额外核验`/proc/PID/cwd`为本仓库。

P2未完成、freezer返回pending或观察遇到暂时I/O错误时，每30秒重查同一绑定。P2完成后在锁内调用已锁freezer，所有科学判据仍由它核验；已有final只验证，不覆盖或补写早期时间。验证数据已经出现而final尚缺失时拒绝迟到冻结。主进程实际退出、身份改变或明确科学错误时停止。缺少可持久化的发布intent则暂缓调用。监控异常有独立状态，不能冒称已完成冻结，也不杀当前科学进程。

合法后继launcher在更新新launch后、同PID exec原auto前接入此监控；精确helper到auto的过渡保持等待。当前监控随所属allocation结束，下一次合法交接为新launch重新绑定，不使用systemd常驻服务。最终34项fake CPU检查及独立审阅见`quality_watch_cpu_verification_v2.json`和`quality_watch_independent_review.json`。2026-10-19 07:00 EDT root实际启动当前launch的PID2659000并核验`WAITING_P2`；事实见`quality_watch_live_verification.json`，不是final已经冻结的证明。
