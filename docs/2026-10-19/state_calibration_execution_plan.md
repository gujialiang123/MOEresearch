MOEresearch：状态来源、聚合校准与自由生成的 Codex 实验计划
日期：2026-10-19
适用仓库：https://github.com/gujialiang123/MOEresearch
审阅基准提交：d43a836a35f5349ab33935ab6281ed66e0e5e539
用途：可直接作为 Codex 的执行任务输入。请按阶段实现、验证并运行实验，交付原始数据、分析和可续跑命令。本文规定科学对照，不预设 GPU 型号、卡数或显存预算。
0 任务目标和执行方式
本轮检验一个候选机制：
在原生 MoE 内部状态上估计的聚合校准，是否会在减少激活专家后的实际运行状态上重新失准？如果会，使用实际运行状态重新估计同一种校准器，能否改善匹配和完整生成行为？

这是假设，不是已经成立的结论。不要预设必须得到正结果，也不要围绕 ACL 投稿目标修改判据或选择有利样本。
执行要求：
1. 先阅读本计划、仓库现有说明和适用的 AGENTS.md，确认正在操作 MOEresearch。
2. 使用当前工作区的实际版本；审阅提交只用于定位背景，不要求回退已有更新。保留用户现有修改及历史结果。
3. 以原项目 Qwen3-30B-A3B-Instruct-2507、BF16 为主实验对象，保持原有模型与问题的连续性。
4. 先做模型接口和正确性预检，再执行 P0–P5。P6–P7 是通过明确证据门槛后的确认与论文扩展；不按某张卡的容量压缩科学设计。
5. 常规实现、修复、建立隔离环境、运行本计划内实验可直接推进，无需逐步询问。机器暂时没有可用运行环境时，完成可完成的实现和预检，交付具体启动命令。
6. 本计划不设默认墙钟截止时间。长任务分阶段保存，支持中断与续跑。用户后来给出的时间安排只影响调度，不改变比较口径。
7. 各方法使用相同 checkpoint、精度、数据、提示词和生成协议；不同进程独立维护 cache、policy 和随机状态。
8. 本轮不训练新 router、不做模型微调。先完整检验状态来源假设，再按结果扩展。
9. 以仓库中的 Markdown 报告、结构化结果和可续跑命令交付。不得覆盖旧 raw 数据或把不同精度、不同协议的结果拼成配对比较。
10. 遇到真实阻塞时，记录具体原因；“没有运行”与“运行得到负结果”必须区分。
首轮成功交付不要求假设成立。成功标准是：实现可信、比较可解释、数据可重算，并明确结论属于 POSITIVE_EXPLORATORY、NEGATIVE、INCONCLUSIVE 或 BLOCKED。
1 研究背景和本轮贡献边界
已经存在的近邻工作包括：
- Training-Free Halving：将计算专家数与归一化参考集合分开，并报告自由生成与截断行为。
- LDA：研究 reduced top-k 导致的尺度和方差失配，使用逐层逐维校准恢复表现。
- To Reason but Not to Stop：研究剪枝后固定参考历史与自由生成中的停止差异，并做激活干预。
因此，“renorm 会改变尺度”“输出可能变长”“校准能恢复”“要看完整生成”都不能直接作为本轮独有贡献。
本轮真正要验证的是：
- H1：已有校准在独立的原生状态上有效，但在少专家模型实际访问的状态上重新失准。
- H2：这种重新失准包含可区分的内部执行状态因素，而不只是换了一批文本、改变样本量或一般数值噪声。
- H3：仅改变校准状态来源，在保持 K、专家选择规则、校准函数形式和精度一致时，可以改善更新后模型自己访问的状态，并带来下游收益。
H1、H2、H3 可以分别成立或不成立。范数改善不能自动推出生成改善；输出变短也不能自动推出推理效率或质量改善。
2 先读哪些现有文件
优先复用以下文件，检查当前版本是否已修复相关问题：
- HANDOFF.md
- MOE_OPTIMIZATION.md
- docs/2026-07-20/v29_v32_experiment_report.md
- docs/2026-07-20/v29_v32_preflight_audit.md
- moe_research/k_policy.py
- moe_research/decode_norm_calib.py
- moe_research/gain_calibration.py
- moe_research/answer_parsing.py
- moe_research/stats.py
- moe_research/trace_schema.py
- scripts/_harness.py
- scripts/run_v29_decode_norm_control.py
- scripts/run_v31_gain_controls.py
- scripts/run_v32_pulse_recovery.py
- tests/test_native_equivalence.py
- tests/test_decode_norm_calibration.py
- tests/test_physical_expert_skip.py
- tests/test_intervention_window.py
- results/2026-07-20_v29_decode_norm_pilot/manifest.json
审阅时确认的旧运行环境是 torch 2.9.1+cu128、Transformers 4.57.1、H200、BF16。它是兼容性线索，不要求在新机器盲目复制整套依赖。
旧 scripts/_harness.py 硬编码了模型路径和整模型 BF16 上卡方式。本轮应增加明确的配置入口，不要照搬旧绝对路径。
已有结论只用作背景：
- v29 的 norm-match 校准与 realized-ratio 诊断主要使用原生模型的状态。
- 现有 no_renorm 是“先按原生 top-8 归一，再删除尾部专家”，不是完全取消归一化。
- 旧数据不足以证明 norm-match 相对 no_renorm 还存在独立的剩余长度机制。
- 旧 strict 与 tolerant accuracy 存在口径差异，不应选择最有利的一列宣布质量保持。
3 固定实验对象与可复现设置
3.1 主实验对象
- 模型：Qwen/Qwen3-30B-A3B-Instruct-2507，沿用项目实际 checkpoint 并记录 revision。
- 模型精度：BF16。统计累计使用 FP32/FP64。
- 原生激活专家数：K0=8；主要干预为 decode K=4。
- prefill：所有主实验臂保持原生策略。
- 生成：使用原 checkpoint 的 tokenizer、chat template 和 EOS 定义。
- 实验主线不自动切换成量化、小模型或更短生成长度。后续若专门研究精度敏感性，建立独立协议，所有对照在该精度内重跑。
NATIVE 指模型按其训练时的原生 top-k 规则运行，不表示激活所有专家。
3.2 记录实现条件
在 manifest 中记录实际的软件版本、模型路径、权重 revision、dtype、attention/expert backend、设备映射和硬件型号，用于复现与解释计时。硬件信息是实验记录，不是选择科学问题的条件。
必须检查：
- gate、experts、单专家线性层的实际类型与参数布局。
- MoE block 和 gate 的 forward 返回格式。
- cache、prefill/decode 阶段判定和模型的原生归一化语义。
- 当前依赖是否兼容仓库 hooks。
优先复用已有可验证环境。需要适配时在隔离环境中修改，锁定版本，保留原始模型路径作为 native identity 参考。不要无条件升级依赖，也不要把不同后端产生的数值差异当成研究效应。
如果当前环境尚不能执行 GPU 主实验，先完成 CPU toy、配置、脚本和分析代码；将真实模型实验标为 BLOCKED，并给出已验证到哪一步及准确启动条件。
4 正确性预检和必要修复
下面这些检查是机制实验的必要条件，不是可选美化。
4.1 模型接口检查
当前 Transformers 的 Qwen MoE 实现可能已使用三维专家参数，而旧仓库依赖逐专家 ModuleList；gate 和 block 的返回格式也可能变化。
- 以当前安装源码和运行时对象为准。
- 不假设设置 eager 就能恢复旧专家布局。
- 专家执行必须保留实际参数布局和数值语义；三维专家参数不能未经核验就按旧 ModuleList 路径处理。
- 遇到不支持布局时明确报错或完成适配，不允许 catch 异常后静默更换算法。
- 记录最终锁定版本及适配器选择。
4.2 原生等价性检查
比较同一模型、同一精度下：
- 完全未修改的原生 forward。
- 强制经过新执行实现的 K=native、单位 gain、单位校准。
必须覆盖短 prefill、携带 cache 的连续 decode、局部 MoE 输出、router 选择与权重、最终 logits 和短 greedy 输出。
先记录重复原生运行的数值噪声；保存最大误差、相对误差、固定前缀 KL 和 argmax agreement。FP32 toy 可以要求严格数值接近；真实 BF16 路径的容差在预检后固定，并与目标干预效应量比较。不得在发现明显偏差后不断放宽阈值。
仅靠 K=native 时直接绕回原始 forward 的短路，不能证明新专家执行实现正确。
还要检查：
- 取消 policy/observer 后能恢复原生行为。
- 不同 arm 不共享 cache、计数器、随机状态或上一个 arm 的校准参数。
- observer 开关不会改变实际返回给网络的结果。
- 对每个实际使用的 arm，在同一 h 上比较 shadow 中的 z4 与该 arm 真实 forward 的 MoE 输出，核对校准位置、cast 与聚合顺序。K8 identity 和 observer neutrality 不能代替这项检查。
- teacher-forced replay 的 A 和 C 能在固定文本上复现各自无 observer 生成时的 logits/状态到已声明的容差。随机数状态和所有原地修改也必须受控。
4.3 确认不该干预的位置保持原生
prefill、未选中的 layer/step、声明为 native identity 的配置，必须保留原生计算和单位校准。
用真实非平凡参数测试：非单位 scalar、非零 bias、真实 shuffled provider。不要用空字典或 provider=None 代替实际覆盖。
审阅发现的旧问题：
- v31 shuffled provider 忽略 phase，可能在 prefill K8 时也随机放大。
- v32 open-loop 的零起点与 closed-loop 的一起点存在错位。
- 现有整段 teacher-force 校准返回 native branch，不能用来代表 student 自身状态。
- hit_max 用 generated_tokens>=cap 判断，会把恰好在最后位置 EOS 的样本混入触顶；本轮应明确 stop_reason。
检查当前版本后修复仍存在的问题。脉冲实验不是本轮必跑项；如果本轮不运行，也要记录其旧结果的使用边界。
4.4 reference-k 与实际专家计算分离
实现 k_compute 和 k_reference 两个独立参数。
以 Qwen 为例，k_compute=4、k_reference=16 时，只能执行 top-4 专家 FFN。计算 top-16 路由概率之和不等于执行 16 个专家。
使用 toy hooks 或专家调用计数确认物理跳过；不能仅把 8 个专家全部算完后将 4 个权重设为零，然后声称计算减半。
诊断中的 shadow top-8 计算属于额外分析开销，不能混进无探针的性能结果。
5 模型策略和第一轮六个强基线
以下精确定义适用于原 Qwen 的 norm_topk_prob=true。不同原生语义的模型需要独立定义和报告。
令 p_i(h) 是所有专家的 router softmax 概率，m_j(h) 是 top-j 概率之和。
原生 K0=8：
y8(h) = sum_{i in top8} p_i(h)/m8(h) * E_i(h)。
保留原权重的 K4：
u4(h) = sum_{i in top4} p_i(h)/m8(h) * E_i(h)。
本轮所有臂 prefill 固定原生 K8，只在 decode 改策略。
ID	Decode 策略	定义或作用
N	Native K8	当前精度的原生参考
F	K4 full renorm	u4 * m8/m4
R8	K4 native reference	u4，等价仓库 relative no_renorm
R16	K4 reference 16	u4 * m8/m16
M_N	Native-state scalar	a_l^N * u4
L_N	Native-state LDA	对 K4 full-renorm 输出施加逐层逐维 affine 校准


F、R8、R16 已组成 reference-k 的最小集合 {4,8,16}。首轮不增加 K6、k_reference=12、额外 beta 网格或更多裁剪比例。
上表是数学定义。BF16 中的具体运算顺序必须固定：F/R8/R16 由各自最终 router 权重调用统一的专家聚合 primitive，softmax、归一化、cast 与专家累加顺序对齐已验证的实现。不能默认“先算 BF16 u4 再乘 m8/m4”与“先归一权重再聚合”数值等价。Scalar/LDA 明确采用聚合后校准：获得原生模型 dtype 的 reduced branch，转 FP32 应用校准，再 cast 回模型 dtype。校准统计、shadow 和真实 driver 遵循同一约定；实际浮点顺序写入 protocol。
标量校准沿用原项目的目标：
a_l^N = sum ||y8(h)|| / sum ||u4(h)||。
LDA 按论文的均值与标准差对齐形式实现：
LDA_l(y4) = (y4 - mu4_l) * sigma8_l / (sigma4_l + eps) + mu8_l。
这里乘除为逐维操作，y4 是明确定义的 reduced-routing branch。
- 不得把一个 scalar 重命名为 LDA。
- 记录 bias、低方差维度处理、epsilon 与所有 clamp。
- 初始以数值保护为目的固定 epsilon，不根据验证结果调参。
- 校准统计使用 FP32/FP64 在线累计。
- 依据原文 Algorithm 1/2 实现；本次审阅未确认官方代码仓库，不假定作者代码已经可用。这里是“LDA 在本研究 decode-only 协议下的受控复现”，必须记录校准文本、位置和应用阶段与原论文的差异。原文逐层校准后由 native K0 传播到下一层，不能把各 K 独立传播的版本标为相同基线。
- 逐维均值/标准差在保存的全部有效拟合 token 上统一估计，可用可合并的在线 moments 实现；不能每个 decode step 只有一个 token 就单独求 std。方差采用总体矩口径（ddof=0）并记录，不直接平均各 prompt 的标准差。若原文或后续官方实现要求其他估计口径，先对照说明并为所有 LDA 来源臂一致更新。
- 公式的 sigma+eps、方差的数值保护以及诊断分母的 eps 分别命名和记录，避免把不同单位的保护常数混用。真实的额外 clamp 作为实现差异报告。
- 本轮不是复现论文表格的绝对分数。声称超过该工作前，必须确认没有弱化公式、数据或参考集选择。
6 校准与评估数据
6.1 固定集合
首先审计已有结果的 split、sample ID 和 question hash。以下区间是默认候选；如数据版本改变或发现冲突，在首次新实验前一次性确定替代集合并写入 protocol.json。
集合	默认 GSM8K train 区间	正式样本数	用途
C_fit	[0,128)	128	拟合校准器，产生校准 rollout
C_diag	[128,256)	128	检查留出泛化、状态失配与更新后自身状态；不拟合参数
D_select	[256,512)	256	比较强基线、选择进入复核的配置
D_validate	[512,1024)	512	冻结方案后的内部复核
D_confirm	依据完整历史记录，从此前未参与本研究选择的样本中封存	预先确定	独立确认，不参与调参


C_diag、D_select、D_validate 的部分样本可能已出现在历史实验中。因此只能称“对本次拟合留出”或“内部复核”，不能称历史从未看过的新确认集。
旧 GSM8K test 已做过多轮实验，可以在冻结后重跑并提供完整 benchmark 分数，但不能重新包装成此前从未使用的确认集。D_confirm 必须审计使用记录并预先封存 question hash；独立性无法确认时如实标注，并为新任务设置未参与选择的确认集。
所有新增校准参数都只由 C_fit 产生。禁止从 C_diag、dev/test 的 gold、结果长度或自身失配直接拟合参数。测试时 shadow 计算仅用于离线诊断，不参与部署策略。
6.2 先验证管线，再使用固定正式样本
- 正确性 smoke：4 个 prompt，最多 128 个新 token。
- 管线 pilot：C_fit 前 16 个、C_diag 前 16 个、D_select 前 32 个 prompt，验证完整执行和数据产物。
- 正式核心实验：使用表中固定的 128/128/256/512 样本数。
- pilot 不作为有无机制的统计结论，也不根据其显著性决定是否偷偷改变正式样本量。
- 校准从 16 扩到 128 后参数会改变，必须建立新 calibration hash；使用旧参数生成的 pilot 不能并入正式主结果。
- 全部 arm 使用相同 prompt 集合和顺序。缺失配对补齐后再计算正式差值。
主实验默认 max_new_tokens=2048。128 或 256 的短输出只用于工程预检，不用于宣称恢复自然停止。
预先规定：若主比较任一臂在 D_select 的行政截断比例超过 1%，追加所有入围臂在固定的 D_select 前 128 个 prompt 上的 4096-token 敏感性实验。原 2048 结果仍完整保留，较长上限结果单独报告；不得只给一个方法的失败样本延长上限。1% 是本计划的工程触发值，不是领域通用标准。
如果更长上限仍有截断，报告被删失的事实及可辨认的指标，不外推未知的真正长度。
6.3 校准位置与权重
默认使用 C_fit 自由生成 trace 的全部有效 decode 位置，上限为主协议的 cap；prefill 不纳入 decode 校准。
- native-state 基线使用 Native 在 C_fit 上生成并按 native driver 回放的状态。
- M_NT 和 M_ST 使用完全相同的 S0 trace、相同有效位置清单和相同 token 数；位置清单在拟合前生成并保存。
- 所有统计默认按有效 token 加权，保存每 prompt 的计数。长输出权重更高是显式设计，不隐藏这一点。
- Native 初始基线与 student-text 重估的总 token 数可能不同，必须报告；对“状态来源”的主要归因依赖 token 数严格匹配的 M_ST vs M_NT。
- 若为统一拟合成本而抽样，先固定位置规则、seed、每 prompt 配额，再执行两条配对路径；不得分别按各路径的失配大小抽样。
- 正式诊断保留按 prompt 等权的敏感性分析，避免结果完全由少数长输出支配。
- 以少量前缀校准的 pilot，不能冒充完整轨迹校准的正式结果。
6.4 生成协议
- 默认 greedy，do_sample=false，batch=1；后续批处理必须对所有相关方法一致，并通过对应 native/replay 等价性检查。
- 使用 checkpoint 对应 chat template，保存完整渲染 prompt。
- 保持现有“step by step + #### final answer”提示词，首次预检后冻结。
- 从实际 generation_config 和 tokenizer 解析完整 EOS 集合；不要只猜一个 token ID。
- 原 Qwen Instruct-2507 是 non-thinking checkpoint，不要硬编码 </think> 作为停止目标。
- 冻结 max_new_tokens、输入截断策略、attention backend、dtype、随机种子和 batch。
- 变更协议后建立新 run_id；只复用 protocol hash 完全相同的结果。
7 核心诊断：把文字和内部状态分开
7.1 定义被诊断的 Student
S0 必须是实际待验证的部署策略。
- 诊断 M_N 时，S0=K4+native-state scalar。
- 诊断 L_N 时，S0=K4+native-state LDA。
- full-renorm 可以作为现象参考，但不能用 F 的坏状态冒充已经校准的 S0 状态。
优先诊断最强、实现完整的校准基线。若 LDA 已经恢复了主要行为，只证明旧 scalar 更弱不能成为新增方法的强动机。
在每个 C_diag prompt 上，无 observer 地生成并保存：
T_N = 原生模型的输出 token IDs；
T_S = S0 的输出 token IDs。
7.2 四条回放路径全部必做
形成“执行策略 × 文本来源”的 2×2 设计：
路径	实际内部执行策略	强制输入的生成文字	用途
A = N on T_N	Native	Native 文字	留出 native 状态的校准有效性
B = S0 on T_N	Student	与 A 完全相同	Native 文本下的内部状态对照
C = S0 on T_S	Student	Student 自己的文字	真实部署状态
D = N on T_S	Native	与 C 完全相同	Student 文本下的内部状态对照


核心受控比较为 C−D：在 student 实际生成的同一段文字上，只改变内部执行策略，校准失配是否增加？B−A 是 native 文本下的支持性比较。
D 从初轮正式诊断开始就做，不根据结果好坏再决定补不补。它只增加一次回放，不增加一个自由生成方法。
7.3 必须逐 token 使用 cache 回放
不能直接 model(full_ids) 然后声称获得 B 或 C。旧仓库会把无 cache 的整段调用判为 prefill，可能导致实际一直用 K8。
正确逻辑：
每条路径单独初始化 cache
用 Native 策略处理 prompt，得到第一个输出 token 的预测和 cache
对 d = 1 ... len(trace)-1：
    输入 trace[d-1]
    按该路径的 driver policy 进行一次 decode
    观察本次各层的实际输入状态
    强制按 trace 继续，而不是使用本次 argmax 决定下一个输入
统一索引约定：
- 第一个生成 token 来自 prefill。
- decode_forward_index=1 消费第一个已生成 token，预测第二个生成 token。
- 生成 T 个 token 通常对应 T-1 次 decode forward。
- 已输出的终止 EOS 不再作为后续模型输入。
- B 即使本会提前预测 EOS，也继续输入固定的 native trace；它是受控回放，不是自由生成完成率实验。
将此约定写入 trace schema，并用可人工检查的短 toy trace 覆盖。
7.4 Observer 不能改变 driver
在路径 q 实际经过的同一个 MoE 输入 h 上，形成：
- y8(h)：同一状态下原生 top8 的输出。
- u4(h)：同一状态下保留 native 权重的 top4 输出。
- y4_full(h)：同一状态下 full-renorm top4 输出。
- z4(h)：当前固定校准器作用后的实际 reduced branch。
同状态意味着 h 完全相同；不能拿 native 轨迹的 y8(h_N) 与 student 轨迹的 u4(h_S) 直接比较，并称为同状态匹配。
探针可以额外计算 shadow experts，但实际反馈到网络的结果必须保持 driver：
- A、D 返回 native 输出。
- B、C 返回 S0 的真实输出。
优先保存并返回原本 driver 已计算的输出，shadow 计算只读，不复用可被覆盖的返回缓冲。不能因为测了 y8，就让 student 继续沿 y8 传播。
尤其不要直接复用旧 DecodeNormCalibrator 的返回 native branch 行为来收集 B/C。
8 记录什么以及怎样判断失准
8.1 每个状态的最小指标
至少记录：
- prompt ID、路径、层号、decode index、token ID、位置 bin。
- ||y8||、||u4||、||z4||。
- 真实 MoE residual addition 输入 r 的范数。
- ||z4-y8|| / (||r||+eps)。
- cos(u4,y8)。
- 误差沿 y8 的投影与正交分量。
这里 r 是实际 residual addition 的输入；不能把经过 RMSNorm 的 MoE 输入 h 当成 residual。
正标量能改变幅度，不能旋转向量。若幅度已匹配但方向误差仍显著，不要把所有问题继续归结为尺度。
实际专家集合可以随路径变化。“固定选择规则”不等于不同生成轨迹在每一步都选中同一批专家。
8.2 标量校准的主要诊断量
对每层和路径：
R_l(q; theta) = sum ||z4_theta(h_q)|| / sum ||y8(h_q)||。
z4_theta 是当前被冻结校准参数作用后的真实 reduced branch。Scalar 时它是 a_l*u4；LDA 时它是实际逐维 affine 输出。A/B/C/D 均使用同一套初始参数 theta0 做观察。更新后的策略分别使用 theta_NT 与 theta_ST，并记录 calibration hash，不能运行新模型却继续用旧标量诊断。
定义：
E_norm(q; theta) = mean_l abs(log R_l(q; theta))。下面省略 theta 时，表示同一次比较内已经固定的校准器。
主要受控比较：
D_state_student_text = E_norm(C) - E_norm(D)。
支持性比较：
D_state_native_text = E_norm(B) - E_norm(A)。
C/D 使用完全相同的 student 文本与有效位置；A/B 使用完全相同的 native 文本与有效位置。对各比较保存相同的 mask hash。先报告留出 A 的匹配情况，再解释 C/D 的差异。
正式诊断覆盖完整有效 decode 轨迹，并分预设位置段报告：
1–32、33–128、129–256、257–512、513–1024、1025–2047。
- 每段报告涉及 prompt 数、有效位置数、逐层有符号 log ratio 与置信区间。
- 长后缀样本少时如实反映区间，不用早段匹配代替后段结论。
- C/D 保留 student 的完整长后缀，不要求它必须短于 native trace。
- 四路径直接汇总比较时，可额外使用每个 prompt 的共同位置范围；它是补充分析，报告覆盖率及被排除后缀。
- 将 B−A 和 C−D 比较为“交互效应”时，先在四路径的共同位置与相同权重下重算，不能直接相减两个覆盖不同长度的估计。
- 共同存活位置上的结果是条件性分析，不能替代完整生成结果。
原生来源与实际来源还可能改变位置和 prompt 权重。因此同时保留按 token 汇总的主分析，以及按 prompt 等权的敏感性分析。
8.3 LDA 必须测自己的目标
LDA 的核心目标是逐维均值与标准差匹配，不能只用一个总 norm ratio 判定其成败。
除 E_norm 外，计算每层：
E_moment_l = sqrt(
    (||mu_z-mu_y8||^2 + ||sigma_z-sigma_y8||^2)
    / (||mu_y8||^2 + ||sigma_y8||^2 + eps)
)。
再对层平均，并报告均值误差与标准差误差两个分量。这里两种 branch 都来自同一个路径的同一批 h。
选择 LDA 作为 S0 时，以 E_moment 的 C−D 为主要状态诊断；E_norm 作为补充。不要因某个指标更有利而临时更换主要目标。
LDA 在 native 状态上成立，不意味着新的实际状态上必然不成立；也不意味着其余向量误差必然能靠矩匹配修复。
8.4 可重算的数据保存
- 不在 GPU 保存全部 prompt×token×layer×hidden 的激活。
- 在线累计统计，逐 prompt 写入 CPU/磁盘。
- 保存每 prompt、每层、每位置 bin 的计数与求和项，保证能按 prompt 重采样。
- LDA 需要逐维 sum/sumsq 时使用压缩数组文件，明确 dtype、count 和来源。
- 少量完整调试样本可以保存，但默认不超过 4 个 prompt。
- 保存 rollout token IDs、解析输出、策略与校准参数 hash。
- observer 的额外时间与显存单列，不计入方法正常生成的速度。
9 只在有失准证据后进行一次重新校准
9.1 首选一个校准器家族
优先对 LDA 做同形比较；如果当前只能可靠实现 scalar，就先完成 scalar 管线，但把“相对 LDA 的新增价值”标为待检验。
不要同时新增很多校准目标、参数容量或正则化技巧。
9.2 两个关键新增臂
这里 M_NT/M_ST 中的 M 泛指本阶段选定的同一个校准器家族；若选 LDA，两者都必须使用逐维 affine LDA。manifest 单独记录 family，不能把 scalar 与 LDA 混在这一配对中。
固定初始 S0，在 C_fit 上生成 T_S0。所有新增拟合只使用 C_fit。
对同一份 T_S0、相同 prompt 和位置 mask，估计：
- M_NT：由 Native 处理 T_S0 产生的状态拟合校准器。
- M_ST：由 S0 处理 T_S0 产生的实际状态拟合同一种校准器。
两者的公式、参数数量、原始 branch 定义、token 数和数值保护必须一致。目标均为每个实际 h 上的 shadow y8(h)，不能改成另一条 native 轨迹上的输出。
M_NT 控制“只是换了 student 生成文字”的收益；M_ST 检验内部状态来源的额外作用。
与原生文本上拟合的 M_N/L_N 一起报告，可以区分：
- 更换文本带来的效果。
- 在同一文本下更换内部执行状态带来的效果。
- 是否超过现有强基线。
第一轮六个基线，加这两个新增臂，最多八个方法配置。只有入围方法进入更大的 D_validate，不要求所有八臂都扩样。
9.3 公式与成本控制
Scalar：
a_l^ST = sum ||y8(h_ST)|| / sum ||u4(h_ST)||。
LDA：
保持原来的逐维 affine 公式，只更换统计来源。不要顺便换成低秩映射、向量回归、停止分类器或不同 loss。
M_NT 与 M_ST 的拟合使用相同 token quota、相同位置抽样和 prompt 集合。原始 M_N/L_N 的 token 数单列，不把文本长度变化隐藏为“完全等预算”。分别报告：
- 拟合使用的 token 数。
- native/student rollout token 数。
- shadow expert 计算次数。
- 总离线校准时间。
- 最终推理时的额外操作。
相同拟合样本量不等于总校准成本相同。M_NT 与 M_ST 可共享 T_S0，但相对原始 LDA 多生成 rollout 的开销必须报告；若声称同总预算优势，应给原生校准一个相应的等预算扩样对照。
9.4 两个更新版本都必须重新运行自身轨迹
冻结 M_NT 与 M_ST，得到两个更新模型 S_NT 和 S_ST。它们分别在 C_diag / D_validate 上重新自由生成，并测量各自实际访问的状态。
- 必须同时测 S_NT on T_NT 与 S_ST on T_ST。
- 只测 M_ST 的新轨迹，无法排除两者共有的“换了校准文本”收益。
- 只展示新参数在旧 S0 状态上匹配得好，也不足以说明部署后有效。
- 为区分参数改变与轨迹改变，可在已经保存的同一批 h 上只读计算两个校准器的输出，作为固定状态的补充比较；不能改变 driver 的返回。
- 报告初始 S0、更新 S_NT、更新 S_ST 各自的部署失配、生成质量、结束行为与请求成本。
- 所有失配使用该策略的真实 z4 和自己的 calibration hash；固定状态交叉观察时同时标注 driver hash 与被观察校准器 hash。
首轮只做一次重估。若更新后再次失配，记录结果；第二轮迭代属于预先声明的新扩展，不自动无限迭代至出现有利结果。
10 下游指标和评分边界
每个自由生成样本至少保留：
- 原始输出文字和 token IDs。
- generated_tokens、prompt_tokens。
- stop_reason：eos、max_new_tokens、timeout、oom 或 error。
- eos_position、是否出现 ####、解析结果。
- strict_correct、tolerant_correct，以及两者是否冲突。
- 重复 4-gram 比例、第一次明确答案标记位置、标记后长度。
- 无 observer 的总生成时间、prefill 时间和可测的 decode 时间。
- 峰值 GPU allocated/reserved memory。
- 实际专家执行计数或明确标注的简化预算。
必须报告：
1. Strict 和 tolerant accuracy 并列。
2. 平均、median、p90 生成长度。
3. 自然 EOS 比例与真正的行政截断比例。
4. 重复指标与答案格式遵循。
5. 每请求完整生成成本。
注意：
- 默认继续使用已冻结的解析器便于探索，但导出最多 20 个有代表性的解析分歧样本供检查。
- tolerant 的“全文最后一个数字”兜底不能直接等同真实答题能力。
- 若两种评分方向相反，将质量结论标为不确定，不挑一列宣布胜出。
- 若修正 parser bug，给 parser 新版本，对所有 arm 一致重算，不据此单独调整某个方法。
- 输出短了但答案更差、直接省略作答或提前终止，不能算成功。
- 首个 EOS 恰好位于 cap 的最后位置仍属于自然结束。
- RMST 到 cap 的值不能恢复 cap 之后未知的真实长度。
- 某个 arm 出现 OOM/错误/超时不能静默丢样本；记录完整状态，并明确哪些配对尚未完成。
性能测量必须关闭 observer，使用同一 backend、同一 batch、同一精度；开始/结束计时处同步 CUDA。一次短热身不计入结果。
在 Python 研究后端上测得的速度只代表该后端。没有部署系统实测时，不声称可直接转成 vLLM/SGLang speedup。
不要把 K×生成长度称为实测专家执行总量；首 token 由 prefill 产生，padding、EOS 后计算、共享专家和非专家计算也可能影响实际成本。
11 统计方法和停止判据
11.1 统计单位
独立单位是 prompt，不是 layer-token。
- 使用配对 prompt bootstrap，默认 10,000 次，固定 seed=20261019。
- 每次抽取一个 prompt 时一起带入它的全部路径、层和位置，重新计算 ratio、E_norm/E_moment 和差值。
- A/B 与 C/D 分别在完全相同 mask 上比较。
- 对生成长度、正确性、自然结束率报告配对差值及 95% CI。
- 如报告准确率 p 值，使用合适的配对检验，并对预先声明的一组主要比较做 Holm 校正。
- 不能用“不显著”证明两组等价。
- 不对每层、每位置都做显著性筛选后只展示有利位置。
校准参数固定时的 prompt bootstrap 主要反映评估样本不确定性，不包含重新拟合参数的全部不确定性。首轮可不做昂贵的嵌套 bootstrap，但必须说明这一点。
11.2 预先声明主要比较
主要比较尽量少：
1. Student 固定文本下的状态差：E(C)-E(D)，并报告 E(B)-E(A) 作为支持性对照。
2. M_ST 与 M_NT 更新后，各自实际轨迹上的失配差。
3. M_ST 与 M_NT 的下游配对差异。
4. M_ST 与事先选定最强既有基线的下游差异。
在 D_select 之后、D_validate 之前冻结最终对照和选择规则；不得一边看 D_validate 一边换最佳基线。
可以在运行前声明“约 5% norm ratio 偏差”作为一个工程上可解释的参考带，并同时报告完整连续数值和区间。这个参考带不是领域共识，也不是因果阈值。不能因为结果略低于它就声称效应不存在。
对优先使用的 LDA，E_moment 有自己的单位和尺度，不能套用 norm ratio 的 5%。正式 C_diag 诊断前，在 protocol.json 中冻结 chosen_mismatch_metric、native_match_tolerance 与 state_effect_tolerance，说明它们与实现噪声及工程意义的关系。未能合理定义实用界限时，仍可完成连续效应与区间分析，但不使用“基本匹配”“已等价”或 NEGATIVE 结论；标为 INCONCLUSIVE。判负需要预先指定等价/实用界限及足以落入该界限的区间。
11.3 按观察决定下一步
观察	判断和行动
Native identity 或 observer neutrality 不通过	FAILED_IMPLEMENTATION；先修实现，不跑正式矩阵
留出原生状态 A 上就明显失配	检查拟合、位置覆盖、留出泛化与实现；尚不能称为 student-state 再失准
C 比 D 更失配，且 C 在自身轨迹确实失配	支持内部执行状态因素，进入同文本的重新校准实验
B−A 与 C−D 方向一致	对不同文本来源下的内部状态作用提供支持
C−D 接近，C 与 D 都明显失配	更符合文本来源或校准覆盖改变；不能强称内部执行状态来源是主因
C−D 与 B−A 不一致	报告文本与状态的交互或位置覆盖差异，做共同 mask 分析，不挑一个对比下结论
完整轨迹上 C、D 都匹配，且 CI 足够窄	对该设定下的实用失准假设给负结果，停止以此为主要扩展动机
正式样本下 CI 仍宽	INCONCLUSIVE；根据效应与配对方差制定新的固定样本计划，不反复看显著性追加
LDA 或 reference 方法已恢复主要行为	报告已有方法有效；只胜过较弱 scalar 不构成充分新增价值
新参数只在旧 S0 状态有效，在新自身轨迹又失准	一次重估未稳定恢复，报告结果
匹配改善但生成未改善	该匹配量不足以说明下游收益，不声称它完全解释生成机制
长度减少但答案质量恶化	不能认定改进；检查提前停止和格式影响


11.4 正结果必须同时满足的证据链
进入确认与论文扩展前，至少同时满足：
1. 校准器在留出 native 状态上基本有效；在 student 同文本 C/D 比较中出现可复现的内部状态来源差异。
2. M_ST 相对同文本、同公式、同拟合 token 数的 M_NT，在更新后的实际运行中改善匹配。
3. M_ST 相对 M_NT 和事先选定的强既有基线，改善预先指定的完成行为或请求成本，并满足明确的质量约束。
在 D_select 后、D_validate 前，把主要下游目标与“实际有用”的最小效应写入冻结文件。例如优先考察完整输出 token 成本和自然结束行为；真实时延作为同后端实测指标。阈值由本任务的使用意义与探索数据的方差确定，不能在 D_validate 后倒推。
方向性正结果标为 POSITIVE_EXPLORATORY。若要声称“准确率下降不超过 2 个百分点”，必须在解封确认集前设置非劣性界限并据配对不一致率确定样本量。这个 2pp 只是可选例子，不是本轮已认可的无损标准；准确率差异不显著不能推出质量保持。
上述证据可以支持“校准状态来源具有因果影响”。它仍不足以证明“测到的范数/矩失配完全中介了长度变化”。
12 分阶段执行与验收
阶段	要回答的问题	工作与最小完成物
P0	新路径是否忠实于原模型？	接口适配、native identity、observer neutrality；preflight.json
P1	全链路能否正确产生可分析数据？	4-prompt smoke 与 16/16/32 管线 pilot；索引、phase、stop reason、评分检查
P2	现有强方法是否已解决问题？	正式 C_fit=128 校准，六臂 D_select=256 配对生成；基线表
P3	失准来自文本还是内部执行状态？	C_diag=128 上完整 A/B/C/D 回放；分层、分位置、配对诊断
P4	改变状态来源能否修复部署表现？	有失准依据时拟合 M_NT/M_ST；两者都重新生成自身轨迹；新结果与成本
P5	冻结后的结论能否在留出样本复核？	入围臂 D_validate=512；冻结对比、区间、报告和后续判定
P6	主结论能否通过独立确认？	解封前固定样本量、指标和五个关键臂；一次确认，不能边跑边改
P7	结论是否跨设置成立？	仅在主线通过后，执行第16节的最小跨模型/任务与校准重复


执行顺序与状态：
- P2/P3 可以复用已经冻结的 trace；有依赖的拟合、生成与分析保持顺序。
- 独立方法可在可用设备上并行，确保 cache、随机状态、目录、policy 完全隔离；同一对比固定数值与批处理条件。
- 按共同 prompt 小块完成待比较 arm，持续保存，便于中断后补齐配对。
- 不改变 cap 来隐藏无法自然结束的样本。
- 支持 resume，按 run_id、protocol hash、model/precision/calibration hash 和 sample_id 核验已有行。
- 对 NaN、运行错误、超时、重复 ID、缺失配对给明确状态。
- 每完成一个阶段写入 STATUS.md，包含已完成、未完成、累计样本、失败原因与精确续跑命令。
- 只有实现不通过、存在真实阻塞或明确的科学停止条件时停下。不得把硬件型号作为降低证据标准的理由。
- P4 的科学前提不成立时，完成 P2/P3 报告及必要的 P5 负结果复核，不为得到一套新方法结果而强行继续。
交付阶段性结果时，必须明确实际完成到哪一阶段。P0/P1 完成不等于机制实验完成；P2–P5 完成也不自动等于论文结论已经通过确认。
13 建议代码与输出结构
优先在现有模块上做小而清楚的扩展，避免复制多份 monkeypatch。
建议新增或等价组织：
- configs/state_calibration_study.yaml
- scripts/run_state_calibration_study.py
- moe_research/state_collection.py
- moe_research/state_calibration.py
- 必要的模型适配器，按实际模块结构分派
- 针对新增风险的少量测试
CLI 可以提供以下子命令：
- preflight
- smoke
- baselines
- diagnose
- recalibrate
- validate
- report
- auto
这些名称是待实现的接口要求，不是声称仓库已存在这些命令。完成后验证 --help 和一个实际 smoke 命令，再记录真正可用的启动命令。
建议结果目录：
results/<study_id>/<run_id>/
    protocol.json
    manifest.json
    preflight.json
    calibration/
    generations/<arm>.jsonl
    diagnostics/<path>/
    summaries/
    figures/
    STATUS.md
    RUN_REPORT.md
    commands.sh
manifest 至少含：
- 仓库 commit、未提交补丁摘要或 diff 文件。
- 模型和 tokenizer revision、权重来源与全部配置。
- 原生 K、compute K、reference K、校准器公式和参数 hash。
- prefill/decode 策略、observer 开关、phase 与 index 约定。
- 软件/硬件版本、backend、batch、dtype、device map。
- 数据集 revision、split、ID、question hash、模板和 parser 版本。
- 校准与生成 token 上限、实际统计 token 数、位置 mask。
- seed、样本计划、计时与显存测量方法。
- 原始任务启动时间、阶段状态、失败原因。
14 默认配置草案
配置字段可按现有工程调整，但语义要保留：
study_id: state_calibration_study
execution:
  main_stages: [P0, P1, P2, P3, P4, P5]
  gated_followup_stages: [P6, P7]
  max_method_updates_core: 1
  max_method_arms_core: 8
  resume: true

runtime:
  model_id: Qwen/Qwen3-30B-A3B-Instruct-2507
  model_path: configurable
  model_revision: pin_before_run
  precision: bfloat16
  batch_size: 1
  do_sample: false
  seed: 20261019
  preserve_native_reference_path: true

generation:
  main_max_new_tokens: 2048
  smoke_max_new_tokens: 128
  followup_max_new_tokens: 4096
  followup_trigger_truncation_rate: 0.01
  followup_select_prompts: 128
  use_model_generation_eos: true

data:
  dataset: gsm8k
  subset: main
  calib_fit_train_range: [0, 128]
  calib_diag_train_range: [128, 256]
  select_train_range: [256, 512]
  validate_train_range: [512, 1024]
  pilot_fit_prompts: 16
  pilot_diag_prompts: 16
  pilot_select_prompts: 32
  audit_historical_sample_usage: true
  seal_confirmation_ids_before_selection: true

calibration:
  fit_split: C_fit
  phase: decode
  positions: all_valid_decode
  weighting: token
  accumulation_dtype: float64
  matched_student_text_positions_for_NT_ST: true
  reuse_pilot_generations_after_refit: false

policy:
  prefill: native
  decode_k: 4
  reference_k_candidates: [4, 8, 16]
  core_baseline_arms: [N, F, R8, R16, M_N, L_N]
  new_source_arms: [M_NT, M_ST]

diagnostics:
  paths: [A_native_native_text, B_student_native_text, C_student_student_text, D_native_student_text]
  primary_state_contrast: C_minus_D
  supporting_state_contrast: B_minus_A
  coverage: full_valid_decode
  position_bins: [[1, 32], [33, 128], [129, 256], [257, 512], [513, 1024], [1025, 2047]]
  preserve_driver_output: true
  evaluate_both_updated_models_on_own_traces: true
  save_all_hidden_states: false
  prompt_equal_weight_sensitivity: true

statistics:
  bootstrap_unit: prompt
  bootstrap_resamples: 10000
  paired: true
  report_95_ci: true
  no_equivalence_claim_from_nonsignificance: true
  freeze_confirmatory_endpoints_before_unsealing: true
15 最终向用户汇报什么
RUN_REPORT.md 必须用直白中文回答：
1. 实际跑的是哪一份模型、什么精度、什么实现版本？是否保持了原项目的科学设定？
2. 修改过哪些关键实现？哪些正确性检查通过，哪些仍未通过？
3. 六个已有方法的结果是什么？最强已有方法是否已经解决主要问题？
4. 原生校准到了 student 状态上，到底有没有再失准？C−D 和 B−A 各自说明什么？长后缀是否覆盖？
5. 两个更新版本 M_NT/M_ST 都重新运行后，M_ST 是否仍有额外收益？
6. 相同文字的 native-state 对照、总校准预算和强基线，是否能解释收益？
7. 答案质量、生成长度、自然结束和实际成本分别如何变化？
8. 当前结论是 POSITIVE_EXPLORATORY、NEGATIVE、INCONCLUSIVE 还是 BLOCKED？
9. 下一步最值得做的一件事是什么？给出精确命令和依据。
报告中明确分开已运行结果、统计推断和后续假设。不要写“完整证明因果链”“精度无损”“已达到 ACL 新颖性”等超过证据的结论。
建议图只做最有解释力的三类：
- Native-state 与 student-state 的逐层失配及区间。
- 更新前后，在各自实际轨迹上的失配与生成行为。
- 入围方法的质量与完整生成成本比较。
原始数据必须能重新生成所有表格。交付可运行的 commands.sh 和 STATUS.md，确保用户换一个 Codex 会话也能继续。
16 通过主线后，怎样扩展成论文证据
16.1 P6：一次封存确认
前提是第11.4节的证据链在内部复核中得到支持。解封前固定比较和样本量，确认阶段保留五个关键臂：
1. Native。
2. 在 D_select 上选定的最佳 reference 策略。
3. 选定的初始 native-state 校准器。
4. M_NT。
5. M_ST。
第一轮 {4,8,16} 只是最小 reference 网格。若要声称优于 reference 方法，应在确认前审查该网格是否弱化近邻工作；需要扩展时在选择集上预先固定适当范围，完成选择后再冻结，不能用确认集挑 k_reference。
确认样本量由最小有用效应与探索集的配对方差决定。256 题可用作方向性确认的起点，但不是证明 2pp 非劣性的充分样本保证。若计算所需样本超过可用独立数据，说明可支持的结论范围。
16.2 P7：三个实验条件构成最小扩展
实验条件	要验证的外推
原 Qwen × GSM8K	完整主机制与方法结果
第二个 MoE checkpoint × GSM8K	是否依赖单一 checkpoint 或单一架构
原 Qwen × 另一项有独立正确性评测的生成任务	是否只属于数学答案格式和 GSM8K 长度


第二模型优先采用不同架构，依据官方配置定义原生 top-k、是否有 shared experts、是否 survivor renorm。必须先通过该模型 native identity。若只用了同家族不同规模，结论称“跨 checkpoint”，不称已经跨架构。
第二任务可选代码生成，例如 MBPP 这类可通过隔离环境中的单元测试评分的任务。进入该阶段前核对官方数据、split、评分协议和模型模板，固定一个任务；若使用代码任务，将生成代码视为待测输入，在受限执行环境评分。该任务自己的校准集、选择集和确认集必须分开，不能用测试通过结果拟合校准器。
两个新增条件各自使用独立 C_fit=128 和预先固定的评估集合，只跑上述五个关键臂。先进行工程 pilot，再使用至少 256 个评估样本或任务约定的完整集合；最终样本数按主要效应所需精度冻结。新的任务可以有任务特定的 cap，但各方法一致并做必要的截断敏感性分析。
16.3 校准抽样稳定性
对初始 native-state 校准、M_NT、M_ST 三个核心臂，增加两组预先抽取的校准样本复验，总计三组校准子集。每组先拟合本组初始校准器，再产生其 S0 rollout，再完成本组同文本重估。
每个子集大小一致，不接触诊断/验证/确认集；报告子集重叠率。若样本重叠，它们是稳健性检查，不当作完全独立的三次研究。
这一阶段反映校准数据选择的不确定性，不能用固定参数的 prompt bootstrap 代替。
16.4 科学贡献与暂缓内容
若主线与确认成功，论文贡献围绕三点组织：
- 诊断：用同文本、不同 driver 的受控设计，定位聚合校准对内部运行状态来源的依赖。
- 方法：在相同校准形式和参数量下，证明基于实际运行状态重估的额外作用，并检查更新后自身状态。
- 评估：用独立确认与有限跨设置实验，明确何时改善生成、何时失效，以及完整成本。
这些仍需与最新相关工作逐项对齐。“使用 student 数据重校准”本身是一般思想；仅更换术语或数据来源不自动产生新颖性。必须由具体的失败条件、可区分的状态因素和有意义的修复结果支撑。
暂缓大范围 K/层/时间窗口扫描、v32 脉冲矩阵、K8 gain 剂量曲线、新 router 训练、强迫 EOS、新 serving 系统与 kernel 优化。只有当前结果指向一个具体剩余问题时，再建立对应的新实验；扩实验数量不能代替主假设证据。
17 参考来源
仓库事实和已知问题以实际 checkout 为准；以下链接提供审阅依据。实现 LDA 或其他论文方法前，要再次读取原文的公式与算法，不只依赖摘要。
1. MOEresearch 仓库
2. 审阅提交的最新实验报告
3. 现有校准实现
4. 旧运行 manifest
5. Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models
6. Distribution-Consistent Inference for Dynamic Sparse Mixture-of-Experts（LDA）
7. To Reason but Not to Stop: Understanding and Recovering Pruned Large Reasoning Models
8. Qwen 官方模型卡
9. Qwen 官方配置
10. Transformers 当前 Qwen MoE 实现
以上文献说明近邻覆盖和实现来源，不表示本轮假设已经被证明，也不保证候选方法具有尚未出现过的全球首次性。