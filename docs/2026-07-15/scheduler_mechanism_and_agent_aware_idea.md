# sglang 请求调度机制报告 + agent-aware 调度 idea 评估

**目的**：搞清 sglang 里 request 如何被接收/排队/组批/处理，并评估"agent 层知道 request 顺序 → 优化 inference scheduler"这个 idea 的可行性与落脚点。
**代码位置**：`python/sglang/srt/managers/scheduler.py`、`schedule_policy.py`、`io_struct.py`（本地 sglang `bbe9c7eeb`）

---

## 1. 请求的完整生命周期（从进来到出去）

```
用户请求 → recv_requests() → process_input_requests() → handle_generate_request()
    → waiting_queue.append(req)          [进等待队列，记录入队时间]
    ↓
event_loop（每个 tick 循环一次）：
    get_next_batch_to_run()：
      ① 先尝试组 prefill batch（get_new_batch_prefill）
         - policy.calc_priority() 对 waiting_queue 排序
         - PrefillAdder 按显存/token 预算逐个准入
      ② prefill 没有可组的，才跑 decode（update_running_batch）
    → run_batch() → process_batch_result()
```

**核心调度循环**（`scheduler.py:1108 event_loop_normal`）：每个 tick 做三件事——收请求、选一个 batch、跑这个 batch。是一个**单线程串行循环**。

---

## 2. 三个决定性的机制

### 机制 A：prefill 绝对优先于 decode（`get_next_batch_to_run:1935-1944`）
```python
if new_batch is not None:      # 有可组的 prefill batch
    ret = new_batch            # → 先跑 prefill
else:                          # 没有 prefill 才
    ret = self.running_batch   # → 跑 decode
```
**含义**：只要 waiting_queue 里有新请求能组成 prefill，调度器**优先插 prefill**，正在 decode 的请求要等这一步 prefill 做完。这是"新请求尽快拿到首 token（TTFT）"的设计，但会**打断 decode**（TBT 抖动）。

### 机制 B：continuous batching（连续批处理）
- 没有"等满一个 batch 再跑"——每个 tick 都把**当前所有在跑的请求**一起做一步 decode（`running_batch`）。
- 新请求 prefill 完就 `merge_batch` 并入 running_batch（`:1912-1919`），完成的请求随时 `filter_batch` 移出。
- 所以 batch 大小是**动态**的，随请求到达/完成实时变化。这就是为什么单流稀疏到达时 batch 很小（GPU 喂不饱）。

### 机制 C：waiting_queue 的排序策略（`schedule_policy.py`）
两大类：
| 类 | 策略 | 排序依据 |
|---|---|---|
| **Cache-aware** | **lpm**（默认）| 按最长前缀匹配排 —— 让共享 prefix 的请求挨在一起，最大化 radix cache 复用 |
| | dfs-weight | 按 radix tree 的 DFS 权重 |
| **Cache-agnostic** | fcfs | 先来先服务（按入队时间戳）|
| | lof | 按最长输出排 |
| | random / routing-key | 随机 / 按路由键 |

**关键**：`lpm` 在 `waiting_queue > 128` 时自动退化成 fcfs（`_determine_active_policy:159`，因为前缀匹配排序太贵）。

---

## 3. ★ 已存在的关键接口：priority scheduling（和你的 idea 直接相关）

**sglang 已经内置 per-request 优先级调度**：
- `GenerateReqInput` / `Req` 有 **`priority: Optional[int]`** 字段（`io_struct.py:249,751`）——**API 层就能给每个请求带一个优先级数字**。
- server 开 `--enable-priority-scheduling` 后，waiting_queue 用 `_sort_by_priority_and_fcfs` 排序（`schedule_policy.py:300`）：**先按 priority、同优先级再按到达时间**。
- 还有 `priority_scheduling_preemption_threshold`——高优先级请求可以**抢占**低优先级的（`PrefillAdder.preempt_to_schedule`）。

**这意味着**：你的 idea"agent 层知道请求顺序 → 指导 scheduler"**不需要改 sglang 核心**，可以直接用现成的 `priority` 字段传信号。

---

## 4. 评估你的 idea：agent 感知的调度

> **原始 idea**：如果 agent 层知道自己会发出请求的顺序/依赖，能不能借此优化 inference 层的 scheduler？

### 这个 idea 站得住，而且有多个具体落点

**为什么有价值**：inference 层的 scheduler 是"盲"的——它只看到孤立的请求流，不知道：
- 哪些请求**属于同一个 agent 任务**（会共享上下文/后续依赖）；
- 哪个请求的结果**会立即触发下一个请求**（在关键路径上）；
- 哪些请求**可以延后**（后台/推测性调用，非关键路径）。

agent 层恰恰知道这些。把这个语义传下去，scheduler 能做得更好。

### 三个具体可落地的方向（按落地难度排序）

**方向 1（最易，用现成接口）：agent 标注 priority**
- agent 知道哪些请求在**关键路径**上（阻塞用户/下一步），给高 priority；后台/推测调用给低 priority。
- 直接用 `--enable-priority-scheduling` + 请求带 `priority` 字段。
- **预期收益**：关键路径请求 TTFT/端到端延迟下降（不被后台请求排在后面/抢占回来）。
- **实验**：混合关键+后台请求流，对比 priority on/off 的关键路径延迟。**这个当天可做**。

**方向 2（中等）：agent 提示 prefix 共享**
- 多个 agent 请求共享同一大段 system prompt / 工具定义 / 历史。agent 知道这个结构，可以提示 scheduler 把它们**攒在一起、同一时间窗调度**，最大化 radix cache 复用（配合 lpm）。
- 落点：可能需要扩展请求元数据（如 session/group id），或用现成的 `routing-key` 策略（`_sort_by_routing_key`）。
- **预期收益**：prefill 计算下降（cache 命中↑）、吞吐↑。

**方向 3（较难，需改核心）：agent 提示到达时间/依赖图**
- agent 知道"请求 B 会在请求 A 返回后 ~Xms 发出"。scheduler 可据此**预留 KV / 预热 / 攒批**，减少 A→B 之间的 GPU idle。
- 这正好对应我们测到的 **serving idle**（86%）——如果 scheduler 提前知道请求要来，就能更好地填满 GPU。
- 落点：需要新接口传依赖/时序，改 scheduler 的准入逻辑。

### 和我们已有发现的连接
- **serving idle（86%）**：方向 3 直接针对它——agent 时序信息能帮 scheduler 减少"等请求"的空转。
- **prefill 打断 decode（机制 A）**：方向 1 的 priority 能缓解——让关键 decode 不被非关键 prefill 反复打断（TBT 更稳）。
- **prefix 复用（机制 C，lpm）**：方向 2 强化它——agent 语义比"盲猜前缀"更准。

---

## 5. 建议的下一步实验（验证 idea）

**实验 P1（最快，方向 1）：priority scheduling 的关键路径收益**
- 构造混合负载：一半"关键"请求（高 priority）+ 一半"后台"请求（低 priority，长输出占资源）。
- A/B：`--enable-priority-scheduling` on/off，测关键请求的 TTFT / 端到端延迟。
- 预期：priority on 时关键请求延迟显著下降 → 证明"agent 传优先级"有用。

**实验 P2（方向 2）：routing-key / group 调度的 cache 复用收益**
- 多组共享 prefix 的请求交错到达，对比默认 lpm vs 按 group 显式聚合，测 prefill 计算量 / cache 命中率。

---

## 附：关键代码索引
| 机制 | 文件:行 |
|---|---|
| event loop | `scheduler.py:1108` |
| prefill 优先 decode | `scheduler.py:1935` |
| continuous batching (merge/filter) | `scheduler.py:1906,1919` |
| 请求入队 | `scheduler.py:1686` |
| policy 排序 | `schedule_policy.py:114 calc_priority` |
| lpm→fcfs 退化 | `schedule_policy.py:159` |
| **priority 字段** | `io_struct.py:249,751` |
| **priority 排序** | `schedule_policy.py:300` |
| PrefillAdder 准入 | `scheduler.py:2030` |
