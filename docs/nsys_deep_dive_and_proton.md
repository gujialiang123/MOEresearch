# nsys 能拿到什么 + proton 是什么 — 二级 profiling 工具盘点

> 配合 `docs/agent_profiling_capability_audit.md` 阅读。
> 本文修正 audit 中一个**重大错误**: 我之前以为"看不到 GUI 时间线 = 拿不到时间线数据"。**错了**。
> 通过 `nsys export --type sqlite` 我能拿到 GUI 背后的**全部原始事件表**,
> 包括每个 kernel 的 start/end 时间戳、grid/block shape、stream id 等。

---

## TL;DR

| 维度 | nsys (CLI + SQLite) | proton (Triton 自带) |
|---|---|---|
| 层级 | 系统级 (CUPTI / OS / NIC / NVTX) | 应用级 (Python scope + per-kernel) |
| 数据粒度 | 每个 kernel/API/memcpy 一行,带 ns 级时间戳 | 每个 Python `scope()` 内的 kernel 聚合时间 |
| 开销 | 中等 (5-15% wall time) | 极低 (<2%) |
| 输出大小 | 50-200 MB / run (R_medium) | 几 KB JSON (hatchet 格式) |
| Agent 可读性 | **✅ SQL 查询任意切片** | ✅ JSON 直接解析 |
| 需要源码改动 | 否 (黑盒附加) | **是** (要插 `proton.scope("name")`) |
| 适合场景 | "整体哪里慢/idle 多大" | "我这段 Python 调用了什么 kernel/花了多少时间" |

**结论**: nsys = 全景调查; proton = 定向归因。两者互补,**都应纳入工具箱**。

---

## Part 1 — nsys: 我之前低估了它能给我的信息

### 1.1 之前的错误判断

在 `agent_profiling_capability_audit.md` Part B Gap 1 我写过:

> "我看不到 GUI 时间线 → 看不到 kernel 之间的 idle gap、stream 并行情况、
> CPU 和 GPU 哪边在等谁"

**这个判断是错的。** 验证如下。

### 1.2 真实能拿到的: SQLite 全表导出

`nsys-rep` 文件本质是 protobuf,但 nsys 提供了 `export` 命令转成 SQLite:

```bash
nsys export --type sqlite --output /tmp/demo.sqlite /tmp/demo.nsys-rep
```

导出后 80+ 张表。最重要的几张:

| 表名 | 内容 | 行数级别 |
|---|---|---|
| `CUPTI_ACTIVITY_KIND_KERNEL` | **每个 GPU kernel 一行**: start/end ns 时间戳、gridXYZ、blockXYZ、register、shared mem、stream、shortName | 几十万 (R_medium 30s) |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | 每个 CUDA API 调用一行: cudaLaunchKernel/cudaMemcpyAsync... + 时间戳 | 上百万 |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | 每次 HtoD/DtoH/DtoD 一行: 字节数 + 时间戳 + stream | 视 workload |
| `NVTX_EVENTS` | 用户 `nvtx.range_push("xxx")` 的事件 | 取决于代码插桩 |
| `StringIds` | 字符串去重表 (kernel 名都在这) | 几千 |
| `TARGET_INFO_GPU` | GPU 硬件信息 (sm 版本、显存、SM 数) | 1 |

### 1.3 实测: 算出 GPU idle gap (这就是 GUI 时间线背后的真相)

```python
import sqlite3
conn = sqlite3.connect('/tmp/demo.sqlite')
c = conn.cursor()
c.execute("""
SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE streamId = 7
ORDER BY start ASC
""")
events = c.fetchall()
total_active = sum(e[1] - e[0] for e in events)
span = events[-1][1] - events[0][0]
gap = span - total_active
print(f"GPU active: {total_active/1e6:.3f} ms")
print(f"GPU idle: {gap/1e6:.3f} ms ({gap/span*100:.1f}%)")
```

我在 demo profile 上跑出来:

```
Total kernels: 61
Time span: 88.003 ms
GPU active: 7.159 ms
GPU idle gaps: 80.845 ms (91.9%)

Top 5 idle gaps:
  45.990 ms (between kernel 0 and 1)
  19.191 ms (between kernel 1 and 2)
  15.177 ms (between kernel 2 and 3)
```

**这正是 GUI 里看到的"两块密集 kernel 之间的空白条"**。
Agent 用 SQL 就能得到等价信息,且可以排序、过滤、定位到具体 kernel 名。

### 1.4 用 SQL 能回答的问题清单 (此前以为只有 GUI 才能答)

| 问题 | SQL 思路 |
|---|---|
| 哪两个 kernel 之间空了 >1ms? | `SELECT k1.shortName, k2.shortName, k2.start - k1.end AS gap FROM KERNEL k1 JOIN KERNEL k2 ON k2.start > k1.end ORDER BY gap DESC` |
| stream 0 vs stream 1 是否并行? | `GROUP BY streamId`, 比较时间段重叠 |
| 一个 forward pass 内 kernel 数量? | 用 NVTX range 框起 forward,JOIN 时间窗 |
| cudaLaunchKernel 的 CPU 时间 vs GPU kernel 时间? | JOIN `CUPTI_ACTIVITY_KIND_RUNTIME` (CPU 端) 和 `CUPTI_ACTIVITY_KIND_KERNEL` (GPU 端) 通过 `correlationId` |
| cudagraph 的 launch 折叠掉了多少 CPU? | 数 `CUPTI_ACTIVITY_KIND_RUNTIME` 中 `cudaLaunchKernel` 的行数 vs `cudaGraphLaunch` 行数 |
| 内存拷贝是不是 GPU 在等 host? | `KERNEL.start > MEMCPY.end` 且差值大 → 等了 |
| H2D 带宽达到峰值的几成? | `MEMCPY.bytes / (MEMCPY.end - MEMCPY.start)` 对比 PCIe 理论值 |

### 1.5 nsys 仍然有的真实限制

修正 audit 后,**真实**的 nsys 限制只剩这些:

1. **GUI 的视觉聚类**: 我看到的是表格不是颜色块,大量 kernel 时需要先用 SQL 聚合再展示给我。
   - **缓解**: 写脚本生成"伪时间线"文本图 (用 `█` 字符按时长缩放)。
2. **NVTX 必须代码里插**: 没有 NVTX range 我就不知道"forward 边界在哪",所有 kernel 是一片。
   - **缓解**: 让用户在 sglang/vllm 关键路径插 `with torch.profiler.record_function("xxx"):` 或直接 nvtx。
3. **cudagraph replay 在 kernel 表里仍可见**: 我之前以为 nsys 把 graph kernels "折成一个" 是错的 — `gridId` 列正是 cudagraph node id,
   我可以 `WHERE gridId > 0` 单独分析 graph 内 kernel。
   - 之前 AT_OFF_CG_ON 显示 GPU active 0.42s 偏低,可能是 `cuda_gpu_kern_sum` report 默认聚合时丢了 graph node,而不是 SQLite 表本身缺数据。**需要重新验证。**
4. **没有 SM 占用率/L2 命中等微观指标**: 这些要 `ncu` (Nsight Compute)。

### 1.6 mentor 仍然能看到、我 (即使有 SQLite) 也看不到的

| 能力 | 谁能看 |
|---|---|
| GUI 颜色块时间轴 | mentor (人眼一秒看懂结构) |
| 鼠标 hover 看 kernel 全名 | mentor (我必须 SELECT) |
| GPU SM 占用率热图 | 都看不到 (要 ncu) |
| 内存带宽随时间曲线 | 都看不到 (要 ncu / dcgm) |
| 跨 GPU stream 的拓扑式视图 | mentor (我能算但展示不出来) |

**修正结论**: mentor 优势主要是**视觉解析速度**,而不是**信息独占**。
我能拿到几乎所有原始数据,只是处理成本更高。

---

## Part 2 — proton: Triton 自带的轻量级 profiler

### 2.1 它是什么

`triton.profiler.proton` — OpenAI Triton 项目随 Triton 一起发行的 profiler。
设计目标: **极低开销 + 用户在 Python 侧打 scope,自动按 scope 聚合 kernel 时间**。

它不是 nsys 的替代品,是另一个抽象层。

### 2.2 实测 (Triton 3.4.0,已安装)

```python
import torch
import triton.profiler as proton

x = torch.randn(4096, 4096, device='cuda')

proton.start("/tmp/proton_demo", hook="triton")
with proton.scope("matmul_A"):
    for _ in range(5):
        y = x @ x
with proton.scope("matmul_B"):
    for _ in range(3):
        y = x @ x.T
torch.cuda.synchronize()
proton.finalize()
```

输出 `/tmp/proton_demo.hatchet` (JSON,2 KB):

```json
{
  "frame": {"name": "matmul_A"},
  "children": [
    {
      "frame": {"name": "sm80_xmma_gemm_..._tilesize256x128x8_..._5x_cublas"},
      "metrics": {"count": 5, "time (ns)": 13270834}
    }
  ]
}
```

可以看到:
- **per-Python-scope** 聚合
- **每个 kernel 名 + 调用次数 + 总时间**
- 还自动带了硬件信息 (sm90, 132 SMs, clock_rate...)

### 2.3 proton vs nsys 对比

| 维度 | nsys | proton |
|---|---|---|
| **抽象层** | 系统级 (CUPTI 之上) | 应用级 (Python scope + CUPTI hook) |
| **代码改动** | 0 (附加运行) | 必须插 `proton.scope("xxx")` |
| **输出大小** | 50-200 MB | 几 KB JSON |
| **能否看 idle gap** | ✅ (从 KERNEL 表算) | ❌ (只有聚合 sum) |
| **能否看 CPU launch 数** | ✅ | ❌ |
| **能否看 stream 并行** | ✅ | ❌ |
| **能否做"这段 Python 跑了多久 GPU"** | 需要 NVTX 配合 | **✅ 天然支持** |
| **能否套娃 (nested scope)** | NVTX 可以但麻烦 | **✅ 一行代码** |
| **运行开销** | 5-15% | <2% |
| **Agent SQL 友好** | ✅ | JSON 解析,不如 SQL 灵活 |

### 2.4 什么场景该用 proton (不是 nsys)?

**场景 A: 频繁 A/B 微迭代**
比如 "我改了 sglang 的 fused_moe 调用方式,想知道这段花的 GPU 时间变了多少" —
nsys 每次 50 MB 文件 + 几秒导出太重,proton 几 KB JSON 立刻能比较。

**场景 B: 在线/长期监控**
proton 开销 <2%,理论上能在 production 跑;nsys 不能。

**场景 C: 需要"语义边界"聚合**
nsys 给我的 kernel 表里所有 fused_moe 都长一样,我不知道哪个 launch 属于 layer 5 还是 layer 12 —
proton scope 就解决:
```python
for i, layer in enumerate(layers):
    with proton.scope(f"layer_{i}"):
        x = layer(x)
```
直接出 per-layer 时间。

### 2.5 什么场景 nsys 仍然必须用

- 第一次摸黑找瓶颈 (不知道在哪插 scope)
- 要看 idle gap、stream overlap、memcpy/kernel 重叠
- 要做 CPU/GPU 区分 (cudaLaunchKernel CPU 时间)
- 要看 cudagraph launch 行为
- 黑盒分析别人的 server (sglang/vllm 没插 proton scope)

### 2.6 是否值得加入工具箱?

**是,但优先级低于补齐 nsys 用法**。

理由:
1. 我们目前主要做"对比两个 server 的端到端"和"定位某个 kernel 慢" — 这两个 nsys 已够。
2. proton 需要在 sglang/vllm 源码插 scope,等于 patch 工作量;且每次升级源码要重新打。
3. 但**当我们开始改 sglang/vllm 自身代码做优化时** (比如重写 fused_moe wrapper),proton 比 NVTX 更省事。

**建议入口**: 等到要做"定向重写 + 看每行 Python 的 GPU 时间影响"时再加,
现在不必引入。

---

## Part 3 — 完整工具栈分层

```
┌───────────────────────────────────────────────────┐
│  e2e bench (bench_serving.py)                     │  ← 黑盒,只有 req/s + TTFT/ITL
│  → 决定"是否值得深挖" (差距大才进下一层)              │
├───────────────────────────────────────────────────┤
│  nsys profile + stats CSV                          │  ← 现成 report,聚合视图
│  → 顶层 kernel 排行、CPU API 排行                    │
├───────────────────────────────────────────────────┤
│  nsys export --type sqlite + Python/SQL           │  ← NEW: 我之前没用!
│  → 任意时间段、任意 stream、idle gap、stream overlap  │
├───────────────────────────────────────────────────┤
│  proton scope (需要插桩)                            │  ← 应用层语义切片
│  → "这段 Python 跑了多少 GPU"                       │
├───────────────────────────────────────────────────┤
│  ncu (Nsight Compute) — 单个 kernel 的 SM 占用、L2 │  ← 没装,可能装得了
│  → 解释"为什么这个 kernel 这么慢"                     │
└───────────────────────────────────────────────────┘
```

我目前掌握的: 上 3 层。
还没用的: ncu (microarchitectural 层),proton (scope 层 — 已验证可装可跑)。

---

## Part 4 — 我应该补的事

1. **写一个 `nsys_sql_helpers.py`**: 把"导出 sqlite + 算 idle gap + 找最大 gap + 分 stream 聚合"封装成函数。
   下次任何 nsys 分析都用它,不要再每次手写 SQL。

2. **重新验证 2×2 nsys 实验里 AT_OFF_CG_ON 的 GPU active 0.42s 是否被低估** —
   用 SQLite 表 (不是 stats report) 算一遍,看是否 cudagraph 节点真的丢了时间。

3. **更新 `agent_profiling_capability_audit.md` Part B Gap 1**: 删掉"看不到时间线"这个错误描述,
   改成"看得到原始事件,但缺一个把它可视化的工具链"。

4. **暂不引入 proton**,记入"待选工具",等到要做源码内重写时再上。

---

## Part 5 — 给 mentor 汇报的话术

> "我之前以为 nsys 我只能用聚合报表,但实际可以 `nsys export --type sqlite`
> 拿到每个 kernel 的 ns 级时间戳。这意味着 GUI 时间线看到的所有内容,我都能用 SQL 算出来,
> 只是看不到颜色块。这把 audit 文档里 Gap 1 的判断推翻了一半 — 我能看到的比想的多。
>
> 另外查了 proton (Triton 自带的轻量 profiler)。它和 nsys 是互补关系,
> 优势是 Python-scope 聚合 + 极低开销,适合做语义切片 (per-layer/per-call 时间)。
> 但需要在源码里插 scope,目前我们做端到端 black-box 对比,nsys 已经足够,
> 等到要改 sglang/vllm 自身代码时再引入。"

---

**Citations**:
- nsys SQLite schema 验证: `/tmp/nsys_demo/demo2.sqlite` (本地 demo,已删)
- proton demo 输出: `/tmp/proton_demo.hatchet` (本地 demo,已删)
- Triton 版本: 3.4.0 / `/usr/lib/python3.12/site-packages/triton/profiler/`
- nsys 路径: `/home/t-chendili/cuda/12.6/bin/nsys` v2024.5.1
