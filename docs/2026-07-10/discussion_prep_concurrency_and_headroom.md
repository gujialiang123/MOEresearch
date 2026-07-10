# 讨论准备：concurrency / serving command / TBT headroom（回应 Dey&Chendi 追问）

> 结论先行：**你的理解是对的** —— 只有 ~20 并发是因为 simulator（mooncake toolagent）按真实到达时间戳回放请求，模拟了请求之间的真实时间间隔。这是 **benchmark（负载）** 的特性，不是 server 或 model 的能力上限。

---

## 0. 你的理解对不对？——对。而且有硬证据

mooncake toolagent 数据集用 `get_mooncake_request_over_time` 按 trace 里的**真实时间戳**逐个放请求（`--mooncake-slowdown-factor 1.0` = 原速回放）。所以并发数由**请求到达速率**决定，不是 server 顶不住。

**证据（同一负载、不同 server cap，并发几乎不变）：**
| 跑法 | `--max-running-requests` | 实测并发 |
|---|---|---|
| v7 | 32 | LFM 6.1 / Qwen 25.4 |
| v9d | **128** | LFM 6.2 / Qwen 19.7 |

> server 容量从 32 提到 128，并发**几乎没变**（6→6，25→20）——说明瓶颈是**到达速率**，不是 server 容量。这直接证明"20 并发"来自 benchmark 的真实到达模拟，不是 serve/model 限制。

---

## 1. 精确的 serving + benchmark 命令

**Server（LFM2.5，Qwen3 只改 model-path + chunked-prefill-size）：**
```bash
python -m sglang.launch_server \
  --model-path /data/hf/LFM2.5-8B-A1B --tokenizer-path /data/hf/LFM2.5-8B-A1B \
  --trust-remote-code --host 127.0.0.1 --port 31230 --tensor-parallel-size 1 \
  --mem-fraction-static 0.85 --chunked-prefill-size 4096 \
  --schedule-policy lpm --max-running-requests 128 \
  --context-length 32768 --moe-runner-backend triton
```
- Qwen3-30B：`--model-path /data/hf/models/Qwen3-30B-A3B-Instruct-2507`，`--chunked-prefill-size 16384`。
- 这是 v8 tuning 出的最优 config（cap=128、triton MoE、各自最优 chunked-prefill）。

**Benchmark（真实到达，simulator 原速回放）：**
```bash
python -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port 31230 --model <model_path> \
  --dataset-name mooncake --mooncake-workload toolagent \
  --num-prompts 200 --mooncake-slowdown-factor 1.0
```
- `--mooncake-slowdown-factor 1.0` = 原速回放真实时间戳 → 并发 6–20（真实到达）。
- 想提高并发就调这个：`0.1` = 10× 加速到达，`--max-concurrency N` = 闭环固定 N 个在飞。

---

## 2. 增加 concurrency 会有帮助吗？——分指标看，有得有失

**怎么增加**：压缩 simulator 时间间隔（`--mooncake-slowdown-factor 0.1`）或闭环限流（`--max-concurrency 64/128`），或真实场景里接更多用户。

| 指标 | 加并发的影响 |
|---|---|
| **GPU 利用率 / server idle** | ✅ 大幅改善——把现在 ~85% 的 idle 填上（v8 实测：cap128 饱和下吞吐比 cap32 高 +48~89%） |
| **吞吐（token/s、req/s）** | ✅ 明显提升 |
| **TBT / TPOT（每 token 延迟）** | ❌ 变差——batch 越大，decode 每步越慢（v8 实测：Qwen cap32 TPOT 20ms → cap128 53ms） |
| **kernel 层 SM 空转** | ⚠️ 不解决——SM 空转是 occupancy/延迟问题，即使喂满 batch 仍占 GPU 39–46%（v9 饱和场景实测） |

**一句话**：加并发能**回收 server idle（吞吐/利用率）**，但**换来更高的 TBT**，且**不触及 kernel 层的 SM 空转**。到底要不要加，取决于你优化的是吞吐还是延迟（SLA）。

> 注意：v8 已确认 `max-running-requests=128` 是拐点，再往上（192/256）吞吐不再涨——所以"加并发"的上限是负载本身，不是无限的。

---

## 3. TBT headroom 是怎么算的（给 copilot-explain 那个问题）

**用的数据（全实测）**：NCU 在最优 config、真实 agent decode 点（in=2700）下测的每个 kernel 的 `Duration`、`SM Throughput%`、`Memory Throughput%`。

**公式（roofline，只算 exact 方法、不改字节/FLOP）**：
```
每个 kernel:  busiest_pipe = max(SM%, Mem%)          # 该 kernel 最忙的硬件通路
             floor_time   = Duration × busiest_pipe/100   # 打满该通路后的理想时长
TBT headroom = Σ Duration / Σ floor_time
```
直觉：一个 kernel 若最忙的资源只用了 45%，而它的活（字节/FLOP）不变，理论上打满该资源后时间能压到 45% → 剩下 55% 是硬件空等，可回收。按 duration 加权汇总得到整步的可压缩倍数。

**各 setup 的上界（实测外推）**：
| 模型 | decode regime | 时间加权 busiest-pipe | TBT 上界 |
|---|---|---|---|
| LFM2.5 | b32 / b64 | 42% / 45% | ~2.4× / ~2.2× |
| Qwen3-30B | b32 / b64 | 54% / 56% | ~1.9× / ~1.8× |

**性质**：是**理论上界**（假设 kernel 能把最忙 pipe 打到 100%，真实达不到）；且只算 exact 方法（量化/投机解码会另加 headroom）。不同 model/setup 差异来自各自 kernel 的实测利用率——Qwen3 的 fused_moe/attention 把带宽压得更高（54%），所以 headroom 比 LFM（42%）小。

---

## 4. 我方观点（讨论主线）

我们在 agent workload 上 profile 出**两个独立的浪费源**，分别对应两个优化层：
- **kernel 层 SM 空转**（decode GPU 时间的 67%/78%，LFM/Qwen）→ 靠 **kernel-level optimization**（提 occupancy、隐藏内存延迟）。TBT 上界 ~2× 。
- **serving idle**（墙钟的 86%/81%）→ 靠 **policy / framework-level**（攒批、连续批处理、多租户、更高负载）。

其中 serving idle 主要是 **benchmark 的真实到达模拟**造成的低负载——真实部署里靠多用户/攒批可回收；它不是 kernel 或硬件的 gap。两个 lever 正交，config tuning 都够不到。
