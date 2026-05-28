# Log layout — Stage 1 RegimeScout

> 🇬🇧 English first · 🇨🇳 [跳转中文版](#-中文版)

Everything produced by a Stage 1 suite run is captured. Nothing is overwritten.

## Suite-level

| File | Content |
|---|---|
| `logs/regime_suite_<ts>.log` | One INFO line per workload start/end with metric summary, plus orchestrator-level events (budget, errors). Plain text, timestamped. |
| `regime_scout/outputs/raw_results.jsonl` | One JSON line per workload run. Stable schema (see `scripts/run_regime_suite.py` docstring). Includes the full normalized `metrics` blob and the path to its `run_dir`. |

## Per-workload (`experiments/tmp/regime_scout/<suite_ts>/run_NNNN_<name>/`)

| File | Content |
|---|---|
| `workload_input.yaml`    | The workload YAML the suite ingested (byte-identical copy of the file under `regime_scout/candidates/`). |
| `workload_snapshot.yaml` | Second copy taken by `run_experiment.py` (redundant safety net). |
| `config_snapshot.yaml`   | The sglang server config used for this run. |
| `server.log`             | Full sglang `launch_server` stdout+stderr: model load, KV cache size, CUDA graph capture report, request-level prefill logs, shutdown signal. Contains the SGLang version banner and the `max_total_num_tokens` / `chunked_prefill_size` / `max_running_requests` / `context_len` summary line. |
| `<mode>_benchmark.log`   | `sglang.bench_serving` stdout+stderr — the human-readable per-run summary table and any Python tracebacks. |
| `<mode>_raw.jsonl`       | `sglang.bench_serving --output-file --output-details`. Single JSON object containing aggregate metrics + per-request arrays: `ttfts[]`, `itls[]`, `input_lens[]`, `output_lens[]`, `errors[]`, `generated_texts[]`. This is the canonical raw data. |
| `<mode>_metrics.json`    | Our normalized metrics derived from `<mode>_raw.jsonl` + log scans (see `scripts/parse_metrics.py`). Stable schema across modes. |
| `orchestrator.log`       | `run_experiment.py` stdout+stderr: launch command, CUDA_VISIBLE_DEVICES, CUDA_HOME, server ready time, parse status. |

## How to inspect a single run after the fact

```bash
RUN=$(ls -td experiments/tmp/regime_scout/*/run_0005_* | head -1)
echo "=== metrics ==="; cat "$RUN"/*_metrics.json
echo "=== server tail ==="; tail -50 "$RUN/server.log"
echo "=== bench raw ==="; python -m json.tool < "$RUN"/*_raw.jsonl | head -60
```

## How to inspect the whole suite

```bash
# Suite log (chronological)
tail -f logs/regime_suite_<ts>.log

# All rows as a compact table
python -c "
import json
for line in open('regime_scout/outputs/raw_results.jsonl'):
    r = json.loads(line)
    m = r.get('metrics') or {}
    print(f\"{r['run_id']:8s} {r['status']:5s} {r['workload_name']:40s} \"
          f\"ttft_p95={m.get('ttft_p95_ms')} out_tps={m.get('output_throughput')}\")
"
```

## Retention

Per-workload `run_dir/` is kept under `experiments/tmp/regime_scout/<suite_ts>/`.
That root is **never overwritten**: every suite invocation creates a fresh
timestamped sub-dir. Cleanup is manual.

---
---

# 🇨🇳 中文版

> 注：本文档仍然用 Stage 1 命名（v0.3 时代），路径仍正确（仓库重构
> 时 `regime_scout/` 没动），但下次写 Problem-Setter 文档时这份会一起
> 被吸收进 setter PLAYBOOK。

Stage 1 suite 跑出来的所有东西都会留下。**任何东西都不会被覆盖**。

## Suite 级

| 文件 | 内容 |
|---|---|
| `logs/regime_suite_<时间戳>.log` | 每个 workload 的开始/结束一行 INFO，含 metric 摘要 + orchestrator 级事件（budget、错误）。带时间戳的纯文本。 |
| `regime_scout/outputs/raw_results.jsonl` | 每 workload 一行 JSON。schema 稳定（见 `scripts/run_regime_suite.py` 的 docstring）。含完整归一化 `metrics` blob 和它的 `run_dir` 路径。 |

## 每 workload（`experiments/tmp/regime_scout/<suite时间戳>/run_NNNN_<name>/`）

| 文件 | 内容 |
|---|---|
| `workload_input.yaml`    | suite 摄入的 workload yaml（与 `regime_scout/candidates/` 下的副本字节完全一致）。 |
| `workload_snapshot.yaml` | `run_experiment.py` 自己留的第二份副本（冗余兜底）。 |
| `config_snapshot.yaml`   | 本 run 用的 sglang server 配置。 |
| `server.log`             | sglang `launch_server` 完整 stdout+stderr：model load、KV cache size、CUDA graph capture 报告、请求级 prefill 日志、shutdown 信号。包含 SGLang 版本 banner 和 `max_total_num_tokens / chunked_prefill_size / max_running_requests / context_len` 摘要行。 |
| `<mode>_benchmark.log`   | `sglang.bench_serving` 的 stdout+stderr —— 人类可读的 per-run 摘要表和任何 Python traceback。 |
| `<mode>_raw.jsonl`       | `sglang.bench_serving --output-file --output-details`。单个 JSON object，含聚合 metric + per-request 数组：`ttfts[]`、`itls[]`、`input_lens[]`、`output_lens[]`、`errors[]`、`generated_texts[]`。这是 canonical raw data。 |
| `<mode>_metrics.json`    | 我们对 `<mode>_raw.jsonl` + log scan 做的归一化 metric（见 `scripts/parse_metrics.py`）。schema 跨 mode 稳定。 |
| `orchestrator.log`       | `run_experiment.py` 的 stdout+stderr：launch 命令、CUDA_VISIBLE_DEVICES、CUDA_HOME、server ready 时间、parse 状态。 |

## 怎么事后看一个 run

```bash
RUN=$(ls -td experiments/tmp/regime_scout/*/run_0005_* | head -1)
echo "=== metrics ==="; cat "$RUN"/*_metrics.json
echo "=== server tail ==="; tail -50 "$RUN/server.log"
echo "=== bench raw ==="; python -m json.tool < "$RUN"/*_raw.jsonl | head -60
```

## 怎么整体看 suite

```bash
# Suite 日志（时间序）
tail -f logs/regime_suite_<时间戳>.log

# 把所有行做成紧凑表
python -c "
import json
for line in open('regime_scout/outputs/raw_results.jsonl'):
    r = json.loads(line)
    m = r.get('metrics') or {}
    print(f\"{r['run_id']:8s} {r['status']:5s} {r['workload_name']:40s} \"
          f\"ttft_p95={m.get('ttft_p95_ms')} out_tps={m.get('output_throughput')}\")
"
```

## 保留策略

每 workload 的 `run_dir/` 留在 `experiments/tmp/regime_scout/<suite时间
戳>/` 下。这个根目录**从不被覆盖**：每次 suite 调用都建新的带时间戳
子目录。清理需要手动。
