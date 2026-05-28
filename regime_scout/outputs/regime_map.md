# Regime Map

- Generated at: 2026-05-28 17:11:09
- Server config: `configs/base.yaml`
- Model: `/data/hf/models/Qwen3-0.6B`
- Hardware: H200
- SGLang version: (unknown)

## Overview

- Workloads run: **12**
- Passed: **12**
- Failed: **0**
- Clusters: **10**

## Regime clusters

### R_scheduler_tail  (scheduler_tail)

- Passed members (3): scheduler_overhead_high_concurrency, scheduler_overhead_high_concurrency__con_16, scheduler_overhead_high_concurrency__con_32
- Primary metric: `ttft_p95_ms` (lower-is-better)
- Median primary: **95.98**
- Median TTFT p99/p50 ratio: **2.97**
- Max suspicion score: **0.735**
- Top workload by score: `scheduler_overhead_high_concurrency`

### R_sanity  (sanity)

- Passed members (1): smoke
- Primary metric: `ttft_p95_ms` (lower-is-better)
- Median primary: **95.98**
- Median TTFT p99/p50 ratio: **4.06**
- Max suspicion score: **0.150**
- Top workload by score: `smoke`

### R_scheduler_or_cuda_graph  (scheduler_or_cuda_graph)

- Passed members (1): short_in_short_out
- Primary metric: `ttft_p95_ms` (lower-is-better)
- Median primary: **98.65**
- Median TTFT p99/p50 ratio: **2.90**
- Max suspicion score: **0.150**
- Top workload by score: `short_in_short_out`

### R_prefill  (prefill)

- Passed members (1): prefill_medium
- Primary metric: `ttft_p95_ms` (lower-is-better)
- Median primary: **49.63**
- Median TTFT p99/p50 ratio: **2.77**
- Max suspicion score: **0.150**
- Top workload by score: `prefill_medium`

### R_prefill_boundary  (prefill_boundary)

- Passed members (1): prefill_long
- Primary metric: `ttft_p95_ms` (lower-is-better)
- Median primary: **120.05**
- Median TTFT p99/p50 ratio: **1.89**
- Max suspicion score: **0.150**
- Top workload by score: `prefill_long`

### R_decode  (decode)

- Passed members (1): decode_medium
- Primary metric: `output_throughput` (higher-is-better)
- Median primary: **5759.07**
- Median TTFT p99/p50 ratio: **5.01**
- Max suspicion score: **0.150**
- Top workload by score: `decode_medium`

### R_decode_saturation  (decode_saturation)

- Passed members (1): decode_heavy
- Primary metric: `output_throughput` (higher-is-better)
- Median primary: **10140.72**
- Median TTFT p99/p50 ratio: **5.42**
- Max suspicion score: **0.150**
- Top workload by score: `decode_heavy`

### R_prefix_cache  (prefix_cache)

- Passed members (1): prefix_reuse_ideal
- Primary metric: `ttft_p95_ms` (lower-is-better)
- Median primary: **177.30**
- Median TTFT p99/p50 ratio: **1.50**
- Max suspicion score: **0.150**
- Top workload by score: `prefix_reuse_ideal`

### R_cache_churn  (cache_churn)

- Passed members (1): prefix_churn
- Primary metric: `ttft_p95_ms` (lower-is-better)
- Median primary: **185.40**
- Median TTFT p99/p50 ratio: **3.11**
- Max suspicion score: **0.150**
- Top workload by score: `prefix_churn`

### R_scheduler_overhead  (scheduler_overhead)

- Passed members (1): tiny_latency
- Primary metric: `ttft_p95_ms` (lower-is-better)
- Median primary: **13.84**
- Median TTFT p99/p50 ratio: **1.43**
- Max suspicion score: **0.032**
- Top workload by score: `tiny_latency`

## Top suspicious workloads (overall)

| Rank | Workload | Regime | Status | Score | Primary metric | Value |
|---:|---|---|---|---:|---|---:|
| 1 | `scheduler_overhead_high_concurrency` | scheduler_tail | pass | 0.735 | ttft_p95_ms | 434.23 |
| 2 | `smoke` | sanity | pass | 0.150 | ttft_p95_ms | 98.66 |
| 3 | `short_in_short_out` | scheduler_or_cuda_graph | pass | 0.150 | ttft_p95_ms | 98.65 |
| 4 | `prefill_medium` | prefill | pass | 0.150 | ttft_p95_ms | 49.63 |
| 5 | `prefill_long` | prefill_boundary | pass | 0.150 | ttft_p95_ms | 120.05 |
| 6 | `decode_medium` | decode | pass | 0.150 | output_throughput | 5759.07 |
| 7 | `decode_heavy` | decode_saturation | pass | 0.150 | output_throughput | 10140.72 |
| 8 | `prefix_reuse_ideal` | prefix_cache | pass | 0.150 | ttft_p95_ms | 177.30 |
| 9 | `prefix_churn` | cache_churn | pass | 0.150 | ttft_p95_ms | 185.40 |
| 10 | `scheduler_overhead_high_concurrency__con_16` | scheduler_tail | pass | 0.150 | ttft_p95_ms | 95.98 |
| 11 | `scheduler_overhead_high_concurrency__con_32` | scheduler_tail | pass | 0.150 | ttft_p95_ms | 121.67 |
| 12 | `tiny_latency` | scheduler_overhead | pass | 0.032 | ttft_p95_ms | 13.84 |
