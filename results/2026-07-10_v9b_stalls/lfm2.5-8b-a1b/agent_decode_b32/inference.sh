#!/bin/bash
export HOME=/home/t-jialianggu
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/home/t-jialianggu/.conda/envs/sglang-dev
export TRITON_CACHE_DIR=/tmp/sglang_triton_cache_v9_lfm
mkdir -p $TRITON_CACHE_DIR
export CPATH=/home/t-jialianggu/.conda/envs/sglang-dev/targets/x86_64-linux/include:/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/nvidia/cublas/include:/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/nvidia/cuda_runtime/include
export LIBRARY_PATH=/home/t-jialianggu/.conda/envs/sglang-dev/lib:/home/t-jialianggu/.conda/envs/sglang-dev/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=/home/t-jialianggu/.conda/envs/sglang-dev/lib:/home/t-jialianggu/.conda/envs/sglang-dev/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/home/t-jialianggu/.conda/envs/sglang-dev/bin:/usr/local/bin:/usr/bin:/bin

exec /home/t-jialianggu/.conda/envs/sglang-dev/bin/python -m sglang.bench_one_batch \
  --model-path /data/hf/LFM2.5-8B-A1B \
  --tokenizer-path /data/hf/LFM2.5-8B-A1B \
  --trust-remote-code \
  --batch-size 32 \
  --input-len 2700 \
  --output-len 32 \
  --profile \
  --profile-activities CUDA_PROFILER \
  --profile-stage decode \
  --profile-filename-prefix /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-10_v9b_stalls/lfm2.5-8b-a1b/agent_decode_b32/sglang_bench \
  --result-filename /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-10_v9b_stalls/lfm2.5-8b-a1b/agent_decode_b32/bench_one_batch_result.jsonl \
  --run-name ncu_v9 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --schedule-policy lpm \
  --moe-runner-backend triton
