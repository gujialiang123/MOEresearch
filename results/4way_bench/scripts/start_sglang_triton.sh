#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate sglang-dev
export TRITON_CACHE_DIR=/tmp/4way_sglang_triton_cache
mkdir -p $TRITON_CACHE_DIR
export CUDA_VISIBLE_DEVICES=0
export CPATH=/home/t-jialianggu/.conda/envs/sglang-dev/targets/x86_64-linux/include:/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/nvidia/cublas/include:/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/nvidia/cuda_runtime/include
cd /home/t-jialianggu/work/MOEresearch
python -m sglang.launch_server \
  --model-path /data/hf/models/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name qwen3-30b-a3b-moe \
  --host 127.0.0.1 \
  --port 30000 \
  --tensor-parallel-size 1 \
  --mem-fraction-static 0.85 \
  --context-length 32768 \
  --max-running-requests 32 \
  --chunked-prefill-size -1 \
  --max-prefill-tokens 16384 \
  --moe-runner-backend triton \
  --trust-remote-code \
  --log-level info \
  > /home/t-jialianggu/work/MOEresearch/results/4way_bench/sglang_triton/server.log 2>&1
