#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate vllm-bench
export CUDA_VISIBLE_DEVICES=0

SGLANG_ENV=/home/t-jialianggu/.conda/envs/sglang-dev
export CUDA_HOME=$SGLANG_ENV
export PATH=$SGLANG_ENV/bin:$PATH
export CPATH=$SGLANG_ENV/targets/x86_64-linux/include:$SGLANG_ENV/lib/python3.11/site-packages/nvidia/cublas/include:$SGLANG_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/include:$CPATH
export LIBRARY_PATH=/usr/lib64:$SGLANG_ENV/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$SGLANG_ENV/lib:$SGLANG_ENV/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export VLLM_USE_FLASHINFER_SAMPLER=0

# Force flashinfer_cutlass MoE backend via vLLM kernel-config
vllm serve /data/hf/models/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name qwen3-30b-a3b-moe \
  --host 127.0.0.1 --port 30001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --dtype bfloat16 \
  --trust-remote-code \
  --kernel-config '{"moe_backend": "flashinfer_cutlass"}' \
  > /home/t-jialianggu/work/MOEresearch/results/4way_bench/vllm_cutlass/server.log 2>&1
