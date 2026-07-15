#!/bin/bash
export HOME=/home/t-jialianggu
export CUDA_VISIBLE_DEVICES=5
export CUDA_HOME=/home/t-jialianggu/.conda/envs/sglang-dev
export TRITON_CACHE_DIR=/tmp/tc_v12_ngram
mkdir -p $TRITON_CACHE_DIR
export CPATH=/home/t-jialianggu/.conda/envs/sglang-dev/targets/x86_64-linux/include:/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/nvidia/cublas/include:/home/t-jialianggu/.conda/envs/sglang-dev/lib/python3.11/site-packages/nvidia/cuda_runtime/include
export LIBRARY_PATH=/home/t-jialianggu/.conda/envs/sglang-dev/lib:/home/t-jialianggu/.conda/envs/sglang-dev/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=/home/t-jialianggu/.conda/envs/sglang-dev/lib:/home/t-jialianggu/.conda/envs/sglang-dev/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/home/t-jialianggu/.conda/envs/sglang-dev/bin:/usr/local/bin:/usr/bin:/bin
exec /home/t-jialianggu/.conda/envs/sglang-dev/bin/python -m sglang.bench_one_batch   --model-path /data/hf/models/Qwen3-30B-A3B-Instruct-2507 --tokenizer-path /data/hf/models/Qwen3-30B-A3B-Instruct-2507 --trust-remote-code   --batch-size 32 --input-len 2700 --output-len 32   --profile --profile-activities CUDA_PROFILER --profile-stage decode   --profile-filename-prefix /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-15_v12_ncu_spec/ngram/sgl --result-filename /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/2026-07-15_v12_ncu_spec/ngram/res.jsonl --run-name v12_ngram   --mem-fraction-static 0.85 --chunked-prefill-size 16384 --attention-backend fa3 --moe-runner-backend triton --speculative-algorithm NGRAM --speculative-num-draft-tokens 8 --speculative-num-steps 4
