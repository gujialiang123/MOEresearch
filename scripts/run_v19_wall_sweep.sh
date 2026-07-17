#!/bin/bash
# v19 Part A: prefill vs decode wall proportion across concurrency, agent (toolagent) workload.
# One server, sweep --max-concurrency. Prefill wall ~= TTFT; Decode wall ~= E2E - TTFT.
# Saves per-run bench_result.jsonl (+ --output-details raw per-request) for post-hoc metrics.
set -e
MODEL_KEY=${MODEL_KEY:-qwen}
GPU=${GPU:-6}
PORT=${PORT:-31890}
NUMP=${NUMP:-300}
CONCS=${CONCS:-"1 4 8 16 32 64"}

if [ "$MODEL_KEY" = "lfm" ]; then
  MODELP=/data/hf/LFM2.5-8B-A1B; CHUNK=4096; NAME=lfm2.5-8b-a1b
else
  MODELP=/data/hf/models/Qwen3-30B-A3B-Instruct-2507; CHUNK=16384; NAME=qwen3-30b-a3b-bf16
fi
CONDA=/home/t-jialianggu/.conda/envs/sglang-dev
OUT=/home/t-jialianggu/work/MOEresearch/results/2026-07-15_v19_wall_sweep/$NAME
mkdir -p $OUT

export HOME=/home/t-jialianggu CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=$CONDA
export PATH=$CONDA/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$CONDA/lib:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR=/tmp/tc_v19_$MODEL_KEY; mkdir -p $TRITON_CACHE_DIR

echo "[1] launch server ($NAME) on GPU$GPU port $PORT ..."
$CONDA/bin/python -m sglang.launch_server \
  --model-path $MODELP --tokenizer-path $MODELP --trust-remote-code \
  --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 \
  --mem-fraction-static 0.85 --chunked-prefill-size $CHUNK \
  --schedule-policy lpm --max-running-requests 128 \
  --context-length 32768 --moe-runner-backend triton \
  > $OUT/server.log 2>&1 &
SRV_PID=$!
echo "  server pid $SRV_PID"
START=$(date +%s)

echo "[2] wait ready..."
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; do
  sleep 5
  if ! kill -0 $SRV_PID 2>/dev/null; then echo "  server died"; tail -20 $OUT/server.log; exit 1; fi
  [ $(( $(date +%s) - START )) -gt 300 ] && { echo "  timeout"; kill $SRV_PID; exit 1; }
done
echo "  ready at $(( $(date +%s) - START ))s"

echo "[3] warmup..."
curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
  -d '{"text":"hello","sampling_params":{"max_new_tokens":8}}' >/dev/null 2>&1 || true

for C in $CONCS; do
  echo "[4] bench max-concurrency=$C ..."
  $CONDA/bin/python -m sglang.bench_serving --backend sglang \
    --host 127.0.0.1 --port $PORT --model $MODELP \
    --dataset-name mooncake --mooncake-workload toolagent \
    --num-prompts $NUMP --request-rate inf --max-concurrency $C \
    --output-details \
    --output-file $OUT/bench_c${C}.jsonl > $OUT/bench_c${C}.log 2>&1 || echo "  c=$C failed (see log)"
  echo "    done c=$C"
done

echo "[5] shutdown server $SRV_PID"
kill $SRV_PID 2>/dev/null || true
echo "ALL DONE $NAME"
