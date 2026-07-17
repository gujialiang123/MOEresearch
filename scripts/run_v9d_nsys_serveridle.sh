#!/bin/bash
# v9d: measure REAL server idle with nsys timeline (delay/duration aligns capture to bench).
set -e
MODEL_KEY=${V9D_MODEL:-lfm}
GPU=${V9D_GPU:-4}
PORT=${V9D_PORT:-31230}
DELAY=${V9D_DELAY:-95}      # seconds before nsys starts recording (must exceed ready+warmup)
DURATION=${V9D_DURATION:-90}
if [ "$MODEL_KEY" = "lfm" ]; then
  MODELP=/data/hf/LFM2.5-8B-A1B; CHUNK=4096; NAME=lfm2.5-8b-a1b
else
  MODELP=/data/hf/models/Qwen3-30B-A3B-Instruct-2507; CHUNK=16384; NAME=qwen3-30b-a3b-bf16
fi
NSYS=/home/t-chendili/cuda/12.9/bin/nsys
CONDA=/home/t-jialianggu/.conda/envs/sglang-dev
OUT=/home/t-jialianggu/work/MOEresearch/results/2026-07-10_v9d_nsys/$NAME
mkdir -p $OUT

export HOME=/home/t-jialianggu CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=$CONDA
export PATH=$CONDA/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$CONDA/lib:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR=/tmp/tc_v9d_$MODEL_KEY; mkdir -p $TRITON_CACHE_DIR

echo "[1] launch server under nsys (delay=${DELAY}s duration=${DURATION}s)..."
$NSYS profile --trace=cuda --sample=none --delay=$DELAY --duration=$DURATION \
  --force-overwrite=true -o $OUT/timeline \
  $CONDA/bin/python -m sglang.launch_server \
    --model-path $MODELP --tokenizer-path $MODELP --trust-remote-code \
    --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 --chunked-prefill-size $CHUNK \
    --schedule-policy lpm --max-running-requests 128 \
    --context-length 32768 --moe-runner-backend triton \
    > $OUT/nsys_server.log 2>&1 &
NSYS_PID=$!
START=$(date +%s)

echo "[2] wait server ready..."
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; do
  sleep 5
  [ $(( $(date +%s) - START )) -gt 200 ] && { echo "  server never ready"; kill $NSYS_PID; exit 1; }
done
READY=$(( $(date +%s) - START ))
echo "  ready at ${READY}s (recording starts at ${DELAY}s)"

echo "[3] warmup..."
curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
  -d '{"text":"hello world","sampling_params":{"max_new_tokens":16}}' >/dev/null 2>&1 || true

echo "[4] wait until nsys recording window is active..."
while [ $(( $(date +%s) - START )) -lt $((DELAY + 2)) ]; do sleep 1; done

echo "[5] run bench (real arrival slowdown 1.0, 200 prompts)..."
BT0=$(date +%s.%N)
$CONDA/bin/python -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port $PORT --model $MODELP \
  --dataset-name mooncake --mooncake-workload toolagent \
  --num-prompts 200 --mooncake-slowdown-factor 1.0 \
  --output-file $OUT/bench_result.jsonl > $OUT/bench.log 2>&1
BT1=$(date +%s.%N)
echo "$(echo "$BT1 - $BT0" | bc)" > $OUT/bench_wall_seconds.txt
echo "  bench wall = $(cat $OUT/bench_wall_seconds.txt)s"

echo "[6] wait for nsys duration to elapse + report write..."
wait $NSYS_PID 2>/dev/null || true
sleep 3

echo "[7] kill any leftover server on port $PORT..."
SP=$(ps -ef | grep "port $PORT" | grep -v grep | awk '{print $2}'); for p in $SP; do kill $p 2>/dev/null || true; done
ls -la $OUT/timeline.nsys-rep 2>/dev/null && echo "REPORT OK" || echo "NO REPORT"
echo "DONE $NAME"
