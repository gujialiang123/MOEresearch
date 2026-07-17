#!/bin/bash
# v11-B2: multi-stream utilization sweep — prove serving idle is recoverable by
# concurrent independent request streams (multi-tenancy). One server, N parallel
# toolagent streams (real arrival, slowdown 1.0). nsys measures GPU busy/idle.
set -e
MODEL_KEY=${MK:-lfm}
GPU=${GPUID:-4}
PORT=${PORTN:-31250}
STREAMS=${NSTREAMS:-1}       # number of concurrent bench_serving streams
DELAY=${DELAYS:-95}
DURATION=${DURS:-120}
if [ "$MODEL_KEY" = "lfm" ]; then
  MODELP=/data/hf/LFM2.5-8B-A1B; CHUNK=4096; NAME=lfm2.5-8b-a1b
else
  MODELP=/data/hf/models/Qwen3-30B-A3B-Instruct-2507; CHUNK=16384; NAME=qwen3-30b-a3b-bf16
fi
NSYS=/home/t-chendili/cuda/12.9/bin/nsys
CONDA=/home/t-jialianggu/.conda/envs/sglang-dev
OUT=/home/t-jialianggu/work/MOEresearch/results/2026-07-15_v11b2_multistream/$NAME/streams$STREAMS
mkdir -p $OUT
export HOME=/home/t-jialianggu CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=$CONDA
export PATH=$CONDA/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$CONDA/lib:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR=/tmp/tc_v11_$MODEL_KEY; mkdir -p $TRITON_CACHE_DIR

echo "[1] launch server under nsys (delay=$DELAY dur=$DURATION, streams=$STREAMS)..."
$NSYS profile --trace=cuda --sample=none --delay=$DELAY --duration=$DURATION \
  --force-overwrite=true -o $OUT/timeline \
  $CONDA/bin/python -m sglang.launch_server \
    --model-path $MODELP --tokenizer-path $MODELP --trust-remote-code \
    --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 --chunked-prefill-size $CHUNK \
    --schedule-policy lpm --max-running-requests 256 \
    --context-length 32768 --moe-runner-backend triton \
    > $OUT/server.log 2>&1 &
NSYS_PID=$!; START=$(date +%s)

echo "[2] wait ready..."
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; do
  sleep 5; [ $(( $(date +%s) - START )) -gt 200 ] && { echo "server timeout"; kill $NSYS_PID; exit 1; }
done
echo "  ready at $(( $(date +%s) - START ))s"
curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' -d '{"text":"hi","sampling_params":{"max_new_tokens":8}}' >/dev/null 2>&1 || true

echo "[3] wait for nsys window..."
while [ $(( $(date +%s) - START )) -lt $((DELAY + 2)) ]; do sleep 1; done

echo "[4] launch $STREAMS parallel toolagent streams..."
PIDS=""
for i in $(seq 1 $STREAMS); do
  $CONDA/bin/python -m sglang.bench_serving --backend sglang \
    --host 127.0.0.1 --port $PORT --model $MODELP \
    --dataset-name mooncake --mooncake-workload toolagent \
    --num-prompts 200 --mooncake-slowdown-factor 1.0 --seed $((i*7)) \
    --output-file $OUT/bench_stream$i.jsonl > $OUT/bench_stream$i.log 2>&1 &
  PIDS="$PIDS $!"
done
echo "  stream pids:$PIDS"
for p in $PIDS; do wait $p 2>/dev/null || true; done
echo "  all streams done"

echo "[5] wait nsys duration + report..."
wait $NSYS_PID 2>/dev/null || true; sleep 3
echo "[6] kill server..."
SP=$(ps -ef | grep "port $PORT" | grep -v grep | awk '{print $2}'); for p in $SP; do kill $p 2>/dev/null || true; done
ls -la $OUT/timeline.nsys-rep 2>/dev/null && echo "REPORT OK" || echo "NO REPORT"
echo "DONE $NAME streams=$STREAMS"
