#!/bin/bash
# Per-regime NCU runner — launches sglang under ncu wrap, runs ONE regime, kills.
# Usage: ncu_one_regime.sh <regime_id>
set -e
REGIME=$1
[ -z "$REGIME" ] && echo "usage: $0 <regime_id>" && exit 1

ENV=/home/t-jialianggu/.conda/envs/sglang-dev
NCU=/home/t-chendili/.conda/pkgs/nsight-compute-2026.1.1.2-h1ff7d1d_0/bin/ncu
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
OUT_DIR=$REPO/results/2026-06-09_sglang_triton_sweep/ncu/$REGIME
mkdir -p $OUT_DIR

# Env that sglang needs (build paths for triton JIT etc.)
export CUDA_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR=/tmp/sglang_triton_cache_ncu
mkdir -p $TRITON_CACHE_DIR
export CPATH=$ENV/targets/x86_64-linux/include:$ENV/lib/python3.11/site-packages/nvidia/cublas/include:$ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/include
export LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$ENV/bin:$PATH

cd $REPO

echo "[$(date +%H:%M:%S)] regime=$REGIME — launching server under ncu" >&2

# Launch sglang under ncu wrap (sudo). full metrics, no kernel filter,
# skip first 10000 launches (warmup), capture next 50 launches.
# Writes CSV to OUT_DIR/ncu_raw.csv
sudo -n $NCU \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "regex:.*" \
  --launch-skip 5000 \
  --launch-count 30 \
  --set full \
  --csv \
  --log-file $OUT_DIR/ncu_raw.csv \
  --force-overwrite \
  -- $ENV/bin/python -m sglang.launch_server \
    --model-path /data/hf/models/Qwen3-30B-A3B-Instruct-2507 \
    --served-model-name qwen3-30b-a3b-moe \
    --host 127.0.0.1 --port 30000 \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    --context-length 32768 \
    --max-running-requests 32 \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 16384 \
    --moe-runner-backend triton \
    --disable-cuda-graph \
    --trust-remote-code \
    --log-level info \
  > $OUT_DIR/server.log 2>&1 &
NCU_PID=$!
echo "[$(date +%H:%M:%S)] ncu PID=$NCU_PID — waiting for server ready" >&2

# Wait for server ready (up to 8 min)
for i in $(seq 1 32); do
  if curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] server ready after ${i}*15s" >&2
    break
  fi
  sleep 15
done

if ! curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] FAIL: server never came up" >&2
  tail -20 $OUT_DIR/server.log >&2
  python3 -c "import os, signal; os.kill($NCU_PID, signal.SIGTERM)" 2>/dev/null
  exit 1
fi

# Run ONE regime
echo "[$(date +%H:%M:%S)] running workload for $REGIME" >&2
python3 $REPO/scripts/ncu_run_one_regime_workload.py $REGIME

# Send SIGINT to ncu (it'll flush remaining samples and exit)
echo "[$(date +%H:%M:%S)] SIGINT to ncu — flush" >&2
python3 -c "
import os, signal, subprocess
try:
    out = subprocess.check_output(['pgrep', '-f', 'ncu.*sglang.launch_server.*30000'], text=True)
    for l in out.strip().split('\n'):
        try: os.kill(int(l), signal.SIGINT); print('SIGINT', l)
        except: pass
except: pass
"
# wait up to 5 min for ncu to flush and exit
for i in $(seq 1 60); do
  if ! pgrep -f "ncu.*sglang.launch_server.*30000" > /dev/null; then
    echo "[$(date +%H:%M:%S)] ncu exited cleanly" >&2
    break
  fi
  sleep 5
done

# Kill any leftover sglang too
python3 -c "
import os, signal, subprocess
for pat in ['sglang.launch_server', 'ncu --target']:
    try:
        out = subprocess.check_output(['pgrep', '-f', pat], text=True)
        for l in out.strip().split('\n'):
            try: os.kill(int(l), signal.SIGTERM)
            except: pass
    except: pass
"
sleep 10

echo "[$(date +%H:%M:%S)] DONE $REGIME" >&2
ls -la $OUT_DIR/
