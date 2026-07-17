#!/usr/bin/env bash
# Wrap sglang server under nsys, run a single regime workload, kill, export sqlite.
#
# Usage: nsys_on_universal_config.sh <regime_id> <gpu_id> <port>
#   regime_id: one of R_short_decode / R_medium_balanced / R_long_prefill / R_concurrent_decode
#   gpu_id:    GPU index
#   port:      port for sglang server
#
# Output: results/2026-06-29_ncu_validation/<regime_id>/
#   - server.log
#   - workload.log
#   - profile.nsys-rep    (raw)
#   - profile.sqlite      (queryable)
#
# Notes:
#   - Uses sglang's CURRENT config (the universal autotuned config — defaults
#     are essentially identical so we use defaults + cudagraph + cutlass)
#   - Runs workload AFTER warmup so we capture steady-state kernels
set -euo pipefail

REGIME=${1:?usage: $0 <regime_id> <gpu_id> <port>}
GPU=${2:?usage: $0 <regime_id> <gpu_id> <port>}
PORT=${3:?usage: $0 <regime_id> <gpu_id> <port>}

REPO=/home/t-jialianggu/work/MOEresearch
NSYS=/home/t-chendili/cuda/12.6/bin/nsys
PY=/home/t-jialianggu/.conda/envs/sglang-dev/bin/python

OUT_DIR=$REPO/results/2026-06-29_ncu_validation/$REGIME
mkdir -p "$OUT_DIR"

# Workload script path
WORKLOAD_PY=$REPO/scripts/ncu_run_one_regime_workload.py

# Patch the workload to use our port (default is 30000)
TMP_WL=$(mktemp /tmp/nsys_workload_XXX.py)
sed "s|http://127.0.0.1:30000|http://127.0.0.1:$PORT|g" "$WORKLOAD_PY" > "$TMP_WL"

echo "[$(date +%H:%M:%S)] $REGIME on GPU $GPU port $PORT"
echo "[$(date +%H:%M:%S)] launching sglang server under nsys (delay=720s, capture duration=90s)"

# Set env for sglang (deep_gemm needs CUDA_HOME)
ENV=/home/t-jialianggu/.conda/envs/sglang-dev
export CUDA_HOME=$ENV
export CPATH=$ENV/targets/x86_64-linux/include:$ENV/lib/python3.11/site-packages/nvidia/cublas/include:$ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/include
export LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export TRITON_CACHE_DIR=/tmp/sglang_triton_cache_nsys
mkdir -p $TRITON_CACHE_DIR

# HF cache redirect (shared cache is owned by another user)
export HF_HOME=/data/hf/gujialiang123/hf_cache
export HF_HUB_CACHE=/data/hf/gujialiang123/hf_cache/hub

# Universal config from Optuna study. Use --delay to skip model loading +
# cudagraph capture (which take ~10 min under nsys instrumentation).
# Then capture for 90s while workload runs.
CUDA_VISIBLE_DEVICES=$GPU "$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cuda-memory-usage=true \
  --delay=720 \
  --duration=90 \
  --force-overwrite=true \
  --output="$OUT_DIR/profile" \
  "$PY" -m sglang.launch_server \
    --model-path /data/hf/models/Qwen3-30B-A3B-Instruct-2507 \
    --served-model-name qwen3-30b-a3b-moe \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    --context-length 32768 \
    --max-running-requests 32 \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 16384 \
    --schedule-policy fcfs \
    --moe-runner-backend flashinfer_cutlass \
    --trust-remote-code \
    --log-level info \
    > "$OUT_DIR/server.log" 2>&1 &
NSYS_PID=$!

# We didn't use --capture-range above (gen "all" run instead).
# Actually use simpler approach: nsys captures everything from launch,
# we run workload then SIGINT.
# Rebuild the command without capture-range since we want full profile.

# Wait for server ready. Without nsys: ~80s. With nsys instrumentation: 10+ min.
# We set --delay=720 (12 min) for nsys; align our wait with that, then run workload.
echo "[$(date +%H:%M:%S)] waiting for /health (up to 15 min)"
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] server ready after ${i}*5s"
    break
  fi
  sleep 5
done

if ! curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] FAIL: server never came up after 15 min"
  tail -30 "$OUT_DIR/server.log"
  exit 1
fi

# At this point nsys --delay may or may not have triggered. Run the workload —
# the --duration=90 will start capturing as soon as --delay elapses and stop
# 90s later. Workload should overlap with that window.
#
# We run the workload TWICE to ensure good coverage:
#   1st pass: kernel JIT/autotune (likely already done from /health probes)
#   2nd pass: steady-state, this is what we want captured
echo "[$(date +%H:%M:%S)] workload pass 1 (warmup)"
"$PY" "$TMP_WL" "$REGIME" 2>&1 | tee "$OUT_DIR/workload_pass1.log"

echo "[$(date +%H:%M:%S)] workload pass 2 (target capture)"
"$PY" "$TMP_WL" "$REGIME" 2>&1 | tee "$OUT_DIR/workload_pass2.log"

# nsys will exit after its --duration=90 elapses; just wait for it.
echo "[$(date +%H:%M:%S)] waiting for nsys to exit (auto after --duration=90 expires)"
for i in $(seq 1 60); do
  if ! kill -0 $NSYS_PID 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] nsys exited"
    break
  fi
  sleep 5
done

# Kill any leftover sglang procs
pgrep -f "sglang.launch_server.*$PORT" | while read pid; do
  echo "[$(date +%H:%M:%S)] killing leftover sglang pid=$pid"
  kill -TERM $pid 2>/dev/null || true
done

sleep 5

# Export sqlite if .nsys-rep exists
if [ -f "$OUT_DIR/profile.nsys-rep" ]; then
  REP_SIZE=$(stat -c%s "$OUT_DIR/profile.nsys-rep")
  echo "[$(date +%H:%M:%S)] profile.nsys-rep size: $((REP_SIZE/1024/1024)) MB"
  echo "[$(date +%H:%M:%S)] exporting to sqlite"
  "$NSYS" export --type sqlite --force-overwrite=true \
    --output "$OUT_DIR/profile.sqlite" "$OUT_DIR/profile.nsys-rep" 2>&1 | tail -5
  echo "[$(date +%H:%M:%S)] DONE $REGIME"
  ls -la "$OUT_DIR/"
else
  echo "[$(date +%H:%M:%S)] WARNING: profile.nsys-rep not produced"
fi

rm -f "$TMP_WL"
