#!/bin/bash
# Per-regime NCU runner — uses sglang.bench_one_batch with --profile-activities CUDA_PROFILER
# so that ncu --profile-from-start off triggers on cudaProfilerStart inside sglang.
# This is MUCH faster than wrapping the full HTTP server.
#
# Usage: bench_ncu_one_regime.sh <regime_id>
set -e
REGIME=$1
[ -z "$REGIME" ] && echo "usage: $0 <regime_id>" && exit 1

ENV=/home/t-jialianggu/.conda/envs/sglang-dev
NCU=/home/t-chendili/.conda/pkgs/nsight-compute-2026.1.1.2-h1ff7d1d_0/bin/ncu
REPO=/home/t-jialianggu/work/EndtoEnd-auto-optimization
OUT_DIR=$REPO/results/2026-06-09_sglang_triton_sweep/ncu/$REGIME
mkdir -p $OUT_DIR

# Read regime params
python3 -c "
import yaml
r = yaml.safe_load(open('$REPO/regimes/qwen3_30b_moe_sglang_perf_sweep.yaml'))['regimes']['$REGIME']
print(f'BATCH={r[\"concurrency\"]}')
print(f'INLEN={r[\"prompt_words\"] * 2}')
print(f'OUTLEN={r[\"max_new\"]}')
print(f'STAGE={\"decode\" if r[\"max_new\"] >= r[\"prompt_words\"] // 4 else \"prefill\"}')
" > /tmp/regime_${REGIME}_params.sh
source /tmp/regime_${REGIME}_params.sh
echo "[$(date +%H:%M:%S)] regime=$REGIME BATCH=$BATCH INLEN=$INLEN OUTLEN=$OUTLEN STAGE=$STAGE" >&2

# Wrapper script that sudo'd ncu invokes — sets env then calls sglang.
# This is necessary because sudo -n strips env (whitelist doesn't permit -E).
WRAPPER=/tmp/sglang_bench_inner_${REGIME}.sh
cat > $WRAPPER << EOF
#!/bin/bash
export HOME=/home/t-jialianggu
export CUDA_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR=/tmp/sglang_triton_cache_ncu
mkdir -p \$TRITON_CACHE_DIR
export CUDA_HOME=$ENV
export CPATH=$ENV/targets/x86_64-linux/include:$ENV/lib/python3.11/site-packages/nvidia/cublas/include:$ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/include
export LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib
export PATH=$ENV/bin:/usr/local/bin:/usr/bin:/bin
exec $ENV/bin/python -m sglang.bench_one_batch \\
  --model-path /data/hf/models/Qwen3-30B-A3B-Instruct-2507 \\
  --tokenizer-path /data/hf/models/Qwen3-30B-A3B-Instruct-2507 \\
  --trust-remote-code \\
  --moe-runner-backend triton \\
  --disable-cuda-graph \\
  --batch-size $BATCH \\
  --input-len $INLEN \\
  --output-len $OUTLEN \\
  --profile \\
  --profile-activities CUDA_PROFILER \\
  --profile-stage $STAGE \\
  --profile-filename-prefix $OUT_DIR/sglang_bench_${REGIME} \\
  --result-filename $OUT_DIR/bench_one_batch_result.jsonl \\
  --run-name ${REGIME}_ncu_full
EOF
chmod +x $WRAPPER

# Launch ncu wrap of wrapper:
# --profile-from-start off → wait for cudaProfilerStart inside sglang
# --set full → all metric sections
# --kernel-name regex:.* → no kernel filter
sudo -n $NCU \
  --target-processes all \
  --profile-from-start off \
  --launch-count 50 \
  --set full \
  --kernel-name-base demangled \
  --kernel-name "regex:.*" \
  --force-overwrite \
  --export $OUT_DIR/${REGIME}_ncu \
  -- $WRAPPER \
  > $OUT_DIR/bench.log 2>&1

echo "[$(date +%H:%M:%S)] regime=$REGIME completed" >&2

# Export to CSV
sudo -n $NCU --import $OUT_DIR/${REGIME}_ncu.ncu-rep --csv \
  > $OUT_DIR/ncu_raw.csv 2>$OUT_DIR/ncu_import.err || true
echo "[$(date +%H:%M:%S)] regime=$REGIME — csv exported $(wc -l < $OUT_DIR/ncu_raw.csv) lines" >&2

ls -la $OUT_DIR/
