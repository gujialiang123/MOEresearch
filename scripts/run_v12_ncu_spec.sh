#!/bin/bash
# v12: NCU on ngram-spec vs baseline decode -> measure how much SM idle (No-Eligible) spec reclaims
set -e
VARIANT=$1   # baseline | ngram
GPU=${GPUID:-4}
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
CONDA=/home/t-jialianggu/.conda/envs/sglang-dev
MODELP=/data/hf/models/Qwen3-30B-A3B-Instruct-2507
OUT=/home/t-jialianggu/work/MOEresearch/results/2026-07-15_v12_ncu_spec/$VARIANT
mkdir -p $OUT
SPEC=""
[ "$VARIANT" = "ngram" ] && SPEC="--speculative-algorithm NGRAM --speculative-num-draft-tokens 8 --speculative-num-steps 4"

cat > $OUT/inf.sh <<EOF
#!/bin/bash
export HOME=/home/t-jialianggu
export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_HOME=$CONDA
export TRITON_CACHE_DIR=/tmp/tc_v12_$VARIANT
mkdir -p \$TRITON_CACHE_DIR
export CPATH=$CONDA/targets/x86_64-linux/include:$CONDA/lib/python3.11/site-packages/nvidia/cublas/include:$CONDA/lib/python3.11/site-packages/nvidia/cuda_runtime/include
export LIBRARY_PATH=$CONDA/lib:$CONDA/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$CONDA/lib:$CONDA/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH
export PATH=$CONDA/bin:/usr/local/bin:/usr/bin:/bin
exec $CONDA/bin/python -m sglang.bench_one_batch \
  --model-path $MODELP --tokenizer-path $MODELP --trust-remote-code \
  --batch-size 32 --input-len 2700 --output-len 32 \
  --profile --profile-activities CUDA_PROFILER --profile-stage decode \
  --profile-filename-prefix $OUT/sgl --result-filename $OUT/res.jsonl --run-name v12_$VARIANT \
  --mem-fraction-static 0.85 --chunked-prefill-size 16384 --attention-backend fa3 --moe-runner-backend triton $SPEC
EOF
chmod +x $OUT/inf.sh

sudo -n $NCU --target-processes all --profile-from-start off --launch-count 24 \
  --kernel-name-base demangled \
  --kernel-name 'regex:fused_moe|nvjet|flash|cutlass|RMSNorm|act_and_mul|topk|conv1d|moe_sum|gemm|eagle|ngram|verify' \
  --section SpeedOfLight --section WarpStateStats --section SchedulerStats --section Occupancy \
  --force-overwrite --export $OUT/ncu -- $OUT/inf.sh > $OUT/bench.log 2>&1
echo "NCU exit=$? for $VARIANT"
sudo -n $NCU --import $OUT/ncu.ncu-rep --csv 2>/dev/null > $OUT/ncu_raw.csv
echo "CSV $(wc -l < $OUT/ncu_raw.csv) lines"
