#!/bin/bash
# Drive 2 Optuna studies sequentially on one GPU.
# Usage: scripts/autotune_two_regimes.sh <gpu_id> <port> <regime_1> <regime_2> <n_trials>
set -euo pipefail

GPU="$1"; PORT="$2"; REGIME_A="$3"; REGIME_B="$4"; N="$5"
REPO="/home/t-jialianggu/work/EndtoEnd-auto-optimization"
PY="/home/t-jialianggu/.conda/envs/sglang-dev/bin/python"

cd "$REPO"

for REGIME in "$REGIME_A" "$REGIME_B"; do
  OUT_DIR="results/2026-06-25_autotuning/per_regime/${REGIME}_gpu${GPU}"
  echo "================================================================"
  echo "[$(date)] Starting study for $REGIME on GPU $GPU port $PORT"
  echo "Output: $OUT_DIR"
  echo "================================================================"
  "$PY" -m harness.autotune \
      --template-spec bench-specs/sglang-triton-bf16-baseline.yaml \
      --target-regime "$REGIME" \
      --gpu-id "$GPU" \
      --port "$PORT" \
      --n-trials "$N" \
      --out-dir "$OUT_DIR" \
      --seed 2026
  echo "[$(date)] Done with $REGIME"
done

echo "[$(date)] All studies on GPU $GPU complete."
