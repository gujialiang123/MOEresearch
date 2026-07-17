#!/usr/bin/env bash
# Run hardware-view profile on 7 MoE optimization-knob variants × R8 regime.
# Outputs to experiments/tmp/moe_opt_levels/<TAG>/ then aggregated by
# scripts/regime_study/aggregate_moe_opt_levels.py
set +e
eval "$(conda shell.bash hook)" && conda activate sglang-dev
cd /home/t-jialianggu/work/MOEresearch

wait_port_free() {
  local port=$1
  for i in $(seq 1 30); do
    if ! ss -ltn 2>/dev/null | grep -q ":${port} "; then return 0; fi
    sleep 1
  done
  return 1
}

WORKLOAD="regime_scout/candidates_regime_study/R8_prefix_sharing.yaml"

for cfg in configs/moe_variants/C0_baseline.yaml \
           configs/moe_variants/C1_torch_compile.yaml \
           configs/moe_variants/C2_no_cuda_graph.yaml \
           configs/moe_variants/C3_chunked_prefill.yaml \
           configs/moe_variants/C4_moe_cutlass.yaml \
           configs/moe_variants/C5_attn_flashinfer.yaml \
           configs/moe_variants/C6_piecewise_cuda_graph.yaml; do
  TAG=$(basename "$cfg" .yaml)
  OUT="experiments/tmp/moe_opt_levels/${TAG}"
  echo "================ ${TAG} ================"
  rm -rf "$OUT"
  wait_port_free 30000 || { echo "[batch] port busy"; continue; }
  if [ "$TAG" = "C1_torch_compile" ]; then START_TO=600; else START_TO=300; fi
  python scripts/regime_study/run_hw_view.py \
    --config "$cfg" \
    --workload "$WORKLOAD" \
    --out-dir "$OUT" \
    --gpu 0 \
    --profile-num-steps 10 \
    --warmup-requests 8 \
    --server-start-timeout $START_TO || echo "WARN: $TAG rc=$?"
  sleep 5
done
echo "ALL DONE"
