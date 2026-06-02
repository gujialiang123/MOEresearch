#!/usr/bin/env bash
# Run hardware-view profile on the 2 REAL kernel-swap MoE variants × R8.
# C7 = flashinfer_cutlass — real CUTLASS MoE kernel from flashinfer
# C8 = triton_kernel       — triton_kernels library's matmul_ogs
set +e
eval "$(conda shell.bash hook)" && conda activate sglang-dev
cd /home/t-jialianggu/work/EndtoEnd-auto-optimization

wait_port_free() {
  local port=$1
  for i in $(seq 1 30); do
    if ! ss -ltn 2>/dev/null | grep -q ":${port} "; then return 0; fi
    sleep 1
  done
  return 1
}

WORKLOAD="regime_scout/candidates_regime_study/R8_prefix_sharing.yaml"

for cfg in configs/moe_variants/C7_moe_flashinfer_cutlass.yaml \
           configs/moe_variants/C8_moe_triton_kernel.yaml; do
  TAG=$(basename "$cfg" .yaml)
  OUT="experiments/tmp/moe_opt_levels/${TAG}"
  echo "================ ${TAG} ================"
  rm -rf "$OUT"
  wait_port_free 30000 || { echo "[batch] port busy"; continue; }
  python scripts/regime_study/run_hw_view.py \
    --config "$cfg" \
    --workload "$WORKLOAD" \
    --out-dir "$OUT" \
    --gpu 0 \
    --profile-num-steps 10 \
    --warmup-requests 8 \
    --server-start-timeout 480 || echo "WARN: $TAG rc=$?"
  sleep 5
done
echo "ALL DONE"
