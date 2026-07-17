#!/bin/bash
# Run all 4 regimes through bench_ncu_one_regime.sh sequentially.
# Logs to /tmp/ncu_batch_<regime>.log per regime.
# Designed to run as detached background job (nohup + &).
set -uo pipefail
SCRIPT=/home/t-jialianggu/work/MOEresearch/scripts/bench_ncu_one_regime.sh
LOGDIR=/tmp/ncu_batch
mkdir -p $LOGDIR
echo "[$(date)] starting NCU batch" > $LOGDIR/index.log

REGIMES=(R_long_prefill R_concurrent_decode R_medium_balanced R_short_decode)
# Order: longest prefill first (shortest ncu wall), short_decode last (slowest)

for REGIME in "${REGIMES[@]}"; do
    echo "[$(date +%H:%M:%S)] >>> START $REGIME" | tee -a $LOGDIR/index.log
    bash $SCRIPT $REGIME > $LOGDIR/$REGIME.log 2>&1 || true
    echo "[$(date +%H:%M:%S)] <<< END $REGIME (exit $?)" | tee -a $LOGDIR/index.log
done

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a $LOGDIR/index.log
