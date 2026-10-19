#!/bin/sh
#SBATCH --job-name=moe-state-study
#SBATCH --partition=contrib-B200
#SBATCH --account=kerenzhou
#SBATCH --qos=kzhou6
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/jgu7/moe-state-study/slurm/%x-%j.log

set -eu
STUDY_RUN_DIR=${1:?Usage: sbatch --dependency=afterany:OLD_JOB scripts/state_study_resume.sbatch RUN OLD_JOB}
STUDY_RUN_DIR=${STUDY_RUN_DIR%/}
STUDY_PREVIOUS_JOB=${2:?Pass the exact predecessor job ID as the second script argument}
case "$STUDY_RUN_DIR" in
    /scratch/jgu7/moe-state-study/runs/*) ;;
    *) echo 'Expected an existing study run directory under scratch' >&2; exit 2 ;;
esac
test -f "$STUDY_RUN_DIR/protocol.json"
export HF_HOME=/scratch/jgu7/cache/huggingface
export UV_CACHE_DIR=/scratch/jgu7/cache/uv
export TMPDIR=/scratch/jgu7/tmp
export XDG_CACHE_HOME=/scratch/jgu7/cache
export MPLCONFIGDIR=/scratch/jgu7/cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
cd /home/jgu7/work/MOEresearch
STUDY_PYTHON=/scratch/jgu7/envs/moe-state/bin/python
STUDY_SLURM_LOG="/scratch/jgu7/moe-state-study/slurm/${SLURM_JOB_NAME:-moe-state-study}-${SLURM_JOB_ID}.log"
STUDY_MAIN_EXIT=0
"$STUDY_PYTHON" -u scripts/launch_state_study_resume.py \
    --run-dir "$STUDY_RUN_DIR" --previous-job-id "$STUDY_PREVIOUS_JOB" \
    --log "$STUDY_SLURM_LOG" || STUDY_MAIN_EXIT=$?

# The shell must remain alive to export the final measured report after the
# exec-replaced helper/auto process returns. Preserve the scientific exit code.
STUDY_RUN_NAME=${STUDY_RUN_DIR##*/}
STUDY_REPORT_DIR="/scratch/jgu7/moe-state-study/reports/$STUDY_RUN_NAME"
STUDY_REPORT_EXIT=0
STUDY_PLOT_EXIT=0
STUDY_STATUS_EXIT=0
mkdir -p "$STUDY_REPORT_DIR" || STUDY_REPORT_EXIT=$?
if [ "$STUDY_REPORT_EXIT" -eq 0 ]; then
    "$STUDY_PYTHON" scripts/summarize_state_study_results.py \
        --run-dir "$STUDY_RUN_DIR" --output-dir "$STUDY_REPORT_DIR" \
        --verify-diagnostic-hashes \
        >"$STUDY_REPORT_DIR/report-${SLURM_JOB_ID}.log" 2>&1 || STUDY_REPORT_EXIT=$?
    "$STUDY_PYTHON" scripts/plot_state_calibration_study.py \
        --run-dir "$STUDY_RUN_DIR" --scope formal \
        --out-dir "$STUDY_REPORT_DIR/figures" \
        >"$STUDY_REPORT_DIR/plots-${SLURM_JOB_ID}.log" 2>&1 || STUDY_PLOT_EXIT=$?
    STUDY_CODES_FILE="$STUDY_REPORT_DIR/batch-exit-codes-${SLURM_JOB_ID}.json"
    printf '{"scientific_exit_code":%s,"report_exit_code":%s,"plot_exit_code":%s}\n' \
        "$STUDY_MAIN_EXIT" "$STUDY_REPORT_EXIT" "$STUDY_PLOT_EXIT" \
        >"$STUDY_CODES_FILE.next" || STUDY_STATUS_EXIT=$?
    if [ "$STUDY_STATUS_EXIT" -eq 0 ]; then
        mv "$STUDY_CODES_FILE.next" "$STUDY_CODES_FILE" || STUDY_STATUS_EXIT=$?
    fi
else
    STUDY_PLOT_EXIT=125
    STUDY_STATUS_EXIT=125
fi
printf 'Scientific main exit=%s; independent report exit=%s; plots exit=%s; status publication exit=%s; report directory=%s\n' \
    "$STUDY_MAIN_EXIT" "$STUDY_REPORT_EXIT" "$STUDY_PLOT_EXIT" "$STUDY_STATUS_EXIT" "$STUDY_REPORT_DIR"
if [ "$STUDY_REPORT_EXIT" -ne 0 ] || [ "$STUDY_PLOT_EXIT" -ne 0 ] || [ "$STUDY_STATUS_EXIT" -ne 0 ]; then
    echo 'Derived report/plot publication failed; inspect the separate postprocessing logs. No result was synthesized.' >&2
fi
exit "$STUDY_MAIN_EXIT"
