#!/usr/bin/env bash
set -euo pipefail

# Configurable arguments with default fallbacks
ARCH="${1:-wrn28-2}"
DATASET="${2:-CIFAR10}"
SHADOWS="${3:-16}"
GPU="${4:-:0}"

# Unique experiment ID incorporating the shadow count to prevent collisions
EXP_ID="${ARCH}_${DATASET}_${SHADOWS}shadows"
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${EXP_ID}.log"

echo "================================================================="
echo " Starting Training Pipeline"
echo " Experiment ID : ${EXP_ID}"
echo " Architecture  : ${ARCH}"
echo " Dataset       : ${DATASET}"
echo " Shadow Models : ${SHADOWS}"
echo " GPU Device    : ${GPU}"
echo " Log file      : ${LOG_FILE}"
echo "================================================================="

# Run the complete attack pipeline and pipe stdout/stderr to console and log file
python -m loss_traces.run_attack_pipeline \
    --exp_id "${EXP_ID}" \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --n-shadows "${SHADOWS}" \
    --full \
    --gpu "${GPU}" 2>&1 | tee "${LOG_FILE}"

echo "================================================================="
echo " Training and attack computation complete for ${EXP_ID}!"
echo " Artifacts saved under:"
echo "   Models    : ./data/models/${EXP_ID}/"
echo "   Losses    : ./data/losses/${EXP_ID}_target.pq"
echo "   CSG Norms : ./data/csg_norms/${EXP_ID}_target.pq"
echo "================================================================="
