#!/usr/bin/env bash
set -euo pipefail

# Configurable arguments matching the training setup
ARCH="${1:-wrn28-2}"
DATASET="${2:-CIFAR10}"
SHADOWS="${3:-16}"
DEVICE="${4:-cuda:0}"
RHO="${5:-1.0}"
INITIAL_LR="${6:-0.1}"

# Target the exact experiment ID created during training
EXP_ID="${ARCH}_${DATASET}_${SHADOWS}shadows"
OUTPUT_DIR="./results_analysis/${EXP_ID}"
LOG_DIR="./logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_${EXP_ID}.log"

echo "================================================================="
echo " Starting Artifact Evaluation"
echo " Experiment ID  : ${EXP_ID}"
echo " Dataset        : ${DATASET}"
echo " Architecture   : ${ARCH}"
echo " Shadow Count   : ${SHADOWS}"
echo " Compute Device : ${DEVICE}"
echo " Output Dir     : ${OUTPUT_DIR}"
echo " Log File       : ${LOG_FILE}"
echo "================================================================="

# Execute standalone analysis
python main_results.py \
    --exp_id "${EXP_ID}" \
    --dataset "${DATASET}" \
    --arch "${ARCH}" \
    --n_shadows "${SHADOWS}" \
    --device "${DEVICE}" \
    --rho "${RHO}" \
    --initial_lr "${INITIAL_LR}" \
    --output_dir "${OUTPUT_DIR}" 2>&1 | tee "${LOG_FILE}"

echo "================================================================="
echo " Evaluation complete for ${EXP_ID}!"
echo " Results stored in: ${OUTPUT_DIR}"
echo "   - table1_lira_precision_recall.csv"
echo "   - table5_union_precision_recall.csv"
echo "   - all_sample_scores.csv"
echo "   - plots/precision_vs_fpr_comparison.png"
echo "   - plots/late_csg_ablation.png"
echo "================================================================="
