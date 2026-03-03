#!/bin/bash -l
# Quick ABN saliency generation on Waterbirds validation split.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/abn_quick_saliency_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/abn_quick_saliency_%j.err

set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
cd "$REPO_ROOT"

DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}"
ABN_CKPT="${ABN_CKPT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/retrain_abn_wb100_fixed_21073587_trial_000/best_balanced_valacc_0.7_epoch_43.ckpt}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
SPLIT="${SPLIT:-val}"
TARGET_MODE="${TARGET_MODE:-label}"
OUT_DIR="${OUT_DIR:-/home/ryreu/guided_cnn/logsWaterbird/abn_quick_saliency_${SLURM_JOB_ID}}"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "ABN ckpt: $ABN_CKPT"
echo "Num samples: $NUM_SAMPLES | split: $SPLIT | target: $TARGET_MODE"
echo "Output: $OUT_DIR"
which python

python -u run_waterbirds_abn_quick_saliency.py \
  --data-path "$DATA_PATH" \
  --abn-ckpt "$ABN_CKPT" \
  --num-samples "$NUM_SAMPLES" \
  --split "$SPLIT" \
  --target-mode "$TARGET_MODE" \
  --output-dir "$OUT_DIR"

