#!/bin/bash -l
# Curated saliency generation for Waterbirds 95/100 from existing checkpoints.
# Uses guided + vanilla + gals only (no CLIP ZS/LR).

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_curated_saliency_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_curated_saliency_%j.err

set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
cd "$REPO_ROOT"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

OUT_DIR="${OUT_DIR:-/home/ryreu/guided_cnn/logsWaterbird/waterbirds_curated_saliency_${SLURM_JOB_ID}}"
echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Output: $OUT_DIR"
which python

python -u run_waterbirds_curated_saliency_from_ckpts.py \
  --run-wb95 \
  --run-wb100 \
  --target-class label \
  --output-dir "$OUT_DIR"

