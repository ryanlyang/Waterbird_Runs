#!/bin/bash -l
# Train fixed AFR (WB95+WB100) then generate curated val saliency for:
# upweight + abn + afr (no CLIP methods).

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_curated_saliency_up_abn_afr_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_curated_saliency_up_abn_afr_%j.err

set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
cd "$REPO_ROOT"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_DISABLED=true

OUT_DIR="${OUT_DIR:-/home/ryreu/guided_cnn/logsWaterbird/waterbirds_curated_saliency_up_abn_afr_${SLURM_JOB_ID}}"
echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Output: $OUT_DIR"
which python

python -u run_waterbirds_curated_saliency_up_abn_afr.py \
  --run-wb95 \
  --run-wb100 \
  --target-mode label \
  --wb95-afr-gamma 11.0 \
  --wb95-afr-reg 0.0 \
  --wb95-afr-seed 2 \
  --wb100-afr-gamma 4.0 \
  --wb100-afr-reg 0.0 \
  --wb100-afr-seed 2 \
  --output-dir "$OUT_DIR"
