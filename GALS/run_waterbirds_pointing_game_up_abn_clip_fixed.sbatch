#!/bin/bash -l
# Pointing-game eval on Waterbirds 95/100 for:
# - upweight (fixed checkpoints)
# - abn (fixed checkpoints)
# - clip_zs
# - clip_lr

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/pointing_game_up_abn_clip_fixed_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/pointing_game_up_abn_clip_fixed_%j.err

set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
LOG_ROOT="${LOG_ROOT:-/home/ryreu/guided_cnn/logsWaterbird}"
mkdir -p "$LOG_ROOT"

cd "$REPO_ROOT"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

UP95_CKPT="${UP95_CKPT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/retrain_upweight_wb95_fixed_21073587_trial_000/best_balanced_valacc_0.86_epoch_169.ckpt}"
UP100_CKPT="${UP100_CKPT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/retrain_upweight_wb100_fixed_21073587_trial_000/best_balanced_valacc_0.69_epoch_1.ckpt}"
ABN95_CKPT="${ABN95_CKPT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/retrain_abn_wb95_fixed_21073587_trial_000/best_balanced_valacc_0.87_epoch_7.ckpt}"
ABN100_CKPT="${ABN100_CKPT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/retrain_abn_wb100_fixed_21073587_trial_000/best_balanced_valacc_0.7_epoch_43.ckpt}"

for f in "$UP95_CKPT" "$UP100_CKPT" "$ABN95_CKPT" "$ABN100_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing checkpoint: $f" >&2
    exit 2
  fi
done

OUT_DIR="${OUT_DIR:-$LOG_ROOT/waterbirds_pointing_game_up_abn_clip_fixed_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}}"
echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Output: $OUT_DIR"
echo "Methods: upweight,abn,clip_zs,clip_lr"
which python

python -u waterbirds_pointing_game_eval.py \
  --methods upweight,abn,clip_zs,clip_lr \
  --datasets 95,100 \
  --split test \
  --target-mode label \
  --upweight95-ckpt "$UP95_CKPT" \
  --upweight100-ckpt "$UP100_CKPT" \
  --abn95-ckpt "$ABN95_CKPT" \
  --abn100-ckpt "$ABN100_CKPT" \
  --output-dir "$OUT_DIR"

