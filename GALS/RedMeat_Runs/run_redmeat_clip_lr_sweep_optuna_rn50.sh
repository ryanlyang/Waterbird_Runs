#!/bin/bash -l
# Explicit RN50 CLIP+LR sweep for RedMeat.
# This is a thin wrapper around the main CLIP+LR sweep script with CLIP_MODEL forced to RN50.
#
# Usage:
#   sbatch RedMeat_Runs/run_redmeat_clip_lr_sweep_optuna_rn50.sh

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_rn50_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_rn50_sweep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Force RN50 backbone for CLIP feature extraction.
export CLIP_MODEL=RN50

# Use RN50-specific default output names unless caller overrides.
export OUT_CSV="${OUT_CSV:-/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_rn50_sweep_${SLURM_JOB_ID}.csv}"
export POST_OUT_CSV="${POST_OUT_CSV:-/home/ryreu/guided_cnn/logsRedMeat/redmeat_clip_lr_rn50_best5_${SLURM_JOB_ID}.csv}"

exec bash "${SCRIPT_DIR}/run_redmeat_clip_lr_sweep_optuna.sh"

