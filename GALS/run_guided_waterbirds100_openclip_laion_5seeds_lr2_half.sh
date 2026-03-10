#!/bin/bash -l
# WB100 local sensitivity: lr2_mult/2 (from 0.123 -> 0.0615), others fixed.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/guided100_sens_lr2half_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/guided100_sens_lr2half_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

export LR2_MULT=0.0615
export SUMMARY_CSV="${SUMMARY_CSV:-/home/ryreu/guided_cnn/logsWaterbird/guided100_sens_lr2half_${SLURM_JOB_ID}.csv}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_guided_waterbirds100_openclip_laion_5seeds.sh"
