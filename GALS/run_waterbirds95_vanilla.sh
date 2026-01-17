#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=3-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_vanilla_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds95_vanilla_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate learntolook

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS
DATA_ROOT=/home/ryreu/guided_cnn/waterbirds
DATA_DIR=waterbird_complete95_forest2water2

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT/$DATA_DIR"
which python

srun --unbuffered python -u main.py \
  --config configs/waterbirds_95_vanilla.yaml \
  --dryrun \
  DATA.ROOT="$DATA_ROOT" \
  DATA.WATERBIRDS_DIR="$DATA_DIR"
