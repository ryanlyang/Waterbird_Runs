#!/bin/bash -l
# AFR-style DecoyMNIST LeNet fixed run (5 seeds by default).
#
# Usage:
#   sbatch run_decoymnist_afr_fixed.sbatch
#
# Optional overrides:
#   sbatch --export=ALL,GAMMA=4.0,REG_COEFF=0.0,SEEDS=0,1,2,3,4 run_decoymnist_afr_fixed.sbatch

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_afr_fixed_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_afr_fixed_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
CONDA_ENV="${CONDA_ENV:-learntolook}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"

SEEDS="${SEEDS:-0,1,2,3,4}"
SPLIT_SEED="${SPLIT_SEED:-0}"

# LeNet CDEP baseline optimizer settings.
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# AFR fixed hyperparameters from your prior selection.
GAMMA="${GAMMA:-4.0}"
REG_COEFF="${REG_COEFF:-0.0}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-19}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-500}"
STAGE2_LR="${STAGE2_LR:-0.001}"
STAGE2_WEIGHT_DECAY="${STAGE2_WEIGHT_DECAY:-0.0}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:--1}"

mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"

OUT_CSV="${LOG_DIR}/decoy_afr_fixed_${SLURM_JOB_ID}.csv"

echo "[$(date)] Host: $(hostname)"
which python
echo "Repo: ${REPO_ROOT}"
echo "Data: ${DATA_PATH}"
echo "Seeds: ${SEEDS}"
echo "Stage1: epochs=${STAGE1_EPOCHS} lr=${LR} wd=${WEIGHT_DECAY}"
echo "Stage2: epochs=${STAGE2_EPOCHS} lr=${STAGE2_LR} wd=${STAGE2_WEIGHT_DECAY} batch=${STAGE2_BATCH_SIZE}"
echo "AFR: gamma=${GAMMA} reg_coeff=${REG_COEFF}"
echo "Output CSV: ${OUT_CSV}"

python -u run_decoymnist_afr_fixed.py \
  --png-root "${DATA_PATH}" \
  --seeds "${SEEDS}" \
  --split-seed "${SPLIT_SEED}" \
  --stage1-epochs "${STAGE1_EPOCHS}" \
  --stage2-epochs "${STAGE2_EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --stage2-lr "${STAGE2_LR}" \
  --stage2-weight-decay "${STAGE2_WEIGHT_DECAY}" \
  --stage2-batch-size "${STAGE2_BATCH_SIZE}" \
  --gamma "${GAMMA}" \
  --reg-coeff "${REG_COEFF}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  --output-csv "${OUT_CSV}"

