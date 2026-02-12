#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_clip_zeroshot_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/waterbirds_clip_zeroshot_debug_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# Default to Waterbirds-95; override DATA_PATH for Waterbirds-100.
DATA_PATH=${DATA_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}
CLIP_MODEL=${CLIP_MODEL:-RN50}
SEEDS=${SEEDS:-0,1,2,3,4}
SPLITS=${SPLITS:-val,test}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}
OUT_CSV=${OUT_CSV:-$LOG_DIR/clip_zeroshot_waterbirds_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[ERROR] Missing DATA_PATH: $DATA_PATH" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_PATH"
echo "CLIP model: $CLIP_MODEL"
echo "Seeds: $SEEDS"
echo "Splits: $SPLITS"
echo "Output CSV: $OUT_CSV"
which python

srun --unbuffered python -u run_clip_zeroshot_waterbirds.py \
  "$DATA_PATH" \
  --clip-model "$CLIP_MODEL" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --seeds "$SEEDS" \
  --splits "$SPLITS" \
  --output-csv "$OUT_CSV"
