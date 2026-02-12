#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/clip_zeroshot_multi_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/clip_zeroshot_multi_debug_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR_WB=${LOG_DIR_WB:-/home/ryreu/guided_cnn/logsWaterbird}
LOG_DIR_RM=${LOG_DIR_RM:-/home/ryreu/guided_cnn/logsRedMeat}
mkdir -p "$LOG_DIR_WB" "$LOG_DIR_RM"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

WB95_PATH=${WB95_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}
WB100_PATH=${WB100_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}
REDMEAT_PATH=${REDMEAT_PATH:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}

CLIP_MODEL=${CLIP_MODEL:-RN50}
SEEDS=${SEEDS:-0,1,2,3,4}
SPLITS=${SPLITS:-val,test}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}

RUN_WB95=${RUN_WB95:-1}
RUN_WB100=${RUN_WB100:-1}
RUN_REDMEAT=${RUN_REDMEAT:-1}

OUT_WB95=${OUT_WB95:-$LOG_DIR_WB/clip_zeroshot_wb95_${SLURM_JOB_ID}.csv}
OUT_WB100=${OUT_WB100:-$LOG_DIR_WB/clip_zeroshot_wb100_${SLURM_JOB_ID}.csv}
OUT_REDMEAT=${OUT_REDMEAT:-$LOG_DIR_RM/clip_zeroshot_redmeat_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "CLIP model: $CLIP_MODEL"
echo "Seeds: $SEEDS"
echo "Splits: $SPLITS"
echo "RUN_WB95=$RUN_WB95 RUN_WB100=$RUN_WB100 RUN_REDMEAT=$RUN_REDMEAT"
which python

run_wb() {
  local data_path="$1"
  local out_csv="$2"
  local tag="$3"

  if [[ ! -d "$data_path" ]]; then
    echo "[WARN] Skipping $tag (missing path): $data_path"
    return 0
  fi

  echo "\n=== [ZERO-SHOT] $tag ==="
  echo "Data: $data_path"
  echo "Output: $out_csv"

  srun --unbuffered python -u run_clip_zeroshot_waterbirds.py \
    "$data_path" \
    --clip-model "$CLIP_MODEL" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --output-csv "$out_csv"
}

run_redmeat() {
  local data_path="$1"
  local out_csv="$2"

  if [[ ! -d "$data_path" ]]; then
    echo "[WARN] Skipping REDMEAT (missing path): $data_path"
    return 0
  fi

  echo "\n=== [ZERO-SHOT] REDMEAT ==="
  echo "Data: $data_path"
  echo "Output: $out_csv"

  srun --unbuffered python -u run_clip_zeroshot_redmeat.py \
    "$data_path" \
    --clip-model "$CLIP_MODEL" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seeds "$SEEDS" \
    --splits "$SPLITS" \
    --output-csv "$out_csv"
}

if [[ "$RUN_WB95" -eq 1 ]]; then
  run_wb "$WB95_PATH" "$OUT_WB95" "WATERBIRDS95"
fi

if [[ "$RUN_WB100" -eq 1 ]]; then
  run_wb "$WB100_PATH" "$OUT_WB100" "WATERBIRDS100"
fi

if [[ "$RUN_REDMEAT" -eq 1 ]]; then
  run_redmeat "$REDMEAT_PATH" "$OUT_REDMEAT"
fi

echo "\n[DONE] Zero-shot runs finished."
echo "  WB95 CSV:    $OUT_WB95"
echo "  WB100 CSV:   $OUT_WB100"
echo "  RedMeat CSV: $OUT_REDMEAT"
