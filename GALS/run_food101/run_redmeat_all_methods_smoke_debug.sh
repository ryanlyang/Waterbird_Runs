#!/bin/bash -l
# Smoke-test all Red Meat sweep strategies with tiny runs.
# This is for "does it run end-to-end without crashing?" checks.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMeat/redmeat_all_methods_smoke_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMeat/redmeat_all_methods_smoke_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsMeat
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
ENV_NAME=${ENV_NAME:-gals_a100}
conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

WB_ROOT=/home/ryreu/guided_cnn/Food101/Waterbird_Runs
GALS_ROOT="$WB_ROOT/GALS"
DATA_ROOT=/home/ryreu/guided_cnn/Food101/data
DATA_DIR=${DATA_DIR:-food-101-redmeat}
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"
ATT_DIR=${ATT_DIR:-clip_vit_attention}
PRIMARY_GT_ROOT=${PRIMARY_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}

SMOKE_EPOCHS=${SMOKE_EPOCHS:-2}
SMOKE_BATCH_SIZE=${SMOKE_BATCH_SIZE:-32}
SMOKE_TRIALS=${SMOKE_TRIALS:-1}
SMOKE_SEED=${SMOKE_SEED:-0}

if [[ ! -f "$DATASET_ROOT/meta/all_images.csv" ]]; then
  echo "[ERROR] Missing metadata CSV: $DATASET_ROOT/meta/all_images.csv" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT/$ATT_DIR" ]]; then
  echo "[ERROR] Missing attention dir: $DATASET_ROOT/$ATT_DIR" >&2
  exit 2
fi
if [[ ! -d "$PRIMARY_GT_ROOT" ]]; then
  echo "[ERROR] Missing guided GT root: $PRIMARY_GT_ROOT" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Conda env: $ENV_NAME"
echo "Data root: $DATASET_ROOT"
echo "Attention dir: $DATASET_ROOT/$ATT_DIR"
echo "Guided GT root: $PRIMARY_GT_ROOT"
echo "Smoke epochs: $SMOKE_EPOCHS"
echo "Smoke batch size: $SMOKE_BATCH_SIZE"
echo "Smoke trials per method: $SMOKE_TRIALS"
which python

run_step() {
  local name="$1"
  shift
  echo "===================="
  echo "[SMOKE] START: $name"
  echo "===================="
  "$@"
  echo "[SMOKE] DONE: $name"
}

cd "$GALS_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

run_step "GALS ViT sweep smoke" \
  python -u run_gals_sweep.py \
    --method gals \
    --config configs/food_gals_vit.yaml \
    --data-root "$DATA_ROOT" \
    --waterbirds-dir "$DATA_DIR" \
    --dataset food_subset \
    --n-trials "$SMOKE_TRIALS" \
    --seed "$SMOKE_SEED" \
    --train-seed "$SMOKE_SEED" \
    --sampler random \
    --keep none \
    --output-csv "$LOG_DIR/redmeat_smoke_gals_${SLURM_JOB_ID}.csv" \
    --logs-dir "$LOG_DIR/redmeat_smoke_gals_logs_${SLURM_JOB_ID}" \
    --weight-min 100.0 \
    --weight-max 100.0 \
    --base-lr-min 1e-4 \
    --base-lr-max 1e-4 \
    --cls-lr-min 1e-4 \
    --cls-lr-max 1e-4 \
    --tune-weight-decay \
    --weight-decay-min 1e-5 \
    --weight-decay-max 1e-5 \
    --post-seeds 0 \
    DATA.FOOD_SUBSET_DIR="$DATA_DIR" \
    DATA.SUBDIR="$DATA_DIR" \
    DATA.ATTENTION_DIR="$ATT_DIR" \
    DATA.BATCH_SIZE="$SMOKE_BATCH_SIZE" \
    EXP.NUM_EPOCHS="$SMOKE_EPOCHS"

run_step "UpWeight sweep smoke" \
  python -u run_gals_sweep.py \
    --method upweight \
    --config configs/food_upweight.yaml \
    --data-root "$DATA_ROOT" \
    --waterbirds-dir "$DATA_DIR" \
    --dataset food_subset \
    --n-trials "$SMOKE_TRIALS" \
    --seed "$SMOKE_SEED" \
    --train-seed "$SMOKE_SEED" \
    --sampler random \
    --keep none \
    --output-csv "$LOG_DIR/redmeat_smoke_upweight_${SLURM_JOB_ID}.csv" \
    --logs-dir "$LOG_DIR/redmeat_smoke_upweight_logs_${SLURM_JOB_ID}" \
    --base-lr-min 1e-4 \
    --base-lr-max 1e-4 \
    --cls-lr-min 1e-4 \
    --cls-lr-max 1e-4 \
    --tune-weight-decay \
    --weight-decay-min 1e-5 \
    --weight-decay-max 1e-5 \
    --post-seeds 0 \
    DATA.FOOD_SUBSET_DIR="$DATA_DIR" \
    DATA.SUBDIR="$DATA_DIR" \
    DATA.BATCH_SIZE="$SMOKE_BATCH_SIZE" \
    EXP.NUM_EPOCHS="$SMOKE_EPOCHS"

run_step "Vanilla SGD sweep smoke" \
  python -u run_vanilla_sgd_sweep.py \
    --config configs/food_vanilla.yaml \
    --data-root "$DATA_ROOT" \
    --food-dir "$DATA_DIR" \
    --dataset food_subset \
    --n-trials "$SMOKE_TRIALS" \
    --seed "$SMOKE_SEED" \
    --train-seed "$SMOKE_SEED" \
    --sampler random \
    --keep none \
    --output-csv "$LOG_DIR/redmeat_smoke_vanilla_${SLURM_JOB_ID}.csv" \
    --logs-dir "$LOG_DIR/redmeat_smoke_vanilla_logs_${SLURM_JOB_ID}" \
    --lr-min 1e-3 \
    --lr-max 1e-3 \
    --weight-decay-min 1e-5 \
    --weight-decay-max 1e-5 \
    --momentum-min 0.9 \
    --momentum-max 0.9 \
    --post-seeds 0 \
    DATA.FOOD_SUBSET_DIR="$DATA_DIR" \
    DATA.SUBDIR="$DATA_DIR" \
    DATA.BATCH_SIZE="$SMOKE_BATCH_SIZE" \
    EXP.NUM_EPOCHS="$SMOKE_EPOCHS"

cd "$WB_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

run_step "Guided RedMeat sweep smoke" \
  env GUIDED_NUM_EPOCHS="$SMOKE_EPOCHS" python -u run_waterbirds/run_guided_redmeat_sweep.py \
    "$DATASET_ROOT" \
    "$PRIMARY_GT_ROOT" \
    --n-trials "$SMOKE_TRIALS" \
    --seed "$SMOKE_SEED" \
    --sampler random \
    --attn-min 0 \
    --attn-max 0 \
    --kl-min 10.0 \
    --kl-max 10.0 \
    --kl-incr-min 1.0 \
    --kl-incr-max 1.0 \
    --base-lr-min 1e-4 \
    --base-lr-max 1e-4 \
    --cls-lr-min 1e-4 \
    --cls-lr-max 1e-4 \
    --lr2-mult-min 1.0 \
    --lr2-mult-max 1.0 \
    --post-seeds 0 \
    --output-csv "$LOG_DIR/redmeat_smoke_guided_${SLURM_JOB_ID}.csv"

echo
echo "[SMOKE] All method smoke tests completed successfully."
echo "[SMOKE] CSV outputs:"
echo "  $LOG_DIR/redmeat_smoke_gals_${SLURM_JOB_ID}.csv"
echo "  $LOG_DIR/redmeat_smoke_upweight_${SLURM_JOB_ID}.csv"
echo "  $LOG_DIR/redmeat_smoke_vanilla_${SLURM_JOB_ID}.csv"
echo "  $LOG_DIR/redmeat_smoke_guided_${SLURM_JOB_ID}.csv"
