#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=15-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsRedMeat/redmeat_vanilla_cnn_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsRedMeat/redmeat_vanilla_cnn_sweep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV_CANDIDATES=(
  "${SCRIPT_DIR}/common_env.sh"
  "${SBATCH_SUBMIT_DIR:-}/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
  "/home/ryreu/guided_cnn/Food101/Waterbird_Runs/GALS/RedMeat_Runs/common_env.sh"
)
COMMON_ENV=""
for candidate in "${COMMON_ENV_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    COMMON_ENV="$candidate"
    break
  fi
done
if [[ -z "$COMMON_ENV" ]]; then
  echo "[ERROR] Could not locate common_env.sh" >&2
  exit 2
fi
source "$COMMON_ENV"

redmeat_set_defaults
redmeat_activate_env
redmeat_prepare_food_layout "$DATA_ROOT" "$DATA_DIR"

REPO_ROOT="$PROJECT_ROOT"
DATASET_ROOT="$DATA_ROOT/$DATA_DIR"

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}

LR_MIN=${LR_MIN:-1e-6}
LR_MAX=${LR_MAX:-1e-2}
WD_MIN=${WD_MIN:-1e-7}
WD_MAX=${WD_MAX:-1e-3}
MOM_MIN=${MOM_MIN:-0.80}
MOM_MAX=${MOM_MAX:-0.98}

OUT_CSV=${OUT_CSV:-$LOG_DIR/redmeat_vanilla_sweep_${SLURM_JOB_ID}.csv}
POST_OUT_CSV=${POST_OUT_CSV:-$LOG_DIR/redmeat_vanilla_best5_${SLURM_JOB_ID}.csv}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/Vanilla_RedMeat_Checkpoints}

export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"

cd "$GALS_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Missing DATASET_ROOT: $DATASET_ROOT" >&2
  exit 2
fi

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATASET_ROOT"
echo "Trials: $N_TRIALS (sampler=$SAMPLER sweep_seed=$SWEEP_SEED train_seed=$TRAIN_SEED)"
echo "Output CSV: $OUT_CSV"
echo "Post CSV: $POST_OUT_CSV"
echo "SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS"
which python

srun --unbuffered python -u RedMeat_Runs/run_vanilla_redmeat_sweep.py \
  "$DATASET_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --train-seed "$TRAIN_SEED" \
  --sampler "$SAMPLER" \
  --num-epochs 150 \
  --lr-min "$LR_MIN" --lr-max "$LR_MAX" \
  --wd-min "$WD_MIN" --wd-max "$WD_MAX" \
  --momentum-min "$MOM_MIN" --momentum-max "$MOM_MAX" \
  --output-csv "$OUT_CSV" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$POST_OUT_CSV" \
  --checkpoint-dir "$CKPT_DIR"
