#!/bin/bash -l
# Retrain missing Waterbirds upweight/ABN checkpoints with fixed hyperparameters.
# Runs 4 single-trial sweeps (fixed ranges => fixed values):
# - upweight WB95
# - upweight WB100
# - abn_cls WB95
# - abn_cls WB100

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/retrain_missing_upweight_abn_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/retrain_missing_upweight_abn_%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsWaterbird}"
DATA_ROOT="${DATA_ROOT:-/home/ryreu/guided_cnn/waterbirds}"
JOB_TAG="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

if [[ "${SKIP_ENV_ACTIVATE:-0}" != "1" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${ENV_NAME:-gals_a100}"
fi

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "Log dir: $LOG_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
which python

python -c "import optuna" 2>/dev/null || {
  echo "[INFO] Installing optuna..."
  pip install -q optuna
}

declare -a RUN_LABELS=()
declare -a RUN_CSVS=()
declare -a RUN_CKPTS=()

run_fixed() {
  local label="$1"
  local method="$2"
  local config="$3"
  local data_dir="$4"
  local base_lr="$5"
  local cls_lr="$6"
  local abn_cls_weight="${7:-}"

  local out_csv="$LOG_DIR/${label}_${JOB_TAG}.csv"
  local logs_dir="$LOG_DIR/${label}_logs_${JOB_TAG}"
  local run_prefix="${label}_${JOB_TAG}"

  local -a args=(
    --method "$method"
    --config "$config"
    --data-root "$DATA_ROOT"
    --waterbirds-dir "$data_dir"
    --n-trials 1
    --seed 0
    --train-seed 0
    --sampler random
    --keep all
    --output-csv "$out_csv"
    --logs-dir "$logs_dir"
    --base-lr-min "$base_lr"
    --base-lr-max "$base_lr"
    --cls-lr-min "$cls_lr"
    --cls-lr-max "$cls_lr"
    --post-seeds 0
    --run-name-prefix "$run_prefix"
  )

  if [[ -n "$abn_cls_weight" ]]; then
    args+=(
      --abn-cls-weight-min "$abn_cls_weight"
      --abn-cls-weight-max "$abn_cls_weight"
    )
  fi

  echo
  echo "=============================="
  echo "[RUN] $label"
  echo "method=$method config=$config data_dir=$data_dir"
  echo "base_lr=$base_lr classifier_lr=$cls_lr abn_cls_weight=${abn_cls_weight:-NONE}"
  echo "csv=$out_csv"
  echo "=============================="

  python -u run_gals_sweep.py "${args[@]}"

  local ckpt=""
  ckpt="$(python - "$out_csv" <<'PY'
import csv, os, sys
p = sys.argv[1]
rows = []
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("")
    raise SystemExit(0)
v = rows[-1].get("checkpoint", "") or ""
if v and not os.path.isabs(v):
    v = os.path.abspath(v)
print(v)
PY
)"

  RUN_LABELS+=("$label")
  RUN_CSVS+=("$out_csv")
  RUN_CKPTS+=("$ckpt")
}

# Upweight WB95
run_fixed \
  "retrain_upweight_wb95_fixed" \
  "upweight" \
  "configs/waterbirds_95_upweight.yaml" \
  "waterbird_complete95_forest2water2" \
  "0.013011143106374497" \
  "0.00034233257609501744"

# Upweight WB100
run_fixed \
  "retrain_upweight_wb100_fixed" \
  "upweight" \
  "configs/waterbirds_100_upweight.yaml" \
  "waterbird_1.0_forest2water2" \
  "0.029039631505422818" \
  "9.11321114874417e-05"

# ABN WB95
run_fixed \
  "retrain_abn_wb95_fixed" \
  "abn_cls" \
  "configs/waterbirds_95_abn.yaml" \
  "waterbird_complete95_forest2water2" \
  "0.01003187525820913" \
  "7.25437836215674e-05" \
  "2.617728751833586"

# ABN WB100
run_fixed \
  "retrain_abn_wb100_fixed" \
  "abn_cls" \
  "configs/waterbirds_100_abn.yaml" \
  "waterbird_1.0_forest2water2" \
  "0.027933817440579763" \
  "0.0008096689727354128" \
  "3.2547104257357056"

echo
echo "===== RETRAIN SUMMARY ====="
for i in "${!RUN_LABELS[@]}"; do
  echo "[${RUN_LABELS[$i]}]"
  echo "  csv: ${RUN_CSVS[$i]}"
  echo "  checkpoint: ${RUN_CKPTS[$i]}"
done
echo "==========================="

UP95_CKPT="${RUN_CKPTS[0]:-}"
UP100_CKPT="${RUN_CKPTS[1]:-}"
ABN95_CKPT="${RUN_CKPTS[2]:-}"
ABN100_CKPT="${RUN_CKPTS[3]:-}"

if [[ "${RUN_POINTING_GAME:-1}" == "1" ]]; then
  for req in "$UP95_CKPT" "$UP100_CKPT" "$ABN95_CKPT" "$ABN100_CKPT"; do
    if [[ -z "$req" || ! -f "$req" ]]; then
      echo "[ERROR] Missing retrained checkpoint needed for pointing game: $req" >&2
      exit 2
    fi
  done

  WB95_DATA_PATH="${WB95_DATA_PATH:-$DATA_ROOT/waterbird_complete95_forest2water2}"
  WB100_DATA_PATH="${WB100_DATA_PATH:-$DATA_ROOT/waterbird_1.0_forest2water2}"
  CUB_MASK_ROOT="${CUB_MASK_ROOT:-$DATA_ROOT/CUB_200_2011/segmentations}"
  WB95_MASK_ROOT="${WB95_MASK_ROOT:-$CUB_MASK_ROOT}"
  WB100_MASK_ROOT="${WB100_MASK_ROOT:-$CUB_MASK_ROOT}"
  PG_OUT_DIR="${PG_OUT_DIR:-$LOG_DIR/waterbirds_pointing_game_up_abn_clip_${JOB_TAG}}"

  echo
  echo "===== POINTING GAME ====="
  echo "WB95 upweight: $UP95_CKPT"
  echo "WB100 upweight: $UP100_CKPT"
  echo "WB95 abn: $ABN95_CKPT"
  echo "WB100 abn: $ABN100_CKPT"
  echo "Output: $PG_OUT_DIR"
  echo "========================="

  python -u waterbirds_pointing_game_eval.py \
    --datasets "${PG_DATASETS:-95,100}" \
    --split "${PG_SPLIT:-test}" \
    --target-mode "${PG_TARGET_MODE:-label}" \
    --max-samples "${PG_MAX_SAMPLES:-0}" \
    --sample-seed "${PG_SAMPLE_SEED:-0}" \
    --seed "${PG_SEED:-0}" \
    --methods "${PG_METHODS:-upweight,abn,clip_zs,clip_lr}" \
    --wb95-data-path "$WB95_DATA_PATH" \
    --wb100-data-path "$WB100_DATA_PATH" \
    --wb95-mask-root "$WB95_MASK_ROOT" \
    --wb100-mask-root "$WB100_MASK_ROOT" \
    --upweight95-ckpt "$UP95_CKPT" \
    --upweight100-ckpt "$UP100_CKPT" \
    --abn95-ckpt "$ABN95_CKPT" \
    --abn100-ckpt "$ABN100_CKPT" \
    --clip-model "${PG_CLIP_MODEL:-RN50}" \
    --clip-lr95-C "${PG_CLIP_LR95_C:-30.481669053249504}" \
    --clip-lr95-penalty "${PG_CLIP_LR95_PENALTY:-l2}" \
    --clip-lr95-solver "${PG_CLIP_LR95_SOLVER:-lbfgs}" \
    --clip-lr95-fit-intercept "${PG_CLIP_LR95_FIT_INTERCEPT:-1}" \
    --clip-lr100-C "${PG_CLIP_LR100_C:-0.2515000498909345}" \
    --clip-lr100-penalty "${PG_CLIP_LR100_PENALTY:-l2}" \
    --clip-lr100-solver "${PG_CLIP_LR100_SOLVER:-lbfgs}" \
    --clip-lr100-fit-intercept "${PG_CLIP_LR100_FIT_INTERCEPT:-1}" \
    --output-dir "$PG_OUT_DIR"
fi
