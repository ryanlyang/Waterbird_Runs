#!/bin/bash -l
# Red Meat guided sweep:
# 1) tune on primary GT path
# 2) rerun best config for 5 seeds on primary + each extra GT path

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsMeat/guided_redmeat_sweep_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMeat/guided_redmeat_sweep_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsMeat
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME=${ENV_NAME:-gals_a100}
BOOTSTRAP_ENV=${BOOTSTRAP_ENV:-0}
RECREATE_ENV=${RECREATE_ENV:-0}
REQ_FILE=/home/ryreu/guided_cnn/Food101/Waterbird_Runs/GALS/requirements.txt

if [[ "$BOOTSTRAP_ENV" -eq 1 ]]; then
  if [[ "$RECREATE_ENV" -eq 1 ]]; then
    conda env remove -n "$ENV_NAME" -y || true
  fi
  if ! conda env list | grep -E "^${ENV_NAME}[[:space:]]" >/dev/null; then
    conda create -y -n "$ENV_NAME" python=3.8
    conda activate "$ENV_NAME"
    conda install -y pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -c nvidia -c conda-forge
    conda install -y -c conda-forge pycocotools
    REQ_TMP=/tmp/${ENV_NAME}_reqs_$$.txt
    grep -v -E '^(opencv-python|pycocotools|torch|torchvision|torchray)' "$REQ_FILE" > "$REQ_TMP"
    pip install -r "$REQ_TMP"
    rm -f "$REQ_TMP"
    pip install torchray==1.0.0.2 --no-deps
    pip install opencv-python==4.6.0.66 optuna
    conda deactivate
  fi
fi

conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=/home/ryreu/guided_cnn/Food101/Waterbird_Runs
DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/Food101/data/food-101-redmeat}

# Primary GT path (used for tuning and best-seed reruns)
SWEEP_GT_ROOT=${SWEEP_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_dinovit/val/prediction_cmap/}

# Extra GT paths (used only for best-seed reruns)
ALT1_GT_ROOT=${ALT1_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openai_xcit/val/prediction_cmap/}
ALT2_GT_ROOT=${ALT2_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_openclip_dinovit/val/prediction_cmap/}
ALT3_GT_ROOT=${ALT3_GT_ROOT:-/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_redmeat_siglip2_dinovit/val/prediction_cmap/}

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-0}
SAMPLER=${SAMPLER:-tpe}
POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}

SWEEP_OUT=${SWEEP_OUT:-$LOG_DIR/guided_redmeat_sweep_${SLURM_JOB_ID}.csv}
POST_OUT=${POST_OUT:-$LOG_DIR/guided_redmeat_sweep_best5_${SLURM_JOB_ID}.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -f "$DATA_ROOT/meta/all_images.csv" ]]; then
  echo "Missing DATA_ROOT metadata: $DATA_ROOT/meta/all_images.csv" >&2
  exit 1
fi
for p in "$SWEEP_GT_ROOT" "$ALT1_GT_ROOT" "$ALT2_GT_ROOT" "$ALT3_GT_ROOT"; do
  if [[ ! -d "$p" ]]; then
    echo "Missing GT root: $p" >&2
    exit 1
  fi
done

python -c "import optuna" 2>/dev/null || {
  echo "[INFO] Installing optuna..."
  pip install -q optuna
}

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data: $DATA_ROOT"
echo "Primary GT (tune + best5): $SWEEP_GT_ROOT"
echo "Extra GT #1: $ALT1_GT_ROOT"
echo "Extra GT #2: $ALT2_GT_ROOT"
echo "Extra GT #3: $ALT3_GT_ROOT"
echo "Trials: $N_TRIALS (sampler=$SAMPLER seed=$SWEEP_SEED)"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
echo "Sweep CSV: $SWEEP_OUT"
echo "Post CSV: $POST_OUT"
which python

srun --unbuffered python -u run_waterbirds/run_guided_redmeat_sweep.py \
  "$DATA_ROOT" \
  "$SWEEP_GT_ROOT" \
  --extra-gt-path "$ALT1_GT_ROOT" \
  --extra-gt-path "$ALT2_GT_ROOT" \
  --extra-gt-path "$ALT3_GT_ROOT" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --sampler "$SAMPLER" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --output-csv "$SWEEP_OUT" \
  --post-output-csv "$POST_OUT"
