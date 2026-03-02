#!/bin/bash -l
# One-job CLIP+LR Optuna sweeps for:
# - Waterbirds-95
# - Waterbirds-100
# - DecoyMNIST
#
# Uses the same wide search ranges for all three datasets.

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/clip_lr_all3_optuna50_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/clip_lr_all3_optuna50_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
ENV_NAME=${ENV_NAME:-gals_a100}
conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}
WB95_PATH=${WB95_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}
WB100_PATH=${WB100_PATH:-/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2}
DECOY_PATH=${DECOY_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}

LOG_WB=${LOG_WB:-/home/ryreu/guided_cnn/logsWaterbird}
LOG_MNIST=${LOG_MNIST:-/home/ryreu/guided_cnn/logsMNIST}
mkdir -p "$LOG_WB" "$LOG_MNIST"

N_TRIALS=${N_TRIALS:-50}
SWEEP_SEED=${SWEEP_SEED:-13}
SAMPLER=${SAMPLER:-tpe}
CLIP_MODEL=${CLIP_MODEL:-RN50}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-0}
OBJECTIVE=${OBJECTIVE:-val_avg_group_acc}

C_MIN=${C_MIN:-1e-6}
C_MAX=${C_MAX:-1e6}
TOL_MIN=${TOL_MIN:-1e-6}
TOL_MAX=${TOL_MAX:-1e-1}
MAX_ITER=${MAX_ITER:-8000}
PENALTY_SOLVERS=${PENALTY_SOLVERS:-l2:lbfgs,l2:saga,l1:saga,elasticnet:saga,l2:liblinear,l1:liblinear}
FEATURE_MODES=${FEATURE_MODES:-l2,raw,zscore}
CLASS_WEIGHT_OPTIONS=${CLASS_WEIGHT_OPTIONS:-none,balanced}

POST_SEEDS=${POST_SEEDS:-5}
POST_SEED_START=${POST_SEED_START:-0}
DECOY_VAL_FRAC=${DECOY_VAL_FRAC:-0.1}
DECOY_SPLIT_SEED=${DECOY_SPLIT_SEED:-0}

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p CLIP/clip
if [[ ! -f CLIP/clip/bpe_simple_vocab_16e6.txt.gz ]]; then
  echo "[INFO] Downloading CLIP BPE vocab..."
  curl -L -o CLIP/clip/bpe_simple_vocab_16e6.txt.gz \
    https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
fi

python -c "import optuna" 2>/dev/null || { echo "[INFO] Installing optuna..."; pip install -q optuna; }
python -c "import sklearn" 2>/dev/null || { echo "[INFO] Installing scikit-learn..."; pip install -q scikit-learn; }

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Sampler: $SAMPLER | Trials per dataset: $N_TRIALS | Seed: $SWEEP_SEED"
echo "Ranges: C=[$C_MIN,$C_MAX] tol=[$TOL_MIN,$TOL_MAX] max_iter=$MAX_ITER"
echo "Penalty/solvers: $PENALTY_SOLVERS"
echo "Feature modes: $FEATURE_MODES | class weights: $CLASS_WEIGHT_OPTIONS"
echo "Post seeds: $POST_SEEDS (start=$POST_SEED_START)"
which python

WB95_CSV=${WB95_CSV:-$LOG_WB/clip_lr95_tpe50_wide_${SLURM_JOB_ID}.csv}
WB95_POST_CSV=${WB95_POST_CSV:-$LOG_WB/clip_lr95_tpe50_wide_best5_${SLURM_JOB_ID}.csv}
WB100_CSV=${WB100_CSV:-$LOG_WB/clip_lr100_tpe50_wide_${SLURM_JOB_ID}.csv}
WB100_POST_CSV=${WB100_POST_CSV:-$LOG_WB/clip_lr100_tpe50_wide_best5_${SLURM_JOB_ID}.csv}
DECOY_CSV=${DECOY_CSV:-$LOG_MNIST/decoy_clip_lr_tpe50_wide_${SLURM_JOB_ID}.csv}
DECOY_POST_CSV=${DECOY_POST_CSV:-$LOG_MNIST/decoy_clip_lr_tpe50_wide_best5_${SLURM_JOB_ID}.csv}

echo
echo "===== [1/3] Waterbirds-95 CLIP+LR sweep ====="
srun --unbuffered python -u run_clip_lr_sweep.py \
  "$WB95_PATH" \
  --clip-model "$CLIP_MODEL" \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --output-csv "$WB95_CSV" \
  --sampler "$SAMPLER" \
  --C-min "$C_MIN" \
  --C-max "$C_MAX" \
  --tol-min "$TOL_MIN" \
  --tol-max "$TOL_MAX" \
  --max-iter "$MAX_ITER" \
  --penalty-solvers "$PENALTY_SOLVERS" \
  --feature-modes "$FEATURE_MODES" \
  --class-weight-options "$CLASS_WEIGHT_OPTIONS" \
  --objective "$OBJECTIVE" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$WB95_POST_CSV"

echo
echo "===== [2/3] Waterbirds-100 CLIP+LR sweep ====="
srun --unbuffered python -u run_clip_lr_sweep.py \
  "$WB100_PATH" \
  --clip-model "$CLIP_MODEL" \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --output-csv "$WB100_CSV" \
  --sampler "$SAMPLER" \
  --C-min "$C_MIN" \
  --C-max "$C_MAX" \
  --tol-min "$TOL_MIN" \
  --tol-max "$TOL_MAX" \
  --max-iter "$MAX_ITER" \
  --penalty-solvers "$PENALTY_SOLVERS" \
  --feature-modes "$FEATURE_MODES" \
  --class-weight-options "$CLASS_WEIGHT_OPTIONS" \
  --objective "$OBJECTIVE" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$WB100_POST_CSV"

echo
echo "===== [3/3] DecoyMNIST CLIP+LR sweep ====="
srun --unbuffered python -u run_clip_lr_sweep_decoymnist.py \
  "$DECOY_PATH" \
  --clip-model "$CLIP_MODEL" \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --n-trials "$N_TRIALS" \
  --seed "$SWEEP_SEED" \
  --sampler "$SAMPLER" \
  --C-min "$C_MIN" \
  --C-max "$C_MAX" \
  --tol-min "$TOL_MIN" \
  --tol-max "$TOL_MAX" \
  --max-iter "$MAX_ITER" \
  --penalty-solvers "$PENALTY_SOLVERS" \
  --feature-modes "$FEATURE_MODES" \
  --class-weight-options "$CLASS_WEIGHT_OPTIONS" \
  --objective "$OBJECTIVE" \
  --val-frac "$DECOY_VAL_FRAC" \
  --split-seed "$DECOY_SPLIT_SEED" \
  --output-csv "$DECOY_CSV" \
  --post-seeds "$POST_SEEDS" \
  --post-seed-start "$POST_SEED_START" \
  --post-output-csv "$DECOY_POST_CSV"

echo
echo "===== [DONE] ====="
echo "WB95 sweep:   $WB95_CSV"
echo "WB95 best5:   $WB95_POST_CSV"
echo "WB100 sweep:  $WB100_CSV"
echo "WB100 best5:  $WB100_POST_CSV"
echo "Decoy sweep:  $DECOY_CSV"
echo "Decoy best5:  $DECOY_POST_CSV"
