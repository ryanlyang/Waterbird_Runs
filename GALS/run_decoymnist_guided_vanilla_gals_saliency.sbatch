#!/bin/bash -l
# Train DecoyMNIST guided/vanilla/gals LeNet models and export RISE saliency
# examples from validation split (default: 15 per digit).

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_guided_vanilla_gals_saliency_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_guided_vanilla_gals_saliency_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS}"
CONDA_ENV="${CONDA_ENV:-learntolook}"
LOG_DIR="${LOG_DIR:-/home/ryreu/guided_cnn/logsMNIST}"
DATA_PATH="${DATA_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png}"
MASK_PATH="${MASK_PATH:-/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png/clip_rn50_attention_gradcam}"

OUT_DIR_DEFAULT="${LOG_DIR}/decoy_guided_vanilla_gals_saliency_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

EPOCHS="${EPOCHS:-19}"
SEED="${SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
SAMPLES_PER_DIGIT="${SAMPLES_PER_DIGIT:-15}"
TARGET_CLASS="${TARGET_CLASS:-label}"

# Vanilla defaults (CDEP-style).
VANILLA_LR="${VANILLA_LR:-0.001}"
VANILLA_WEIGHT_DECAY="${VANILLA_WEIGHT_DECAY:-1e-4}"

# Guided locked defaults (trial40-like).
GUIDED_LR="${GUIDED_LR:-0.001}"
GUIDED_WEIGHT_DECAY="${GUIDED_WEIGHT_DECAY:-1e-4}"
GUIDED_GRAD_WEIGHT="${GUIDED_GRAD_WEIGHT:-72503.48035960984}"
GUIDED_GRAD_CRITERION="${GUIDED_GRAD_CRITERION:-L1}"

# GALS locked defaults (separate knobs; set to your current best).
GALS_LR="${GALS_LR:-0.001}"
GALS_WEIGHT_DECAY="${GALS_WEIGHT_DECAY:-1e-4}"
GALS_GRAD_WEIGHT="${GALS_GRAD_WEIGHT:-72503.48035960984}"
GALS_GRAD_CRITERION="${GALS_GRAD_CRITERION:-L1}"

# RISE
RISE_NUM_MASKS="${RISE_NUM_MASKS:-2000}"
RISE_GRID_SIZE="${RISE_GRID_SIZE:-8}"
RISE_P1="${RISE_P1:-0.1}"
RISE_GPU_BATCH="${RISE_GPU_BATCH:-16}"
RISE_SEED="${RISE_SEED:-0}"
RISE_MASKS_PATH="${RISE_MASKS_PATH:-}"

mkdir -p "${LOG_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

# Ensure runtime deps used by utils/rise.py + saliency rendering are present.
python -c "import skimage" 2>/dev/null || {
  echo "[INFO] Installing scikit-image..."
  pip install -q scikit-image
}
python -c "import cv2" 2>/dev/null || {
  echo "[INFO] Installing opencv-python..."
  pip install -q opencv-python
}
python -c "import tqdm" 2>/dev/null || {
  echo "[INFO] Installing tqdm..."
  pip install -q tqdm
}

cd "${REPO_ROOT}"

echo "[$(date)] Host: $(hostname)"
which python
echo "Repo: ${REPO_ROOT}"
echo "Data: ${DATA_PATH}"
echo "Masks: ${MASK_PATH}"
echo "Output: ${OUT_DIR}"
echo "epochs=${EPOCHS} seed=${SEED} split_seed=${SPLIT_SEED}"
echo "samples_per_digit=${SAMPLES_PER_DIGIT} target_class=${TARGET_CLASS}"
echo "vanilla: lr=${VANILLA_LR} wd=${VANILLA_WEIGHT_DECAY}"
echo "guided:  lr=${GUIDED_LR} wd=${GUIDED_WEIGHT_DECAY} grad_weight=${GUIDED_GRAD_WEIGHT} grad_criterion=${GUIDED_GRAD_CRITERION}"
echo "gals:    lr=${GALS_LR} wd=${GALS_WEIGHT_DECAY} grad_weight=${GALS_GRAD_WEIGHT} grad_criterion=${GALS_GRAD_CRITERION}"

CMD=(
  python -u run_decoymnist_guided_vanilla_gals_saliency.py
  --png-root "${DATA_PATH}"
  --mask-root "${MASK_PATH}"
  --output-dir "${OUT_DIR}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --split-seed "${SPLIT_SEED}"
  --samples-per-digit "${SAMPLES_PER_DIGIT}"
  --target-class "${TARGET_CLASS}"
  --vanilla-lr "${VANILLA_LR}"
  --vanilla-weight-decay "${VANILLA_WEIGHT_DECAY}"
  --guided-lr "${GUIDED_LR}"
  --guided-weight-decay "${GUIDED_WEIGHT_DECAY}"
  --guided-grad-weight "${GUIDED_GRAD_WEIGHT}"
  --guided-grad-criterion "${GUIDED_GRAD_CRITERION}"
  --gals-lr "${GALS_LR}"
  --gals-weight-decay "${GALS_WEIGHT_DECAY}"
  --gals-grad-weight "${GALS_GRAD_WEIGHT}"
  --gals-grad-criterion "${GALS_GRAD_CRITERION}"
  --rise-num-masks "${RISE_NUM_MASKS}"
  --rise-grid-size "${RISE_GRID_SIZE}"
  --rise-p1 "${RISE_P1}"
  --rise-gpu-batch "${RISE_GPU_BATCH}"
  --rise-seed "${RISE_SEED}"
  --num-workers "${SLURM_CPUS_PER_TASK:-1}"
)

if [[ -n "${RISE_MASKS_PATH}" ]]; then
  CMD+=(--rise-masks-path "${RISE_MASKS_PATH}")
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --unbuffered "${CMD[@]}"
else
  "${CMD[@]}"
fi
