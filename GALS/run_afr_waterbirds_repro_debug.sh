#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/afr_waterbirds_repro_debug_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/afr_waterbirds_repro_debug_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-gals_a100}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
AFR_ROOT="${AFR_ROOT:-${REPO_ROOT}/afr}"

DATA_DIR="${DATA_DIR:-/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/guided_cnn/logsWaterbird/afr_repro_${SLURM_JOB_ID}}"
LOGS_ROOT="${LOGS_ROOT:-/home/ryreu/guided_cnn/logsWaterbird/afr_repro_logs_${SLURM_JOB_ID}}"

SEEDS="${SEEDS:-0,21,42}"
FULL_PAPER_GRID="${FULL_PAPER_GRID:-1}"
GAMMAS="${GAMMAS:-4,6,8,10,12,14,16,18,20}"
REG_COEFFS="${REG_COEFFS:-0,0.1,0.2,0.3,0.4}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-500}"
STAGE2_LR="${STAGE2_LR:-0.01}"

cd "${SCRIPT_DIR}"

echo "[$(date)] Host: $(hostname)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "AFR_ROOT=${AFR_ROOT}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LOGS_ROOT=${LOGS_ROOT}"
echo "SEEDS=${SEEDS}"
echo "FULL_PAPER_GRID=${FULL_PAPER_GRID}"
echo "STAGE1_EPOCHS=${STAGE1_EPOCHS}"
echo "STAGE2_EPOCHS=${STAGE2_EPOCHS}"
echo "STAGE2_LR=${STAGE2_LR}"
echo "GAMMAS=${GAMMAS}"
echo "REG_COEFFS=${REG_COEFFS}"
which python

python - <<'PY'
import importlib.util
import sys
mods = ["torch", "torchvision", "timm", "pandas", "numpy", "wandb"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("[ERROR] Missing Python packages:", ", ".join(missing), file=sys.stderr)
    print("Install AFR deps first (see afr/scripts/requirements.sh).", file=sys.stderr)
    raise SystemExit(2)
PY

ARGS=(
  --afr-root "${AFR_ROOT}"
  --data-dir "${DATA_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --logs-root "${LOGS_ROOT}"
  --python-exe "$(which python)"
  --seeds "${SEEDS}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage2-lr "${STAGE2_LR}"
)

if [[ "${FULL_PAPER_GRID}" == "1" ]]; then
  ARGS+=(--full-paper-grid)
else
  ARGS+=(--gammas "${GAMMAS}" --reg-coeffs "${REG_COEFFS}")
fi

srun --unbuffered python -u run_afr_waterbirds_repro.py "${ARGS[@]}"
