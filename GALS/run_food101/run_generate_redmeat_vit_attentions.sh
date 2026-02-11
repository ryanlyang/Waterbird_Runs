#!/bin/bash -l
# Generate CLIP ViT attention maps for Red Meat subset.
#
# Expected dataset layout (default):
#   /home/ryreu/guided_cnn/Food101/data/food-101-redmeat/
#     images/
#     meta/all_images.csv
#     train -> images
#
# Outputs:
#   <DATA_ROOT>/<FOOD_DIR>/clip_vit_attention/*.pth

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=/home/ryreu/guided_cnn/logsMeat/redmeat_vit_attention_gen_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMeat/redmeat_vit_attention_gen_%j.err
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
    pip install opencv-python==4.6.0.66
    conda deactivate
  fi
fi

conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

REPO_ROOT=/home/ryreu/guided_cnn/Food101/Waterbird_Runs/GALS
DATA_ROOT=${DATA_ROOT:-/home/ryreu/guided_cnn/Food101/data}
FOOD_DIR=${FOOD_DIR:-food-101-redmeat}

CSV_FILE="$DATA_ROOT/$FOOD_DIR/meta/all_images.csv"

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

if [[ ! -d "$DATA_ROOT/$FOOD_DIR/images" ]]; then
  echo "[ERROR] Missing dataset images dir: $DATA_ROOT/$FOOD_DIR/images" >&2
  exit 2
fi
if [[ ! -f "$CSV_FILE" ]]; then
  echo "[ERROR] Missing metadata CSV: $CSV_FILE" >&2
  exit 2
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "Food subset dir: $FOOD_DIR"
which python

CHUNK_SIZE=${CHUNK_SIZE:-1000}
if [[ -z "${FOOD_N:-}" ]]; then
  FOOD_N=$(( $(wc -l < "$CSV_FILE") - 1 ))
fi
echo "[GEN] Total rows in all_images.csv: $FOOD_N"
echo "[GEN] Chunk size: $CHUNK_SIZE"

for ((start=0; start<FOOD_N; start+=CHUNK_SIZE)); do
  end=$((start+CHUNK_SIZE))
  echo "[GEN] RedMeat chunk: START_IDX=$start END_IDX=$end"
  srun --unbuffered python -u extract_attention.py \
    --config configs/food_attention.yaml \
    DATA.ROOT="$DATA_ROOT" \
    DATA.FOOD_SUBSET_DIR="$FOOD_DIR" \
    SAVE_FOLDER=clip_vit_attention \
    MODEL_TYPE="ViT-B/32" \
    ATTENTION_TYPE=transformer \
    DISABLE_VIS=true \
    SKIP_EXISTING=true \
    START_IDX="$start" \
    END_IDX="$end"
done

echo "[GEN] Done."

