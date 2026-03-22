#!/bin/bash
#SBATCH -J get_logits_multi
#SBATCH -c 4
#SBATCH --gres=gpu:a100:1
#SBATCH -p normal
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# ========================
# basic env
# ========================
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
export TMPDIR=/scratch-ssd/$USER/tmp

set +u
/scratch-ssd/oatml/run_locked.sh \
  /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate llm
set -u
# ========================
# Params
# ========================
#RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_ID="basic_logits_analysis01"

MODE=${MODE:-both}
TEMP=${TEMP:-1.0}
N_SAMPLES=${N_SAMPLES:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-200}

# ========================
# model list
# ========================
MODELS=(
  "Qwen/Qwen3.5-0.8B"
  "Qwen/Qwen2.5-1.5B"
  # "Qwen/Qwen2.5-Math-7B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "deepseek-ai/deepseek-llm-7b-chat"
  # "mistralai/Mistral-7B-v0.1"
  # "microsoft/Phi-3-mini-4k-instruct"
  # "meta-llama/Llama-3.2-3B"
)

# ========================
# Loop
# ========================
for MODEL_NAME in "${MODELS[@]}"; do
  MODEL_TAG=$(basename "$MODEL_NAME")

  echo "=========================="
  echo "Running model: $MODEL_NAME"
  echo "=========================="

  LOCAL_ROOT="/scratch-ssd/$USER/SFT_VS_RL/$MODEL_TAG/$RUN_ID"
  HOME_ROOT="$HOME/SFT_VS_RL/$MODEL_TAG/$RUN_ID"

  mkdir -p "$LOCAL_ROOT"
  mkdir -p "$HOME_ROOT"
  # -------- SFT logits --------
  srun python -u get_logits_sft.py \
    --config ./configs/train_basic.yaml \
    --model_name "$MODEL_NAME" \
    --output_dir "$LOCAL_ROOT/sft" \
    --do_extract_logits \
    --extract_n_samples "$N_SAMPLES"

  # # -------- RL logits --------
  # srun python -u get_rl_logits.py \
  #   --config ./configs/train_basic.yaml \
  #   --model_name "$MODEL_NAME" \
  #   --output_dir "$LOCAL_ROOT/rl" \
  #   --mode "$MODE" \
  #   --temperature "$TEMP" \
  #   --max_new_tokens "$MAX_NEW_TOKENS" \
  #   --n_samples "$N_SAMPLES"


  REMOTE_HOST="josren@oat0.cs.ox.ac.uk"
  REMOTE_BASE="/scratch-ssd/$USER/SFT_VS_RL/$MODEL_TAG/$RUN_ID"
  echo "Creating remote dir..."
  ssh "$REMOTE_HOST" "mkdir -p $REMOTE_BASE"

  echo "Syncing large files to $REMOTE_BASE"
  echo "Syncing metadata to $HOME_ROOT"
  # -------- sync back HOME --------
  # 1) 大文件转存到 /scratch
  rsync -av --partial \
    --include="*/" \
    --include="*.pt" \
    --include="*.bin" \
    --include="*.safetensors" \
    --exclude="*" \
    "$LOCAL_ROOT/" "$REMOTE_HOST:$REMOTE_BASE/"

  # 2) 小文件同步到 HOME
  rsync -av --partial \
    --include="*/" \
    --include="*.json" \
    --include="*.jsonl" \
    --include="*.yaml" \
    --include="*.yml" \
    --include="*.txt" \
    --include="*.csv" \
    --include="*.png" \
    --exclude="*" \
    "$LOCAL_ROOT/" "$HOME_ROOT/"

done