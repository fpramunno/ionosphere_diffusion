#!/bin/bash
#SBATCH --job-name=multiscale_emb
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -A sk035
#SBATCH --output=./logs/out/out_ViT_15step_ddp_NOCOND_v1_BS1_resume.log
#SBATCH --error=./logs/err/err_ViT_15step_ddp_NOCOND_v1_BS1_resume.log

# =============================================================================
# ✅ Environment setup
# =============================================================================
source ${IONO_VENV:-/path/to/venv}/bin/activate

# -----------------------------------------------------------------------------
# 🚨 Clean all stale FSDP variables (switching to DDP)
# -----------------------------------------------------------------------------
unset ACCELERATE_USE_FSDP
unset ACCELERATE_FSDP_SHARDING_STRATEGY
unset ACCELERATE_FSDP_STATE_DICT_TYPE
unset ACCELERATE_FSDP_BACKWARD_PREFETCH
unset ACCELERATE_FSDP_AUTO_WRAP_POLICY
unset ACCELERATE_FSDP_CPU_RAM_EFFICIENT_LOADING
unset ACCELERATE_FSDP_OFFLOAD_PARAMS
unset ACCELERATE_FSDP_SYNC_MODULE_STATES
unset ACCELERATE_FSDP_USE_ORIG_PARAMS
unset ACCELERATE_FSDP_USE_LOW_PRECISION_GRADIENTS
unset ACCELERATE_FSDP_TRANSFORMER_CLS_TO_WRAP


# -----------------------------------------------------------------------------
# ✅ Export DDP variables
# -----------------------------------------------------------------------------
export TE_DISABLE_FLASH_ATTN_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_TRACE_BUFFER_SIZE=10485760
export PYTHONUNBUFFERED=1

# Multi-node coordination
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "---- Environment check ----"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "Using DDP (not FSDP)"

# =============================================================================
# ✅ Configurations (export for use in srun subshell)
# =============================================================================
export SEQUENCE_LENGTH=22
export PREDICT_STEPS=7
export CONFIG_PATH="${IONO_REPO:-/path/to/ionosphere_diffusion}/configs/forecast_iono_15_big_cosine_solar_classic.json"
export CSV_PATH="${IONO_HOME_ROOT:-/path/to/home_root}/data/ionosphere/l1_to_map_matched_2020_2025.csv"
export BATCH_SIZE=1
export DIR_NAME="ViT_forecast_cond15_pred7_absolute_max_ddp_NOCOND_v1_BS1"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
export WANDB_RUN_NAME="ViT_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_absolute_max_ddp_NOCOND_v1"

export NORM_TYPE="absolute_max"
export PREPROCESS_SCALING="log10"

mkdir -p ${IONO_HOME_ROOT:-/path/to/home_root}/logs/out
mkdir -p ${IONO_HOME_ROOT:-/path/to/home_root}/logs/err

# =============================================================================
# ✅ Run (Single-node multi-GPU with accelerate DDP)
# =============================================================================
accelerate launch \
  --config_file ${IONO_REPO:-/path/to/ionosphere_diffusion}/configs/accelerate_config_ddp.yaml \
  ${IONO_REPO:-/path/to/ionosphere_diffusion}/training_pred.py \
  --config $CONFIG_PATH \
  --sequence-length $SEQUENCE_LENGTH \
  --predict-steps $PREDICT_STEPS \
  --csv-path $CSV_PATH \
  --batch-size $BATCH_SIZE \
  --dir-name $DIR_NAME \
  --wandb-runname $WANDB_RUN_NAME \
  --max-steps 200000 \
  --evaluate-every 5000 \
  --save-every 10000 \
  --normalization-type $NORM_TYPE \
  --mixed-precision bf16 \
  --use-wandb \
  --grad-accum-steps 1 \
  --only-complete-sequences \
  --cartesian-transform \
  --num-workers 8 \
  --use-iterable-dataset \
  --val-steps 1000 \
  --no-mapping-cond \
  --wandb-runid "98hcyngr"
