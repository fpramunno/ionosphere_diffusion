#!/bin/bash
#SBATCH --job-name=overfit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32           # For data loading workers + prefetching
#SBATCH -A sk035
#SBATCH --output=/users/framunno/logs/out/out_overfit_15step_v4_cartesian_coordinates_absmax_debug.log
#SBATCH --error=/users/framunno/logs/err/err_overfit_15step_v4_cartesian_coordinates_absmax_debug.log

# =============================================================================
# ✅ Environment setup
# =============================================================================
source /users/framunno/envs/ionosphere/bin/activate

# ----------------------------------------------------------------------------- 
# 🚨 Clean all stale Accelerate/FSDP variables (fixes FULL_SHARD crash)
# -----------------------------------------------------------------------------
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
# ✅ Export only necessary variables
# -----------------------------------------------------------------------------
export ACCELERATE_USE_FSDP=true
export TE_DISABLE_FLASH_ATTN_VERSION_CHECK=1  # silences flash-attn warning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

# Quick check (optional): make sure only ACCELERATE_USE_FSDP remains
echo "---- Environment check ----"
env | grep ACCELERATE || echo "No stale ACCELERATE vars — clean environment ✅"

# =============================================================================
# ✅ Configurations
# =============================================================================
SEQUENCE_LENGTH=30
PREDICT_STEPS=15
CONFIG_PATH="/users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_unet.json"
CSV_PATH="/users/framunno/data/ionosphere/l1_earth_associated_with_maps.csv"
BATCH_SIZE=1
DIR_NAME="unet_forecast_15frames_overfit_absolute_max_cscs_debug"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
WANDB_RUN_NAME="unet_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_absolute_max_cscs_debug"

NORM_TYPE="absolute_max"
PREPROCESS_SCALING="log10"

mkdir -p /users/framunno/logs/out
mkdir -p /users/framunno/logs/err

# =============================================================================
# ✅ Run (Accelerate handles multi-GPU / multi-rank FSDP)
# =============================================================================
accelerate launch \
  --config_file /users/framunno/projects/ionosphere_diffusion/configs/accelerate_config_fsdp.yaml \
  /users/framunno/projects/ionosphere_diffusion/training_pred.py \
  --config $CONFIG_PATH \
  --sequence-length $SEQUENCE_LENGTH \
  --predict-steps $PREDICT_STEPS \
  --csv-path $CSV_PATH \
  --batch-size $BATCH_SIZE \
  --dir-name $DIR_NAME \
  --wandb-runname $WANDB_RUN_NAME \
  --max-epochs 500 \
  --evaluate-every 5 \
  --normalization-type $NORM_TYPE \
  --mixed-precision bf16 \
  --use-wandb \
  --overfit-single \
  --grad-accum-steps 1 \
  --only-complete-sequences \
  --cartesian-transform \
  --num-workers 8 \
  --use-iterable-dataset
