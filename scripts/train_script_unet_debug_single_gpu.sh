#!/bin/bash
#SBATCH --job-name=overfit_1gpu
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -A sk035
#SBATCH --output=/users/framunno/logs/out/out_overfit_15step_v4_cartesian_coordinates_absmax_debug_1gpu.log
#SBATCH --error=/users/framunno/logs/err/err_overfit_15step_v4_cartesian_coordinates_absmax_debug_1gpu.log

# =============================================================================
# ✅ Environment setup
# =============================================================================
source /users/framunno/envs/ionosphere/bin/activate

# -----------------------------------------------------------------------------
# ✅ Export environment variables (NO FSDP)
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
export TE_DISABLE_FLASH_ATTN_VERSION_CHECK=1  # silences flash-attn warning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

echo "---- Environment check ----"
echo "Running on single GPU (no distributed training)"

# =============================================================================
# ✅ Configurations
# =============================================================================
SEQUENCE_LENGTH=30
PREDICT_STEPS=15
CONFIG_PATH="/users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_unet.json"
CSV_PATH="/users/framunno/data/ionosphere/l1_earth_associated_with_maps.csv"
BATCH_SIZE=12
DIR_NAME="unet_forecast_15frames_overfit_absolute_max_cscs_debug_1gpu"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
WANDB_RUN_NAME="unet_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_absolute_max_cscs_debug_1gpu"

NORM_TYPE="absolute_max"
PREPROCESS_SCALING="log10"

mkdir -p /users/framunno/logs/out
mkdir -p /users/framunno/logs/err

# =============================================================================
# ✅ Run (Single GPU - no accelerate)
# =============================================================================
python /users/framunno/projects/ionosphere_diffusion/training_pred.py \
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
