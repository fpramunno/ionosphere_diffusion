#!/bin/bash
#SBATCH --job-name=overfit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32           # Increased for data loading workers + prefetching
#SBATCH -A sk035
#SBATCH --output=/users/framunno/logs/out/out_overfit_15step_v4_cartesian_coordinates_absmax_debug.log
#SBATCH --error=/users/framunno/logs/err/err_overfit_15step_v4_cartesian_coordinates_absmax_debug.log

# Load your environment
source /users/framunno/envs/ionosphere/bin/activate

# Configuration for UNet with log10 preprocessing
SEQUENCE_LENGTH=30
PREDICT_STEPS=15
CONFIG_PATH="/users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_unet.json"
CSV_PATH="/users/framunno/data/ionosphere/l1_earth_associated_with_maps.csv"
BATCH_SIZE=12
DIR_NAME="unet_forecast_15frames_overfit_absolute_max_cscs_debug"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
WANDB_RUN_NAME="unet_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_absolute_max_cscs_debug"

# Preprocessing settings
NORM_TYPE="absolute_max" #"ionosphere_preprocess"  # Use new SDO-style preprocessing
PREPROCESS_SCALING="log10"  # Options: None, "log10", "sqrt", "symlog"

mkdir -p /users/framunno/logs/out
mkdir -p /users/framunno/logs/err

#python3 if i want to use 1 gpu
#accelerate launch training_pred.py \
accelerate launch --config_file /users/framunno/projects/ionosphere_diffusion/configs/accelerate_config.yaml \
    training_pred.py \
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
    # --preprocess-scaling $PREPROCESS_SCALING

# MAJOR OPTIMIZATIONS ENABLED:
# - IterableDataset with proper GPU/worker sharding (no overlap!)
# - CSV loaded once and cached (not reloaded on every sample)
# - Cartesian transform coordinates cached
# - persistent_workers=True (workers stay alive between epochs)
# - prefetch_factor=4 (4 batches prefetched per worker)
# - Sequential disk I/O (much faster than random seeks)
