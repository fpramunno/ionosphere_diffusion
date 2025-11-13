#!/bin/bash
#SBATCH --job-name=fsdp_train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32           # give enough CPU threads to feed 4 GPUs with prefetch
#SBATCH -A sk035
#SBATCH --output=/users/framunno/logs/out/out_fsdp_15step_cartesian_bf16_optimized.log
#SBATCH --error=/users/framunno/logs/err/err_fsdp_15step_cartesian_bf16_optimized.log
#SBATCH --time=24:00:00

# Load your environment
source /users/framunno/envs/ionosphere/bin/activate

# Configuration for UNet with FSDP + all optimizations
SEQUENCE_LENGTH=30
PREDICT_STEPS=15
CONFIG_PATH="/users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_15_big.json"
CSV_PATH="/users/framunno/data/ionosphere/l1_earth_associated_with_maps.csv"
BATCH_SIZE=16                        # Per GPU! Global batch = 16 * 4 = 64
DIR_NAME="fsdp_forecast_15frames_bf16_optimized"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
WANDB_RUN_NAME="fsdp_4gpu_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}x4_bf16_optimized"

# Preprocessing settings
NORM_TYPE="absolute_max"

mkdir -p /users/framunno/logs/out
mkdir -p /users/framunno/logs/err

# Launch with FSDP config + all optimizations
accelerate launch --config_file /users/framunno/projects/ionosphere_diffusion/configs/accelerate_config_fsdp.yaml \
    training_pred.py \
    --config $CONFIG_PATH \
    --sequence-length $SEQUENCE_LENGTH \
    --predict-steps $PREDICT_STEPS \
    --csv-path $CSV_PATH \
    --batch-size $BATCH_SIZE \
    --dir-name $DIR_NAME \
    --wandb-runname $WANDB_RUN_NAME \
    --max-epochs 100 \
    --evaluate-every 5 \
    --normalization-type $NORM_TYPE \
    --mixed-precision bf16 \
    --use-wandb \
    --grad-accum-steps 1 \
    --only-complete-sequences \
    --cartesian-transform \
    --use-fsdp \
    --activation-checkpointing \
    --compile-model \
    --num-workers 8

echo "✅ FSDP Training completed!"
