#!/bin/bash
#SBATCH --job-name=overfit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12           # give enough CPU threads to feed 4 GPUs
#SBATCH --time=24:00:00
#SBATCH --output=./logs/out/out_overfit_15step_v4_cartesian_coordinates.log
#SBATCH --error=./logs/err/err_overfit_15step_v4_cartesian_coordinates.log

# Default values - you can modify these or pass them as command line arguments
SEQUENCE_LENGTH=30
PREDICT_STEPS=15
CONFIG_PATH="/users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_15_big_cosine_solar.json"
CSV_PATH="/users/framunno/data/ionosphere/l1_earth_associated_with_maps.csv"
BATCH_SIZE=6
DIR_NAME="cond_forecasting_15frames_overfit_v4_cartesian_coordinates_cscs"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
WANDB_RUN_NAME="iono_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_overfit_v4_cartesian_coordinates_cscs"
NORM_TYPE="absolute_max"  # "absolute_max" or "mean_sigma_tanh"

#python3 if i want to use 1 gpu
accelerate launch training_pred.py \ 
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
    --cartesian-transform

