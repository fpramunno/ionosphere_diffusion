#!/bin/bash
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_iono_cond5_forecasting_5frame_L1COND"
#SBATCH --error=./logs/err/err_training_iono_cond5_forecasting_5frame_L1COND.log
#SBATCH --out=./logs/out/out_training_iono_cond5_forecasting_5frame_L1COND.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Default values - you can modify these or pass them as command line arguments
SEQUENCE_LENGTH=10
PREDICT_STEPS=5
CONFIG_PATH="/mnt/nas05/data01/francesco/progetto_simone/ionosphere/configs/forecast_iono_5_cond5.json"
CSV_PATH="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/merged_params_solar_wind.csv"
BATCH_SIZE=64
DIR_NAME="cond_forecasting_cfg_5predictedstep_v1_L1COND_cond5"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
WANDB_RUN_NAME="iono_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_L1COND_cond5"
NORM_TYPE="absolute_max"  # "absolute_max" or "mean_sigma_tanh"

python3 training_pred.py \
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
    --use-wandb

