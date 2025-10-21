#!/bin/bash
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1
#SBATCH --nodelist=server0094
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_iono_cond5_forecasting_15frame_L1COND_big_cosine"
#SBATCH --error=./logs/err/err_training_iono_cond5_forecasting_15frame_L1COND_big_cosine.log
#SBATCH --out=./logs/out/out_training_iono_cond5_forecasting_15frame_L1COND_big_cosine.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Default values - you can modify these or pass them as command line arguments
SEQUENCE_LENGTH=30
PREDICT_STEPS=15
CONFIG_PATH="/mnt/nas05/data01/francesco/progetto_simone/ionosphere/configs/forecast_iono_15_big_cosine.json"
CSV_PATH="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/l1_earth_associated_with_maps.csv"
BATCH_SIZE=32
DIR_NAME="cond_forecasting_cfg_5predictedstep_L1COND_cond15_v3_big_cosine"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
WANDB_RUN_NAME="iono_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_L1COND_cond15_v3_big_cosine"
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
    --mixed-precision bf16 \
    --use-wandb

