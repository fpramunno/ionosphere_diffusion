#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodelist=server0099,server0103,server0105,server0107,server0109,server0094
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_iono_forecasting_oneframe"
#SBATCH --error=./logs/err/err_training_iono_forecasting_15frame.log
#SBATCH --out=./logs/out/out_training_iono_forecasting_15frame.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Default values - you can modify these or pass them as command line arguments
SEQUENCE_LENGTH=30
PREDICT_STEPS=15
CONFIG_PATH="/mnt/nas05/data01/francesco/progetto_simone/ionosphere/configs/forecast_iono_15.json"
CSV_PATH="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/npy_metrics.csv"
TRANSFORM_COND_CSV="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/params.csv"
BATCH_SIZE=64
DIR_NAME="cond_forecasting_cfg_15predictedstep_v1"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
WANDB_RUN_NAME="iono_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}"

python3 training_pred.py \
    --config $CONFIG_PATH \
    --sequence-length $SEQUENCE_LENGTH \
    --predict-steps $PREDICT_STEPS \
    --csv-path $CSV_PATH \
    --transform-cond-csv $TRANSFORM_COND_CSV \
    --batch-size $BATCH_SIZE \
    --dir-name $DIR_NAME \
    --wandb-runname $WANDB_RUN_NAME \
    --max-epochs 500 \
    --evaluate-every 5 \
    --use-wandb

