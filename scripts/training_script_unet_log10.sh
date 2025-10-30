#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=server0094
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="unet_log10"
#SBATCH --error=./logs/err/err_unet_15step_absolute_max.log
#SBATCH --out=./logs/out/out_unet_15step_absolute_max.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Configuration for UNet with log10 preprocessing
SEQUENCE_LENGTH=30
PREDICT_STEPS=15
CONFIG_PATH="/mnt/nas05/data01/francesco/progetto_simone/ionosphere/configs/forecast_iono_unet.json"
CSV_PATH="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/l1_earth_associated_with_maps.csv"
BATCH_SIZE=1
DIR_NAME="unet_forecast_15frames_overfit_absolute_max"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
WANDB_RUN_NAME="unet_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_absolute_max"

# Preprocessing settings
NORM_TYPE="absolute_max" #"ionosphere_preprocess"  # Use new SDO-style preprocessing
PREPROCESS_SCALING="log10"  # Options: None, "log10", "sqrt", "symlog"

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
    --use-wandb \
    --overfit-single \
    --grad-accum-steps 6 \
    --only-complete-sequences \
    --cartesian-transform
    # --preprocess-scaling $PREPROCESS_SCALING \
