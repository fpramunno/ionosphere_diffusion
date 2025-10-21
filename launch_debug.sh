#!/bin/bash

echo "🐛 Starting debug training with wandb OFF"
echo "========================================"

python training_pred.py \
    --config /mnt/nas05/data01/francesco/progetto_simone/ionosphere/configs/forecast_iono.json \
    --batch-size 4 \
    --max-epochs 2 \
    --sequence-length 16 \
    --predict-steps 1 \
    --csv-path "/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/merged_params_solar_wind.csv" \
    --no-wandb \
    --dir-name "debug_run" \
    --normalization-type "absolute_max" \
    --num-workers 2