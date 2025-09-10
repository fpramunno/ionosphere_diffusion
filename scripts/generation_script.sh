#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodelist=server0099,server0103,server0105,server0107,server0109,server0094
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="generation_multi_window_1frame"
#SBATCH --error=./logs/err/err_generation_multi_window_1frame.log
#SBATCH --out=./logs/out/out_generation_multi_window_1frame.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Default values - you can modify these or pass them as command line arguments
MODEL_CKPT="/mnt/nas05/data01/francesco/progetto_simone/ionosphere/models_results/models_cond_forecasting_cfg_1predictedstep_v1/model_epoch_0499.pth"
CONFIG_PATH="/mnt/nas05/data01/francesco/progetto_simone/ionosphere/configs/forecast_iono.json"
OUTPUT_DIR="/mnt/nas05/data01/francesco/progetto_simone/ionosphere/generated_data/generation_results_1frame"
CSV_PATH="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/npy_metrics.csv"
TRANSFORM_COND_CSV="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/params.csv"
NUM_GENERATIONS=5
MAX_SAMPLES=100
CONDITIONING_FRAMES=15

python3 generate_multi_window.py \
    --model_ckpt $MODEL_CKPT \
    --config $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --num_generations $NUM_GENERATIONS \
    --max_samples $MAX_SAMPLES \
    --conditioning_frames $CONDITIONING_FRAMES \
    --csv_path $CSV_PATH \
    --transform_cond_csv $TRANSFORM_COND_CSV \
    --split valid \
    --seed 42