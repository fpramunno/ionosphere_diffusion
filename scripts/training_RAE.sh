#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=server0107
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_RAE_stage1_ionosphere"
#SBATCH --error=/mnt/nas05/data01/francesco/progetto_simone/ionosphere/logs/err/err_training_RAE_stage1.log
#SBATCH --out=/mnt/nas05/data01/francesco/progetto_simone/ionosphere/logs/out/out_training_RAE_stage1.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Wandb settings
export ENTITY="francescopio"
export PROJECT="ionosphere-rae-stage1"
export WANDB_KEY="000ad2f3e0a4da21d61c7df4d767a9b6e70591df"  # Replace with your actual wandb API key from https://wandb.ai/authorize

# Paths
RAE_DIR="/mnt/nas05/data01/francesco/progetto_simone/ionosphere/RAE"
CONFIG_PATH="${RAE_DIR}/configs/stage1/training/Ionosphere_DINOv2-B_decXL.yaml"
DATA_PATH="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/pickled_maps"
RESULTS_DIR="${RAE_DIR}/results_ionosphere_stage1"

# Training settings
IMAGE_SIZE=256
PRECISION="bf16"

# Create log directories if they don't exist (relative to ionosphere root)
mkdir -p /mnt/nas05/data01/francesco/progetto_simone/ionosphere/logs/err
mkdir -p /mnt/nas05/data01/francesco/progetto_simone/ionosphere/logs/out

# Change to RAE directory
cd $RAE_DIR

# Run training
python src/train_stage1.py \
    --config $CONFIG_PATH \
    --data-path $DATA_PATH \
    --results-dir $RESULTS_DIR \
    --image-size $IMAGE_SIZE \
    --precision $PRECISION \
    --wandb

echo "Training completed!"

