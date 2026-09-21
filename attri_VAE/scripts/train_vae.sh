#!/bin/bash
#SBATCH --job-name=attri_vae
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=15:00:00
#SBATCH -A sk035
#SBATCH --output=/capstor/scratch/cscs/framunno/logs/out/out_attri_vae_perpceptual0_convae2d_%j_LARGEV2_lr3e_4.log
#SBATCH --error=/capstor/scratch/cscs/framunno/logs/err/err_attri_vae_perpceptual0_convae2d_%j_LARGEV2_lr3e_4.log

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# =============================================================================
# ✅ DDP environment variables
# =============================================================================
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# =============================================================================
# ✅ Environment setup
# =============================================================================
source /users/framunno/envs/ionosphere/bin/activate

# =============================================================================
# Configurations
# =============================================================================
BATCH_SIZE=16
EPOCHS=500
LEARNING_RATE=3e-4 #1e-4 0.0001
LATENT_SIZE=128
SEQUENCE_LENGTH=1
HDIM=512
# Loss hyperparameters
BETA=2.0          # KL loss weight
ALPHA=1.0         # MLP loss weight
GAMMA=10.0        # AR loss weight (10.0 needed for good attribute correlation)
FACTOR=100.0      # AR loss factor # before it was 100.0
PERCEPTUAL_WEIGHT=0  # Weight for perceptual loss component
# Skip connection mode: 'full', 'none', or 'dropout'
# - full: use all skip connections (best reconstruction, but latent manipulation won't work)
# - none: no skip connections (latent manipulation works, but lower quality)
# - dropout: randomly drop skip connections during training (best of both worlds)
SKIP_MODE="none" # choose from 'full', 'none', 'dropout' or 'weighted'
SKIP_SCALE=0  # scaling factor for weighted skip connections
SKIP_DROPOUT_PROB=0.0  # probability of dropping each skip connection
MODEL="CONVAE2D"
KL_ANNEAL_EPOCHS=100  # Number of epochs over which to anneal KL weight
KL_ANNEAL_START=0.1    # Epoch to start KL annealing

# =============================================================================
# Disc loss flag
LAMBDA_GAN=0.01  # Weight for GAN loss component

# WANDB RUN ID:
WANDB_RUN_ID="cfaxuggm"  # to resume previous run, set the wandb run ID here

# =============================================================================
# Paths
CSV_PATH="/users/framunno/data/ionosphere/l1_to_map_matched_even_minutes_test_v3_interpolated_deduplicated.csv"
TRANSFORM_COND_CSV="/users/framunno/data/ionosphere/params.csv"  # UPDATE THIS
# SAVING_PATH="/mnt/nas05/data01/francesco/progetto_simone/results_attri_vae_skip${SKIP_MODE}_gamma${GAMMA}_dropout${SKIP_DROPOUT_PROB}_hdim${HDIM}_latentsize${LATENT_SIZE}_perceptual${PERCEPTUAL_WEIGHT}_model${MODEL}/"
# SAVING_PATH="/users/framunno/data/ionosphere/results_attri_vae_skipnone_gamma10.0_dropout0.5/"
# SAVING_PATH="/mnt/nas05/data01/francesco/progetto_simone/STABLEDIFF_VAE/"
# SAVING_PATH="/capstor/scratch/cscs/framunno/models_results/results_attri_vae_gamma${GAMMA}_hdim${HDIM}_latentsize${LATENT_SIZE}_perceptual${PERCEPTUAL_WEIGHT}_model${MODEL}_GANlambda${LAMBDA_GAN}/"
SAVING_PATH="/capstor/scratch/cscs/framunno/models_results/results_attri_vae_gamma${GAMMA}_hdim${HDIM}_latentsize${LATENT_SIZE}_perceptual${PERCEPTUAL_WEIGHT}_model${MODEL}_LARGEV2_lr${LEARNING_RATE}/"
# Wandb
WANDB_PROJECT="attri-vae"
# WANDB_RUN_NAME="unet_vae_skip${SKIP_MODE}_gamma${GAMMA}_beta${BETA}_dropout${SKIP_DROPOUT_PROB}_hdim${HDIM}_latentsize${LATENT_SIZE}_perceptual${PERCEPTUAL_WEIGHT}_skipscale${SKIP_SCALE}_model${MODEL}"
# WANDB_RUN_NAME="model${MODEL}_vae_gamma${GAMMA}_hdim${HDIM}_latentsize${LATENT_SIZE}_perceptual${PERCEPTUAL_WEIGHT}_GANlambda${LAMBDA_GAN}"
WANDB_RUN_NAME="model${MODEL}_vae_gamma${GAMMA}_hdim${HDIM}_latentsize${LATENT_SIZE}_perceptual${PERCEPTUAL_WEIGHT}_LARGEV2_lr${LEARNING_RATE}"
# WANDB_RUN_NAME="unet_vae_skipnone_gamma10.0_beta2.0_dropout0.5"
# WANDB_RUN_NAME="stable_diff_vae"
# Create output directories
# mkdir -p /mnt/nas05/data01/francesco/progetto_simone/logs/out
# mkdir -p /mnt/nas05/data01/francesco/progetto_simone/logs/err
mkdir -p $SAVING_PATH

echo "========================================"
echo "Starting Attri-VAE Training"
echo "========================================"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Latent size: $LATENT_SIZE"
echo "Beta: $BETA, Alpha: $ALPHA, Gamma: $GAMMA"
echo "Skip mode: $SKIP_MODE (dropout prob: $SKIP_DROPOUT_PROB)"
echo "========================================"

# =============================================================================
# Run training
# =============================================================================
accelerate launch \
    --config_file /users/framunno/projects/ionosphere_diffusion/configs/accelerate_config_ddp.yaml \
    /users/framunno/projects/ionosphere_diffusion/attri_VAE/training_main.py \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --latent-size $LATENT_SIZE \
    --hdim $HDIM \
    --sequence-length $SEQUENCE_LENGTH \
    --perceptual-weight $PERCEPTUAL_WEIGHT \
    --beta $BETA \
    --alpha $ALPHA \
    --gamma $GAMMA \
    --factor $FACTOR \
    --csv-path $CSV_PATH \
    --transform-cond-csv $TRANSFORM_COND_CSV \
    --saving-path $SAVING_PATH \
    --normalization-type absolute_max \
    --num-workers 8 \
    --use-iterable-dataset \
    --only-complete-sequences \
    --cartesian-transform \
    --use-wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-runname $WANDB_RUN_NAME \
    --skip-mode $SKIP_MODE \
    --skip-dropout-prob $SKIP_DROPOUT_PROB \
    --skip-scale $SKIP_SCALE \
    --kl-anneal-epochs $KL_ANNEAL_EPOCHS \
    --kl-anneal-start $KL_ANNEAL_START \
    # --wandb-run-id $WANDB_RUN_ID \
    # --resume-training 
    # --disc-loss \
    # --lambda-gan $LAMBDA_GAN \

echo "Training completed!"
