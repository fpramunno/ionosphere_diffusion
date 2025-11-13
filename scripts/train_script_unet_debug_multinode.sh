#!/bin/bash
#SBATCH --job-name=overfit_multinode
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH -A sk035
#SBATCH --output=/users/framunno/logs/out/out_overfit_15step_v4_cartesian_coordinates_absmax_debug_multinode_NOCONDITION_IN_OUTPUT.log
#SBATCH --error=/users/framunno/logs/err/err_overfit_15step_v4_cartesian_coordinates_absmax_debug_multinode_NOCONDITION_IN_OUTPUT.log

# =============================================================================
# ✅ Environment setup
# =============================================================================
source /users/framunno/envs/ionosphere/bin/activate

# -----------------------------------------------------------------------------
# 🚨 Clean all stale Accelerate/FSDP variables (fixes FULL_SHARD crash)
# -----------------------------------------------------------------------------
unset ACCELERATE_FSDP_SHARDING_STRATEGY
unset ACCELERATE_FSDP_STATE_DICT_TYPE
unset ACCELERATE_FSDP_BACKWARD_PREFETCH
unset ACCELERATE_FSDP_AUTO_WRAP_POLICY
unset ACCELERATE_FSDP_CPU_RAM_EFFICIENT_LOADING
unset ACCELERATE_FSDP_OFFLOAD_PARAMS
unset ACCELERATE_FSDP_SYNC_MODULE_STATES
unset ACCELERATE_FSDP_USE_ORIG_PARAMS
unset ACCELERATE_FSDP_USE_LOW_PRECISION_GRADIENTS
unset ACCELERATE_FSDP_TRANSFORMER_CLS_TO_WRAP

# -----------------------------------------------------------------------------
# ✅ Export only necessary variables
# -----------------------------------------------------------------------------
export ACCELERATE_USE_FSDP=true
export TE_DISABLE_FLASH_ATTN_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=hsn
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCKING_WAIT=1

# Multi-node coordination
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "---- Environment check ----"
echo "ACCELERATE_USE_FSDP=${ACCELERATE_USE_FSDP}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "Note: SLURM_NODEID will be set per-node inside srun (0 or 1)"

# =============================================================================
# ✅ Configurations (export for use in srun subshell)
# =============================================================================
export SEQUENCE_LENGTH=30
export PREDICT_STEPS=15
export CONFIG_PATH="/users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_unet.json"
export CSV_PATH="/users/framunno/data/ionosphere/l1_earth_associated_with_maps.csv"
export BATCH_SIZE=1
export DIR_NAME="unet_forecast_15frames_overfit_absolute_max_cscs_debug_multinode_NOCONDITION_IN_OUTPUT"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
export WANDB_RUN_NAME="unet_forecast_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_absolute_max_cscs_debug_multinode_NOCONDITION_IN_OUTPUT"

export NORM_TYPE="absolute_max"
export PREPROCESS_SCALING="log10"

mkdir -p /users/framunno/logs/out
mkdir -p /users/framunno/logs/err

# =============================================================================
# ✅ Run (Multi-node with srun + accelerate)
# =============================================================================
srun bash -c 'echo "Node $(hostname): SLURM_NODEID=$SLURM_NODEID starting accelerate..."; accelerate launch \
  --config_file /users/framunno/projects/ionosphere_diffusion/configs/accelerate_config_fsdp_multinode.yaml \
  --main_process_ip '"$MASTER_ADDR"' \
  --main_process_port '"$MASTER_PORT"' \
  --machine_rank $SLURM_NODEID \
  --num_machines '"$SLURM_NNODES"' \
  /users/framunno/projects/ionosphere_diffusion/training_pred.py \
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
  --grad-accum-steps 12 \
  --only-complete-sequences \
  --cartesian-transform \
  --num-workers 8 \
  --use-iterable-dataset'
