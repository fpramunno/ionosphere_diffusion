#!/bin/bash
#SBATCH --job-name=unet_benchmark
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -A sk035
#SBATCH --output=./logs/out/out_unet_benchmark_v1_BS1.log
#SBATCH --error=./logs/err/err_unet_benchmark_v1_BS1.log

source ${IONO_VENV:-/path/to/venv}/bin/activate

unset ACCELERATE_USE_FSDP
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

export TE_DISABLE_FLASH_ATTN_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_TRACE_BUFFER_SIZE=10485760
export PYTHONUNBUFFERED=1

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}, SLURM_NODELIST=${SLURM_NODELIST}"

export SEQUENCE_LENGTH=22
export PREDICT_STEPS=7
export CSV_PATH="${IONO_HOME_ROOT:-/path/to/home_root}/data/ionosphere/l1_to_map_matched_2020_2025.csv"
export BATCH_SIZE=1
export DIR_NAME="unet_benchmark_v1_BS1"
CONDITIONING_LENGTH=$((SEQUENCE_LENGTH - PREDICT_STEPS))
export WANDB_RUN_NAME="unet_cond${CONDITIONING_LENGTH}_pred${PREDICT_STEPS}_bs${BATCH_SIZE}_benchmark_v1"

mkdir -p ${IONO_DATA_ROOT:-/path/to/data_root}/logs/out
mkdir -p ${IONO_DATA_ROOT:-/path/to/data_root}/logs/err

accelerate launch \
  --config_file ${IONO_REPO:-/path/to/ionosphere_diffusion}/configs/accelerate_config_ddp.yaml \
  ${IONO_REPO:-/path/to/ionosphere_diffusion}/training_unet.py \
  --sequence-length $SEQUENCE_LENGTH \
  --predict-steps $PREDICT_STEPS \
  --csv-path $CSV_PATH \
  --batch-size $BATCH_SIZE \
  --dir-name $DIR_NAME \
  --wandb-runname $WANDB_RUN_NAME \
  --max-steps 200000 \
  --evaluate-every 5000 \
  --save-every 10000 \
  --normalization-type absolute_max \
  --mixed-precision bf16 \
  --use-wandb \
  --only-complete-sequences \
  --cartesian-transform \
  --num-workers 8 \
  --val-steps 200 \
  --base-channels 256 \
  --channel-mults 1 2 4 4 \
  --num-res-blocks 8
  # --wandb-runid ""
