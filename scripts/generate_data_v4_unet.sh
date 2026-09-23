#!/bin/bash
#SBATCH --job-name=generate_v4_unet
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -A sk035
#SBATCH --output=./logs/out/out_generate_v4_unet_%j.log
#SBATCH --error=./logs/err/err_generate_v4_unet_%j.log

source ${IONO_VENV:-/path/to/venv}/bin/activate

export PYTHONUNBUFFERED=1

# =============================================================================
# Configuration
# =============================================================================

# --- model type ---
export MODEL_TYPE="unet"

# --- unet ---
export CKPT_PATH="${IONO_DATA_ROOT:-/path/to/data_root}/models_results/models_unet_benchmark_v1_BS1/model_step_0200000.pth"
export BASE_CHANNELS=256
export CHANNEL_MULTS="1 2 4 4"
export NUM_RES_BLOCKS=8

# --- args: $1=dynamics (low|high)  $2=activity (low|high)  $3=ar (optional, enables autoregressive) ---
DYNAMICS="${1:?pass dynamics as \$1: low or high}"
ACTIVITY="${2:?pass activity as \$2: low or high}"

# --- sequence / rollout ---
if [ "${3:-}" = "ar" ]; then
    export SEQUENCE_LENGTH=50
    AR_FLAG="--autoregressive"
    MODE_TAG="ar50"
elif [ "${3:-}" = "ar3" ]; then
    export SEQUENCE_LENGTH=36
    AR_FLAG="--autoregressive"
    MODE_TAG="ar3"
elif [ "${3:-}" = "ark50" ]; then
    # K=50 AR step -> 15 + 7*50 = 365
    export SEQUENCE_LENGTH=365
    AR_FLAG="--autoregressive"
    MODE_TAG="ark50"
else
    export SEQUENCE_LENGTH=22
    AR_FLAG=""
    MODE_TAG="singlepass"
fi

# --- data ---
export CSV_PATH="${IONO_HOME_ROOT:-/path/to/home_root}/data/ionosphere/l1_to_map_matched_2020_2025.csv"
export NORMALIZATION_TYPE="absolute_max"
if [ "$MODE_TAG" = "ar3" ]; then
    export DYNAMICS_CACHE="--dynamics-cache ${IONO_DATA_ROOT:-/path/to/data_root}/dynamics_scores_val_all_seq36.json"
elif [ "$MODE_TAG" = "ark50" ]; then
    export DYNAMICS_CACHE="--dynamics-cache ${IONO_DATA_ROOT:-/path/to/data_root}/dynamics_scores_val_all_seq365.json"
else
    export DYNAMICS_CACHE="--dynamics-cache ${IONO_DATA_ROOT:-/path/to/data_root}/dynamics_scores_val_all_seq50.json"
fi

# --- output ---
if   [ "${3:-}" = "ar"    ]; then MODE="ar50"
elif [ "${3:-}" = "ar3"   ]; then MODE="ar3"
elif [ "${3:-}" = "ark50" ]; then MODE="ark50"
else                                MODE="singlepass"
fi
if [ "$MODE" = "ark50" ]; then
    # capstor e' sopra la quota file (1M+ inode) -- ark50 salva su iopsstor, vuoto e senza limite di file
    export OUTPUT_DIR="${IONO_FAST_ROOT:-/path/to/fast_scratch_root}/results_unet_${MODE}_step200k_dyn${DYNAMICS}_act${ACTIVITY}"
else
    export OUTPUT_DIR="${IONO_DATA_ROOT:-/path/to/data_root}/results_unet_${MODE}_step200k_dyn${DYNAMICS}_act${ACTIVITY}"
fi

# =============================================================================
# Run
# =============================================================================
mkdir -p ${IONO_FAST_ROOT:-/path/to/fast_scratch_root}/logs/out
mkdir -p ${IONO_FAST_ROOT:-/path/to/fast_scratch_root}/logs/err
mkdir -p $OUTPUT_DIR

echo ">>> dynamics=${DYNAMICS} activity=${ACTIVITY} → $OUTPUT_DIR"

python3 ${IONO_REPO:-/path/to/ionosphere_diffusion}/generate_data_v4.py \
    --model-type $MODEL_TYPE \
    --ckpt $CKPT_PATH \
    --output-dir $OUTPUT_DIR \
    --csv-path $CSV_PATH \
    --sequence-length $SEQUENCE_LENGTH \
    --normalization-type $NORMALIZATION_TYPE \
    --activity-filter $ACTIVITY \
    --base-channels $BASE_CHANNELS \
    --channel-mults $CHANNEL_MULTS \
    --num-res-blocks $NUM_RES_BLOCKS \
    --cartesian-transform \
    --dynamics-filter $DYNAMICS \
    $AR_FLAG \
    $DYNAMICS_CACHE
