#!/bin/bash
#SBATCH --job-name=generate_v4
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -A sk035
#SBATCH --output=/iopsstor/scratch/cscs/framunno/logs/out/out_generate_v4_diffusion_%j.log
#SBATCH --error=/iopsstor/scratch/cscs/framunno/logs/err/err_generate_v4_diffusion_%j.log

source /users/framunno/envs/ionosphere/bin/activate

export PYTHONUNBUFFERED=1

# =============================================================================
# Args: $1=dynamics (low|high)  $2=activity (low|high)  $3=model (classic|nocond)  $4=ar (optional)
# =============================================================================
DYNAMICS="${1:?pass dynamics as \$1: low or high}"
ACTIVITY="${2:?pass activity as \$2: low or high}"
MODEL="${3:?pass model as \$3: classic or nocond}"

if [ "${4:-}" = "ar" ]; then
    SEQUENCE_LENGTH=50
    AR_FLAG="--autoregressive"
    MODE_TAG="ar50"
elif [ "${4:-}" = "ar3" ]; then
    SEQUENCE_LENGTH=36
    AR_FLAG="--autoregressive"
    MODE_TAG="ar3"
elif [ "${4:-}" = "ark50" ]; then
    # K=50 AR step (50 applicazioni del modello, 7 frame ciascuna) -> 15 + 7*50 = 365
    # tag "ark50" per non collidere con "ar" (che produce gia' MODE_TAG=ar50, ma e' K=5/seq_length=50)
    SEQUENCE_LENGTH=365
    AR_FLAG="--autoregressive"
    MODE_TAG="ark50"
else
    SEQUENCE_LENGTH=22
    AR_FLAG=""
    MODE_TAG="singlepass"
fi

if [ "$MODEL" = "classic" ]; then
    CKPT_PATH="/capstor/scratch/cscs/framunno/models_results/models_ViT_forecast_cond15_pred7_absolute_max_ddp_CLASSIC_v1_BS1/model_step_0200000.pth"
    NO_MAPPING_COND=""
else
    CKPT_PATH="/capstor/scratch/cscs/framunno/models_results/models_ViT_forecast_cond15_pred7_absolute_max_ddp_NOCOND_v1_BS1/model_step_0200000.pth"
    NO_MAPPING_COND="--no-mapping-cond"
fi

# --- redirect logs to descriptive filenames (iopsstor: capstor e' sopra la quota file) ---
mkdir -p /iopsstor/scratch/cscs/framunno/logs/out
mkdir -p /iopsstor/scratch/cscs/framunno/logs/err
exec >> /iopsstor/scratch/cscs/framunno/logs/out/out_generate_v4_diffusion_${MODEL}_dyn${DYNAMICS}_act${ACTIVITY}_${MODE_TAG}.log 2>> /iopsstor/scratch/cscs/framunno/logs/err/err_generate_v4_diffusion_${MODEL}_dyn${DYNAMICS}_act${ACTIVITY}_${MODE_TAG}.log

# =============================================================================
# Fixed config
# =============================================================================
MODEL_TYPE="diffusion"
CONFIG_PATH="/users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_15_big_cosine_solar_classic.json"
N_SAMPLES=10
SAMPLER="dpmpp_2m_sde"
DIFFUSION_STEPS=50
CSV_PATH="/users/framunno/data/ionosphere/l1_to_map_matched_2020_2025.csv"
NORMALIZATION_TYPE="absolute_max"
if [ "$MODE_TAG" = "ar3" ]; then
    DYNAMICS_CACHE="--dynamics-cache /capstor/scratch/cscs/framunno/dynamics_scores_val_all_seq36.json"
elif [ "$MODE_TAG" = "ark50" ]; then
    DYNAMICS_CACHE="--dynamics-cache /capstor/scratch/cscs/framunno/dynamics_scores_val_all_seq365.json"
else
    DYNAMICS_CACHE="--dynamics-cache /capstor/scratch/cscs/framunno/dynamics_scores_val_all_seq50.json"
fi
if [ "$MODE_TAG" = "ark50" ]; then
    # capstor e' sopra la quota file (1M+ inode) -- ark50 salva su iopsstor, vuoto e senza limite di file
    OUTPUT_DIR="/iopsstor/scratch/cscs/framunno/results_diffusion_${MODEL}_${MODE_TAG}_step200k_dyn${DYNAMICS}_act${ACTIVITY}"
else
    OUTPUT_DIR="/capstor/scratch/cscs/framunno/results_diffusion_${MODEL}_${MODE_TAG}_step200k_dyn${DYNAMICS}_act${ACTIVITY}"
fi

mkdir -p $OUTPUT_DIR
echo ">>> model=${MODEL} dynamics=${DYNAMICS} activity=${ACTIVITY} mode=${MODE_TAG} → $OUTPUT_DIR"

# =============================================================================
# Run
# =============================================================================
python3 /users/framunno/projects/ionosphere_diffusion/generate_data_v4.py \
    --model-type $MODEL_TYPE \
    --config $CONFIG_PATH \
    --ckpt $CKPT_PATH \
    --output-dir $OUTPUT_DIR \
    --csv-path $CSV_PATH \
    --sequence-length $SEQUENCE_LENGTH \
    --normalization-type $NORMALIZATION_TYPE \
    --activity-filter $ACTIVITY \
    --n-samples $N_SAMPLES \
    --sampler $SAMPLER \
    --diffusion-steps $DIFFUSION_STEPS \
    --cartesian-transform \
    $AR_FLAG \
    $NO_MAPPING_COND \
    --dynamics-filter $DYNAMICS \
    $DYNAMICS_CACHE
