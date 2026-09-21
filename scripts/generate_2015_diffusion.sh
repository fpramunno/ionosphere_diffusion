#!/bin/bash
#SBATCH --job-name=gen_2015_diffusion
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -A sk035
#SBATCH --output=/capstor/scratch/cscs/framunno/logs/out/out_generate_2015_diffusion.log
#SBATCH --error=/capstor/scratch/cscs/framunno/logs/err/err_generate_2015_diffusion.log

source /users/framunno/envs/ionosphere/bin/activate
export PYTHONUNBUFFERED=1

# =============================================================================
# PRED_START: passed as $1 (single window) or uses default 5-window set
# =============================================================================
if [ -n "$1" ]; then
    PRED_STARTS=("$1")
else
    PRED_STARTS=(
        "2015-03-16 04:34"
        "2015-03-16 19:52"
        "2015-03-17 06:00"
    )
fi
ROLLOUT_END="2015-03-19T00:00:00+00:00"

# =============================================================================
# Fixed config
# =============================================================================
CSV_PATH="/users/framunno/data/ionosphere/l1_to_map_matched_2015_march_10_20_deduplicated.csv"
NORM_CSV="/users/framunno/data/ionosphere/l1_to_map_matched_2020_2025.csv"
NORMALIZATION_TYPE="absolute_max"
SEQUENCE_LENGTH=50
N_SAMPLES=3
SAMPLER="dpmpp_2m_sde"
DIFFUSION_STEPS=50

mkdir -p /capstor/scratch/cscs/framunno/logs/out
mkdir -p /capstor/scratch/cscs/framunno/logs/err

# =============================================================================
# Loop over PRED_START windows
# =============================================================================
for PRED_START in "${PRED_STARTS[@]}"; do

    EPOCH=$(date -d "$PRED_START" +%s)
    CENTER=$(date -d "@$((EPOCH + 1200))" "+%Y-%m-%dT%H:%M")
    RANGE_START=$(date -d "@$EPOCH" "+%Y-%m-%dT%H:%M")
    RANGE_END=$(date -d "@$((EPOCH + 3600))" "+%Y-%m-%dT%H:%M")
    TIME_RANGE="${RANGE_START}:${RANGE_END}"
    PRED_START_TAG=$(echo "$PRED_START" | tr ' ' 'T' | tr ':' 'h')

    echo ">>> ============================================================"
    echo ">>> Prediction starts at : $PRED_START"
    echo ">>> Center time          : $CENTER"
    echo ">>> Time range filter    : $TIME_RANGE"

    # CLASSIC
    OUTPUT_DIR="/capstor/scratch/cscs/framunno/results_2015_CLASSIC_AR50_pred${PRED_START_TAG}"
    mkdir -p $OUTPUT_DIR
    echo ">>> Running CLASSIC → $OUTPUT_DIR"
    python3 /users/framunno/projects/ionosphere_diffusion/generate_data_v4.py \
        --model-type diffusion \
        --config /users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_15_big_cosine_solar_classic.json \
        --ckpt /capstor/scratch/cscs/framunno/models_results/models_ViT_forecast_cond15_pred7_absolute_max_ddp_CLASSIC_v1_BS1/model_step_0200000.pth \
        --output-dir $OUTPUT_DIR \
        --csv-path $CSV_PATH \
        --sequence-length $SEQUENCE_LENGTH \
        --normalization-type $NORMALIZATION_TYPE \
        --n-samples $N_SAMPLES \
        --sampler $SAMPLER \
        --diffusion-steps $DIFFUSION_STEPS \
        --cartesian-transform \
        --autoregressive \
        --rollout-end-time "$ROLLOUT_END" \
        --norm-csv $NORM_CSV \
        --split eval \
        --time-ranges $TIME_RANGE

    # NOCOND
    OUTPUT_DIR="/capstor/scratch/cscs/framunno/results_2015_NOCOND_AR50_pred${PRED_START_TAG}"
    mkdir -p $OUTPUT_DIR
    echo ">>> Running NOCOND → $OUTPUT_DIR"
    python3 /users/framunno/projects/ionosphere_diffusion/generate_data_v4.py \
        --model-type diffusion \
        --config /users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_15_big_cosine_solar_classic.json \
        --ckpt /capstor/scratch/cscs/framunno/models_results/models_ViT_forecast_cond15_pred7_absolute_max_ddp_NOCOND_v1_BS1/model_step_0200000.pth \
        --output-dir $OUTPUT_DIR \
        --csv-path $CSV_PATH \
        --sequence-length $SEQUENCE_LENGTH \
        --normalization-type $NORMALIZATION_TYPE \
        --n-samples $N_SAMPLES \
        --sampler $SAMPLER \
        --diffusion-steps $DIFFUSION_STEPS \
        --cartesian-transform \
        --autoregressive \
        --no-mapping-cond \
        --rollout-end-time "$ROLLOUT_END" \
        --norm-csv $NORM_CSV \
        --split eval \
        --time-ranges $TIME_RANGE

done
