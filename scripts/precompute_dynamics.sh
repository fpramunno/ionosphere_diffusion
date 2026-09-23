#!/bin/bash
#SBATCH --job-name=precomp_dyn
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH -A sk035
#SBATCH --output=./logs/out/out_precompute_dynamics_seq36_new.log
#SBATCH --error=./logs/err/err_precompute_dynamics_seq36_new.log

source ${IONO_VENV:-/path/to/venv}/bin/activate
export PYTHONUNBUFFERED=1

mkdir -p ${IONO_DATA_ROOT:-/path/to/data_root}/logs/out
mkdir -p ${IONO_DATA_ROOT:-/path/to/data_root}/logs/err

python3 ${IONO_REPO:-/path/to/ionosphere_diffusion}/precompute_dynamics_scores.py \
    --csv-path ${IONO_HOME_ROOT:-/path/to/home_root}/data/ionosphere/l1_to_map_matched_2020_2025.csv \
    --cache-path ${IONO_DATA_ROOT:-/path/to/data_root}/dynamics_scores_val_all_seq36.json \
    --split valid \
    --sequence-length 36 \
    --cartesian-transform \
    --epsilon-high-quantile 0.75 \
    --epsilon-low-quantile 0.25 \
    --num-workers 32
