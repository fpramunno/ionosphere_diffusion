#!/bin/bash
#SBATCH --job-name=precomp_dyn_ark50
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH -A sk035
#SBATCH --output=/capstor/scratch/cscs/framunno/logs/out/out_precompute_dynamics_seq365.log
#SBATCH --error=/capstor/scratch/cscs/framunno/logs/err/err_precompute_dynamics_seq365.log

source /users/framunno/envs/ionosphere/bin/activate
export PYTHONUNBUFFERED=1

mkdir -p /capstor/scratch/cscs/framunno/logs/out
mkdir -p /capstor/scratch/cscs/framunno/logs/err

python3 /users/framunno/projects/ionosphere_diffusion/precompute_dynamics_scores.py \
    --csv-path /users/framunno/data/ionosphere/l1_to_map_matched_2020_2025.csv \
    --cache-path /capstor/scratch/cscs/framunno/dynamics_scores_val_all_seq365.json \
    --split valid \
    --sequence-length 365 \
    --cartesian-transform \
    --epsilon-high-quantile 0.75 \
    --epsilon-low-quantile 0.25 \
    --num-workers 32
