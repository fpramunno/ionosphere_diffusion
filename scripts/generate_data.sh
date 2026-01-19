#!/bin/bash
#SBATCH --job-name=generate_multiscale_emb
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -A sk035
#SBATCH --output=/capstor/scratch/cscs/framunno/logs/out/out_generate_NOCOND_v3.log
#SBATCH --error=/capstor/scratch/cscs/framunno/logs/err/err_generate_NOCOND_v3.log

# =============================================================================
# ✅ Environment setup
# =============================================================================
source /users/framunno/envs/ionosphere/bin/activate

python3 /users/framunno/projects/ionosphere_diffusion/generate_data_v3.py