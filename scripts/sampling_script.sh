#!/bin/bash
#SBATCH --job-name=overfit_ddp
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -A sk035
#SBATCH --output=/users/framunno/logs/out/out_generate_valid_ViT_800mln.log
#SBATCH --error=/users/framunno/logs/err/err_generate_valid_ViT_800mln.log

python3 /users/framunno/projects/ionosphere_diffusion/generate_data_v3.py