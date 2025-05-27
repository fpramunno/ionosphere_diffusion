#!/bin/bash
#SBATCH --gres=gpu:0  
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="obtain_df_params"
#SBATCH --error=err_obtain_df_params.err
#SBATCH --out=out_obtain_df_params.log
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

python3 obtain_paramsdf.py
