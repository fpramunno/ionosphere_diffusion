#!/bin/bash
#SBATCH --gres=gpu:0  
#SBATCH --nodelist=server0099,server0103,server0105,server0107,server0109,server0094
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --partition=performance
#SBATCH --job-name="obtain_df_params_new"
#SBATCH --error=./logs/err/err_obtain_df_params_new.err
#SBATCH --out=./logs/out/out_obtain_df_params_new.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

python3 /mnt/nas05/data01/francesco/progetto_simone/ionosphere/merge_data.py
