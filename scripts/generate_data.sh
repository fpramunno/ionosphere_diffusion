#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodelist=server0099,server0103,server0105,server0107,server0109,server0094
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="generate_1frame_test"
#SBATCH --error=err_generate_1frame_test.log
#SBATCH --out=out_generate_1frame_test.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

python3 /mnt/nas05/data01/francesco/progetto_simone/ionosphere/generation_script.py