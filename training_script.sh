#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodelist=server0103,server0105,server0107,server0109,server0094
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_iono_batch32"
#SBATCH --error=err_training_iono_batch32.err
#SBATCH --out=out_training_iono_batch32.log
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

python3 train_3dmag.py

