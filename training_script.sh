#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodelist=server0099,server0103,server0105,server0107,server0109,server0094
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_iono_forecasting"
#SBATCH --error=err_training_iono_forecasting.log
#SBATCH --out=out_training_iono_forecasting.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

python3 train_3dmag.py

