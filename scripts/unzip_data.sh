#!/bin/bash
#SBATCH --job-name=unzip_ionosphere
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -A sk035
#SBATCH --output=/users/framunno/logs/out/out_unzip_ionosphere.log
#SBATCH --error=/users/framunno/logs/err/err_unzip_ionosphere.log

source /users/framunno/envs/ionosphere/bin/activate

mkdir -p /users/framunno/logs/out
mkdir -p /users/framunno/logs/err

echo "Starting unzip job on $(hostname) at $(date)"

python /users/framunno/projects/ionosphere_diffusion/unzip_data.py

echo "Done at $(date)"
