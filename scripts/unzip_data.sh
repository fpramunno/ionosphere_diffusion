#!/bin/bash
#SBATCH --job-name=unzip_ionosphere
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -A sk035
#SBATCH --output=./logs/out/out_unzip_ionosphere.log
#SBATCH --error=./logs/err/err_unzip_ionosphere.log

source ${IONO_VENV:-/path/to/venv}/bin/activate

mkdir -p ${IONO_HOME_ROOT:-/path/to/home_root}/logs/out
mkdir -p ${IONO_HOME_ROOT:-/path/to/home_root}/logs/err

echo "Starting unzip job on $(hostname) at $(date)"

python ${IONO_REPO:-/path/to/ionosphere_diffusion}/unzip_data.py

echo "Done at $(date)"
