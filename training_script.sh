#!/bin/bash -l
#SBATCH -p gpu
# SBATCH -p gpupreempt -q gpupreempt
#SBATCH --time=144:00:00
#SBATCH --mem=450G
#SBATCH -C a100-40gb  # a100-40gb # a100-80gb # h100 # h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH -o /mnt/home/framunno/ceph/logs/diffusion_3dmag/%j.out
# SBATCH --open-mode=append

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=6
# export CUDA_LAUNCH_BLOCKING=1

master_node=$SLURMD_NODENAME

source /mnt/home/framunno/sde_mag2mag_v2/sde_mag2mag_v2/bin/activate  
set -x

python3 train_3dmag.py

