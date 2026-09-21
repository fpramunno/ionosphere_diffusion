#!/bin/bash
#SBATCH --job-name=metrics_v3_ark50
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -A sk035
#SBATCH --output=/iopsstor/scratch/cscs/framunno/logs/out/out_metrics_v3_ark50.log
#SBATCH --error=/iopsstor/scratch/cscs/framunno/logs/err/err_metrics_v3_ark50.log

source /users/framunno/envs/ionosphere/bin/activate
export PYTHONUNBUFFERED=1

mkdir -p /iopsstor/scratch/cscs/framunno/logs/out
mkdir -p /iopsstor/scratch/cscs/framunno/logs/err

BASE=/iopsstor/scratch/cscs/framunno

for DYN in high low; do
    for ACT in high low; do
        echo ">>> ============================================================"
        echo ">>> metrics v3 ark50  dyn=${DYN}  act=${ACT}"
        python /users/framunno/projects/ionosphere_diffusion/compute_metrics_v3.py \
            --results-dirs \
                ${BASE}/results_diffusion_classic_ark50_step200k_dyn${DYN}_act${ACT} \
                ${BASE}/results_diffusion_nocond_ark50_step200k_dyn${DYN}_act${ACT} \
                ${BASE}/results_unet_ark50_step200k_dyn${DYN}_act${ACT} \
            --labels classic nocond unet \
            --output-dir /users/framunno/res_iono/metrics_v3_ark50_dyn${DYN}_act${ACT} \
            --lpips-net alex
    done
done
