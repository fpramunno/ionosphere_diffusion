#!/bin/bash
#SBATCH --job-name=metrics_v3_singlepass
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -A sk035
#SBATCH --output=./logs/out/out_metrics_v3_singlepass.log
#SBATCH --error=./logs/err/err_metrics_v3_singlepass.log

source ${IONO_VENV:-/path/to/venv}/bin/activate
export PYTHONUNBUFFERED=1

mkdir -p ${IONO_DATA_ROOT:-/path/to/data_root}/logs/out
mkdir -p ${IONO_DATA_ROOT:-/path/to/data_root}/logs/err

BASE=${IONO_DATA_ROOT:-/path/to/data_root}

for DYN in high low; do
    for ACT in high low; do
        echo ">>> ============================================================"
        echo ">>> metrics v3 singlepass  dyn=${DYN}  act=${ACT}"
        python ${IONO_REPO:-/path/to/ionosphere_diffusion}/compute_metrics_v3.py \
            --results-dirs \
                ${BASE}/results_diffusion_classic_singlepass_step200k_dyn${DYN}_act${ACT} \
                ${BASE}/results_diffusion_nocond_singlepass_step200k_dyn${DYN}_act${ACT} \
                ${BASE}/results_unet_singlepass_step200k_dyn${DYN}_act${ACT} \
            --labels classic nocond unet \
            --output-dir ${IONO_HOME_ROOT:-/path/to/home_root}/res_iono/metrics_v3_singlepass_dyn${DYN}_act${ACT} \
            --lpips-net alex
    done
done
