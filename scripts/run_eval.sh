#!/bin/bash                                                                                                                                                                                           
#SBATCH --job-name=run_eval
#SBATCH --partition=normal                                                                                                                                                                            
#SBATCH --nodes=1                                                                                                                                                                                   
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=09:00:00
#SBATCH -A sk035
#SBATCH --output=./logs/out/out_runeval_CLASSIC_MEAN_v4_interdata.log
#SBATCH --error=./logs/err/err_runeval_CLASSIC_MEAN_v4_interdata.log

source ${IONO_VENV:-/path/to/venv}/bin/activate

python3 ${IONO_REPO:-/path/to/ionosphere_diffusion}/evaluation/eval_model.py \
    --directory ${IONO_DATA_ROOT:-/path/to/data_root}/results_ViT_800mln_CLASSIC_MEAN_v4_075_new_epoch160 \
    --num_frames 15 \
    --output_path_csv ${IONO_HOME_ROOT:-/path/to/home_root}/res_iono \
    --name_csv evaluation_metrics_CLASSIC_MEAN_v4_interdata.csv