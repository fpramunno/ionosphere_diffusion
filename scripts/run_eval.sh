#!/bin/bash                                                                                                                                                                                           
#SBATCH --job-name=run_eval
#SBATCH --partition=normal                                                                                                                                                                            
#SBATCH --nodes=1                                                                                                                                                                                   
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=09:00:00
#SBATCH -A sk035
#SBATCH --output=/capstor/scratch/cscs/framunno/logs/out/out_runeval_CLASSIC_MEAN_v4_interdata.log
#SBATCH --error=/capstor/scratch/cscs/framunno/logs/err/err_runeval_CLASSIC_MEAN_v4_interdata.log

source /users/framunno/envs/ionosphere/bin/activate

python3 /users/framunno/projects/ionosphere_diffusion/evaluation/eval_model.py \
    --directory /capstor/scratch/cscs/framunno/results_ViT_800mln_CLASSIC_MEAN_v4_075_new_epoch160 \
    --num_frames 15 \
    --output_path_csv /users/framunno/res_iono \
    --name_csv evaluation_metrics_CLASSIC_MEAN_v4_interdata.csv