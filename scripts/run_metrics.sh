#!/bin/bash
#SBATCH --gres=gpu:0 
# SBATCH --nodelist=server0099,server0103,server0105,server0107,server0109,server0094
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="compute_metrics"
#SBATCH --error=err_compute_metrics.log
#SBATCH --out=out_compute_metrics.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Run the metrics computation script
python3 compute_metrics.py \
    --results_dir "/mnt/nas05/data01/francesco/progetto_simone/results_1frame_test" \
    --output_file "/mnt/nas05/data01/francesco/progetto_simone/metrics_1frame_test/metrics_results.csv" \
    --save_plots
