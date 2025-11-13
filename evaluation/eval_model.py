"""
Script to evaluate the performance of the ionosphere model over the following metrics:

1) PSNR (Peak Signal-to-Noise Ratio) frame by frame
2) SSIM (Structural Similarity Index Measure) frame by frame
3) Difference between max and min values per frame in the phsyics domain
4) Fréchet Video Distance (FVD) between ground truth and predicted sequences
"""

import argparse
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from IPython import embed
import os
import pandas as pd
from tqdm import tqdm

def main():
    
    p = argparse.ArgumentParser(description="Evaluate ionosphere model performance")
    p.add_argument("--directory", type=str, required=True, help="Directory containing generated, input, and ground truth images")
    p.add_argument("--device", type=str, default="cuda:0", help="Device to use for FVD computation")
    p.add_argument("--num_frames", type=int, default=15, help="Number of frames predicted by the model")
    p.add_argument("--output_path_csv", type=str, default="evaluation_metrics.csv", help="Path to save the evaluation metrics CSV file")
    args = p.parse_args()

    # Path to your directory
    directory = args.directory
    output_path_csv = args.output_path_csv

    # Get all filenames in the directory
    files_gen = os.listdir(os.path.join(directory, 'generated_imgs'))
    files_gt = os.listdir(os.path.join(directory, 'ground_truth'))

    files_gen = sorted(files_gen)
    files_gt = sorted(files_gt)

    num_frames = args.num_frames

    psnr_values = []
    ssim_values = []
    physics_differences = []

    for i in tqdm(range(len(files_gen)), desc="Evaluating files"):

        # Load predicted and ground truth frames
        print(f"Loading file {files_gen[i]} and {files_gt[i]}")
        predictions = np.load(os.path.join(directory, 'generated_imgs', files_gen[i]), allow_pickle=True)
        ground_truth = np.load(os.path.join(directory, 'ground_truth', files_gt[i]), allow_pickle=True)
        print(f"Shapes - Predictions: {predictions.shape}, Ground Truth: {ground_truth.shape}")

        psnr_frames = []
        ssim_frames = []
        physics_frames = []
        for t in range(num_frames):
            psnr_samples = []
            ssim_samples = []
            physics_samples = []

            gt_frame = ground_truth[0, t]

            for s in range(20):
                pred_frame = predictions[s, t]

                # Compute PSNR
                psnr_val = psnr(gt_frame, pred_frame, data_range=gt_frame.max() - gt_frame.min())
                psnr_samples.append(psnr_val)

                # Compute SSIM
                ssim_val = ssim(gt_frame, pred_frame, data_range=gt_frame.max() - gt_frame.min())
                ssim_samples.append(ssim_val)

                # Compute physics domain difference (max - min) for the entire sequence
                physics_diff = ((np.max(pred_frame) - np.min(pred_frame)) - (np.max(gt_frame) - np.min(gt_frame))) / (np.max(gt_frame) - np.min(gt_frame))
                physics_samples.append(physics_diff)

            psnr_frames.append(psnr_samples)
            ssim_frames.append(ssim_samples)
            physics_frames.append(physics_samples) 
        
        psnr_values.append(psnr_frames)
        ssim_values.append(ssim_frames)
        physics_differences.append(physics_frames)

    df = pd.DataFrame(columns=['PSNR', 'SSIM', 'Physics_Difference'])
    df['PSNR'] = psnr_values
    df['SSIM'] = ssim_values
    df['Physics_Difference'] = physics_differences

    df.to_csv(os.path.join(output_path_csv, 'evaluation_metrics.csv'), index=False)

if __name__ == "__main__":
    main()