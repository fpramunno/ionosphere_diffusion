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
from scipy.signal.windows import tukey
from powerbox import get_power


def revert_normalization(array):

    array = ((array + 1) / 2) * (80000 - (-80000)) + (-80000)

    return array

def compute_isotropic_power(x, boxlength=None, apply_window=True):
    """
    Compute isotropic 1D power spectrum on images.

    Args:
        x: np.ndarray of shape (..., H, W)
        boxlength: float — physical or normalized box size
        apply_window: bool — whether to apply Tukey window to reduce edge artifacts

    Returns:
        p: np.ndarray of shape (..., F) — power spectra
        k: np.ndarray of shape (F,) — frequency bins
    """
    *batch_shape, H, W = x.shape
    x = x.reshape(-1, H, W)  # treat all other dimensions as batch dimension

    # Precompute window
    window2d = 1.0
    if apply_window:
        w_h = tukey(H, alpha=0.5)
        w_w = tukey(W, alpha=0.5)
        window2d = np.outer(w_h, w_w)
    x = x * window2d
    
    # Compute power-spectrum
    p = []
    for i in range(x.shape[0]):
        pk, k = get_power(x[i], boxlength=boxlength or [H, W], bins_upto_boxlen=False)
        p.append(pk)
    p = np.array(p)
    
    p = p.reshape(batch_shape + [-1,])

    return p, k  # (..., F), (F,)

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
    power_spectra_differences = []
    for i in tqdm(range(len(files_gen)), desc="Evaluating files"):

        # Load predicted and ground truth frames
        print(f"Loading file {files_gen[i]} and {files_gt[i]}")
        predictions = np.load(os.path.join(directory, 'generated_imgs', files_gen[i]), allow_pickle=True)
        ground_truth = np.load(os.path.join(directory, 'ground_truth', files_gt[i]), allow_pickle=True)
        print(f"Shapes - Predictions: {predictions.shape}, Ground Truth: {ground_truth.shape}")

        psnr_frames = []
        ssim_frames = []
        physics_frames = []
        power_spectra_frames = []

        for t in range(num_frames):
            psnr_samples = []
            ssim_samples = []
            physics_samples = []
            power_spectra_samples = []
            gt_frame = ground_truth[0, t]

            # embed()

            gt_ps = compute_isotropic_power(gt_frame[np.newaxis, :, :], boxlength=[64, 64], apply_window=True)[0][0]

            for s in range(20):
                pred_frame = predictions[s, t]

                # Compute PSNR
                psnr_val = psnr(gt_frame, pred_frame, data_range=gt_frame.max() - gt_frame.min())
                psnr_samples.append(psnr_val)

                # Compute SSIM
                ssim_val = ssim(gt_frame, pred_frame, data_range=gt_frame.max() - gt_frame.min())
                ssim_samples.append(ssim_val)

                pred_frame_renorm = revert_normalization(pred_frame)
                gt_frame_renorm = revert_normalization(gt_frame)

                # embed()

                # Compute physics domain difference (max - min) for the entire sequence
                physics_diff = ((np.max(pred_frame_renorm) - np.min(pred_frame_renorm)) - (np.max(gt_frame_renorm) - np.min(gt_frame_renorm))) / (np.max(gt_frame_renorm) - np.min(gt_frame_renorm))
                physics_samples.append(physics_diff)

                # Compute power spectrum difference
                pred_ps = compute_isotropic_power(pred_frame[np.newaxis, :, :], boxlength=[64, 64], apply_window=True)[0][0]
                ps_diff = np.mean(np.abs(pred_ps - gt_ps) / gt_ps)
                power_spectra_samples.append(ps_diff)

            psnr_frames.extend(psnr_samples)
            ssim_frames.extend(ssim_samples)
            physics_frames.extend(physics_samples)
            power_spectra_frames.extend(power_spectra_samples)
        
        psnr_values.append(psnr_frames)
        ssim_values.append(ssim_frames)
        physics_differences.append(physics_frames)
        power_spectra_differences.append(power_spectra_frames)

    df = pd.DataFrame(columns=['PSNR', 'SSIM', 'Physics_Difference', 'Power_Spectra'])
    df['PSNR'] = psnr_values
    df['SSIM'] = ssim_values
    df['Physics_Difference'] = physics_differences
    df['Power_Spectra'] = power_spectra_differences

    df.to_csv(os.path.join(output_path_csv, 'evaluation_metrics.csv'), index=False)

if __name__ == "__main__":
    main()