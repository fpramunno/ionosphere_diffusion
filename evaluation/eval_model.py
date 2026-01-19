"""
Script to evaluate the performance of the ionosphere model over the following metrics:

Image Quality Metrics:
1) PSNR (Peak Signal-to-Noise Ratio) frame by frame
2) SSIM (Structural Similarity Index Measure) frame by frame

Physics Domain Metrics (on renormalized data):
3) Relative L2 Error: ||pred - gt||_2 / ||gt||_2
4) Normalized RMSE: RMSE / std(gt)
5) Mean Error: (mean(pred) - mean(gt)) / std(gt)
6) Std Ratio: std(pred) / std(gt)

CPCP Metrics (Cross-Polar Cap Potential = 99th - 1st percentile):
7) MAE: Mean Absolute Error of CPCP
8) RMSE: Root Mean Squared Error of CPCP
9) Normalized MAE: MAE / mean(gt_cpcp)
10) Normalized RMSE: RMSE / std(gt_cpcp)
11) Mean Bias: Mean signed error
12) Mean Bias %: (Mean bias / mean(gt_cpcp)) * 100
13) Correlation: Correlation between predicted and ground truth CPCP

Power Spectrum Metrics:
14) Log-space MSE: MSE of log10(power spectra)
15) Log-space Relative L2: relative error in log-space
16) Filtered Relative Error: relative error excluding low-power frequencies
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

def compute_physics_metrics(pred_renorm, gt_renorm):
    """
    Compute robust physics domain metrics.

    Returns:
        dict with multiple physics metrics
    """
    # Relative L2 error (normalized RMSE)
    rel_l2 = np.linalg.norm(pred_renorm - gt_renorm) / np.linalg.norm(gt_renorm)

    # Normalized RMSE (relative to std of ground truth)
    nrmse = np.sqrt(np.mean((pred_renorm - gt_renorm)**2)) / np.std(gt_renorm)

    # Statistical moments comparison
    mean_error = (np.mean(pred_renorm) - np.mean(gt_renorm)) / np.std(gt_renorm)
    std_ratio = np.std(pred_renorm) / np.std(gt_renorm)

    return {
        'rel_l2': rel_l2,
        'nrmse': nrmse,
        'mean_error': mean_error,
        'std_ratio': std_ratio
    }

def compute_cpcp_value(field_renorm):
    """
    Compute Cross-Polar Cap Potential (CPCP) using robust percentile method.

    CPCP = 99th percentile - 1st percentile (robust to outliers)

    Args:
        field_renorm: Renormalized ionospheric potential field

    Returns:
        float: CPCP value
    """
    cpcp = np.percentile(field_renorm, 99) - np.percentile(field_renorm, 1)
    return cpcp

def compute_cpcp_aggregate_metrics(pred_cpcps, gt_cpcps):
    """
    Compute aggregate CPCP metrics across multiple samples.

    Args:
        pred_cpcps: array of predicted CPCP values
        gt_cpcps: array of ground truth CPCP values

    Returns:
        dict with aggregate metrics
    """
    pred_cpcps = np.array(pred_cpcps)
    gt_cpcps = np.array(gt_cpcps)

    # 1. Overall metrics
    mae = np.mean(np.abs(pred_cpcps - gt_cpcps))
    rmse = np.sqrt(np.mean((pred_cpcps - gt_cpcps)**2))

    # 2. Normalized versions
    mean_gt_cpcp = np.mean(np.abs(gt_cpcps))
    std_gt_cpcp = np.std(gt_cpcps)

    normalized_mae = mae / mean_gt_cpcp if mean_gt_cpcp > 0 else 0.0
    normalized_rmse = rmse / std_gt_cpcp if std_gt_cpcp > 0 else 0.0

    # 3. Bias
    mean_bias = np.mean(pred_cpcps - gt_cpcps)
    mean_bias_pct = (mean_bias / mean_gt_cpcp * 100) if mean_gt_cpcp > 0 else 0.0

    # 4. Correlation (only if we have variation in gt)
    if len(pred_cpcps) > 1 and np.std(gt_cpcps) > 0:
        correlation = np.corrcoef(pred_cpcps, gt_cpcps)[0, 1]
    else:
        correlation = np.nan

    return {
        'cpcp_mae': mae,
        'cpcp_rmse': rmse,
        'cpcp_normalized_mae': normalized_mae,
        'cpcp_normalized_rmse': normalized_rmse,
        'cpcp_mean_bias': mean_bias,
        'cpcp_mean_bias_pct': mean_bias_pct,
        'cpcp_correlation': correlation
    }

def compute_power_spectrum_metric(pred_ps, gt_ps, epsilon=1e-10):
    """
    Compute robust power spectrum comparison in log-space.

    This avoids instability from dividing by small values.
    """
    # Log-space MSE (more stable for wide dynamic range)
    log_mse = np.mean((np.log10(pred_ps + epsilon) - np.log10(gt_ps + epsilon))**2)

    # Relative L2 error in log-space
    log_rel_l2 = np.sqrt(np.sum((np.log10(pred_ps + epsilon) - np.log10(gt_ps + epsilon))**2)) / \
                 np.sqrt(np.sum((np.log10(gt_ps + epsilon))**2))

    # Alternative: filter out low-power frequencies and compute relative error
    # Only use frequencies where gt_ps is significant
    threshold = np.max(gt_ps) * 1e-3  # 0.1% of max power
    mask = gt_ps > threshold
    if np.sum(mask) > 0:
        rel_error_filtered = np.mean(np.abs(pred_ps[mask] - gt_ps[mask]) / gt_ps[mask])
    else:
        rel_error_filtered = np.nan

    return {
        'log_mse': log_mse,
        'log_rel_l2': log_rel_l2,
        'rel_error_filtered': rel_error_filtered
    }

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
    p.add_argument("--name_csv", type=str, default="evaluation_metrics_cross_att_physics.csv", help="Name of the output CSV file")
    args = p.parse_args()

    # Path to your directory
    directory = args.directory
    output_path_csv = args.output_path_csv
    name_csv = args.name_csv

    # Get all filenames in the directory
    files_gen = os.listdir(os.path.join(directory, 'generated_imgs'))
    files_gt = os.listdir(os.path.join(directory, 'ground_truth'))

    files_gen = sorted(files_gen)
    files_gt = sorted(files_gt)

    num_frames = args.num_frames

    psnr_values = []
    ssim_values = []

    # Physics metrics
    physics_rel_l2 = []
    physics_nrmse = []
    physics_mean_error = []
    physics_std_ratio = []

    # CPCP values (to compute aggregate metrics later)
    cpcp_pred_values = []
    cpcp_gt_values = []

    # Power spectrum metrics
    ps_log_mse = []
    ps_log_rel_l2 = []
    ps_rel_error_filtered = []

    for i in tqdm(range(len(files_gen)), desc="Evaluating files"):

        # Load predicted and ground truth frames
        print(f"Loading file {files_gen[i]} and {files_gt[i]}")
        predictions = np.load(os.path.join(directory, 'generated_imgs', files_gen[i]), allow_pickle=True)
        ground_truth = np.load(os.path.join(directory, 'ground_truth', files_gt[i]), allow_pickle=True)
        print(f"Shapes - Predictions: {predictions.shape}, Ground Truth: {ground_truth.shape}")

        psnr_frames = []
        ssim_frames = []

        phys_rel_l2_frames = []
        phys_nrmse_frames = []
        phys_mean_err_frames = []
        phys_std_ratio_frames = []

        cpcp_pred_frames = []
        cpcp_gt_frames = []

        ps_log_mse_frames = []
        ps_log_rel_l2_frames = []
        ps_rel_filt_frames = []

        for t in range(num_frames):
            psnr_samples = []
            ssim_samples = []

            phys_rel_l2_samples = []
            phys_nrmse_samples = []
            phys_mean_err_samples = []
            phys_std_ratio_samples = []

            cpcp_pred_samples = []
            cpcp_gt_samples = []

            ps_log_mse_samples = []
            ps_log_rel_l2_samples = []
            ps_rel_filt_samples = []

            gt_frame = ground_truth[0, t]
            gt_frame_renorm = revert_normalization(gt_frame)
            gt_ps = compute_isotropic_power(gt_frame[np.newaxis, :, :], boxlength=[64, 64], apply_window=True)[0][0]

            for s in range(20):
                pred_frame = predictions[s, t]

                # Compute PSNR
                psnr_val = psnr(gt_frame, pred_frame, data_range=gt_frame.max() - gt_frame.min())
                psnr_samples.append(psnr_val)

                # Compute SSIM
                ssim_val = ssim(gt_frame, pred_frame, data_range=gt_frame.max() - gt_frame.min())
                ssim_samples.append(ssim_val)

                # Compute improved physics domain metrics
                pred_frame_renorm = revert_normalization(pred_frame)
                phys_metrics = compute_physics_metrics(pred_frame_renorm, gt_frame_renorm)
                phys_rel_l2_samples.append(phys_metrics['rel_l2'])
                phys_nrmse_samples.append(phys_metrics['nrmse'])
                phys_mean_err_samples.append(phys_metrics['mean_error'])
                phys_std_ratio_samples.append(phys_metrics['std_ratio'])

                # Compute CPCP values (Cross-Polar Cap Potential)
                cpcp_pred = compute_cpcp_value(pred_frame_renorm)
                cpcp_gt = compute_cpcp_value(gt_frame_renorm)
                cpcp_pred_samples.append(cpcp_pred)
                cpcp_gt_samples.append(cpcp_gt)

                # Compute improved power spectrum metrics
                pred_ps = compute_isotropic_power(pred_frame[np.newaxis, :, :], boxlength=[64, 64], apply_window=True)[0][0]
                ps_metrics = compute_power_spectrum_metric(pred_ps, gt_ps)
                ps_log_mse_samples.append(ps_metrics['log_mse'])
                ps_log_rel_l2_samples.append(ps_metrics['log_rel_l2'])
                ps_rel_filt_samples.append(ps_metrics['rel_error_filtered'])

            psnr_frames.extend(psnr_samples)
            ssim_frames.extend(ssim_samples)

            phys_rel_l2_frames.extend(phys_rel_l2_samples)
            phys_nrmse_frames.extend(phys_nrmse_samples)
            phys_mean_err_frames.extend(phys_mean_err_samples)
            phys_std_ratio_frames.extend(phys_std_ratio_samples)

            cpcp_pred_frames.extend(cpcp_pred_samples)
            cpcp_gt_frames.extend(cpcp_gt_samples)

            ps_log_mse_frames.extend(ps_log_mse_samples)
            ps_log_rel_l2_frames.extend(ps_log_rel_l2_samples)
            ps_rel_filt_frames.extend(ps_rel_filt_samples)

        psnr_values.append(psnr_frames)
        ssim_values.append(ssim_frames)

        physics_rel_l2.append(phys_rel_l2_frames)
        physics_nrmse.append(phys_nrmse_frames)
        physics_mean_error.append(phys_mean_err_frames)
        physics_std_ratio.append(phys_std_ratio_frames)

        cpcp_pred_values.append(cpcp_pred_frames)
        cpcp_gt_values.append(cpcp_gt_frames)

        ps_log_mse.append(ps_log_mse_frames)
        ps_log_rel_l2.append(ps_log_rel_l2_frames)
        ps_rel_error_filtered.append(ps_rel_filt_frames)

    # Compute aggregate CPCP metrics for each file
    cpcp_mae_list = []
    cpcp_rmse_list = []
    cpcp_normalized_mae_list = []
    cpcp_normalized_rmse_list = []
    cpcp_mean_bias_list = []
    cpcp_mean_bias_pct_list = []
    cpcp_correlation_list = []

    for pred_vals, gt_vals in zip(cpcp_pred_values, cpcp_gt_values):
        metrics = compute_cpcp_aggregate_metrics(pred_vals, gt_vals)
        cpcp_mae_list.append(metrics['cpcp_mae'])
        cpcp_rmse_list.append(metrics['cpcp_rmse'])
        cpcp_normalized_mae_list.append(metrics['cpcp_normalized_mae'])
        cpcp_normalized_rmse_list.append(metrics['cpcp_normalized_rmse'])
        cpcp_mean_bias_list.append(metrics['cpcp_mean_bias'])
        cpcp_mean_bias_pct_list.append(metrics['cpcp_mean_bias_pct'])
        cpcp_correlation_list.append(metrics['cpcp_correlation'])

    df = pd.DataFrame({
        'PSNR': psnr_values,
        'SSIM': ssim_values,
        'Physics_RelL2': physics_rel_l2,
        'Physics_NRMSE': physics_nrmse,
        'Physics_MeanError': physics_mean_error,
        'Physics_StdRatio': physics_std_ratio,
        'CPCP_MAE': cpcp_mae_list,
        'CPCP_RMSE': cpcp_rmse_list,
        'CPCP_NormalizedMAE': cpcp_normalized_mae_list,
        'CPCP_NormalizedRMSE': cpcp_normalized_rmse_list,
        'CPCP_MeanBias': cpcp_mean_bias_list,
        'CPCP_MeanBiasPct': cpcp_mean_bias_pct_list,
        'CPCP_Correlation': cpcp_correlation_list,
        'PS_LogMSE': ps_log_mse,
        'PS_LogRelL2': ps_log_rel_l2,
        'PS_RelErrorFiltered': ps_rel_error_filtered
    })

    df.to_csv(os.path.join(output_path_csv, name_csv), index=False)
    print(f"\nMetrics saved to {os.path.join(output_path_csv, name_csv)}")
    print(f"\nSummary statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()