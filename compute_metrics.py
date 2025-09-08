import sys
sys.path.append("/mnt/nas05/data01/francesco/progetto_simone/ionosphere")
import torch
import src as K
import argparse
import numpy as np
from src.data.dataset import get_sequence_data_objects
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import glob
from collections import defaultdict
import pandas as pd

def compute_metrics(pred, target):
    """
    Compute various metrics between prediction and target
    
    Args:
        pred: predicted image (numpy array)
        target: target image (numpy array)
    
    Returns:
        dict: dictionary containing all computed metrics
    """
    # Ensure both arrays have the same shape
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"
    
    # Flatten arrays for some metrics
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Basic metrics
    mse = np.mean((pred_flat - target_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - target_flat))
    rmse = np.sqrt(mse)
    
    # Normalized metrics
    mse_norm = mse / (np.var(target_flat) + 1e-8)
    
    # PSNR
    max_val = np.max(target_flat)
    psnr_val = psnr(target, pred, data_range=max_val)
    
    # SSIM
    ssim_val = ssim(target, pred, data_range=max_val)
    
    # Correlation coefficient
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # Relative error
    relative_error = np.mean(np.abs(pred_flat - target_flat) / (np.abs(target_flat) + 1e-8))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mse_norm': mse_norm,
        'psnr': psnr_val,
        'ssim': ssim_val,
        'correlation': correlation,
        'relative_error': relative_error
    }

def load_generated_samples(results_dir):
    """
    Load all generated samples from the results directory
    
    Args:
        results_dir: path to results directory
    
    Returns:
        dict: dictionary mapping sample index to generated data
    """
    samples = {}
    pattern = os.path.join(results_dir, "sample_forecasting_*.npy")
    
    for file_path in glob.glob(pattern):
        # Extract index from filename
        filename = os.path.basename(file_path)
        index = int(filename.split('_')[-1].split('.')[0])
        
        # Load the sample
        sample_data = np.load(file_path)
        samples[index] = sample_data
    
    return samples

def main():
    # Setup argument parser
    p = argparse.ArgumentParser(description='Compute metrics between generated samples and targets')
    p.add_argument('--results_dir', type=str, 
                   default="/mnt/nas05/data01/francesco/progetto_simone/results_1frame_test",
                   help='Directory containing generated samples')
    p.add_argument('--output_file', type=str, default='metrics_results.csv',
                   help='Output file for metrics results')
    p.add_argument('--save_plots', action='store_true',
                   help='Save comparison plots')
    
    args = p.parse_args()
    
    # Create output directories if they don't exist
    metrics_dir = "/mnt/nas05/data01/francesco/progetto_simone/metrics_1frame_test/"
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"Created/verified metrics directory: {metrics_dir}")
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset, val_sampler, val_dl = get_sequence_data_objects(
        csv_path="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/npy_metrics.csv",
        transform_cond_csv="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/params.csv",
        batch_size=1,
        distributed=False,
        num_data_workers=1,
        split='test',
        seed=42,
        sequence_length=30
    )
    
    # Load generated samples
    print("Loading generated samples...")
    generated_samples = load_generated_samples(args.results_dir)
    print(f"Found {len(generated_samples)} generated samples")
    
    if len(generated_samples) == 0:
        print("No generated samples found!")
        return
    
    # Compute metrics for each sample
    all_metrics = []
    
    print("Computing metrics...")
    for k, batch in enumerate(tqdm(val_dl, desc="Computing metrics")):
        if k not in generated_samples:
            print(f"Warning: No generated sample found for index {k}")
            continue
            
        # Get target from validation data
        inpt = batch[0].contiguous().float()
        inpt = inpt.squeeze(2)  # shape: (8, 120, 24, 360)
        target_img = inpt[0, 29, :, :]  # target is the 30th frame (index 29)
        
        # Get generated sample
        generated_sample = generated_samples[k][0][0]
        
        print(f"Processing sample {k}:")
        print(f"  Target shape: {target_img.shape}")
        print(f"  Generated shape: {generated_sample.shape}")
        
        # Ensure shapes match
        if generated_sample.shape != target_img.shape:
            print(f"Shape mismatch for sample {k}: generated {generated_sample.shape}, target {target_img.shape}")
            continue
        
        # Compute metrics
        metrics = compute_metrics(generated_sample, target_img.numpy())
        metrics['sample_index'] = k
        
        all_metrics.append(metrics)
        
        # Save comparison plots if requested
        # if args.save_plots:
        print('Saving comparison plots...')
        save_comparison_plot(generated_sample, target_img.numpy(), k, "/mnt/nas05/data01/francesco/progetto_simone/metrics_1frame_test/")
    
    # Compute summary statistics
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Save detailed results
        df.to_csv(args.output_file, index=False)
        print(f"Detailed metrics saved to {args.output_file}")
        
        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        numeric_columns = [col for col in df.columns if col != 'sample_index']
        
        for metric in numeric_columns:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"{metric}:")
                print(f"  Mean: {values.mean():.6f}")
                print(f"  Std:  {values.std():.6f}")
                print(f"  Min:  {values.min():.6f}")
                print(f"  Max:  {values.max():.6f}")
                print()
        
        # Save summary statistics
        summary_stats = df[numeric_columns].describe()
        summary_file = args.output_file.replace('.csv', '_summary.csv')
        summary_stats.to_csv(summary_file)
        print(f"Summary statistics saved to {summary_file}")
        
    else:
        print("No metrics computed!")

def save_comparison_plot(generated, target, index, results_dir):
    """
    Save comparison plots between generated and target images
    
    Args:
        generated: generated image array
        target: target image array
        index: sample index
        results_dir: directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot target
    im1 = axes[0].imshow(target, cmap='viridis', aspect='auto')
    axes[0].set_title('Target')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot generated
    im2 = axes[1].imshow(generated, cmap='viridis', aspect='auto')
    axes[1].set_title('Generated')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot difference
    diff = generated - target
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto')
    axes[2].set_title('Difference (Generated - Target)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(results_dir, f'comparison_{index}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Verify the file was created
    if os.path.exists(plot_path):
        print(f"Saved comparison plot: {plot_path}")
    else:
        print(f"Warning: Failed to save plot to {plot_path}")

if __name__ == "__main__":
    main() 