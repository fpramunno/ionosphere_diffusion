#!/usr/bin/env python3
"""
Multi-Window Diffusion Model Generation Script

Usage:
python generate_multi_window.py --model_ckpt model.pth --config config.json --output_dir results/ --num_generations 5
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import src as K
import argparse
import numpy as np
from copy import deepcopy
from util import generate_samples
from src.data.dataset import get_sequence_data_objects
from tqdm import tqdm
import json
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate samples from diffusion models with different prediction windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model_ckpt', type=str, required=True,
                       help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the model configuration (.json file)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save generated samples')
    
    # Optional arguments
    parser.add_argument('--num_generations', type=int, default=5,
                       help='Number of generations per input (for uncertainty estimation)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for generation (recommended: 1)')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of input samples to process')
    parser.add_argument('--sampler', type=str, default='dpmpp_2m_sde',
                       choices=['euler', 'euler_ancestral', 'heun', 'dpm_2', 
                               'dpm_2_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde'],
                       help='Sampling method to use')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detected if not specified')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Dataset arguments
    parser.add_argument('--csv_path', type=str, 
                       default="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/npy_metrics.csv",
                       help='Path to the dataset CSV file')
    parser.add_argument('--transform_cond_csv', type=str,
                       default="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/params.csv",
                       help='Path to the condition transformation CSV file')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--conditioning_frames', type=int, default=15,
                       help='Number of conditioning frames (input sequence length)')
    
    return parser.parse_args()


def load_model(config_path, model_ckpt_path, device):
    """Load the diffusion model from checkpoint."""
    print(f"Loading model configuration from: {config_path}")
    config = K.config.load_config(config_path)
    
    print(f"Creating model architecture...")
    inner_model = K.config.make_model(config)
    inner_model_ema = deepcopy(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)
    
    print(f"Loading model weights from: {model_ckpt_path}")
    ckpt = torch.load(model_ckpt_path, map_location='cpu')
    model_ema.inner_model.load_state_dict(ckpt['model_ema'])
    model_ema.to(device)
    model_ema.eval()
    
    # Extract prediction window from config
    prediction_window = config['model'].get('out_channels', 1)
    
    return model_ema, prediction_window, config


def setup_dataset(args, prediction_window):
    """Setup the dataset and dataloader."""
    print(f"Setting up dataset...")
    print(f"  CSV path: {args.csv_path}")
    print(f"  Transform CSV: {args.transform_cond_csv}")
    print(f"  Split: {args.split}")
    print(f"  Conditioning frames: {args.conditioning_frames}")
    
    # Calculate total sequence length (conditioning + prediction frames)
    total_sequence_length = args.conditioning_frames + prediction_window #15 
    
    dataset, sampler, dataloader = get_sequence_data_objects(
        csv_path=args.csv_path,
        transform_cond_csv=args.transform_cond_csv,
        batch_size=args.batch_size,
        distributed=False,
        num_data_workers=1,
        split=args.split,
        seed=args.seed,
        sequence_length=total_sequence_length
    )
    
    return dataset, dataloader


def generate_multiple_samples(model, conditioning_input, condition_labels, 
                            prediction_frames, num_generations, args, device):
    """Generate multiple samples for uncertainty estimation."""
    all_samples = []
    
    for gen_idx in range(num_generations):
        # Set different seed for each generation to get diverse samples
        torch.manual_seed(args.seed + gen_idx)
        np.random.seed(args.seed + gen_idx)
        
        samples = generate_samples(
            model=model,
            num_samples=1,
            device=device,
            cond_label=condition_labels,
            sampler=args.sampler,
            step=args.steps,
            cond_img=conditioning_input,
            num_pred_frames=prediction_frames
        )
        
        all_samples.append(samples.cpu())
    
    # Stack all samples along a new dimension [num_generations, pred_frames, H, W]
    all_samples = torch.stack(all_samples, dim=0)
    
    return all_samples


def save_results(sample_idx, conditioning_input, target_frames, condition_labels, 
                generated_samples, prediction_window, output_dir, dataset):
    """Save generation results to disk."""
    # Create sample-specific directory
    sample_dir = output_dir / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(exist_ok=True)
    
    # Save conditioning frames (input)
    np.save(sample_dir / "conditioning_input.npy", conditioning_input.cpu().numpy())
    
    # Save ground truth frames for evaluation
    np.save(sample_dir / "ground_truth_frames.npy", target_frames.cpu().numpy())
    
    # Save condition labels (normalized)
    np.save(sample_dir / "condition_labels_normalized.npy", condition_labels.cpu().numpy())
    
    # Save condition labels (original scale) for visualization
    cond_original = dataset.revert_condition_normalization(condition_labels[0].cpu())
    np.save(sample_dir / "condition_labels_original.npy", cond_original.numpy())
    
    # Save generated samples [num_generations, pred_frames, H, W]
    np.save(sample_dir / "generated_samples.npy", generated_samples.numpy())
    
    # Calculate and save uncertainty metrics
    mean_sample = generated_samples.mean(dim=0)  # [pred_frames, H, W]
    std_sample = generated_samples.std(dim=0)    # [pred_frames, H, W]
    
    np.save(sample_dir / "generated_mean.npy", mean_sample.numpy())
    np.save(sample_dir / "generated_std.npy", std_sample.numpy())
    
    # Save metadata
    metadata = {
        "sample_idx": int(sample_idx),
        "prediction_window": int(prediction_window),
        "num_generations": int(generated_samples.shape[0]),
        "conditioning_frames": int(conditioning_input.shape[1]),
        "spatial_shape": list(conditioning_input.shape[-2:]),
        "condition_params": {
            "float1": float(cond_original[0, 0]),
            "float2": float(cond_original[0, 1]), 
            "float3": float(cond_original[0, 2]),
            "float4": float(cond_original[0, 3])  # velocity
        }
    }
    
    with open(sample_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return sample_dir


def main():
    """Main generation function."""
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Arguments: {vars(args)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments for reproducibility
    with open(output_dir / "generation_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load model
    model, prediction_window, config = load_model(args.config, args.model_ckpt, device)
    
    print(f"Model prediction window: {prediction_window}")
    
    # Setup dataset
    dataset, dataloader = setup_dataset(args, prediction_window)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Starting generation...")
    
    # Generation loop
    processed_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating samples")):
            if processed_samples >= args.max_samples:
                break
                
            # Extract data from batch
            sequence_data = batch[0].contiguous().float().to(device)  # [batch, seq_len, H, W]
            condition_labels = batch[1].to(device)  # [batch, seq_len, 4]
            
            # Remove channel dimension if present
            if sequence_data.dim() == 5:
                sequence_data = sequence_data.squeeze(2)
            
            batch_size = sequence_data.shape[0]
            
            for i in range(batch_size):
                if processed_samples >= args.max_samples:
                    break
                    
                # Extract single sample
                single_sequence = sequence_data[i]  # [seq_len, H, W]
                single_conditions = condition_labels[i]  # [seq_len, 4]
                
                # Split into conditioning and target frames
                conditioning_input = single_sequence[:args.conditioning_frames]  # [cond_frames, H, W]
                target_frames = single_sequence[args.conditioning_frames:args.conditioning_frames+prediction_window]
                
                # Prepare condition labels (take average over conditioning frames)
                cond_labels_input = single_conditions[:args.conditioning_frames]  # [cond_frames, 4]
                
                # Reshape for generation
                conditioning_input = conditioning_input.unsqueeze(0)  # [1, cond_frames, H, W]
                cond_labels_input = cond_labels_input.unsqueeze(0)    # [1, cond_frames, 4]
                
                print(f"\nGenerating sample {processed_samples + 1}/{min(args.max_samples, len(dataset))}")
                print(f"  Conditioning frames: {conditioning_input.shape}")
                print(f"  Prediction window: {prediction_window}")
                print(f"  Target frames shape: {target_frames.shape}")
                
                # Generate multiple samples
                generated_samples = generate_multiple_samples(
                    model=model,
                    conditioning_input=conditioning_input,
                    condition_labels=cond_labels_input,
                    prediction_frames=prediction_window,
                    num_generations=args.num_generations,
                    args=args,
                    device=device
                )
                
                print(f"  Generated samples shape: {generated_samples.shape}")
                
                # Save results
                sample_dir = save_results(
                    sample_idx=processed_samples,
                    conditioning_input=conditioning_input[0],
                    target_frames=target_frames,
                    condition_labels=cond_labels_input,
                    generated_samples=generated_samples.squeeze(1),  # Remove batch dim
                    prediction_window=prediction_window,
                    output_dir=output_dir,
                    dataset=dataset
                )
                
                print(f"  Saved to: {sample_dir}")
                
                processed_samples += 1
    
    print(f"\nGeneration complete!")
    print(f"Generated {processed_samples} samples")
    print(f"Results saved to: {output_dir}")
    
    # Save generation summary
    summary = {
        "total_samples": processed_samples,
        "prediction_window": prediction_window,
        "num_generations_per_sample": args.num_generations,
        "conditioning_frames": args.conditioning_frames,
        "model_config": config,
        "generation_args": vars(args)
    }
    
    with open(output_dir / "generation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {output_dir}/generation_summary.json")


if __name__ == "__main__":
    main()