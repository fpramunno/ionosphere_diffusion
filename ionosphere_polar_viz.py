#!/usr/bin/env python3
"""
Ionosphere Polar Plot Visualization Script

Simple script to create polar plots of ionosphere data.
Compatible with the existing ionosphere project structure.

Usage:
    python ionosphere_polar_viz.py --csv /path/to/csv --animate
    python ionosphere_polar_viz.py --data /path/to/data.npy --single
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
import imageio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.data.dataset import get_sequence_data_objects
except ImportError:
    print("Warning: Could not import dataset module")

# Set high quality plots
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'


def plot_polar_ionosphere(data, title="Ionosphere TEC", cmap='plasma', figsize=(8, 8)):
    """
    Plot ionosphere data in polar coordinates.
    
    Args:
        data: 2D array (24, 360) - ionosphere data
        title: Plot title
        cmap: Colormap
        figsize: Figure size
    """
    # Convert tensor to numpy if needed
    if hasattr(data, 'numpy'):
        data = data.numpy()
    if hasattr(data, 'squeeze'):
        data = data.squeeze()
    
    # Create polar coordinates
    theta = np.linspace(0, 2*np.pi, data.shape[1])  # 360 longitude points
    r = np.linspace(0, 1, data.shape[0])  # 24 latitude points (normalized)
    
    # Create meshgrid
    Theta, R = np.meshgrid(theta, r)
    
    # Create polar plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Plot data
    im = ax.pcolormesh(Theta, R, data, cmap=cmap, shading='auto')
    
    # Configure polar plot
    ax.set_theta_zero_location('N')  # North at top
    ax.set_theta_direction(-1)       # Clockwise
    ax.set_ylim(0.1, 1)             # Avoid white dot at center
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('TEC Value', rotation=270, labelpad=20)
    
    # Set title
    ax.set_title(title, pad=30, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def create_polar_animation(data_sequence, output_path, duration=0.5):
    """
    Create animated GIF of ionosphere polar plots.
    
    Args:
        data_sequence: Sequence of data arrays [seq_len, 24, 360]
        output_path: Path to save GIF
        duration: Frame duration in seconds
    """
    if hasattr(data_sequence, 'numpy'):
        data_sequence = data_sequence.numpy()
    
    if data_sequence.ndim == 4:  # Remove channel dim if present
        data_sequence = data_sequence.squeeze(1)
    
    frames = []
    seq_len = data_sequence.shape[0]
    
    # Find global min/max for consistent coloring
    vmin, vmax = data_sequence.min(), data_sequence.max()
    
    print(f"Creating animation with {seq_len} frames...")
    
    for i in range(seq_len):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Create coordinates
        theta = np.linspace(0, 2*np.pi, data_sequence.shape[2])
        r = np.linspace(0, 1, data_sequence.shape[1])
        Theta, R = np.meshgrid(theta, r)
        
        # Plot with consistent color scale
        im = ax.pcolormesh(Theta, R, data_sequence[i], 
                          cmap='plasma', vmin=vmin, vmax=vmax, shading='auto')
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0.1, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Ionosphere Frame {i+1}/{seq_len}', pad=30, fontsize=14)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('TEC Value', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig)
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{seq_len} frames")
    
    # Save GIF
    imageio.mimsave(output_path, frames, duration=duration*1000, loop=0)
    print(f"Animation saved to {output_path}")


def plot_sequence_grid(data_sequence, max_frames=9, figsize=(15, 10)):
    """
    Plot multiple frames in a grid layout.
    
    Args:
        data_sequence: Sequence of data arrays
        max_frames: Maximum number of frames to show
        figsize: Figure size
    """
    if hasattr(data_sequence, 'numpy'):
        data_sequence = data_sequence.numpy()
    
    if data_sequence.ndim == 4:
        data_sequence = data_sequence.squeeze(1)
    
    n_frames = min(data_sequence.shape[0], max_frames)
    cols = 3
    rows = (n_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, 
                           subplot_kw=dict(projection='polar'))
    
    if n_frames == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    # Global color scale
    vmin, vmax = data_sequence[:n_frames].min(), data_sequence[:n_frames].max()
    
    for i in range(n_frames):
        theta = np.linspace(0, 2*np.pi, data_sequence.shape[2])
        r = np.linspace(0, 1, data_sequence.shape[1])
        Theta, R = np.meshgrid(theta, r)
        
        im = axes[i].pcolormesh(Theta, R, data_sequence[i], 
                               cmap='plasma', vmin=vmin, vmax=vmax, shading='auto')
        
        axes[i].set_theta_zero_location('N')
        axes[i].set_theta_direction(-1)
        axes[i].set_ylim(0.1, 1)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'Frame {i+1}', fontsize=12)
        
        # Small colorbar for each subplot
        cbar = fig.colorbar(im, ax=axes[i], shrink=0.6, aspect=15)
        cbar.set_label('TEC', rotation=270, labelpad=15, fontsize=8)
    
    # Hide unused subplots
    for i in range(n_frames, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def load_single_data(data_path):
    """Load single .npy file."""
    try:
        data = np.load(data_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and len(data) > 0:
            return data[0]  # Return the map data
        return data
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return None


def load_sequence_data(csv_path, sequence_length=16):
    """Load sequence data from CSV."""
    try:
        dataset, sampler, dataloader = get_sequence_data_objects(
            csv_path=csv_path,
            batch_size=1,
            distributed=False,
            num_data_workers=2,
            split='train',
            sequence_length=sequence_length,
            normalization_type="absolute_max",
            use_l1_conditions=True,
            seed=42
        )
        
        # Get first batch
        batch_iter = iter(dataloader)
        data_seq, cond_seq = next(batch_iter)
        
        return data_seq[0], cond_seq[0], dataset
    except Exception as e:
        print(f"Error loading sequence data: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Ionosphere Polar Visualization')
    parser.add_argument('--csv', type=str, help='CSV file path for sequence data')
    parser.add_argument('--data', type=str, help='NPY file path for single data')
    parser.add_argument('--output', type=str, default='./ionosphere_viz', help='Output directory')
    parser.add_argument('--animate', action='store_true', help='Create animation')
    parser.add_argument('--single', action='store_true', help='Plot single frame')
    parser.add_argument('--grid', action='store_true', help='Plot frames in grid')
    parser.add_argument('--seq-len', type=int, default=16, help='Sequence length')
    parser.add_argument('--max-frames', type=int, default=9, help='Max frames for grid')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.single and args.data:
        # Single data visualization
        data = load_single_data(args.data)
        if data is None:
            return
        
        fig, ax = plot_polar_ionosphere(data, f"Ionosphere - {Path(args.data).stem}")
        
        save_path = output_dir / f"ionosphere_single_{Path(args.data).stem}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Single plot saved to {save_path}")
        plt.show()
        
    elif args.csv:
        # Sequence data visualization
        data_seq, cond_seq, dataset = load_sequence_data(args.csv, args.seq_len)
        if data_seq is None:
            return
        
        # Revert normalization for visualization
        data_original = data_seq * 55000.0
        
        if args.animate:
            # Create animation
            save_path = output_dir / "ionosphere_animation.gif"
            create_polar_animation(data_original, save_path)
            
        elif args.grid:
            # Create grid plot
            fig, axes = plot_sequence_grid(data_original, args.max_frames)
            save_path = output_dir / "ionosphere_grid.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grid plot saved to {save_path}")
            plt.show()
            
        else:
            # Plot first frame only
            fig, ax = plot_polar_ionosphere(data_original[0], "Ionosphere Sequence - Frame 1")
            save_path = output_dir / "ionosphere_first_frame.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"First frame saved to {save_path}")
            plt.show()
            
    else:
        print("Please provide either --csv for sequence data or --data with --single for single frame")
        parser.print_help()


if __name__ == "__main__":
    main()