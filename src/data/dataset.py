import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
from torchvision.transforms import Compose, Normalize
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from typing import Iterator
from datetime import datetime, timedelta
from collections.abc import Sized
from IPython import embed
from scipy.interpolate import griddata

# Preprocessing config similar to SDO approach
IONOSPHERE_PREPROCESS = {
    "electric_potential": {
        "min": -80000,
        "max": 80000,
        "scaling": None,  # Options: None, "log10", "sqrt", "symlog"
    }
}

def get_ionosphere_transform(data_tensor, config=None):
    """
    Apply preprocessing transform to ionosphere data following SDO approach exactly.

    Pipeline (same as SDO):
    1. Clamp to [min, max]
    2. Apply scaling transformation (log10, sqrt, or none)
    3. Normalize to [0, 1] using (x - mean) / std
    4. Normalize to [-1, 1] using 2x - 1 (equivalent to Normalize(mean=0.5, std=0.5))

    Args:
        data_tensor: Input tensor
        config: Preprocessing config dict with 'min', 'max', 'scaling'

    Returns:
        Transformed tensor in range [-1, 1]
    """
    if config is None:
        config = IONOSPHERE_PREPROCESS["electric_potential"]

    # Step 1: Clamp to valid range
    data_clamped = torch.clamp(data_tensor,
                               min=config["min"],
                               max=config["max"])

    # Step 2: Apply scaling transformation and compute mean/std for normalization
    if config["scaling"] == "log10":
        # Symmetric log10 for data with negative values
        epsilon = 1.0
        sign = torch.sign(data_clamped)
        abs_data = torch.abs(data_clamped)
        data_transformed = sign * torch.log10(abs_data + epsilon)

        # Compute mean and std for normalization (maps to [0, 1])
        mean = np.sign(config["min"]) * np.log10(abs(config["min"]) + epsilon)
        std = np.sign(config["max"]) * np.log10(abs(config["max"]) + epsilon) - mean

    elif config["scaling"] == "sqrt":
        # Symmetric square root
        sign = torch.sign(data_clamped)
        abs_data = torch.abs(data_clamped)
        data_transformed = sign * torch.sqrt(abs_data)

        # Compute mean and std for normalization
        mean = np.sign(config["min"]) * np.sqrt(abs(config["min"]))
        std = np.sign(config["max"]) * np.sqrt(abs(config["max"])) - mean

    elif config["scaling"] == "symlog":
        # Symmetric log: sign(x) * log(1 + |x|/C) where C is a scale factor
        scale_factor = config.get("scale_factor", 10000)
        sign = torch.sign(data_clamped)
        abs_data = torch.abs(data_clamped)
        data_transformed = sign * torch.log1p(abs_data / scale_factor)

        # Compute mean and std for normalization
        mean = np.sign(config["min"]) * np.log1p(abs(config["min"]) / scale_factor)
        std = np.sign(config["max"]) * np.log1p(abs(config["max"]) / scale_factor) - mean

    else:  # No scaling
        data_transformed = data_clamped
        mean = config["min"]
        std = config["max"] - config["min"]

    # Step 3: First normalization - map to [0, 1]
    # Equivalent to: Normalize(mean=[mean], std=[std])
    data_normalized_01 = (data_transformed - mean) / std

    # Step 4: Second normalization - map [0, 1] to [-1, 1]
    # Equivalent to: Normalize(mean=0.5, std=0.5)
    # Which is: (x - 0.5) / 0.5 = 2x - 1
    data_normalized = (data_normalized_01 - 0.5) / 0.5

    return data_normalized


def reverse_ionosphere_transform(normalized_tensor, config=None):
    """
    Reverse the preprocessing transform to get back original scale.

    Reverses the exact SDO pipeline:
    1. Reverse second normalization: from [-1, 1] to [0, 1]
    2. Reverse first normalization: from [0, 1] to transformed range
    3. Reverse scaling transformation: from transformed to original

    Args:
        normalized_tensor: Normalized tensor in range [-1, 1]
        config: Preprocessing config dict

    Returns:
        Denormalized tensor in original scale
    """
    if config is None:
        config = IONOSPHERE_PREPROCESS["electric_potential"]

    # Step 1: Reverse second normalization - map [-1, 1] to [0, 1]
    # Reverse of: (x - 0.5) / 0.5
    data_01 = normalized_tensor * 0.5 + 0.5

    # Compute mean and std based on scaling type (same as forward)
    if config["scaling"] == "log10":
        epsilon = 1.0
        mean = np.sign(config["min"]) * np.log10(abs(config["min"]) + epsilon)
        std = np.sign(config["max"]) * np.log10(abs(config["max"]) + epsilon) - mean

    elif config["scaling"] == "sqrt":
        mean = np.sign(config["min"]) * np.sqrt(abs(config["min"]))
        std = np.sign(config["max"]) * np.sqrt(abs(config["max"])) - mean

    elif config["scaling"] == "symlog":
        scale_factor = config.get("scale_factor", 10000)
        mean = np.sign(config["min"]) * np.log1p(abs(config["min"]) / scale_factor)
        std = np.sign(config["max"]) * np.log1p(abs(config["max"]) / scale_factor) - mean

    else:  # No scaling
        mean = config["min"]
        std = config["max"] - config["min"]

    # Step 2: Reverse first normalization - map [0, 1] to transformed range
    # Reverse of: (x - mean) / std
    data_transformed = data_01 * std + mean

    # Step 3: Reverse scaling transformation
    if config["scaling"] == "log10":
        epsilon = 1.0
        sign = torch.sign(data_transformed)
        abs_log = torch.abs(data_transformed)
        data_original = sign * (torch.pow(10, abs_log) - epsilon)

    elif config["scaling"] == "sqrt":
        sign = torch.sign(data_transformed)
        abs_sqrt = torch.abs(data_transformed)
        data_original = sign * (abs_sqrt ** 2)

    elif config["scaling"] == "symlog":
        scale_factor = config.get("scale_factor", 10000)
        sign = torch.sign(data_transformed)
        abs_log = torch.abs(data_transformed)
        data_original = sign * scale_factor * (torch.exp(abs_log) - 1)

    else:  # No scaling
        data_original = data_transformed

    return data_original 

class MeanSigmaTanhNormalizer:
    """
    Normalizer that applies mean-sigma normalization followed by tanh.
    
    Forward: tanh((x - mean) / std)
    Reverse: atanh(y) * std + mean
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        
    def fit(self, data):
        """Fit the normalizer to the data by computing mean and std."""
        if isinstance(data, torch.Tensor):
            self.mean = data.mean().item()
            self.std = data.std().item()
        else:
            self.mean = float(np.mean(data))
            self.std = float(np.std(data))
        
        # Avoid division by zero
        if self.std == 0:
            self.std = 1.0
            
    def forward(self, x):
        """Apply normalization: tanh((x - mean) / std)"""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted first!")
        
        # Standardize then apply tanh
        normalized = (x - self.mean) / self.std
        return torch.tanh(normalized)
    
    def reverse(self, y):
        """Reverse normalization: atanh(y) * std + mean"""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted first!")
        
        # Clamp y to valid range for atanh to avoid numerical issues
        y_clamped = torch.clamp(y, -0.999999, 0.999999)
        
        # Apply inverse tanh then denormalize
        denormalized = torch.atanh(y_clamped) * self.std + self.mean
        return denormalized
    
    def get_stats(self):
        """Get the fitted mean and std statistics."""
        return {"mean": self.mean, "std": self.std}

class IonoDataset(Dataset): # type: ignore
    def __init__(
        self,
        resolution=(24, 360),
        path=None,
        train_val_test=[0.8, 0.10, 0.10],
        split="train",
        transforms=True,
        normalization_type="absolute_max",  # "absolute_max" or "mean_sigma_tanh"
        seed=42
    ):

        # path to dataset
        base_path = path
        self.base_path = Path(base_path)

        # Transformations
        self.are_transform = transforms
        self.normalization_type = normalization_type
        
        # Initialize normalizer based on type
        if self.normalization_type == "mean_sigma_tanh":
            self.normalizer = MeanSigmaTanhNormalizer()
            # We'll fit the normalizer after loading all data
            self._fit_normalizer()
        else:
            self.normalizer = None

        # Splitting the dataset

        self.train_perc = train_val_test[0]
        self.valid_perc = train_val_test[1]
        self.test_perc = train_val_test[2]
        self.seed = seed
        self.split = split
        self.files_paths = self.get_filespaths()
        

    def get_filespaths(self):
        # Get all files in the base path
        files = list(self.base_path.glob("*.npy"))
        # Sort the files
        files.sort()
        
        # Shuffle deterministically
        rng = random.Random(self.seed)
        rng.shuffle(files)

        # Compute sizes
        total_size = len(files)
        train_size = int(self.train_perc * total_size)
        val_size = int(self.valid_perc * total_size)
        test_size = total_size - train_size - val_size  # remainder

        # Split the list
        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        test_files = files[train_size + val_size:]

        file_paths = {
            'train': train_files,
            'valid': val_files,
            'test': test_files,
        }[self.split]

        return file_paths

    def _fit_normalizer(self):
        """Fit the normalizer by computing statistics on a sample of the data."""
        if self.normalizer is None:
            return
            
        # Load a sample of data to compute statistics
        sample_data = []
        max_samples = min(100, len(self.files_paths))  # Use max 100 files for fitting
        
        print(f"Fitting normalizer on {max_samples} samples...")
        for i in range(0, len(self.files_paths), max(1, len(self.files_paths) // max_samples)):
            if len(sample_data) >= max_samples:
                break
            try:
                file_path = self.files_paths[i]
                data = np.load(file_path, allow_pickle=True)
                sample_data.append(data[0])
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        if sample_data:
            # Stack all sample data
            all_data = np.stack(sample_data, axis=0)
            
            # Fit the normalizer
            self.normalizer.fit(all_data)
            stats = self.normalizer.get_stats()
            print(f"Normalizer fitted: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        else:
            print("Warning: No data could be loaded for fitting normalizer!")

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        # Load the data
        file_path = self.files_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        
        data_tensor = torch.from_numpy(data[0]).float().unsqueeze(0)
        # Apply transformations if any
        if self.are_transform:
            if self.normalization_type == "mean_sigma_tanh" and self.normalizer is not None:
                # Use mean-sigma + tanh normalization
                data_tensor = self.normalizer.forward(data_tensor)
            else:
                # Use original absolute max normalization
                data_tensor = torch.clamp(data_tensor, -80000, 80000) #/ 55000 # maximum max value among the whole dataset can be changed
                data_tensor = 2* (data_tensor - 80000) / (80000 - (-80000)) -1  # normalize to [-1, 1]

        condition_tensor = torch.tensor([data[1], data[2], data[3], data[4]], dtype=torch.float32)

        return data_tensor, condition_tensor
    
    def reverse_normalization(self, normalized_tensor):
        """Reverse the normalization applied to the data."""
        if not self.are_transform:
            return normalized_tensor
            
        if self.normalization_type == "mean_sigma_tanh" and self.normalizer is not None:
            return self.normalizer.reverse(normalized_tensor)
        else:
            # Reverse absolute max normalization
            return 2 * ((normalized_tensor - 80000) / (80000 - 80000)) - 1 #normalized_tensor * 55000.0

class RandomSamplerSeed(Sampler[int]):
    """Overwrite the RandomSampler to allow for a seed for each epoch.
    Effectively going over the same data at same epochs."""

    def __init__(
        self,
        dataset: Sized, 
        num_samples: int | None = None,
        generator=None, 
        epoch: int = 0
    ):
        self.dataset = dataset
        self._num_samples = num_samples
        self.generator = generator
        self.epoch = epoch

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.dataset)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if self.generator is None:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            seed = int(torch.empty((), dtype=torch.int64).random_(generator=g).item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=generator).tolist()
        yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return len(self.dataset)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    
def get_data_objects(
    batch_size,
    distributed,
    num_data_workers,
    train_val_test=[0.8, 0.10, 0.10],
    rank=None,
    world_size=None,
    split="train",
    path=None,
    transforms=True,
    normalization_type="absolute_max",  # "absolute_max" or "mean_sigma_tanh"
    seed=42,
):

    dataset = IonoDataset(
        path=path,
        transforms=transforms,
        split=split,
        seed=seed,
        train_val_test=train_val_test,
        normalization_type=normalization_type
    )
    
    # sampler
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, seed=0
        )
    else:
        sampler = RandomSamplerSeed(dataset)
    
    # dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        shuffle=False,  # shuffle determined by the sampler
        sampler=sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

    return dataset, sampler, dataloader

def extract_timestamp(filename):
            base = Path(filename).stem
            parts = base.split('_')
            try:
                dt = datetime(
                    int(parts[-6]), int(parts[-5]), int(parts[-4]),
                    int(parts[-3]), int(parts[-2]), int(parts[-1])
                )
            except Exception:
                dt = None
            return dt


def latlon_to_cartesian_grid(data, output_size=224):
    """
    Convert (24, 360) lat/lon ionosphere data to (output_size, output_size) Cartesian grid.
    Creates a circular disk representation.

    Args:
        data: (24, 360) numpy array
        output_size: output grid size (default 224)

    Returns:
        grid: (output_size, output_size) array with circular data
    """
    # Squeeze if needed
    if data.ndim > 2:
        data = data.squeeze()

    # Original lat/lon coordinates
    mag_lat = np.linspace(-90, -66, data.shape[0])  # 24 latitude points
    mag_lon = np.linspace(0, 360, data.shape[1], endpoint=False)  # 360 longitude points
    lon_grid, lat_grid = np.meshgrid(mag_lon, mag_lat)

    # Convert to polar (r, theta)
    r = 90 - np.abs(lat_grid.flatten())  # radial distance
    theta = np.deg2rad(lon_grid.flatten())  # angle in radians

    # Convert polar to Cartesian (x, y)
    x_src = r * np.cos(theta)
    y_src = r * np.sin(theta)

    # Create target Cartesian grid (output_size x output_size)
    max_r = r.max()
    x_target = np.linspace(-max_r, max_r, output_size)
    y_target = np.linspace(-max_r, max_r, output_size)
    x_grid, y_grid = np.meshgrid(x_target, y_target)

    # Interpolate data onto Cartesian grid
    points = np.column_stack((x_src, y_src))
    grid_values = griddata(points, data.flatten(), (x_grid, y_grid), method='linear', fill_value=0)

    # Mask values outside the circle
    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    grid_values[r_grid > max_r] = 0  # Zero outside the circle

    return grid_values
class IonoSequenceDataset(Dataset):
    def __init__(
        self,
        csv_path,
        transform_cond_csv=None,
        resolution=(24, 360),
        train_val_test=[0.8, 0.10, 0.10],
        split="train",
        sequence_length=5,
        transforms=True,
        normalization_type="absolute_max",  # "absolute_max", "mean_sigma_tanh", "per_file_log", or "ionosphere_preprocess"
        use_l1_conditions=False,  # New parameter to switch between modes
        min_center_distance=15,  # Minimum distance between sequence centers (frames)
        cartesian_transform=False,  # Convert to Cartesian circular grid
        output_size=224,  # Output grid size for Cartesian transform
        per_file_stats_path=None,  # Path to per-file normalization stats JSON
        only_complete_sequences=False,  # Only use sequences with no missing frames
        preprocess_config=None,  # Preprocessing config for ionosphere_preprocess normalization
        seed=42
    ):
        self.csv_path = csv_path
        self.transform_cond_csv = transform_cond_csv
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.normalization_type = normalization_type
        self.use_l1_conditions = use_l1_conditions
        self.min_center_distance = min_center_distance
        self.cartesian_transform = cartesian_transform
        self.output_size = output_size
        self.only_complete_sequences = only_complete_sequences
        self.preprocess_config = preprocess_config  # Store preprocessing config
        self.seed = seed
        self.split = split
        self.train_perc = train_val_test[0]
        self.valid_perc = train_val_test[1]
        self.test_perc = train_val_test[2]

        # Load global stats if provided (for global log normalization)
        self.global_mean = None
        self.global_std = None
        if per_file_stats_path is not None:
            import json
            with open(per_file_stats_path, 'r') as f:
                stats = json.load(f)
                # Check if it's global stats (dict with 'mean' and 'std')
                if isinstance(stats, dict) and 'mean' in stats and 'std' in stats:
                    # Global stats
                    self.global_mean = stats['mean']
                    self.global_std = stats['std']
                    print(f"Loaded global stats: mean={self.global_mean:.2f}, std={self.global_std:.2f}")
                else:
                    raise ValueError("Expected global stats JSON with 'mean' and 'std' keys")

        # Initialize normalizer based on type
        if self.normalization_type == "mean_sigma_tanh":
            self.normalizer = MeanSigmaTanhNormalizer()
            # We'll fit the normalizer after loading sequence info
        else:
            self.normalizer = None

        # Read CSV and extract filenames and timestamps
        df = pd.read_csv(self.csv_path)

        if self.use_l1_conditions:
            # Using L1 conditions from merged CSV
            if 'proton_vx_gsm' not in df.columns:
                raise ValueError("CSV file must contain L1 columns when use_l1_conditions=True")
            
            # Store min and max for L1 columns for normalization (exclude missing values)
            l1_cols = ['bx_gsm', 'by_gsm', 'bz_gsm', 'proton_vx_gsm']
            df_clean = df[l1_cols].replace(-99999, np.nan)  # Replace missing values with NaN
            self.cond_min = df_clean.min().values.astype(np.float32)  # min ignores NaN
            self.cond_max = df_clean.max().values.astype(np.float32)  # max ignores NaN
        else:
            # Using original projected bow shock conditions from separate CSV
            if self.transform_cond_csv is None:
                raise ValueError("transform_cond_csv must be provided when use_l1_conditions=False")
            
            df_cond = pd.read_csv(self.transform_cond_csv)
            df_cond["float4"] = df_cond["float4"] * -1 # do this becauuse it is a velocity and it was stored in the opposite
            # Store min and max for each float column for normalization (exclude missing values)
            df_cond_clean = df_cond[["float1", "float2", "float3", "float4"]].replace(-99999, np.nan)
            self.cond_min = df_cond_clean.min().values.astype(np.float32)  # min ignores NaN
            self.cond_max = df_cond_clean.max().values.astype(np.float32)  # max ignores NaN
        df['timestamp'] = df['filename'].apply(extract_timestamp)
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        self.all_files = df['filename'].tolist()
        self.all_timestamps = df['timestamp'].tolist()

        # Build overlapping sequences with minimum center distance
        # Fill missing frames with zeros instead of rejecting sequences
        self.sequences = []  # Store center indices
        n = len(self.all_files)
        i = 0
        while i + self.sequence_length <= n:
            center_idx = i + self.sequence_length // 2

            # Check if we have minimum distance from last sequence's center
            if not self.sequences or (center_idx - self.sequences[-1] >= self.min_center_distance):
                self.sequences.append(center_idx)
                i += self.min_center_distance  # Move by minimum distance
            else:
                i += 1  # Try next window

        # Split sequences by month (based on center timestamp)
        train_seqs, val_seqs, test_seqs = [], [], []
        for center_idx in self.sequences:
            month = self.all_timestamps[center_idx].month
            if 1 <= month <= 8:
                train_seqs.append(center_idx)
            elif 9 <= month <= 10:
                val_seqs.append(center_idx)
            elif 11 <= month <= 12:
                test_seqs.append(center_idx)

        split_seqs = {
            'train': train_seqs,
            'valid': val_seqs,
            'test': test_seqs,
        }[self.split]
        self.sequences = split_seqs

        # Filter out sequences with missing frames if requested
        if self.only_complete_sequences:
            original_count = len(self.sequences)
            self.sequences = self._filter_complete_sequences()
            filtered_count = len(self.sequences)
            print(f"Filtered sequences for {self.split}: {original_count} -> {filtered_count} "
                  f"({filtered_count/original_count*100:.1f}% complete sequences)")

        # Fit the normalizer if using mean_sigma_tanh
        if self.normalization_type == "mean_sigma_tanh" and self.normalizer is not None:
            self._fit_normalizer_sequence()

    def _fit_normalizer_sequence(self):
        """Fit the normalizer by computing statistics on a sample of the sequence data."""
        if self.normalizer is None:
            return

        # Load a sample of data to compute statistics
        sample_data = []
        max_samples = min(10, len(self.sequences))  # Use max 10 sequences for fitting

        print(f"Fitting sequence normalizer on {max_samples} sequences...")
        for seq_idx in range(0, len(self.sequences), max(1, len(self.sequences) // max_samples)):
            if len(sample_data) >= max_samples * 5:  # 5 files per sequence max
                break
            try:
                center_idx = self.sequences[seq_idx]
                start_idx = center_idx - self.sequence_length // 2

                # Sample a few files from each sequence
                for i in range(min(5, self.sequence_length)):  # Take first 5 files
                    file_idx = start_idx + i
                    if 0 <= file_idx < len(self.all_files):
                        file_path = self.all_files[file_idx]
                        data = np.load('/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/pickled_maps/' + file_path, allow_pickle=True)
                        sample_data.append(data[0])
            except Exception as e:
                print(f"Warning: Could not load sequence {seq_idx}: {e}")
                continue

        if sample_data:
            # Stack all sample data
            all_data = np.stack(sample_data, axis=0)

            # Fit the normalizer
            self.normalizer.fit(all_data)
            stats = self.normalizer.get_stats()
            print(f"Sequence normalizer fitted: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        else:
            print("Warning: No sequence data could be loaded for fitting normalizer!")

    def _filter_complete_sequences(self):
        """
        Filter out sequences that have missing frames.
        Returns a list of center indices for complete sequences only.
        """
        complete_sequences = []

        for center_idx in self.sequences:
            start_idx = center_idx - self.sequence_length // 2
            center_time = self.all_timestamps[center_idx]
            expected_start_time = center_time - timedelta(minutes=2 * (self.sequence_length // 2))

            # Check if all frames in this sequence exist
            is_complete = True
            for frame_offset in range(self.sequence_length):
                file_idx = start_idx + frame_offset
                expected_time = expected_start_time + timedelta(minutes=2 * frame_offset)

                # Check if this frame exists and matches the expected timestamp
                frame_exists = False
                if 0 <= file_idx < len(self.all_files):
                    actual_time = self.all_timestamps[file_idx]
                    # Allow small tolerance (1 second) for timestamp matching
                    if abs((actual_time - expected_time).total_seconds()) <= 1:
                        frame_exists = True

                if not frame_exists:
                    is_complete = False
                    break

            if is_complete:
                complete_sequences.append(center_idx)

        return complete_sequences

    def __len__(self):
        return len(self.sequences)
    
    def revert_condition_normalization(self, cond_normalized):
        """
        Revert normalized conditions from [-1, 1] back to original scale
        
        Args:
            cond_normalized: Tensor or array of shape (..., 4) with values in [-1, 1]
            
        Returns:
            Tensor or array of original condition values
            
        Note:
            - Each column was normalized independently using per-column min/max
            - Column 3 (index 3) represents -float4 (velocity was negated)
            - Formula: (norm + 1) / 2 * (max - min) + min
        """
        # Convert tensor to numpy if needed
        is_tensor = torch.is_tensor(cond_normalized)
        if is_tensor:
            cond_norm = cond_normalized.numpy()
        else:
            cond_norm = cond_normalized
            
        # Revert normalization: (norm + 1) / 2 * (max - min) + min
        cond_original = (cond_norm + 1) / 2 * (self.cond_max - self.cond_min) + self.cond_min
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            cond_original = torch.from_numpy(cond_original).float()
            
        return cond_original

    def __getitem__(self, idx):
        """
        Get a sequence by building it from the center index.
        Missing frames (due to temporal gaps) are filled with zeros.
        """
        center_idx = self.sequences[idx]
        start_idx = center_idx - self.sequence_length // 2
        end_idx = start_idx + self.sequence_length

        data_tensors = []
        cond_tensors = []

        # Get the CSV dataframe for condition lookup if using L1 conditions
        if self.use_l1_conditions:
            df = pd.read_csv(self.csv_path)

        # Calculate expected timestamps for the sequence (2-min cadence)
        center_time = self.all_timestamps[center_idx]
        expected_start_time = center_time - timedelta(minutes=2 * (self.sequence_length // 2))

        for frame_offset in range(self.sequence_length):
            file_idx = start_idx + frame_offset
            expected_time = expected_start_time + timedelta(minutes=2 * frame_offset)

            # Check if this frame exists and matches the expected timestamp
            frame_exists = False
            if 0 <= file_idx < len(self.all_files):
                actual_time = self.all_timestamps[file_idx]
                # Allow small tolerance (1 second) for timestamp matching
                if abs((actual_time - expected_time).total_seconds()) <= 1:
                    frame_exists = True

            if frame_exists:
                # Load the actual frame
                file_path = self.all_files[file_idx]
                data = np.load('/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/pickled_maps/' + file_path, allow_pickle=True)

                # Apply Cartesian transformation if enabled
                data_map = data[0].astype(np.float32)
                if self.cartesian_transform:
                    data_map = latlon_to_cartesian_grid(data_map, output_size=self.output_size)

                data_tensor = torch.from_numpy(data_map).float().unsqueeze(0)

                if self.transforms:
                    if self.normalization_type == "mean_sigma_tanh" and self.normalizer is not None:
                        data_tensor = self.normalizer.forward(data_tensor)
                    elif self.normalization_type == "per_file_log" and self.global_mean is not None and self.global_std is not None:
                        # Use global symmetric log normalization
                        # Center the data
                        centered = data_tensor - self.global_mean

                        # Symmetric log: sign(x) * log(1 + |x| / scale)
                        sign = torch.sign(centered)
                        abs_centered = torch.abs(centered)
                        log_transformed = sign * torch.log1p(abs_centered / self.global_std)

                        # Scale to approximately [-1, 1] based on expected range
                        # For ~3 std, log(1 + 3) ≈ 1.39
                        max_expected = np.log1p(3)  # ≈ 1.39
                        data_tensor = log_transformed / max_expected
                    elif self.normalization_type == "ionosphere_preprocess":
                        # Use new SDO-style preprocessing with configurable scaling
                        data_tensor = get_ionosphere_transform(data_tensor, config=self.preprocess_config)
                    else:
                        # Default absolute_max normalization
                        # data_tensor = torch.clamp(data_tensor, -55000, 55000) / 55000.0

                        data_tensor = torch.clamp(data_tensor, -80000, 80000) #/ 55000 # maximum max value among the whole dataset can be changed
                        data_tensor = 2 * (data_tensor - (-80000)) / (80000 - (-80000)) - 1 # normalize to [-1, 1]

                if self.use_l1_conditions:
                    filename = os.path.basename(file_path)
                    file_row = df[df['filename'] == filename]

                    if len(file_row) == 0:
                        raise ValueError(f"Filename {filename} not found in CSV")

                    cond_raw = np.array([
                        file_row['bx_gsm'].iloc[0],
                        file_row['by_gsm'].iloc[0],
                        file_row['bz_gsm'].iloc[0],
                        file_row['proton_vx_gsm'].iloc[0],
                    ], dtype=np.float32)
                else:
                    cond_raw = np.array([data[1], data[2], data[3], data[4]], dtype=np.float32)
                
                # Negate velocity
                # cond_raw[3] = -cond_raw[3]

                # Normalize conditions
                cond_norm = 2 * (cond_raw - self.cond_min) / (self.cond_max - self.cond_min) - 1

                cond_tensor = torch.tensor(cond_norm, dtype=torch.float32)
            else:
                # Frame is missing - fill data with zeros, conditions with 2.0 (outside normalized range)
                if self.cartesian_transform:
                    data_tensor = torch.zeros(1, self.output_size, self.output_size, dtype=torch.float32)
                else:
                    data_tensor = torch.zeros(1, 24, 360, dtype=torch.float32)
                cond_tensor = torch.full((4,), 2.0, dtype=torch.float32)

            data_tensors.append(data_tensor)
            cond_tensors.append(cond_tensor)

        data_seq = torch.stack(data_tensors, dim=0)
        cond_seq = torch.stack(cond_tensors, dim=0)
        return data_seq, cond_seq

    def reverse_data_normalization(self, normalized_tensor):
        """
        Reverse the normalization applied to the data.

        Args:
            normalized_tensor: Normalized tensor to denormalize

        Returns:
            Denormalized tensor
        """
        if not self.transforms:
            return normalized_tensor

        if self.normalization_type == "mean_sigma_tanh" and self.normalizer is not None:
            return self.normalizer.reverse(normalized_tensor)
        elif self.normalization_type == "per_file_log":
            if self.global_mean is None or self.global_std is None:
                raise ValueError("Global stats not loaded for per_file_log denormalization")

            # Reverse the scaling
            max_expected = np.log1p(11)  # ≈ 1.39
            log_transformed = normalized_tensor * max_expected

            # Reverse symmetric log: sign(y) * std * (exp(|y|) - 1)
            sign = torch.sign(log_transformed)
            abs_log = torch.abs(log_transformed)
            centered = sign * self.global_std * (torch.exp(abs_log) - 1)

            # Add back the mean
            denormalized = centered + self.global_mean
            return denormalized
        elif self.normalization_type == "ionosphere_preprocess":
            # Use new SDO-style reverse preprocessing
            return reverse_ionosphere_transform(normalized_tensor, config=self.preprocess_config)
        else:
            # Reverse absolute max normalization
            # Original: data_tensor = 2 * (data_tensor - (-80000)) / (80000 - (-80000)) - 1
            # Reverse: x = ((norm + 1) / 2) * (80000 - (-80000)) + (-80000)
            return ((normalized_tensor + 1) / 2) * (80000 - (-80000)) + (-80000)


def get_sequence_data_objects(
    batch_size,
    distributed,
    num_data_workers,
    sequence_length,
    train_val_test=[0.8, 0.10, 0.10],
    rank=None,
    world_size=None,
    split="train",
    csv_path=None,
    transform_cond_csv=None,
    transforms=True,
    normalization_type="absolute_max",  # "absolute_max", "mean_sigma_tanh", "per_file_log", or "ionosphere_preprocess"
    use_l1_conditions=False,  # New parameter
    min_center_distance=15,  # Minimum distance between sequence centers
    cartesian_transform=False,  # Convert to Cartesian circular grid
    output_size=224,  # Output grid size for Cartesian transform
    per_file_stats_path=None,  # Path to per-file normalization stats JSON
    only_complete_sequences=False,  # Only use sequences with no missing frames
    preprocess_config=None,  # Preprocessing config for ionosphere_preprocess normalization
    seed=42,
):
    dataset = IonoSequenceDataset(
        csv_path=csv_path,
        transform_cond_csv=transform_cond_csv,
        transforms=transforms,
        split=split,
        seed=seed,
        train_val_test=train_val_test,
        sequence_length=sequence_length,
        normalization_type=normalization_type,
        use_l1_conditions=use_l1_conditions,
        min_center_distance=min_center_distance,
        cartesian_transform=cartesian_transform,
        output_size=output_size,
        per_file_stats_path=per_file_stats_path,
        only_complete_sequences=only_complete_sequences,
        preprocess_config=preprocess_config
    )
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, seed=0
        )
    else:
        sampler = RandomSamplerSeed(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )
    return dataset, sampler, dataloader


class IonoRAEDataset(Dataset):
    """
    Dataset for RAE Stage 1 training on ionospheric data.
    Returns individual 2D frames transformed to Cartesian grid.

    Args:
        data_path: Path to directory containing .npy files
        csv_path: Path to CSV file with metadata
        split: 'train', 'valid', or 'test'
        train_val_test: Split ratios [train, val, test]
        normalization: Normalization type - 'minmax' (maps to [-1,1]) or 'none'
        triplicate_channels: Triplicate to (3, H, W) for pretrained encoders
        cartesian_transform: If True, convert (24, 360) to (output_size, output_size) circular grid
        output_size: Size of output grid if cartesian_transform=True
        seed: Random seed for splitting
    """

    def __init__(
        self,
        data_path,
        csv_path=None,
        split='train',
        train_val_test=[0.8, 0.10, 0.10],
        normalization='minmax',
        triplicate_channels=True,  # Triplicate to (3, H, W) for pretrained encoders
        cartesian_transform=True,  # Convert to Cartesian circular grid
        output_size=224,  # Output grid size for Cartesian transform
        per_file_stats_path=None,  # Path to per-file normalization stats JSON
        seed=42
    ):
        self.data_path = Path(data_path)
        self.csv_path = csv_path
        self.split = split
        self.normalization = normalization
        self.triplicate_channels = triplicate_channels
        self.cartesian_transform = cartesian_transform
        self.output_size = output_size
        self.seed = seed

        # Split ratios
        self.train_perc = train_val_test[0]
        self.valid_perc = train_val_test[1]
        self.test_perc = train_val_test[2]

        # Load per-file stats if provided (for per-file mean_sigma_tanh normalization)
        self.stats_dict = None
        if per_file_stats_path is not None:
            import json
            with open(per_file_stats_path, 'r') as f:
                stats_list = json.load(f)
                # Convert to dict for fast lookup: filename -> {mean, std}
                self.stats_dict = {s['file']: s for s in stats_list}
            print(f"Loaded per-file stats for {len(self.stats_dict)} files")

        # Load file paths and split
        self.file_paths = self._get_file_paths()

        print(f"IonoRAEDataset [{split}]: {len(self.file_paths)} samples")
        print(f"  Normalization: {normalization}")
        print(f"  Cartesian transform: {cartesian_transform}")
        if cartesian_transform:
            input_channels = 3 if triplicate_channels else 1
            print(f"  Input shape: ({input_channels}, {output_size}, {output_size})")
            print(f"  Target shape: (1, {output_size}, {output_size})")
        else:
            input_shape = "(3, 24, 360)" if triplicate_channels else "(1, 24, 360)"
            print(f"  Input shape: {input_shape}, Target shape: (1, 24, 360)")

    def _get_file_paths(self):
        """Get file paths and split by month (same as sequence dataset)."""
        # If CSV is provided, use it for splitting by month
        if self.csv_path is not None:
            df = pd.read_csv(self.csv_path)

            df['timestamp'] = df['filename'].apply(extract_timestamp)
            df = df.dropna(subset=['timestamp'])

            # Split by month
            train_files = df[df['timestamp'].dt.month.between(1, 8)]['filename'].tolist()
            val_files = df[df['timestamp'].dt.month.between(9, 10)]['filename'].tolist()
            test_files = df[df['timestamp'].dt.month.between(11, 12)]['filename'].tolist()

            split_files = {
                'train': train_files,
                'valid': val_files,
                'test': test_files,
            }[self.split]

            # Return full paths
            return [self.data_path / f for f in split_files]

        else:
            # Fall back to random splitting
            files = sorted(list(self.data_path.glob("*.npy")))

            # Shuffle deterministically
            rng = random.Random(self.seed)
            rng.shuffle(files)

            # Compute sizes
            total_size = len(files)
            train_size = int(self.train_perc * total_size)
            val_size = int(self.valid_perc * total_size)

            # Split the list
            train_files = files[:train_size]
            val_files = files[train_size:train_size + val_size]
            test_files = files[train_size + val_size:]

            split_files = {
                'train': train_files,
                'valid': val_files,
                'test': test_files,
            }[self.split]

            return split_files

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Returns:
            input_tensor: (C, H, W) where C=3 if triplicate_channels else 1
            target: (1, H, W)
            Both normalized to [-1, 1] via min-max scaling
        """
        file_path = self.file_paths[idx]

        # Load data (stored as pickled array with [data, cond1, cond2, cond3, cond4])
        data = np.load(file_path, allow_pickle=True)

        # Extract 2D map (24, 360)
        data_map = data[0].astype(np.float32)

        # Apply Cartesian transformation if enabled
        if self.cartesian_transform:
            data_map = latlon_to_cartesian_grid(data_map, output_size=self.output_size)

        # Convert to tensor and add channel dimension: (1, H, W)
        data_tensor = torch.from_numpy(data_map).unsqueeze(0)

        # Apply normalization
        if self.normalization == 'minmax':
            # Clamp to [-80000, 80000] and normalize to [-1, 1]
            data_tensor = torch.clamp(data_tensor, -80000, 80000)
            data_tensor = 2 * (data_tensor - (-80000)) / (80000 - (-80000)) - 1
        elif self.normalization == 'per_file_tanh' and self.stats_dict is not None:
            # Use per-file mean/std normalization with tanh
            filename = file_path.name
            if filename in self.stats_dict:
                file_stats = self.stats_dict[filename]
                mean = file_stats['mean']
                std = file_stats['std']
                # Standardize and apply tanh
                normalized = (data_tensor - mean) / std
                data_tensor = torch.tanh(normalized)

        # Target is always single channel (1, H, W)
        target = data_tensor

        # Input: triplicate if needed for pretrained RGB encoders
        if self.triplicate_channels:
            input_tensor = data_tensor.repeat(3, 1, 1)  # (3, H, W)
        else:
            input_tensor = data_tensor  # (1, H, W)

        return input_tensor, target

    def denormalize(self, normalized_tensor, filename=None):
        """
        Reverse normalization to get back original scale.

        Args:
            normalized_tensor: Normalized tensor to denormalize
            filename: Required for 'per_file_tanh' normalization - the filename to look up stats

        Returns:
            Denormalized tensor
        """
        if self.normalization == 'minmax':
            # Reverse: x = ((norm + 1) / 2) * (max - min) + min
            return ((normalized_tensor + 1) / 2) * (80000 - (-80000)) + (-80000)
        elif self.normalization == 'per_file_tanh':
            if filename is None:
                raise ValueError("filename must be provided for per_file_tanh denormalization")
            if self.stats_dict is None:
                raise ValueError("stats_dict not loaded for per_file_tanh denormalization")

            if filename not in self.stats_dict:
                print(f"Warning: No stats found for {filename}, cannot denormalize accurately")
                return normalized_tensor

            file_stats = self.stats_dict[filename]
            mean = file_stats['mean']
            std = file_stats['std']

            # Reverse tanh: atanh(y)
            # Clamp to valid range for atanh
            y_clamped = torch.clamp(normalized_tensor, -0.999999, 0.999999)
            denormalized = torch.atanh(y_clamped) * std + mean
            return denormalized
        else:
            return normalized_tensor


def get_ionosphere_rae_dataloaders(
    data_path,
    csv_path=None,
    batch_size=32,
    num_workers=4,
    normalization='minmax',
    triplicate_channels=True,
    cartesian_transform=True,
    output_size=224,
    per_file_stats_path=None,  # Path to per-file normalization stats JSON
    seed=42,
    train_val_test=[0.8, 0.10, 0.10],
):
    """
    Convenience function to create train/val dataloaders for RAE Stage 1.

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    train_dataset = IonoRAEDataset(
        data_path=data_path,
        csv_path=csv_path,
        split='train',
        train_val_test=train_val_test,
        normalization=normalization,
        triplicate_channels=triplicate_channels,
        cartesian_transform=cartesian_transform,
        output_size=output_size,
        per_file_stats_path=per_file_stats_path,
        seed=seed
    )

    val_dataset = IonoRAEDataset(
        data_path=data_path,
        csv_path=csv_path,
        split='valid',
        train_val_test=train_val_test,
        normalization=normalization,
        triplicate_channels=triplicate_channels,
        cartesian_transform=cartesian_transform,
        output_size=output_size,
        per_file_stats_path=per_file_stats_path,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, train_dataset, val_dataset
