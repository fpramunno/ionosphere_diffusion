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
from datetime import datetime
from collections.abc import Sized

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
                data_tensor = torch.clamp(data_tensor, -55000, 55000) / 55000 # maximum max value among the whole dataset can be changed

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
            return normalized_tensor * 55000.0

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
        normalization_type="absolute_max",  # "absolute_max" or "mean_sigma_tanh"
        use_l1_conditions=False,  # New parameter to switch between modes
        seed=42
    ):
        self.csv_path = csv_path
        self.transform_cond_csv = transform_cond_csv
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.normalization_type = normalization_type
        self.use_l1_conditions = use_l1_conditions
        self.seed = seed
        self.split = split
        self.train_perc = train_val_test[0]
        self.valid_perc = train_val_test[1]
        self.test_perc = train_val_test[2]
        
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
            l1_cols = ['proton_vx_gsm', 'bx_gsm', 'by_gsm', 'bz_gsm']
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

        # Build only temporally consistent, non-overlapping sequences (2 min apart)
        self.sequences = []
        n = len(self.all_files)
        i = 0
        while i + self.sequence_length <= n:
            seq_files = self.all_files[i:i+self.sequence_length]
            seq_times = self.all_timestamps[i:i+self.sequence_length]
            is_consistent = True
            for j in range(1, self.sequence_length):
                delta = (seq_times[j] - seq_times[j-1]).total_seconds()
                if delta != 120:  # 2 minutes = 120 seconds
                    is_consistent = False
                    break
            if is_consistent:
                self.sequences.append(seq_files)
                i += self.sequence_length  # non-overlapping
            else:
                i += 1  # try next window

        # Helper to get month from first file in each sequence
        def get_month(seq):
            idx = self.all_files.index(seq[0])
            return self.all_timestamps[idx].month

        train_seqs, val_seqs, test_seqs = [], [], []
        for seq in self.sequences:
            month = get_month(seq)
            if 1 <= month <= 8:
                train_seqs.append(seq)
            elif 9 <= month <= 10:
                val_seqs.append(seq)
            elif 11 <= month <= 12:
                test_seqs.append(seq)
        split_seqs = {
            'train': train_seqs,
            'valid': val_seqs,
            'test': test_seqs,
        }[self.split]
        self.sequences = split_seqs
        
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
                seq_files = self.sequences[seq_idx]
                # Sample a few files from each sequence
                for i, file_path in enumerate(seq_files[:5]):  # Take first 5 files
                    data = np.load(file_path, allow_pickle=True)
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
        seq_files = self.sequences[idx]
        data_tensors = []
        cond_tensors = []
        time = []
        
        # Get the CSV dataframe for condition lookup if using L1 conditions
        if self.use_l1_conditions:
            df = pd.read_csv(self.csv_path)
        
        for file_path in seq_files:
            # Load only the map data (first element) from npy file
            data = np.load('/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/pickled_maps/'+file_path, allow_pickle=True)
            data_tensor = torch.from_numpy(data[0]).float().unsqueeze(0)
            
            if self.transforms:
                if self.normalization_type == "mean_sigma_tanh" and self.normalizer is not None:
                    # Use mean-sigma + tanh normalization
                    data_tensor = self.normalizer.forward(data_tensor)
                else:
                    # Use original absolute max normalization
                    data_tensor = torch.clamp(data_tensor, -55000, 55000) / 55000.0
            
            if self.use_l1_conditions:
                # Get L1 conditions from CSV based on filename
                filename = os.path.basename(file_path)
                file_row = df[df['filename'] == filename]
                
                if len(file_row) == 0:
                    raise ValueError(f"Filename {filename} not found in CSV")
                
                # Use L1 conditions from merged CSV
                cond_raw = np.array([
                    file_row['bx_gsm'].iloc[0], 
                    file_row['by_gsm'].iloc[0],
                    file_row['bz_gsm'].iloc[0],
                    file_row['proton_vx_gsm'].iloc[0],
                ], dtype=np.float32)
                
            else:
                # Use original projected bow shock conditions from npy file
                cond_raw = np.array([data[1], data[2], data[3], data[4]], dtype=np.float32)
            
            # Negate velocity (proton_vx_gsm is at index 3 for L1, index 3 for original)
            if self.use_l1_conditions:
                cond_raw[3] = -cond_raw[3]  # proton_vx_gsm at index 3
            else:
                cond_raw[3] = -cond_raw[3]  # float4 at index 3
            
            # Normalize first, then handle missing values
            cond_norm = 2 * (cond_raw - self.cond_min) / (self.cond_max - self.cond_min) - 1
            
            # Replace missing values AFTER normalization with a special "no info" value
            missing_mask = (cond_raw == -99999)
            cond_norm[missing_mask] = 0.0  # 0 in normalized space means "neutral/no information"
            
            cond_tensor = torch.tensor(cond_norm, dtype=torch.float32)
            
            data_tensors.append(data_tensor)
            cond_tensors.append(cond_tensor)
            time.append(extract_timestamp(file_path))
            
        data_seq = torch.stack(data_tensors, dim=0)
        cond_seq = torch.stack(cond_tensors, dim=0)
        return data_seq, cond_seq #, time

    def reverse_data_normalization(self, normalized_tensor):
        """Reverse the normalization applied to the data."""
        if not self.transforms:
            return normalized_tensor
            
        if self.normalization_type == "mean_sigma_tanh" and self.normalizer is not None:
            return self.normalizer.reverse(normalized_tensor)
        else:
            # Reverse absolute max normalization
            return normalized_tensor * 55000.0


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
    normalization_type="absolute_max",  # "absolute_max" or "mean_sigma_tanh"
    use_l1_conditions=False,  # New parameter
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
        use_l1_conditions=use_l1_conditions
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

