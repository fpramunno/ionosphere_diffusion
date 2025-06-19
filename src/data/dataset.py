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

class IonoDataset(Dataset): # type: ignore
    def __init__(
        self,
        resolution=(24, 360),
        path=None,
        train_val_test=[0.8, 0.10, 0.10],
        split="train",
        transforms=True,
        seed=42
    ):

        # path to dataset
        base_path = path
        self.base_path = Path(base_path)

        # Transformations
        self.are_transform = transforms
        # if self.are_transform:
        #     self.transforms = Compose([
        #                             Normalize(mean=[-1385.47], std=[7235.46]),  # z-score normalization
        #                             # Normalize(mean=[0.5], std=[0.5])            # maps to [-1, 1]
        #                         ])

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

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        # Load the data
        file_path = self.files_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        
        data_tensor = torch.from_numpy(data[0]).float().unsqueeze(0)
        # Apply transformations if any
        if self.are_transform:
            data_tensor = data_tensor / 108154.0 # maximum max value among the whole dataset can be changed
            # data_tensor = self.transforms(data_tensor) 

        condition_tensor = torch.tensor([data[1], data[2], data[3], data[4]], dtype=torch.float32)


        return data_tensor, condition_tensor

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
    seed=42,
):

    dataset = IonoDataset(
        path=path,
        transforms=transforms,
        split=split,
        seed=seed,
        train_val_test=train_val_test
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
        resolution=(24, 360),
        train_val_test=[0.8, 0.10, 0.10],
        split="train",
        sequence_length=5,
        transforms=True,
        seed=42
    ):
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.transforms = transforms
        self.seed = seed
        self.split = split
        self.train_perc = train_val_test[0]
        self.valid_perc = train_val_test[1]
        self.test_perc = train_val_test[2]

        # Read CSV and extract filenames and timestamps
        df = pd.read_csv(self.csv_path)
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

        # Also build sequences by simply grouping every self.sequence_length images, regardless of temporal consistency
        # self.grouped_sequences = []
        # for i in range(0, n - self.sequence_length + 1, self.sequence_length):
        #     seq_files = self.all_files[i:i+self.sequence_length]
        #     self.grouped_sequences.append(seq_files)

        # Split sequences by timestamp: first 8 months train, next 2 val, last 2 test
        # Assumes timestamps are sorted and cover a single year

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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_files = self.sequences[idx]
        data_tensors = []
        cond_tensors = []
        time = []
        for file_path in seq_files:
            data = np.load(file_path, allow_pickle=True)
            data_tensor = torch.from_numpy(data[0]).float().unsqueeze(0)
            if self.transforms:
                data_tensor = data_tensor / 108154.0
            cond_tensor = torch.tensor([data[1], data[2], data[3], data[4]], dtype=torch.float32)
            data_tensors.append(data_tensor)
            cond_tensors.append(cond_tensor)
            time.append(extract_timestamp(file_path))
        data_seq = torch.stack(data_tensors, dim=0)
        cond_seq = torch.stack(cond_tensors, dim=0)
        return data_seq, cond_seq #, time


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
    transforms=True,
    seed=42,
):
    dataset = IonoSequenceDataset(
        csv_path=csv_path,
        transforms=transforms,
        split=split,
        seed=seed,
        train_val_test=train_val_test,
        sequence_length=sequence_length
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

