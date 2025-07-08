import json
from datetime import datetime
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Iterator
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from astropy.io import fits
import math
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision.transforms import Normalize, Compose

from src.paths import HMI_DATA_PATH, AIA_DATA_PATH, SUN_REGION_DATA_PATH
from src.models.temporal_context import (
    UniformContext, MultiscaleSymmetricalContext, LongRangeFixedPastContext,
    Hierarchy2Context
)
from src.utils import subsample

CHANNEL_PREPROCESS = {
    "0094A": {"min": 0.1, "max": 800, "scaling": "log10"},
    "0131A": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "0171A": {"min": 5, "max": 3500, "scaling": "log10"},
    "0193A": {"min": 20, "max": 5500, "scaling": "log10"},
    "0211A": {"min": 7, "max": 3500, "scaling": "log10"},
    "0304A": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "0335A": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600A": {"min": 10, "max": 800, "scaling": "log10"},
    "1700A": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500A": {"min": 4000, "max": 20000, "scaling": "log10"},
}


def get_default_transforms(channel):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)

    Apply the normalization necessary for the SDO ML Dataset. Depending on the channel, it:
      - clips the "pixels" data in the predefined range (see above)
      - applies a log10() on the data
      - normalizes the data to the [0, 1] range
      - normalizes the data around 0 (standard scaling)
    also refer to
       - https://pytorch.org/vision/stable/transforms.html
       - https://github.com/i4Ds/SDOBenchmark/blob/master/dataset/data/load.py#L363
       - https://gitlab.com/jdonzallaz/solarnet-thesis/-/blob/master/solarnet/data/transforms.py

    Args:
        channel (str, optional): [The SDO channel]. Defaults to 171.
    Returns:
        [Transform]
    """
    transforms = []
    preprocess_config = CHANNEL_PREPROCESS[channel]

    if preprocess_config["scaling"] == "log10":
        def lambda_transform(x): return torch.log10(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - math.log10(preprocess_config["min"])
    elif preprocess_config["scaling"] == "sqrt":
        def lambda_transform(x): return torch.sqrt(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.sqrt(preprocess_config["min"])
        std = math.sqrt(preprocess_config["max"]) - math.sqrt(preprocess_config["min"])
    else:
        def lambda_transform(x): return torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        )
        mean = preprocess_config["min"]
        std = preprocess_config["max"] - preprocess_config["min"]

    # def test_lambda_func(x):
    #     return lambda_transform(x)
    transforms.append(lambda_transform)
    transforms.append(Normalize(mean=[mean], std=[std]))
    transforms.append(Normalize(mean=(0.5), std=(0.5)))

    return Compose(transforms)


class ArcsinhTransform:
    def __init__(self, factor=1.0):
        self.factor = factor

    def __call__(self, x):
        return torch.arcsinh(x / self.factor)

    def inverse(self, x):
        return torch.sinh(x) * self.factor
    

class ScaleTransform:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, x):
        return x / self.scale

    def inverse(self, x):
        return x * self.scale


class RandomSamplerSeed(Sampler[int]):
    """Overwrite the RandomSampler to allow for a seed for each epoch.
    Effectively going over the same data at same epochs."""

    def __init__(
        self,
        dataset: Dataset, 
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


def load_full_hmi(date_time=None, date=None, time=None, base_path_override=None, fields=None):
    """Load the full disk magnetic field components (Br, Bt, Bp) for a given date and time.
    
    Args:
        date_time (str, optional): Combined date and time string in format 'YYYYMMDDTHHMM'. 
        date (str, optional): Date string in format 'YYYYMMDD'.
        time (str, optional): Time string in format 'HHMM'.
            
    Returns:
        tuple: Returns (b_disk, fits_files) where:
            - b_disk is a numpy array containing concatenated Br, Bt, Bp components
            - fits_files is a tuple of the opened FITS files for each component
        or None if files not found
    """
    if date_time is None:
        date_time = f"{date}T{time}"
    else:
        date_time = date_time.strftime("%Y%m%dT%H%M")
    if base_path_override is None:
        base_path_override = HMI_DATA_PATH
    if fields is None:
        fields = ["Br", "Bp", "Bt"]
    # load full disk images
    hmi_paths = {field: base_path_override / date_time[:8] / f"sdo_hmi_h2_{date_time}00_{field}_v1.fits" for field in fields}
    hmi_data = {}
    for field in fields:
        if not hmi_paths[field].exists():
            print(f"file {hmi_paths[field]} does not exist")
            continue
        file = fits.open(hmi_paths[field])
        hmi_data[field] = file[1].data
        file.close()
    return hmi_data


def load_full_aia(date_time=None, date=None, time=None, base_path_override=None, wavelengths=None):
    """Load the full disk AIA data for a given date and time.
    
    Args:
        date_time (str, optional): Combined date and time string in format 'YYYYMMDDTHHMM'. 
        date (str, optional): Date string in format 'YYYYMMDD'.
        time (str, optional): Time string in format 'HHMM'.
        base_path_override (str, optional): Path to the base directory for AIA data.
        wavelengths (list, optional): List of wavelengths to load.
    """
    if date_time is None:
        date_time = f"{date}T{time}"
    else:
        date_time = date_time.strftime("%Y%m%dT%H%M")
    if base_path_override is None:
        base_path_override = AIA_DATA_PATH
    if wavelengths is None:
        wavelengths = [wv.value for wv in AIABand]
    # load full disk images
    aia_paths = {wv: base_path_override / date_time[:8] / f"sdo_aia_h2_{date_time}00_{wv:04d}_v1.fits" for wv in wavelengths}
    aia_data = {}
    for wl in wavelengths:
        if not aia_paths[wl].exists():
            print(f"file {aia_paths[wl]} does not exist")
            continue
        file = fits.open(aia_paths[wl])
        aia_data_wl = file[1].data
        aia_data[wl] = aia_data_wl
        file.close()
    return aia_data


class AIABand(Enum):
    """Enum representing the AIA bands."""
    AIA_94 = "0094A"    # EUV
    AIA_131 = "0131A"   # EUV
    AIA_171 = "0171A"   # EUV
    AIA_193 = "0193A"   # EUV
    AIA_211 = "0211A"   # EUV
    AIA_304 = "0304A"   # EUV
    AIA_335 = "0335A"   # EUV
    AIA_1600 = "1600A"  # UV
    AIA_1700 = "1700A"  # UV


class SunRegionDataset(Dataset):
    def __init__(
        self,
        n_past=None,
        n_future=None,
        temporal_context=None,
        resolution=(512,512),
        use_interpolate=True,
        base_path_override=None,
        AIA_channels=[],
        train_val_test=[0.7, 0.15, 0.15],  # this is hardcoded for now
        split="train",
        include_filters=None,
        exclude_filters=None,
        active_regions_only=False,
        transform_hmi=None,
        transform_aia=None
    ):
        # path to dataset
        base_path = base_path_override or SUN_REGION_DATA_PATH
        self.base_path = Path(base_path)
        self.include_filters = include_filters  # TODO: should cast to list
        self.exclude_filters = exclude_filters
        self.active_regions_only = active_regions_only
        self.AIA_channels = AIA_channels
        if self.base_path.name == "hmi_zoomed_freeze":  # TODO: ugly backward compatibility
            self.AIA_channels = []
        
        # temporal context: separation between past / future
        assert temporal_context is not None or (n_past is not None and n_future is not None), "Must provide either temporal_context or (n_past and n_future)"
        if temporal_context is None:
            self.temporal_context = UniformContext(past_frames=n_past, future_frames=n_future)
        else:
            self.temporal_context = temporal_context
        self.past_frames = self.temporal_context.past_frames
        self.past_horizon = self.temporal_context.past_horizon
        self.future_frames = self.temporal_context.future_frames
        self.future_horizon = self.temporal_context.future_horizon
        if isinstance(self.temporal_context, UniformContext):
            self.slicing_method = "direct"
        else:
            horizon = self.past_horizon + self.future_horizon + 1
            self.slicing_method = "block" if horizon < 20 else "direct"

        # transform
        self.transform_hmi = transform_hmi
        self.transform_aia = transform_aia

        # train / val / test split
        self.train_val_test = train_val_test
        self.split = split
        self.files_paths = self.get_filespaths()

        # assess the number of videos available in each file
        data_file_shape = self.infer_data_shape()
        self.sr_per_file = data_file_shape[0]  # nb of sun regions per file
        self.steps_per_file = data_file_shape[1]  # nb of time steps per file
        self.videos_per_file = self.steps_per_file - self.past_horizon - self.future_horizon
        self.native_resolution = data_file_shape[-2:]
        self.resolution = resolution
        self.use_interpolate = use_interpolate

        # get index mapping
        self.index_mapping = self.get_index_mapping()  # idx -> (file_idx, sr_idx, step)

        assert self.videos_per_file > 0, "Not enough time steps available in the dataset"
        assert all(r <= n for r, n in zip(self.resolution, self.native_resolution)), "resolution exceeds native resolution"

    def get_filespaths(self):
        # retrieve all files in the dataset
        days_paths = list(self.base_path.iterdir())
        # ignore empty directories
        filenames = []
        for dpath in days_paths:
            # ignore empty directories
            if dpath.is_dir() and any(dpath.iterdir()):
                # get the first file in the directory
                filenames.append(next(dpath.iterdir()))
        filenames = sorted(filenames)
    
        # define train/valid/test periods: 70%/15%/15%
        train_periods = [
            ("20140101", "20180531"),
        ]
        valid_periods = [
            ("20130101", "20130531"), 
            ("20180630", "20190130")
        ]
        test_periods = [
            ("20130801", "20131130"), 
            ("20190301", "20190930")
        ]
        def convert_periods_to_dates(periods):
            dates = []
            for start, end in periods:
                date_range = pd.date_range(
                    # start=datetime.datetime.strptime(start, "%Y%m%d"),
                    # end=datetime.datetime.strptime(end, "%Y%m%d"), There was a conflict with the import datetime and from datetime import datetime
                    start=datetime.strptime(start, "%Y%m%d"),
                    end=datetime.strptime(end, "%Y%m%d"),
                    freq='D'
                )
                dates.extend(date.strftime("%Y%m%d") for date in date_range)
            return dates
        train_dates = convert_periods_to_dates(train_periods)   
        valid_dates = convert_periods_to_dates(valid_periods)
        test_dates = convert_periods_to_dates(test_periods)

        # now retrieve the filenames for each period
        train_filenames = [f for f in filenames if f.parent.name in train_dates]
        valid_filenames = [f for f in filenames if f.parent.name in valid_dates]
        test_filenames = [f for f in filenames if f.parent.name in test_dates]
    
        file_paths = {
            'train': train_filenames,
            'valid': valid_filenames,
            'test': test_filenames,
        }[self.split]

        # filepaths to exclude
        if self.include_filters is not None:
            file_paths = [
                path for path in file_paths 
                if any(filter in path.name for filter in self.include_filters)
            ]
        if self.exclude_filters is not None:
            file_paths = [
                path for path in file_paths 
                if not any(filter in path.name for filter in self.exclude_filters)
            ]

        return file_paths
    
    def get_index_mapping(self):
        """Assemble the index -> (file_index, sr_index, step) mapping."""
        idx = 0
        active_regions = self.base_path / "active_regions.json"
        mapping = {}
        with open(active_regions, "r") as f:
            active_regions = json.load(f)
        for (i_file, i_traj) in product(range(len(self.files_paths)), range(self.sr_per_file)):
            # do we include this trajectory?
            day = self.files_paths[i_file].parent.name
            include_this_traj = not self.active_regions_only or active_regions[f"{day}_{i_traj}"]
            if include_this_traj:
                for i_step in range(self.videos_per_file):
                    mapping[(i_file, i_traj, i_step)] = idx
                    idx += 1
        return {v: k for k, v in mapping.items()}  # idx -> (file_idx, sr_region, step)

    def __len__(self):
        return len(self.index_mapping)

    def infer_data_shape(self):
        with h5py.File(self.files_paths[0], 'r') as h5_file:
            return h5_file['br'].shape
    
    def read_h5_dataset(self, d, sr_region, time_idces):
        """ Read a dataset from a h5 file. If method="block", accelerate the 
        reading by first reading an entire (contiguous) block of data, and then
        selecting the relevant time indices. It is faster than reading directly
        non-contiguous time indices, but uses more CPU memory.

        Args:
            d: the dataset to read
            sr_region: the region to read
            time_idces: the indices to read
        """
        if self.slicing_method == "block":
            min_i, max_i = time_idces.min(), time_idces.max()
            block = d[sr_region, min_i:max_i+1]
            return block[time_idces-min_i]
        elif self.slicing_method == "direct":
            return d[sr_region, time_idces]
        else:
            raise ValueError(f"Invalid method: {self.slicing_method}")

    def __getitem__(self, idx):
        # the files, trajectory, and step where to start looking for data
        file_idx, sr_region, step = self.index_mapping[idx]

        # the indices in the past and future for the video
        template = np.array(self.temporal_context.sample_template(seed=idx))
        video_idces = template + self.past_horizon + step
        # mask 
        mask = self.temporal_context.sample_mask(seed=idx)

        with h5py.File(self.files_paths[file_idx], 'r') as h5_file:
            # vector magnetogram data
            br = self.read_h5_dataset(h5_file['br'], sr_region, video_idces)
            bt = self.read_h5_dataset(h5_file['bt'], sr_region, video_idces)
            bp = self.read_h5_dataset(h5_file['bp'], sr_region, video_idces)
            # aia data
            aia = [
                self.read_h5_dataset(h5_file["aia_"+band[:-1]], sr_region, video_idces)
                for band in self.AIA_channels
            ]
            x = np.stack([br, bt, bp, *aia], axis=1)  # T x C x H x W
            x = torch.from_numpy(x).to(torch.float32)

            # error data
            br_err = self.read_h5_dataset(h5_file['br_err'], sr_region, video_idces)
            bt_err = self.read_h5_dataset(h5_file['bt_err'], sr_region, video_idces)
            bp_err = self.read_h5_dataset(h5_file['bp_err'], sr_region, video_idces)

            # TODO: Aia error data? Not providing anything for now
            x_err = np.stack([br_err, bt_err, bp_err], axis=1)
            x_err = torch.from_numpy(x_err).to(torch.float32)

            # position of the patch on the original full disk image
            pixel_y = self.read_h5_dataset(h5_file['pixel_y'], sr_region, video_idces)
            pixel_x = self.read_h5_dataset(h5_file['pixel_x'], sr_region, video_idces)
            center = np.stack([pixel_y, pixel_x], axis=1)
            center = torch.from_numpy(center).to(torch.int32)

            # times of the past and future images
            source_fnames = self.read_h5_dataset(h5_file['source_file'], sr_region, video_idces)
            times_str = [f[11:26].decode('utf-8') for f in source_fnames]
            # convert times to hours from start
            times_past = times_str[:self.past_frames]
            times_future = times_str[self.past_frames:]
            
            # longitude, latitude, (and mu)
            lat = self.read_h5_dataset(h5_file['lat'], sr_region, video_idces)
            lon = self.read_h5_dataset(h5_file['lon'], sr_region, video_idces)
            mu = self.read_h5_dataset(h5_file['mu'], sr_region, video_idces)
            # convert times to hours from present
            hours_delta = np.tile(template[:,None,None], (1,*lat.shape[1:]))
            coord = np.stack([hours_delta, lat, lon, mu], axis=1)  # Add hours as first channel
            coord = torch.from_numpy(coord).to(torch.float32)

            # transform
            if self.use_interpolate:
                # this is a new behavior for interpolating directly the tensor
                # rather than going from tensor to numpy then numpy to tensor
                x = F.interpolate(x, size=self.resolution, mode='bilinear')
                x_err = F.interpolate(x_err, size=self.resolution, mode='bilinear')
                coord = F.interpolate(coord, size=self.resolution, mode='bilinear')
            else:
                # will be deprecated
                x = subsample(x, self.resolution)
                x_err = subsample(x_err, self.resolution)
                coord = subsample(coord, self.resolution)
            if self.transform_hmi is not None:
                x[:, :3, :, :] = self.transform_hmi(x[:, :3, :, :])
                # TODO: should implement a transform for the error map as well
            if self.transform_aia is not None and self.AIA_channels:
                for i, band in enumerate(self.AIA_channels):
                    x[:, 3+i:3+i+1, :, :] = self.transform_aia[band](x[:, 3+i:3+i+1, :, :])

            return {
                # input (past)
                'input': x[:self.past_frames,:,:,:],
                'input_err': x_err[:self.past_frames,:,:,:],
                'input_coord': coord[:self.past_frames,:,:,:],
                'input_center': center[:self.past_frames,:],
                'input_times': times_past,
                # target (future)
                'target': x[self.past_frames:,:,:,:],
                'target_err': x_err[self.past_frames:,:,:,:],
                'target_coord': coord[self.past_frames:,:,:,:],
                'target_center': center[self.past_frames:,:],
                'target_times': times_future,
                # mask
                'mask': mask,
            }


def get_data_objects(
    batch_size,
    train_val_test,
    context_type,
    context_params,
    n_past,
    n_future,
    resolution,
    AIA_channels,
    distributed,
    num_data_workers,
    rank=None,
    world_size=None,
    split="train",
    include_filters=None,
    exclude_filters=None,
    active_regions_only=False,
    base_path_override=None,
    use_interpolate=True,
):
    # dataset
    # transform
    transform_hmi = ScaleTransform(scale=1000.0)
    transform_aia = {
        band: get_default_transforms(channel=band) for band in AIA_channels
    }
    # temporal context
    if context_type == "uniform":
        temporal_context = UniformContext(past_frames=n_past, future_frames=n_future)
    elif context_type == "multiscale_symmetrical":
        past_horizon, flexible = context_params
        temporal_context = MultiscaleSymmetricalContext(past_frames=n_past, past_horizon=past_horizon, flexible=flexible)
    elif context_type == "longrange_fixedpast":
        future_horizon = context_params[0]
        temporal_context = LongRangeFixedPastContext(past_frames=n_past, future_frames=n_future, future_horizon=future_horizon)
    elif context_type == "hierarchical2":
        temporal_context = Hierarchy2Context(past_frames=n_past, future_frames=n_future)
    else:
        raise ValueError(f"Invalid context type: {context_type}")
    dataset = SunRegionDataset(
        temporal_context=temporal_context,
        AIA_channels=AIA_channels,
        resolution=(resolution,resolution), 
        train_val_test=train_val_test,
        split=split,
        include_filters=include_filters,
        exclude_filters=exclude_filters,
        base_path_override=base_path_override,
        active_regions_only=active_regions_only,
        transform_hmi=transform_hmi,
        transform_aia=transform_aia,
        use_interpolate=use_interpolate,
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


if __name__ == "__main__":

    from time import time
    from src.utils import set_seed

    transform_hmi = ScaleTransform(scale=1000.0)
    transform_aia = {
        band.value: get_default_transforms(channel=band.value) for band in AIABand
    }
    dataset = SunRegionDataset(
        # temporal_context=MultiscaleSymmetricalContext(past_frames=4, alpha_min=1.1, alpha_max=2.5),
        temporal_context=UniformContext(past_frames=4, future_frames=3),
        resolution=(64, 64),
        train_val_test=[0.8, 0.1, 0.1],
        split="test",
        AIA_channels=[AIABand.AIA_171.value, AIABand.AIA_193.value],
        include_filters=None,
        exclude_filters=None,
        base_path_override="/mnt/home/polymathic/ceph/solar/sun_regions",
        active_regions_only=True,
        transform_hmi=transform_hmi,
        transform_aia=transform_aia,
    )
    with set_seed(0, backend="numpy"):
        idx = np.random.randint(0, len(dataset), 10)
    t1 = time()
    for i in idx:
        batch = dataset[int(i)]
    t2 = time() - t1
    print(batch["input"].shape)
    print(batch["target"].shape)
    print(batch["input_times"])
    print(len(dataset))