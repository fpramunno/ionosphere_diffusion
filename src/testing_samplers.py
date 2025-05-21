# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:24:16 2024

@author: pio-r
"""

import os
# os.chdir(r"/mnt/nas05/data01/francesco/sdo_img2img")
import numpy as np
os.chdir(r"\\10.35.146.35\data01\francesco\sdo_img2img\k_diffusion\k_diffusion")
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Normalize
import pandas as pd
from sunpy.visualization.colormaps import cm
# Preprocessing

from torchvision.transforms import Compose, Resize, Normalize, Lambda
import math

#Channels that correspond to HMI Magnetograms 
HMI_WL = ['Bx','By','Bz']
#A colormap for visualizing HMI
HMI_CM = LinearSegmentedColormap.from_list("bwrblack", ["#0000ff","#000000","#ff0000"])

def channel_to_map(name):
    """Given channel name, return colormap"""
    return HMI_CM if name in HMI_WL else cm.cmlist.get('sdoaia%d' % int(name))

def get_clip(X, name):
    """Given an image and the channel name, get the right clip"""
    return get_signed_pct_clip(X) if name in HMI_WL else get_pct_clip(X)

def get_pct_clip(X):
    """Return the 99.99th percentile"""
    return (0,np.quantile(X.ravel(),0.999))

def get_signed_pct_clip(X):
    """Return the 99.99th percentile by magnitude, but symmetrize it so 0 is in the middle"""
    v = np.quantile(np.abs(X.ravel()),0.999)
    return (-v,v)

def vis(X, cm, clip=None):
    """Given image, colormap, and a clipping, visualize results"""
    Xc = X
    if clip:
        Xc = np.clip((X-clip[0])/(clip[1]-clip[0]),0,1)
    Xcv = cm(Xc)
    return (Xcv[:,:,:3]*255).astype(np.uint8)

def show_grid(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    nrows=int(len(imgs)/4)
    ncols=4
    if nrows <= 0:
        nrows = 1
        ncols = len(imgs)
    fix, axs = plt.subplots(figsize=(24,24), ncols=ncols, nrows=nrows, squeeze=False)
    row = 0
    for i, img in enumerate(imgs):
        col = i % 4
        if i != 0 and i % 4 == 0:
            row = row + 1
        axs[row, col].imshow(img)
        axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.pause(10)
    plt.show()
    
def tensor_to_img(img, wave):
    V = []
    imgs = []
    for i in range(img.shape[0]):
        imgs.append(img[i])
    for images in imgs:
        images = images.permute(1, 2, 0)
        images = np.squeeze(images.cpu().numpy())
        v = vis(images, channel_to_map(wave))
        v = Image.fromarray(v)
        V.append(v)
    return show_grid(V)

CHANNEL_PREPROCESS = {
    "94A": {"min": 0.1, "max": 800, "scaling": "log10"},
    "131A": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "171A": {"min": 5, "max": 3500, "scaling": "log10"},
    "193A": {"min": 20, "max": 5500, "scaling": "log10"},
    "211A": {"min": 7, "max": 3500, "scaling": "log10"},
    "304A": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "335A": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600A": {"min": 10, "max": 800, "scaling": "log10"},
    "1700A": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500A": {"min": 4000, "max": 20000, "scaling": "log10"},
    "continuum": {"min": 0, "max": 65535, "scaling": None},
    "magnetogram": {"min": -3000, "max": 3000, "scaling": None},
    "Bx": {"min": -250, "max": 250, "scaling": None},
    "By": {"min": -250, "max": 250, "scaling": None},
    "Bz": {"min": -250, "max": 250, "scaling": None},
}


def get_default_transforms(target_size=256, channel="171", mask_limb=False, radius_scale_factor=1.0):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.

    Apply the normalization necessary for the SDO ML Dataset. Depending on the channel, it:
      - masks the limb with 0s
      - clips the "pixels" data in the predefined range (see above)
      - applies a log10() on the data
      - normalizes the data to the [0, 1] range
      - normalizes the data around 0 (standard scaling)

    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 256.
        channel (str, optional): [The SDO channel]. Defaults to 171.
        mask_limb (bool, optional): [Whether to mask the limb]. Defaults to False.
        radius_scale_factor (float, optional): [Allows to scale the radius that is used for masking the limb]. Defaults to 1.0.
    Returns:
        [Transform]
    """

    transforms = []

    # also refer to
    # https://pytorch.org/vision/stable/transforms.html
    # https://github.com/i4Ds/SDOBenchmark/blob/master/dataset/data/load.py#L363
    # and https://gitlab.com/jdonzallaz/solarnet-thesis/-/blob/master/solarnet/data/transforms.py
    preprocess_config = CHANNEL_PREPROCESS[channel]

    if preprocess_config["scaling"] == "log10":
        # TODO does it make sense to use vflip(x) in order to align the solar North as in JHelioviewer?
        # otherwise this has to be done during inference
        def lambda_transform(x): return torch.log10(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - \
            math.log10(preprocess_config["min"])
    elif preprocess_config["scaling"] == "sqrt":
        def lambda_transform(x): return torch.sqrt(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.sqrt(preprocess_config["min"])
        std = math.sqrt(preprocess_config["max"]) - \
            math.sqrt(preprocess_config["min"])
    else:
        def lambda_transform(x): return torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        )
        mean = preprocess_config["min"]
        std = preprocess_config["max"] - preprocess_config["min"]

    def limb_mask_transform(x):
        h, w = x.shape[1], x.shape[2]  # C x H x W

        # fixed disk size of Rs of 976 arcsec, pixel size in the scaled image (512x512) is ~4.8 arcsec
        original_resolution = 4096
        scaled_resolution = h
        pixel_size_original = 0.6
        radius_arcsec = 976.0
        radius = (radius_arcsec / pixel_size_original) / \
            original_resolution * scaled_resolution

        mask = create_circular_mask(
            h, w, radius=radius*radius_scale_factor)
        mask = torch.as_tensor(mask, device=x.device)
        return torch.where(mask, x, torch.tensor(0.0))

    if mask_limb:
        def mask_lambda_func(x):
            return limb_mask_transform(x)
        transforms.append(mask_lambda_func)
        # transforms.append(Lambda(lambda x: limb_mask_transform(x)))

    transforms.append(Resize((target_size, target_size)))
    # TODO find out if these transforms make sense
    def test_lambda_func(x):
        return lambda_transform(x)
    transforms.append(test_lambda_func)
    # transforms.append(Lambda(lambda x: lambda_transform(x)))
    transforms.append(Normalize(mean=[mean], std=[std]))
    # required to remove strange distribution of pixels (everything too bright)
    transforms.append(Normalize(mean=(0.5), std=(0.5)))

    return Compose(transforms)

def create_circular_mask(h, w, center=None, radius=None):
    # TODO investigate the use of a circular mask to prevent focussing to much on the limb
    # https://gitlab.com/jdonzallaz/solarnet-app/-/blob/master/src/prediction.py#L9

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask




def save_images(img, path):
    V = []
    imgs = []
    for i in range(img.shape[0]):
        imgs.append(img[i])
    for images in imgs:
        images = images.permute(1, 2, 0)
        images = np.squeeze(images.cpu().numpy())
        # v = vis(images, channel_to_map(171))
        v = Image.fromarray(images)
        V.append(v)
    for value in V:
        value.save(path)

def save_tensor_as_png(tensor, filename):
    # Make sure the tensor is in the CPU and detach it from the computational graph
    tensor = tensor.detach().cpu()

    # Convert the tensor to a PIL image
    if tensor.shape[0] == 1:
        # If the input tensor has only 1 channel, convert it to a grayscale image
        image = Image.fromarray((tensor.squeeze(0).numpy() * 255).astype('uint8'), mode='L')
    else:
        # If the input tensor has 3 channels, convert it to an RGB image
        image = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))

    # Save the image to the specified filename
    image.save(filename)


# Data loading

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from astropy.io import fits

mapping = {'A': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from astropy.io import fits
import glymur

mapping = {'A': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}

# lb_used = []

class PairedJP2Dataset(Dataset):
    def __init__(self, dir2, dir3, dir_171, dir_1600, dir_1700, dir_doppl, labels_list, time_mag_list, transform2=None,
                 transform_171=None, transform_1600=None, transform_1700=None, transform_doppl=None):
        super(PairedJP2Dataset, self).__init__()
        
        # Ensure directories exist
        assert os.path.isdir(dir2), f"{dir2} is not a directory."
        assert os.path.isdir(dir3), f"{dir3} is not a directory."
        assert os.path.isdir(dir_171), f"{dir_171} is not a directory."
        assert os.path.isdir(dir_1600), f"{dir_1600} is not a directory."
        assert os.path.isdir(dir_1700), f"{dir_1700} is not a directory."
        assert os.path.isdir(dir_doppl), f"{dir_doppl} is not a directory."

        self.dir2_files = sorted([os.path.join(dir2, fname) for fname in os.listdir(dir2) if fname.endswith('.fits')])
        self.dir3_files = sorted([os.path.join(dir3, fname) for fname in os.listdir(dir3) if fname.endswith('.fits')])
        self.dir_171_files = sorted([os.path.join(dir_171, fname) for fname in os.listdir(dir_171) if fname.endswith('.fits')])
        self.dir_1600_files = sorted([os.path.join(dir_1600, fname) for fname in os.listdir(dir_1600) if fname.endswith('.fits')])
        self.dir_1700_files = sorted([os.path.join(dir_1700, fname) for fname in os.listdir(dir_1700) if fname.endswith('.fits')])
        self.dir_doppl_files = sorted([os.path.join(dir_doppl, fname) for fname in os.listdir(dir_doppl) if fname.endswith('.fits')])
        assert len(self.dir2_files) == len(self.dir3_files), "Directories have different number of .fits files."
        self.transform2 = transform2
        self.transform_171 = transform_171
        self.transform_1600 = transform_1600
        self.transform_1700 = transform_1700
        self.transform_doppl = transform_doppl
        self.labels = labels_list
        self.time_mag = time_mag_list

    def __len__(self):
        return len(self.dir2_files)

    def __getitem__(self, idx):
        
        
        with fits.open(self.dir2_files[idx]) as hdul:
            data2 = hdul[1].data
            data2 = np.nan_to_num(data2, nan=np.nanmin(data2))
            header_1 = hdul[1].header
            
        
        with fits.open(self.dir3_files[idx]) as hdul:
            data3 = hdul[1].data
            data3 = np.nan_to_num(data3, nan=np.nanmin(data3))
            header_2 = hdul[1].header
        
        with fits.open(self.dir_171_files[idx]) as hdul:
            data171 = hdul[1].data
            data171 = np.nan_to_num(data171, nan=np.nanmin(data171))
        
        
        with fits.open(self.dir_1600_files[idx]) as hdul:
            data1600 = hdul[1].data
            data1600 = np.nan_to_num(data1600, nan=np.nanmin(data1600))
            
        
        with fits.open(self.dir_1700_files[idx]) as hdul:
            data1700 = hdul[1].data
            data1700 = np.nan_to_num(data1700, nan=np.nanmin(data1700))
            
        
        i = 0
        while True:
            with fits.open(self.dir_doppl_files[idx]) as hdul:
                datadoppl = hdul[i].data
            if isinstance(datadoppl, np.ndarray):
                break
            else:
                i =+ 1
                continue
            
        datadoppl = np.nan_to_num(datadoppl, nan=np.nanmin(datadoppl))
            
        # try:
        #     datadoppl = to_tensor(datadoppl.newbyteorder('=').byteswap())
        # except:
        
        
        alpha = 25
        beta = 6
        flattened_tensor1 = data2.flatten()
        flattened_tensor2 = data3.flatten()
        # flattened_tensor3 = datadoppl.flatten()
        
        data2 = np.arcsinh(flattened_tensor1 / alpha) / beta
        data3 = np.arcsinh(flattened_tensor2 / alpha) / beta
        # datadoppl = np.arcsinh(flattened_tensor3 / alpha) / beta
        
        data2 = rotate(res(torch.from_numpy(data2).reshape(1, 1024, 1024))).float()
        data3 = rotate(res(torch.from_numpy(data3).reshape(1, 1024, 1024))).float()
        # datadoppl = rotate(res(torch.from_numpy(datadoppl).reshape(1, 1024, 1024))).float()
        data171 = (to_tensor(data171)).float()
        data1600 = (to_tensor(data1600)).float()
        data1700 = (to_tensor(data1700)).float()
        
        if self.transform_171:
            data171 = self.transform_171(data171)
            
        
        if self.transform_1600:
            data1600 = self.transform_1600(data1600)
        
        
        if self.transform_1700:
            data1700 = self.transform_1700(data1700)
            
        datadoppl = to_tensor(datadoppl)
        
        if self.transform_doppl:
            datadoppl = self.transform_doppl(datadoppl)
            
        
            
        data2norm = 2*(data2 - torch.min(data2)) / (torch.max(data2) - torch.min(data2)) - 1
        data3norm = 2*(data3 - torch.min(data3)) / (torch.max(data3) - torch.min(data3)) - 1
        datadoppl = 2*(datadoppl - torch.min(datadoppl)) / (torch.max(datadoppl) - torch.min(datadoppl)) - 1
                     
            
        label = self.labels[idx]
        time_mag = self.time_mag[idx]    
        
        
        return data2norm, data3norm, data171, data1600, data1700, datadoppl, header_1['T_OBS'], header_2['T_OBS'], self.dir2_files[idx]

# Example usage:
# dir2 = '/mnt/nas05/data01/francesco/sdo_data/final_dataset/filtered_mag_24_all/'
dir2 = '//10.35.146.35/data01/francesco/sdo_data/final_dataset/filtered_mag_24_all/'
# dir3 = '/mnt/nas05/data01/francesco/sdo_data/final_dataset/filtered_mag_flare_all/'
dir3 = '//10.35.146.35/data01/francesco/sdo_data/final_dataset/filtered_mag_flare_all/'
# dir171 = '/mnt/nas05/data01/francesco/sdo_data/final_dataset/filtered_171_24_1024_all/'
dir171 = '//10.35.146.35/data01/francesco/sdo_data/final_dataset/filtered_171_24_1024_all/'
# dir1600 = '/mnt/nas05/data01/francesco/sdo_data/final_dataset/filtered_1600_24_1024_all/'
dir1600 = '//10.35.146.35/data01/francesco/sdo_data/final_dataset/filtered_1600_24_1024_all/'
# dir1700 = '/mnt/nas05/data01/francesco/sdo_data/final_dataset/filtered_1700_24_1024_all/'
dir1700 = '//10.35.146.35/data01/francesco/sdo_data/final_dataset/filtered_1700_24_1024_all/'
# dirdoppl = '/mnt/nas05/data01/francesco/sdo_data/final_dataset/filtered_dopplergram_corrected_all/'
dirdoppl = '//10.35.146.35/data01/francesco/sdo_data/final_dataset/filtered_dopplergram_corrected_all/'


# df_lab = pd.read_csv("/mnt/nas05/data01/francesco/sdo_data/final_dataset/df_all_final_v2.csv")
df_lab = pd.read_csv("//10.35.146.35/data01/francesco/sdo_data/final_dataset/df_all_final_v2.csv")


# df_lab.to_csv('./lab_used.csv')

# transform_hmi = get_default_transforms(
#     target_size=64, channel="continuum", mask_limb=False, radius_scale_factor=1.0)
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class CustomRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(img, self.angle)


res = Resize((256, 256))

transform_2 = get_default_transforms(
    target_size=256, channel="magnetogram", mask_limb=False, radius_scale_factor=1.0)

transform_171 = get_default_transforms(
    target_size=256, channel="171A", mask_limb=False, radius_scale_factor=1.0)

transform_1600 = get_default_transforms(
    target_size=256, channel="1600A", mask_limb=False, radius_scale_factor=1.0)

transform_1700 = get_default_transforms(
    target_size=256, channel="1700A", mask_limb=False, radius_scale_factor=1.0)


transform_doppl = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomHorizontalFlip(p=1.0),  # This line adds a 90-degree rotation to the right
    # transforms.Normalize(mean=(-3000), std=(6000)),
    # transforms.Normalize(mean=(0.5), std=(0.5))
])

rotate = transforms.Compose([
    
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomHorizontalFlip(p=1.0),])

to_tensor = transforms.ToTensor()

dataset = PairedJP2Dataset(dir2, dir3, dir171, dir1600, dir1700, dirdoppl, df_lab['Label'], df_lab['MAG'],
                           transform2=transform_2,
                           transform_171=transform_171,
                           transform_1600=transform_1600,
                           transform_1700=transform_1700, transform_doppl=transform_doppl)

import torch
import concurrent.futures
from tqdm import tqdm




from torch.utils.data import random_split

total_samples = len(dataset)
train_size = int(0.7 * total_samples)  # Using 80% for training as an example
val_size = total_samples - train_size


torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_data = DataLoader(train_dataset, batch_size=1,
                          shuffle=False,)
                            # pin_memory=True,# pin_memory set to True
                            # num_workers=12,
                            # prefetch_factor=4,
                            # drop_last=False)

val_data = DataLoader(val_dataset, batch_size=1,
                          shuffle=False,)
                            # pin_memory=True,# pin_memory set to True
                            # num_workers=12,
                            # prefetch_factor=4,  # pin_memory set to True
                            # drop_last=False)

print('Train loader and Valid loader are up!')

from modules import PaletteModelV2, EMA
from diffusion import Diffusion_cond
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)


dataloader = train_data
dataloader_val = val_data
device = 'cuda'
model = PaletteModelV2(c_in=6, c_out=1, num_classes=5,  image_size=int(64), true_img_size=64).to(device)
ckpt = torch.load(r"\\10.35.146.35\data01\francesco\sdo_img2img\models_256_DDPM_171_1600_1700_DOPPL_V2\DDPM_Conditional\checkpoint.pt")
model.load_state_dict(ckpt['model_state'])

diffusion = Diffusion_cond(img_size=256, device=device, noise_steps=50)
from external import DiscreteEpsDDPMDenoiser
# class CustomDenoiser(DiscreteEpsDDPMDenoiser):
#     def __init__(self, model, diffusion, y=None, labels=None):
#         # Initialize the parent class with alphas_cumprod
#         super().__init__(model, diffusion.alpha_hat, quantize=False)
#         self.diffusion = diffusion
#         self.y = y
#         self.labels = labels

#     def sigma_to_t(self, sigma):
#         # Convert sigma to the corresponding timestep
#         return self.diffusion.sigma_to_timestep(sigma)

#     def forward(self, x, sigma, **kwargs):
#         # Map sigma to the corresponding timestep
#         t = self.sigma_to_t(sigma)
#         # Ensure t is a tensor of appropriate shape
#         # t = torch.tensor([t] * x.size(0), device=x.device, dtype=torch.long)
#         # Call the model with the mapped timestep and additional inputs
#         return self.inner_model(x, self.y, self.labels, t)

alphas_cumprod = diffusion.return_alphacumprof()
denoiser = DiscreteEpsDDPMDenoiser(model, alphas_cumprod, True)

from sampling import get_sigmas_karras
from sampling import sample_euler
pbar_val = tqdm(val_data)
model.train()

for i, (image_24, image_peak, image_171, image_1600, image_1700, image_doppl, time1, time2, path) in enumerate(pbar_val):
    img_24 = image_24.to(device).float()
    img_peak = image_peak.to(device).float()
    image_171 = image_171.to(device).float()
    image_1600 = image_1600.to(device).float()
    image_1700 = image_1700.to(device).float()
    image_doppl = image_doppl.to(device).float()
    
    img_input = torch.cat([img_24, image_171, image_1600, image_1700, image_doppl], dim=1)
    
    # denoiser = CustomDenoiser(model, diffusion, y=img_input, labels=None)
    
    
    sigmas = diffusion.return_sigma()
    
    sigmas = get_sigmas_karras(n=50, sigma_min=sigmas.min().item(), sigma_max=sigmas.max().item(), rho=7., device=device)
    
   
    # alphas_cumprod = diffusion.return_alphacumprof()
    # sigmas = diffusion.return_sigma()

    x = torch.randn((1, 1, 256, 256), device=device) * sigmas[0]
    
    x = torch.cat([x, img_input], dim=1)
    # plt.imshow(x[0].permute(1, 2, 0).cpu().numpy())
    
    samples = sample_euler(denoiser, x, sigmas)
    # samples = sample_heun(denoiser, x, sigmas)

    
    ema_sampled_images = diffusion.sample(model, y=img_input[0].reshape(1, 5, 256, 256), labels=None, n=1)
    
    plt.imshow(samples[0][5].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy())
    # plt.imshow(x[0][0].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy())
    plt.imshow(ema_sampled_images[0].permute(1, 2, 0).cpu().numpy())