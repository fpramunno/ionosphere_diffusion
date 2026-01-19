import sys
sys.path.append("/mnt/nas05/data01/francesco/progetto_simone/ionosphere")
import torch
import src as K
import argparse
from copy import deepcopy
from util import generate_samples
import numpy as np
from src.data.dataset import get_sequence_data_objects_iterable
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import numpy as np
import torch
import os
from IPython import embed

## DATASET SETUP

val_dataset, val_sampler, val_dl = get_sequence_data_objects_iterable(
        csv_path="/users/framunno/data/ionosphere/l1_to_map_matched_even_minutes_test_v3_deduplicated.csv",
        transform_cond_csv="/users/framunno/data/ionosphere/params.csv",
        batch_size=1,
        # distributed=False,
        num_data_workers=1,
        split='valid',
        seed=42,
        sequence_length=30,
        normalization_type="absolute_max",
        use_l1_conditions=True,
        min_center_distance=30,
        cartesian_transform=True,  # Convert to Cartesian circular grid
        output_size=64,  # Output grid size for Cartesian transform
        only_complete_sequences=True,  # Filter out sequences with missing frames
        activity_filter='high',  # Only sequences with epsilon ≥ 75th percentile
        epsilon_high_quantile=0.75
    )


## SETUP MODEL

# embed()

p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--config', type=str, required=True,
            help='the configuration file')

args = p.parse_args(["--config", "/users/framunno/projects/ionosphere_diffusion/configs/forecast_iono_15_big_cosine_solar.json"])

config = K.config.load_config(args.config)
inner_model = K.config.make_model(config)
model_ema = K.config.make_denoiser_wrapper(config)(inner_model)

# embed()
ckpt = torch.load("/capstor/scratch/cscs/framunno/models_results/models_ViT_forecast_15frames_absolute_max_ddp_bs1_NOCOND_v3_interpolated_deduplicated/model_epoch_0100.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ema.inner_model.load_state_dict(ckpt['model_ema'])
model_ema.to(device)
model_ema.eval()

import os

# Change only this line for different experiments
base_dir = "/capstor/scratch/cscs/framunno/results_ViT_800mln_NOCOND_V3_075"

os.makedirs(os.path.join(base_dir, "input_imgs"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "generated_imgs"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "gifs"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "ground_truth"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "conditions"), exist_ok=True)

cartesian_transform = True
no_mapping_cond = True

with torch.no_grad():
    for k, batch in enumerate(tqdm(val_dl, desc="Validation")):
        inpt = batch[0].contiguous().float().to(device, non_blocking=True)
        inpt = inpt.squeeze(2)  # shape: (8, 120, 24, 360)
        cond_img = inpt[:, :15, :, :]    # first 60 time steps :15
        target_img = inpt[:, 15:, :, :].unsqueeze(1)  # last 60 time steps  15:
        cond_label = batch[1].to(device, non_blocking=True)

        # embed()

        cond_label_inp = cond_label[:, :, :].repeat(20, 1, 1) # :16

        if cartesian_transform:
            spatial_shape = (64, 64)
        else:
            spatial_shape = (24, 360)

        cond_label_sample = None if no_mapping_cond else cond_label_inp[:, :, :]

        samples = generate_samples(model_ema, 20, device, cond_label=cond_label_sample, sampler="dpmpp_2m_sde", cond_img=cond_img[0].reshape(1, 15, *spatial_shape).repeat(20, 1, 1, 1), num_pred_frames=15).cpu()

        # Save the original sample
        np.save(os.path.join(base_dir, f"input_imgs/original_forecasting_{k}.npy"), cond_img[0].cpu().numpy())
        # Save the generated sample
        np.save(os.path.join(base_dir, f"generated_imgs/sample_forecasting_{k}.npy"), samples.cpu().numpy())
        # Save the target sample
        np.save(os.path.join(base_dir, f"ground_truth/target_forecasting_{k}.npy"), target_img[0].cpu().numpy())
        # Save the condition
        np.save(os.path.join(base_dir, f"conditions/cond_{k}.npy"), cond_label[0].cpu().numpy())