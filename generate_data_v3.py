import sys
sys.path.append("/mnt/nas05/data01/francesco/progetto_simone/ionosphere")
import torch
import src as K
import argparse
from copy import deepcopy
from util import generate_samples
import numpy as np
from src.data.dataset import get_sequence_data_objects
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import numpy as np
import torch
import os
from IPython import embed

# LOAD DATA
train_dataset, train_sampler, train_dl = get_sequence_data_objects(
        csv_path="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/l1_earth_associated_with_maps.csv",
        transform_cond_csv="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/params.csv",
        batch_size=1,
        distributed=False,
        num_data_workers=1,
        split='train',
        seed=42,
        sequence_length=30,
        normalization_type="absolute_max",
        use_l1_conditions=True,
        min_center_distance=5,
        cartesian_transform=True,  # Convert to Cartesian circular grid
        output_size=64,  # Output grid size for Cartesian transform
        only_complete_sequences=True,  # Filter out sequences with missing frames
    )

val_dataset, val_sampler, val_dl = get_sequence_data_objects(
        csv_path="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/l1_earth_associated_with_maps.csv",
        transform_cond_csv="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/params.csv",
        batch_size=1,
        distributed=False,
        num_data_workers=1,
        split='valid',
        seed=42,
        sequence_length=30,
        normalization_type="absolute_max",
        use_l1_conditions=True,
        min_center_distance=5,
        cartesian_transform=True,  # Convert to Cartesian circular grid
        output_size=64,  # Output grid size for Cartesian transform
        only_complete_sequences=True,  # Filter out sequences with missing frames
    )

overfit_single = True

# Overfitting on a single trajectory
if overfit_single:

    # Get one batch from the training set (batch_images, batch_conditions)
    batch_images, batch_conditions = next(iter(train_dl))

    # Extract the FIRST sample from the batch (remove batch dimension)
    # batch_images: [batch_size, seq_len, C, H, W] -> [seq_len, C, H, W]
    # batch_conditions: [batch_size, seq_len, num_cond] -> [seq_len, num_cond]
    single_image = batch_images[0]
    single_condition = batch_conditions[0]

    # Create a dataset that repeats this single sample
    class SingleSampleDataset(torch.utils.data.Dataset):
        def __init__(self, image, condition, repeat=1000):
            self.image = image
            self.condition = condition
            self.repeat = repeat

        def __len__(self):
            return self.repeat

        def __getitem__(self, idx):
            return self.image, self.condition

    # Repeat the single sample enough times for a reasonable epoch
    # With batch_size, this gives ~100 batches per epoch for quick overfitting tests
    repeat_count = 100 * 1
    single_dataset = SingleSampleDataset(single_image, single_condition, repeat=repeat_count)


    # Create new dataloader with the single sample
    # Use original batch_size so the model sees the expected batch dimension
    train_dl = torch.utils.data.DataLoader(
        single_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Also use the same single sample for validation
    val_dl = torch.utils.data.DataLoader(
        single_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

## SETUP MODEL

# embed()

p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--config', type=str, required=True,
            help='the configuration file')

args = p.parse_args(["--config", "/mnt/nas05/data01/francesco/progetto_simone/ionosphere/configs/forecast_iono_15_big_cosine_solar.json"])

config = K.config.load_config(args.config)
inner_model = K.config.make_model(config)
model_ema = K.config.make_denoiser_wrapper(config)(inner_model)

ckpt = torch.load("/mnt/nas05/data01/francesco/progetto_simone/ionosphere/models_results/models_cond_forecasting_15frames_overfit_15step_v4_cartesian_coordinates/model_epoch_0000.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ema.inner_model.load_state_dict(ckpt['model_ema'])
model_ema.to(device)
model_ema.eval()

import os

os.mkdir("/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/input_imgs")
os.mkdir("/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/generated_imgs")
os.mkdir("/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/gifs")
os.mkdir("/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/ground_truth")
os.mkdir("/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/conditions")

cartesian_transform = True

with torch.no_grad():
    for k, batch in enumerate(tqdm(val_dl, desc="Validation")):
        inpt = batch[0].contiguous().float().to(device, non_blocking=True)
        inpt = inpt.squeeze(2)  # shape: (8, 120, 24, 360)
        cond_img = inpt[:, :15, :, :]    # first 60 time steps :15
        target_img = inpt[:, 15:, :, :].unsqueeze(1)  # last 60 time steps  15:
        cond_label = batch[1].to(device, non_blocking=True)

        cond_label_inp = cond_label[:, :, :]  # :16

        if cartesian_transform:
            spatial_shape = (64, 64)
        else:
            spatial_shape = (24, 360)

        # embed()
        # samples = generate_samples(model_ema, 1, device, cond_label=cond_label_inp[:, :, :], sampler="dpmpp_2m_sde", cond_img=cond_img[0].reshape(1, 15, 24, 360), num_pred_frames=15, step=50)
        samples = generate_samples(model_ema, 1, device, cond_label=cond_label_inp[:, :, :], sampler="dpmpp_2m_sde", cond_img=cond_img[0].reshape(1, 15, *spatial_shape), num_pred_frames=15).cpu()

        # Save the oeiginal sample
        np.save(f"/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/input_imgs/original_forecasting_{k}.npy", cond_img[0].cpu().numpy())
        # Save the generated sampl
        np.save(f"/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/generated_imgs/sample_forecasting_{k}.npy", samples.cpu().numpy())
        # Save the target sample
        np.save(f"/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/ground_truth/arget_forecasting_{k}.npy", target_img[0].cpu().numpy())
        # Save the condition
        np.save(f"/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/conditions/cond_{k}.npy", cond_label.cpu().numpy())

        frames = []
        data_seq = samples[0].unsqueeze(1)  # shape: [20, 1, 24, 360]
        # Dynamically set figsize based on image shape for better fit
        img_h, img_w = data_seq.shape[2], data_seq.shape[3]
        aspect = img_w / img_h
        base_height = 4  # inches
        figsize = (base_height * aspect, base_height)

        for t in range(data_seq.shape[0]):
            img = data_seq[t, 0].cpu().numpy()  # shape: [24, 360]
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(img, cmap='viridis', aspect='auto')
            ax.set_title(f"Time step {t}")
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # Convert plot to image array
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)

        # Save as gif
        imageio.mimsave(f'/mnt/nas05/data01/francesco/progetto_simone/results_30predstep_v3_big_cosine_overfit_v4/gifs/sequence_gen_{k}.gif', frames, duration=1)