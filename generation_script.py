# import debugpy

# debugpy.connect(("v000675", 5678))  # VS Code listens on login node
# print("✅ Connected to VS Code debugger!")
# debugpy.wait_for_client()
# print("🎯 Debugger attached!")


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

p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--config', type=str, required=True,
            help='the configuration file')

args = p.parse_args(["--config", "./configs/forecast_iono.json"])

config = K.config.load_config(args.config)
inner_model = K.config.make_model(config)
inner_model_ema = deepcopy(inner_model)
model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

ckpt = torch.load("/mnt/nas05/data01/francesco/progetto_simone/ionosphere/model_cond_forecasting_cfg_oneframe/model_00093975.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ema.inner_model.load_state_dict(ckpt['model_ema'])
model_ema.to(device)
model_ema.eval()

with torch.no_grad():
    for k, batch in enumerate(tqdm(val_dl, desc="Validation")):
        inpt = batch[0].contiguous().float().to(device, non_blocking=True)
        inpt = inpt.squeeze(2)  # shape: (8, 120, 24, 360)
        cond_img = inpt[:, :29, :, :]    # first 60 time steps :15
        target_img = inpt[:, 29, :, :].unsqueeze(1)  # last 60 time steps  15:
        cond_label = batch[1].to(device, non_blocking=True)

        cond_label_inp = cond_label[:, :, :]  # :16

        samples = generate_samples(model_ema, 1, device, cond_label=cond_label_inp[0].reshape(1, 30, 4), sampler="dpmpp_2m_sde", cond_img=cond_img[0].reshape(1, 29, 24, 360))

        # Save the oeiginal sample
        np.save(f"/mnt/nas05/data01/francesco/progetto_simone/results_1frame_test/original_forecasting_{k}.npy", cond_img[0].cpu().numpy())
        # Save the generated sampl
        np.save(f"/mnt/nas05/data01/francesco/progetto_simone/results_1frame_test/sample_forecasting_{k}.npy", samples.cpu().numpy())
        # Save the condition
        np.save(f"/mnt/nas05/data01/francesco/progetto_simone/results_1frame_test/cond_{k}.npy", cond_label.cpu().numpy())

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
        imageio.mimsave(f'/mnt/nas05/data01/francesco/progetto_simone/results_1frame_test/sequence_gen_{k}.gif', frames, duration=1)
