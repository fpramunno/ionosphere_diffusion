import debugpy

debugpy.connect(("v000675", 5678))  # VS Code listens on login node
print("✅ Connected to VS Code debugger!")
debugpy.wait_for_client()
print("🎯 Debugger attached!")


import sys
sys.path.append("/mnt/nas05/data01/francesco/progetto_simone/ionosphere")
from src.data.dataset import IonoDataset
import torch
import src as K
import argparse
from copy import deepcopy
from util import generate_samples
import numpy as np


train_dataset = IonoDataset(
    path="/mnt/nas05/data01/francesco/sdo_img2img/sde_mag2mag_v2/progetto_simone/data/pickled_maps",
    transforms=True,
    split='train',
    seed=42
)

inpt, cond_label = train_dataset[0]

p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--config', type=str, required=True,
            help='the configuration file')

args = p.parse_args(["--config", "./configs/3dmag.json"])

config = K.config.load_config(args.config)
inner_model = K.config.make_model(config)
inner_model_ema = deepcopy(inner_model)
model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

ckpt = torch.load("/mnt/nas05/data01/francesco/progetto_simone/ionosphere/model_cond_generation/model_00315780.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ema.inner_model.load_state_dict(ckpt['model_ema']).to(device)

samples = generate_samples(model_ema, 1, device, cond_label=cond_label.unsqueeze(0).to(device), sampler="dpmpp_2m_sde")

# Save the oeiginal sample
np.save("/mnt/nas05/data01/francesco/progetto_simone/ionosphere/np_generated_data/original.npy", inpt[0].cpu().numpy())
# Save the generated sample
np.save("/mnt/nas05/data01/francesco/progetto_simone/ionosphere/np_generated_data/sample.npy", samples[0][0].cpu().numpy())
# Save the condition
np.save("/mnt/nas05/data01/francesco/progetto_simone/ionosphere/np_generated_data/cond.npy", cond_label.cpu().numpy())
