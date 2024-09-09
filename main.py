from utils import generate
from config import unet_model_path, vae_model_path, denoising_timesteps
from models import *
from train import train

from diffusers import DDPMScheduler
import torch

import os

def sample():
    if os.path.exists(unet_model_path) and os.path.exists(vae_model_path):
        vae.load_state_dict(torch.load(vae_model_path))
        unet.load_state_dict(torch.load(unet_model_path))
        noise_scheduler = DDPMScheduler(num_train_timesteps=denoising_timesteps)
        generate(vae, unet, noise_scheduler, 101)
    else:
        print("Models not trained, training...")
        train()

if __name__ == "__main__":
    sample()