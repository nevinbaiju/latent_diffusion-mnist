from config import (latent_channels, vae_plots_path, 
                    unet_plots_path, device)

from diffusers import DDPMScheduler
import torch

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from tqdm import tqdm

def create_path_if_not_exists(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

def revert_images(imgs: torch.tensor) -> np.array:
    """
    Converts the torch.tensor of images to np.array by remapping values in the range of (0-255)
    """
    h = imgs.shape[-1]
    imgs = imgs.cpu().detach().numpy()
    min_vals = imgs.min(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    max_vals = imgs.max(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]

    imgs = ((max_vals - imgs)/(max_vals-min_vals))*255
    if imgs.shape[1] == 1:
        imgs = imgs.astype(int).reshape(-1, h, h)

    return imgs

def plot_side_by_side(images_y: torch.tensor, images_pred: torch.tensor, 
                      latents: torch.tensor, epoch: int):
    """
    Visualize the performance of VAE and its latent channels
    """

    images_y, images_pred = revert_images(images_y), revert_images(images_pred)
    latents = revert_images(latents)
    idx = np.random.randint(0, images_y.shape[0])
    fig, axs = plt.subplots(1, 2)
    
    # Plot input image and Output image
    axs[0].imshow(images_y[idx], cmap='gray')
    axs[0].axis('off')  
    axs[0].set_title("Input")
    
    axs[1].imshow(images_pred[idx], cmap='gray')
    axs[1].axis('off')  
    axs[1].set_title("Output")
    plt.savefig(os.path.join(vae_plots_path, f'epoch_{epoch}_input_output.png'))
    plt.clf()

    latent_channels = latents.shape[1]
    fig, axs = plt.subplots(1, 4)

    # Plot the different latent channels
    for i in range(latent_channels):
        axs[i].imshow(latents[idx, i, :, :], cmap='gray')
        axs[i].axis('off')  
        axs[i].set_title(f"Latent channel: {i}", fontsize=8)       
    plt.savefig(os.path.join(vae_plots_path, f'epoch_{epoch}_latent_channels.png'))
    plt.clf()

def generate(vae: torch.nn.Module, unet: torch.nn.Module, noise_scheduler: DDPMScheduler, epoch: int):
    """
    Generate the samples from Unet.
    """
    def plot(recon_imgs, timesteps, epoch):
        create_path_if_not_exists(os.path.join(unet_plots_path, f'epoch_{epoch}'))
        recon_imgs = revert_images(recon_imgs.sample)
        fig, axs = plt.subplots(2, 5)
        for i in range(10):
            axs[i//5][i%5].imshow(recon_imgs[i], cmap='gray')
            axs[i//5][i%5].axis('off')  
            axs[i//5][i%5].set_title(str(i))
        plt.suptitle(f"Timesteps: {timesteps}")
        plt.savefig(os.path.join(unet_plots_path, f'epoch_{epoch}', f'plot {timesteps}.png'))
        plt.clf()
        
    latents = torch.randn((10, latent_channels, 8, 8)).to(device)
    labels = torch.arange(10).to(device)

    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():
            noise_pred = unet(latents, t, class_labels=labels, encoder_hidden_states=None).sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            recon_imgs = vae.decode(latents)
            if t == 999 or t%100 == 0:
                plot(recon_imgs, t, epoch)