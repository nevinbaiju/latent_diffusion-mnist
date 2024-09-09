from utils import *
from config import *
from models import *
from data import *

import torch
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from diffusers import DDPMScheduler

from tqdm import tqdm
import numpy as np
import os

def train_vae():
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=lr_vae)
    for epoch in range(vae_epochs):
        losses = []
        for step, (images, _) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            vae_optimizer.zero_grad()
            
            # VAE forward pass
            posterior = vae.encode(images)
            outputs = vae.decode(posterior['latent_dist'].sample())
            # Compute VAE loss
            recon_loss = F.mse_loss(outputs.sample, images, reduction="mean")
            kl_loss = (posterior.latent_dist.kl()/(batch_size*img_size*img_size)).mean()
            vae_loss = recon_loss + 0.5*kl_loss

            losses.append(vae_loss.item())
            vae_loss.backward()
            vae_optimizer.step()
            
        print(f"VAE Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")
        plot_side_by_side(images, outputs.sample, posterior.latent_dist.sample(), epoch+1)
        with torch.no_grad():
            losses = []
            for step, (images, _) in enumerate(tqdm(test_loader)):
                images = images.to(device)
                
                # VAE forward pass
                posterior = vae.encode(images)
                outputs = vae.decode(posterior['latent_dist'].sample())
                # Compute VAE loss
                recon_loss = F.mse_loss(outputs.sample, images, reduction="mean")
                kl_loss = (posterior.latent_dist.kl()/(64*28*28)).mean()
                vae_loss = recon_loss + 0.5*kl_loss
        
                losses.append(vae_loss.item())
        print(f"VAE Epoch {epoch+1}, Test Loss: {np.mean(losses):.4f}")

    torch.save(vae.state_dict(), vae_model_path)

def train_unet():
    vae.load_state_dict(torch.load(vae_model_path))
    noise_scheduler = DDPMScheduler(num_train_timesteps=denoising_timesteps)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr_unet)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=(len(train_loader) * 100),
    )

    accelerator = Accelerator()
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )
    ema_model = EMAModel(model.parameters(), decay=0.9999, use_ema_warmup=True)
    for epoch in range(100):

        losses = []
        for step, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            labels_raw = labels
            latents = vae.encode(images).latent_dist.sample()
            noise = torch.randn(latents.shape).to(device)
            bs = images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict the noise residual
            optimizer.zero_grad()
            noise_pred =  unet(sample=noisy_images, timestep=timesteps, encoder_hidden_states=None, class_labels=labels)
            noise_pred = noise_pred.sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            ema_model.step(model.parameters())
            
            losses.append(loss.item())

        print(f'Epoch: {epoch+1}, Train loss: {np.mean(losses)}')
        generate(vae, unet, noise_scheduler, epoch)

        losses = []
        with torch.no_grad():
            for step, (images, labels) in enumerate(tqdm(test_loader)):
                images = images.to(device)
                labels = labels.to(device)
                latents = vae.encode(images).latent_dist.sample()
                noise = torch.randn(latents.shape).to(device)
                bs = images.shape[0]
        
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
        
                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)
                
                noise_pred =  unet(sample=noisy_images, timestep=timesteps, encoder_hidden_states=None, class_labels=labels)
                noise_pred = noise_pred.sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                losses.append(loss.item())
            
        print(f'Epoch: {epoch+1}, Test loss: {np.mean(losses)}')
    torch.save(unet.state_dict(), unet_model_path)

def train():
    create_path_if_not_exists('models')
    create_path_if_not_exists('plots')
    create_path_if_not_exists(vae_plots_path)
    create_path_if_not_exists(unet_plots_path)

    if not os.path.exists(vae_model_path):
        train_vae()

    if not os.path.exists(unet_model_path):
        train_unet()


if __name__ == "__main__":
    train()