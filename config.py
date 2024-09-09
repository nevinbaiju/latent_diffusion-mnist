import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
img_size = 32

latent_channels = 4

lr_vae = 1e-4
vae_epochs = 50

lr_unet = 1e-4
denoising_timesteps = 1000
num_warmup_steps = 500
unet_epochs = 100

vae_model_path = 'models/vae.pth'
unet_model_path = 'models/unet.pth'

vae_plots_path = 'plots/vae/'
unet_plots_path = 'plots/unet/'