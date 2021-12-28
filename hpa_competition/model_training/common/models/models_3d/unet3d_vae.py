import torch.nn as nn
from hpa_competition.model_training.common.models.models_3d.unet3d import UNet3d
from hpa_competition.model_training.common.models.models_3d.vae import VAE


class UNet3dVae(nn.Module):
    def __init__(self, input_shape, in_channels, out_channels):
        super(UNet3dVae, self).__init__()
        self.unet = UNet3d(in_channels, out_channels)
        self.vae = VAE(input_shape, out_channels)

    def forward(self, x):
        if self.training:
            emb, y_pred = self.unet(x)
            x_recovered, mu, sigma = self.vae(emb)
            return y_pred, x_recovered, mu, sigma
        else:
            y_pred = self.unet(x)
            return y_pred
