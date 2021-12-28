import torch
from .segmentation_3d import Segmentation3dModelAdapter
from ..models.models_3d.vae import VAE
from .. losses import get_loss

__all__ = ['SegmentationVAEAdapter']


class SegmentationVAEAdapter(Segmentation3dModelAdapter):
    def __init__(self, config, log_path):
        super(SegmentationVAEAdapter, self).__init__(config, log_path)

        self._initialize_vae_losses(config)
        self.mse_weight = 1
        self.kl_weight = 1

    def _initialize_vae_losses(self, config):
        self.mse_weight = config["model"]["vae_loss"]["mse_weight"]
        self.kl_weight = config["model"]["vae_loss"]["parametric_kl_weight"]

        self.mse_loss = get_loss({"name": "mse"})
        self.kl_loss = get_loss({"name": "parametric_kl"})


    def get_loss(self, y_pred, data):
        if self.mode == 'train':
            X = data[0].to(self.device)
            y_pred, x_recon, mu, sigma = y_pred

            loss = super().get_loss(y_pred, data)
            loss += self.mse_weight * self.mse_loss(x_recon, X)
            loss += self.kl_weight * self.kl_loss((mu, sigma, torch.prod(torch.tensor(X.shape[2:]))))

            return loss
        elif self.mode == 'val':
            return super().get_loss(y_pred, data)
