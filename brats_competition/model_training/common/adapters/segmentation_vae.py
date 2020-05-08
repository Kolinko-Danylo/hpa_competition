import torch
from .segmentation_3d import Segmentation3dModelAdapter
from ..models.models_3d.vae import VAE

__all__ = ['SegmentationVAEAdapter']


class SegmentationVAEAdapter(Segmentation3dModelAdapter):
    def __init__(self, config, log_path):
        super(SegmentationVAEAdapter, self).__init__(config, log_path)
        self.vae = VAE(input_shape=(256, 16, 16, 19), out_channels=4)

        self.__encoded_tensor = None
        self.target_module = self.get_target_layer(self.model, config['model']['target_layer'])
        self.target_module.register_forward_hook(self.__save_encoding)

    @staticmethod
    def get_target_layer(model, target_layer):
        modules_path = target_layer.split('.')
        module = model
        for subpath in modules_path:
            for name, current_module in module.named_children():
                if name == subpath:
                    module = current_module
                    break
            else:
                raise ValueError(f"Module path {target_layer} is not valid for current module.")

        return module

    def __save_encoding(self, module, input_tensor, output_tensor):
        self.__encoded_tensor = output_tensor

    def forward(self, data):
        X = data[0]
        if self.mode == 'train':
            y = self.model(X)
            x_recon, mu, sigma = self.vae(self.__encoded_tensor)
            return y, x_recon, mu, sigma
        elif self.mode == 'val':
            return self.model(X)

    def get_loss(self, y_pred, data):
        X, y_true = data[0].to(self.device), data[1].to(self.device)
        y_pred, x_recon, mu, sigma = y_pred

        dice_loss, dice_weight = self.criterion['mean_dice']
        mse_loss, mse_weight = self.criterion['mse']
        kl_loss, kl_weight = self.criterion['git ad']

        loss = 0
        loss += dice_weight * dice_loss(y_pred, y_true)
        loss += mse_weight * mse_loss(x_recon, X)
        loss += kl_weight * kl_loss((mu, sigma, torch.prod(X.shape[2:])))
