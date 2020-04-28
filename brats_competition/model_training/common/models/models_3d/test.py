from brats_competition.model_training.common.models.models_3d import unet3d, vae
import torch
import time


class TestModel(object):

    def __init__(self, input_shape, target_layer):
        self.device = torch.device('cuda:2')
        vae_input_shape = (256, input_shape[1] // 8, input_shape[2] // 8, input_shape[3] // 8)

        self.model = unet3d.UNet3d(in_channels=input_shape[0], out_channels=3).to(self.device)
        self.vae = vae.VAE(input_shape=vae_input_shape, out_channels=input_shape[0]).to(self.device)

        self.encoded_tensor__ = None
        self.target_module = self.get_target_layer(self.model, target_layer)
        self.target_module.register_forward_hook(self.save_encoding__)

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

    def save_encoding__(self, module, input_tensor, output_tensor):
        self.encoded_tensor__ = output_tensor

    def forward(self, x):
        y = self.model(x)
        x_recon, mu, sigma = self.vae(self.encoded_tensor__)
        return y, x_recon, mu, sigma


if __name__ == '__main__':
    input_shape = (4, 128, 128, 128)
    tester = TestModel(input_shape=input_shape, target_layer='block13')
    times = []
    for x in range(100):
        x = torch.rand(1, *input_shape, device='cuda:2')
        start = time.time()
        list(map(lambda z: print(z.shape), tester.forward(x)))
        times.append(time.time() - start)
    print(sum(times) / len(times))
