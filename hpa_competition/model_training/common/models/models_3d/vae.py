import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from hpa_competition.model_training.common.models.models_3d.nnutils import ResBlock3d


class VAE(nn.Module):
    def __init__(self, input_shape, out_channels):
        super(VAE, self).__init__()
        in_channels, x_, y_, z_ = input_shape
        self.embed1 = nn.Sequential(OrderedDict([
            ('bn', nn.GroupNorm(num_groups=in_channels // 8, num_channels=in_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv3d(in_channels, 16, (3, 3, 3), padding=1, stride=2))
        ]))
        self.dense2 = nn.Linear(16 * x_ // 2 * y_ // 2 * z_ // 2, in_channels)

        self.dense3 = nn.Linear(in_channels // 2, in_channels * x_ // 2 * y_ // 2 * z_ // 2)

        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(in_channels, in_channels, (1, 1, 1))

        self.conv5 = nn.Conv3d(in_channels, in_channels // 2, (1, 1, 1))

        self.block6 = ResBlock3d(in_channels // 2, channels_per_group=8)
        self.conv7 = nn.Conv3d(in_channels // 2, in_channels // 4, (1, 1, 1))

        self.block8 = ResBlock3d(in_channels // 4, channels_per_group=8)
        self.conv9 = nn.Conv3d(in_channels // 4, in_channels // 8, (1, 1, 1))

        self.block10 = ResBlock3d(in_channels // 8, channels_per_group=8)
        self.conv11 = nn.Conv3d(in_channels // 8, out_channels, (1, 1, 1))

    def forward(self, x):
        batch_size, in_channels, x_, y_, z_ = x.size()
        encoded = self.embed1(x)
        encoded = encoded.view(batch_size, -1)
        encoded = self.dense2(encoded)
        latent_dim = encoded.size(1) // 2

        sample = torch.randn(batch_size, latent_dim, device=x.device)
        mu, sigma = encoded[:, :latent_dim], encoded[:, latent_dim:]
        sample = sample * sigma + mu

        out = self.dense3(sample)
        out = out.view(batch_size, in_channels, x_ // 2, y_ // 2, z_ // 2)

        out = self.relu4(out)
        out = self.conv4(out)
        out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)

        out = self.conv5(out)
        out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)

        out = self.block6(out)
        out = self.conv7(out)
        out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)

        out = self.block8(out)
        out = self.conv9(out)
        out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)

        out = self.block10(out)
        out = self.conv11(out)

        return out, mu, sigma


if __name__ == '__main__':
    import torchsummary

    model = VAE((256, 16, 16, 19), out_channels=4)
    torchsummary.summary(model, input_size=(256, 16, 16, 19), device='cpu')
