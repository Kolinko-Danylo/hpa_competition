import torch.nn as nn
from collections import OrderedDict


class ResBlock3d(nn.Module):

    def __init__(self, channels, channels_per_group):
        super(ResBlock3d, self).__init__()

        self.block = nn.Sequential(OrderedDict([
            ('gn1', nn.GroupNorm(num_groups=channels // channels_per_group, num_channels=channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv3d(channels, channels, (3, 3, 3), padding=1)),
            ('gn2', nn.GroupNorm(num_groups=channels // channels_per_group, num_channels=channels)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(channels, channels, (3, 3, 3), padding=1))
        ]))

    def forward(self, x):
        out = self.block(x)
        return out + x


if __name__ == '__main__':
    from torchsummary import summary

    block = ResBlock3d(channels=32, channels_per_group=8)
    summary(block, input_size=(32, 128, 128, 128), batch_size=1, device='cpu')
