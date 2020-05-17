import torch.nn as nn
import torch.nn.functional as F
from .nnutils import ResBlock3d


class UNet3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3d, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 32, (3, 3, 3), padding=1)
        self.block2 = ResBlock3d(32, channels_per_group=8)
        self.downsample3 = nn.Conv3d(32, 64, (3, 3, 3), stride=2, padding=1)

        self.block4 = ResBlock3d(64, channels_per_group=8)
        self.block5 = ResBlock3d(64, channels_per_group=8)
        self.downsample6 = nn.Conv3d(64, 128, (3, 3, 3), stride=2, padding=1)

        self.block7 = ResBlock3d(128, channels_per_group=8)
        self.block8 = ResBlock3d(128, channels_per_group=8)
        self.downsample9 = nn.Conv3d(128, 256, (3, 3, 3), stride=2, padding=1)

        self.block10 = ResBlock3d(256, channels_per_group=8)
        self.block11 = ResBlock3d(256, channels_per_group=8)
        self.block12 = ResBlock3d(256, channels_per_group=8)
        self.block13 = ResBlock3d(256, channels_per_group=8)

        self.conv14 = nn.Conv3d(256, 128, (1, 1, 1))
        self.block15 = ResBlock3d(128, channels_per_group=8)

        self.conv16 = nn.Conv3d(128, 64, (1, 1, 1))
        self.block17 = ResBlock3d(64, channels_per_group=8)

        self.conv18 = nn.Conv3d(64, 32, (1, 1, 1))
        self.block19 = ResBlock3d(32, channels_per_group=8)

        self.conv20 = nn.Conv3d(32, out_channels, (1, 1, 1))

    def forward(self, x):
        # encoder
        enc32 = self.conv1(x)
        enc32 = self.block2(enc32)
        enc64 = self.downsample3(enc32)

        enc64 = self.block4(enc64)
        enc64 = self.block5(enc64)
        enc128 = self.downsample6(enc64)

        enc128 = self.block7(enc128)
        enc128 = self.block8(enc128)
        enc256 = self.downsample9(enc128)

        enc256 = self.block10(enc256)
        enc256 = self.block11(enc256)
        enc256 = self.block12(enc256)
        enc256 = self.block13(enc256)

        # decoder
        out = self.conv14(enc256)
        out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
        out += enc128
        out = self.block15(out)

        out = self.conv16(out)
        out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
        out += enc64
        out = self.block17(out)

        out = self.conv18(out)
        out = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
        out += enc32
        out = self.block19(out)

        out = self.conv20(out)

        if self.training:
            return enc256, out
        else:
            return out
