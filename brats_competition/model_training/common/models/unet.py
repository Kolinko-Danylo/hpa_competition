"""The network definition that was used for a second place solution at the DeepGlobe Building Detection challenge."""

import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torch.nn import functional as F
from collections import OrderedDict
from torchsummary import summary

from copy import deepcopy


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_, out):
        super(Conv2dReLU, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_, out, 3, padding=1)),
            ('relu', nn.ReLU(inplace=True))
        ]))


class UNetResNet(nn.Module):

    def __init__(self, encoder_depth, num_classes=2, num_filters=32, dropout_rate=0.2,
                 pretrained=True, is_deconv=True):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if encoder_depth == 34:
            encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048

        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        # make first layer accept 4 channels (copy first 3 from ResNet pretrained on ImageNet)
        # https://discuss.pytorch.org/t/how-to-transfer-the-pretrained-weights-for-a-standard-resnet50-to-a-4-channel/52252
        first_conv = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            first_conv.weight[:, :3] = encoder.conv1.weight

        self.conv1 = nn.Sequential(first_conv,
                                   deepcopy(encoder.bn1),
                                   deepcopy(encoder.relu),
                                   nn.MaxPool2d(2, 2))

        self.conv2 = deepcopy(encoder.layer1)
        self.conv3 = deepcopy(encoder.layer2)
        self.conv4 = deepcopy(encoder.layer3)
        self.conv5 = deepcopy(encoder.layer4)

        self.center = DecoderCenter(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv=False)

        self.dec5 = DecoderBlock(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 2, is_deconv)
        self.dec4 = DecoderBlock(bottom_channel_nr // 2 + num_filters * 2, num_filters * 8, num_filters * 2, is_deconv)
        self.dec3 = DecoderBlock(bottom_channel_nr // 4 + num_filters * 2, num_filters * 4, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2, num_filters * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2, num_filters, num_filters * 2, is_deconv)
        self.dec0 = Conv2dReLU(num_filters * 10, num_filters * 2)
        self.conv2_drop = nn.Dropout2d(p=self.dropout_rate)
        self.final = nn.Conv2d(num_filters * 2, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center = self.center(conv5)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        f = torch.cat((
            dec1,
            F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(dec3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(dec4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(dec5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)

        dec0 = self.dec0(self.conv2_drop(f))
        return self.final(dec0)


class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        if is_deconv:
            super(DecoderBlock, self).__init__(OrderedDict([
                ('conv_relu', Conv2dReLU(in_channels, middle_channels)),
                ('conv_transpose',
                 nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU(inplace=True))
            ]))
        else:
            super(DecoderBlock, self).__init__(OrderedDict([
                ('upsample', nn.Upsample(scale_factor=2, mode='bilinear')),
                ('conv_relu1', Conv2dReLU(in_channels, middle_channels)),
                ('conv_relu2', Conv2dReLU(middle_channels, out_channels)),
            ]))
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                Conv2dReLU(in_channels, middle_channels),
                Conv2dReLU(middle_channels, out_channels),
            )


class DecoderCenter(nn.Sequential):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """
            super(DecoderCenter, self).__init__(OrderedDict([
                ('conv_relu', Conv2dReLU(in_channels, middle_channels)),
                ('conv_transpose',
                 nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU(inplace=True))
            ]))
        else:
            super(DecoderCenter, self).__init__(OrderedDict([
                ('conv_relu1', Conv2dReLU(in_channels, middle_channels)),
                ('conv_relu2', Conv2dReLU(middle_channels, out_channels)),
            ]))


if __name__ == '__main__':
    model = UNetResNet(encoder_depth=34, num_classes=4).cuda()
    print(summary(model, input_size=(4, 224, 224)))
