from torch import nn
import segmentation_models_pytorch as smp


class DeepLabV3(nn.Module):
    def __init__(self, encoder, num_classes):
        super(DeepLabV3, self).__init__()
        self.model = smp.DeepLabV3(encoder, classes=num_classes if num_classes != 2 else 1, in_channels=4)

    def forward(self, x):
        return self.model(x)

    def get_params_groups(self):
        return (
            list(self.model.encoder.parameters()),
            list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())
        )
