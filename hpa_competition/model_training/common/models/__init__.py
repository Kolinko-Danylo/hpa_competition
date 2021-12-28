from .unet import UNetResNet
from .deeplab import DeepLabV3
from .models_3d.unet3d import UNet3d
from .models_3d.unet3d_vae import UNet3dVae
import torchvision.models as models
import pretrainedmodels
from torch import nn
def get_network(model_config):
    """
    Create model form configuration
    Args:
        model_config (dict): dictionary of model config
    Return:
        model (torch.nn.Module): model created from config
    """
    # arch = model_config['arch']
    # del model_config['arch']
    model_name = 'resnet50'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    # model = models.resnet50(pretrained=True)
    # output = model.features(input_224)

    nb_classes = 19
    class LinearSigmoid(nn.Module):

        def __init__(self, feat, classes):
            super(LinearSigmoid, self).__init__()
            self.fc = nn.Linear(feat, classes)
            self.sigm = nn.Sigmoid()
        def forward(self, x):
            return self.sigm(self.fc(x))

    model.last_linear = LinearSigmoid(model.last_linear.in_features, nb_classes)
    return model




    # features extraction
    # model.last_linear = pretrained.utils.Identity()
    # output = model(input_224)
    # print(output.size())  # (1,2048)

    # if arch == 'unet_resnet':
    #     return UNetResNet(model_config['encoder_depth'], model_config['classes'])
    # elif arch == "deeplab_v3":
    #     return DeepLabV3(model_config["encoder"], model_config["classes"])
    # elif arch == "unet_3d":
    #     return UNet3d(model_config["input_channels"], model_config["classes"])
    # elif arch == "unet_3d_vae":
    #     return UNet3dVae((256, 14, 14, 18), model_config["input_channels"], model_config["classes"])
    # else:
    #     raise ValueError(f'Model architecture [{arch}] not recognized.')
