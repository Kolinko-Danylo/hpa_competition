from .unet import UNetResNet
from .deeplab import DeepLabV3
from .models_3d.unet3d import UNet3d
from .models_3d.unet3d_vae import UNet3dVae


def get_network(model_config):
    """
    Create model form configuration
    Args:
        model_config (dict): dictionary of model config
    Return:
        model (torch.nn.Module): model created from config
    """
    arch = model_config['arch']
    del model_config['arch']

    if arch == 'unet_resnet':
        return UNetResNet(model_config['encoder_depth'], model_config['classes'])
    elif arch == "deeplab_v3":
        return DeepLabV3(model_config["encoder"], model_config["classes"])
    elif arch == "unet_3d":
        return UNet3d(model_config["input_channels"], model_config["classes"])
    elif arch == "unet_3d_vae":
        return UNet3dVae((256, 14, 14, 18), model_config["input_channels"], model_config["classes"])
    else:
        raise ValueError(f'Model architecture [{arch}] not recognized.')
