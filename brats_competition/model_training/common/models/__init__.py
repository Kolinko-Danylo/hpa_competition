from .unet import UNetResNet


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
        return UNetResNet(**model_config)
    else:
        raise ValueError(f'Model architecture [{arch}] not recognized.')
