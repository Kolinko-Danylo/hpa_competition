import torch



def get_network(model_config):
    """
    Create model form configuration
    Args:
        model_config (dict): dictionary of model config
    Return:
        model (torch.nn.Module): model created from config
    """
    arch = model_config["arch"]

    pass

    # if arch == "deeplabv3plus_resnet50":
    #     return DeepLabV3Plus(num_classes=model_config["classes"])
