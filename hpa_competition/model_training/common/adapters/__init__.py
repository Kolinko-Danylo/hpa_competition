from .segmentation import *
from .segmentation_3d import *
from .segmentation_vae import *
from .base import *

def get_model_adapter(config, log_path):
    return ModelAdapter(config, log_path)
    if config['task'] == 'classification':
        return SegmentationModelAdapter(config, log_path)
    if config['task'] == 'segmentation_3d':
        return Segmentation3dModelAdapter(config, log_path)
    if config['task'] == 'segmentation_vae':
        return SegmentationVAEAdapter(config, log_path)
    else:
        raise ValueError(f'Unrecognized task [{config["task"]}]')
