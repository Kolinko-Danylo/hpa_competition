from .segmentation import *


def get_model_adapter(config, log_path):
    if config['task'] == 'segmentation':
        return SegmentationModelAdapter(config, log_path)
    else:
        raise ValueError(f'Unrecognized task [{config["task"]}]')