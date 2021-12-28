import numpy as np
import torch
from torchvision.utils import make_grid

from .base import ModelAdapter

__all__ = ['SegmentationModelAdapter']


class SegmentationModelAdapter(ModelAdapter):

    def __init__(self, config, log_path):
        super(SegmentationModelAdapter, self).__init__(config, log_path)
        self.num_classes = config['model']['classes']

        self.class_colors = np.random.randint(0, 255, (self.num_classes, 3), dtype=np.uint8)
        self.class_colors[0] = (0, 0, 0)

    def make_tensorboard_grid(self, batch_sample):
        data, y_pred = batch_sample['data'], batch_sample['y_pred']
        y = data[1]
        _, y_pred = y_pred.max(1)
        return make_grid(torch.cat([
            self.decode_segmap(y, self.num_classes),
            self.decode_segmap(y_pred.to(y.device), self.num_classes)
        ]), nrow=y.shape[0])

    def decode_segmap(self, image, nc=201):
        out = torch.empty(image.shape[0], 3, *image.shape[1:], dtype=torch.float32, device=image.device)
        for l in range(0, nc):
            idx = image == l
            out[:, 0].masked_fill_(idx, self.class_colors[l][0])
            out[:, 1].masked_fill_(idx, self.class_colors[l][1])
            out[:, 2].masked_fill_(idx, self.class_colors[l][2])

        return out / 255.0
