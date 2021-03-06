import os
import numpy as np
import torch
from hpa_competition.preprocessing.preprocess import read_instance

__all__ = ['Brats3dDataset']


class Brats3dDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.folder_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
        self.folder_paths.sort()

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, i):
        x, y = read_instance(self.folder_paths[i])
        y[y == 4] = 3

        if self.transform is None:
            return torch.from_numpy(x.astype(np.float32))[..., 2:-1], torch.from_numpy(y).long()[..., 2:-1]
        else:
            x, y = self.transform(x.transpose(1, 2, 3, 0), y)
            return x.permute(3, 0, 1, 2)[..., 5:-6], y.squeeze()[..., 5:-6]
