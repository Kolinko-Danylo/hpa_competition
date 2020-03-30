import os
import numpy as np
import torch

__all__ = ['Brats2dDataset']


class Brats2dDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.x_path = os.path.join(path, "X")
        self.y_path = os.path.join(path, "y")

        self.length = len(os.listdir(self.x_path))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x = np.load(os.path.join(self.x_path, f"{i}.npy"))
        y = np.load(os.path.join(self.y_path, f"{i}.npy"))
        y[y == 4] = 3

        if self.transform is None:
            return torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1), torch.from_numpy(y).long()
        else:
            x, y = self.transform(x, y)
            return x.permute(2, 0, 1), y.squeeze()
