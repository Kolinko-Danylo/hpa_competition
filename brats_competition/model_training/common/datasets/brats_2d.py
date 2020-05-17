import os
import numpy as np
import torch

__all__ = ['Brats2dDataset']


class Brats2dDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, return_original=False):
        self.transform = transform
        self.return_original = return_original
        self.x_path = os.path.join(path, "X")
        self.y_path = os.path.join(path, "y")

        self.length = len(os.listdir(self.x_path))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x_orig = np.load(os.path.join(self.x_path, f"{i}.npy"))
        y = np.load(os.path.join(self.y_path, f"{i}.npy"))
        y[y == 4] = 3
        if self.transform is None:
            rv = (torch.from_numpy(x_orig.astype(np.float32)).permute(2, 0, 1), torch.from_numpy(y).long())
        else:
            x, y = self.transform(x_orig, y)
            rv = (x.permute(2, 0, 1), y.squeeze())

        if self.return_original:
            rv = rv + (x_orig.astype(np.int32), )

        return rv

