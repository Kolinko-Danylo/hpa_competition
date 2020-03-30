import os
import numpy as np
import nibabel as nib
import torch
from brats_competition.preprocessing.preprocess import read_instance


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
            return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y).long()
        else:
            return self.transform(x.permute(1, 2, 3, 0), y)
