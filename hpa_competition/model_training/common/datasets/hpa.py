
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image

class HPADataset(torch.utils.data.Dataset):
    def __init__(self, path, df, transform=None, return_original=False):
        self.path = path
        self.transform = transform
        self.return_original = return_original
        self.df = df

    #         self.x_path = os.path.join(path, "X")
    #         self.y_path = os.path.join(path, "y")

    #         self.length = len(os.listdir(self.x_path))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        item = self.df.loc[i]
        x_orig_path = self.get_x(item)
        x = np.array(Image.open(x_orig_path))
        y = self.get_y(item)

        if self.transform is not None:
            x = self.transform(x, mask=None).permute(2, 0, 1)
        return x, y

    def get_x(self, r):
        return os.path.join(self.path, 'cells/', (r['image_id'] + '_' + str(r['cell_id']) + '.jpg'))

    def get_y(self, r):
        NUM_CLASS = 19
        lbl_list = r['image_labels'].split('|')
        base = torch.zeros(NUM_CLASS, dtype=torch.float)
        for lbl in lbl_list:
            base[int(lbl)] = 1
        return base


class HPADatasetTest(torch.utils.data.Dataset):
    def __init__(self, path, df, transform=None, return_original=False):
        self.path = path
        self.transform = transform
        self.return_original = return_original
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        item = self.df.loc[i]
        x_orig_path = self.get_x(item)
        x = np.array(Image.open(x_orig_path))
        # y = self.get_y(item)

        if self.transform is not None:
            x = self.transform(x, mask=None).permute(2, 0, 1)
        return x

    def get_x(self, r):
        return os.path.join(self.path, 'cells/', (r['image_id'] + '_' + str(r['cell_id']) + '.jpg'))
