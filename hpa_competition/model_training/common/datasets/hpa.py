
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image

class HPADataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, return_original=False):
        self.path = path
        self.transform = transform
        self.return_original = return_original
        df = pd.read_csv(os.path.join(path, 'cell_df.csv'))
        df = df.loc[df.size1 >= 224]
        df = df.loc[df.size2 >= 224]

        labels = [str(i) for i in range(19)]
        for x in labels:
            df[x] = df['image_labels'].apply(lambda r: int(x in r.split('|')))
        dfs = df.sample(frac=0.1, random_state=42)
        dfs = dfs.reset_index(drop=True)
        # unique_counts = {lbl: len(dfs[dfs.image_labels == lbl]) for lbl in labels}
        #
        # full_counts = {lbl: dfs[lbl].sum() for lbl in labels}
        #
        # counts = list(zip(full_counts.keys(), full_counts.values(), unique_counts.values()))
        # counts = np.array(sorted(counts, key=lambda x: -x[1]))
        # counts = pd.DataFrame(counts, columns=['label', 'full_count', 'unique_count'])
        # counts.set_index('label').T
        self.df = dfs

    #         self.x_path = os.path.join(path, "X")
    #         self.y_path = os.path.join(path, "y")

    #         self.length = len(os.listdir(self.x_path))

    def __len__(self):
        return self.dfs.shape[0]

    def __getitem__(self, i):
        item = self.df.loc[i]
        x_orig_path = self.get_x(item)
        x = np.array(Image.open(x_orig_path))
        y = self.get_y(item)

        if self.transform is not None:
            x = self.transform(x, mask=None)
        return x, y

    def get_x(self, r):
        return os.path.join(self.path, 'cells/', (r['image_id'] + '_' + str(r['cell_id']) + '.jpg'))

    def get_y(self, r):
        return r['image_labels'].split('|')
