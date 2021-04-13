
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
from hpa_competition.model_training.classification.utils import load_RGBY_image, build_image_names
from hpa_competition.PuzzleCAM.tools.ai.torch_utils import one_hot_embedding

import hpacellseg.cellsegmentator as cellsegmentator

from hpacellseg.utils import label_cell, label_nuclei


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



class HPADatasetCAM(Dataset):

    def __init__(self, path, df, transform, img_size=224, yellow=False):
        self.NUM_CL = 19
        self.path = path
        self.list_IDs = df['ID'].values
        self.labels = df['Label'].values
        self.img_size = img_size
        self.transform = transform
        self.yellow=yellow

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        X = load_RGBY_image(root_path=self.path, image_id=ID, train_or_test='train', image_size=None, yellow_channel=self.yellow)
        if self.transform is not None:
            X = self.transform(np.transpose(X, (1, 2, 0)), mask=None).permute(2, 0, 1)
        y = self.labels[index]
        y = y.split('|')
        y = list(map(int, y))
        y = one_hot_embedding(y, self.NUM_CL)
        return X, y.astype(np.float32), ID



class HPADatasetCAMTest(Dataset):

    def __init__(self, path, df, transform, img_size=224, yellow=False, segm_models=None):
        self.NUM_CL = 19
        self.path = path
        self.list_IDs = df['ID'].values
        self.img_size = img_size
        self.transform = transform
        self.yellow=yellow
        # self.segmentator = cellsegmentator.CellSegmentator(
        #     segm_models['nuclei_path'],
        #     segm_models['cell_path'],
        #     scale_factor=0.25,
        #     device='cpu',
        #     padding=True,
        #     multi_channel_model=True
        # )

    def __len__(self):
        return len(self.list_IDs)




    def __getitem__(self, index):
        ID = self.list_IDs[index]

        X = load_RGBY_image(root_path=self.path, image_id=ID, train_or_test='test', image_size=None, yellow_channel=self.yellow)
        if self.transform is not None:
            X = self.transform(np.transpose(X, (1, 2, 0)), mask=None).permute(2, 0, 1)
        return X, ID

        # #TODO: loaded twice
        # ppath = os.path.join(self.path, 'test')
        # images = build_image_names(ID, ppath)[-1]
        # nuc_segmentations = self.segmentator.pred_nuclei(images[2])
        #
        # cell_segmentations = self.segmentator.pred_cells(images)
        # nuclei_mask, cell_mask = label_cell(nuc_segmentations, cell_segmentations)
        #
        # return X, ID, nuclei_mask, cell_mask
