
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
from hpa_competition.model_training.classification.utils import load_RGBY_image, build_image_names
from hpa_competition.PuzzleCAM.tools.ai.torch_utils import one_hot_embedding
from hpa_competition.PuzzleCAM.core.aff_utils import PathIndex, GetAffinityLabelFromIndices
import imageio

import hpacellseg.cellsegmentator as cellsegmentator

from hpacellseg.utils import label_cell, label_nuclei


class HPADataset(torch.utils.data.Dataset):
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

class AffinityBaseDatasetCAM(Dataset):
    def __init__(self, config, df, train):
        self.NUM_CL = config['model']['classes']
        self.train_str = 'train' if train else 'val'
        self.path = config[self.train_str]['path']
        self.list_IDs = df['ID'].values
        self.channels = config['model']['in_channels']


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = load_RGBY_image(self.path, 'train', ID, channels=self.channels)
        return X, ID


class HPADatasetCAM(Dataset):
    def __init__(self, config, df, transform, train):
        self.NUM_CL = config['model']['classes']
        self.train_str = 'train' if train else 'val'
        self.path = config[self.train_str]['path']
        self.list_IDs = df['ID'].values
        self.labels = df['Label'].values
        self.transform = transform
        self.channels = config['model']['in_channels']
        self.load_masks = config[self.train_str]['load_mask']
        self.load_masks_path = config[self.train_str]['mask_path']
        self.cell_input = config[self.train_str]['cell_input']
        self.nuclei_input = config[self.train_str]['nuclei_input']
        self.b8 = config[self.train_str]['b8']
        self.segmentation = config['model']['segmentation']
        self.hpasegm_path = config['hpasegm_predictions']
        self.additional_data_path = False
        print(config[self.train_str])

        self.selfsupervise = config[self.train_str]['transform']['supervision']

        if train:
            self.additional_data_path = config['train']['additional_data_path']
            if self.additional_data_path:
                self.additional_df = pd.read_csv(self.additional_data_path)



    def __len__(self):
        return len(self.list_IDs) + (self.additional_df.shape[0] if self.additional_data_path else 0)

    def load_mask(self, base_path, ID, mask_type='cell' ):
        cell_subdir = f'hpa_{mask_type}_mask'
        cell_dir = os.path.join(base_path, cell_subdir, f'{ID}.npz')
        cell_mask = np.load(cell_dir)['arr_0']

        if 'public' in  base_path:
            cell_mask = cv2.resize(cell_mask, (1024, 1024), cv2.INTER_NEAREST)
        return cell_mask

    def load_mask_pred(self, base_path, ID, mask_type='cell' ):

        cell_subdir = mask_type
        cell_dir = os.path.join(base_path, cell_subdir, f'{ID}.npy')
        cell_mask = np.load(cell_dir)
        if mask_type=='nuc':
            cell_mask = cv2.resize(cell_mask, (1024, 1024))
            cell_mask = cell_mask[..., [1, 2]]

        return cell_mask

    def __getitem__(self, index):
        ss_mask = None
        if self.additional_data_path and index >= len(self.list_IDs):
            return self.getitem_additional(index - len(self.list_IDs))

        ID = self.list_IDs[index]
        # return np.nan, np.nan, ID, (np.nan), np.nan
        X = load_RGBY_image(self.path, 'train', ID, channels=self.channels, b8=self.b8)

        y = list(map(int, self.labels[index].split('|')))
        y = one_hot_embedding(y, self.NUM_CL)
        if self.segmentation:
            ss_mask = self.load_mask_pred(self.hpasegm_path, ID, 'cell')
            ss_mask1 = self.load_mask_pred(self.hpasegm_path, ID, 'nuc')
            ss_mask1 = (cv2.resize(ss_mask1, ss_mask.shape[:-1]))
            ss_mask = np.concatenate((ss_mask, ss_mask1), axis=-1)
        if self.load_masks:
            cell_mask = self.load_mask(self.load_masks_path, ID, 'cell')

        if self.cell_input:
            X = np.concatenate((X, np.expand_dims(cell_mask, axis=0)), axis=0)

        out1 = self.transform_func(X, y, ID, cell_mask, ss_mask)

        return out1



    def getitem_additional(self, index):
        ID = self.additional_df.ID.values[index]
        ppths = '/common/danylokolinko/publichpa'

        X = load_RGBY_image(ppths, 'train', ID, channels=self.channels, b8=self.b8)

        y = list(map(int, self.additional_df.Label.values[index].split('|')))
        y = one_hot_embedding(y, self.NUM_CL)

        ss_mask=None
        if self.segmentation:
            ppth = ppths + '_mask_semantic'
            ss_mask = self.load_mask_pred(ppth, ID, 'cell')
            ss_mask1 = self.load_mask_pred(ppth, ID, 'nuc')
            ss_mask1 = (cv2.resize(ss_mask1, ss_mask.shape[:-1]))
            ss_mask = np.concatenate((ss_mask, ss_mask1), axis=-1)
        if self.load_masks:
            cell_mask = self.load_mask(ppths +'_mask', ID, 'cell')

        if self.cell_input:
            X = np.concatenate((X, np.expand_dims(cell_mask, axis=0)), axis=0)

        out1 = self.transform_func(X, y, ID, cell_mask, ss_mask)

        return out1

    def transform_func(self, X, y, ID, cell_mask, ss_mask=None):
        if self.transform is not None:
            if (self.load_masks and not (self.cell_input)):
                masks = [cell_mask.astype(int)]
                if self.segmentation:
                    masks.append(ss_mask)
                X, cell_masks = self.transform(np.transpose(X, (1, 2, 0)), masks=masks)

            elif self.segmentation:
                X, cell_masks = self.transform(np.transpose(X, (1, 2, 0)), masks=[ss_mask])
                cell_masks = [None, cell_masks[0]]

            else:
                X = self.transform(np.transpose(X, (1, 2, 0)), masks=None)
        if self.selfsupervise:
            X2 = X[1]
            X2 = X2.permute(2, 0, 1)

            X = X[0]

        X = X.permute(2, 0, 1)

        return X, y.astype(np.float32), ID, (np.nan if not self.load_masks else cell_masks[0]), (
            np.nan if not self.segmentation else (cell_masks[1].permute(2, 0, 1) > 100).type(torch.uint8)), (X2 if self.selfsupervise else np.nan)





class HPADatasetCAMTest(Dataset):

    def __init__(self, config, df, transform):
        self.NUM_CL = config['model']['classes']
        self.path = config['test']['path']
        self.list_IDs = df['ID'].values
        self.transform = transform
        self.channels = config['model']['in_channels']
        self.b8 = config['test']['b8']

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = load_RGBY_image(root_path=self.path,  train_or_test='test', image_id=ID, channels=self.channels, image_size=None, b8=self.b8)
        if self.transform is not None:
            X, X_unnorm = self.transform(np.transpose(X, (1, 2, 0)), masks=None)
        return X.permute(2, 0, 1), X_unnorm, ID



class General_Dataset_For_Affinity(AffinityBaseDatasetCAM):
    def __init__(self, root_dir, domain, path_index, label_dir, transform, config, df, train):
        super().__init__(config, df, train)

        # data = read_json('./data/VOC_2012.json')

        # self.class_dic = data['class_dic']
        # self.classes = data['classes']

        self.transform = transform

        self.label_dir = label_dir
        self.path_index = path_index

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(self.path_index.src_indices,
                                                                self.path_index.dst_indices)

    def __getitem__(self, idx):
        X, image_id = super().__getitem__(idx)

        label = imageio.imread(os.path.join(self.label_dir, f'{image_id}.png'))
        # label = Image.fromarray(label)

        if self.transform is not None:
            X, label = self.transform(np.transpose(X, (1, 2, 0)), mask=label.astype(int))


        return np.transpose(X, (2, 0, 1)), self.extract_aff_lab_func(label)
