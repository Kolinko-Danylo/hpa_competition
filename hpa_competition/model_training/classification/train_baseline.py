import yaml
import torch
import numpy as np
import os
import random

from hpa_competition.model_training.common.trainer import Trainer
from hpa_competition.model_training.common.datasets import HPADataset
from hpa_competition.model_training.common.augmentations import get_transforms
from hpa_competition.PuzzleCAM import train_classification_with_puzzle
from utils import get_df
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
with open(os.path.join(os.path.dirname(__file__), 'config', 'test.yaml')) as config_file:
    config = yaml.full_load(config_file)

train_transform = get_transforms(config['train']['transform'])
val_transform = get_transforms(config['val']['transform'])

train_df, val_df = get_df(path=config['train']['path'])
path = config['train']['path']
train_ds = HPADataset(path, train_df, transform=train_transform)
val_ds = HPADataset(path, val_df, transform=val_transform)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12)

trainer = Trainer(config, train_dl, val_dl)
trainer.train()



from itertools import groupby
from pycocotools import mask as mutils
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import pickle
import cv2
from multiprocessing import Pool
# import matplotlib.pyplot as plt



# exp_name = "v3"
# conf_name = "mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco"
# cell_mask_dir = '/datasets/kolinko/hpa-mask/hpa_cell_mask'
# ROOT = '/datasets/kolinko/hpa/'
# train_or_test = 'train'
# img_dir = f'result/baseline_{exp_name}_{train_or_test}'
# df = pd.read_csv(os.path.join(ROOT, 'train.csv'))
#
# # this script takes more than 9hours for full data.
# debug = True
# if debug:
#     df = df[:4]
#
#
#
# cell_mask_dir = '/datasets/kolinko/hpa-mask/hpa_cell_mask'
# for idx in range(3):
#     image_id = df.iloc[idx].ID
#     cell_mask = np.load(f'{cell_mask_dir}/{image_id}.npz')['arr_0']
#     # print_masked_img(image_id, cell_mask)


import numpy as np
# from fastai.vision.all import PILImage, RandomResizedCrop, Normalize
import pickle
import os
import torch



# X = torch.rand(1, 3, 224, 224)
# from hpa_competition.model_training.common.models import get_network
# model = get_network(None)
# print(model(X).shape)

# cell_mask = np.load(f'{cell_dir}/{image_id}.npz')['arr_0']
# nucl_mask = np.load(f'{nucl_dir}/{image_id}.npz')['arr_0']
#
#
#
# cell_mask_dir = '../input/hpa-mask/hpa_cell_mask'
# def get_masks(cell_mask_path, df):
#     for idx in df.index:
#         image_id = df.iloc[idx].ID
#         cell_mask = np.load(f'{cell_mask_path}/{image_id}.npz')['arr_0']
#
#         # print_masked_img(image_id, cell_mask)
#
# MAX_THRE = 4 # set your avarable CPU count.
# p = Pool(processes=MAX_THRE)
# annos = []
# len_df = len(df)
# for anno, idx, image_id in p.imap(mk_ann, range(len(df))):
#     if len(anno['ann']) > 0:
#         annos.append(anno)
#     print(f'{idx+1}/{len_df}, {image_id}')
