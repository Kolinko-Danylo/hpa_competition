import os
import pandas as pd
import numpy as np
import yaml
import sys
sys.path.append('/home/danylokolinko/hpa')
from hpa_competition.model_training.common.augmentations import get_transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from hpa_competition.model_training.common.datasets import HPADatasetCAMTest, HPADatasetCAM

from hpa_competition.model_training.classification.utils import  get_df_cam, build_image_names, read_img, get_cam, load_RGBY_image, print_masked_img
from torch.utils.data import Dataset, DataLoader
from inference_cam import Inferrer
import cv2
import matplotlib.pyplot as plt
import hpacellseg.cellsegmentator as cellsegmentator
from tqdm import tqdm

with open(os.path.join('config', 'cam_avenga.yaml')) as config_file:
    config = yaml.full_load(config_file)

# model = Classifier(config['model']['arch'], config['model']['pretreined'],
#                num_classes=config['model']['classes'], mode=config['args']['mode'])
# trainer = CAMTrainer(config, None, None)

# train_transform = get_transforms(config['train']['transform'])
# test_transform = get_transforms(config['test']['transform'])
#
# testdf = pd.read_csv(os.path.join(config['test']['path'], 'sample_submission.csv'))
#
# test_dataset = HPADatasetCAMTest(config, testdf, transform=test_transform)
#
# test_loader = DataLoader(test_dataset, batch_size=20, num_workers=config['args']['num_workers'],
#                          shuffle=False,
#                          drop_last=False)
#
# train_df = get_df_cam(path=config['train']['path'])
#
# train_dataset = HPADatasetCAM(config, train_df, transform=test_transform, train=True)
#
# train_loader = DataLoader(train_dataset, batch_size=100, num_workers=config['args']['num_workers'],
#                           shuffle=True,
#                           drop_last=True)
#
#
#
#
# inferrer = Inferrer(config, test_loader, testdf, False)
import hpacellseg.cellsegmentator as cellsegmentator

# from hpacellseg.utils import label_cell, label_nuclei
import glob
save_dir = '/common/danylokolinko/publichpa/HPA-Challenge-2021-trainset-extra/'

mt = glob.glob(save_dir  + '*_red.png')
er = [f.replace('red', 'yellow') for f in mt]
nu = [f.replace('red', 'blue') for f in mt]
names = [f.replace('_red.png', '').split('/')[-1] for f in mt]


segm_config = config['segm_model']
NUC_MODEL = segm_config['nuclei_path']
CELL_MODEL = segm_config['cell_path']

segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.5,
    device="cuda",
    padding=False,
    multi_channel_model=True,
)

# For nuclei
# nuc_segmentations = segmentator.pred_nuclei(images[2])

otherdir = '/common/danylokolinko/publichpa_mask_semantic/cell'
for i in tqdm(range(len(names))):
    fname = os.path.join(otherdir, names[i])
    if os.path.isfile(fname):
        continue
    images = [[mt[i]], [er[i]], [nu[i]]]

    if not (os.path.isfile(images[0][0]) and os.path.isfile(images[1][0]) and os.path.isfile(images[2][0])):
        continue

    cell_segmentations = segmentator.pred_cells(images)
#     print(cell_segmentations[0].shape)
#     for step, idx in enumerate(image_id):
#     plt.imshow(cell_segmentations[0])

    np.save(fname, cell_segmentations[0][..., [1, 2]])
    # break
