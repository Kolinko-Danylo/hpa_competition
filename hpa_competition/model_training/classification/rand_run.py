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
test_transform = get_transforms(config['test']['transform'])

# testdf = pd.read_csv(os.path.join(config['test']['path'], 'sample_submission.csv'))
#
# # test_dataset = HPADatasetCAMTest(config, testdf, transform=test_transform)
#
# test_loader = DataLoader(test_dataset, batch_size=20, num_workers=config['args']['num_workers'],
#                          shuffle=False,
#                          drop_last=False)

train_df = get_df_cam(path=config['train']['path'])

train_dataset = HPADatasetCAM(config, train_df, transform=test_transform, train=True)

train_loader = DataLoader(train_dataset, batch_size=100, num_workers=config['args']['num_workers'],
                          shuffle=True,
                          drop_last=True)




# inferrer = Inferrer(config, test_loader, testdf, False)

segmentator = cellsegmentator.CellSegmentator(
            config['segm_model']['nuclei_path'],
            config['segm_model']['cell_path'],
            scale_factor=0.25,
            device='cuda',
            padding=True,
            multi_channel_model=True
        )


img_dir = '/common/danylokolinko/hpa/train/'
img_save_dir = '/common/danylokolinko/hpa_mask_semantic/cell'
# print(train_df.ID.values.tolist())
for image_id in tqdm(train_df.ID.values.tolist()):
# for image_id in tqdm(['3f903a34-bbb2-11e8-b2ba-ac1f6b6435d0']):
    # print(image_id)
    fname = os.path.join(img_save_dir, f'{image_id}.npy')
    if os.path.isfile(fname):
        continue
    images = build_image_names([image_id], img_dir)[-1]



    cell_segmentations = segmentator.pred_cells(images)
    np.save(fname, cell_segmentations[0][..., [1, 2]])

# for step, (image, empty_label, image_id, empty_cell, empty_nuclei) in (enumerate(tqdm(train_loader))):
#     images = build_image_names(image_id, img_dir)[-1]
#     #     nuc_segmentations = segmentator.pred_nuclei(images[2])
#     cell_segmentations = segmentator.pred_cells(images)
#     #     st = time.time()
#
#     for step, idx in enumerate(image_id):
#         fname = os.path.join(img_save_dir, idx)
#         if os.path.isfile(fname):
#             continue
#         np.save(fname, cell_segmentations[step][..., [1, 2]])
