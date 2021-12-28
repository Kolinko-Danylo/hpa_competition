# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import pandas as pd
from hpa_competition.model_training.common.datasets import HPADatasetCAM
from hpa_competition.model_training.common.augmentations import get_transforms

from tqdm import tqdm
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
import yaml
from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', default='resnet50@seed=0@bs=16@ep=5@nesterov@train@scale=0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--fg_threshold', default=0.30, type=float)
parser.add_argument('--bg_threshold', default=0.05, type=float)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    with open(os.path.join('/home/danylokolinko/hpa/hpa_competition/model_training/classification', 'config',
                           'cam_inf_puzzle.yaml')) as config_file:
        config = yaml.full_load(config_file)
    ###################################################################################
    # Arguments
    ##################################################################################

    experiment_name = config['model']['model_path'].split('/')[-1].split('.')[0]
    args = config['args']
    if 'train' in args['domain']:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@scale=%s' % args['scales']

    pred_dir = f'/home/danylokolinko/logs/inference/{experiment_name}/'

    model_path = config['model']['model_path']

    aff_dir = create_directory(
        '/home/danylokolinko/logs/inference/{}@aff_fg={:.2f}_bg={:.2f}/'.format(experiment_name, args['fg_threshold'],
                                                                                args['bg_threshold']))

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # for mIoU

    #################################################################################################
    # Convert
    #################################################################################################
    tag_str = config['model']['model_path'].split('/')[-1].split('.')[0]
    print(tag_str)
    print(f'loading csv from {tag_str}')

    df = pd.read_csv(os.path.join(config['log_path'], 'csv', tag_str))
    print(df.head())

    train_df = df.loc[~df.is_valid].reset_index(drop=True)
    length = train_df.shape[0]
    ids = np.arange(length)
    from copy import deepcopy
    def init_func(config, train_df):
        global aff_dir, pred_dir, model_path, dataset

        config = deepcopy(config)
        train_transform = get_transforms(config['train']['transform'])
        dataset = HPADatasetCAM(config, deepcopy(train_df), transform=train_transform, train=True)
        args = parser.parse_args()

        experiment_name = config['model']['model_path'].split('/')[-1].split('.')[0]
        args = config['args']
        if 'train' in args['domain']:
            experiment_name += '@train'
        else:
            experiment_name += '@val'

        experiment_name += '@scale=%s' % args['scales']

        pred_dir = f'/home/danylokolinko/logs/inference/{experiment_name}/'

        model_path = config['model']['model_path']

        aff_dir = '/home/danylokolinko/logs/inference/{}@aff_fg={:.2f}_bg={:.2f}/'.format(experiment_name,
                                                                                    args['fg_threshold'],
                                                                                    args['bg_threshold'])


    dataset = None
    def make_aff1(arg):
        make_aff(*arg)

    def make_aff(indx):
        (ori_image, _, image_id, _, _) = dataset[indx]
        ori_image = ori_image.permute(1, 2, 0)
        png_path = aff_dir + image_id + '.png'
        # if os.path.isfile(png_path):
        #     continue

        # load
        image = np.asarray(ori_image).astype((np.uint8))[..., :3]
        cam_dict = np.load(pred_dir + image_id + '.npy', allow_pickle=True).item()
        ori_h, ori_w, c = image.shape
        
        keys = cam_dict['keys']
        keys = np.pad(keys + 1, (1, 0), mode='constant')

        cams = cam_dict['hr_cam']

        # 1. find confident fg & bg
        fg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args['fg_threshold'])

        fg_cam = np.argmax(fg_cam, axis=0)
        # fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0])]




        
        bg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args['bg_threshold'])
        bg_cam = np.argmax(bg_cam, axis=0)
        # bg_conf = bg_cam
        # bg_conf = keys[crf_inference_label(image, bg_cam, n_labels=keys.shape[0])]
        fg_conf = keys[fg_cam]
        bg_conf = keys[bg_cam]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0
        
        imageio.imwrite(png_path, conf.astype(np.uint8))
        return True
        # sys.stdout.write('\r# Convert [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), conf.shape))
        # sys.stdout.flush()

    # with tqdm(total=len(ids)) as pbar:
    #     with ThreadPoolExecutor(max_workers=10) as ex:
    #         futures = [ex.submit(make_aff, id) for id in ids]
    #         for future in as_completed(futures):
    #             result = future.result()
    #             pbar.update(1)
    with ProcessPoolExecutor(20, initializer=init_func, initargs=(config, train_df)) as executor:
        # print('here')
        a = list(tqdm(executor.map(make_aff, ids), total=len(ids)))
