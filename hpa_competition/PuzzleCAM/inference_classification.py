# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import pandas as pd
from hpa_competition.model_training.common.datasets import HPADatasetCAM
from hpa_competition.model_training.common.augmentations import get_transforms
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
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

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
import yaml
if __name__ == '__main__':
    with open(os.path.join('/home/danylokolinko/hpa/hpa_competition/model_training/classification', 'config',
                           'cam_inf_puzzle.yaml')) as config_file:
        config = yaml.full_load(config_file)
    ###################################################################################
    # Arguments
    ###################################################################################
    # args = parser.parse_args()

    experiment_name = config['model']['model_path'].split('/')[-1].split('.')[0]
    args = config['args']
    if 'train' in args['domain']:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@scale=%s' % args['scales']

    pred_dir = create_directory(f'/home/danylokolinko/logs/inference/{experiment_name}/')

    model_path = config['model']['model_path']

    # set_seed(args.seed)
    # log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # rgby_mean = [0.08123, 0.05293, 0.05398, 0.08153]
    # rgby_std = [0.13028, 0.08611, 0.14256, 0.12620]

    # normalize_fn = Normalize(rgby_mean, rgby_std)
    
    # for mIoU
    # meta_dic = read_json('./data/VOC_2012.json')
    # dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    model = Classifier(config['model'])

    model = model.cuda()
    model.eval()

    tag_str = config['model']['model_path'].split('/')[-1].split('.')[0]
    print(tag_str)
    print(f'loading csv from {tag_str}')

    df = pd.read_csv(os.path.join(config['log_path'], 'csv', tag_str))
    print(df.head())
    train_df = df.loc[~df.is_valid].reset_index(drop=True)
    train_transform = get_transforms(config['train']['transform'])

    dataset = HPADatasetCAM(config, train_df, transform=train_transform, train=True)

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        model = nn.DataParallel(model)

    load_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    #################################################################################################
    # Evaluation
    #################################################################################################

    eval_timer = Timer()
    scales = [float(scale) for scale in args['scales'].split(',')]
    
    model.eval()
    eval_timer.tik()

    def get_cam(ori_image, scale, ori_w, ori_h):
        # preprocessing
        # image = copy.deepcopy(ori_image)
        # image = image.resize(, resample=PIL.Image.CUBIC)
        image = resize_for_tensors(ori_image.unsqueeze(0), (round(ori_w*scale), round(ori_h*scale)))[0]
        # image = normalize_fn(image)
        # image = image.transpose((2, 0, 1))

        # image = torch.from_numpy(image)
        vflipped_image = torch.flip(image, (1, ))
        hflipped_image = torch.flip(image, (2, ))


        
        images = torch.stack([image, vflipped_image, hflipped_image])
        images = images.cuda()
        
        # inferenece
        _, features = model(images, with_cam=True)

        # postprocessing
        cams = F.relu(features)
        cams = cams[0] + torch.flip(cams[1], (1, )) + torch.flip(cams[2], (2, ))

        return cams


    loader = DataLoader(dataset, batch_size=1, num_workers=config['args']['num_workers'],
                              shuffle=False,
                              drop_last=False)

    with torch.no_grad():

        length = len(loader)
        for step, (ori_image, label, image_id, gt_mask, ana) in enumerate(tqdm(loader)):
            # ori_w, ori_h = ori_image.size
            image_id = image_id[0]
            ori_image = ori_image.squeeze(0)
            label = label.squeeze(0)

            npy_path = pred_dir + image_id + '.npy'

            if os.path.isfile(npy_path):
                continue
            ori_h, ori_w = ori_image.size(1), ori_image.size(2)
            strided_size = get_strided_size((ori_h, ori_w), 4)
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

            cams_list = [get_cam(ori_image, scale, ori_w, ori_h) for scale in scales]

            strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
            strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)

            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]
            
            keys = torch.nonzero(label)[:, 0]
            # keys = torch.nonzero(torch.from_numpy(label))[:, 0]

            strided_cams = strided_cams[keys]
            strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5
            
            hr_cams = hr_cams[keys]

            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5
            keys = np.pad(keys + 1, (1, 0), mode='constant')

            np.save(npy_path, {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})

            # sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), hr_cams.size()))
            # sys.stdout.flush()


