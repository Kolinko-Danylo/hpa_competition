# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from hpa_competition.model_training.common.augmentations import get_transforms

from hpa_competition.PuzzleCAM.core.networks import *
from hpa_competition.PuzzleCAM.core.datasets import *

from hpa_competition.PuzzleCAM.tools.general.io_utils import *
from hpa_competition.PuzzleCAM.tools.general.time_utils import *
from hpa_competition.PuzzleCAM.tools.general.json_utils import *

from hpa_competition.PuzzleCAM.tools.ai.log_utils import *
from hpa_competition.PuzzleCAM.tools.ai.demo_utils import *
from hpa_competition.PuzzleCAM.tools.ai.optim_utils import *
from hpa_competition.PuzzleCAM.tools.ai.torch_utils import *
from hpa_competition.PuzzleCAM.tools.ai.evaluate_utils import *

from hpa_competition.PuzzleCAM.tools.ai.augment_utils import *
from hpa_competition.PuzzleCAM.tools.ai.randaugment import *
from hpa_competition.model_training.common.datasets import HPADatasetCAMTest
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)


def get_cam(ori_image, scale):
    # preprocessing
    image = copy.deepcopy(ori_image)
    # image = image.resize((round(ori_w * scale), round(ori_h * scale)), resample=PIL.Image.CUBIC)

    # image = normalize_fn(image)
    # image = image.transpose((2, 0, 1))

    # image = torch.from_numpy(image)
    flipped_image = image.flip(-1)

    images = torch.stack([image, flipped_image])
    # images = images.cuda()

    # inferenece
    _, features = model(images, with_cam=True)

    # postprocessing
    cams = F.relu(features)
    cams = cams[0] + cams[1].flip(-1)

    return cams

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'config', 'cam.yaml')) as config_file:
        config = yaml.full_load(config_file)
    args = parser.parse_args()
    # experiment_name = args.tag
    #
    # if 'train' in args.domain:
    #     experiment_name += '@train'
    # else:
    #     experiment_name += '@val'
    #
    # experiment_name += '@scale=%s' % args.scales
    #
    # pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')
    #
    # model_path = './experiments/models/' + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    model = Classifier(config['model']['arch'], config['model']['classes'], mode=args.mode)

    # model = model.cuda()
    model.eval()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model(model, config['model']['model_path'], parallel=the_number_of_gpu > 1)

    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]

    model.eval()
    eval_timer.tik()

    df = pd.read_csv(os.path.join(config['test']['path'], 'sample_submission.csv' ))
    test_transform = get_transforms(config['test']['transform'])


    dataset = HPADatasetCAMTest(config['test']['path'], df, transform=test_transform, img_size=None)


    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id) in enumerate(dataset):
            # print(ori_image.shape)
            _, ori_w, ori_h = ori_image.shape

            npy_path = os.path.join(config['pred_dir'], f'{image_id}.npy')

            strided_size = get_strided_size((ori_h, ori_w), 4)
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)
            scales = [0]
            cams_list = [get_cam(ori_image, scale) for scale in scales]


            strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
            strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)

            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]

            # keys = torch.nonzero(torch.from_numpy(label))[:, 0]
            #
            # strided_cams = strided_cams[keys]
            # strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5
            #
            # hr_cams = hr_cams[keys]
            # hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

            # save cams
            # keys = np.pad(keys + 1, (1, 0), mode='constant')
            # np.save(npy_path, {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})

            # sys.stdout.write(
            #     '\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100,
            #                                                       (ori_h, ori_w), hr_cams.size()))
            # sys.stdout.flush()
        # print()

    # if args.domain == 'train_aug':
    #     args.domain = 'train'

    # print("python3 evaluate.py --experiment_name {} --domain {}".format(experiment_name, args.domain))
