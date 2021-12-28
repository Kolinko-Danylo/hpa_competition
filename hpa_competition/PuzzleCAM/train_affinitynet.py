# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

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
LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles",
             "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments",
             "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol",
             "Vesicles", "Negative"]
###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=3, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)

parser.add_argument('--pred_dir', default='./experiments/predictions/', type=str)
parser.add_argument('--label_name', default='resnet50@seed=0@nesterov@train_aug@bg=0.20@scale=0.5,1.0,1.5,2.0@aff', type=str)
import yaml

if __name__ == '__main__':
    with open(os.path.join('/home/danylokolinko/hpa/hpa_competition/model_training/classification', 'config',
                           'cam_inf_puzzle.yaml')) as config_file:
        config = yaml.full_load(config_file)
    experiment_name = config['model']['model_path'].split('/')[-1].split('.')[0]

    args = config['args']
    # if 'train' in args['domain']:
    #     experiment_name += '@train'
    # else:
    #     experiment_name += '@val'
    #
    # experiment_name += '@scale=%s' % args['scales']
    #
    # pred_dir = f'/home/danylokolinko/logs/inference/{experiment_name}/'
    #
    # model_path = config['model']['model_path']
    #
    # aff_dir = create_directory(
    #     '/home/danylokolinko/logs/inference/{}@aff_fg={:.2f}_bg={:.2f}/'.format(experiment_name, args['fg_threshold'],
    #                                                                             args['bg_threshold']))
    bas = '/home/danylokolinko/logs/affinity'
    log_dir = create_directory(os.path.join(bas, 'logs'))
    data_dir = create_directory(os.path.join(bas, 'data'))
    model_dir = create_directory(os.path.join(bas, 'models'))
    tagg = experiment_name + '_affinity'
    tensorboard_dir = create_directory(f'{bas}/{tagg}/')

    log_path = os.path.join(log_dir, f'{tagg}.txt')
    data_path = os.path.join(data_dir, f'{tagg}.json')
    model_path = os.path.join(model_dir, f'{tagg}.pth')

    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(tagg))
    log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]

    # normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    # train_transform = transforms.Compose([
    #     RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
    #     RandomHorizontalFlip_For_Segmentation(),
    #
    #     Normalize_For_Segmentation(imagenet_mean, imagenet_std),
    #     RandomCrop_For_Segmentation(args.image_size),
    #
    #     Transpose_For_Segmentation(),
    #     Resize_For_Mask(args.image_size // 4),
    # ])
    
    # meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(LBL_NAMES)
    def_size = config['train']['transform']['size']
    path_index = PathIndex(radius=10, default_size=(def_size, def_size))
    from hpa_competition.model_training.common.augmentations import get_transforms
    from hpa_competition.model_training.common.datasets.hpa import General_Dataset_For_Affinity

    import pandas as pd
    train_transform = get_transforms(config['train']['transform'])
    tag_str = config['model']['model_path'].split('/')[-1].split('.')[0]
    print(tag_str)
    print(f'loading csv from {tag_str}')

    df = pd.read_csv(os.path.join(config['log_path'], 'csv', tag_str))
    print(df.head())

    pathh = 'tf_efficientnet_b4-2021-04-26-23-11-24_channel_dropout@train@scale=0.8, 1, 1.2@aff_fg=0.50_bg=0.20'
    base_path = os.path.join('/home/danylokolinko/logs/inference/', pathh)

    train_df = df.loc[~df.is_valid].reset_index(drop=True)
    train_dataset = General_Dataset_For_Affinity('whatever', 'whatever', path_index=path_index, label_dir=base_path, transform=train_transform, df=train_df, train=True, config=config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=args['num_workers'], shuffle=True, drop_last=True)
    

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args['print_ratio'])
    max_iteration = config['num_epochs'] * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    ###################################################################################
    # Network
    ###################################################################################
    model = AffinityNet(config['model'], path_index)


    param_groups = list(model.edge_layers.parameters())

    model = model.cuda()
    model.train()


    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'
    
    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn_for_backup = lambda: save_model(model, model_path.replace('.pth', f'_backup.pth'), parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    optimizer = PolyOptimizer([
        {'params': param_groups, 'lr': args['lr'], 'weight_decay': args['wd']},
    ], lr=args['lr'], momentum=0.9, weight_decay=args['wd'], max_step=max_iteration, nesterov=args['nesterov'])
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
    }

    train_timer = Timer()
    train_meter = Average_Meter([
        'loss', 
        'bg_loss', 'fg_loss', 'neg_loss',
    ])
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)

    for iteration in range(max_iteration):
        images, labels = train_iterator.get()

        images = images.cuda()

        bg_pos_label = labels[0].cuda(non_blocking=True)
        fg_pos_label = labels[1].cuda(non_blocking=True)
        neg_label = labels[2].cuda(non_blocking=True)
        
        #################################################################################################
        # Affinity Matrix
        #################################################################################################
        edge, aff = model(images, with_affinity=True)

        ###############################################################################
        # The part is to calculate losses.
        ###############################################################################
        pos_aff_loss = (-1) * torch.log(aff + 1e-5)
        neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

        bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
        fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)

        pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
        neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

        loss = (pos_aff_loss + neg_aff_loss) / 2
        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 

            'bg_loss' : bg_pos_aff_loss.item(),
            'fg_loss' : fg_pos_aff_loss.item(),
            'neg_loss' : neg_aff_loss.item(),
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss, bg_loss, fg_loss, neg_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,

                'bg_loss' : bg_loss,
                'fg_loss' : fg_loss,
                'neg_loss' : neg_loss,

                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                bg_loss={bg_loss:.4f}, \
                fg_loss={fg_loss:.4f}, \
                neg_loss={neg_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/bg_loss', bg_loss, iteration)
            writer.add_scalar('Train/fg_loss', fg_loss, iteration)
            writer.add_scalar('Train/neg_loss', neg_loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            save_model_fn()
            
    save_model_fn()

    write_json(data_path, data_dic)
    writer.close()

