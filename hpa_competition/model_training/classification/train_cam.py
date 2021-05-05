from datetime import datetime

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from hpa_competition.PuzzleCAM.core.networks import *
from hpa_competition.PuzzleCAM.core.datasets import *
from hpa_competition.PuzzleCAM.tools.general.io_utils import *
from hpa_competition.PuzzleCAM.tools.general.time_utils import *
from hpa_competition.PuzzleCAM.tools.general.json_utils import *
from hpa_competition.PuzzleCAM.tools.ai.log_utils import *
from hpa_competition.PuzzleCAM.tools.ai.optim_utils import *
from hpa_competition.PuzzleCAM.tools.ai.torch_utils import *
import torch

import os
from hpa_competition.model_training.common.datasets import HPADatasetCAM
from hpa_competition.model_training.common.augmentations import get_transforms
from utils  import get_df, get_df_cam
import yaml
from cam_pipeline import CAMTrainer
from hpa_competition.PuzzleCAM.core.networks import Classifier
from torch.backends import cudnn

import tqdm
y_file = 'cam_avenga.yaml'
if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.enabled = True
    with open(os.path.join(os.path.dirname(__file__), 'config', y_file)) as config_file:
        config = yaml.full_load(config_file)

    # model = Classifier(config['model']['arch'], config['model']['pretreined'],
    #                num_classes=config['model']['classes'], mode=config['args']['mode'])
    # trainer = CAMTrainer(config, None, None)


    set_seed(config['args']['seed'])

    train_transform = get_transforms(config['train']['transform'])
    val_transform = get_transforms(config['val']['transform'])

    print(config['model']['load_weights'])
    if config['model']['load_weights']:
        tag_str = config['model']['model_path'].split('/')[-1].split('.')[0]
        # if tag_str.endswith('cell_level'):
        csvtag_str = tag_str.strip('_cell_level')
        print(tag_str)
        print(f'loading csv from {tag_str}')
        filec = os.path.join(config['log_path'], 'csv', csvtag_str)
        if config['pretrain'] or (not os.path.isfile(filec) and os.path.isfile(os.path.join(config['log_path'], 'csv', csvtag_str+'_pretraining'))):
            print('not_loading')
            df = get_df_cam(path=config['train']['path'])
        else:
            df = pd.read_csv(filec)
            print(df.head())
    else:
        print('not_loading')
        df = get_df_cam(path=config['train']['path'])


    train_df = df.loc[~df.is_valid].reset_index(drop=True)
    val_df = df.loc[df.is_valid].reset_index(drop=True)
    path = config['train']['path']
    train_dataset = HPADatasetCAM(config, train_df, transform=train_transform, train=True)
    val_dataset = HPADatasetCAM(config, val_df, transform=val_transform, train=False)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['args']['num_workers'], shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['args']['num_workers'], shuffle=True,
                              drop_last=True)

    trainer = CAMTrainer(config, train_loader, val_loader)
    sstr = trainer.tag_str + ('_pretraining' if config['pretrain'] else '')
    df.to_csv(os.path.join(config['log_path'], 'csv', sstr), index=False)


    trainer.train()
