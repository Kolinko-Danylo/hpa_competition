import yaml
import torch
import numpy as np
import os
import random

from brats_competition.model_training.common.trainer import Trainer
from brats_competition.model_training.common.datasets import Brats2dDataset
from brats_competition.model_training.common.augmentations import get_transforms

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(os.path.join(os.path.dirname(__file__), 'config', 'unet2d.yaml')) as config_file:
    config = yaml.full_load(config_file)

train_transform = get_transforms(config['train']['transform'])
val_transform = get_transforms(config['val']['transform'])

train_ds = Brats2dDataset(config['train']['path'], transform=train_transform)
val_ds = Brats2dDataset(config['val']['path'], transform=val_transform)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True, num_workers=12)

trainer = Trainer(config, train_dl, val_dl)
trainer.train()
