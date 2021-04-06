from datetime import datetime

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



if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'config', 'cam.yaml')) as config_file:
        config = yaml.full_load(config_file)
    set_seed(config['args']['seed'])

    train_transform = get_transforms(config['train']['transform'])
    val_transform = get_transforms(config['val']['transform'])

    df = get_df_cam(path=config['train']['path'])
    train_df = df.loc[~df.is_valid].reset_index(drop=True)
    val_df = df.loc[df.is_valid].reset_index(drop=True)

    df.to_csv(os.path.join(config['log_path'], 'csv'))
    path = config['train']['path']

    train_dataset = HPADatasetCAM(config['train']['path'], train_df, transform=train_transform)
    val_dataset = HPADatasetCAM(config['train']['path'], val_df, transform=val_transform)


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['args']['num_workers'], shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['args']['num_workers'], shuffle=True,
                              drop_last=True)

    trainer = CAMTrainer(config, train_loader, val_loader)
    trainer.train()





    # log_iteration = int(val_iteration * args.print_ratio)
    # val_epoch_iteration = len(val_loader)



    # data_dic = {
    #     'train': [],
    #     'validation': []
    # }

# data = {
#     'iteration': iteration + 1,
#     'learning_rate': learning_rate,
#     'alpha': alpha,
#     'loss': loss,
#     'class_loss': class_loss,
#     'p_class_loss': p_class_loss,
#     're_loss': re_loss,
#     'conf_loss': conf_loss,
#     'time': train_timer.tok(clear=True),
# }
# data_dic['train'].append(data)



# write_json(data_path, data_dic)
