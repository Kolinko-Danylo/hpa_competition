from datetime import datetime

# from torchvision import transforms
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
from hpa_competition.PuzzleCAM.tools.ai.randaugment import *
from hpa_competition.model_training.common.metrics import AveragePrecision
import torch
import os
import pandas as pd
import numpy as np
from hpa_competition.model_training.common.datasets import HPADatasetCAM
from hpa_competition.model_training.common.augmentations import get_transforms
from utils  import get_df, get_df_cam
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--print_ratio', default=0.1, type=float)
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--num_pieces', default=4, type=int)
parser.add_argument('--loss_option', default='cl_pcl_re', type=str)
parser.add_argument('--level', default='feature', type=str)
parser.add_argument('--re_loss', default='L1_Loss', type=str)  # 'L1_Loss', 'L2_Loss'
parser.add_argument('--re_loss_option', default='masking', type=str)  # 'none', 'masking', 'selection'
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--alpha_schedule', default=0.50, type=float)

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'config', 'cam.yaml')) as config_file:
        config = yaml.full_load(config_file)

    args = parser.parse_args()
    log_path = config['log_path']
    log_dir = create_directory(os.path.join(log_path, 'logs'))
    data_dir = create_directory(os.path.join(log_path, 'data'))
    model_dir = create_directory(os.path.join(log_path, 'models'))
    tag_str = f'{config["tag"]}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    tensorboard_dir = create_directory(os.path.join(log_path,  f'tensorboards/{tag_str}/'))

    log_path = log_dir + f'{tag_str}.txt'
    data_path = data_dir + f'{tag_str}.json'
    model_path = model_dir + f'{tag_str}.pth'
    set_seed(args.seed)

    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(tag_str))
    log_func()

    train_transform = get_transforms(config['train']['transform'])
    val_transform = get_transforms(config['val']['transform'])

    # train_transforms = [
    #     transforms.RandomResizedCrop(args.image_size),
    #     transforms.RandomHorizontalFlip(),
    # ]size


    # train_transform = transforms.Compose(train_transform)
    # df = pd.read_csv('/datasets/kolinko/hpa/train.csv')

    train_df, val_df = get_df_cam(path=config['train']['path'])
    path = config['train']['path']
    # train_ds = HPADataset(path, train_df, transform=train_transform)
    # val_ds = HPADataset(path, val_df, transform=val_transform)

    train_dataset = HPADatasetCAM(config['train']['path'], train_df, transform=train_transform)
    val_dataset = HPADatasetCAM(config['train']['path'], val_df, transform=val_transform)


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=args.num_workers, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=args.num_workers, shuffle=True,
                              drop_last=True)


    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration
    val_epoch_iteration = len(val_loader)


    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    model = Classifier(config['model']['arch'], num_classes=config['model']['classes'], mode=args.mode)
    param_groups = model.get_parameter_groups(print_fn=None)

    gap_fn = model.global_average_pooling_2d

    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(config['model']['arch']))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        print(f"use_gpu:{use_gpu}\n\n\n\n\n\n\n\n\n")
    except KeyError:
        use_gpu = '0'
    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)



    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)

    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()

    if args.re_loss == 'L1_Loss':
        re_loss_fn = L1_Loss
    else:
        re_loss_fn = L2_Loss

    log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)

    data_dic = {
        'train': [],
        'validation': []
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'class_loss', 'p_class_loss', 're_loss', 'conf_loss', 'alpha'])
    val_meter = Average_Meter(['loss', 'class_loss', 'p_class_loss', 're_loss', 'conf_loss', 'alpha'])
    best_val_loss = None

    best_val_ap = None

    best_train_mIoU = -1
    thresholds = list(np.arange(0.10, 0.50, 0.05))

    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)
    val_iterator = Iterator(val_loader)
    val_ap = AveragePrecision(classes=config['model']['classes'], device=torch.device('cuda'))

    loss_option = args.loss_option.split('_')

    for iteration in range(max_iteration):
        model.train()
        images, labels = train_iterator.get()
        images, labels = images.cuda(), labels.cuda()

        logits, features = model(images, with_cam=True)
        tiled_images = tile_features(images, args.num_pieces)
        tiled_logits, tiled_features = model(tiled_images, with_cam=True)
        re_features = merge_features(tiled_features, args.num_pieces, config['batch_size'])
        if args.level == 'cam':
            features = make_cam(features)
            re_features = make_cam(re_features)

        class_loss = class_loss_fn(logits, labels).mean()

        if 'pcl' in loss_option:
            p_class_loss = class_loss_fn(gap_fn(re_features), labels).mean()
        else:
            p_class_loss = torch.zeros(1).cuda()

        if 're' in loss_option:
            if args.re_loss_option == 'masking':
                class_mask = labels.unsqueeze(2).unsqueeze(3)
                re_loss = re_loss_fn(features, re_features) * class_mask
                re_loss = re_loss.mean()
            elif args.re_loss_option == 'selection':
                re_loss = 0.
                for b_index in range(labels.size()[0]):
                    class_indices = labels[b_index].nonzero(as_tuple=True)
                    selected_features = features[b_index][class_indices]
                    selected_re_features = re_features[b_index][class_indices]

                    re_loss_per_feature = re_loss_fn(selected_features, selected_re_features).mean()
                    re_loss += re_loss_per_feature
                re_loss /= labels.size()[0]
            else:
                re_loss = re_loss_fn(features, re_features).mean()
        else:
            re_loss = torch.zeros(1).cuda()

        if 'conf' in loss_option:
            conf_loss = shannon_entropy_loss(tiled_logits)
        else:
            conf_loss = torch.zeros(1).cuda()

        if args.alpha_schedule == 0.0:
            alpha = args.alpha
        else:
            alpha = min(args.alpha * iteration / (max_iteration * args.alpha_schedule), args.alpha)

        loss = class_loss + p_class_loss + alpha * re_loss + conf_loss
        #################################################################################################

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss': loss.item(),
            'class_loss': class_loss.item(),
            'p_class_loss': p_class_loss.item(),
            're_loss': re_loss.item(),
            'conf_loss': conf_loss.item(),
            'alpha': alpha,
        })


        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss, class_loss, p_class_loss, re_loss, conf_loss, alpha = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))

            data = {
                'iteration': iteration + 1,
                'learning_rate': learning_rate,
                'alpha': alpha,
                'loss': loss,
                'class_loss': class_loss,
                'p_class_loss': p_class_loss,
                're_loss': re_loss,
                'conf_loss': conf_loss,
                'time': train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)

            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                alpha={alpha:.2f}, \
                loss={loss:.4f}, \
                class_loss={class_loss:.4f}, \
                p_class_loss={p_class_loss:.4f}, \
                re_loss={re_loss:.4f}, \
                conf_loss={conf_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
                     )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/class_loss', class_loss, iteration)
            writer.add_scalar('Train/p_class_loss', p_class_loss, iteration)
            writer.add_scalar('Train/re_loss', re_loss, iteration)
            writer.add_scalar('Train/conf_loss', conf_loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
            writer.add_scalar('Train/alpha', alpha, iteration)

            write_json(data_path, data_dic)


        if (iteration + 1) % val_iteration == 0:
        # if (iteration) == 60:


            model.eval()
            val_ap.reset()
            for vali in range(val_epoch_iteration):
                images, labels = val_iterator.get()
                images, labels = images.cuda(), labels.cuda()

                logits, features = model(images, with_cam=True)
                tiled_images = tile_features(images, args.num_pieces)
                tiled_logits, tiled_features = model(tiled_images, with_cam=True)
                re_features = merge_features(tiled_features, args.num_pieces, config['batch_size'])
                if args.level == 'cam':
                    features = make_cam(features)
                    re_features = make_cam(re_features)

                val_ap.add(logits, labels) #TODO: LOGITS
                class_loss = class_loss_fn(logits, labels).mean()

                if 'pcl' in loss_option:
                    p_class_loss = class_loss_fn(gap_fn(re_features), labels).mean()
                else:
                    p_class_loss = torch.zeros(1).cuda()

                if 're' in loss_option:
                    if args.re_loss_option == 'masking':
                        class_mask = labels.unsqueeze(2).unsqueeze(3)
                        re_loss = re_loss_fn(features, re_features) * class_mask
                        re_loss = re_loss.mean()
                    elif args.re_loss_option == 'selection':
                        re_loss = 0.
                        for b_index in range(labels.size()[0]):
                            class_indices = labels[b_index].nonzero(as_tuple=True)
                            selected_features = features[b_index][class_indices]
                            selected_re_features = re_features[b_index][class_indices]

                            re_loss_per_feature = re_loss_fn(selected_features, selected_re_features).mean()
                            re_loss += re_loss_per_feature
                        re_loss /= labels.size()[0]
                    else:
                        re_loss = re_loss_fn(features, re_features).mean()
                else:
                    re_loss = torch.zeros(1).cuda()

                if 'conf' in loss_option:
                    conf_loss = shannon_entropy_loss(tiled_logits)
                else:
                    conf_loss = torch.zeros(1).cuda()

                val_loss = class_loss + p_class_loss + alpha * re_loss + conf_loss
                #################################################################################################

                val_meter.add({
                    'loss': val_loss.item(),
                    'class_loss': class_loss.item(),
                    'p_class_loss': p_class_loss.item(),
                    're_loss': re_loss.item(),
                    'conf_loss': conf_loss.item(),
                    'alpha': alpha,
                })
            ap_score = val_ap.get()
            val_loss, class_loss, p_class_loss, re_loss, conf_loss, alpha = val_meter.get(clear=True)
            if (best_val_ap is None) or (best_val_ap < ap_score):
                print(f'saving model to {model_path}')
                save_model_fn()
                best_val_ap = ap_score

            val_ap.write_to_tensorboard(writer, iteration, prefix='Val/')
            writer.add_scalar('Val/loss', val_loss, iteration)
            writer.add_scalar('Val/class_loss', class_loss, iteration)
            writer.add_scalar('Val/p_class_loss', p_class_loss, iteration)
            writer.add_scalar('Val/re_loss', re_loss, iteration)
            writer.add_scalar('Val/conf_loss', conf_loss, iteration)
            # writer.add_scalar('Val/learning_rate', learning_rate, iteration)
            writer.add_scalar('Val/alpha', alpha, iteration)



    write_json(data_path, data_dic)
    writer.close()

    print(tag_str)
