import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from hpa_competition.model_training.common.metrics import AveragePrecision, AveragePrecision_SingleClass_Instance, \
    IoUMetricCell
from hpa_competition.model_training.common.losses import get_loss
import yaml
import cv2

import hpa_competition.model_training.common.lovasz_losses as L

from hpa_competition.PuzzleCAM.tools.ai.torch_utils import make_cam, L1_Loss, L2_Loss, shannon_entropy_loss, load_model, \
    save_model, get_learning_rate_from_optimizer, resize_for_tensors
from hpa_competition.PuzzleCAM.core.puzzle_utils import tile_features, merge_features
from hpa_competition.PuzzleCAM.core.datasets import Iterator
from hpa_competition.PuzzleCAM.core.networks import Classifier
from hpa_competition.PuzzleCAM.tools.ai.optim_utils import PolyOptimizer

from hpa_competition.PuzzleCAM.tools.general.io_utils import create_directory
from hpa_competition.PuzzleCAM.tools.ai.log_utils import log_print, Average_Meter
from hpa_competition.PuzzleCAM.tools.general.time_utils import Timer

from utils import build_image_names, unnorm_features
from datetime import datetime
import os
import hpacellseg.cellsegmentator as cellsegmentator
import tqdm


def CXE(predicted, target):
    e = 0.0001
    return -(target * torch.log(predicted + e) + ((1 - target) * (torch.log(1 - predicted + e))))
    # print(target.amax())
    # print(target.amin())
    # print(predicted.amax())
    # print(predicted.amin())


class CAMTrainer:
    def __init__(self, config, train_dl, val_dl):
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.segmentation = self.config['model']['segmentation']
        self.selfsupervision = self.config['selfsupervise']

        self.mask_supervised = config['train']['load_mask'] and not config['train']['cell_input']

        log_path = config['log_path']

        log_dir = os.path.join(log_path, 'logs')
        data_dir = os.path.join(log_path, 'data')
        model_dir = os.path.join(log_path, 'models')
        yamls_dir = os.path.join(log_path, 'yamls')

        tag_str = f'{config["model"]["arch"]}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}_{self.config["expchange"]}'
        self.tag_str = tag_str
        # self.segmentator = cellsegmentator.CellSegmentator(
        #     config['segm_model']['nuclei_path'],
        #     config['segm_model']['cell_path'],
        #     scale_factor=0.25,
        #     device='cuda',
        #     padding=True,
        #     multi_channel_model=True
        # )

        with open(f'{tag_str}.yml', 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

        self.model_path = os.path.join(model_dir, f'{tag_str}.pth')

        self.model_last_path = os.path.join(model_dir, f'{tag_str}_last.pth')
        if not self.config['pretrain']:
            self.model_cell_path = os.path.join(model_dir, f'{tag_str}_cell_level.pth')
        # self.model_nuclei_path = os.path.join(model_dir, f'{tag_str}_nuclei_level.pth')

        tensorboard_dir = create_directory(os.path.join(log_path, f'tensorboards/{tag_str}/'))
        log_txt_path = os.path.join(log_dir, f'{tag_str}.txt')

        self.log_func = lambda string='': log_print(string, log_txt_path)
        self.log_func('[i] {}'.format(tag_str))
        # load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
        # save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
        self.sigmoid = nn.Sigmoid()

        self.train_timer = Timer()
        self.eval_timer = Timer()

        loss_lst = ['loss', 'class_loss', 'p_class_loss', 're_loss', 'conf_loss', 'alpha', 'l1_segmentation', 'lovasz']

        self.train_meter = Average_Meter(loss_lst)
        self.val_meter = Average_Meter(loss_lst)
        self.writer = SummaryWriter(tensorboard_dir)
        self.img_dir = os.path.join(config['train']['path'], 'train')
        self.bs = self.config['batch_size']

        self.best_val_ap = None
        self.best_val_ap_cell = None
        self.segm_losss = CXE
        # self.best_val_ap_nuclei = None

        self.use_amp = self.config['use_amp']
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train(self):
        self._init_params()
        for ep in range(self.config['num_epochs']):

            self.epoch = ep
            self.alpha = 1.0
            print(f'Epoch: {self.epoch}')
            # train_metric_lst = [self.train_ap, self.train_ap_cell, self.train_ap_nuclei]
            # train_metric_lst = [self.train_ap, self.train_ap_cell]
            # val_metric_lst = [self.val_ap, self.val_ap_cell]
            #
            # ap_score_lst = self.run_epoch(metric_lst=self.val_metric_lst, loss_meter=self.val_meter,
            #                               data_loader=self.val_dl,
            #                               train=False)
            # print(ap_score_lst)
            self.run_epoch(metric_lst=self.train_metric_lst, loss_meter=self.train_meter,
                           data_loader=self.train_dl, train=True, break_after=201)
            # val_metric_lst = [self.val_ap, self.val_ap_cell, self.val_ap_nuclei]

            ap_score_lst = self.run_epoch(metric_lst=self.val_metric_lst, loss_meter=self.val_meter,
                                          data_loader=self.val_dl,
                                          train=False)
            ap_score = ap_score_lst[0]
            ap_score_cell = ap_score_lst[-1]
            # ap_score_nuclei = ap_score_lst[2]

            print(f'ap_score: {ap_score}')
            print(f'ap_score_cell: {ap_score_cell}')
            # print(f'ap_score_nuclei: {ap_score_nuclei}')
            self._save_checkpoint(self.model_last_path)

            # save_model(self.model, self.model_last_path, parallel=self.the_number_of_gpu > 1)
            # self.best_val_ap_nuclei = self.sv_mdl(self.best_val_ap_nuclei, ap_score_nuclei, self.model_nuclei_path)
            if not self.config['pretrain']:
                self.best_val_ap_cell = self.sv_mdl(self.best_val_ap_cell, ap_score_cell, self.model_cell_path)

            self.best_val_ap = self.sv_mdl(self.best_val_ap, ap_score, self.model_path)

        self.writer.close()

    def sv_mdl(self, bst_scr, current, path):
        if (bst_scr is None) or (bst_scr < current):
            print(f'saving model to {path}')
            self._save_checkpoint(path)
            return current
        else:
            return bst_scr

    def _save_checkpoint(self, path):
        torch.save(
            {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                # 'loss': self.config['model']['loss'],
                # 'scheduler': self.scheduler.state_dict()
            },
            path)

    def _init_params(self):
        self.model = Classifier(self.config['model'])
        # self.param_groups = self.model.get_parameter_groups(print_fn=None)
        if not self.segmentation:
            self.model.classifier.cuda()
            self.model.model.encoder.cuda()
        else:
            self.model.cuda()


        self.gap_fn = self.model.global_average_pooling_2d
        # self.params = self.model.get_parameter_groups(print_fn=None)

        # try:
        #     use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        #     print(f"use_gpu:{use_gpu}\n\n")
        # except KeyError:
        #     use_gpu = '0'
        # the_number_of_gpu = len(use_gpu.split(','))
        # self.the_number_of_gpu = the_number_of_gpu

        # if the_number_of_gpu > 1:
        #     self.log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        #     self.model = nn.DataParallel(self.model)
        def freeze_params(module):
            for p in module.parameters():
                p.requires_grad = False

        if self.config['model']['load_weights']:

            d = torch.load(self.config['model']['model_path'])['model']
            if self.segmentation:

                for mod in [self.model.model.encoder.conv_stem, self.model.model.encoder.bn1, *self.model.model.encoder.blocks[:3], self.model.classifier]:
                    freeze_params(mod)
            d.pop('model.segmentation_head.0.weight')
            d.pop('model.segmentation_head.0.bias')
            for k in list(filter(lambda x: x.startswith('model.decoder'), list(d.keys()))):
                d.pop(k)
            # d.pop('model)
            pretrained_dict = d
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)


            # self.model.load_state_dict(d)
            # .requires_grad = False
            # self.model.model.encoder.bn1.requires_grad = False
            # self.model.model.encoder.act1.requires_grad = False
            # .requires_grad = False
            # .requires_grad = False
            # # inp = cl.model.encoder.bn1(inp)
            # inp = cl.model.encoder.act1(inp)
            # inp = cl.model.encoder.blocks[:3](inp)

            # for name, param in self.model.named_parameters():
            #
            #     if  name.startswith('classifier'):
            #         param.requires_grad = False
            #     if name.startswith('model.encoder'):
            #         if name.startswith(f'model.encoder.blocks') and not any([name.startswith(f'model.encoder.blocks.{i}') for i in range(3, 8)]):
            #             param.requires_grad = False
            #
            #         if not any([name.startswith(f'model.encoder.{nname}') for nname in ['conv_head', 'bn2', 'act2', 'global_pool', 'blocks']]):
            #             param.requires_grad = False

            # nlst = []
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad and name.startswith('model.encoder'):
            #         nnn = name.replace('model.encoder.', '')
            #         nnn = nnn[:min(8, len(nnn))]
            #         nlst.append(nnn)
            # for s in set(nlst):
            #     print(s)

                    # print(name)





            # self.model.model.encoder.load_state_dict(torch.load(self.config['model']['model_path'])['model'].model.encoder)
            # self.model.model.decoder.load_state_dict(torch.load(self.config['model']['model_path'])['model'].model.decoder)
            # self.model.classifier.load_state_dict(torch.load(self.config['model']['model_path'])['model'].classifier)

            # self.model.model.segmentation_head[0] = torch.nn.Conv2d(16, 4, kernel_size=(3, 3), stride=(1, 1),
            #                                                         padding=(1, 1)).cuda()
            # load_model(self.model, self.config['model']['model_path'], parallel=the_number_of_gpu > 1)

        self.class_loss_fn = get_loss(self.config['model']['loss'])

        if self.config['args']['re_loss'] == 'L1_Loss':

            self.re_loss_fn = L1_Loss
        else:
            self.re_loss_fn = L2_Loss
        # self.run_epoch(metric_lst=self.train_metric_lst, loss_meter=self.train_meter,
        #

        self._init_optimizer()

        self.val_ap = AveragePrecision(classes=self.config['model']['classes'], device=torch.device('cuda'))
        self.train_ap = AveragePrecision(classes=self.config['model']['classes'], device=torch.device('cuda'))
        self.val_metric_lst = [self.val_ap]
        self.train_metric_lst = [self.train_ap]

        if not self.config['pretrain']:
            self.val_ap_cell = AveragePrecision_SingleClass_Instance(classes=self.config['model']['classes'],
                                                                     device=torch.device('cuda'), masktype='cell')
            self.train_ap_cell = AveragePrecision_SingleClass_Instance(classes=self.config['model']['classes'],
                                                                       device=torch.device('cuda'), masktype='cell')
            self.val_metric_lst.append(self.val_ap_cell)
            self.train_metric_lst.append(self.train_ap_cell)

            # self.val_ap_nuclei = AveragePrecision_SingleClass_Instance(classes=self.config['model']['classes'], device=torch.device('cuda'), masktype='nuclei')
        # self.train_ap_nuclei = AveragePrecision_SingleClass_Instance(classes=self.config['model']['classes'], device=torch.device('cuda'), masktype='nuclei')
        self.loss_option = self.config['args']['loss_option'].split('_')
        self.iters2acc = self.config['args']['iters2acc']

    def _init_optimizer(self):
        # param_groups = self.params
        # lr = self.config['args']['lr']
        # wd = self.config['args']['wd']
        # nesterov = self.config['args']['nesterov']

        # self.optimizer = PolyOptimizer([
        #     {'params': param_groups[0], 'lr': lr, 'weight_decay': wd},
        #     {'params': param_groups[1], 'lr': 2 * lr, 'weight_decay': 0},
        #     {'params': param_groups[2], 'lr': 10 * lr, 'weight_decay': wd},
        #     {'params': param_groups[3], 'lr': 20 * lr, 'weight_decay': 0},
        # ], lr=lr, momentum=0.9, weight_decay=wd, max_step=self.config['num_epochs']*len(self.train_dl), nesterov=nesterov)
        lr_list = self.config['optimizer']['lr']
        param_groups = self.model.get_groups()
        optim_params = [{'params': group, 'lr': lr_value} for group, lr_value in zip(param_groups, lr_list)]

        self.optimizer = torch.optim.Adam(params=optim_params, weight_decay=self.config['optimizer']['weight_decay'])

        if self.config['model']['load_weights']:
            # self.optimizer.load_state_dict(torch.load(self.config['model']['model_path'])['optimizer'])
            pass

    def get_losses(self, features, re_features, logits, tiled_logits, labels):
        class_loss = self.class_loss_fn(logits, labels).mean()

        p_class_loss = self.class_loss_fn(self.gap_fn(re_features),
                                          labels).mean() if 'pcl' in self.loss_option else torch.zeros(1).cuda()
        re_loss = (self.re_loss_fn(features, re_features) * labels.unsqueeze(2).unsqueeze(
            3)).mean() if 're' in self.loss_option else torch.zeros(1).cuda()
        conf_loss = shannon_entropy_loss(tiled_logits) if 'conf' in self.loss_option else torch.zeros(1).cuda()
        return class_loss, p_class_loss, re_loss, conf_loss

    def run_epoch(self, metric_lst, loss_meter, data_loader, train=True, break_after=201):

        data_iterator = Iterator(data_loader)
        iteration_num = len(data_loader)

        status_bar = tqdm.tqdm(total=iteration_num)

        torch.set_grad_enabled(train)

        if train:
            self.model.train()
            if self.segmentation:
                for mod in [self.model.model.encoder.conv_stem, self.model.model.encoder.bn1, *self.model.model.encoder.blocks[:3], self.model.classifier]:
                    mod.eval()

        else:
            self.model.eval()
            for metric in metric_lst:
                metric.reset()
        # nlst = []
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad and name.startswith('model.encoder'):
        #         nnn = name.replace('model.encoder.', '')
        #         nnn = nnn[:min(8, len(nnn))]
        #         nlst.append(nnn)
        # for s in set(nlst):
        #     print(s)

        for iter in range(iteration_num):
            with torch.cuda.amp.autocast(enabled=self.use_amp):

                images, labels, ids, cell_masks, cell_masks_ss, images2 = data_iterator.get()
                # print(labels)
                single_label = (labels.sum(-1) == 1)
                images, labels = images.cuda(), labels.cuda()
                cell_masks = cell_masks.cuda()

                if self.segmentation:
                    cell_masks_ss = cell_masks_ss.cuda()
                # if not self.config['pretrain']:
                multiplier = 2 if (self.selfsupervision and train) else 1
                if self.selfsupervision and train:
                    images2 = images2.cuda()
                    images = torch.cat([images, images2])
                tiled_images = tile_features(images, self.config['args']['num_pieces'])
                tiled_logits, tiled_features, tiled_masks = self.model(tiled_images, with_cam=True)
                re_features = merge_features(tiled_features, self.config['args']['num_pieces'],
                                                 self.config['batch_size']*multiplier)

                logits, features, mask = self.model(images, with_cam=True, no_decoder=not self.segmentation)
                labels = torch.cat([labels]*multiplier)
                class_loss, p_class_loss, re_loss, conf_loss = self.get_losses(features,
                    (torch.cat([re_features[self.bs:], re_features[:self.bs]]) if (self.selfsupervision and train) else re_features),
                    logits, tiled_logits,labels)


                if self.segmentation and train:
                    A1 = mask
                    A2 = cell_masks_ss
                    segm_loss = 0
                    ssize = A2.size(1)
                    for i in range(ssize):
                        segm_loss += L.lovasz_hinge(A1[:, i], A2[:, i])
                    segm_loss /= ssize
                else:
                    segm_loss = torch.zeros(1).cuda()

                features += re_features
                features /= 2

                if "lovasz" in self.loss_option:
                    features = resize_for_tensors(features, cell_masks.shape[1:])
                    tru = (cell_masks > 0).type(torch.uint8)
                    out = features.amax(1)
                    lovasz_loss = L.lovasz_hinge(out, tru)
                else:
                    lovasz_loss = torch.zeros(1).cuda()
                if train:
                    tr_it = self.epoch * len(self.train_dl) + iter
                    max_it = self.config['num_epochs'] * len(self.train_dl)
                    self.alpha = min(
                        self.config['args']['alpha'] * tr_it / (max_it * self.config['args']['alpha_schedule']),
                        self.config['args']['alpha'])

                loss = class_loss + p_class_loss + (self.alpha * re_loss + conf_loss) + self.alpha * 0.01 * (
                    segm_loss) + self.alpha * 0.005 * lovasz_loss
                status_bar.update()
                status_bar.set_postfix(loss=class_loss.item())
                # status_bar.set_postfix(loss=class_loss.item())

            if train:

                self.scaler.scale(loss / self.iters2acc).backward()
                if (iter + 1) % self.iters2acc == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            loss_dict = {
                'loss': loss.item(),
                'class_loss': class_loss.item(),
                'p_class_loss': p_class_loss.item(),
                're_loss': re_loss.item(),
                'conf_loss': conf_loss.item(),
                'lovasz': lovasz_loss.item(),
                'alpha': self.alpha,
                'l1_segmentation': segm_loss.item()
            }

            loss_meter.add(loss_dict)
            ost = ((iter + 1) % int(len(self.train_dl) * self.config['args']['print_ratio']))
            west = ((iter + 1) // int(len(self.train_dl) * self.config['args']['print_ratio']))

            if not train:
                probs = self.sigmoid(logits).cpu().detach().numpy()
                labels = labels.int().cpu().detach().numpy()

                metric_lst[0].add(probs, labels)
                if single_label.sum().item() and not self.config['pretrain']:
                    metric_lst[1].add(cams=features[single_label].detach(),
                                      cell_masks=cell_masks[single_label].detach(), labels=labels[single_label])

                    # metric_lst[2].add(cams=features[single_label].detach(), cell_masks=nuclei_masks[single_label].detach(), labels=labels[single_label])
            elif not ost:

                tr_itt = int((self.epoch / self.config['args']['print_ratio'])) + west
                self.log_results(tr_itt, loss_meter, train)
                if break_after == west:
                    return None

        status_bar.close()

        if not train:

            metric_score_lst = [metric.get() for metric in metric_lst]

            # tr_itt = (self.epoch+1)*len(self.train_dl)
            it = int((self.epoch / self.config['args']['print_ratio']))
            metric_lst[0].write_to_tensorboard(self.writer, it, prefix='VAL_AP/')  # TODO: change for each metric
            if not self.config['pretrain']:
                metric_lst[1].write_to_tensorboard(self.writer, it,
                                                   prefix='VAL_AP_CLASS/')  # TODO: change for each metric
            # metric_lst[2].write_to_tensorboard(self.writer, it, prefix='VAL_AP_NUCLEI/') #TODO: change for each metric

            self.log_results(it, loss_meter, train)
            return metric_score_lst

    def log_results(self, iteration, meter, train=True):

        loss, class_loss, p_class_loss, re_loss, conf_loss, alpha, segm_loss, lovasz_loss = meter.get(
            clear=True)

        learning_rate = float(get_learning_rate_from_optimizer(self.optimizer))

        prefix = 'Train' if train else 'Val'
        self.writer.add_scalar(f'{prefix}/loss', loss, iteration)
        self.writer.add_scalar(f'{prefix}/class_loss', class_loss, iteration)
        self.writer.add_scalar(f'{prefix}/p_class_loss', p_class_loss, iteration)
        self.writer.add_scalar(f'{prefix}/re_loss', re_loss, iteration)
        self.writer.add_scalar(f'{prefix}/conf_loss', conf_loss, iteration)
        self.writer.add_scalar(f'{prefix}/learning_rate', learning_rate, iteration)
        self.writer.add_scalar(f'{prefix}/alpha', alpha, iteration)
        if self.segmentation:
            self.writer.add_scalar(f'{prefix}/segm_l1', segm_loss, iteration)
        if "lovasz" in self.loss_option:
            self.writer.add_scalar(f'{prefix}/lovasz', lovasz_loss, iteration)
