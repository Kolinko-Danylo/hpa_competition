import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from hpa_competition.model_training.common.metrics import AveragePrecision, AveragePrecision_SingleClass_Instance, IoUMetricCell
from hpa_competition.model_training.common.losses import get_loss
import yaml

from hpa_competition.PuzzleCAM.tools.ai.torch_utils import make_cam, L1_Loss, L2_Loss, shannon_entropy_loss, load_model, save_model, get_learning_rate_from_optimizer, resize_for_tensors
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
    return -(target * torch.log(predicted + e) + ((1 - target) * ( torch.log(1 - predicted + e))))
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

        loss_lst = ['loss', 'class_loss', 'p_class_loss', 're_loss', 'conf_loss', 'alpha', 'pos_loss', 'neg_loss']
        if self.segmentation:
            loss_lst.append('l1_segmentation')

        self.train_meter = Average_Meter(loss_lst)
        self.val_meter = Average_Meter(loss_lst)
        self.writer = SummaryWriter(tensorboard_dir)
        self.img_dir = os.path.join(config['train']['path'], 'train')

        self.best_val_ap = None
        self.best_val_ap_cell = None
        self.segm_losss=CXE
        # self.best_val_ap_nuclei = None

        self.use_amp = self.config['use_amp']
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train(self):
        self._init_params()
        for ep in range(self.config['num_epochs']):


            self.epoch = ep
            self.alpha=1.0
            print(f'Epoch: {self.epoch}')
            # train_metric_lst = [self.train_ap, self.train_ap_cell, self.train_ap_nuclei]
            # train_metric_lst = [self.train_ap, self.train_ap_cell]
            # val_metric_lst = [self.val_ap, self.val_ap_cell]


            self.run_epoch(metric_lst=self.train_metric_lst, loss_meter=self.train_meter,
                                                        data_loader=self.train_dl, train=True, break_after=11)
            # val_metric_lst = [self.val_ap, self.val_ap_cell, self.val_ap_nuclei]

            ap_score_lst = self.run_epoch(metric_lst=self.val_metric_lst, loss_meter=self.val_meter, data_loader=self.val_dl,
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
        if ( bst_scr is None) or (bst_scr < current):
            print(f'saving model to {path}')
            self._save_checkpoint(path)
            return current
        else: return bst_scr

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
        self.model = Classifier(self.config['model'] )
        # self.param_groups = self.model.get_parameter_groups(print_fn=None)
        self.model.cuda()
        self.gap_fn = self.model.global_average_pooling_2d
        self.params = self.model.get_parameter_groups(print_fn=None)

        try:
            use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
            print(f"use_gpu:{use_gpu}\n\n")
        except KeyError:
            use_gpu = '0'
        the_number_of_gpu = len(use_gpu.split(','))
        self.the_number_of_gpu = the_number_of_gpu

        if the_number_of_gpu > 1:
            self.log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
            self.model = nn.DataParallel(self.model)

        if self.config['model']['load_weights']:
            self.model.load_state_dict(torch.load(self.config['model']['model_path'])['model'])
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
            self.val_ap_cell = AveragePrecision_SingleClass_Instance(classes=self.config['model']['classes'], device=torch.device('cuda'), masktype='cell')
            self.train_ap_cell = AveragePrecision_SingleClass_Instance(classes=self.config['model']['classes'], device=torch.device('cuda'), masktype='cell')
            self.val_metric_lst.append(self.val_ap_cell)
            self.train_metric_lst.append(self.train_ap_cell)


            # self.val_ap_nuclei = AveragePrecision_SingleClass_Instance(classes=self.config['model']['classes'], device=torch.device('cuda'), masktype='nuclei')
        # self.train_ap_nuclei = AveragePrecision_SingleClass_Instance(classes=self.config['model']['classes'], device=torch.device('cuda'), masktype='nuclei')
        self.loss_option = self.config['args']['loss_option'].split('_')

    def _init_optimizer(self):
        param_groups = self.params
        lr = self.config['args']['lr']
        wd = self.config['args']['wd']
        nesterov = self.config['args']['nesterov']

        # self.optimizer = PolyOptimizer([
        #     {'params': param_groups[0], 'lr': lr, 'weight_decay': wd},
        #     {'params': param_groups[1], 'lr': 2 * lr, 'weight_decay': 0},
        #     {'params': param_groups[2], 'lr': 10 * lr, 'weight_decay': wd},
        #     {'params': param_groups[3], 'lr': 20 * lr, 'weight_decay': 0},
        # ], lr=lr, momentum=0.9, weight_decay=wd, max_step=self.config['num_epochs']*len(self.train_dl), nesterov=nesterov)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr = self.config['optimizer']['lr'], weight_decay=self.config['optimizer']['weight_decay'])
        if self.config['model']['load_weights']:
            self.optimizer.load_state_dict(torch.load(self.config['model']['model_path'])['optimizer'])

    def run_epoch(self, metric_lst, loss_meter, data_loader, train=True, break_after=11):
        data_iterator = Iterator(data_loader)
        iteration_num = len(data_loader)

        status_bar = tqdm.tqdm(total=iteration_num)

        torch.set_grad_enabled(train)

        if train:
            self.model.train()
        else:
            self.model.eval()
            for metric in metric_lst:
                metric.reset()

        pos_loss = torch.zeros(1).cuda()
        neg_loss = torch.zeros(1).cuda()
        for iter in range(iteration_num):
            with torch.cuda.amp.autocast(enabled=self.use_amp):

                images, labels, ids, cell_masks, cell_masks_ss = data_iterator.get()
                # print(labels)
                single_label = (labels.sum(-1) == 1)
                images, labels = images.cuda(), labels.cuda()
                cell_masks_ss = cell_masks_ss.cuda()
                if not self.config['pretrain']:
                    cell_masks = cell_masks.cuda()
                # cell_masks_ss
                # nuclei_masks = nuclei_masks.cuda()




                    # hard_code = (16, 16)
                    # cell_masks_super = resize_for_tensors(cell_masks_super, hard_code, mode='nearest', align_corners=None)

                logits, features, mask = self.model(images, with_cam=True)
                # print(features.shape)
                # print(mask.shape, cell_masks_ss.shape)
                if self.segmentation:
                    A1 = mask
                    A2 = cell_masks_ss/255

                    segm_loss = self.segm_losss(A1, A2)
                    # print(A2.min())

                    segm_loss = segm_loss.mean()


                # print(features.shape)
                # image_paths = build_image_names(ids, self.img_dir)[-1]
                # cell_segmentations = self.segmentator.pred_cells(image_paths)
                # cell_segmentations = [cell_s[..., 2] for cell_s in cell_segmentations]

                if 'pos' in self.loss_option:
                    cell_masks_super = (cell_masks.unsqueeze(1) != 0).to(torch.float32) if self.mask_supervised else None
                    hard_core = (128, 128)
                    cell_masks_super = resize_for_tensors(cell_masks_super, hard_core)
                    # u_features = self.sigmoid(unnorm_features(model=self.model, features=features))
                    u_features = self.sigmoid(features)


                    u_features = resize_for_tensors(u_features, hard_core)

                    pos_loss = self.pos_loss(u_features, cell_masks_super, labels)


                # print(logits.shape,  features.shape)
                # print(images.shape)
                tiled_images = tile_features(images, self.config['args']['num_pieces'])

                tiled_logits, tiled_features, tiled_masks = self.model(tiled_images, with_cam=True)
                # print(tiled_images.shape)
                # print(tiled_features.shape)
                re_features = merge_features(tiled_features, self.config['args']['num_pieces'], self.config['batch_size'])
                # print(re_features.shape)
                class_loss = self.class_loss_fn(logits, labels).mean()

                if 'pcl' in self.loss_option:
                    p_class_loss = self.class_loss_fn(self.gap_fn(re_features), labels).mean()
                else:
                    p_class_loss = torch.zeros(1).cuda()

                if 're' in self.loss_option:
                    if self.config['args']['re_loss_option'] == 'masking':
                        class_mask = labels.unsqueeze(2).unsqueeze(3)
                        # print(features.shape, re_features.shape)

                        re_loss = self.re_loss_fn(features, re_features) * class_mask
                        re_loss = re_loss.mean()
                    elif self.config['args']['re_loss_option'] == 'selection':
                        re_loss = 0.
                        for b_index in range(labels.size()[0]):
                            class_indices = labels[b_index].nonzero(as_tuple=True)
                            selected_features = features[b_index][class_indices]
                            selected_re_features = re_features[b_index][class_indices]

                            re_loss_per_feature = self.re_loss_fn(selected_features, selected_re_features).mean()
                            re_loss += re_loss_per_feature
                        re_loss /= labels.size()[0]
                    else:
                        re_loss = self.re_loss_fn(features, re_features).mean()
                else:
                    re_loss = torch.zeros(1).cuda()

                if 'conf' in self.loss_option:
                    conf_loss = shannon_entropy_loss(tiled_logits)
                else:
                    conf_loss = torch.zeros(1).cuda()

                if train:
                    tr_it = self.epoch*len(self.train_dl) + iter
                    max_it = self.config['num_epochs']*len(self.train_dl)
                    self.alpha = min(self.config['args']['alpha'] * tr_it / (max_it * self.config['args']['alpha_schedule']), self.config['args']['alpha'])

                loss = class_loss + p_class_loss + self.alpha * re_loss + conf_loss +  (self.alpha/4 * pos_loss) + 2*self.alpha*segm_loss
                status_bar.update()
                status_bar.set_postfix(loss=class_loss.item())

            if train:

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            loss_dict = {
                'loss': loss.item(),
                'class_loss': class_loss.item(),
                'p_class_loss': p_class_loss.item(),
                're_loss': re_loss.item(),
                'conf_loss': conf_loss.item(),
                'pos_loss': pos_loss.item(),
                'neg_loss': neg_loss.item(),
                'alpha': self.alpha,
            }
            if self.segmentation:
                loss_dict['l1_segmentation'] = segm_loss.item()
            # print(loss_dict)

            loss_meter.add(loss_dict)
            ost = ( (iter + 1) % int(len(self.train_dl) * self.config['args']['print_ratio']))
            west = ( (iter + 1) // int(len(self.train_dl) * self.config['args']['print_ratio']))

            if not train:
                probs = self.sigmoid(logits).cpu().detach().numpy()
                labels = labels.int().cpu().detach().numpy()
                metric_lst[0].add(probs, labels)
                if single_label.sum().item() and not self.config['pretrain']:

                    metric_lst[1].add(cams=features[single_label].detach(), cell_masks=cell_masks[single_label].detach(), labels=labels[single_label])

                    # metric_lst[2].add(cams=features[single_label].detach(), cell_masks=nuclei_masks[single_label].detach(), labels=labels[single_label])
            elif not ost:

                tr_itt = int((self.epoch/ self.config['args']['print_ratio'])) + west
                self.log_results(tr_itt , loss_meter, train)
                if break_after == west:
                    return None

        status_bar.close()

        if not train:

            metric_score_lst = [metric.get() for metric in metric_lst]

            # tr_itt = (self.epoch+1)*len(self.train_dl)
            it = int((self.epoch/ self.config['args']['print_ratio']))
            metric_lst[0].write_to_tensorboard(self.writer, it, prefix='VAL_AP/') #TODO: change for each metric
            if not self.config['pretrain']:
                metric_lst[1].write_to_tensorboard(self.writer, it, prefix='VAL_AP_CLASS/') #TODO: change for each metric
            # metric_lst[2].write_to_tensorboard(self.writer, it, prefix='VAL_AP_NUCLEI/') #TODO: change for each metric



            self.log_results(it, loss_meter, train)
            return metric_score_lst








    def log_results(self, iteration, meter, train=True):
        if not self.segmentation:
            loss, class_loss, p_class_loss, re_loss, conf_loss, alpha, pos_loss, neg_loss = meter.get(clear=True)
        else:
            loss, class_loss, p_class_loss, re_loss, conf_loss, alpha, pos_loss, neg_loss, segm_loss = meter.get(clear=True)
        learning_rate = float(get_learning_rate_from_optimizer(self.optimizer))



        prefix = 'Train' if train else 'Val'
        self.writer.add_scalar(f'{prefix}/loss', loss, iteration)
        self.writer.add_scalar(f'{prefix}/class_loss', class_loss, iteration)
        self.writer.add_scalar(f'{prefix}/p_class_loss', p_class_loss, iteration)
        self.writer.add_scalar(f'{prefix}/re_loss', re_loss, iteration)
        self.writer.add_scalar(f'{prefix}/neg_loss', neg_loss, iteration)
        self.writer.add_scalar(f'{prefix}/pos_loss', pos_loss, iteration)

        self.writer.add_scalar(f'{prefix}/conf_loss', conf_loss, iteration)
        self.writer.add_scalar(f'{prefix}/learning_rate', learning_rate, iteration)
        self.writer.add_scalar(f'{prefix}/alpha', alpha, iteration)
        if self.segmentation:
            self.writer.add_scalar(f'{prefix}/segm_l1', segm_loss, iteration)

        # if train:
        #     self.log_func(f'[i] \
        #         iteration={iteration}, \
        #         learning_rate={learning_rate:.4f}, \
        #         alpha={self.alpha:.2f}, \
        #         loss={loss:.4f}, \
        #         class_loss={class_loss:.4f}, \
        #         p_class_loss={p_class_loss:.4f}, \
        #         re_loss={re_loss:.4f}, \
        #         pos_loss={pos_loss:.4f}, \
        #         neg_loss={neg_loss:.4f}, \
        #         conf_loss={conf_loss:.4f},')


    def neg_loss(self, features, cell_segmentations, labels):
        # print(features.shape)
        # print(cell_segmentations.shape)

        segms = features
        segms = segms.flatten(end_dim=1)

        neg_labels = (labels == 0).flatten()
        ret = segms[neg_labels].mean()
        return ret




    def pos_loss(self, features, cell_segmentations, labels):
        segms = features * (1 - cell_segmentations)
        segms = segms.flatten(end_dim=1)


        pos_labels = (labels != 0).flatten()
        return segms[pos_labels].mean()
        # total = 0
        #
        # for i in range(segms.size(0)):
        #     total += segms[i, pos_labels[i]].mean()/segms.size(0)
        # return total




def rescale_segm(cell_segmentations, shape):
    nrew = []
    for arr in cell_segmentations:
        cur_inp = torch.from_numpy(arr[..., 2]).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
        cur = resize_for_tensors(cur_inp, shape).squeeze(1).squeeze(1)
        nrew.append(cur)

    cell_segmentations = torch.stack(nrew)
    cell_segmentations /= 255
    return cell_segmentations.cuda()








# class OtherTrainer:
#     def __init__(self, config, train_dl, val_dl):
#         self.config = config
#         self.train_dl = train_dl
#         self.val_dl = val_dl
#
#         self.mask_supervised = config['train']['load_mask'] and not config['train']['cell_input']
#         self.rrange = np.arange(0.05, 1, 0.05)
#         self.ious = [IoUMetricCell(classes=2, device=torch.device('cuda'), thresh_range=self.rrange)]
#
#
#         self.sigmoid = nn.Sigmoid()
#
#     def infer(self):
#         self._init_params()
#         #add IoU
#         ap_score_lst = self.run_epoch(metric_lst=self.ious, data_loader=self.train_dl,
#                                       train=False)
#         ap_score_lst = ap_score_lst[0]
#         print(ap_score_lst)
#         for step, i in enumerate(self.rrange):
#             print(f'thresh {i}: {ap_score_lst[step]}')
#
#     def _init_params(self):
#         self.model = Classifier(self.config['model'] )
#         # self.param_groups = self.model.get_parameter_groups(print_fn=None)
#         self.model.cuda()
#         self.gap_fn = self.model.global_average_pooling_2d
#         self.params = self.model.get_parameter_groups(print_fn=None)
#         self.use_amp = True
#         try:
#             use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
#             print(f"use_gpu:{use_gpu}\n\n")
#         except KeyError:
#             use_gpu = '0'
#         the_number_of_gpu = len(use_gpu.split(','))
#         self.the_number_of_gpu = the_number_of_gpu
#
#         if the_number_of_gpu > 1:
#             self.log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
#             self.model = nn.DataParallel(self.model)
#
#         if self.config['model']['load_weights']:
#             load_model(self.model, self.config['model']['model_path'], parallel=the_number_of_gpu > 1)
#
#
#     def _init_optimizer(self):
#         self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr = self.config['optimizer']['lr'], weight_decay=self.config['optimizer']['weight_decay'])
#
#     def run_epoch(self, metric_lst, data_loader, train=True):
#         data_iterator = Iterator(data_loader)
#         iteration_num = len(data_loader)
#
#         status_bar = tqdm.tqdm(total=iteration_num)
#
#         torch.set_grad_enabled(train)
#
#         if train:
#             self.model.train()
#         else:
#             self.model.eval()
#             for metric in metric_lst:
#                 metric.reset()
#
#         for iter in range(iteration_num):
#             with torch.cuda.amp.autocast(enabled=self.use_amp):
#
#                 images, labels, ids, cell_masks, cell_masks_ss = data_iterator.get()
#                 images, labels = images.cuda(), labels.cuda()
#                 cell_masks = cell_masks.cuda()
#
#                 logits, features = self.model(images, with_cam=True)
#                 status_bar.update()
#
#             if not train:
#                 for i in range(len(self.ious)):
#                     metric_lst[i].add(features, cell_masks)
#             ap_score_lst = [metric.get() for metric in metric_lst]
#             ap_score_lst = ap_score_lst[0]
#             for step, i in enumerate(self.rrange):
#                 print(f'thresh {i}: {ap_score_lst[step]}')
#
#         status_bar.close()
#
#         if not train:
#
#             metric_score_lst = [metric.get() for metric in metric_lst]
#
#             return metric_score_lst
