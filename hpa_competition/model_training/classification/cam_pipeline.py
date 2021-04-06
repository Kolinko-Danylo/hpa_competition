import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from hpa_competition.model_training.common.metrics import AveragePrecision


from hpa_competition.PuzzleCAM.tools.ai.torch_utils import make_cam, L1_Loss, L2_Loss, shannon_entropy_loss, load_model, save_model, get_learning_rate_from_optimizer
from hpa_competition.PuzzleCAM.core.puzzle_utils import tile_features, merge_features
from hpa_competition.PuzzleCAM.core.datasets import Iterator
from hpa_competition.PuzzleCAM.core.networks import Classifier
from hpa_competition.PuzzleCAM.tools.ai.optim_utils import PolyOptimizer

from hpa_competition.PuzzleCAM.tools.general.io_utils import create_directory
from hpa_competition.PuzzleCAM.tools.ai.log_utils import log_print, Average_Meter
from hpa_competition.PuzzleCAM.tools.general.time_utils import Timer

from datetime import datetime
import os


class CAMTrainer:
    def __init__(self, config, train_dl, val_dl):
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl

        log_path = config['log_path']

        log_dir = os.path.join(log_path, 'logs')
        data_dir = os.path.join(log_path, 'data')
        model_dir = os.path.join(log_path, 'models')


        tag_str = f'{config["model"]["arch"]}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'


        self.model_path = os.path.join(model_dir, f'{tag_str}.pth')
        tensorboard_dir = create_directory(os.path.join(log_path, f'tensorboards/{tag_str}/'))
        log_txt_path = log_dir + f'{tag_str}.txt'
        self.log_func = lambda string='': log_print(string, log_txt_path)
        self.log_func('[i] {}'.format(tag_str))
        # load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
        # save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
        self.sigmoid = nn.Sigmoid()

        self.train_timer = Timer()
        self.eval_timer = Timer()

        self.train_meter = Average_Meter(['loss', 'class_loss', 'p_class_loss', 're_loss', 'conf_loss', 'alpha'])
        self.val_meter = Average_Meter(['loss', 'class_loss', 'p_class_loss', 're_loss', 'conf_loss', 'alpha'])
        self.writer = SummaryWriter(tensorboard_dir)

        self.best_val_ap = None

    def train(self):
        self._init_params()
        for ep in range(self.config['num_epochs']):
            self.epoch = ep
            # self.alpha=1.0
            self.run_epoch(metric=self.train_ap, loss_meter=self.train_meter,
                                                        data_loader=self.train_dl, train=True)

            ap_score = self.run_epoch(metric=self.val_ap, loss_meter=self.train_meter, data_loader=self.val_dl,
                                      train=False)


            if (self.best_val_ap is None) or (self.best_val_ap < ap_score):
                print(f'saving model to {self.model_path}')
                save_model(self.model, self.model_path, parallel=self.the_number_of_gpu > 1)
                self.best_val_ap = ap_score
        self.writer.close()



    def _init_params(self):
        self.model = Classifier(self.config['model']['arch'], num_classes=self.config['model']['classes'], mode=self.config['args']['mode'])
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
            load_model(self.model, self.config['model']['model_path'], parallel=the_number_of_gpu > 1)

        self.class_loss_fn = nn.BCEWithLogitsLoss(reduction='none').cuda()

        if self.config['args']['re_loss'] == 'L1_Loss':
            self.re_loss_fn = L1_Loss
        else:
            self.re_loss_fn = L2_Loss

        self._init_optimizer()
        self.val_ap = AveragePrecision(classes=self.config['model']['classes'], device=torch.device('cuda'))
        self.train_ap = AveragePrecision(classes=self.config['model']['classes'], device=torch.device('cuda'))

        self.loss_option = self.config['args']['loss_option'].split('_')

    def _init_optimizer(self):
        param_groups = self.params
        lr = self.config['args']['lr']
        wd = self.config['args']['wd']
        nesterov = self.config['args']['nesterov']

        self.optimizer = PolyOptimizer([
            {'params': param_groups[0], 'lr': lr, 'weight_decay': wd},
            {'params': param_groups[1], 'lr': 2 * lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10 * lr, 'weight_decay': wd},
            {'params': param_groups[3], 'lr': 20 * lr, 'weight_decay': 0},
        ], lr=lr, momentum=0.9, weight_decay=wd, max_step=self.config['num_epochs']*len(self.train_dl), nesterov=nesterov)

    def run_epoch(self, metric, loss_meter, data_loader, train=True):
        data_iterator = Iterator(data_loader)
        iteration_num = len(data_loader)
        if train:
            self.model.train()
        else:
            self.model.eval()
            metric.reset()

        for iter in range(iteration_num):

            images, labels, ids = data_iterator.get()
            # print(images.shape,  labels.shape)
            images, labels = images.cuda(), labels.cuda()

            logits, features = self.model(images, with_cam=True)
            # print(logits.shape,  features.shape)

            tiled_images = tile_features(images, self.config['args']['num_pieces'])
            tiled_logits, tiled_features = self.model(tiled_images, with_cam=True)
            re_features = merge_features(tiled_features, self.config['args']['num_pieces'], self.config['batch_size'])
            # print(re_features.shape)
            # if args.level == 'cam':
            # features = make_cam(features)
            # re_features = make_cam(re_features)

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

            loss = class_loss + p_class_loss + self.alpha * re_loss + conf_loss

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_meter.add({
                'loss': loss.item(),
                'class_loss': class_loss.item(),
                'p_class_loss': p_class_loss.item(),
                're_loss': re_loss.item(),
                'conf_loss': conf_loss.item(),
                'alpha': self.alpha,
            })

            if not train:
                metric.add(self.sigmoid(logits), labels)
            elif not ( (iter + 1) % int(len(self.train_dl) * self.config['args']['print_ratio'])):
                tr_itt = (self.epoch) * len(self.train_dl)
                self.log_results(iter + tr_itt , loss_meter, train)



        if not train:
            metric_score = metric.get()
            tr_itt = (self.epoch+1)*len(self.train_dl)
            metric.write_to_tensorboard(self.writer, tr_itt, prefix='VAL_AP/')
            self.log_results(tr_itt, loss_meter, train)
            return metric_score





    def log_results(self, iteration, meter, train=True):
        loss, class_loss, p_class_loss, re_loss, conf_loss, alpha = meter.get(clear=True)
        learning_rate = float(get_learning_rate_from_optimizer(self.optimizer))



        prefix = 'Train' if train else 'Val'
        self.writer.add_scalar(f'{prefix}/loss', loss, iteration)
        self.writer.add_scalar(f'{prefix}/class_loss', class_loss, iteration)
        self.writer.add_scalar(f'{prefix}/p_class_loss', p_class_loss, iteration)
        self.writer.add_scalar(f'{prefix}/re_loss', re_loss, iteration)
        self.writer.add_scalar(f'{prefix}/conf_loss', conf_loss, iteration)
        self.writer.add_scalar(f'{prefix}/learning_rate', learning_rate, iteration)
        self.writer.add_scalar(f'{prefix}/alpha', alpha, iteration)

        if train:
            self.log_func(f'[i] \
                iteration={iteration}, \
                learning_rate={learning_rate:.4f}, \
                alpha={self.alpha:.2f}, \
                loss={loss:.4f}, \
                class_loss={class_loss:.4f}, \
                p_class_loss={p_class_loss:.4f}, \
                re_loss={re_loss:.4f}, \
                conf_loss={conf_loss:.4f},')
