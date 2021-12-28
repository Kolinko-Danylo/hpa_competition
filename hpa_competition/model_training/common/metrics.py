import torch
import itertools
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix as cm
from hpa_competition.PuzzleCAM.tools.ai.torch_utils import resize_for_tensors
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
__EPSILON = 1e-6
from torch import nn

LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles",
             "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments",
             "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol",
             "Vesicles", "Negative"]

class Metric:
    def add(self, y_pred, y_true):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def write_to_tensorboard(self, writer, epoch):
        raise NotImplementedError


class IoUMetric(Metric):
    NAME = 'meanIoU'

    def __init__(self, classes, device, ignore_value=255):
        self.classes = classes
        self.device = device
        self.ignore_value = ignore_value
        self.reset()

    def add(self, output, target):
        output = torch.argmax(output, dim=1).view(-1)
        target = target.view(-1)
        valid_idx = target != self.ignore_value
        target[~valid_idx] = 0

        for i, j in itertools.product(torch.unique(target), torch.unique(output)):
            self.conf_matrix[i, j] += torch.sum((target[valid_idx] == i) & (output[valid_idx] == j))

    def get(self):
        conf_matrix = self.conf_matrix.float()
        true_positives = torch.diagonal(conf_matrix)
        false_positives = torch.sum(conf_matrix, 0) - true_positives
        false_negatives = torch.sum(conf_matrix, 1) - true_positives

        iou_per_class = true_positives / (true_positives + false_negatives + false_positives)
        self.score = torch.mean(iou_per_class).item()
        return self.score

    def reset(self):
        self.conf_matrix = torch.zeros((self.classes, self.classes), dtype=torch.int64).to(self.device)

    def write_to_tensorboard(self, writer, epoch):
        writer.add_scalar(self.NAME, self.score, epoch)


class DiceMetric(Metric):
    NAME = 'meanDiceScore'

    def __init__(self, classes, device, ignore_value=255):
        self.classes = classes
        self.device = device
        self.ignore_value = ignore_value
        self.reset()

    def add(self, output, target):
        output = torch.argmax(output, dim=1).view(-1)
        target = target.view(-1)
        valid_idx = target != self.ignore_value
        target[~valid_idx] = 0

        for i, j in itertools.product(torch.unique(target), torch.unique(output)):
            self.conf_matrix[i, j] += torch.sum((target[valid_idx] == i) & (output[valid_idx] == j))

    def get(self):
        conf_matrix = self.conf_matrix.float()
        true_positives = torch.diagonal(conf_matrix)
        false_positives = torch.sum(conf_matrix, 0) - true_positives
        false_negatives = torch.sum(conf_matrix, 1) - true_positives

        dice_per_class = 2 * true_positives / (2 * true_positives + false_negatives + false_positives)

        self.score = torch.mean(dice_per_class).item()
        return self.score

    def reset(self):
        self.conf_matrix = torch.zeros((self.classes, self.classes), dtype=torch.int64).to(self.device)

    def write_to_tensorboard(self, writer, epoch):
        writer.add_scalar(self.NAME, self.score, epoch)

class AveragePrecision(Metric):
        NAME = 'AveragePrecision'

        def __init__(self, classes, device):
            self.classes = classes
            self.device = device
            self.reset()

        def add(self, y_pred, y_true):
            y_pred = y_pred
            y_true = y_true

            assert not np.isnan(y_pred).any(), f'{np.isnan(y_pred).sum()}'
            assert not np.isinf(y_pred).any()


            self.pred_stacked = np.vstack((self.pred_stacked,  y_pred))
            self.true_stacked = np.vstack((self.true_stacked,  y_true))

        def get(self):

            self.score_lst = [average_precision_score(self.true_stacked[:, i], self.pred_stacked[:, i]) for i in range(self.classes)]
            score_dict = {str(i): self.score_lst[i] for i in range(self.classes)}


            self.score = np.nanmean(np.array(self.score_lst))
            return self.score

        def reset(self):
            self.pred_stacked = np.empty(shape=(0, self.classes))
            self.true_stacked = np.empty(shape=(0, self.classes))
            # self.score=None
            # self.score_lst=None

        def write_to_tensorboard(self, writer, epoch, prefix=''):
            writer.add_scalar(self.NAME, self.score, epoch)
            for i in range(self.classes):
                sc = self.score_lst[i]
                sc = sc if sc == sc else 0
                # print(sc)
                #TODO: fix in distributing val and train
                writer.add_scalar(f'{prefix}{self.NAME}_CLASS_{i}_{LBL_NAMES[i]}', sc, epoch)


class AveragePrecision_SingleClass_Instance(Metric):
    NAME = 'AveragePrecision_SingleClass_Instance'

    def __init__(self, classes, device, masktype):
        self.classes = classes
        self.device = device
        self.masktype=masktype
        self.sigm = nn.Sigmoid()
        self.reset()
        self.relu = nn.ReLU()

    def add(self, cams, cell_masks, labels):
        cell_masks = cell_masks.cuda()
        CAMS = resize_for_tensors(cams, cell_masks.shape[1:]) #B, C, W, H
        for i in range(CAMS.size(0)):

            cur_cams = CAMS[i].unsqueeze(0)
            cur_cell_masks = cell_masks[i]
            _, counts = torch.unique(cur_cell_masks, return_counts=True)
            non_z = counts[1:].unsqueeze(-1)

            cur_sz = non_z.size(0)

            label_c = np.tile(labels[i].copy(), (cur_sz, 1))

            cur_cell_masks = cur_cell_masks.unsqueeze(0).expand(cur_sz, -1, -1, -1)
            inst_ = torch.arange(1, cur_sz+1, device=torch.device('cuda'))

            a_cur_cell_masks = (cur_cell_masks == inst_.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            # print(torch.isinf(a_cur_cell_masks).sum())
            divs = 100
            result = (cur_cams * (a_cur_cell_masks)) / divs
            # print(torch.isinf(result).sum())

            summa = result.sum(dim=(2, 3))
            # assert not torch.isinf(summa).any()

            y_pred = divs * torch.div(summa, non_z)
            class_pred = cur_cams.squeeze(0).mean(dim=(1, 2))





            assert not torch.isnan(y_pred + class_pred).any(), f'{torch.isnan(y_pred).sum()},\n {torch.isnan(class_pred).sum()}'
            assert not torch.isinf(y_pred + class_pred).any()

            self.pred_stacked = [np.vstack(    (self.pred_stacked[idx],    self.sigm(((1 - weight/100)*y_pred  +  (weight/100)*class_pred )).cpu().numpy())   )    for idx, weight in enumerate(range(0, 110, 10))]

            self.true_stacked = np.vstack((self.true_stacked, label_c))


    def get(self):
        self.score_lst = []
        for weight_idx in range(11):
            # print(weight_idx)
            # print((self.pred_stacked[10].shape))
            temp = [average_precision_score(self.true_stacked[:, i], self.pred_stacked[weight_idx][:, i]) for i in
                          range(self.classes)]
            self.score_lst.append(temp)

        # score_dict = {str(i): self.score_lst[i] for i in range(self.classes)}

        self.scores = [np.nanmean(np.array(self.score_lst[weight_idx])) for weight_idx in range(11)]
        return self.scores[0]

    def reset(self):
        self.pred_stacked = [np.empty(shape=(0, self.classes)) for i in range(11)]

        self.true_stacked = np.empty(shape=(0, self.classes))

        # self.score=None
        # self.score_lst=None

    def write_to_tensorboard(self, writer, epoch, prefix=''):
        nm=self.NAME+('nuclei' if self.masktype=='nuclei' else '')
        writer.add_scalar(nm, self.scores[0], epoch)

        for i, weight in enumerate(range(0, 110, 10)):
            nm=('nuclei' if self.masktype=='nuclei' else 'cell')
            writer.add_scalar(f'WeightedAP{nm}/image_level_w_{weight}%', self.scores[i], epoch)

        for i in range(self.classes):
            sc = self.score_lst[0][i]
            sc = sc if sc == sc else 0
            # print(sc)
            # TODO: fix in distributing val and train
            writer.add_scalar(f'{prefix}{self.NAME}_CLASS_{i}_{LBL_NAMES[i]}', sc, epoch)


class IoUMetricCell(Metric):
    NAME = 'meanIoUcell'

    def __init__(self, classes, device, thresh_range, ignore_value=255 ):
        self.relu = nn.ReLU()
        self.classes = classes
        self.device = device
        self.ignore_value = ignore_value
        self.thresh_range = thresh_range
        # self.background = background
        self.reset()

    def add(self, output, target):
        output = self.relu(output)
        output = resize_for_tensors(output, (target.size(-2), target.size(-1)))
        output -=  torch.amin(output, dim=(1,2,3), keepdim=True)
        output /= (torch.amax(output, dim=(1,2,3), keepdim=True))
        # output -= output.min(1, keepdim=True)[0]
        # print(output)
        # output /= output.max(1, keepdim=True)[0]

        output = torch.amax(output, dim=1).squeeze()

        target = (target > 0).type(torch.IntTensor)
        target = target.view(-1)
        valid_idx = target != self.ignore_value

        target[~valid_idx] = 0
        for st, thresh in enumerate(self.thresh_range):
            cur_output =  (output > thresh).type(torch.IntTensor).view(-1)
            for i, j in itertools.product(torch.unique(target), torch.unique(cur_output)):
                self.conf_matrix_lst[st][i, j] += torch.sum((target[valid_idx] == i) & (cur_output[valid_idx] == j))

    def get(self):
        return [self.get_mat(self.conf_matrix_lst[i]) for i in range(len(self.thresh_range))]

    def get_mat(self, mat):
        conf_matrix = mat.float()
        true_positives = torch.diagonal(conf_matrix)
        false_positives = torch.sum(conf_matrix, 0) - true_positives
        false_negatives = torch.sum(conf_matrix, 1) - true_positives

        iou_per_class = true_positives / (true_positives + false_negatives + false_positives)
        # self.score = torch.mean(iou_per_class).item()
        return iou_per_class

    def reset(self):
        self.conf_matrix_lst = [torch.zeros((self.classes, self.classes), dtype=torch.int64).to(self.device) for i in range(len(self.thresh_range))]




def get_metric(metric_name, num_classes, device):
    if metric_name == 'find_thresh':
        return IoUMetricCell(num_classes, device)
    if metric_name == 'AP':
        return AveragePrecision(num_classes, device)
    if metric_name == "iou":
        return IoUMetric(num_classes, device)
    elif metric_name == "dice":
        return DiceMetric(num_classes, device)
    else:
        raise ValueError(f"Metric [{metric_name}] not recognized.")

