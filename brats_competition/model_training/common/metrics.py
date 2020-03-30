import torch
import itertools

__EPSILON = 1e-6


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


def get_metric(metric_name, num_classes, device):
    if metric_name == "iou":
        return IoUMetric(num_classes, device)
    else:
        raise ValueError(f"Metric [{metric_name}] not recognized.")
