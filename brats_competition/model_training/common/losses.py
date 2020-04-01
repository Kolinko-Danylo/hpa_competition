import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    """
    Implementation of mean soft-IoU loss for semantic segmentation
    """
    __EPSILON = 1e-6

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
        y_pred: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        y_true: a tensor of shape [B, H, W].
        Returns:
        float: soft-iou loss.
        """
        num_classes = y_pred.shape[1]

        y_true_dummy = torch.eye(num_classes)[y_true.squeeze(1)]
        y_true_dummy = y_true_dummy.permute(0, 3, 1, 2).to(y_true.device)

        y_pred_proba = F.softmax(y_pred, dim=1)

        intersection = torch.sum(y_pred_proba * y_true_dummy, dim=(2, 3))
        union = torch.sum(y_pred_proba ** 2 + y_true_dummy ** 2, dim=(2, 3)) - intersection
        iou_loss = ((intersection + self.__EPSILON) / (union + self.__EPSILON))

        return 1 - iou_loss.mean()


def get_loss(loss_config):
    loss_name = loss_config['name']
    if loss_name == 'categorical_cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'mean_iou':
        return IoULoss()
    else:
        raise ValueError(f"Loss [{loss_name}] not recognized.")
