import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Implementation of mean soft-dice loss for semantic classification
    """

    __EPSILON = 1e-6

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
        y_pred: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        y_true: a tensor of shape [B, H, W].
        Returns:
        float: soft-iou loss.
        """
        # Convert 3d masks to 2d
        if len(y_pred.shape) == 5:
            y_pred = y_pred.view(*y_pred.shape[:-2], -1)
            y_true = y_true.view(*y_true.shape[:-2], -1)

        num_classes = y_pred.shape[1]

        y_true_dummy = torch.eye(num_classes)[y_true.squeeze(1)]
        y_true_dummy = y_true_dummy.permute(0, 3, 1, 2).to(y_true.device)

        y_pred_proba = F.softmax(y_pred, dim=1)

        intersection = torch.sum(y_pred_proba * y_true_dummy, dim=(2, 3))
        dice_loss = ((2 * intersection + self.__EPSILON) / (
                torch.sum(y_pred_proba ** 2 + y_true_dummy ** 2, dim=(2, 3)) + self.__EPSILON))

        return 1 - dice_loss.mean()


class IoULoss(nn.Module):
    """
    Implementation of mean soft-IoU loss for semantic classification
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

        # Convert 3d masks to 2d
        if len(y_pred.shape) == 5:
            y_pred = y_pred.view(*y_pred.shape[:-2], -1)
            y_true = y_true.view(*y_true.shape[:-2], -1)

        num_classes = y_pred.shape[1]

        y_true_dummy = torch.eye(num_classes)[y_true.squeeze(1)]
        y_true_dummy = y_true_dummy.permute(0, 3, 1, 2).to(y_true.device)

        y_pred_proba = F.softmax(y_pred, dim=1)

        intersection = torch.sum(y_pred_proba * y_true_dummy, dim=(2, 3))
        union = torch.sum(y_pred_proba ** 2 + y_true_dummy ** 2, dim=(2, 3)) - intersection
        iou_loss = ((intersection + self.__EPSILON) / (union + self.__EPSILON))

        return 1 - iou_loss.mean()


# class FocalLoss(nn.Module):
#     def __init__(self, gamma, weight=None, ignore_index=-100, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index = ignore_index
#         if reduction not in ('mean', 'sum', 'none'):
#             raise ValueError(f"Reduction type must be one of following: 'none' | 'mean' | 'sum'")
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         inputs = torch.sigmoid(inputs)
#         cce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
#         focal_loss = (1 - torch.exp(-cce_loss)) ** self.gamma * cce_loss
#         if self.weight is not None:
#             focal_loss = self.weight[targets] * focal_loss
#
#         if self.reduction == 'mean':
#             return focal_loss.mean() if self.weight is None else focal_loss.sum() / self.weight[targets].sum()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

salpha = [15,
 32,
 14,
 28,
 25,
 19,
 46,
 19,
 37,
 35,
 43,
 # 457,
100,
 20,
 25,
 25,
 # 141,
100,
 15,
 60,
 # 1050
 100]

class FocalLoss(nn.Module):
    def __init__(self, alpha=salpha, gamma=2, div=1,logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        print(alpha, gamma)
        self.div=div
        self.alpha = ((torch.FloatTensor(alpha).cuda())/ self.div)
        # self.alpha = (self.alpha / self.alpha.sum())
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha*(1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss




class ParametricKLDivergence(nn.Module):
    def forward(self, dist_params):
        mu, sigma, numel = dist_params
        batch_size = mu.size(0)
        return torch.sum(mu ** 2 + sigma ** 2 - torch.log(sigma ** 2) - 1) / numel / batch_size


def get_loss(loss_config):
    # return nn.BCELoss()
    loss_name = loss_config['name']
    if loss_name == 'bce_w_logits':
        return nn.BCEWithLogitsLoss(reduction='none')

    # if loss_name == 'categorical_cross_entropy':
    #     class_weights = loss_config.get("class_weights", None)
    #     if class_weights is not None:
    #         class_weights = torch.FloatTensor(class_weights)
    #     return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == 'focal_w_logits':

        class_weights = loss_config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).cuda()
        return FocalLoss(gamma=loss_config['gamma'], alpha=salpha, div=loss_config['div'], logits=True)
    # elif loss_name == 'mean_iou':
    #     return IoULoss()
    # elif loss_name == 'mean_dice':
    #     return DiceLoss()
    # elif loss_name == 'mse':
    #     return nn.MSELoss()
    # elif loss_name == 'parametric_kl':
    #     return ParametricKLDivergence()
    else:
        raise ValueError(f"Loss [{loss_name}] not recognized.")
