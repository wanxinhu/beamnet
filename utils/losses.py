import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


# Soft Morphological Operations for clDice
def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_erode(img)
    skel = F.relu(img - soft_open(img))
    for j in range(iter_):
        img = soft_erode(img)
        skel = skel + F.relu(img - soft_open(img))
    return torch.clamp(skel, 0, 1)


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iter_=3, smooth=1.):
        super(SoftCLDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)

        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)

        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


# 实例化全局损失函数
focal_loss_fn = FocalLoss()
cldice_loss_fn = SoftCLDiceLoss(iter_=3)


def adjust_lr_with_warmup(optimizer, current_epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    """
    结合了线性 Warm-up 和余弦退火的学习率调度器
    """
    if warmup_epochs and current_epoch <= warmup_epochs:
        lr = min_lr + (base_lr - min_lr) * (current_epoch / warmup_epochs)
    else:
        progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def combined_vessel_loss(pred_logits, mask, use_cldice=True, w=1):
    """
    联合损失：Focal Loss + Dice Loss + [可选的 clDice Loss]
    """
    loss_focal = focal_loss_fn(pred_logits, mask)
    pred_probs = torch.sigmoid(pred_logits)
    intersection = torch.sum(pred_probs * mask)
    loss_dice = 1.0 - (2.0 * intersection + 1.0) / (torch.sum(pred_probs) + torch.sum(mask) + 1.0)

    if use_cldice:
        loss_cldice = cldice_loss_fn(pred_probs, mask)
        total_loss = w * (0.2 * loss_focal + 0.7 * loss_dice + 0.1 * loss_cldice)
    else:
        # 消融实验退化版本
        total_loss = w * (0.5 * loss_focal + 0.5 * loss_dice)

    return total_loss


