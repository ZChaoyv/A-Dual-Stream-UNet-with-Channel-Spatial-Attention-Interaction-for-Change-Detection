import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


################################################################################
#                                                                              
#              🎯 LOSS FUNCTIONS & OPTIMIZATION OBJECTIVES                     
#                  🎯 损失函数与优化目标中心                                     
#                        👤 AUTHOR: Caoyv                                      
#                      📄 PAPER: DCSI_UNet                                     
#           🔗 https://ieeexplore.ieee.org/document/11299285                   
#                                                                              
#   Description: This script defines the objective functions used to train      
#   the DCSI_UNet, including Cross Entropy and Dice Loss for Change Detection.  
#   代码说明：该脚本定义了训练 DCSI_UNet 的目标函数，包括变化检测常用的交叉熵与 Dice 损失。 
#                                                                              
################################################################################


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


""" 骰子 损失函数(DiceLoss) """
def dice_loss(predicted, true):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = predicted.shape[1]
    device = true.device  # 获取 true 张量所在的设备
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1, device=device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(predicted)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes, device=device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.log_softmax(predicted, dim=1)
    true_1_hot = true_1_hot.type(predicted.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + 1e-6)).mean()
    return (1 - dice_loss)

