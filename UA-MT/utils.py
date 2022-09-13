import torch
import torch.nn.functional as F
import numpy as np

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    x = torch.sum(score * target)
    y = torch.sum(score * score)
    z = torch.sum(target * target)
    loss = 1 - (2 * x + smooth) / (y + z + smooth)
    return loss

def softmax_mse_loss(score, target):
    score = F.softmax(score, dim=1)
    target = F.softmax(target, dim=1)
    loss = F.mse_loss(score, target)
    return loss

def softmax_kl_loss(score, target):
    score = F.log_softmax(score, dim=1)
    target = F.softmax(target, dim=1)
    loss = F.kl_div(score, target, reduction='none')
    return loss

def sigmod_rampup(current, rampup_length):
    if current < rampup_length:
        p = max(0.0, float(current)) / float(rampup_length)
        p = 1.0 - p
        return float(np.exp(-p*p*5.0))
    else:
        return 1.0