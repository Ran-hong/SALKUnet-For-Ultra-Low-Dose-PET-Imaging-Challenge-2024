import torch.nn.functional as F
import torch
import math

@torch.no_grad()
def compute_psnr(pred, target, eps=1e-8):
    pred = pred.detach()
    pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
    target = target.detach()
    target = (target - target.min(target)) / (torch.max(target) - torch.min(target))
    mse = F.mse_loss(pred, target)
    psnr = 10 * math.log10(1. / (mse + eps))
    return psnr