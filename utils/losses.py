import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(input, target):
    return F.mse_loss(input, target)

def mae_loss(input, target):
    return F.l1_loss(input, target)

def nmse_loss(input, target):
    return F.mse_loss(input, target) / F.mse_loss(torch.zeros_like(input), target)

def rmse_loss(input, target):
    return torch.sqrt(F.mse_loss(input, target))
