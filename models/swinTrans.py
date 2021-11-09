'''
Author: your name
Date: 2021-11-09 20:38:45
LastEditTime: 2021-11-09 21:09:45
LastEditors: Please set LastEditors
Description: Timm Implement of Swin transformer
FilePath: \PetFinderNew\models\swinTrans.py
'''
import os
import torch
import torch.optim as optim
import torch.nn as nn
from timm import create_model

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

class SwinModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = 