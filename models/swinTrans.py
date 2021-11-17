'''
Author: your name
Date: 2021-11-09 20:38:45
LastEditTime: 2021-11-17 13:44:57
LastEditors: wht
Description: Timm Implement of Swin transformer
FilePath: \PetFinderNew\models\swinTrans.py
'''
import os
from timm.models.layers import config
import torch
import torch.optim as optim
import torch.nn as nn
from timm import create_model
from box import Box

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
        self.backbone = create_model(
            self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )
    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out

if __name__ == "__main__":
    cfg = {
        'model':{
            'name': 'swin_tiny_patch4_window7_224',
            'output_dim': 1
        }
    }
    conf = Box(cfg)
    data = torch.rand(1,3,224,224)
    model = SwinModel(conf)
    out = model(data)
    print(out.shape)