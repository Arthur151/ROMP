import sys, os
import torch
import torch.nn as nn
from models.hrnet_32 import HigherResolutionNet
from models.resnet_50 import ResNet_50
from models.modelv1 import ROMP
from models.modelv6 import BEV
from models.trace import TROMPv2

Backbones = {'hrnet': HigherResolutionNet, 'resnet': ResNet_50}
Heads = {1: ROMP, 6:BEV}
THeads = {2: TROMPv2}

def build_model(backbone, model_version, **kwargs):
    if backbone in Backbones:
        backbone = Backbones[backbone]()
    else:
        raise NotImplementedError("Backbone is not recognized")
    if model_version in Heads:
        ROMP = Heads[model_version]
    else:
        raise NotImplementedError("Head is not recognized")
    model = ROMP(backbone=backbone, **kwargs)
    return model

def build_temporal_model(model_type='conv3D', head=1):
    model = THeads[head](model_type=model_type)
    return model

if __name__ == '__main__':
    net = build_model()
    nx = torch.rand(4,512,512,3).float().cuda()
    y = net(nx)
    
    for idx, item in enumerate(y):
        if isinstance(item,dict):
            for key, it in item.items():
                print(key,it.shape)
        else:
            print(idx,item.shape)
