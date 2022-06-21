import sys, os
import torch
import torch.nn as nn
from config import args
from models.hrnet_32 import HigherResolutionNet
from models.resnet_50 import ResNet_50
from models.romp_model import ROMP
from models.bev_model import BEV

Backbones = {'hrnet': HigherResolutionNet, 'resnet': ResNet_50}
Heads = {1: ROMP, 6:BEV}

def build_model():
    if args().backbone in Backbones:
        backbone = Backbones[args().backbone]()
    else:
        raise NotImplementedError("Backbone is not recognized")
    if args().model_version in Heads:
        head = Heads[args().model_version]
    else:
        raise NotImplementedError("Head is not recognized")
    model = head(backbone=backbone)
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
