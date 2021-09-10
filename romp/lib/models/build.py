import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import torch
import torch.nn as nn
from config import args
from models.hrnet_32 import HigherResolutionNet
from models.resnet_50 import ResNet_50
from models.modelv1 import ROMP as ROMPv1

Backbones = {'hrnet': HigherResolutionNet, 'resnet': ResNet_50}
Heads = {1: ROMPv1}

def build_model():
    if args().backbone in Backbones:
        backbone = Backbones[args().backbone]()
    else:
        raise NotImplementedError("Backbone is not recognized")
    if args().model_version in Heads:
        ROMP = Heads[args().model_version]
    else:
        raise NotImplementedError("Head is not recognized")
    model = ROMP(backbone=backbone)
    return model

def build_teacher_model(backbone='hrnet', head=1):
    model = Heads[head](backbone=Backbones[backbone]())
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
