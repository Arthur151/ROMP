import os,sys

sys.path.append(os.path.abspath(__file__).replace('model/model.py',''))
import torch
import torch.nn as nn
from config import args
from models import modelv5

def get_pose_net(params_num = 79):
    model = modelv5.get_pose_net(params_num = params_num)
    return model

if __name__ == '__main__':
    net = get_pose_net()
    net = nn.DataParallel(net).cuda()
    nx = torch.rand(4,512,512,3).float().cuda()
    y = net(nx)
    
    for idx, item in enumerate(y):
        if isinstance(item,dict):
            for key, it in item.items():
                print(key,it.shape)
        else:
            print(idx,item.shape)
