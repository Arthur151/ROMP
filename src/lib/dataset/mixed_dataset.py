import torch
import numpy as np
from .pw3d import PW3D
from .internet import Internet

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from dataset.image_base import *
import config
from config import args

dataset_dict = {'pw3d':PW3D, 'internet':Internet}

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None,**kwargs):
        assert dataset in dataset_dict, print('dataset {} not found while creating data loader!'.format(dataset))
        self.dataset = dataset_dict[dataset](**kwargs)
        self.length = len(self.dataset)            

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    config.datasets_used = ['pw3d','crowdpose','posetrack','oh']
    datasets = MixedDataset(train_flag=True)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset = datasets,batch_size = 64,shuffle = True,drop_last = True, pin_memory = True,num_workers =1)
    for data in enumerate(data_loader):
        pass