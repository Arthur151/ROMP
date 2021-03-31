import glob
import numpy as np
import random
import cv2
import torch
import shutil
import time
import copy
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from dataset.image_base import *
from utils.demo_utils import img_preprocess
import config
from config import args
import constants

class Internet(Dataset):
    def __init__(self, image_folder=None, **kwargs):
        super(Internet,self).__init__()
        self.file_paths = glob.glob(os.path.join(image_folder,'*'))
        sorted(self.file_paths)
        
        print('Loading {} internet data from:{}'.format(len(self), image_folder))

    def get_image_info(self,index):
        return self.file_paths[index]
        
    def resample(self):
        return self.__getitem__(random.randint(0,len(self)))

    def get_item_single_frame(self,index):

        imgpath = self.get_image_info(index)
        image = cv2.imread(imgpath)
        if image is None:
            index = self.resample()
            imgpath = self.get_image_info(index)
            image = cv2.imread(imgpath)

        input_data = img_preprocess(image, imgpath, input_size=args.input_size)

        return input_data


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        try:
            return self.get_item_single_frame(index)
        except Exception as error:
            print(error)
            index = np.random.randint(len(self))
            return self.get_item_single_frame(index)

def test_dataset():
    save_dir = os.path.join(config.project_dir,'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    image_folder = os.path.join(config.project_dir,'demo','images')
    dataset = Internet(image_folder=image_folder)
    length = len(dataset)
    for idx in range(length):
        r = dataset.__getitem__(idx)
        for key, value in r.items():
            if isinstance(value, str):
                print(key,value)
            else:
                print(key,value.shape)
        image = r['image'].numpy().astype(np.uint8)[:,:,::-1]
        cv2.imwrite('{}/{}.jpg'.format(save_dir,idx), image)
        image_org = r['image_org'].numpy().astype(np.uint8)[:,:,::-1]
        cv2.imwrite('{}/{}_org.jpg'.format(save_dir,idx), image_org)

if __name__ == '__main__':
    test_dataset()