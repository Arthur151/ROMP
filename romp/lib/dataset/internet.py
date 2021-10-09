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

from dataset.image_base import *
import config
from config import args
import constants

class Internet(Dataset):
    def __init__(self, file_list=[], **kwargs):
        super(Internet,self).__init__()
        assert isinstance(file_list, list), print('Error: Input file_list is supposed to be a list!')
        self.file_paths = file_list
        
        print('Loading {} images to process'.format(len(self)))

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

        input_data = img_preprocess(image, imgpath=imgpath, input_size=args().input_size)

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

def img_preprocess(image, imgpath=None, input_size=512, ds='internet', single_img_input=False):
    image = image[:,:,::-1]
    image_org, offsets = process_image(image)
    image = torch.from_numpy(cv2.resize(image_org, (input_size,input_size), interpolation=cv2.INTER_CUBIC))
    
    offsets = torch.from_numpy(offsets).float()
    
    if single_img_input:
        image = image.unsqueeze(0).contiguous()
        offsets = offsets.unsqueeze(0).contiguous()
        
    input_data = {
        'image': image,
        'offsets': offsets,
        'data_set': ds } #

    if imgpath is not None:
        name = os.path.basename(imgpath)
        imgpath, name = imgpath, name
        input_data.update({'imgpath': imgpath, 'name': name})
    return input_data

def test_dataset(image_folder=None):
    save_dir = os.path.join(config.project_dir,'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    if image_folder is None:
        image_folder = os.path.join(config.project_dir,'demo','internet_image')
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
