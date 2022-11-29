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
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs
import config
from config import args
import constants

default_mode = args().image_loading_mode

def Internet(base_class=default_mode):
    class Internet(Base_Classes[base_class]):
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

            input_data = img_preprocess(image, imgpath, input_size=args().input_size)

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
    return Internet

def img_preprocess(image, imgpath, input_size=512, ds='internet', single_img_input=False):
    image = image[:,:,::-1]
    image_org, offsets = process_image(image)
    image = torch.from_numpy(cv2.resize(image_org, (input_size,input_size), interpolation=cv2.INTER_CUBIC))
    #image_1024 = torch.from_numpy(cv2.resize(image_org, (1024,1024), interpolation=cv2.INTER_CUBIC))
    
    offsets = torch.from_numpy(offsets).float()
    name = os.path.basename(imgpath)

    if single_img_input:
        image = image.unsqueeze(0).contiguous()
        #image_1024 = image_1024.unsqueeze(0).contiguous()
        offsets = offsets.unsqueeze(0).contiguous()
        imgpath, name, ds = [imgpath], [name], [ds]
    input_data = {
        'image': image,
        #'image_1024': image_1024,
        'imgpath': imgpath,
        'offsets': offsets,
        'name': name,
        'data_set':ds }
    return input_data

def test_dataset(image_folder=None):
    save_dir = os.path.join(config.project_dir,'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    file_list = glob.glob(os.path.join(image_folder, '*'))
    dataset = Internet(file_list=file_list)
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

if __name__ == '__main__':
    test_dataset('/home/yusun/data_drive/dataset/AGORA/test')
