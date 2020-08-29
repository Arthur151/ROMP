import sys,os
import glob
import numpy as np
import random
import cv2
import torch
import shutil
import time
import smplx
import copy
from PIL import Image
sys.path.append(os.path.abspath(__file__).replace('dataset/internet.py',''))
import torchvision
from torch.utils.data import Dataset, DataLoader
import config
from config import args
import constants

class Internet(Dataset):
    def __init__(self, image_folder=None, **kwargs):
        super(Internet,self).__init__()
        self.file_paths = glob.glob(os.path.join(image_folder,'*'))
        sorted(self.file_paths)
        self.input_size = args.input_size
        
        print('Loading {} internet data from:{}'.format(len(self), image_folder))

    def get_image_info(self,index):
        return self.file_paths[index]
        
    def resample(self):
        return self.__getitem__(random.randint(0,len(self)))

    def get_item_single_frame(self,index):

        imgpath = self.get_image_info(index)
        image = cv2.imread(imgpath)[:,:,::-1]
        image_size = image.shape[:2][::-1]
        image_org = Image.fromarray(image)
        
        resized_image_size = (float(self.input_size)/max(image_size) * np.array(image_size) // 2 * 2).astype(np.int)[::-1]
        padding = tuple((self.input_size-resized_image_size)[::-1]//2)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resized_image_size, interpolation=3),
            torchvision.transforms.Pad(padding, fill=0, padding_mode='constant'),
            #torchvision.transforms.ToTensor(),
            ])
        image = torch.from_numpy(np.array(transform(image_org)))

        padding_org = tuple((max(image_size)-np.array(image_size))//2)
        transform_org = torchvision.transforms.Compose([
            torchvision.transforms.Pad(padding_org, fill=0, padding_mode='constant'),
            torchvision.transforms.Resize((self.input_size*2, self.input_size*2), interpolation=3), #max(image_size)//2,max(image_size)//2
            #torchvision.transforms.ToTensor(),
            ])
        image_org = torch.from_numpy(np.array(transform_org(image_org)))
        padding_org = (np.array(list(padding_org))*float(self.input_size*2/max(image_size))).astype(np.int)
        if padding_org[0]>0:
            image_org[:,:padding_org[0]] = 255 
            image_org[:,-padding_org[0]:] = 255
        if padding_org[1]>0:
            image_org[:padding_org[1]] = 255 
            image_org[-padding_org[1]:] = 255 

        offsets = np.array([image_size[1],image_size[0],resized_image_size[1],\
            resized_image_size[1]+padding[1],resized_image_size[0],resized_image_size[0]+padding[0],padding[1],\
            resized_image_size[1],padding[0],resized_image_size[0]],dtype=np.int)

        input_data = {
            'image': image.float(),
            'image_org': image_org,
            'imgpath': imgpath,
            'offsets': torch.from_numpy(offsets).float(),
            'name': os.path.basename(imgpath),
            'data_set':'internet'}

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