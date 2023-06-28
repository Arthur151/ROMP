import glob
import numpy as np
import random
import cv2
import torch
import shutil
import os
from datasets.image_base import *
import config
from config import args
from datasets.base import Base_Classes, Test_Funcs

default_mode = args().video_loading_mode if args().video else args().image_loading_mode

def InternetVideo(base_class=default_mode):
    class InternetVideo(Base_Classes[base_class]):
        def __init__(self, video_path_list=[], frame_path_dict=None, img_ext='.jpg', load_entire_sequence=True, seq_name_level=1,**kwargs):
            super(InternetVideo,self).__init__(load_entire_sequence=load_entire_sequence, train_flag=False)
            assert isinstance(video_path_list, list), print('Error: Input file_list is supposed to be a list!')
            self.prepare_video_sequence(video_path_list, frame_path_dict, img_ext, seq_name_level)
            self.video_clip_ids = self.prepare_video_clips()
            
            print('Loading {} image sequences to process'.format(len(self)))
        
        def prepare_video_sequence(self, video_path_list, frame_path_dict, img_ext, seq_name_level):
            self.sequence_dict = {}
            if frame_path_dict is not None:
                self.sequence_dict = frame_path_dict
            else:
                for video_path in video_path_list:
                    frame_list = sorted(glob.glob(os.path.join(video_path,'*'+img_ext)))
                    if len(frame_list) ==0:
                        frame_list = sorted(glob.glob(os.path.join(video_path,'*.png')))
                    self.sequence_dict[video_path] = frame_list

            self.sequence_dict = OrderedDict(self.sequence_dict)
            self.file_paths, self.sequence_ids, self.sid_video_name = [], [], {}
            for sid, video_path in enumerate(self.sequence_dict):
                video_name = os.path.basename(video_path)
                if seq_name_level == 2:
                    video_name = video_path.split(os.path.sep)[-2]+'-'+video_name
                self.sid_video_name[sid] = video_name
                self.sequence_ids.append([])
                for fid, frame_path in enumerate(self.sequence_dict[video_path]):
                    self.file_paths.append([sid,fid,os.path.join(video_path, frame_path)])
                    self.sequence_ids[sid].append(len(self.file_paths)-1)
            print('sid_video_name:',self.sid_video_name)

        def get_image_info(self,index):
            return self.file_paths[index]
            
        def resample(self):
            return self.__getitem__(random.randint(0,len(self)))

        def get_item_single_frame(self, index, **kwargs):
            sid, frame_id, imgpath = self.get_image_info(index)
            seq_name = self.sid_video_name[sid]
            end_frame_flag = frame_id == (len(self.sequence_ids[sid])-1)
            image = cv2.imread(imgpath)
            if image is None:
                print('image load None', imgpath)

            input_data = img_preprocess(image, imgpath, input_size=args().input_size)
            input_data['seq_info'] = torch.Tensor([sid, frame_id, end_frame_flag])
            return input_data

    return InternetVideo

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
        #'name': name,
        'data_set':ds }
    return input_data

def test_dataset(image_folder=None):
    save_dir = os.path.join(config.project_dir,'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    if image_folder is None:
        image_folder = os.path.join(config.data_dir,'demo','internet_image')
    datasets = InternetVideo(image_folder=image_folder)
    length = len(datasets)
    for idx in range(length):
        r = datasets.__getitem__(idx)
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
    test_dataset('/home/sunyu15/datasets/AGORA/test')
