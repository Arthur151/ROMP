import cv2
import keyboard
import imageio
import torch
import numpy as np
import random
from transforms3d.axangles import axangle2mat
import pickle
from PIL import Image
import torchvision
import time
import os,sys

import config
import constants
from config import args
from utils import save_obj

    
def get_video_bn(video_file_path):
    basename = os.path.basename(video_file_path)
    for ext in constants.video_exts:
        basename.replace(ext, '')
    return basename

def save_meshes(reorganize_idx, outputs, output_dir, smpl_faces):
    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx==vid)[0]
        img_path = outputs['meta_data']['imgpath'][verts_vids[0]]
        obj_name = os.path.join(output_dir, '{}'.format(os.path.basename(img_path))).replace('.mp4','').replace('.jpg','').replace('.png','')+'.obj'
        for subject_idx, batch_idx in enumerate(verts_vids):
            save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), \
                smpl_faces,obj_name.replace('.obj', '_{}.obj'.format(subject_idx)))



class OpenCVCapture:
    def __init__(self, video_file=None, show=False):
        if video_file is None:
            self.cap = cv2.VideoCapture(int(args().cam_id))
        else:
            self.cap = cv2.VideoCapture(video_file)
            self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.whether_to_show=show


    def read(self, return_rgb=True):
        flag, frame = self.cap.read()
        if not flag:
          return None
        if self.whether_to_show:
            cv2.imshow('webcam',cv2.resize(frame, (240,320)))
            cv2.waitKey(1)
        if return_rgb:
            frame = np.flip(frame, -1).copy() # BGR to RGB
        return frame

class Image_Reader:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_list = os.listdir(self.image_folder)
        self.current_num=0

    def read(self):
        frame = cv2.imread(os.path.join(self.image_folder,self.image_list[self.current_num]))
        self.current_num+=1
        if self.current_num==len(self.image_list):
            self.current_num=0
        return np.flip(frame, -1).copy() # BGR to RGB


class Time_counter():
    def __init__(self,thresh=0.1):
        self.thresh=thresh
        self.runtime = 0
        self.frame_num = 0

    def start(self):
        self.start_time = time.time()

    def count(self, frame_num=1):
        time_cost = time.time()-self.start_time
        if time_cost<self.thresh:
            self.runtime+=time_cost
            self.frame_num+=frame_num
        self.start()

    def fps(self):
        print('average per-frame runtime:',self.runtime/self.frame_num)
        print('FPS: {}, not including visualization time. '.format(self.frame_num/self.runtime))

    def reset(self):
        self.runtime = 0
        self.frame_num = 0

def video2frame(video_name, frame_save_dir=None):
    cap = OpenCVCapture(video_name)
    os.makedirs(frame_save_dir, exist_ok=True)
    frame_list = []
    for frame_id in range(int(cap.length)):
        frame = cap.read(return_rgb=False)
        save_path = os.path.join(frame_save_dir, '{:06d}.jpg'.format(frame_id))
        cv2.imwrite(save_path, frame)
        frame_list.append(save_path)
    return frame_list

def frames2video(images_path, video_name,fps=30):
    writer = imageio.get_writer(video_name, format='mp4', mode='I', fps=fps)
    for path in images_path:
        image = imageio.imread(path)
        writer.append_data(image)
    writer.close()