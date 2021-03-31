import sys, os
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch
import shutil
import time
import pickle
import copy
import joblib
import logging
import scipy.io as scio
import quaternion
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader

root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from utils import *
from config import args
import config
import constants

class Image_base(Dataset):
    def __init__(self, train_flag=True, regress_smpl = False):
        super(Image_base,self).__init__()
        self.heatmap_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.COCO_17)

        self.input_shape = [args.input_size, args.input_size]
        self.high_resolution=args.high_resolution
        self.vis_size = 512 if self.high_resolution else 256
        self.labels, self.images, self.file_paths = [],[],[]

        self.root_inds = None
        self.torso_ids = [constants.SMPL_ALL_54[part] for part in ['Neck', 'Neck_LSP', 'R_Shoulder', 'L_Shoulder','Pelvis', 'R_Hip', 'L_Hip']]
        self.heatmap_res = 128
        self.joint_number = len(list(constants.SMPL_ALL_54.keys()))
        self.max_person = args.max_person

    def process_kps(self,kps,img_size,set_minus=True):
        kps = kps.astype(np.float32)
        kps[:,0] = kps[:,0]/ float(img_size[1])
        kps[:,1] = kps[:,1]/ float(img_size[0])
        kps[:,:2] = 2.0 * kps[:,:2] - 1.0

        if kps.shape[1]>2 and set_minus:
            kps[kps[:,2]<=0.03] = -2.
        kps_processed=kps[:,:2]
        for inds, kp in enumerate(kps_processed):
            x,y = kp
            if x > 1 or x < -1 or y < -1 or y > 1:
                kps_processed[inds] = -2.

        return kps_processed

    def map_kps(self,joint_org,maps=None):
        kps = joint_org[maps].copy()
        kps[maps==-1] = -2.
        return kps

    def _calc_center_(self, kps):
        vis = kps[self.torso_ids,0]>-1
        if vis.sum()>0:
            center = kps[self.torso_ids][vis].mean(0)
        elif (kps[:,0]>-1).sum()>0:
            center = kps[kps[:,0]>-1].mean(0)
        return center

    def parse_multiperson_kps(self,image, full_kps, subject_id):
        full_kps = [self.process_kps(kps_i,image.shape) for kps_i in full_kps]
        full_kp2d = np.ones((self.max_person, self.joint_number, 2))*-2.
        person_centers = np.ones((self.max_person, 2))*-2.
        subject_ids = np.ones(self.max_person)*-2
        used_person_inds = []
        filled_id = 0

        for inds in range(min(len(full_kps),self.max_person)):
            kps_i = full_kps[inds]
            center = self._calc_center_(kps_i)
            if center is None or len(center)==0:
                continue
            person_centers[filled_id] = center
            full_kp2d[filled_id] = kps_i
            subject_ids[filled_id] = subject_id[inds]
            filled_id+=1
            used_person_inds.append(inds)

        return person_centers, full_kp2d, subject_ids, used_person_inds
        
    def process_3d(self, info_3d, used_person_inds):
        dataset_name, kp3ds, params, kp3d_hands = info_3d

        kp3d_flag = np.zeros(self.max_person, dtype=np.bool)
        if kp3ds is not None:
            kp_num = kp3ds[0].shape[0]
            # -2 serves as an invisible flag
            kp3d_processed = np.ones((self.max_person,kp_num,3), dtype=np.float32)*-2.
            for inds, used_id in enumerate(used_person_inds):
                kp3d = kp3ds[used_id]
                valid_mask = self._check_kp3d_visible_parts_(kp3d)
                kp3d[~valid_mask] = -2.
                kp3d_flag[inds] = True
                kp3d_processed[inds] = kp3d
        else:
            kp3d_processed = np.ones((self.max_person,self.joint_number,3), dtype=np.float32)*-2.
            
        params_processed = np.ones((self.max_person,76), dtype=np.float32)*-10
        smpl_flag = np.zeros(self.max_person, dtype=np.bool)
        if params is not None:
            for inds, used_id in enumerate(used_person_inds):
                param = params[used_id]
                theta, beta = param[:66], param[-10:]
                params_processed[inds] = np.concatenate([theta, beta])
                smpl_flag[inds] = True
        
        return kp3d_processed, params_processed, kp3d_flag, smpl_flag

    def get_item_single_frame(self,index):

        info_2d, info_3d = self.get_image_info(index)
        dataset_name, imgpath, image, full_kps, root_trans, subject_id = info_2d
        full_kp2d_vis = [kps_i[:,-1] for kps_i in full_kps]
        img_size = (image.shape[1], image.shape[0])

        img_info = process_image(image, full_kps)
        image, image_wbg, full_kps, kps_offset = img_info
        person_centers, full_kp2d, subject_ids, used_person_inds = \
            self.parse_multiperson_kps(image, full_kps, subject_id)

        offset,lt,rb,size,_ = kps_offset
        offsets = np.array([image.shape[1],image.shape[0],lt[1],rb[1],lt[0],rb[0],offset[1],size[1],offset[0],size[0]],dtype=np.int)
        dst_image = cv2.resize(image, tuple(self.input_shape), interpolation = cv2.INTER_CUBIC)
        org_image = cv2.resize(image_wbg, (self.vis_size, self.vis_size), interpolation=cv2.INTER_CUBIC)
        kp3d, params, kp3d_flag, smpl_flag = self.process_3d(info_3d, used_person_inds)

        input_data = {
            'image': torch.from_numpy(dst_image).float(),
            'image_org': torch.from_numpy(org_image),
            'full_kp2d': torch.from_numpy(full_kp2d).float(),
            # rectify the x, y order, from x-y to y-x
            'person_centers':torch.from_numpy(person_centers[:,::-1].copy()).float(),
            'subject_ids':torch.from_numpy(subject_ids).long(),
            'kp_3d': torch.from_numpy(kp3d).float(),
            'params': torch.from_numpy(params).float(),
            'smpl_flag':torch.from_numpy(smpl_flag).bool(),
            'kp3d_flag': torch.from_numpy(kp3d_flag).bool(),
            'offsets': torch.from_numpy(offsets).float(),
            'imgpath': imgpath,
            'data_set':dataset_name,}

        return input_data

    def get_image_info(self,index):
        raise NotImplementedError

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return self.get_item_single_frame(index)

    def _check_kp3d_visible_parts_(self, kp3ds, invisible_flag=-2.):
        visible_parts_mask = (kp3ds!=invisible_flag).sum(-1) == kp3ds.shape[-1]
        return visible_parts_mask

def calc_aabb(ptSets):
    ptLeftTop     = np.array([np.min(ptSets[:,0]),np.min(ptSets[:,1])])
    ptRightBottom = np.array([np.max(ptSets[:,0]),np.max(ptSets[:,1])])

    return np.array([ptLeftTop, ptRightBottom])

def process_image(originImage, full_kps):
    height       = originImage.shape[0]
    width        = originImage.shape[1]
    
    original_shape = originImage.shape
    channels     = originImage.shape[2] if len(originImage.shape) >= 3 else 1
    scale = 1.
    leftTop = np.array([0.,0.])
    rightBottom = np.array([width,height],dtype=np.float32)
    leftTop, rightBottom = get_image_cut_box(leftTop, rightBottom, scale)

    lt = [int(leftTop[0]), int(leftTop[1])]
    rb = [int(rightBottom[0]), int(rightBottom[1])]

    lt[0] = max(0, lt[0])
    lt[1] = max(0, lt[1])
    rb[0] = min(rb[0], width)
    rb[1] = min(rb[1], height)

    leftTop      = np.array([int(leftTop[0]), int(leftTop[1])])
    rightBottom  = np.array([int(rightBottom[0] + 0.5), int(rightBottom[1] + 0.5)])

    length = max(rightBottom[1] - leftTop[1]+1, rightBottom[0] - leftTop[0]+1)

    dstImage = np.zeros(shape = [length,length,channels], dtype = np.uint8)
    orgImage_white_bg = np.ones(shape = [length,length,channels], dtype = np.uint8)*255
    offset = np.array([lt[0] - leftTop[0], lt[1] - leftTop[1]])
    size   = [rb[0] - lt[0], rb[1] - lt[1]]
    dstImage[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0],:]
    orgImage_white_bg[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0],:]

    return dstImage, orgImage_white_bg, [off_set_pts(kps_i, leftTop) for kps_i in full_kps],(offset,lt,rb,size,original_shape[:2])

def off_set_pts(keyPoints, leftTop):
    result = keyPoints.copy()
    result[:, 0] -= leftTop[0]#-offset[0]
    result[:, 1] -= leftTop[1]#-offset[1]
    return result

def check_and_mkdir(dir):
    os.makedirs(dir,exist_ok=True)

def denormalize_kp2ds(mat, img_size=args.input_size):
    return (mat+1)/2*img_size