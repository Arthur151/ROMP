from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from asyncio import constants

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import pickle
import numpy as np
import math
from config import args
import constants
from loss_funcs.keypoints_loss import calc_pampjpe, calc_pj2d_error
from loss_funcs.relative_loss import relative_depth_scale_loss
from loss_funcs.video_loss import _calc_world_trans_loss_, extract_sequence_inds
              

class  Learnable_Loss(nn.Module):
    """docstring for  Learnable_Loss"""
    def __init__(self, ID_num=0):
        super(Learnable_Loss, self).__init__()
        self.loss_class = {'det':['CenterMap', 'CenterMap_3D'],\
            'loc':['Cam', 'init_pj2d', 'cams_init', 'P_KP2D'],\
            'reg':['MPJPE', 'PAMPJPE', 'P_KP2D', 'Pose', 'Shape', 'Prior', 'ortho']}
        
        if args().learn_relative:
            self.loss_class['rel'] = ['R_Age', 'R_Gender', 'R_Weight', 'R_Depth', 'R_Depth_scale']
        if args().video:
            self.loss_class['temp'] = ['temp_rot_consist', 'temp_cam_consist', 'temp_shape_consist']
        if args().dynamic_augment:
            self.loss_class['dynamic'] = ['world_cams_consist', 'world_cams', 'world_pj2D', 'world_foot', 'wrotsL2',\
                                'world_cams_init_consist', 'world_cams_init', 'init_world_pj2d', 'world_grots', 'world_trans']
        if args().learn_motion_offset3D:
            self.loss_class['motion'] = ['motion_offsets3D', 'associate_offsets3D']
        
        self.all_loss_names = np.concatenate([loss_list for task_name, loss_list in self.loss_class.items()]).tolist()

    def forward(self, outputs, new_training=False):
        loss_dict = outputs['loss_dict']
        if args().model_return_loss and args().calc_mesh_loss and not new_training:
            if args().PAMPJPE_weight>0 and outputs['detection_flag'].sum()>0:
                try:
                    kp3d_mask = outputs['meta_data']['valid_masks'][:,1].to(outputs['j3d'].device)
                    kp3d_gt = outputs['meta_data']['kp_3d'][kp3d_mask].contiguous().to(outputs['j3d'].device)
                    preds_kp3d = outputs['j3d'][kp3d_mask, :kp3d_gt.shape[1]].contiguous()
                    if len(preds_kp3d)>0:
                        loss_dict['PAMPJPE'] = calc_pampjpe(kp3d_gt.contiguous().float(), preds_kp3d.contiguous().float()).mean() * args().PAMPJPE_weight
                except Exception as exp_error:
                    print('PA_MPJPE calculation failed! at Learnable_Loss', exp_error)
        
        if args().model_return_loss and args().dynamic_augment:
            meta_data = outputs['meta_data']
            sequence_mask = outputs['pred_seq_mask']
            pred_batch_ids = outputs['pred_batch_ids'][sequence_mask].detach().long() - meta_data['batch_ids'][0]
            subject_ids = meta_data['subject_ids'][sequence_mask]
            
            clip_frame_ids = meta_data['seq_inds'][pred_batch_ids,1]
            video_seq_ids = meta_data['seq_inds'][pred_batch_ids,0]
            sequence_inds = extract_sequence_inds(subject_ids, video_seq_ids, clip_frame_ids)

            world_cam_masks = meta_data['world_cam_mask'][sequence_mask]
            world_trans_gts = meta_data['world_root_trans'][sequence_mask]
            world_trans_preds = outputs['world_trans'][sequence_mask].float()
            loss_dict['world_trans'] = _calc_world_trans_loss_(world_trans_preds, world_trans_gts, world_cam_masks, sequence_inds)
            loss_dict['world_trans'] = loss_dict['world_trans'] * args().world_trans_weight
        
        loss_dict = {key:value.mean() for key, value in loss_dict.items() if not isinstance(value, int)}

        loss_list = []
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                if not torch.isnan(value):
                    if value.item()<args().loss_thresh:
                        loss_list.append(value)
                    else:
                        loss_list.append(value/(value.item()/args().loss_thresh))
        loss = sum(loss_list)
        
        loss_tasks = {}
        for loss_class in self.loss_class:
            loss_tasks[loss_class] = sum(
                [loss_dict[item] for item in self.loss_class[loss_class] if item in loss_dict])

        left_loss = sum([loss_dict[loss_item] for loss_item in loss_dict if loss_item not in self.all_loss_names])
        if left_loss!=0:
            loss_tasks.update({'Others': left_loss})

        outputs['loss_dict'] = dict(loss_tasks, **loss_dict)

        return loss, outputs
    
