from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import pickle
import numpy as np
import math
from config import args
from loss_funcs.keypoints_loss import batch_kp_2d_l2_loss, calc_mpjpe, calc_pampjpe

class  Learnable_Loss(nn.Module):
    """docstring for  Learnable_Loss"""
    def __init__(self, ID_num=0):
        super(Learnable_Loss, self).__init__()
        self.loss_class = {'det':['CenterMap','CenterMap_3D'],'reg':['MPJPE','PAMPJPE','P_KP2D','Pose','Shape','Cam', 'Prior']}
        self.all_loss_names = np.concatenate([loss_list for task_name, loss_list in self.loss_class.items()]).tolist()

        if args().learn_2dpose:
            self.loss_class['reg'].append('heatmap')
        if args().learn_AE:
            self.loss_class['reg'].append('AE')
        if args().learn_relative:
            self.loss_class['rel'] = ['R_Age', 'R_Gender', 'R_Weight', 'R_Depth', 'R_Depth_scale']

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
                    print('PA_MPJPE calculation failed! ll', exp_error)
        
        loss_dict = {key:value.mean() for key, value in loss_dict.items() if not isinstance(value, int)}
        
        if new_training and args().model_version==6:
            loss_dict['CenterMap_3D'] = loss_dict['CenterMap_3D'] / 1000.
            loss_dict = {key: loss_dict[key] for key in self.loss_class['det']}
        
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