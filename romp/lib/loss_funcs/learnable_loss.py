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
        self.loss_class = {'det':['CenterMap'],'reg':['MPJPE','PAMPJPE','P_KP2D','Pose','Shape','Prior']}
        self.all_loss_names = np.concatenate([loss_list for task_name, loss_list in self.loss_class.items()]).tolist()

        if args().learn_2dpose:
            self.loss_class['reg'].append('heatmap')
        if args().learn_AE:
            self.loss_class['reg'].append('AE')

    def forward(self, outputs):
        loss_dict = outputs['loss_dict']
        if args().model_return_loss:
            if args().PAMPJPE_weight>0 and outputs['detection_flag']:
                try:
                    kp3d_mask = outputs['meta_data']['valid_masks'][:,1]
                    kp3d_gt = outputs['meta_data']['kp_3d'][kp3d_mask].contiguous().to(outputs['j3d'].device)
                    preds_kp3d = outputs['j3d'][kp3d_mask, :kp3d_gt.shape[1]].contiguous()
                    if len(preds_kp3d)>0:
                        loss_dict['PAMPJPE'] = calc_pampjpe(kp3d_gt.contiguous(), preds_kp3d.contiguous()).mean() * args().PAMPJPE_weight
                except Exception as exp_error:
                    print('PA_MPJPE calculation failed!', exp_error)

        loss_dict = {key:value.mean() for key, value in loss_dict.items() if not isinstance(value, int)}
        loss = sum([value if value.item()<args().loss_thresh else value/(value.item()/args().loss_thresh) for key, value in loss_dict.items()])
        
        det_loss = sum([loss_dict[item] for item in self.loss_class['det'] if item in loss_dict])
        reg_loss = sum([loss_dict[item] for item in self.loss_class['reg'] if item in loss_dict])
        loss_tasks = {'reg': reg_loss, 'det': det_loss}

        left_loss = sum([loss_dict[loss_item] for loss_item in loss_dict if loss_item not in self.all_loss_names])
        if left_loss!=0:
            loss_tasks.update({'Others': left_loss})

        outputs['loss_dict'] = dict(loss_tasks, **loss_dict)

        return loss, outputs