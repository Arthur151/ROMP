from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import time
import pickle
import numpy as np

import config
from config import args
import constants

from utils.center_utils import denormalize_center
from loss_funcs.params_loss import batch_l2_loss_param,batch_l2_loss
from loss_funcs.keypoints_loss import batch_kp_2d_l2_loss, calc_mpjpe, calc_pampjpe
from loss_funcs.maps_loss import focal_loss, JointsMSELoss
from loss_funcs.prior_loss import angle_prior, MaxMixturePrior
from maps_utils.centermap import CenterMap

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.gmm_prior = MaxMixturePrior(prior_folder=args().smpl_model_path,num_gaussians=8,dtype=torch.float32).cuda()
        if args().HMloss_type=='focal':
            args().heatmap_weight /=1000
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        self.joint_lossweights = torch.from_numpy(constants.SMPL54_weights).float()
        self.align_inds_MPJPE = np.array([constants.SMPL_ALL_54['L_Hip'], constants.SMPL_ALL_54['R_Hip']])
        self.shape_pca_weight = torch.Tensor([1, 0.32, 0.16, 0.16, 0.08, 0.08, 0.08, 0.04, 0.02, 0.01]).unsqueeze(0).float()

    def forward(self, outputs, **kwargs):
        meta_data = outputs['meta_data']
        detect_loss_dict = self._calc_detection_loss(outputs, meta_data)
        detection_flag = outputs['detection_flag'].sum() if args().model_return_loss else outputs['detection_flag']
        if detection_flag or args().model_return_loss:
            kp_loss_dict, kp_error = self._calc_keypoints_loss(outputs, meta_data) #, kp_acc_dict
            params_loss_dict = self._calc_param_loss(outputs, meta_data)
            loss_dict = dict(detect_loss_dict, **kp_loss_dict, **params_loss_dict)
        else:
            loss_dict = detect_loss_dict
            kp_error = None

        loss_names = list(loss_dict.keys())
        for name in loss_names:
            if isinstance(loss_dict[name],tuple):
                loss_dict[name] = loss_dict[name][0]
            elif isinstance(loss_dict[name],int):
                loss_dict[name] = torch.zeros(1,device=outputs['center_map'].device)
            loss_dict[name] = loss_dict[name].mean() * eval('args().{}_weight'.format(name))

        return {'loss_dict':loss_dict, 'kp_error':kp_error}

    def _calc_detection_loss(self, outputs, meta_data):
        device = outputs['center_map'].device
        detect_loss_dict = {'CenterMap': 0}
        all_person_mask = meta_data['all_person_detected_mask'].to(device)
        if all_person_mask.sum()>0:
            detect_loss_dict['CenterMap'] = focal_loss(outputs['center_map'][all_person_mask], meta_data['centermap'][all_person_mask].to(device)) #((centermaps-centermaps_gt)**2).sum(-1).sum(-1).mean(-1) #
        return detect_loss_dict

    def _calc_keypoints_loss(self, outputs, meta_data):
        kp_loss_dict, error = {'P_KP2D':0, 'MPJPE':0, 'PAMPJPE':0}, {'3d':{'error':[], 'idx':[]},'2d':{'error':[], 'idx':[]}}
        if 'pj2d' in outputs:
            real_2d = meta_data['full_kp2d'].to(outputs['pj2d'].device)
            if args().model_version == 3:
                kp_loss_dict['joint_sampler'] = self.joint_sampler_loss(real_2d, outputs['joint_sampler_pred'])
            kp_loss_dict['P_KP2D'] = batch_kp_2d_l2_loss(real_2d.float(), outputs['pj2d'].float(), weights=self.joint_lossweights)
        
            kp3d_mask = meta_data['valid_masks'][:,1]#.to(outputs['j3d'].device)
            if (~kp3d_mask).sum()>1:
                error['2d']['error'].append(kp_loss_dict['P_KP2D'][~kp3d_mask].detach()*1000)
                error['2d']['idx'].append(torch.where(~kp3d_mask)[0])
        
        if kp3d_mask.sum()>1 and 'j3d' in outputs:
            kp3d_gt = meta_data['kp_3d'][kp3d_mask].contiguous().to(outputs['j3d'].device)
            preds_kp3d = outputs['j3d'][kp3d_mask, :kp3d_gt.shape[1]].contiguous()
            
            if args().MPJPE_weight>0:
                mpjpe_each = calc_mpjpe(kp3d_gt, preds_kp3d, align_inds=self.align_inds_MPJPE)
                kp_loss_dict['MPJPE'] = mpjpe_each
                error['3d']['error'].append(mpjpe_each.detach()*1000)
                error['3d']['idx'].append(torch.where(kp3d_mask)[0])
            if not args().model_return_loss and args().PAMPJPE_weight>0 and len(preds_kp3d)>0:
                try:
                    pampjpe_each = calc_pampjpe(kp3d_gt.contiguous(), preds_kp3d.contiguous())
                    kp_loss_dict['PAMPJPE'] = pampjpe_each
                except Exception as exp_error:
                    print('PA_MPJPE calculation failed!', exp_error)

        return kp_loss_dict, error

    def _calc_param_loss(self, outputs, meta_data):
        params_loss_dict = {'Pose': 0, 'Shape':0, 'Prior':0}

        if 'params' in outputs:
            _check_params_(meta_data['params'])
            device = outputs['params']['body_pose'].device
            grot_masks, smpl_pose_masks, smpl_shape_masks = meta_data['valid_masks'][:,3].to(device), meta_data['valid_masks'][:,4].to(device), meta_data['valid_masks'][:,5].to(device)

            if grot_masks.sum()>0:
                params_loss_dict['Pose'] += batch_l2_loss_param(meta_data['params'][grot_masks,:3].to(device).contiguous(), outputs['params']['global_orient'][grot_masks].contiguous()).mean()

            if smpl_pose_masks.sum()>0:
                params_loss_dict['Pose'] += batch_l2_loss_param(meta_data['params'][smpl_pose_masks,3:22*3].to(device).contiguous(), outputs['params']['body_pose'][smpl_pose_masks,:21*3].contiguous()).mean()

            if smpl_shape_masks.sum()>1:
                # beta annots in datasets are for each gender (male/female), not for our neutral. 
                smpl_shape_diff = meta_data['params'][smpl_shape_masks,-10:].to(device).contiguous() - outputs['params']['betas'][smpl_shape_masks,:10].contiguous()
                params_loss_dict['Shape'] += torch.norm(smpl_shape_diff, p=2, dim=-1).mean() / 20.

            if (~smpl_shape_masks).sum()>1:
                params_loss_dict['Shape'] += (outputs['params']['betas'][~smpl_shape_masks,1:10]**2).mean() / 10.

            gmm_prior_loss = self.gmm_prior(outputs['params']['body_pose'], outputs['params']['betas']).mean()/100.
            angle_prior_loss = angle_prior(outputs['params']['body_pose']).mean()/5.
            params_loss_dict['Prior'] = gmm_prior_loss + angle_prior_loss

        return params_loss_dict

    def joint_sampler_loss(self, real_2d, joint_sampler):
        batch_size = joint_sampler.shape[0]
        joint_sampler = joint_sampler.view(batch_size, -1, 2)
        joint_gt = real_2d[:,constants.joint_sampler_mapper]
        loss = batch_kp_2d_l2_loss(joint_gt, joint_sampler)
        return loss

def _check_params_(params):
    assert params.shape[0]>0, logging.error('meta_data[params] dim 0 is empty, params: {}'.format(params))
    assert params.shape[1]>0, logging.error('meta_data[params] dim 1 is empty, params shape: {}, params: {}'.format(params.shape, params))
