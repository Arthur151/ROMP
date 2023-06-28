import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import config
from config import args
import constants

from smpl_family.create_smpl_models import create_model
from utils.projection import vertices_kp3d_projection_withfov, vertices_kp3d_projection
from utils.rot_6D import rot6D_to_angular
from maps_utils.relative_parser import parse_age_cls_results
from maps_utils.debug_utils import print_dict

import torch.nn.functional as F

class SMPLWrapper(nn.Module):
    def __init__(self):
        super(SMPLWrapper, self).__init__()
        logging.info('Building SMPL family for relative learning in temporal!')

        self.smpl_model = create_model(args().smpl_model_type)
        #self.part_name = ['cam_trans', 'global_orient', 'body_pose', 'smpl_betas']
        #self.part_idx = [3, 3,  (args().smpl_joint_num-1) * 3, 11 if not args().separate_smil_betas else 21]
        self.part_name = ['cam', 'global_orient', 'body_pose', 'smpl_betas']
        self.part_idx = [3, 6,  (args().smpl_joint_num-1) * 6, 11 if not args().separate_smil_betas else 21]
        self.params_num = np.array(self.part_idx).sum()

    def parse_params_pred(self, params_pred, without_cam=False):
        params_dict = self.pack_params_dict(
            params_pred, without_cam=without_cam)
        params_dict['smpl_betas'], cls_dict = self.process_betas(
            params_dict['smpl_betas'])
        vertices, joints44_17 = self.smpl_model(betas=params_dict['smpl_betas'], \
                            poses=params_dict['smpl_thetas'], separate_smil_betas=args().separate_smil_betas)
        return vertices, joints44_17, params_dict, cls_dict

    def forward(self, outputs, meta_data, calc_pj2d_org=True):
        vertices, joints44_17, params_dict, cls_dict = self.parse_params_pred(outputs['params_pred'])

        if 'world_global_rots' in outputs:
            world_vertices, world_joints44_17 = self.smpl_model(betas=params_dict['smpl_betas'].detach(), \
                            poses=torch.cat([outputs['world_global_rots'], params_dict['smpl_thetas'][:,3:].detach()],1), separate_smil_betas=args().separate_smil_betas)
            outputs.update({'world_verts': world_vertices, 'world_j3d': world_joints44_17[:, :args().joint_num], 'world_joints_h36m17': world_joints44_17[:, args().joint_num:]})

        outputs.update(
            {'verts': vertices, 'j3d': joints44_17[:, :args().joint_num], 'joints_h36m17': joints44_17[:, args().joint_num:], **params_dict, **cls_dict})

        outputs.update(vertices_kp3d_projection(outputs['j3d'], outputs['joints_h36m17'], outputs['cam'], \
                input2orgimg_offsets=meta_data['offsets'] if calc_pj2d_org else None, presp=args().perspective_proj, vertices=outputs['verts'] if args().compute_verts_org else None))  
        if args().dynamic_augment:
            dyna_pouts = vertices_kp3d_projection(outputs['world_j3d'].detach(), outputs['joints_h36m17'].detach(), outputs['world_cams'], \
                input2orgimg_offsets=meta_data['offsets'] if calc_pj2d_org else None, presp=args().perspective_proj, vertices=outputs['verts'] if args().compute_verts_org else None)
            outputs.update({'world_pj2d': dyna_pouts['pj2d'], 'world_trans': dyna_pouts['cam_trans'], 'world_joints_h36m17': dyna_pouts['pj2d_h36m17']})
            if args().compute_verts_org:
                outputs.update({'world_verts_camed_org': dyna_pouts['verts_camed_org']})
            
        return outputs

    def pack_params_dict(self, params_pred, without_cam=False):
        idx_list, params_dict = [0], {}
        for i, (idx, name) in enumerate(zip(self.part_idx, self.part_name)):
            if without_cam and i == 0:
                idx_list.append(0)
                continue
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = params_pred[:, idx_list[i]: idx_list[i+1]].contiguous()
        #print_dict(params_dict)
        if params_dict['global_orient'].shape[-1] == 6:
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(
            N, 6).to(params_dict['body_pose'].device)], 1)
        params_dict['smpl_thetas'] = torch.cat(
            [params_dict['global_orient'], params_dict['body_pose']], 1)
        return params_dict

    def process_betas(self, betas_pred):
        # TODO: don't have video training data with kid offset
        if not args().learn_relative_age:
            betas_pred[:, 10] = 0

        # the obvious variance of SMPL body shape is in -8~8.
        #smpl_betas = betas_pred[:, :10]
        #smpl_betas[:,:2] = smpl_betas[:,:2]* 6.
        kid_offsets = betas_pred[:, 10]
        #genders_pred = self.softmax(betas_pred[:,11:])
        #age_genders = torch.cat([kid_offsets.unsqueeze(1), genders_pred],1).contiguous()
        Age_preds = parse_age_cls_results(kid_offsets)
        #Gender_cls_preds = Age_Gender_cls_preds[:,1]
        #smpl_gender = torch.index_select(torch.tensor([[1,0],[0,1],[0,0]],device=Gender_cls_preds.device),0,Gender_cls_preds).float().to(smpl_betas.device)

        # whether to detach the learning of kid_offsets in vertex
        #betas_pred = torch.cat([smpl_betas, kid_offsets.unsqueeze(1).detach()],1).contiguous()
        betas_pred = betas_pred[:, :10]

        cls_dict = {'Age_preds': Age_preds, 'kid_offsets_pred': kid_offsets}
        return betas_pred, cls_dict


def merge_smpl_outputs(results_list):
    if len(results_list) == 1:
        return results_list[0][0]
    results = {k: None for k in results_list[0][0].keys()}
    map_inds = torch.cat([torch.where(mask)[0] for _, mask in results_list], 0)
    for key in list(results.keys()):
        results[key] = torch.cat([result[key]
                                  for result, _ in results_list], 0)[map_inds]
    return results


"""
            outputs.update(vertices_kp3d_projection_withfov(outputs['j3d'], outputs['joints_h36m17'], \
                outputs['cam_trans'], outputs['pred_fovs'],
                input2orgimg_offsets=meta_data['offsets'] if calc_pj2d_org else None, vertices=outputs['verts'] if args().inference_video else None))
            if args().dynamic_augment:
                dyna_pouts = vertices_kp3d_projection_withfov(outputs['j3d'].detach(), outputs['joints_h36m17'].detach(), \
                    outputs['world_trans'], outputs['pred_fovs'], input2orgimg_offsets=meta_data['offsets'] if calc_pj2d_org else None, vertices=outputs['verts'] if args().inference_video else None)
                outputs.update({'world_pj2d': dyna_pouts['pj2d'], 'world_pj2d_h36m17': dyna_pouts['pj2d_h36m17']})
"""