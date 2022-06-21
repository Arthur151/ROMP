import torch
import torch.nn as nn
import numpy as np 
import logging
import sys, os
import config
from config import args
import constants
from smpl_family.smpla import SMPLA_parser
from utils.projection import vertices_kp3d_projection
from utils.rot_6D import rot6D_to_angular
import torch.nn.functional as F

class SMPLWrapper(nn.Module):
    def __init__(self):
        super(SMPLWrapper,self).__init__()
        logging.info('Building SMPL family for relative learning!!')
        self.smpl_model = SMPLA_parser(args().smpla_model_path, args().smil_model_path, baby_thresh=args().baby_threshold)
            
        self.part_name = ['cam', 'global_orient', 'body_pose', 'betas']
        self.part_idx = [args().cam_dim, args().rot_dim,  (args().smpl_joint_num-1)*args().rot_dim, 11]
        self.params_num = np.array(self.part_idx).sum()
        self.global_orient_nocam = torch.from_numpy(constants.global_orient_nocam).unsqueeze(0)
        self.joint_mapper_op25 = torch.from_numpy(constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()

    def forward(self, outputs, meta_data):
        params_dict = self.pack_params_dict(outputs['params_pred'])

        params_dict['betas'], cls_dict = self.process_betas(params_dict['betas'])

        vertices, joints54_17 = self.smpl_model(betas=params_dict['betas'], poses=params_dict['poses']) #, root_align=args().smpl_mesh_root_align
        outputs.update({'params': params_dict, 'verts': vertices, 'j3d':joints54_17[:,:54], 'joints_h36m17':joints54_17[:,54:], **cls_dict})
        
        outputs.update(vertices_kp3d_projection(outputs['j3d'], outputs['params']['cam'], joints_h36m17_preds=outputs['joints_h36m17'], \
            input2orgimg_offsets=meta_data['offsets'], presp=args().perspective_proj, vertices=outputs['verts']))  
        return outputs
    
    def add_template_mesh_pose(self, params):
        template_mesh = self.template_mesh.to(params['poses'].device).repeat(len(params['poses']), 1, 1)
        template_joint = self.template_joint.to(params['poses'].device).repeat(len(params['poses']), 1, 1)
        return {'verts': template_mesh, 'j3d':template_joint, 'joints_smpl24':template_joint}

    def pack_params_dict(self, params_pred):
        idx_list, params_dict = [0], {}
        for i,  (idx, name) in enumerate(zip(self.part_idx,self.part_name)):
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = params_pred[:, idx_list[i]: idx_list[i+1]].contiguous()
        if args().Rot_type=='6D':
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(
            N, 6).to(params_dict['body_pose'].device)], 1)
        params_dict['poses'] = torch.cat(
            [params_dict['global_orient'], params_dict['body_pose']], 1)

        return params_dict

    def process_betas(self, betas_pred):
        smpl_betas = betas_pred[:,:10] 
        kid_offsets = betas_pred[:,10]
        Age_preds = parse_age_cls_results(kid_offsets)

        cls_dict = {'Age_preds':Age_preds, 'kid_offsets_pred': kid_offsets}
        return betas_pred, cls_dict

def parse_age_cls_results(age_probs):
    age_preds = torch.ones_like(age_probs).long()*-1
    age_preds[(age_probs<=constants.age_threshold['adult'][2])&(age_probs>constants.age_threshold['adult'][0])] = 0
    age_preds[(age_probs<=constants.age_threshold['teen'][2])&(age_probs>constants.age_threshold['teen'][0])] = 1
    age_preds[(age_probs<=constants.age_threshold['kid'][2])&(age_probs>constants.age_threshold['kid'][0])] = 2
    age_preds[(age_probs<=constants.age_threshold['baby'][2])&(age_probs>constants.age_threshold['baby'][0])] = 3
    return age_preds


def merge_smpl_outputs(results_list):
    if len(results_list)==1:
        return results_list[0][0]
    results = {k: None for k in results_list[0][0].keys()}
    map_inds = torch.cat([torch.where(mask)[0] for _,mask in results_list], 0)
    for key in list(results.keys()):
        results[key] = torch.cat([result[key] for result,_ in results_list], 0)[map_inds]
    return results