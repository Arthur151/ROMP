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

from maps_utils.relative_parser import parse_age_cls_results
import torch.nn.functional as F

class SMPLWrapper(nn.Module):
    def __init__(self):
        super(SMPLWrapper,self).__init__()
        logging.info('Building SMPL family for relative learning!!')
        
        #self.smpl_family = SMPL_family(os.path.join(args().smpl_model_path,'smpl'), J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, kid_template_path=None)
        #self.smil = SMIL(os.path.join(args().smpl_model_path,'smil', 'smil_web.pkl'), sparse=False,J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path)
        self.smpl_model = SMPLA_parser(args().smpla_model_path, args().smil_model_path, baby_thresh=args().baby_threshold)
            
        self.part_name = ['cam', 'global_orient', 'body_pose', 'smpl_betas']
        self.part_idx = [args().cam_dim, args().rot_dim,  (args().smpl_joint_num-1)*args().rot_dim, 11 if not args().separate_smil_betas else 21]
        self.params_num = np.array(self.part_idx).sum()
        self.global_orient_nocam = torch.from_numpy(constants.global_orient_nocam).unsqueeze(0)
        self.joint_mapper_op25 = torch.from_numpy(constants.joint_mapping(constants.SMPL_ALL_44, constants.OpenPose_25)).long()
        #self.softmax = nn.Softmax(dim=1)
    
    def parse_params_pred(self, params_pred, without_cam=False):
        params_dict = self.pack_params_dict(params_pred, without_cam=without_cam)
        params_dict['smpl_betas'], cls_dict = self.process_betas(params_dict['smpl_betas'])
        vertices, joints44_17 = self.smpl_model(betas=params_dict['smpl_betas'], poses=params_dict['smpl_thetas'],separate_smil_betas=args().separate_smil_betas) #, root_align=args().smpl_mesh_root_align
        return vertices, joints44_17, params_dict, cls_dict

    def forward(self, outputs, meta_data, calc_pj2d_org=True):
        vertices, joints44_17, params_dict, cls_dict = self.parse_params_pred(outputs['params_pred'])
        outputs.update({'verts': vertices, 'j3d':joints44_17[:,:args().joint_num], 'joints_h36m17':joints44_17[:,args().joint_num:], **params_dict, **cls_dict})

        outputs.update(vertices_kp3d_projection(outputs['j3d'], outputs['joints_h36m17'], outputs['cam'], \
            input2orgimg_offsets=meta_data['offsets'] if calc_pj2d_org else None, presp=args().perspective_proj, vertices=outputs['verts']))  
        return outputs
    
    def add_template_mesh_pose(self, params):
        template_mesh = self.template_mesh.to(params['smpl_thetas'].device).repeat(len(params['smpl_thetas']), 1, 1)
        template_joint = self.template_joint.to(params['smpl_thetas'].device).repeat(len(params['smpl_thetas']), 1, 1)
        return {'verts': template_mesh, 'j3d':template_joint, 'joints_smpl24':template_joint}

    # def convert_parameters_to_mesh(self, params, baby_thresh=0.8):
    #     is_baby = params['smpl_betas'][:,-1] > baby_thresh
    #     results = self.smpl_family(betas=params['smpl_betas'], poses=params['smpl_thetas'], root_align=True)
    #     if is_baby.sum()>0:
    #         baby_results = self.smil(betas=params['smpl_betas'][is_baby,:10], poses=params['smpl_thetas'][is_baby], root_align=True)
    #         keys = list(results.keys())
    #         for k in keys:
    #             results[k][is_baby] = baby_results[k].float()

    #     return results

    def pack_params_dict(self, params_pred, without_cam=False):
        idx_list, params_dict = [0], {}
        for i, (idx, name) in enumerate(zip(self.part_idx,self.part_name)):
            if without_cam and i==0:
                idx_list.append(0)
                continue
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = params_pred[:, idx_list[i]: idx_list[i+1]].contiguous()
        if params_dict['global_orient'].shape[-1] == 6:
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(
            N, 6).to(params_dict['body_pose'].device)], 1)
        params_dict['smpl_thetas'] = torch.cat(
            [params_dict['global_orient'], params_dict['body_pose']], 1)
        #params_dict['cam'][:, 0] = torch.exp(1.6*params_dict['cam'][:, 0])

        return params_dict

    def process_betas(self, betas_pred):
        # the obvious variance of SMPL body shape is in -8~8.
        smpl_betas = betas_pred[:,:10] 
        #smpl_betas[:,:2] = smpl_betas[:,:2]* 6.
        kid_offsets = betas_pred[:,10]
        #genders_pred = self.softmax(betas_pred[:,11:])
        #age_genders = torch.cat([kid_offsets.unsqueeze(1), genders_pred],1).contiguous()
        Age_preds = parse_age_cls_results(kid_offsets)
        #Gender_cls_preds = Age_Gender_cls_preds[:,1]
        #smpl_gender = torch.index_select(torch.tensor([[1,0],[0,1],[0,0]],device=Gender_cls_preds.device),0,Gender_cls_preds).float().to(smpl_betas.device)

        # whether to detach the learning of kid_offsets in vertex
        #betas_pred = torch.cat([smpl_betas, kid_offsets.unsqueeze(1).detach()],1).contiguous()

        cls_dict = {'Age_preds':Age_preds, 'kid_offsets_pred': kid_offsets}
        return betas_pred, cls_dict


def merge_smpl_outputs(results_list):
    if len(results_list)==1:
        return results_list[0][0]
    results = {k: None for k in results_list[0][0].keys()}
    map_inds = torch.cat([torch.where(mask)[0] for _,mask in results_list], 0)
    for key in list(results.keys()):
        results[key] = torch.cat([result[key] for result,_ in results_list], 0)[map_inds]
    return results
    

'''
        if not args().calc_smpl_mesh:
            #import pickle
            #smpl_model_path = os.path.join(args().smpl_model_path,'smpl', 'SMPL_NEUTRAL.pkl')
            #self.template_mesh = torch.from_numpy(pickle.load(open(smpl_model_path, 'rb'),encoding='latin1')['v_template']).float()[None]
            template_mesh_joints = np.load(os.path.join(args().smpl_model_path,'smpl_template_mesh_joints.npz'),allow_pickle=True)
            self.template_mesh = torch.from_numpy(template_mesh_joints['mesh']).float()[None]
            self.template_mesh[:,:,1] *= -1
            self.template_joint = torch.from_numpy(template_mesh_joints['joint']).float()[None]
            self.template_joint[:,:,1] *= -1


if outputs['params_pred'].shape[-1] <= self.params_num:
            params_dict_sum = {}
            for mode_id in range(outputs['params_pred'].shape[-1] // self.params_num):
                params_dict_sum[mode_id] = self.pack_params_dict(outputs['params_pred'][:, mode_id*self.params_num:(mode_id+1)*self.params_num])
            smpl_outputs, params_dict = self.convert_sum_parameters_to_mesh(params_dict_sum, class_labels)
            outputs.update({'params': params_dict, **smpl_outputs})


    def convert_parameters_to_mesh(self, params_dict, class_labels):
        baby_mask, male_mask, female_mask, neutral_mask = self.parse_depth_mask(class_labels)

        results = []
        if baby_mask.sum()>0:
            smil_outs = self.smpl_family(**{k:v[baby_mask].contiguous() for k, v in params_dict.items()}, model_inds=3)
            results.append([smil_outs, baby_mask])
        for ind, smpl_mask in enumerate([male_mask, female_mask, neutral_mask]):
            if smpl_mask.sum()>0:
                smpl_outs_dict = self.smpl_family(**{k:v[smpl_mask].contiguous() for k, v in params_dict.items()}, model_inds=ind)
                results.append([smpl_outs_dict, smpl_mask])
        return merge_smpl_outputs(results)



    def parse_depth_mask(self, class_labels):
        if args.train_first_round:
            return torch.zeros(len(class_labels)).bool(), torch.zeros(len(class_labels)).bool(),\
            torch.zeros(len(class_labels)).bool(), torch.ones(len(class_labels)).bool()
        ages, genders = class_labels[:,0], class_labels[:,1]
        baby_mask = ages==3
        male_mask = (genders==0) & ~baby_mask
        female_mask = (genders==1) & ~baby_mask
        neutral_mask = ~male_mask & ~female_mask & ~baby_mask
        return baby_mask, male_mask, female_mask, neutral_mask


    def convert_sum_parameters_to_mesh(self, params_dict_sum, class_labels):
        baby_mask, male_mask, female_mask, neutral_mask = self.parse_depth_mask(class_labels)

        results, params_list = [], []
        if baby_mask.sum()>0:
            smil_outs = self.smpl_family(**{k:v[baby_mask].contiguous() for k, v in params_dict_sum[3].items()}, model_inds=3)
            results.append([smil_outs, baby_mask])
            params_list.append([params_dict_sum[3], baby_mask])
        for ind, smpl_mask in enumerate([male_mask, female_mask, neutral_mask]):
            if smpl_mask.sum()>0:
                smpl_outs_dict = self.smpl_family(**{k:v[smpl_mask].contiguous() for k, v in params_dict_sum[ind].items()}, model_inds=ind)
                results.append([smpl_outs_dict, smpl_mask])
                params_list.append([params_dict_sum[ind], smpl_mask])
        return merge_smpl_outputs(results), merge_smpl_outputs(params_list)
'''
