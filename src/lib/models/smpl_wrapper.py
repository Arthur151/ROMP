import torch
import torch.nn as nn
import numpy as np 

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import config
from config import args
import constants
import models.smpl as smpl_model
from utils.projection import vertices_kp3d_projection
from utils.rot_6D import rot6D_to_angular

class SMPLWrapper(nn.Module):
    def __init__(self):
        super(SMPLWrapper,self).__init__()
        self.smpl_model = smpl_model.create(args().smpl_model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, \
            batch_size=args().batch_size,model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',flat_hand_mean=True, use_pca=False)
        if '-1' not in args().gpu:
            self.smpl_model = self.smpl_model.cuda()
        self.part_name = ['cam', 'global_orient', 'body_pose', 'betas']
        self.part_idx = [args().cam_dim, args().rot_dim, (args().smpl_joint_num-1)*args().rot_dim, 10]
        self.params_num = np.array(self.part_idx).sum()

    def forward(self, outputs, meta_data):
        idx_list, params_dict = [0], {}
        for i,  (idx, name) in enumerate(zip(self.part_idx,self.part_name)):
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = outputs['params_pred'][:, idx_list[i]: idx_list[i+1]].contiguous()

        if args().Rot_type=='6D':
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N,6).to(params_dict['body_pose'].device)],1)        
        
        smpl_outs = self.smpl_model(**params_dict, return_verts=True, return_full_pose=True)

        outputs.update({'params': params_dict, 'verts': smpl_outs.vertices, 'j3d':smpl_outs.joints, \
            'joints_h36m17':smpl_outs.joints_h36m17, 'joints_smpl24':smpl_outs.joints_smpl24, 'poses':smpl_outs.full_pose})

        outputs.update(vertices_kp3d_projection(outputs))        
        
        return outputs
