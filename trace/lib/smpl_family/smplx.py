#  -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import Optional, Dict, Union
import os
import os.path as osp

import pickle

import numpy as np

import torch
import torch.nn as nn

from smpl_family.smpl import SMPL, lbs, regress_joints_from_vertices
import constants

from collections import namedtuple

class VertexJointSelector(nn.Module):
    """
    Different from SMPL which directly sellect the face/hand/foot joints as specific vertex points from mesh surface \
        via torch.index_select(vertices, 1, self.extra_joints_idxs)
    The right joints should be regressed in SMPL-X joints manner via joint regressor. 
    """
    def __init__(self, extra_joints_idxs, J_regressor_extra9, J_regressor_h36m17, dtype=torch.float32, sparse_joint_regressor=False):
        super(VertexJointSelector, self).__init__()
        if not sparse_joint_regressor:
            J_regressor_extra9 = J_regressor_extra9.to_dense()
            J_regressor_h36m17 = J_regressor_h36m17.to_dense()

        self.register_buffer('facial_foot_joints_idxs', extra_joints_idxs)
        self.register_buffer('J_regressor_extra9', J_regressor_extra9)
        self.register_buffer('J_regressor_h36m17', J_regressor_h36m17)
        
    def forward(self, vertices, joints):
        facial_foot_joints9 = torch.index_select(vertices, 1, self.facial_foot_joints_idxs)
        extra_joints9 = regress_joints_from_vertices(vertices, self.J_regressor_extra9)
        joints_h36m17 = regress_joints_from_vertices(vertices, self.J_regressor_h36m17)
        # 73 joints = 24 smpl joints + 9 face & feet joints + 9 extra joints from different datasets + 17 joints from h36m
        joints73_17 = torch.cat([joints, facial_foot_joints9, extra_joints9, joints_h36m17], dim=1)

        return joints73_17

class SMPLX(SMPL):
    """
    sparse_joint_regressor: 
        True: using sparse coo matrix for joint regressor, 
        when batch size = 1, faster (65%, 8.45e-3 v.s. 3e-3) on CPU, while slower (25%, 1.5e-3 v.s. 1.2e-3) on GPU. 
        when batch size >4, on GPU, they cost equal GPU memory, and direct dense matrix multiplation is always faster than the sparse one. 
        Maybe sparse matrix multiplation is not optmized as good as dense matrix multiplation.
             
    """
    def __init__(self, model_path, model_type='smplx', sparse_joint_regressor=True,\
        pca_hand_pose_num=0, flat_hand_mean=True, expression_dim=10, dtype=torch.float32):
        super(SMPLX, self).__init__(model_path, model_type='smpl')

        model_info = torch.load(model_path)
        self.vertex_joint_selector = VertexJointSelector(model_info['extra_joints_index'], \
            model_info['J_regressor_extra9'], model_info['J_regressor_h36m17'], \
            dtype=self.dtype, sparse_joint_regressor=sparse_joint_regressor)
        if not sparse_joint_regressor:
            self.J_regressor = self.J_regressor.to_dense()

        self.expression_dim=expression_dim
        self.register_buffer('expr_dirs', model_info['expr_dirs'][...,:expression_dim])

        self.pca_hand_pose = pca_hand_pose_num>0
        self.pca_hand_pose_num = pca_hand_pose_num
        if self.pca_hand_pose:
            self.hand_pose_dim = self.pca_hand_pose_num
            self.register_buffer('left_hand_components', model_info['hands_componentsl'][:pca_hand_pose_num])
            self.register_buffer('right_hand_components', model_info['hands_componentsr'][:pca_hand_pose_num])
        else:
            self.hand_pose_dim = 45
        self.flat_hand_mean = flat_hand_mean
        if not self.flat_hand_mean:
            self.register_buffer('left_hand_mean', model_info['hands_meanl'])
            self.register_buffer('right_hand_mean', model_info['hands_meanr'])
    
        #self.SMPLX55_to_SMPL24 = np.array([i for i in range(22)] + [30, 45])
        
    def forward(self, betas=None, poses=None, head_poses=None, expression=None, \
                left_hand_pose=None, right_hand_pose=None, root_align=True, **kwargs):
        ''' Forward pass for the SMPL model
            Parameters
            ----------
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            head_poses: Bx(3*3), 3 joints including jaw_pose,leye_pose,reye_pose
            expression: Bxexpression_dim, usually first 10 parameters to control the facial expression
            left_hand_pose / right_hand_pose: 
                if self.pca_hand_pose is True, then use PCA hand pose space, (B,self.pca_hand_pose_num)
                    else (B,(15*3)), each finger has 3 joints to control the hand pose. 

            Return
            ----------
            outputs: dict, {'verts': vertices of body meshes, (B x 6890 x 3),
                            'joints54': 73 joints of body meshes, (B x 73 x 3), }
                            #'joints_h36m17': 17 joints of body meshes follow h36m skeleton format, (B x 17 x 3)}
        '''
        if isinstance(betas,np.ndarray):
            betas = torch.from_numpy(betas).type(self.dtype)
        if isinstance(poses,np.ndarray):
            poses = torch.from_numpy(poses).type(self.dtype)
        
        if left_hand_pose is None:
            left_hand_pose = torch.zeros(len(poses), self.hand_pose_dim, dtype=poses.dtype, device=poses.device)
        if right_hand_pose is None:
            right_hand_pose = torch.zeros(len(poses), self.hand_pose_dim, dtype=poses.dtype, device=poses.device)
        if head_poses is None:
            head_poses = torch.zeros(len(poses), 3*3, dtype=poses.dtype, device=poses.device)
        if expression is None:
            expression = torch.zeros(len(poses), self.expression_dim, dtype=poses.dtype, device=poses.device)

        if self.pca_hand_pose:           
            left_hand_pose = torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])
        if not self.flat_hand_mean:
            left_hand_pose = left_hand_pose + self.left_hand_mean
            right_hand_pose = right_hand_pose + self.right_hand_mean

        default_device = self.shapedirs.device
        betas, poses = betas.to(default_device), poses.to(default_device)

        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        full_pose = torch.cat([poses, head_poses, left_hand_pose, right_hand_pose], dim=1)
        
        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)
        joints73_17 = self.vertex_joint_selector(vertices, joints)

        if root_align:
            # use the Pelvis of most 2D image, not the original Pelvis
            root_trans = joints73_17[: ,[constants.SMPL_ALL_44['R_Hip'],constants.SMPL_ALL_44['L_Hip']]].mean(1).unsqueeze(1)
            joints73_17 = joints73_17 - root_trans
            vertices = vertices - root_trans

        return vertices, joints73_17

        if return_shaped: # just to supervise the pure shape
            v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
