from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import time
import pickle
import numpy as np

import config
import constants
from config import args
from utils import batch_rodrigues, rotation_matrix_to_angle_axis


def batch_l2_loss(real,predict):
    loss_batch = torch.norm(real-predict, p=2, dim=1)
    return loss_batch.mean()

def batch_l2_loss_param(real,predict):
    # convert to rot mat, multiple angular maps to the same rotation with Pi as a period.
    batch_size = real.shape[0]
    real = batch_rodrigues(real.reshape(-1,3)).contiguous()#(N*J)*3 -> (N*J)*3*3
    predict = batch_rodrigues(predict.reshape(-1,3)).contiguous()#(N*J)*3 -> (N*J)*3*3
    loss = torch.norm((real-predict).view(-1,9), p=2, dim=-1)#self.sl1loss(real,predict)#
    loss = loss.reshape(batch_size, -1).mean(-1)
    return loss

def _calc_MPJAE(rel_pose_pred,rel_pose_real):
    global_pose_rotmat_pred = trans_relative_rot_to_global_rotmat(rel_pose_pred, with_global_rot=True)
    global_pose_rotmat_real = trans_relative_rot_to_global_rotmat(rel_pose_real, with_global_rot=True)
    MPJAE_error = _calc_joint_angle_error(global_pose_rotmat_pred, global_pose_rotmat_real).cpu().numpy()
    return MPJAE_error


def trans_relative_rot_to_global_rotmat(params, with_global_rot=False):
    '''
    calculate absolute rotation matrix in the global coordinate frame of K body parts. 
    The rotation is the map from the local bone coordinate frame to the global one.
    K= 9 parts in the following order: 
    root (JOINT 0) , left hip  (JOINT 1), right hip (JOINT 2), left knee (JOINT 4), right knee (JOINT 5), 
    left shoulder (JOINT 16), right shoulder (JOINT 17), left elbow (JOINT 18), right elbow (JOINT 19).
    parent kinetic tree [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    '''
    batch_size, param_num = params.shape[0], params.shape[1]//3
    pose_rotmat = batch_rodrigues(params.reshape(-1,3)).view(batch_size, param_num, 3, 3).contiguous()
    if with_global_rot:
        sellect_joints = np.array([0,1,2,4,5,16,17,18,19],dtype=np.int)
        results = [pose_rotmat[:, 0]]
        for idx in range(param_num-1):
            i_val = int(idx + 1)
            joint_rot = pose_rotmat[:, i_val]
            parent = constants.kintree_parents[i_val]
            glob_transf_mat = torch.matmul(results[parent], joint_rot)
            results.append(glob_transf_mat)
    else:
        sellect_joints = np.array([1,2,4,5,16,17,18,19],dtype=np.int)-1
        results = [torch.eye(3,3)[None].cuda().repeat(batch_size,1,1)]
        for i_val in range(param_num-1):
            #i_val = int(idx + 1)
            joint_rot = pose_rotmat[:, i_val]
            parent = constants.kintree_parents[i_val+1]
            glob_transf_mat = torch.matmul(results[parent], joint_rot)
            results.append(glob_transf_mat)
    global_rotmat = torch.stack(results, axis=1)[:, sellect_joints].contiguous()
    return global_rotmat

def _calc_joint_angle_error(pred_mat, gt_mat, return_axis_angle=False):
    """
    Compute the geodesic distance between the two input matrices.
    :param pred_mat: predicted rotation matrices. Shape: ( Seq, 9g, 3, 3)
    :param gt_mat: ground truth rotation matrices. Shape: ( Seq, 9, 3, 3)
    :return: Mean geodesic distance between input matrices.
    """

    # Reshape the matrices into B x 3 x 3 arrays
    r1 = pred_mat.reshape(-1,3,3)
    r2 = gt_mat.reshape(-1,3,3)
    # Transpose gt matrices
    r2t = r2.permute(0,2,1)
    r = torch.matmul(r1, r2t)
    # Convert rotation matrix to axis angle representation and find the angle
    axis_angles = rotation_matrix_to_angle_axis(r)
    angles = torch.norm(axis_angles, dim=-1)*(180./np.pi)

    if return_axis_angle:
        return angles,axis_angles
    return angles
