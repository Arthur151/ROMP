from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sys, os
import constants

import time
import pickle
import numpy as np
import torch.nn.functional as F
from evaluation import compute_error_verts, compute_similarity_transform, compute_similarity_transform_torch, \
                    batch_compute_similarity_transform_torch, compute_mpjpe


def batch_kp_2d_l2_loss(real, pred):
    """ 
    Directly supervise the 2D coordinates of global joints, like torso
    While supervise the relative 2D coordinates of part joints, like joints on face, feets
    """
    # invisible joints have been set to -2. in data pre-processing
    vis_mask = ((real > -1.99).sum(-1) == real.shape[-1]).float()

    for parent_joint, leaf_joints in constants.joint2D_tree.items():
        parent_id = constants.SMPL_ALL_54[parent_joint]
        leaf_ids = np.array([constants.SMPL_ALL_54[leaf_joint] for leaf_joint in leaf_joints])
        vis_mask[:, leaf_ids] = vis_mask[:, [parent_id]] * vis_mask[:, leaf_ids]
        real[:, leaf_ids] -= real[:, [parent_id]]
        pred[:, leaf_ids] -= pred[:, [parent_id]]
    bv_mask = torch.logical_and(vis_mask.sum(-1) > 0, (real - pred).sum(-1).sum(-1) != 0)
    vis_mask = vis_mask[bv_mask]
    loss = 0
    if vis_mask.sum() > 0:
        # diff = F.mse_loss(real[bv_mask], pred[bv_mask]).sum(-1)
        diff = torch.norm(real[bv_mask] - pred[bv_mask], p=2, dim=-1)
        loss = (diff * vis_mask).sum(-1) / (vis_mask.sum(-1) + 1e-4)
        # loss = (torch.norm(real[bv_mask]-pred[bv_mask],p=2,dim=-1) * vis_mask).sum(-1) / (vis_mask.sum(-1)+1e-4)

        if torch.isnan(loss).sum() > 0 or (loss > 1000).sum() > 0:
            return 0
            print('CAUTION: meet nan of pkp2d loss again!!!!')
            non_position = torch.isnan(loss)
            print('batch_kp_2d_l2_loss, non_position:', non_position, \
                  'diff results', diff, \
                  'real kp 2d vis', real[bv_mask][non_position][vis_mask[non_position].bool()], \
                  'pred kp 2d vis', pred[bv_mask][non_position][vis_mask[non_position].bool()])
            return 0
    return loss

def calc_pj2d_error(real, pred, joint_inds=None):
    if joint_inds is not None:
        real, pred = real[:,joint_inds], pred[:,joint_inds]
    vis_mask = ((real>-1.99).sum(-1)==real.shape[-1])
    bv_mask = torch.logical_and(vis_mask.float().sum(-1)>0, (real-pred).sum(-1).sum(-1)!=0)
    batch_errors = torch.ones(len(pred)) * 10000
    for bid in torch.where(bv_mask)[0]:
        vmask = vis_mask[bid]
        diff = torch.norm((real[bid][vmask]-pred[bid][vmask]), p=2, dim=-1).mean()
        batch_errors[bid] = diff.item()
    return batch_errors

def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)

def calc_mpjpe(real, pred, align_inds=None, sample_wise=True, trans=None, return_org=False):
    vis_mask = real[:,:,0] != -2.
    if align_inds is not None:
        pred_aligned = align_by_parts(pred,align_inds=align_inds)
        if trans is not None:
            pred_aligned += trans
        real_aligned = align_by_parts(real,align_inds=align_inds)
    else:
        pred_aligned, real_aligned = pred, real
    mpjpe_each = compute_mpjpe(pred_aligned, real_aligned, vis_mask, sample_wise=sample_wise)
    if return_org:
        return mpjpe_each, (real_aligned, pred_aligned, vis_mask)
    return mpjpe_each

def calc_pampjpe(real, pred, sample_wise=True,return_transform_mat=False):
    real, pred = real.float(), pred.float()
    # extracting the keypoints that all samples have the annotations
    vis_mask = (real[:,:,0] != -2.).sum(0)==len(real)
    pred_tranformed, PA_transform = batch_compute_similarity_transform_torch(pred[:,vis_mask], real[:,vis_mask])
    pa_mpjpe_each = compute_mpjpe(pred_tranformed, real[:,vis_mask], sample_wise=sample_wise)
    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each


def _calc_pck_loss(real_3d, predicts, PCK_thresh = 0.05, align_inds=None):
    SMPL_MAJOR_JOINTS = np.array([1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21])
    mpjpe_pck_batch = calc_pck(real_3d, predicts, align_inds=align_inds, pck_joints=SMPL_MAJOR_JOINTS)
    mpjpe_pck_sellected = mpjpe_pck_batch[mpjpe_pck_batch>PCK_thresh] - PCK_thresh
    return mpjpe_pck_sellected

def calc_pck(real, pred, align_inds=None, pck_joints=None):
    vis_mask = real[:,:,0] != -2.
    pred_aligned = align_by_parts(pred,align_inds=align_inds)
    real_aligned = align_by_parts(real,align_inds=align_inds)
    mpjpe_pck_batch = compute_mpjpe(pred_aligned, real_aligned, vis_mask, pck_joints=pck_joints)
    return mpjpe_pck_batch

