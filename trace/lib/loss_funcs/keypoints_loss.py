from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sys, os
import constants
import cv2

import time
import pickle
import numpy as np
import torch.nn.functional as F
from evaluation import compute_error_verts, compute_similarity_transform, compute_similarity_transform_torch, \
                    batch_compute_similarity_transform_torch, compute_mpjpe


def batch_kp_2d_l2_loss(real, pred, images):
    """ 
    Directly supervise the 2D coordinates of global joints, like torso
    While supervise the relative 2D coordinates of part joints, like joints on face, feets
    """
    # unlabelled joints have been set to -2. during data pre-processing
    vis_mask = ((real > -1.99).sum(-1) == real.shape[-1]).float()

    for parent_joint, leaf_joints in constants.joint2D_tree.items():
        parent_id = constants.SMPL_ALL_44[parent_joint]
        leaf_ids = np.array([constants.SMPL_ALL_44[leaf_joint] for leaf_joint in leaf_joints])
        vis_mask[:, leaf_ids] = vis_mask[:, [parent_id]] * vis_mask[:, leaf_ids]
        real[:, leaf_ids] -= real[:, [parent_id]]
        pred[:, leaf_ids] -= pred[:, [parent_id]]
    bv_mask = torch.logical_and(vis_mask.sum(-1) > 0, (real - pred).sum(-1).sum(-1) != 0)
    vis_mask = vis_mask[bv_mask]
    loss = 0
    if vis_mask.sum() > 0:
        #show_kp2ds(real, pred, bv_mask, vis_mask, images)
        # diff = F.mse_loss(real[bv_mask], pred[bv_mask]).sum(-1)
        diff = torch.norm(real[bv_mask] - pred[bv_mask], p=2, dim=-1)
        loss = (diff * vis_mask).sum(-1) / (vis_mask.sum(-1) + 1e-4)
        # loss = (torch.norm(real[bv_mask]-pred[bv_mask],p=2,dim=-1) * vis_mask).sum(-1) / (vis_mask.sum(-1)+1e-4)

        if torch.isnan(loss).sum() > 0 or (loss > 1000).sum() > 0:
            return 0
    
    return loss

def show_kp2ds(real, pred, bv_mask, vis_mask, images):
    from visualization.visualization import draw_skeleton
    for ind in range(len(images)):
        kpgt = (real[ind][(bv_mask[ind]*vis_mask[ind]).bool()]+1)/2 * 512
        print(ind, kpgt) #, ((pred[ind][(bv_mask[ind]*vis_mask[ind]).bool()]+1)/2 * 512)
        ri, pi = ((real[ind]+1)/2 * 512), ((pred[ind]+1)/2 * 512)
        image = images[ind].cpu().numpy().astype(np.uint8)
        r_img = draw_skeleton(image.copy(), ri.detach().cpu().numpy(), bones=constants.All73_connMat, cm=constants.cm_All54)
        p_img = draw_skeleton(image.copy(), pi.detach().cpu().numpy(), bones=constants.All73_connMat, cm=constants.cm_All54)
        cv2.imshow('r_p', np.concatenate([r_img, p_img],1))
        cv2.waitKey()


def calc_pj2d_error(real, pred, joint_inds=None):
    if joint_inds is not None:
        real, pred = real[:,joint_inds], pred[:,joint_inds]
    vis_mask = ((real>-1.99).sum(-1)==real.shape[-1])
    bv_mask = torch.logical_and(vis_mask.float().sum(-1)>0, (real-pred).sum(-1).sum(-1)!=0)
    batch_errors = torch.zeros(len(pred))
    for bid in torch.where(bv_mask)[0]:
        vmask = vis_mask[bid]
        diff = torch.norm((real[bid][vmask]-pred[bid][vmask]), p=2, dim=-1).mean()
        batch_errors[bid] = diff.item()
    return batch_errors

def batch_kp_2d_l2_loss_old(real, pred, top_limit=100):
    vis_mask = (real > -1.99).sum(-1) == real.shape[-1]
    #print('kp2ds masked real pred',real[vis_mask],pred[vis_mask])
    loss = torch.norm(real[vis_mask]-pred[vis_mask], p=2, dim=-1)
    loss = loss[~torch.isnan(loss)] # to avoid nan value
    loss = loss[loss<top_limit] # to avoid inf value
    if len(loss) == 0:
        return 0
    return loss

def batch_kp_2d_l2_loss_with_uncertainty(real, pred, uncertainty):
    vis = (real>-1.).sum(-1)==real.shape[-1]
    if vis.sum()>0:
        valid1, valid2 = torch.where(vis)
        diff = pred[valid1, valid2] - real[valid1, valid2]
        uc_xy = torch.sqrt(uncertainty[valid1, :2]**2)
        error = (torch.sqrt((diff/(uc_xy+1e-6))**2) + torch.log(2*uc_xy)).sum(-1)
        loss = error.sum(-1) / (1e-6+vis.sum(-1))
    else:
        loss = 0
    return loss

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

def calc_pck(real, pred, align_inds=None, pck_joints=None):
    vis_mask = real[:,:,0] != -2.
    pred_aligned = align_by_parts(pred,align_inds=align_inds)
    real_aligned = align_by_parts(real,align_inds=align_inds)
    mpjpe_pck_batch = compute_mpjpe(pred_aligned, real_aligned, vis_mask, pck_joints=pck_joints)
    return mpjpe_pck_batch

