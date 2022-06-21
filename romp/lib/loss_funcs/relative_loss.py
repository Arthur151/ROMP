from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sys, os

import time
import pickle
import numpy as np

import config
import constants
from config import args
from utils.cam_utils import convert_scale_to_depth


def match_batch_subject_ids(reorganize_idx, subject_ids, torso_pj2d_errors, a_id, b_id, pj2d_thresh=0.1):
    matched_inds = [[],[]]
    a_mask = reorganize_idx == a_id
    b_mask = reorganize_idx == b_id
    # intersection of two sets
    all_subject_ids = set(subject_ids[a_mask].cpu().numpy()).intersection(set(subject_ids[b_mask].cpu().numpy()))
    if len(all_subject_ids) == 0:
        return matched_inds
    
    for ind, sid in enumerate(all_subject_ids):
        a_ind = torch.where(torch.logical_and(subject_ids == sid, a_mask))[0][0]
        b_ind = torch.where(torch.logical_and(subject_ids == sid, b_mask))[0][0]
        a_error, b_error = torso_pj2d_errors[a_ind], torso_pj2d_errors[b_ind]
        if a_error<pj2d_thresh and b_error<pj2d_thresh:
            # We sellect the better prediction with lower torso_pj2d_error to serve as anchor gt value matched_inds[0] for supervision,
            # Punish the depth of the prediction with large torso_pj2d_error only, instead of passing the same loss for both items.
            if a_error>b_error:
                matched_inds[0].append(b_ind)
                matched_inds[1].append(a_ind)
            else:
                matched_inds[0].append(a_ind)
                matched_inds[1].append(b_ind)
    matched_inds[0] = torch.Tensor(matched_inds[0]).long()
    matched_inds[1] = torch.Tensor(matched_inds[1]).long()
    return matched_inds


def relative_depth_loss(pred_depths, depth_ids, reorganize_idx, dist_thresh=0.3, uncertainty=None, matched_mask=None):
    depth_ordering_loss = []
    depth_ids = depth_ids.to(pred_depths.device)
    depth_ids_vmask = depth_ids != -1
    pred_depths_valid = pred_depths[depth_ids_vmask]
    valid_inds = reorganize_idx[depth_ids_vmask]
    depth_ids = depth_ids[depth_ids_vmask]
    if uncertainty is not None:
        uncertainty_valid = uncertainty[depth_ids_vmask]
    
    for b_ind in torch.unique(valid_inds):
        sample_inds = valid_inds == b_ind
        if matched_mask is not None:
            sample_inds = sample_inds * matched_mask[depth_ids_vmask]
        did_num = sample_inds.sum()
        if did_num > 1:
            pred_depths_sample = pred_depths_valid[sample_inds]
            triu_mask = torch.triu(torch.ones(did_num, did_num), diagonal=1).bool()
            dist_mat = (pred_depths_sample.unsqueeze(0).repeat(did_num, 1) - pred_depths_sample.unsqueeze(1).repeat(1,did_num))[triu_mask]
            did_mat = (depth_ids[sample_inds].unsqueeze(0).repeat(did_num, 1) - depth_ids[sample_inds].unsqueeze(1).repeat(1,did_num))[triu_mask]
            sample_loss = []
            
            if args().depth_loss_type == 'Piecewise':
                eq_mask = did_mat==0
                cd_mask = did_mat<0
                cd_mask[did_mat<0] = cd_mask[did_mat<0] * (dist_mat[did_mat<0] - did_mat[did_mat<0]*dist_thresh)>0
                fd_mask = did_mat>0
                fd_mask[did_mat>0] = fd_mask[did_mat>0] * (dist_mat[did_mat>0] - did_mat[did_mat>0]*dist_thresh)<0
                if eq_mask.sum()>0:
                    sample_loss.append(dist_mat[eq_mask]**2)
                if cd_mask.sum()>0:
                    cd_loss = torch.log(1+torch.exp(dist_mat[cd_mask]))
                    sample_loss.append(cd_loss)
                if fd_mask.sum()>0:
                    fd_loss = torch.log(1+torch.exp(-dist_mat[fd_mask]))
                    sample_loss.append(fd_loss)
            elif args().depth_loss_type == 'Log':
                eq_loss = dist_mat[did_mat==0]**2
                cd_loss = torch.log(1+torch.exp(dist_mat[did_mat<0]))
                fd_loss = torch.log(1+torch.exp(-dist_mat[did_mat>0]))
                sample_loss = [eq_loss, cd_loss, fd_loss]
            else:
                raise NotImplementedError
            
            if len(sample_loss)>0:
                this_sample_loss = torch.cat(sample_loss).mean()
                depth_ordering_loss.append(this_sample_loss)
    
    if len(depth_ordering_loss) == 0:
        depth_ordering_loss = 0
    else:
        depth_ordering_loss = sum(depth_ordering_loss)/len(depth_ordering_loss)

    return depth_ordering_loss
                                    

def kid_offset_loss(kid_offset_preds, kid_offset_gts, matched_mask=None):
    device = kid_offset_preds.device
    kid_offset_gts = kid_offset_gts.to(device)
    age_vmask = kid_offset_gts != -1
    if matched_mask is not None:
        age_vmask = age_vmask * matched_mask
    if age_vmask.sum()==0:
        return 0
    return ((kid_offset_preds[age_vmask] - kid_offset_gts[age_vmask])**2).mean()


def relative_age_loss(kid_offset_preds, age_gts, matched_mask=None):
    device = kid_offset_preds.device
    age_gts = age_gts.to(device)
    age_vmask = age_gts != -1
    if matched_mask is not None:
        age_vmask = age_vmask * matched_mask
    if age_vmask.sum()==0:
        return 0
    adult_loss = (kid_offset_preds * (age_gts==0))**2
    teen_thresh = constants.age_threshold['teen']
    teen_loss = ((kid_offset_preds - teen_thresh[1]) * (kid_offset_preds>teen_thresh[2]).float() * (age_gts==1).float())**2 + \
                ((kid_offset_preds - teen_thresh[1]) * (kid_offset_preds<=teen_thresh[0]).float() * (age_gts==1).float())**2
    kid_thresh = constants.age_threshold['kid']
    kid_loss = ((kid_offset_preds - kid_thresh[1]) * (kid_offset_preds>kid_thresh[2]).float() * (age_gts==2).float())**2 + \
                ((kid_offset_preds - kid_thresh[1]) * (kid_offset_preds<=kid_thresh[0]).float() * (age_gts==2).float())**2
    baby_thresh = constants.age_threshold['baby']
    baby_loss = ((kid_offset_preds - baby_thresh[1]) * (kid_offset_preds>baby_thresh[2]).float() * (age_gts==3).float())**2 + \
                ((kid_offset_preds - baby_thresh[1]) * (kid_offset_preds<=baby_thresh[0]).float() * (age_gts==3).float())**2
    age_loss = adult_loss.mean() + teen_loss.mean() + kid_loss.mean() + baby_loss.mean()
    # if age_vmask.sum()>0:
    #     age_loss = age_loss[age_vmask].mean()

    return age_loss

def relative_shape_loss(pred_betas, body_type_gts):
    device = pred_betas.device
    body_type_gts = body_type_gts.to(device)
    body_type_vmask = body_type_gts != -1
    
    fat_level_preds = pred_betas[:,1]
    not_fat_loss = (fat_level_preds * (fat_level_preds<-3).float() * (body_type_gts==0).float())**2 + \
                    (fat_level_preds * (fat_level_preds>2).float() * (body_type_gts==0).float())**2
    slightly_fat_loss = ((fat_level_preds+4.5) * (fat_level_preds<-6).float() * (body_type_gts==1).float())**2 + \
                    ((fat_level_preds+4.5) * (fat_level_preds>-3).float() * (body_type_gts==1).float())**2
    fat_loss = ((fat_level_preds+7.5) * (fat_level_preds<-9).float() * (body_type_gts==2).float())**2 + \
                    ((fat_level_preds+7.5) * (fat_level_preds>-6).float() * (body_type_gts==2).float())**2
    body_type_loss = not_fat_loss + slightly_fat_loss + fat_loss
    if body_type_vmask.sum()>0:
        body_type_loss = body_type_loss[body_type_vmask]

    return body_type_loss


def test_depth_ordering_loss():
    pred_cams = torch.rand(6,3)
    depth_info = torch.randint(5,(6,4))
    depth_info[1] = -1
    reorganize_idx = torch.Tensor([0,0,0,1,1,1])
    print('pred_cams', pred_cams)
    print('depth_info', depth_info[:,3])
    print('reorganize_idx', reorganize_idx)
    loss = relative_depth_loss(pred_cams, depth_info, reorganize_idx, dist_thresh=0.3)
    print(loss)
    return loss

def test_relative_shape_loss():
    pred_betas = torch.rand(4,13)
    depth_info = torch.randint(5,(4,4))
    depth_info[:,0] = torch.randint(4,(4,))
    depth_info[:,1] = torch.randint(1,(4,))
    depth_info[:,2] = torch.randint(2,(4,))
    depth_info[1] = -1
    print('pred_betas', pred_betas[:,10:])
    print('depth_info', depth_info)
    loss = relative_shape_loss(pred_betas, depth_info)
    print(loss)


if __name__ == '__main__':
    #test_depth_ordering_loss()
    test_relative_shape_loss()
