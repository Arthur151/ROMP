from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sys
import os
import config

import time
import pickle
import numpy as np
from utils.center_utils import denormalize_center

DEFAULT_DTYPE = torch.float32

def motion_offset_loss(trajectory3Ds,trajectory2Ds,motion_offsets,):
    """ matching motion offsets with gt trajectorys according to the subject id of matched centers, then calculate the offset error. """
    pass

trajectory_weight = torch.Tensor([[0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4]]).float()

def dynamic_center3d_loss_tragectory_matching(Center3D, Cam3Ds, trajectory2Ds, Tj_vmask, reorganize_idx, dynamic_supervise_ids, CM, target_frame_id=3):
    dynamic_center3d_loss = 0
    if Tj_vmask.sum()>0:
        valid_bids, valid_xyzs = [], []
        with torch.no_grad():
            detached_Cam3Ds = Cam3Ds.detach()
            for bid, batch_id in enumerate(dynamic_supervise_ids):
                Cam3D = detached_Cam3Ds[bid]
                subject_ids = torch.where(reorganize_idx==batch_id)[0]
                subject_ids = subject_ids[Tj_vmask[subject_ids]]
                valid_pc_xyz = []
                for sid in subject_ids:
                    cyx = trajectory2Ds[sid, target_frame_id]
                    cy,cx = denormalize_center(cyx, size=Center3D.shape[-1])
                    Tj_list = Cam3D[:, :, cy, cx].transpose(1,0).reshape(32,7,3)
                    Tj2D_candidate = Tj_list[:,:,:2]
                    trajectory_dist = torch.sqrt(((trajectory2Ds[[sid]]-Tj2D_candidate)**2).sum(-1)) * trajectory_weight.to(trajectory2Ds.device)
                    closet_depth = torch.argmin(trajectory_dist.mean(-1))
                    valid_pc_xyz.append([cx.item(),cy.item(), closet_depth.item()])
                
                if len(valid_pc_xyz)>0:
                    valid_pc_xyz = np.array(valid_pc_xyz)
                    valid_xyzs.append(valid_pc_xyz)
                    valid_bids.append(bid)
        if len(valid_xyzs)>0:
            gt_Center3D = CM.generate_centermap_3dheatmap_adaptive_scale_batch(valid_xyzs, device=Center3D.device)
            dynamic_center3d_loss = focal_loss_3D(Center3D[np.array(valid_bids)], gt_Center3D)
            del gt_Center3D
    return dynamic_center3d_loss


def dynamic_center3d_loss(preds, person_centers, CM, st=1):
    gt_cmap3Ds, valid_bids, valid_xyzs = [], [], []
    with torch.no_grad():
        detached_preds = preds.detach()
        for bid, (centermap_3D, cyx) in enumerate(zip(detached_preds, person_centers)):
            valid_mask = cyx[:,0]>0
            if valid_mask.sum()==0:
                continue
            valid_pc_zyx = []
            for vid in torch.where(valid_mask)[0]:
                (cy, cx) = cyx[vid]
                area = centermap_3D[:, cy-st:cy+st, cx-st:cx+st]
                lz, ly, lx = area.shape
                ranked_indices = torch.sort(area.reshape(-1),descending=True).indices.cpu()
                ranked_indices = torch.stack([ranked_indices // (ly * lx), (ranked_indices % (ly * lx)) // lx, ranked_indices % (ly * lx) % lx],dim=1).float()
                for loc in ranked_indices:
                    if len(valid_pc_zyx)==0:
                        valid_pc_zyx.append(loc)
                        break
                    if torch.norm(torch.stack(valid_pc_zyx)-loc[None], p=2, dim=-1).min()>=st:
                        valid_pc_zyx.append(loc)
                        break
            if len(valid_pc_zyx)>0:
                valid_pc_xyz = torch.stack(valid_pc_zyx)[:,[2,1,0]].numpy()
                valid_xyzs.append(valid_pc_xyz)
                valid_bids.append(bid)
    if len(valid_xyzs)>0:
        gt_cmap3Ds = CM.generate_centermap_3dheatmap_adaptive_scale_batch(valid_xyzs, device=preds.device)
        return focal_loss_3D(preds[np.array(valid_bids)], gt_cmap3Ds)
    else:
        return 0


def dynamic_center3d_loss_adaptive_scale(preds, person_centers, person_scales, CM):
    gt_cmap3Ds, valid_bids = [], []
    with torch.no_grad():
        detached_preds = preds.detach()
        for bid, (centermap_3D, cyx, cs) in enumerate(zip(detached_preds, person_centers, person_scales)):
            valid_mask = (cyx[:,0]>0) * (cs>0)
            if valid_mask.sum()==0:
                continue
            valid_pc_zyx = []
            for vid in torch.where(valid_mask)[0]:
                (cy, cx), s = cyx[vid], cs[vid]//2
                area = centermap_3D[:, cy-s:cy+s, cx-s:cx+s]
                lz, ly, lx = area.shape
                ranked_indices = torch.sort(area.reshape(-1),descending=True).indices.cpu()
                ranked_indices = torch.stack([ranked_indices // (ly * lx), (ranked_indices % (ly * lx)) // lx, ranked_indices % (ly * lx) % lx],dim=1).float()
                for loc in ranked_indices:
                    if len(valid_pc_zyx)==0:
                        valid_pc_zyx.append(loc)
                        break
                    if torch.norm(torch.stack(valid_pc_zyx)-loc[None], p=2, dim=-1).min()>=s.cpu():
                        valid_pc_zyx.append(loc)
                        break
            if len(valid_pc_zyx)>0:
                valid_pc_xyz = torch.stack(valid_pc_zyx)[:,[2,1,0]].numpy()
                gt_cmap3D, flag = CM.generate_centermap_3dheatmap_adaptive_scale(valid_pc_xyz)
                if flag:
                    valid_bids.append(bid)
                    gt_cmap3Ds.append(gt_cmap3D)
    if len(gt_cmap3Ds)>0:
        gt_cmap3Ds = torch.stack(gt_cmap3Ds).to(preds.device).float()
        return focal_loss_3D(preds[np.array(valid_bids)], gt_cmap3Ds)
    else:
        return 0


def constraining_log_inputs(x, up=1-1e-4, low=1e-4):
    if low is not None:
        x[x < low] = x[x < low] - (x[x < low] - low)
    if up is not None:
        x[x > (1-1e-4)] = x[x > (1-1e-4)] - (x[x > (1-1e-4)] - (1-1e-4))
    return x


def focal_loss(pred, gt, max_loss_limit=0.8):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = torch.zeros(gt.size(0)).to(pred.device)

    # when x <=0, log(x) lead to nan loss, collipsed
    # To take care of 1-pred_log in neg_loss, pred_log < 1-1e-4
    pred_log = torch.clamp(pred.clone(), min=1e-3, max=1-1e-3)
    pos_loss = torch.log(pred_log) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred_log) * torch.pow(pred, 2) * neg_weights * neg_inds

    # if not visible or not labelled, ignore the corresponding joints loss
    num_pos  = pos_inds.float().sum(-1).sum(-1)
    pos_loss = pos_loss.sum(-1).sum(-1)
    neg_loss = neg_loss.sum(-1).sum(-1)

    mask = num_pos>0
    loss[~mask] = loss[~mask] - neg_loss[~mask]
    loss[mask] = loss[mask] - (pos_loss[mask] + neg_loss[mask]) / (num_pos[mask]+1e-4) # 
    #print('Centermap focal loss', loss)
    while (loss > max_loss_limit).sum() > 0:
        exclude_mask = loss > max_loss_limit
        #print('huge loss of centermap', loss[exclude_mask])
        loss[exclude_mask] = loss[exclude_mask] / 4
    return loss.mean(-1)

def focal_loss_3D(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x z x h x w)
      gt_regr (batch x z x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = torch.zeros(gt.size(0)).to(pred.device)

    pred[torch.isnan(pred)] = 1e-3
    # log(0) lead to nan loss, collipsed
    # To take care of 1-pred_log in neg_loss, pred_log < 1-1e-4
    pred_log = torch.clamp(pred.clone(), min=1e-3, max=1-1e-3)
    pos_loss = torch.log(pred_log) * torch.pow(1 - pred, 2) * pos_inds
    #if torch.isnan(pos_loss).sum()>0:
    #    print(torch.log(pred_log)[torch.isnan(pos_loss)], torch.pow(1 - pred, 2)[torch.isnan(pos_loss)])
    neg_loss = torch.log(1 - pred_log) * torch.pow(pred, 2) * neg_weights * neg_inds
    #if torch.isnan(neg_loss).sum()>0:
    #    print(torch.log(1 - pred_log)[torch.isnan(neg_loss)], pred_log[torch.isnan(neg_loss)])

    # if not visible or not labelled, ignore the corresponding joints loss
    num_pos  = pos_inds.float().sum(-1).sum(-1).sum(-1)
    pos_loss = pos_loss.sum(-1).sum(-1).mean(-1)
    neg_loss = neg_loss.sum(-1).sum(-1).mean(-1)
    mask = num_pos>0
    loss[~mask] = loss[~mask] - neg_loss[~mask]
    loss[mask] = loss[mask] - (pos_loss[mask] + neg_loss[mask]) / (num_pos[mask]+1e-4)

    if torch.isnan(loss).sum()>0:
       print('focal_loss_3D nan index {} : pos {} | neg {}'.format(torch.where(torch.isnan(loss)), pos_loss[torch.isnan(loss)], neg_loss[torch.isnan(loss)]))
       loss[torch.isnan(loss)] = 0
    return loss.mean(-1)

def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp.cuda()
    return inp

class Centermap3dLoss(nn.Module):
    def __init__(self,):
        pass

class HeatmapLoss(nn.Module):
    def __init__(self, loss_type='MSE'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, gt):
        assert pred.size() == gt.size(), print('pred, gt heatmap size mismatch: {}|{}'.format(pred.size(), gt.size()))

        if self.loss_type == 'focal':
            loss = focal_loss(pred, gt)
        elif self.loss_type == 'MSE':
            mask = gt.float().sum(dim=3).sum(dim=2).gt(0).float()
            loss = (((pred - gt)**2).mean(dim=3).mean(dim=2) * mask).sum()/mask.sum()
        else:
            raise NotImplementedError
        return loss


class AELoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """
        associative embedding loss for one image
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return make_input(torch.zeros(1).float()), \
                make_input(torch.zeros(1).float())
        elif num_tags == 1:
            return make_input(torch.zeros(1).float()), \
                pull/(num_tags)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')

        return push/((num_tags - 1) * num_tags) * 0.5, \
            pull/(num_tags)

    def forward(self, tags, joints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class Heatmap_AE_loss(nn.Module):
    def __init__(self, num_joints, loss_type_HM='MSE', loss_type_AE='exp'):
        super().__init__()
        self.num_joints = num_joints
        self.heatmaps_loss = HeatmapLoss(loss_type_HM)
        self.heatmaps_loss_factor = 1.
        self.ae_loss = AELoss(loss_type_AE)
        self.push_loss_factor = 1. #0.001 #1.
        self.pull_loss_factor = 1. #0.001 #0.1

    def forward(self, outputs, heatmaps, joints):
        # TODO(bowen): outputs and heatmaps can be lists of same length
        heatmaps_pred = outputs[:, :self.num_joints]
        tags_pred = outputs[:, self.num_joints:]

        heatmaps_loss = None
        push_loss = None
        pull_loss = None

        if self.heatmaps_loss is not None:
            heatmaps_loss = self.heatmaps_loss(heatmaps_pred, heatmaps)
            heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor

        if self.ae_loss is not None:
            batch_size = tags_pred.size()[0]
            tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

            push_loss, pull_loss = self.ae_loss(tags_pred, joints)
            push_loss = push_loss * self.push_loss_factor
            pull_loss = pull_loss * self.pull_loss_factor

        return heatmaps_loss, push_loss, pull_loss


def test_ae_loss():
    import numpy as np
    t = torch.tensor(
        np.arange(0, 32).reshape(1, 2, 4, 4).astype(np.float32)*0.1,
        requires_grad=True
    )
    t.register_hook(lambda x: print('t', x))

    ae_loss = AELoss(loss_type='exp')

    joints = np.zeros((2, 2, 2))
    joints[0, 0] = (3, 1)
    joints[1, 0] = (10, 1)
    joints[0, 1] = (22, 1)
    joints[1, 1] = (30, 1)
    joints = torch.LongTensor(joints)
    joints = joints.view(1, 2, 2, 2)

    t = t.contiguous().view(1, -1, 1)
    l = ae_loss(t, joints)

    print(l)


if __name__ == '__main__':
    from utils.target_generators import HeatmapGenerator
    num_joints = 25
    output_res = 128
    hg = HeatmapGenerator(output_res,num_joints)

    x = torch.rand(2,num_joints,2).cuda()*2-1
    x[0,:2] = -2.
    heatmaps = hg.batch_process(x)
    print(heatmaps)
    loss = focal_loss(torch.sigmoid(heatmaps+torch.rand(1,25,128,128).cuda()),heatmaps)
    print(loss)
    #test_ae_loss()
