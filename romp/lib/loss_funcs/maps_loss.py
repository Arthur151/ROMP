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

DEFAULT_DTYPE = torch.float32

def focal_loss(pred, gt):
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

    # log(0) lead to nan loss, collipsed
    pred_log = pred.clone()
    pred_log[pred<1e-6] = 1e-6
    pred_log[pred>1-1e-6] = 1-1e-6
    pos_loss = torch.log(pred_log) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred_log) * torch.pow(pred, 2) * neg_weights * neg_inds

    # if not visible or not labelled, ignore the corresponding joints loss
    num_pos  = pos_inds.float().sum(-1).sum(-1)
    pos_loss = pos_loss.sum(-1).sum(-1)
    neg_loss = neg_loss.sum(-1).sum(-1)
    mask = num_pos>0
    #loss[~mask] = loss[~mask] - neg_loss[~mask]
    loss[mask] = loss[mask] - (pos_loss[mask] + neg_loss[mask]) / num_pos[mask]
    return loss.mean(-1)

def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp.cuda()
    return inp


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
        np.arange(0, 32).reshape(1, 2, 4, 4).astype(np.float)*0.1,
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