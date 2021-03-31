import torch
import numpy as np
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import constants
from config import args

def denormalize_center(center, size=512):
    center = ((center+1)/2*size).long()
    center[center<1] = 1
    center[center>size - 1] = size - 1
    return center

def process_gt_center(center_normed):
    center_list = []
    valid_mask = center_normed[:,:,0]>-1
    valid_inds = torch.where(valid_mask)
    valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
    center_gt = ((center_normed+1)/2*args.centermap_size).long()
    center_gt_valid = center_gt[valid_mask]
    return (valid_batch_inds, valid_person_ids, center_gt_valid)