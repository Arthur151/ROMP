import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import config
import constants
from config import args

def process_cam_params(cam_maps):
    # to make sure that scale is always a positive value
    cam_maps[..., 0] = (cam_maps[..., 0] + 1.) / 2.
    return cam_maps

def convert_scale_to_depth(scale):
    return 1 / (scale * tan_fov + 1e-3)

def denormalize_cam_params_to_trans(normed_cams, positive_constrain=False):
    #trans = torch.zeros_like(normed_cams)
    #tan_fov = np.tan(np.radians(args().FOV/2.))
    #convert the predicted camera parameters to 3D translation in camera space.
    scale = normed_cams[:, 0]
    if positive_constrain:
        positive_mask = (normed_cams[:, 0] > 0).float()
        scale = scale * positive_mask

    trans_XY_normed = torch.flip(normed_cams[:, 1:],[1])
    # convert from predicted scale to depth
    depth = 1 / (scale * tan_fov + 1e-3).unsqueeze(1)
    # convert from predicted X-Y translation on image plane to X-Y coordinates on camera space.
    trans_XY = trans_XY_normed * depth * tan_fov
    trans = torch.cat([trans_XY, depth], 1)

    return trans
