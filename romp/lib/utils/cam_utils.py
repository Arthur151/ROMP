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

def convert_cam_to_3d_trans(cams, tan_fov=np.tan(np.radians(args().FOV/2.))):
    trans3d = []
    (s, tx, ty) = cams
    depth, dx, dy = 1./s, tx/s, ty/s
    trans3d = np.array([dx, dy, depth])/tan_fov
    return trans3d