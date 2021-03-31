import torch
import numpy as np

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import constants
from config import args

def vertices_kp3d_projection(outputs):
    params_dict, vertices, j3ds = outputs['params'], outputs['verts'], outputs['j3d']
    verts_camed = batch_orth_proj(vertices, params_dict['cam'], mode='3d',keep_dim=True)
    pj3d = batch_orth_proj(j3ds, params_dict['cam'], mode='2d')
    projected_outputs = {'verts_camed': verts_camed, 'pj2d': pj3d[:,:,:2]}
    return projected_outputs


def project_2D(kp3d, cams,keep_dim=False):
    d,f, t = cams[0], cams[1], cams[2:].unsqueeze(0)
    pose2d = kp3d[:,:2]/(kp3d[:,2][:,None]+d)
    pose2d = pose2d*f+t
    if keep_dim:
        kp3d[:,:2] = pose2d
        return kp3d
    else:
        return pose2d

def batch_orth_proj(X, camera, mode='2d',keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed

def preprocess_kps(kps,image,set_minus=False):
    kps[:,0] /= image.shape[0]
    kps[:,1] /= image.shape[1]
    kps[:,:2] = 2.0 * kps[:,:2] - 1.0

    if kps.shape[1]>2 and set_minus:
        kps[kps[:,2]<0,:2] = -2.
    kps=kps[:,:2]
    return kps