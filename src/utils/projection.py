import torch
import constants
from config import args


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
        s = torch.pow(1.1,camera[:, :, 0])
        X_camed = X[:,:,:2] * s.unsqueeze(-1)
        X_camed += camera[:, :, 1:]
        if keep_dim:
            X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
        return X_camed