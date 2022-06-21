import sys,os
import torch
import torch.nn as nn
import config
import numpy as np 
from .smpl import SMPL
from config import args

class SMPLR(nn.Module):
    def __init__(self, use_gender=False):
        super(SMPLR, self).__init__()
        model_path = os.path.join(config.model_dir,'parameters','smpl')
        self.smpls = {}
        self.smpls['n'] = SMPL(args().smpl_model_path, model_type='smpl')
        if use_gender:
            self.smpls['f'] = SMPL(os.path.join(config.smpl_model_dir,'SMPL_FEMALE.pth'))
            self.smpls['m'] = SMPL(os.path.join(config.smpl_model_dir,'SMPL_MALE.pth'))

    def forward(self, pose, betas, gender='n'):
        if isinstance(pose, np.ndarray):
            pose, betas = torch.from_numpy(pose).float(),torch.from_numpy(betas).float()
        if len(pose.shape)==1:
            pose, betas = pose.unsqueeze(0), betas.unsqueeze(0)
        verts, joints54_17 = self.smpls[gender](poses=pose, betas=betas)

        return verts.numpy(), joints54_17[:,:54].numpy()