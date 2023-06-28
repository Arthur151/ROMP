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
        #SMPL(model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, gender='neutral',create_transl=False)
        if use_gender:
            self.smpls['f'] = SMPL(args().smpl_model_path.replace('NEUTRAL', 'FEMALE'))
            #SMPL(model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, gender='female',create_transl=False)
            self.smpls['m'] = SMPL(args().smpl_model_path.replace('NEUTRAL', 'MALE'))
            #SMPL(model_path,J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, gender='male',create_transl=False)

    def forward(self, pose, betas, gender='n', root_align=True):
        if isinstance(pose, np.ndarray):
            pose, betas = torch.from_numpy(pose).float(),torch.from_numpy(betas).float()
        if len(pose.shape)==1:
            pose, betas = pose.unsqueeze(0), betas.unsqueeze(0)
        verts, joints44_17 = self.smpls[gender](poses=pose, betas=betas, root_align=root_align)

        return verts.numpy(), joints44_17[:,:args().joint_num].numpy()