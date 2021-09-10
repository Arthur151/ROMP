import sys,os
import torch
import torch.nn as nn
import config
import numpy as np 
from .smpl import SMPL
sys.path.append(os.path.abspath(__file__).replace('models/smpl_regressor.py',''))
from config import args

class SMPLR(nn.Module):
    def __init__(self, use_gender=False):
        super(SMPLR, self).__init__()
        model_path = args().smpl_model_path
        J_reg_extra_path = args().smpl_J_reg_extra_path
        
        if use_gender:
            self.smpl_female = SMPL(model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, gender='female',create_transl=False)
            self.smpl_male = SMPL(model_path,J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, gender='male',create_transl=False)
            self.smpls = {'f':self.smpl_female, 'm':self.smpl_male}
        else:
            self.smpl_neutral = SMPL(model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, gender='neutral',create_transl=False)
            self.smpls = {'n':self.smpl_neutral}

    def forward(self, pose, betas, gender='n'):
        if isinstance(pose, np.ndarray):
            pose, betas = torch.from_numpy(pose).float(),torch.from_numpy(betas).float()
        if len(pose.shape)==1:
            pose, betas = pose.unsqueeze(0), betas.unsqueeze(0)
        outputs = self.smpls[gender](poses=pose, betas=betas)

        return outputs