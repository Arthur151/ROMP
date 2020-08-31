import sys,os
sys.path.append(os.path.abspath(__file__).replace('utils/smpl_regressor.py',''))
import torch
import torch.nn as nn
import config
import numpy as np 
from models.smpl import SMPL
from utils.jointmapper import JointMapper,smpl_to_openpose

class SMPLR(nn.Module):
    def __init__(self,joint_format='coco25'):
        super(SMPLR, self).__init__()
        self.smpl_male = SMPL(os.path.join(config.model_dir,'smpl_models','smpl'),gender='male',create_transl=False)
        self.smpl_female = SMPL(os.path.join(config.model_dir,'smpl_models','smpl'),gender='female',create_transl=False)
        self.smpl_neutral = SMPL(os.path.join(config.model_dir,'smpl_models','smpl'),gender='neutral',create_transl=False)
        self.smpls = {'f':self.smpl_female, 'm':self.smpl_male, 'n':self.smpl_neutral}

        self.joint_format = joint_format
        self.J_regressor = torch.from_numpy(np.load(os.path.join(config.model_dir,"spin_data/J_regressor_h36m.npy"))).float().unsqueeze(0)
        #self.joint_mapper_smpl = JointMapper(smpl_to_openpose(model_type='smpl', openpose_format='coco25'))

    def forward(self, pose, betas, gender='n',batch=False):
        if not batch:
            pose, betas = torch.from_numpy(pose).unsqueeze(0).float(), torch.from_numpy(betas).unsqueeze(0).float()
        joints = self.smpls[gender](global_orient=pose[:,:3], body_pose=pose[:,3:], betas=betas).joints[:,:24]
        if not batch:
            joints = joints.numpy()

        return joints