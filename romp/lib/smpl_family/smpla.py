import torch
from smpl_family.smpl import SMPL
import torch.nn as nn

class SMPLA_parser(nn.Module):
    def __init__(self, smpla_path, smil_path, baby_thresh=0.8):
        super(SMPLA_parser, self).__init__()
        self.smil_model = SMPL(smil_path, model_type='smpl')
        self.smpla_model = SMPL(smpla_path, model_type='smpla')
        self.baby_thresh = baby_thresh
    
    def forward(self, betas=None, poses=None, root_align=True):
        baby_mask = betas[:,10] > self.baby_thresh
        if baby_mask.sum()>0:
            adult_mask = ~baby_mask
            verts, joints = torch.zeros(len(poses), 6890, 3, device=poses.device, dtype=poses.dtype), torch.zeros(len(poses), 54+17, 3, device=poses.device, dtype=poses.dtype)

            # SMIL beta - 10 dims, only need the estimated betas, kid_offsets are not used
            verts[baby_mask], joints[baby_mask] = self.smil_model(betas=betas[baby_mask,:10], poses=poses[baby_mask], root_align=root_align)
            
            # SMPLA beta - 11 dims, the estimated betas (10) + kid_offsets (1)
            if adult_mask.sum()>0:
                verts[adult_mask], joints[adult_mask] = self.smpla_model(betas=betas[adult_mask,:11], poses=poses[adult_mask], root_align=root_align)
        else:
            verts, joints = self.smpla_model(betas=betas[:,:11], poses=poses, root_align=root_align)

        return verts, joints