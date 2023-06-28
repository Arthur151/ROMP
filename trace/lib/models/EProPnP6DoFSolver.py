import torch
from torch import nn

from epropnp.epropnp import EProPnP6DoF
from epropnp.levenberg_marquardt import LMSolver, RSLMSolver
from epropnp.camera import PerspectiveCamera
from epropnp.cost_fun import AdaptiveHuberPnPCost

def prepare_camera_mats(fovs, length, device):
    cam_mats = torch.zeros(length, 3, 3).to(device)
    if fovs is not None:      
        cam_mats[:,0,0] = cam_mats[:,1,1] = fovs
    else:
        cam_mats[:,0,0] = cam_mats[:,1,1] = 1./np.tan(np.radians(args().FOV / 2))
    cam_mats[:,2,2] = 1
    return cam_mats

class EProPnP6DoFSolver(nn.Module):
    def __init__(self,):
        super(EProPnP6DoFSolver, self).__init__()
        self.epropnp = EProPnP6DoF(
                mc_samples=512, num_iter=4,
                solver=LMSolver(dof=6, num_iter=10,
                    init_solver=RSLMSolver(dof=6, num_points=8, num_proposals=128, num_iter=5)))
        self.camera = PerspectiveCamera()
        self.cost_fun = AdaptiveHuberPnPCost(relative_delta=0.5)
        self.log_weight_scale = nn.Parameter(torch.zeros(2))
    
    def epro_pnp_train(self, x3d, x2d, w2d, cam_mats, out_pose):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            self.camera.set_param(cam_mats)
            self.cost_fun.set_param(x2d, w2d)  # compute dynamic delta
            pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = self.epropnp.monte_carlo_forward(
                x3d.detach(), x2d, w2d, self.camera, self.cost_fun,
                pose_init=out_pose, force_init_solve=True, with_pose_opt_plus=True)  # True for derivative regularization loss
            
        return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt
    
    def epro_pnp_inference(self, x3d, x2d, w2d, cam_mats, fast_mode=False):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            self.camera.set_param(cam_mats)
            self.cost_fun.set_param(x2d, w2d)
            # returns a mode of the distribution
            pose_opt, _, _, _ = self.epropnp(
                x3d.detach(), x2d, w2d, self.camera, self.cost_fun,
                fast_mode=fast_mode)  # fast_mode=True activates Gauss-Newton solver (no trust region)
        return pose_opt
    
    def solve(self, x3ds, x2ds, w2ds, fovs, fast_mode=True):
        cam_mats = prepare_camera_mats(fovs, len(x3ds), x3ds.device)
        self.camera.set_param(cam_mats)
        self.cost_fun.set_param(x2ds.detach(), w2ds)
        # returns a mode of the distribution
        pose_opt, _, _, _ = self.epropnp(
            x3ds, x2ds, w2ds, self.camera, self.cost_fun,
            fast_mode=fast_mode)
        return pose_opt