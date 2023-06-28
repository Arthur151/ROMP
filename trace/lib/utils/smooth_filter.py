import numpy as np
import torch


def smooth_global_rot_matrix(pred_rots, OE_filter):
    rot_mat = batch_rodrigues(pred_rots[None]).squeeze(0)
    smoothed_rot_mat = OE_filter.process(rot_mat)
    smoothed_rot = rotation_matrix_to_angle_axis(smoothed_rot_mat.reshape(1,3,3)).reshape(-1)
    return smoothed_rot

    device = pred_rots.device
    #print('before',pred_rots)
    rot_euler = transform_rot_representation(pred_rots.cpu().numpy(), input_type='vec',out_type='mat')
    smoothed_rot = OE_filter.process(rot_euler)
    smoothed_rot = transform_rot_representation(smoothed_rot, input_type='mat',out_type='vec')
    smoothed_rot = torch.from_numpy(smoothed_rot).float().to(device)
    #print('after',smoothed_rot)
    return smoothed_rot

class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
        s = value
    else:
        s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s

class OneEuroFilter:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x, print_inter=False):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    
    if isinstance(edx, float):
        cutoff = self.mincutoff + self.beta * np.abs(edx)
    elif isinstance(edx, np.ndarray):
        cutoff = self.mincutoff + self.beta * np.abs(edx)
    elif isinstance(edx, torch.Tensor):
        cutoff = self.mincutoff + self.beta * torch.abs(edx)
    if print_inter:
        print(self.compute_alpha(cutoff))
    return self.x_filter.process(x, self.compute_alpha(cutoff))

def check_filter_state(OE_filters, signal_ID, show_largest=False, smooth_coeff=3.):
    if len(OE_filters)>100:
        del OE_filters
    if signal_ID not in OE_filters:
        if show_largest:
            OE_filters[signal_ID] = create_OneEuroFilter(smooth_coeff)
        else:
            OE_filters[signal_ID] = {}
    if len(OE_filters[signal_ID])>1000:
        del OE_filters[signal_ID]

def create_OneEuroFilter(smooth_coeff):
    return {'smpl_thetas': OneEuroFilter(smooth_coeff, 0.7), 'cam': OneEuroFilter(1.6, 0.7), 'smpl_betas': OneEuroFilter(0.6, 0.7), 'global_rot': OneEuroFilter(smooth_coeff, 0.7)}