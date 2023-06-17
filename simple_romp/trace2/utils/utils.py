import torch
from torch.nn import functional as F
import numpy as np
import cv2, os

#-----------------------------------------------------------------------------------------#
#                                          save_paths                                     #
#-----------------------------------------------------------------------------------------#

class preds_save_paths(object):
    def __init__(self, results_save_dir, prefix='test'):
        self.seq_save_dir = os.path.join(results_save_dir, prefix)
        os.makedirs(self.seq_save_dir, exist_ok=True)

        self.tracking_matrix_save_path = os.path.join(self.seq_save_dir, 'TRACE_{}.txt'.format(prefix))
        #print('tracking_matrix_save_path', self.tracking_matrix_save_path)
        self.seq_results_save_path = os.path.join(results_save_dir, prefix+'.npz')     
        self.seq_tracking_results_save_path = os.path.join(results_save_dir, prefix+'_tracking.npz')    

def print_dict(td):
    keys = collect_keyname(td)
    print(keys)

def get_size(item):
    if isinstance(item, list) or isinstance(item, tuple):
        return len(item)
    elif isinstance(item, torch.Tensor) or isinstance(item, np.ndarray):
        return item.shape
    else:
        return item

def collect_keyname(td):
    keys = []
    for key in td:
        if isinstance(td[key], dict):
            keys.append([key, collect_keyname(td[key])])
        else:
            keys.append([key, get_size(td[key])])
    return keys

def download_model(remote_url, local_path, name):
    try:
        os.makedirs(os.path.dirname(local_path),exist_ok=True)
        try:
            import wget
        except:
            print('Installing wget to download model data.')
            os.system('pip install wget')
            import wget
        print('Downloading the {} model from {} and put it to {} \n Please download it by youself if this is too slow...'.format(name, remote_url, local_path))
        wget.download(remote_url, local_path)
    except Exception as error:
        print(error)
        print('Failure in downloading the {} model, please download it by youself from {}, and put it to {}'.format(name, remote_url, local_path))


#-----------------------------------------------------------------------------------------#
#                                       load parameters                                   #
#-----------------------------------------------------------------------------------------#

def copy_state_dict(cur_state_dict, pre_state_dict, prefix='module.', drop_prefix='', fix_loaded=False):
    success_layers, failed_layers = [], []
    def _get_params(key):
        key = key.replace(drop_prefix,'')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix!='':
                k=k.split(prefix)[1]
            success_layers.append(k)
        except:
            print('copy param {} failed, mismatched'.format(k)) # logging.info
            continue
    print('missing parameters of layers:{}'.format(failed_layers))

    if fix_loaded and len(failed_layers)>0:
        fixed_layers = []
        for k in cur_state_dict.keys():
            try:
                if k in success_layers:
                    cur_state_dict[k].requires_grad=False
                    fixed_layers.append(k)
            except:
                print('fixing the layer {} failed'.format(k))
        #print('fixed layers:', fixed_layers)

    return success_layers

def load_model(path, model, prefix = 'module.', drop_prefix='',optimizer=None, **kwargs):
    print('using fine_tune model: {}'.format(path))
    if os.path.exists(path):
        pretrained_model = torch.load(path)
        current_model = model.state_dict()
        if isinstance(pretrained_model, dict):
            if 'model_state_dict' in pretrained_model:
                pretrained_model = pretrained_model['model_state_dict']
        copy_state_dict(current_model, pretrained_model, prefix = prefix, drop_prefix=drop_prefix, **kwargs)
    else:
        print('model {} not exist!'.format(path))
    return model

def load_config_dict(self, config_dict):
    hparams_dict = {}
    for i, j in config_dict.items():
        setattr(self,i,j)
        hparams_dict[i] = j
    return hparams_dict

#-----------------------------------------------------------------------------------------#
#                                   pre-processing image                                  #
#-----------------------------------------------------------------------------------------#

def padding_image(image):
    h, w = image.shape[:2]
    side_length = max(h, w)
    pad_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)
    top, left = int((side_length - h) // 2), int((side_length - w) // 2)
    bottom, right = int(top+h), int(left+w)
    pad_image[top:bottom, left:right] = image
    image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
    return pad_image, image_pad_info
    
def img_preprocess(image, input_size=512):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pad_image, image_pad_info = padding_image(image)
    input_image = torch.from_numpy(cv2.resize(pad_image, (input_size,input_size), interpolation=cv2.INTER_CUBIC))[None].float()
    return input_image, image_pad_info

#-----------------------------------------------------------------------------------------#
#                                    Infering reorganize                                  #                                    
#-----------------------------------------------------------------------------------------#

def reorganize_items(items, reorganize_idx):
    items_new = [[] for _ in range(len(items))]
    for idx, item in enumerate(items):
        for ridx in reorganize_idx:
            items_new[idx].append(item[ridx])
    return items_new

#-----------------------------------------------------------------------------------------#
#                                   3D to 2D projection                                   #                                    
#-----------------------------------------------------------------------------------------#
focal_length = 548
tan_fov = np.tan(np.radians(50/2.))

def filter_out_incorrect_trans(kp_3ds, trans, kp_2ds, thresh=20, focal_length=focal_length, center_offset=torch.Tensor([512, 512])/2.):
    valid_mask = np.logical_and(kp_3ds[:,:,-1]!=-2., kp_2ds[:,:,-1]>0)
    projected_kp2ds = perspective_projection(kp_3ds, translation=trans, camera_center=center_offset,focal_length=focal_length, normalize=False)
    dists = (np.linalg.norm(projected_kp2ds.numpy()-kp_2ds, axis=-1, ord=2) * valid_mask).sum(-1) / (valid_mask.sum(-1)+1e-3)
    cam_mask = dists<thresh
    assert len(trans)==len(cam_mask), print('len(trans)==len(cam_mask) fail, trans {}; cam_mask {}'.format(trans, cam_mask))
    cam_mask[trans[:,2].numpy()<=0] = False
    trans = trans[cam_mask]
    return trans, cam_mask

def convert_kp2d_from_input_to_orgimg(kp2ds, offsets):
    offsets = offsets.float().to(kp2ds.device)
    img_pad_size, crop_trbl, pad_trbl = offsets[:,:2], offsets[:,2:6], offsets[:,6:10]
    leftTop = torch.stack([crop_trbl[:,3]-pad_trbl[:,3], crop_trbl[:,0]-pad_trbl[:,0]],1)
    kp2ds_on_orgimg = (kp2ds[:,:,:2] + 1) * img_pad_size.unsqueeze(1) / 2 + leftTop.unsqueeze(1)
    if kp2ds.shape[-1] == 3:
        kp2ds_on_orgimg = torch.cat([kp2ds_on_orgimg, (kp2ds[:,:,[2]] + 1) * img_pad_size.unsqueeze(1)[:,:,[0]] / 2 ], -1)
    return kp2ds_on_orgimg

def convert_cam_to_3d_trans(cams, weight=2.):
    (s, tx, ty) = cams[:,0], cams[:,1], cams[:,2]
    depth, dx, dy = 1./s, tx/s, ty/s
    trans3d = torch.stack([dx, dy, depth], 1)*weight
    return trans3d

def convert_kp2ds2org_images(projected_outputs, input2orgimg_offsets):
    projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d'], input2orgimg_offsets)
    if 'verts_camed' in projected_outputs:
        projected_outputs['verts_camed_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['verts_camed'], input2orgimg_offsets)
    if 'pj2d_h36m17' in projected_outputs:
        projected_outputs['pj2d_org_h36m17'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d_h36m17'], input2orgimg_offsets)
    return projected_outputs

def perspective_projection(points, translation=None,rotation=None, keep_dim=False, 
                           focal_length=focal_length, camera_center=None, img_size=512, normalize=True):
    if isinstance(points,np.ndarray):
        points = torch.from_numpy(points).float()
    if isinstance(translation,np.ndarray):
        translation = torch.from_numpy(translation).float()
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    if camera_center is not None:
        K[:,-1, :-1] = camera_center

    # Transform points
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)
    if translation is not None:
        points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / (points[:,:,-1].unsqueeze(-1)+1e-4)
    if torch.isnan(points).sum()>0 or torch.isnan(projected_points).sum()>0:
        print('Error!!! translation prediction is nan')

    projected_points = torch.matmul(projected_points.contiguous(), K.contiguous())
    if not keep_dim:
        projected_points = projected_points[:, :, :-1].contiguous()

    if normalize:
        return projected_points/float(img_size)*2.

    return projected_points

def convert_scale_to_depth(scale, fovs):
    return fovs / (scale + 1e-6)

def denormalize_cam_params_to_trans(normed_cams, fovs=1/tan_fov, positive_constrain=False):
    #convert the predicted camera parameters to 3D translation in camera space.
    scale = normed_cams[..., [0]]
    if positive_constrain:
        positive_mask = (scale > 0).float()
        scale = scale * positive_mask
    trans_XY_normed = torch.flip(normed_cams[..., 1:],[-1])
    if isinstance(fovs, torch.Tensor):
        fovs_uns = fovs.unsqueeze(-1)
    # convert from predicted scale to depth
    depth = convert_scale_to_depth(scale, fovs)
    # convert from predicted X-Y translation on image plane to X-Y coordinates on camera space.
    trans_XY = trans_XY_normed * depth / fovs
    trans = torch.cat([trans_XY, depth], -1)
    return trans

def vertices_kp3d_projection(j3d_preds, joints_h36m17_preds, cam_preds, vertices=None, input2orgimg_offsets=None, presp=True):
    pred_cam_t = denormalize_cam_params_to_trans(cam_preds, positive_constrain=False)
    pj3d = perspective_projection(j3d_preds, translation=pred_cam_t, focal_length=focal_length, normalize=True)
    pj3d_h36m17 = perspective_projection(joints_h36m17_preds, translation=pred_cam_t, focal_length=focal_length, normalize=True)
    projected_outputs = {'cam_trans':pred_cam_t, 'pj2d': pj3d[:,:,:2].float(), 'pj2d_h36m17':pj3d_h36m17[:,:,:2].float()}
    if vertices is not None:
        projected_outputs['verts_camed'] = perspective_projection(vertices.clone().detach(),translation=pred_cam_t,focal_length=focal_length, normalize=True, keep_dim=True)
        projected_outputs['verts_camed'][:,:,2] = vertices[:,:,2]
        
    if input2orgimg_offsets is not None:
        projected_outputs = convert_kp2ds2org_images(projected_outputs, input2orgimg_offsets)

    return projected_outputs

#-----------------------------------------------------------------------------------------#
#                                       Utilizes                                          #                                    
#-----------------------------------------------------------------------------------------#

def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)

age_threshold = {'adult': [-0.05,0,0.15], 'teen':[0.15, 0.3, 0.45], 'kid':[0.45, 0.6, 0.75], 'baby':[0.75,0.9,1]}
def parse_age_cls_results(age_probs):
    age_preds = torch.ones_like(age_probs).long()*-1
    age_preds[(age_probs<=age_threshold['adult'][2])&(age_probs>age_threshold['adult'][0])] = 0
    age_preds[(age_probs<=age_threshold['teen'][2])&(age_probs>age_threshold['teen'][0])] = 1
    age_preds[(age_probs<=age_threshold['kid'][2])&(age_probs>age_threshold['kid'][0])] = 2
    age_preds[(age_probs<=age_threshold['baby'][2])&(age_probs>age_threshold['baby'][0])] = 3
    return age_preds

#-----------------------------------------------------------------------------------------#
#                                       Smooth filter                                     #                                    
#-----------------------------------------------------------------------------------------#

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

#-----------------------------------------------------------------------------------------#
#                                Rotation format transfer                                 #                                    
#-----------------------------------------------------------------------------------------#

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa

def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    denominator = denominator.clone()
    denominator[denominator.abs() < eps] += eps
    return numerator / denominator

def rotation_matrix_to_quaternion(
    rotation_matrix: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.
    Args:
        rotation_matrix: the rotation matrix to convert.
        eps: small value to avoid zero division.
    Return:
        the rotation in quaternion.The quaternion vector has components in (w, x, y, z) format.
    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    rotation_matrix_vec: torch.Tensor = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt((trace + 1.0).clamp_min(eps)) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1():
        sq = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(eps)) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2():
        sq = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(eps)) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3():
        sq = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(eps)) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.
    The quaternion should be in (w, x, y, z) format.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(
                quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    r"""Convert 3d vector of axis-angle rotation to 3x3 rotation matrix
    Args:
        angle_axis (torch.Tensor): tensor of 3d vector of axis-angle rotations.
    Returns:
        torch.Tensor: tensor of 3x3 rotation matrices.
    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 3, 3)`
    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_axis_to_rotation_matrix(input)  # Nx3x3
    """
    if not isinstance(angle_axis, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError(
            "Input size must be a (*, 3) tensor. Got {}".format(
                angle_axis.shape))

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4

#-----------------------------------------------------------------------------------------#
#                              6D Rotation transformation                                 #                                    
#-----------------------------------------------------------------------------------------#

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(
        pred_rotmat.reshape(-1, 3, 3)).reshape(batch_size, -1)
    return pose

def rot6d_to_rotmat(x):
    x = x.view(-1,3,2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats