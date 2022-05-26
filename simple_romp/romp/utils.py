from __future__ import print_function
import torch
from torch.nn import functional as F
import numpy as np
import cv2, os, sys
import os.path as osp
from time import time
#from scipy.spatial.transform import Rotation as R
from threading import Thread
import re

#-----------------------------------------------------------------------------------------#
#                                  IO utilizes                                  
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

def convert_tensor2numpy(outputs, del_keys=['verts_camed','smpl_face', 'pj2d', 'verts_camed_org']):
    for key in del_keys:
        if key in outputs:
            del outputs[key]

    result_keys = list(outputs.keys())
    for key in result_keys:
        if isinstance(outputs[key], torch.Tensor):
            outputs[key] = outputs[key].cpu().numpy()
    return outputs

class ResultSaver:
    def __init__(self, mode='image', save_path=None, save_npz=True):
        self.is_dir = len(osp.splitext(save_path)[1]) == 0
        self.mode = mode
        self.save_path = save_path
        self.save_npz = save_npz
        self.save_dir = save_path if self.is_dir else osp.dirname(save_path)
        if self.mode in ['image', 'video']:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.mode == 'video':
            self.frame_save_paths = []
    
    def __call__(self, outputs, input_path, prefix=None, img_ext='.png'):
        if self.mode == 'video' or self.is_dir:
            save_name = osp.basename(input_path)
            save_path = osp.join(self.save_dir, osp.splitext(save_name)[0])+img_ext
        elif self.mode == 'image':
            save_path = self.save_path

        if prefix is not None:
            save_path = osp.splitext(save_path)[0]+f'_{prefix}'+osp.splitext(save_path)[1]

        rendered_image = None
        if outputs is not None:
            if 'rendered_image' in outputs:
                rendered_image = outputs.pop('rendered_image')
            if self.save_npz:
                np.savez(osp.splitext(save_path)[0]+'.npz', results=outputs)
        if rendered_image is None:
            rendered_image = cv2.imread(input_path)
        
        cv2.imwrite(save_path, rendered_image)    
        if self.mode == 'video':
            self.frame_save_paths.append(save_path)
    
    def save_video(self, save_path, frame_rate=24):
        if len(self.frame_save_paths)== 0:
            return 
        height, width = cv2.imread(self.frame_save_paths[0]).shape[:2]
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        for frame_path in self.frame_save_paths:
            writer.write(cv2.imread(frame_path))
        writer.release()


def save_video_results(frame_save_paths):
    video_results = {}
    video_sequence_results = {}
    for frame_id, save_path in enumerate(frame_save_paths):
        npz_path = osp.splitext(save_path)[0]+'.npz'
        frame_results = np.load(npz_path, allow_pickle=True)['results'][()]
        base_name = osp.basename(save_path)
        video_results[base_name] = frame_results
        
        if 'track_ids' not in frame_results:
            continue
        for subj_ind, track_id in enumerate(frame_results['track_ids']):
            if track_id not in video_sequence_results:
                video_sequence_results[track_id] = {'frame_id':[]}
            video_sequence_results[track_id]['frame_id'].append(frame_id)
            for key in frame_results:
                if key not in video_sequence_results[track_id]:
                    video_sequence_results[track_id][key] = []
                video_sequence_results[track_id][key].append(frame_results[key][subj_ind])

    video_results_save_path = osp.join(osp.dirname(frame_save_paths[0]), 'video_results.npz')
    np.savez(video_results_save_path, results=video_results, sequence_results=video_sequence_results)


class WebcamVideoStream(object):
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        try:
            self.stream = cv2.VideoCapture(src)
        except:
            self.stream = cv2.VideoCapture("/dev/video{}".format(src), cv2.CAP_V4L2)
        
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
        return self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def video2frame(video_path, frame_save_dir=None):
    cap = cv2.VideoCapture(video_path)
    for frame_id in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        success_flag, frame = cap.read()
        if success_flag:
            save_path = os.path.join(frame_save_dir, '{:08d}.jpg'.format(frame_id))
            cv2.imwrite(save_path, frame)

def collect_frame_path(video_path, save_path):
    assert osp.exists(video_path), video_path + 'not exist!'

    is_dir = len(osp.splitext(save_path)[1]) == 0
    if is_dir:
        save_dir = save_path
        save_name = osp.splitext(osp.basename(video_path))[0] + '.mp4'
    else:
        save_dir = osp.dirname(save_path)
        save_name = osp.splitext(osp.basename(save_path))[0] + '.mp4'
    video_save_path = osp.join(save_dir, save_name)

    if osp.isfile(video_path):
        video_name, video_ext = osp.splitext(osp.basename(video_path))
        
        frame_save_dir = osp.join(save_dir, video_name+'_frames')
        print(f'Extracting the frames of input {video_path} to {frame_save_dir}')
        os.makedirs(frame_save_dir, exist_ok=True)
        try:
            video2frame(video_path, frame_save_dir)
        except:
            raise Exception(f"Failed in extracting the frames of {video_path} to {frame_save_dir}! \
                Please check the video. If you want to do this by yourself, please extracte frames to {frame_save_dir} and take it as input to ROMP. \
                For example, the first frame name is supposed to be {osp.join(frame_save_dir, '00000000.jpg')}")
    else:
        frame_save_dir = video_path

    assert osp.isdir(frame_save_dir), frame_save_dir + 'is supposed to be a folder containing video frames.'
    frame_paths = [osp.join(frame_save_dir, frame_name) for frame_name in sorted(os.listdir(frame_save_dir))]
    return frame_paths, video_save_path

#-----------------------------------------------------------------------------------------#
#                         tracking & temporal optimization utils                                    
#-----------------------------------------------------------------------------------------#

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


def smooth_results(filters, body_pose=None, body_shape=None, cam=None):
    if body_pose is not None:
        global_rot = smooth_global_rot_matrix(body_pose[:3], filters['global_rot'])
        body_pose = torch.cat([global_rot, filters['smpl_thetas'].process(body_pose[3:])], 0)
    if body_shape is not None:
        body_shape = filters['smpl_betas'].process(body_shape)
    if cam is not None:
        cam = filters['cam'].process(cam)
    return body_pose, body_shape, cam


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def get_tracked_ids(detections, tracked_objects):
    tracked_ids_out = np.array([obj.id for obj in tracked_objects])
    tracked_points = np.array([obj.last_detection.points for obj in tracked_objects])
    org_points = np.array([obj.points for obj in detections])
    tracked_ids = [tracked_ids_out[np.argmin(np.linalg.norm(tracked_points-point[None], axis=1))] for point in org_points]
    return tracked_ids

def get_tracked_ids3D(detections, tracked_objects):
    tracked_ids_out = np.array([obj.id for obj in tracked_objects])
    tracked_points = np.array([obj.last_detection.points for obj in tracked_objects])
    org_points = np.array([obj.points for obj in detections])
    tracked_ids = [tracked_ids_out[np.argmin(np.linalg.norm(tracked_points.reshape(-1,4)-point.reshape(1,4), axis=1))] for point in org_points]
    return tracked_ids

#-----------------------------------------------------------------------------------------#
#                               3D-to-2D projection utils                                    
#-----------------------------------------------------------------------------------------#

INVALID_TRANS=np.ones(3)*-1
def convert_kp2d_from_input_to_orgimg(kp2ds, offsets):
    offsets = offsets.float().to(kp2ds.device)
    img_pad_size, crop_trbl, pad_trbl = offsets[:,:2], offsets[:,2:6], offsets[:,6:10]
    leftTop = torch.stack([crop_trbl[:,3]-pad_trbl[:,3], crop_trbl[:,0]-pad_trbl[:,0]],1)
    kp2ds_on_orgimg = (kp2ds + 1) * img_pad_size.unsqueeze(1) / 2 + leftTop.unsqueeze(1)
    return kp2ds_on_orgimg

def convert_cam_to_3d_trans(cams, weight=2.):
    (s, tx, ty) = cams[:,0], cams[:,1], cams[:,2]
    depth, dx, dy = 1./s, tx/s, ty/s
    trans3d = torch.stack([dx, dy, depth], 1)*weight
    return trans3d

def batch_orth_proj(X, camera, mode='2d',keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed

def vertices_kp3d_projection(outputs, meta_data=None, presp=False):
    vertices, j3ds = outputs['verts'], outputs['j3d']
    verts_camed = batch_orth_proj(vertices, outputs['cam'], mode='3d',keep_dim=True)
    pj3d = batch_orth_proj(j3ds, outputs['cam'], mode='2d')
    predicts_j3ds = j3ds[:,:24].contiguous().detach().cpu().numpy()
    predicts_pj2ds = (pj3d[:,:,:2][:,:24].detach().cpu().numpy()+1)*256
    cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, \
                                focal_length=443.4, img_size=np.array([512,512])).to(vertices.device)
    projected_outputs = {'verts_camed': verts_camed, 'pj2d': pj3d[:,:,:2], 'cam_trans':cam_trans}

    if meta_data is not None:
        projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d'], meta_data['offsets'])
    return projected_outputs

def estimate_translation_cv2(joints_3d, joints_2d, focal_length=600, img_size=np.array([512.,512.]), proj_mat=None, cam_dist=None):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0,0], camK[1,1] = focal_length, focal_length
        camK[:2,2] = img_size//2
    else:
        camK = proj_mat
    ret, rvec, tvec,inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                              flags=cv2.SOLVEPNP_EPNP,reprojectionError=20,iterationsCount=100)

    if inliers is None:
        return INVALID_TRANS
    else:
        tra_pred = tvec[:,0]            
        return tra_pred

def estimate_translation_np(joints_3d, joints_2d, joints_conf, focal_length=600, img_size=np.array([512.,512.]), proj_mat=None):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = joints_3d.shape[0]
    if proj_mat is None:
        # focal length
        f = np.array([focal_length,focal_length])
        # optical center
        center = img_size/2.
    else:
        f = np.array([proj_mat[0,0],proj_mat[1,1]])
        center = proj_mat[:2,2]

    # transformations
    Z = np.reshape(np.tile(joints_3d[:,2],(2,1)).T,-1)
    XY = np.reshape(joints_3d[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans

def estimate_translation(joints_3d, joints_2d, pts_mnum=4,focal_length=600, proj_mats=None, cam_dists=None,img_size=np.array([512.,512.])):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """
    if torch.is_tensor(joints_3d):
        joints_3d = joints_3d.detach().cpu().numpy()
    if torch.is_tensor(joints_2d):
        joints_2d = joints_2d.detach().cpu().numpy()
    
    if joints_2d.shape[-1]==2:
        joints_conf = joints_2d[:, :, -1]>-2.
    elif joints_2d.shape[-1]==3:
        joints_conf = joints_2d[:, :, -1]>0
    joints3d_conf = joints_3d[:, :, -1]!=-2.
    
    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float)
    if proj_mats is None:
        proj_mats = [None for _ in range(len(joints_2d))]
    if cam_dists is None:
        cam_dists = [None for _ in range(len(joints_2d))]
    # Find the translation for each example in the batch
    for i in range(joints_3d.shape[0]):
        S_i = joints_3d[i]
        joints_i = joints_2d[i,:,:2]
        valid_mask = joints_conf[i]*joints3d_conf[i]
        if valid_mask.sum()<pts_mnum:
            trans[i] = INVALID_TRANS
            continue
        if len(img_size.shape)==1:
            imgsize = img_size
        elif len(img_size.shape)==2:
            imgsize = img_size[i]
        else:
            raise NotImplementedError
        try:
            trans[i] = estimate_translation_cv2(S_i[valid_mask], joints_i[valid_mask], 
                focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i], cam_dist=cam_dists[i])
        except:
            trans[i] = estimate_translation_np(S_i[valid_mask], joints_i[valid_mask], valid_mask[valid_mask].astype(np.float32), 
                focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i])

    return torch.from_numpy(trans).float()


#-----------------------------------------------------------------------------------------#
#                                Body joints definition                                    
#-----------------------------------------------------------------------------------------#

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format),dtype=np.int)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)

SMPL_24 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23
    }

SMPL_EXTRA_30 = {
    'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28, \
    'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34, \
    'L_Hand_thumb':35, 'L_Hand_index': 36, 'L_Hand_middle':37, 'L_Hand_ring':38, 'L_Hand_pinky':39, \
    'R_Hand_thumb':40, 'R_Hand_index':41,'R_Hand_middle':42, 'R_Hand_ring':43, 'R_Hand_pinky': 44, \
    'R_Hip': 45, 'L_Hip':46, 'Neck_LSP':47, 'Head_top':48, 'Pelvis':49, 'Thorax_MPII':50, \
    'Spine_H36M':51, 'Jaw_H36M':52, 'Head':53
    }

SMPL_ALL_54 = {**SMPL_24, **SMPL_EXTRA_30}

#-----------------------------------------------------------------------------------------#
#               3D vector to 6D rotation representation conversion utils                                    
#-----------------------------------------------------------------------------------------#

def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(batch_size, -1)
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

def batch_rodrigues(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 3) 
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

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
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q



def transform_rot_representation(rot, input_type='mat',out_type='vec'):
    '''
    make transformation between different representation of 3D rotation
    input_type / out_type (np.array):
        'mat': rotation matrix (3*3)
        'quat': quaternion (4)
        'vec': rotation vector (3)
        'euler': Euler degrees in x,y,z (3)
    '''
    from scipy.spatial.transform import Rotation as R
    if input_type=='mat':
        r = R.from_matrix(rot)
    elif input_type=='quat':
        r = R.from_quat(rot)
    elif input_type =='vec':
        r = R.from_rotvec(rot)
    elif input_type =='euler':
        if rot.max()<4:
            rot = rot*180/np.pi
        r = R.from_euler('xyz',rot, degrees=True)
    
    if out_type=='mat':
        out = r.as_matrix()
    elif out_type=='quat':
        out = r.as_quat()
    elif out_type =='vec':
        out = r.as_rotvec()
    elif out_type =='euler':
        out = r.as_euler('xyz', degrees=False)
    return out


#-----------------------------------------------------------------------------------------#
#                                       utilizes                                  
#-----------------------------------------------------------------------------------------#

def time_cost(name='ROMP'):
    def time_counter(func):
        # This function shows the execution time of 
        # the function object passed
        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            cost_time = t2-t1
            fps = 1./cost_time
            print(f'{name} {func.__name__!r} executed in {cost_time:.4f}s, FPS {fps:.1f}')
            return result
        return wrap_func
    return time_counter

def determine_device(gpu_id):
    if gpu_id != -1:
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    return device

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

def wait_func(mode):
    if mode == 'image':
        print('Press ESC to exit...')
        while 1:
            if cv2.waitKey() == 27:
                break 
    elif mode == 'webcam' or mode == 'video':
        cv2.waitKey(1)

class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = "%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go \n"

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='-',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)
    
def progress_bar(it):
    progress = ProgressBar(len(it), fmt=ProgressBar.FULL)
    for i, item in enumerate(it):
        yield item
        progress.current += 1
        progress()
    progress.done()
