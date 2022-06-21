import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys, os
import config
import constants
from config import args

tan_fov = np.tan(np.radians(args().FOV/2.))
cam3dmap_anchor = torch.from_numpy(constants.get_cam3dmap_anchor(args().FOV,args().centermap_size)).float()
scale_num = len(cam3dmap_anchor)

def process_cam_params(cam_maps):
    # to make sure that scale is always a positive value
    cam_maps[..., 0] = (cam_maps[..., 0] + 1.) / 2.
    return cam_maps

def convert_scale_to_depth_level(scale):
    cam3dmap_anchors = cam3dmap_anchor.to(scale.device)[None]
    return torch.argmin(torch.abs(scale[:,None].repeat(1, scale_num) - cam3dmap_anchors), dim=1)

def convert_cam_params_to_centermap_coords(cam_params):
    center_coords = torch.ones_like(cam_params)
    center_coords[:,1:] = cam_params[:,1:].clone()
    cam3dmap_anchors = cam3dmap_anchor.to(cam_params.device)[None]
    if len(cam_params) != 0:
        center_coords[:,0] = torch.argmin(torch.abs(cam_params[:,[0]].repeat(1, scale_num) - cam3dmap_anchors), dim=1).float()/args().centermap_size * 2. - 1.
    
    return center_coords

def normalize_trans_to_cam_params(trans):
    # calculate (scale, Y trans, X trans) as camera parameters
    normed_cams = np.zeros_like(trans)
    #tan_fov = np.tan(np.radians(args().FOV/2.))
    normed_cams[:,0] = 1 / (trans[:,2] * tan_fov)
    normed_cams[:,1] = trans[:,1]/(trans[:,2]*tan_fov)
    normed_cams[:,2] = trans[:,0]/(trans[:,2]*tan_fov)

    _check_valid_cam(normed_cams)
    return normed_cams

def convert_scale_to_depth(scale):
    return 1 / (scale * tan_fov + 1e-3)

def denormalize_cam_params_to_trans(normed_cams, positive_constrain=False):
    #convert the predicted camera parameters to 3D translation in camera space.
    scale = normed_cams[:, 0]
    if positive_constrain:
        positive_mask = (normed_cams[:, 0] > 0).float()
        scale = scale * positive_mask

    trans_XY_normed = torch.flip(normed_cams[:, 1:],[1])
    # convert from predicted scale to depth
    depth = convert_scale_to_depth(scale).unsqueeze(1)
    # convert from predicted X-Y translation on image plane to X-Y coordinates on camera space.
    trans_XY = trans_XY_normed * depth * tan_fov
    trans = torch.cat([trans_XY, depth], 1)

    return trans

INVALID_TRANS=np.ones(3)*-1

def _check_valid_cam(normed_cams):
    # scale value is in 0~1
    assert ((normed_cams[:,0]<0)*(normed_cams[:,0]>1)).sum()==0, print('camera scale must in 0~1, but we get {}'.format(normed_cams[:,0])) 
    # normalized translation in X-Y axis must in -1~1
    assert ((normed_cams[:,1]<-1)*(normed_cams[:,1]>1)).sum()==0, print('Y translation must in -1~1, but we get {}'.format(normed_cams[:,1])) 
    assert ((normed_cams[:,2]<-1)*(normed_cams[:,2]>1)).sum()==0, print('X translation must in -1~1, but we get {}'.format(normed_cams[:,2])) 


def estimate_translation_cv2(joints_3d, joints_2d, focal_length=args().focal_length, img_size=np.array([512.,512.]), proj_mat=None, cam_dist=None):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0,0], camK[1,1] = focal_length, focal_length
        camK[:2,2] = img_size//2
    else:
        camK = proj_mat
    ret, rvec, tvec,inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                              flags=cv2.SOLVEPNP_EPNP,reprojectionError=20,iterationsCount=100)

    if inliers is None:
        #print('cv2.solvePnPRansac failed, with valid kp number as {}'.format(joints_3d.shape))
        return INVALID_TRANS
    else:
        #rot_pred = np.eye(3)
        tra_pred = tvec[:,0]
        #cv2.Rodrigues(rvec, rot_pred)                
        return tra_pred

def estimate_translation_np(joints_3d, joints_2d, joints_conf, focal_length=args().focal_length, img_size=np.array([512.,512.]), proj_mat=None):
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


def estimate_translation(joints_3d, joints_2d, pts_mnum=4,focal_length=args().focal_length, proj_mats=None, cam_dists=None,img_size=np.array([512.,512.]), pnp_algorithm='cv2'):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """

    #device = joints_3d.device
    # Use only joints 25:49 (GT joints)
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
        if pnp_algorithm=='cv2':
            trans[i] = estimate_translation_cv2(S_i[valid_mask], joints_i[valid_mask], 
                focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i], cam_dist=cam_dists[i])
        elif pnp_algorithm=='np':
            trans[i] = estimate_translation_np(S_i[valid_mask], joints_i[valid_mask], valid_mask[valid_mask].astype(np.float32), 
                focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i])
        else:
            raise NotImplementedError
    return torch.from_numpy(trans).float()

if __name__ == '__main__':
    cam_params = torch.rand(2,3)
    centermap_coords = convert_cam_params_to_centermap_coords(cam_params)
    print(cam_params,centermap_coords)