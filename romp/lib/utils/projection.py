import torch
import numpy as np

import sys, os, cv2
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import constants
from config import args

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

def vertices_kp3d_projection(outputs, meta_data=None, presp=args().model_version>3):
    params_dict, vertices, j3ds = outputs['params'], outputs['verts'], outputs['j3d']
    verts_camed = batch_orth_proj(vertices, params_dict['cam'], mode='3d',keep_dim=True)
    pj3d = batch_orth_proj(j3ds, params_dict['cam'], mode='2d')
    predicts_j3ds = j3ds[:,:24].contiguous().detach().cpu().numpy()
    predicts_pj2ds = (pj3d[:,:,:2][:,:24].detach().cpu().numpy()+1)*256
    cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, \
                                focal_length=args().focal_length, img_size=np.array([512,512])).to(vertices.device)
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

def batch_orth_proj(X, camera, mode='2d',keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed

def project_2D(kp3d, cams,keep_dim=False):
    d,f, t = cams[0], cams[1], cams[2:].unsqueeze(0)
    pose2d = kp3d[:,:2]/(kp3d[:,2][:,None]+d)
    pose2d = pose2d*f+t
    if keep_dim:
        kp3d[:,:2] = pose2d
        return kp3d
    else:
        return pose2d
