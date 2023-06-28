import torch
import numpy as np

import sys, os
import constants
from config import args
from utils.cam_utils import denormalize_cam_params_to_trans
#from pudb import set_trace; set_trace(paused=False)

def filter_out_incorrect_trans(kp_3ds, trans, kp_2ds, thresh=20, focal_length=args().focal_length, center_offset=torch.Tensor([args().input_size, args().input_size])/2.):
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

def vertices_kp3d_projection(j3d_preds, joints_h36m17_preds, cam_preds, vertices=None, input2orgimg_offsets=None, presp=args().model_version>3):
    if presp:
        pred_cam_t = denormalize_cam_params_to_trans(cam_preds, positive_constrain=False)
        pj3d = perspective_projection(j3d_preds,translation=pred_cam_t,focal_length=args().focal_length, normalize=True)
        pj3d_h36m17 = perspective_projection(joints_h36m17_preds,translation=pred_cam_t,focal_length=args().focal_length, normalize=True)
        projected_outputs = {'cam_trans':pred_cam_t, 'pj2d': pj3d[:,:,:2].float(), 'pj2d_h36m17':pj3d_h36m17[:,:,:2].float()}
        if vertices is not None:
            projected_outputs['verts_camed'] = perspective_projection(vertices.clone().detach(),translation=pred_cam_t,focal_length=args().focal_length, normalize=True, keep_dim=True)
            projected_outputs['verts_camed'][:,:,2] = vertices[:,:,2]
    else:
        pj3d = batch_orth_proj(j3d_preds, cam_preds, mode='2d')
        pj3d_h36m17 = batch_orth_proj(joints_h36m17_preds, cam_preds, mode='2d')
        pred_cam_t = convert_cam_to_3d_trans(cam_preds)
        projected_outputs = {'pj2d': pj3d[:,:,:2], 'cam_trans':pred_cam_t, 'pj2d_h36m17':pj3d_h36m17[:,:,:2].float()}
        if vertices is not None:
            projected_outputs['verts_camed'] = batch_orth_proj(vertices, cam_preds, mode='3d',keep_dim=True)
        
    if input2orgimg_offsets is not None:
        projected_outputs = convert_kp2ds2org_images(projected_outputs, input2orgimg_offsets)

    return projected_outputs


def vertices_kp3d_projection_withfov(j3d_preds, joints_h36m17_preds, pred_cam_t, fovs, vertices=None, input2orgimg_offsets=None):
    pj3d = perspective_projection_withfovs(
        j3d_preds, translation=pred_cam_t, fovs=fovs)
    pj3d_h36m17 = perspective_projection_withfovs(
        joints_h36m17_preds, translation=pred_cam_t, fovs=fovs)
    projected_outputs = {'pj2d': pj3d[:, :, :2].float(), 'pj2d_h36m17': pj3d_h36m17[:, :, :2].float()}
    if vertices is not None:
        projected_outputs['verts_camed'] = perspective_projection_withfovs(
            vertices.clone().detach(), translation=pred_cam_t, fovs=fovs, keep_dim=True)
        projected_outputs['verts_camed'][:, :, 2] = vertices[:, :, 2]

    if input2orgimg_offsets is not None:
        projected_outputs = convert_kp2ds2org_images(projected_outputs, input2orgimg_offsets)

    return projected_outputs

def project_2D(kp3d, cams,keep_dim=False):
    d,f, t = cams[0], cams[1], cams[2:].unsqueeze(0)
    pose2d = kp3d[:,:2]/(kp3d[:,2][:,None]+d)
    pose2d = pose2d*f+t
    if keep_dim:
        kp3d[:,:2] = pose2d
        return kp3d
    else:
        return pose2d

def batch_orth_proj(X, camera, mode='2d',keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed

def calc_trans_xyz(tan_fov, depth_pred, pred_camera):
    trans_xy = pred_camera[:, 1:]
    trans_xy = trans_xy * tan_fov.unsqueeze(-1) * depth_pred.unsqueeze(-1)
    trans_xyz = torch.cat([trans_xy, depth_pred.unsqueeze(-1)], dim=-1)
    return trans_xyz


def batch_persp_depth(pred_joints, trans_xyz, depth_pred, tan_fov, keep_dim=False):
    pred_joints_proj = perspective_projection(pred_joints, trans_xyz, tan_fov)
    if not keep_dim:
        pred_joints_proj = pred_joints_proj[:, :, :-1]
    return pred_joints_proj

def perspective_projection_normed(points, translation, FOV=args().FOV):
    '''
    This function computes the perspective projection of 3D points 
        and output 2D coordinates on normalized image plane (-1 ~ 1)
        points: torch.float32, B x N x 3, 3D body joints
        translation: torch.float32, B x 3,  predicted camera parameters (scale, trans_y on image, trans_x on image) 
        FOV: int, Field of view in degree, here we adopt the FOV of a standard camera, 50 degree. 
    '''

    tan_fov = np.tan(np.radians(FOV/2.))

    points = points + translation.unsqueeze(1)
    
    # Apply perspective distortion
    projected_points = points[:,:,:-1] / (points[:,:,-1].unsqueeze(-1)+1e-9) / tan_fov
    return projected_points

'''
Brought from https://github.com/nkolot/SPIN
'''

def perspective_projection(points, translation=None,rotation=None, keep_dim=False, 
                           focal_length=args().focal_length, camera_center=None, img_size=512, normalize=True):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
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
        #print('translation:', translation[torch.where(torch.isnan(translation))[0]])
        #print('points nan value number:', len(torch.where(torch.isnan(points))[0]))
        #print('projected_points nan value:', projected_points[torch.where(torch.isnan(projected_points))[0]])
        # print('projected_points nan', torch.where(torch.isnan(projected_points)))
        # import pdb; pdb.set_trace()

    # Apply camera intrinsics
    # projected_points = torch.einsum('bij,bkj->bki', K, projected_points)[:, :, :-1]
    projected_points = torch.matmul(projected_points.contiguous(), K.contiguous())
    if not keep_dim:
        projected_points = projected_points[:, :, :-1].contiguous()

    if normalize:
        return projected_points/float(img_size)*2.

    return projected_points


def perspective_projection_withfovs(points, translation=None, rotation=None, keep_dim=False, fovs=None):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()
    if isinstance(translation, np.ndarray):
        translation = torch.from_numpy(translation).float()
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = fovs
    K[:, 1, 1] = fovs
    K[:, 2, 2] = 1.

    # Transform points
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)
    if translation is not None:
        points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / (points[:, :, -1].unsqueeze(-1)+1e-4)
    if torch.isnan(points).sum() > 0 or torch.isnan(projected_points).sum() > 0:
        print('Error!!! translation prediction is nan')
        #print('translation:', translation[torch.where(
        #   torch.isnan(translation))[0]])
        #print('points nan value number:', len(
        #   torch.where(torch.isnan(points))[0]))

    # Apply camera intrinsics
    # projected_points = torch.einsum('bij,bkj->bki', K, projected_points)[:, :, :-1]
    projected_points = torch.matmul(
        projected_points.contiguous(), K.contiguous())
    if not keep_dim:
        projected_points = projected_points[:, :, :-1].contiguous()

    return projected_points

if __name__ == '__main__':
    test_projection_depth()

"""
def centralize_kps(kps, enlarge=1.2):
    center = kps.mean(1).unsqueeze(1)
    lengths = torch.norm(kps-center, p=2, dim=-1)
    length = lengths.max(1).values * enlarge
    length[length<1e-3] = 1e-3
    kps_ceneterlized = (kps - center)/torch.abs(length.unsqueeze(1).unsqueeze(1)+1e-4)

    return kps_ceneterlized

def preprocess_kps(kps,image,set_minus=False):
    kps[:,0] = kps[:,0] / image.shape[0]
    kps[:,1] = kps[:,1] / image.shape[1]
    kps[:,:2] = 2.0 * kps[:,:2] - 1.0

    if kps.shape[1]>2 and set_minus:
        kps[kps[:,2]<0,:2] = -2.
    kps=kps[:,:2]
    return kps


def process_hand2Dkp(hand_kps, bodycoco25_kps, left=True):
    coco25_elbow_wrist_idx = [6,7] if left else [3,4]
    #ew_vector = bodycoco25_kps[:,coco25_elbow_wrist_idx[0]] - bodycoco25_kps[:,coco25_elbow_wrist_idx[1]]
    #ew_vector_normed = ew_vector/(torch.norm(ew_vector, p=2, dim=-1, keepdim=True)+1e-6)
    hand_in = centralize_kps(hand_kps)
    #ew_vnorm = hand_in[:,0] - ew_vector_normed
    #hand_in = torch.cat([ew_vnorm.unsqueeze(1), hand_in], 1)
    return hand_in

def process_face2Dkp(face_kps, bodycoco25_kps):
    #coco25_lrshoulder_idx = [5,2]
    face_in = centralize_kps(face_kps)
    return face_in

def process_2Dkp(kps):
    fnum, kps_list = 0, []
    for i, num in enumerate(constants.bhf_kps_num):
        kps_list.append(kps[:,fnum:fnum+num])
        fnum += num
    bodycoco25_kps, lhand_kps, rhand_kps, face_kps = kps_list
    connect_kps = kps[:,constants.bhf_connect_kps_num]

    face_in = face_kps if face_kps[0,0,0]!=0 or face_kps[0,0,0]!=-2. else None
    lhand_in = lhand_kps if lhand_kps[0,0,0]!=0 or lhand_kps[0,0,0]!=-2. else None
    rhand_in = rhand_kps if rhand_kps[0,0,0]!=0 or rhand_kps[0,0,0]!=-2. else None

    #face_in = process_face2Dkp(face_kps, bodycoco25_kps) if face_kps[0,0,0]!=0 else None
    #lhand_in = process_hand2Dkp(lhand_kps, bodycoco25_kps, left=True) if lhand_kps[0,0,0]!=0 else None
    #rhand_in = process_hand2Dkp(rhand_kps, bodycoco25_kps, left=False) if rhand_kps[0,0,0]!=0 else None
    return face_in, lhand_in, rhand_in, connect_kps

def batch_persp_proj(X, cam,keep_dim=False,mode=1):
    trans = cam.view(-1, 1, 4)
    z = X[:,:,2]
    if mode==0:
        s = 1./(trans[:,:,0] + (trans[:,:,1]+5-z)/(trans[:,:,1]+5))
    elif mode==4:
        d = 4+trans[:,:,1]
        s = trans[:,:,0] + f*z/(d*(d-z))
    elif mode== 3:
        s = 1./(torch.pow(0.9,trans[:,:,0]) * (5+trans[:,:,1]-z)/(5+trans[:,:,0]))
    
    elif mode==2:
        f = 5+trans[:,:,0]
        d = trans[:,:,1]
        s = f/(d-z)
    
    else:
        s=trans[:,:,0]
    X_camed = X[:,:,:2] * s.unsqueeze(-1)
    if mode==2:
        X_camed += trans[:,:,2:]* s.unsqueeze(-1)
    else:
        X_camed += trans[:,:,2:]
    if mode==1:
        d = args().d0+trans[:,:,1]
        s_i = d/(d-z)
        X_camed *= s_i.unsqueeze(-1)

    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed
"""
