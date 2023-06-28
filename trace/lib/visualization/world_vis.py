import numpy as np
import os, sys
import glob
import copy
import torch
import cv2
from torch import nn
import pytorch3d
import tqdm
import constants
from visualization.visualization import draw_skeleton
import quaternion

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles, convention=('X', 'Y', 'Z')):
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

class CamPose_IR(nn.Module):
    def __init__(self, world_kp3d, pj2d, cam_K, init_pitch_tx, device=torch.device('cuda:0')):
        super().__init__()
        self.device = device 

        self.register_buffer('world_kp3d', world_kp3d.float().cuda()) 
        self.register_buffer('pj2d', pj2d.float().cuda())  
        self.register_buffer('cam_K', torch.from_numpy(cam_K).float().cuda()) 

        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_pitch_tx = nn.Parameter(init_pitch_tx)

    def forward(self):
        camera_euler_angles = torch.cat([self.camera_pitch_tx[:2], torch.zeros(1).cuda()],0)
        cam_rot_mat = euler_angles_to_matrix(camera_euler_angles)

        points = torch.einsum('ij,kj->ki', cam_rot_mat, self.world_kp3d)
        points[:,0] = points[:,0] + self.camera_pitch_tx[2]

        projected_points = points / (points[:,-1].unsqueeze(-1))
        projected_points = torch.matmul(self.cam_K[:3,:3], projected_points.contiguous().T).T
        projected_points = projected_points[...,:2]
        
        loss = torch.norm(projected_points - self.pj2d, p=2, dim=-1).mean()
        #print(loss, self.camera_pitch_tx)
        return loss, projected_points

def solve_camera_poses(world_j3ds, world_trans, pj2ds, cam_Ks, image_list, image_h, image_w, vis=False):
    world_j3ds = world_j3ds + world_trans.unsqueeze(1)
    pj2ds = (pj2ds + 1) / 2 * max(image_h, image_w)
    if image_w>image_h:
        pj2ds[:,:,1] = pj2ds[:,:,1] - (image_w - image_h)//2
    elif image_h>image_w:
        pj2ds[:,:,0] = pj2ds[:,:,0] - (image_h - image_w)//2    

    frame_num = len(world_j3ds)
    init_pitch_tx = torch.Tensor([0,0,0]).float().cuda()
    cam_pitch_txs = []
    max_iter_num = 100
    for ind in tqdm.tqdm(range(frame_num)):
        regressor = CamPose_IR(world_j3ds[ind], pj2ds[ind], cam_Ks[ind], init_pitch_tx)
        optimizer = torch.optim.Adam(regressor.parameters(), lr=0.01)
        for i in range(max_iter_num):
            optimizer.zero_grad()
            loss, projected_points = regressor()
            loss.backward()
            optimizer.step()
            # loop.set_description('Optimizing (loss %.4f)' % loss.data)
            
            if loss.item() < 8:
                break
        
            if vis:
                image = cv2.imread(image_list[ind])
                image = draw_skeleton(image, projected_points.detach().cpu().numpy(), bones=constants.All73_connMat, cm=constants.cm_All54)
                image = draw_skeleton(image, pj2ds[ind].detach().cpu().numpy(), bones=constants.All73_connMat, cm=constants.cm_All54)
                cv2.imshow('frame', image)
                cv2.waitKey()
        init_pitch_tx = regressor.camera_pitch_tx
        cam_pitch_txs.append(regressor.camera_pitch_tx.detach().cpu().numpy())
    return cam_pitch_txs

def vis_global_view_org(smpl_poses, smpl_betas, global_trans, global_orient, cam_Rts, cam_Ks, image_list, mesh_color=[(149/255, 149/255, 149/255, 0.8)]):
    from visualization.call_aitviewer import GlobalViewer
    viewer_cfgs_update = {'fps':25, 'playback_fps':25.0}
    global_viewer = GlobalViewer(viewer_cfgs_update=viewer_cfgs_update)

    subj_num = smpl_poses.shape[1]
    for subj_ind in range(subj_num):
        gtrans, grots = copy.deepcopy(global_trans[:,subj_ind]), copy.deepcopy(global_orient[:,subj_ind])
        global_viewer.add_smpl_sequence2scene(smpl_poses[:,subj_ind], smpl_betas[:,subj_ind], gtrans, grots,\
                color=mesh_color[subj_ind], draw_outline=True)

    mean_cam_position = cam_Rts[:, :3, 3].mean(0)
    mean_subj_position = global_trans.mean(0).mean(0)
    world2dynamic_div_dynamic2people = 4
    dynamic2people_distance = np.linalg.norm(global_trans, ord=2, axis=-1).max()+3
    
    global_viewer.add_camera2scene(cam_Ks, cam_Rts)
    global_viewer.add_dynamic_image2scene(image_list, distance=dynamic2people_distance) #
    
    cam_position = mean_cam_position + world2dynamic_div_dynamic2people * (mean_cam_position - mean_subj_position)
    cam_target = mean_subj_position

    # Set initial camera position and target
    # in ait-viewer z is up, x is right
    global_viewer.viewer.scene.camera.position = cam_position[[0,2,1]] # np.array((0.0, 2, 0))
    global_viewer.viewer.scene.camera.target = cam_target[[0,2,1]] # np.array((0, 0, -2))

    # Viewer settings
    global_viewer.viewer.scene.floor.enabled = False
    global_viewer.viewer.scene.fps = 30.0
    global_viewer.viewer.playback_fps = 30.0
    global_viewer.viewer.shadows_enabled = False
    global_viewer.viewer.auto_set_camera_target = False

    global_viewer.viewer.run()


def vis_global_view(smpl_poses, smpl_betas, global_trans, global_orient, cam_Rts, cam_Ks, image_list, mesh_color=[(149/255, 149/255, 149/255, 0.8)], draw_outline=True):
    from visualization.call_aitviewer import GlobalViewer
    viewer_cfgs_update = {'fps':25, 'playback_fps':25.0}
    global_viewer = GlobalViewer(viewer_cfgs_update=viewer_cfgs_update)

    # in ait-viewer, y is up, z is front, x is right
    # therefore, we have to rotate the image plane from x-y to x-z
    XY2XZ = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], np.float32)

    subj_num = smpl_poses.shape[1]
    print('subj_num', subj_num, 'image', len(image_list), smpl_poses.shape)
    for subj_ind in range(subj_num):
        gtrans, grots = copy.deepcopy(global_trans[:,subj_ind]), copy.deepcopy(global_orient[:,subj_ind])
        for gind in range(len(gtrans)):
            body_rot = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(grots[gind]))
            body_rot_tran = np.eye(4)
            body_rot_tran[:3,:3] = body_rot.T
            body_rot_tran[:3,3] = gtrans[gind]
            body_rot_tran = XY2XZ @ body_rot_tran
            gtrans[gind] = body_rot_tran[:3,3]
            grots[gind] = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(body_rot_tran[:3,:3]))
        global_viewer.add_smpl_sequence2scene(smpl_poses[:,subj_ind], smpl_betas[:,subj_ind], gtrans, grots,\
                color=mesh_color[subj_ind], draw_outline=draw_outline)

    cam_Rts = global_viewer.transformRt2viewer_coord(cam_Rts)
    mean_cam_position = cam_Rts[:, :3, 3].mean(0)
    mean_subj_position = global_trans.mean(0).mean(0)
    world2dynamic_div_dynamic2people = 4
    dynamic2people_distance = np.linalg.norm(global_trans, ord=2, axis=-1).max()+3
    
    global_viewer.add_camera2scene(cam_Ks, cam_Rts)
    global_viewer.add_dynamic_image2scene(image_list, distance=dynamic2people_distance) #
    
    # cam_position = mean_cam_position + world2dynamic_div_dynamic2people * (mean_cam_position - mean_subj_position)
    # cam_target = mean_subj_position

    # Set initial camera position and target
    # in ait-viewer, y is up, z is front, x is right
    global_viewer.viewer.scene.camera.position = np.array((0.0, 3, 6)) # np.array((0.0, 10, 20))
    global_viewer.viewer.scene.camera.target = np.array([0, 0, 1]) # np.array([0, 6, 1])

    # Viewer settings
    global_viewer.viewer.scene.floor.enabled = False
    global_viewer.viewer.scene.fps = 30.0
    global_viewer.viewer.playback_fps = 30.0
    global_viewer.viewer.shadows_enabled = False
    global_viewer.viewer.auto_set_camera_target = False

    global_viewer.viewer.run()

def get_cam_K(image_h, image_w, fov=50, move_up=0):
    fy = max(image_h, image_w) / 2 * 1./np.tan(np.radians(fov/2))
    fx = max(image_h, image_w) / 2 * 1./np.tan(np.radians(fov/2))
    cam_K = np.array([[fx, 0, image_w / 2, 0], [0, fy, image_h / 2, move_up], [0, 0, 1, 0]], dtype=np.float32)
    return cam_K

def convertRT2transform(R, T):
    transform4x4 = np.eye(4)
    transform4x4[:3, :3] = R
    transform4x4[:3, 3] = T
    return transform4x4

def camera_pitch_yaw2rotation_matrix(pitch, yaw):
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(pitch))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(yaw))
    R = R2 @ R1
    return R.T

def get_cam_RTs(cam_pitch_txs):
    cam_RTs = []
    for ind, cam_pitch_tx in enumerate(cam_pitch_txs):
        camera_R_mat = camera_pitch_yaw2rotation_matrix(cam_pitch_tx[1], cam_pitch_tx[0])
        camera_T = np.array([cam_pitch_tx[2], 0, 0])
        cam_RT = convertRT2transform(camera_R_mat, camera_T)
        cam_RTs.append(cam_RT)
    cam_RTs = np.stack(cam_RTs)
    return cam_RTs

def resize_images(image_list, image_h, image_w, max_size=720):
    max_edge = max(image_h, image_w)
    new_size = (np.array([image_w, image_h]) * max_size / max_edge).astype(np.int32)
    print('resize images from ', image_h, image_w, 'to', *new_size)
    for image_path in image_list:
        cv2.imwrite(image_path, cv2.resize(cv2.imread(image_path), (new_size[0], new_size[1]), interpolation=cv2.INTER_CUBIC))

def visualize_global_trajectory(outputs, image_folder, solve_camera=False, motion_snapshot=False):
    smpl_thetas, smpl_betas = outputs['smpl_thetas'].numpy(), outputs['smpl_betas'].numpy()
    global_orient = outputs['world_global_rots'].numpy()
    global_trans = outputs['world_trans'].numpy()
    
    image_list = sorted(glob.glob(os.path.join(image_folder, '*')))
    image_h, image_w = cv2.imread(image_list[0]).shape[:2]
    # if max(image_h, image_w) > 720:
    #     resize_images(image_list, image_h, image_w) # will change the size of original images.
    #     image_h, image_w = cv2.imread(image_list[0]).shape[:2]
    
    cam_K = get_cam_K(image_h, image_w, fov=50, move_up=10)
    cam_Ks = cam_K[None].repeat(len(global_trans), 0)

    #print(list(outputs.keys()))
    if solve_camera:
        cam_pitch_txs = solve_camera_poses(outputs['world_j3d'], outputs['world_trans'], outputs['pj2d'], cam_Ks, image_list, image_h, image_w)
        cam_Rts = get_cam_RTs(cam_pitch_txs)
    else:
        cam_Rts = np.eye(4)[None].repeat(len(global_trans), 0)
    
    print(smpl_thetas.shape,'frame_num', len(image_list))
    draw_outline = True
    if smpl_thetas.shape[0]<=len(image_list):
        smpl_poses, smpl_betas, global_trans, global_orient = smpl_thetas[:,3:][:,None], smpl_betas[:,None], global_trans[:,None], global_orient[:,None]
        mesh_color=[(.9, .9, .8, 0.96)]
        frame_num = len(smpl_thetas)
        subj_num = 1
    else:
        track_ids = outputs['track_ids'].numpy()
        tids = np.unique(track_ids)
        subj_num = len(tids)
        frame_num = smpl_thetas.shape[0] // subj_num
        subject_inds = np.array([np.where(track_ids==tid)[0] for tid in tids]).transpose((1,0))
        smpl_poses, smpl_betas, global_trans, global_orient = smpl_thetas[:,3:][subject_inds], smpl_betas[subject_inds], global_trans[subject_inds], global_orient[subject_inds]
        #smpl_poses, smpl_betas, global_trans, global_orient = smpl_thetas[:,3:].reshape(subj_num,frame_num, -1), smpl_betas.reshape(subj_num,frame_num, -1), global_trans.reshape(subj_num,frame_num, 3), global_orient.reshape(subj_num,frame_num, 3)
        #smpl_poses, smpl_betas, global_trans, global_orient = smpl_poses.transpose((1,0,2)), smpl_betas.transpose((1,0,2)), global_trans.transpose((1,0,2)), global_orient.transpose((1,0,2))
        mesh_color=[(.6*i/frame_num+0.1, .9, .8, 0.9) for i in range(subj_num)]
    
    used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])
    used_org_inds = np.stack([np.array(used_org_inds) for _ in range(subj_num)], 0).reshape(-1)
    image_list = [image_list[ii] for ii in used_org_inds]
    
    if motion_snapshot:
        sample_num = 4
        smpl_poses, smpl_betas, global_trans, global_orient = smpl_poses[::sample_num].transpose((1,0,2)), smpl_betas[::sample_num].transpose((1,0,2)), global_trans[::sample_num].transpose((1,0,2)), global_orient[::sample_num].transpose((1,0,2))
        
        cam_Rts, cam_Ks, image_list = cam_Rts[[0]], cam_Ks[[0]], [image_list[0]]
        frame_num = smpl_poses.shape[1]
        mesh_color = [(.6*i/frame_num+0.1, .9, .8, 0.9) for i in range(frame_num)]
        draw_outline = False
    #global_trans[...,0] = global_trans[...,0]*2 + 2
    vis_global_view(smpl_poses, smpl_betas, global_trans, global_orient, cam_Rts, cam_Ks, image_list, mesh_color, draw_outline=draw_outline)


def process_idx(reorganize_idx, vids=None):
    result_size = reorganize_idx.shape[0]
    reorganize_idx = reorganize_idx.cpu().numpy()
    used_org_inds = np.unique(reorganize_idx)
    per_img_inds = [np.where(reorganize_idx==org_idx)[0] for org_idx in used_org_inds]

    return used_org_inds, per_img_inds

def visulize_result(renderer, outputs, seq_data, rendering_cfgs, save_dir, alpha=1):
    used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])

if __name__ == '__main__':
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/InterNet_video_tracking_results3/running2rightforward.npz', '/home/yusun/data_drive3/temporal_attempts/running2rightforward'
    #result_path, image_folder = '/home/yusun/data_drive3/tracking_results/InterNet_video_tracking_results3/Lola_Run_Front_2.npz', '/home/yusun/data_drive3/temporal_attempts/LOLA_running_seriers/Lola_Run_Front_2'
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/InterNet_video_tracking_results3/DAVIS16-parkour1.npz', '/home/yusun/data_drive3/temporal_attempts/DAVIS16-parkour1'
    #result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000/TROMP_v6_outputs/020910_mpii_test.npz', \
    #    '/home/yusun/DataCenter2/datasets/3DPW/Dyna3DPW/020910_mpii_test'
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000/TROMP_v6_outputs/pano-fitness5_0-1-0.npz', \
        '/home/yusun/data_drive3/datasets/static2dynamic_camera/pano_video_frames/pano-fitness5_0-1-0'
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000/TROMP_v6_outputs/pano-fitness5_0-3-0.npz', \
        '/home/yusun/data_drive3/datasets/static2dynamic_camera/pano_video_frames/pano-fitness5_0-3-0'
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000/TROMP_v6_outputs/pano-running_0-0-1.npz', \
        '/home/yusun/data_drive3/datasets/static2dynamic_camera/pano_video_frames/pano-running_0-0-1'
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000/TROMP_v6_outputs/pano-sports_0-4-1.npz', \
        '/home/yusun/data_drive3/datasets/static2dynamic_camera/pano_video_frames/pano-sports_0-4-1'    
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000/TROMP_v6_outputs/pano-running_0-9-1.npz', \
        '/home/yusun/data_drive3/datasets/static2dynamic_camera/pano_video_frames/pano-running_0-9-1' 
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_40002/TROMP_v6_outputs/slam_dunk-ZL1-2.npz',\
        '/home/yusun/data_drive3/temporal_attempts/slam_dunk-ZL1-2'
    
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_40002/TROMP_v6_outputs/DAVIS16-parkour1.npz', '/home/yusun/data_drive3/temporal_attempts/DAVIS16-parkour1'
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_40002/TROMP_v6_outputs/return_running2.npz', \
        '/home/yusun/data_drive3/temporal_attempts/return_running2'

    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000-3/TROMP_v6_outputs/TAO-bicycle-18.npz',\
        '/home/yusun/data_drive3/temporal_attempts/TAO-bicycle-18'
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000-3/TROMP_v6_outputs/TAO-bicycle-19.npz',\
        '/home/yusun/data_drive3/temporal_attempts/TAO-bicycle-19'
    
    result_path, image_folder = '/home/yusun/data_drive3/tracking_results/internet_video-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000-3/TROMP_v6_outputs/downtown_cafe_00.npz',\
        '/home/yusun/DataCenter2/datasets/Dyna3DPW/downtown_cafe_00'
    motion_snapshot = False
    outputs = np.load(result_path, allow_pickle=True)['outputs'][()]
    visualize_global_trajectory(outputs, image_folder, solve_camera=False, motion_snapshot=motion_snapshot)
