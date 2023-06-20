import os
import sys
import argparse
import numpy as np
from .utils.open3d_gui import visualize_world_annots
# pip install MarkupSafe==2.0.1 Werkzeug==2.0.3
import torch
import glob
from smplx import SMPL

smpl_model_folder = '/Users/mac/Desktop/Githubs/'

def show_settings(input_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='TRACE: 5D Temporal Regression of Avatars with Dynamic Cameras in 3D Environments')
    parser.add_argument('--smpl_model_folder', type=str, default='smpl_model_data', help = 'Folder contains SMPL_NEUTRAL.pkl')
    parser.add_argument('--preds_path', type=str, default=None, help = 'Path to save the .npz results')
    parser.add_argument('--frame_dir', type=str, default=None, help = 'Path to folder of input video frames')
    args = parser.parse_args(input_args)

    return args

def obtain_smpl_verts(smpl_thetas, smpl_betas, smpl_model):
    # smpl_thetas is in shape N x 24 x 3 x 3
    world_grot_mat = smpl_thetas[:,0]
    body_pose = smpl_thetas[:,1:]
    smpl_output = smpl_model(global_orient=world_grot_mat, body_pose=body_pose, betas=smpl_betas)
    #verts = smpl_output.vertices.cpu().numpy()
    
    return smpl_output.joints.cpu().numpy(), world_grot_mat.cpu().numpy()

def visualize_subject_world_results(seq_name, annots, seq_frame_dir, smpl_model_folder, img_ext='jpg'):
    print(f'Annotation keys include: ', list(annots.keys()))
    smpl_model = SMPL(smpl_model_folder, gender='neutral').eval()

    # ['person_id', 'poses', 'betas', 'world_grots', 'world_trans', 'kp3ds', 'kp2ds', 'frame_ids', \
    #  'camera_intrinsics', 'camera_extrinsics', 'camera_extrinsics_aligned', 'world_grots_aligned', 'world_trans_aligned']
    smpl_thetas = annots['smpl_thetas']
    if len(smpl_thetas.shape) == 2:
        frame_num = smpl_thetas.shape[0]
        subject_num = 1
    elif len(smpl_thetas.shape) == 3:
        subject_num, frame_num = smpl_thetas.shape[:2]
    body_pose = torch.from_numpy(smpl_thetas[:, 3:].reshape(subject_num, frame_num, 23*3)).float()
    smpl_betas = torch.from_numpy(annots['smpl_betas']).float().reshape(subject_num, frame_num, 10)
    if 'world_grots_aligned' in annots:
        world_grots = torch.from_numpy(annots['world_grots_aligned']).float()
        world_trans = annots['world_trans_aligned']
        camera_intrinsics = annots['camera_intrinsics']
        camera_extrinsics = annots['camera_extrinsics_aligned']
    else:
        world_grots = torch.from_numpy(annots['world_global_rots']).float().reshape(subject_num, frame_num, 3)
        world_trans = np.array(annots['world_trans']).astype(np.float64).reshape(subject_num, frame_num, 3)
        camera_intrinsics = np.repeat([np.array([[548,0,256], [0,548,256], [0,0,1]])], len(world_grots), axis=0).astype(np.float64) #annots['camera_intrinsics']
        camera_extrinsics = np.repeat([np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])], len(world_grots), axis=0).astype(np.float64) #annots['camera_extrinsics']
    vertices = []
    for subject_id in range(subject_num):
        vertex = smpl_model(global_orient=world_grots[subject_id], body_pose=body_pose[subject_id], betas=smpl_betas[subject_id]).vertices.detach().cpu().numpy()
        vertices.append(vertex)
    
    frame_paths = sorted(glob.glob(os.path.join(seq_frame_dir, f'*.{img_ext}')))
    visualize_world_annots(seq_name, vertices, world_trans, camera_intrinsics, camera_extrinsics, frame_paths, np.asarray(smpl_model.faces))

if __name__ == '__main__':
    args = show_settings()
    results = np.load(args.preds_path, allow_pickle=True)['outputs'][()]
    visualize_subject_world_results('prediction', results, args.frame_dir, args.smpl_model_folder, img_ext=args.img_ext)