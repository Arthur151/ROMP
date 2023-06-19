from bev import BEV
import argparse
import os, sys
import os.path as osp
import numpy as np
import cv2
import torch
import pickle
from romp import ResultSaver
from romp.utils import progress_bar

set_id = 0
set_name = ['validation', 'test'][set_id]

# preparing data:
# 1. Register and download 1280x720 images of test set (5GB) and validation set (2GB) from https://agora.is.tue.mpg.de/download.php
# 2. Please change the dataset_dir to the path where AGORA are placed, meanwhile, output_save_dir to where you want to put the results
# 3. Please run this scipt on 'validation' (set_id=0) or 'test' (set_id=1) set. 
# 4.1 To get the results on validation set, please run 
# cd simple_romp/evaluation/
# python -m agora_evaluation.evaluate_agora --numBetas 11 --pred_path /home/yusun/data_drive/evaluation_results/AGORA/CVPR22_camera_ready_val/predictions --result_savePath /home/yusun/data_drive/evaluation_results/AGORA/CVPR22_camera_ready_val/

# DEBUG:
# BUG1: ValueError: Must have equal len keys and value when setting with an iterable, pip install pandas==1.0.3

# 4.2 To get the results on test set, please pack the generated predictions folder to predictions.zip and submit it to https://agora-evaluation.is.tuebingen.mpg.de/

visualize_results = False

dataset_dir = '/home/yusun/data_drive/dataset/AGORA'
output_save_dir = '/home/yusun/data_drive/evaluation_results/AGORA/CVPR22_camera_ready_{}'.format(set_name)
if osp.isdir(output_save_dir):
    import shutil
    shutil.rmtree(output_save_dir)
os.makedirs(output_save_dir,exist_ok=True)
prediction_save_dir = os.path.join(output_save_dir, 'predictions')
os.makedirs(prediction_save_dir,exist_ok=True)


default_eval_settings = argparse.Namespace(GPU=0, calc_smpl=True, center_thresh=[0.15,0.25][set_id], nms_thresh=40,\
    render_mesh = visualize_results, renderer = 'pyrender', show = False, show_largest = False, \
    input=None, frame_rate=24, temporal_optimize=False, smooth_coeff=3.0, relative_scale_thresh=2, overlap_ratio=0.46,\
    mode='image', model_path = '/home/yusun/CenterMesh/trained_models/BEV_Tabs/BEV_ft_agora.pth', onnx=False, crowd=False,\
    save_path = osp.join(output_save_dir,'visualization'), save_video=False, show_items='mesh', show_patch_results=False, \
    smpl_path='/home/yusun/.romp/smpla_packed_info.pth', smil_path='/home/yusun/.romp/smil_packed_info.pth')

if set_id == 0:
    annots = np.load('/home/yusun/data_drive/dataset/AGORA/annots_validation.npz',allow_pickle=True)['annots'][()]

def estimate_translation_cv2(joints_3d, joints_2d, proj_mat=None, cam_dist=None):
    camK = proj_mat
    ret, rvec, tvec,inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                              flags=cv2.SOLVEPNP_EPNP,reprojectionError=20,iterationsCount=100)
    if inliers is None:
        return None
    else:
        tra_pred = tvec[:,0]              
        return tra_pred

def estimate_translation(joints_3d, joints_2d, org_trans, proj_mats=None, cam_dists=None):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """
    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float32)
    if cam_dists is None:
        cam_dists = [None for _ in range(len(joints_2d))]
    # Find the translation for each example in the batch
    for i in range(joints_3d.shape[0]):
        trans_i = estimate_translation_cv2(joints_3d[i], joints_2d[i], 
                proj_mat=proj_mats, cam_dist=cam_dists[i])
        trans[i] = trans_i if trans_i is not None else org_trans[i]

    return torch.from_numpy(trans).float()

def save_agora_predictions_v6(image_path, outputs, save_dir):
    if set_id == 0:
        if os.path.basename(image_path) in annots:
            cam_params = annots[os.path.basename(image_path)][0]['camMats']
        else:
            cam_params = np.array([[995.55555556, 0., 640.],[0.,995.55555556,360.],[0.,0.,1.]])

        predicts_j3ds = outputs['joints']
        predicts_pj2ds = outputs['pj2d_org']
        predicts_j3ds = predicts_j3ds[:,:24] - predicts_j3ds[:,[0]]
        predicts_pj2ds = predicts_pj2ds[:,:24]
        outputs['cam_trans'] = estimate_translation(predicts_j3ds, predicts_pj2ds, outputs['cam_trans'],\
            proj_mats=cam_params)

    img_name = os.path.basename(image_path).strip('.png')
    for ind in range(len(outputs['smpl_thetas'])):
        result_dict = {'params':{}, 'pose2rot': True, 'num_betas': 11, 'gender': 'neutral', 'age': 'kid', 'kid_flag': True}
        result_dict['params']['global_orient'] = outputs['smpl_thetas'][ind,:3].reshape(1,1,3)
        result_dict['params']['body_pose'] = outputs['smpl_thetas'][ind,3:].reshape(1,23,3)
        result_dict['params']['betas'] = outputs['smpl_betas'][ind][None]
        result_dict['params']['transl'] = outputs['cam_trans'][ind][None]
        result_dict['joints'] = (outputs['pj2d_org'][ind][:24]+1)*3840/1280.

        save_name = os.path.join(save_dir,'{}_personId_{}.pkl'.format(img_name, ind))
        with open(save_name,'wb') as outfile:
            pickle.dump(result_dict, outfile, pickle.HIGHEST_PROTOCOL)

@torch.no_grad()
def get_results_on_AGORA(set_name='test'):
    image_folder = osp.join(dataset_dir, set_name)
    file_list = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
    
    model = BEV(default_eval_settings)

    if visualize_results:
        saver = ResultSaver(default_eval_settings.mode, default_eval_settings.save_path, save_npz=False)
    for image_path in progress_bar(file_list):
        image = cv2.imread(image_path)
        outputs = model(image)
        if outputs is None:
            continue
        if visualize_results:
            saver(outputs, image_path)
        save_agora_predictions_v6(image_path, outputs, prediction_save_dir)
    

if __name__ == '__main__':
    get_results_on_AGORA(set_name)