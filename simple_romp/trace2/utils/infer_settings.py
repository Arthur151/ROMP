import os, sys
import numpy as np
import argparse
import torch
import copy
from .utils import download_model

trace_model_dir = os.path.join(os.path.expanduser("~"),'.romp', 'TRACE_models')

def trace_settings(input_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='TRACE: 5D Temporal Regression of Avatars with Dynamic Cameras in 3D Environments')
    parser.add_argument('-m', '--mode', type=str, default='video', help = 'trace only support video mode for now')
    parser.add_argument('-i', '--input', type=str, default=os.path.join(os.path.expanduser("~"),'Desktop', 'trace_demo'), help = 'Path to the input image / video')
    parser.add_argument('-o', '--save_path', type=str, default=os.path.join(os.path.expanduser("~"),'TRACE_results'), help = 'Path to save the results')
    parser.add_argument('--GPU', type=int, default=0, help = 'The gpu device number to run the inference on. If GPU=-1, then running in cpu mode')
    parser.add_argument('--center_thresh', type=float, default=0.05, help = 'The confidence threshold of positive detection in 2D human body center heatmap.')
    
    parser.add_argument('--save_video', action='store_true', help = 'Whether to save the video results')
    parser.add_argument('--show_tracking', action='store_true', help = 'Whether to save the video results')
    parser.add_argument('--renderer', type=str, default='sim3dr', help = 'Choose the renderer for visualizaiton: pyrender (great but slow), sim3dr (fine but fast), open3d (webcam)')
    parser.add_argument('--show', action='store_true', help = 'Whether to show the rendered results')
    parser.add_argument('--frame_rate', type=int, default=24, help = 'The frame_rate of saved video results')

    parser.add_argument('--smpl_path', type=str, default=os.path.join(os.path.expanduser("~"),'.romp','SMPLA_NEUTRAL.pth'), help = 'The path of SMPL-A model file')
    parser.add_argument('--trace_head_model_path', type=str, default=os.path.join(trace_model_dir,'trace_head.pkl'), help = 'The path of TRACE head checkpoint')
    parser.add_argument('--image_backbone_model_path', type=str, default=os.path.join(trace_model_dir,'trace_image_backbone.pkl'), help = 'The path of TRACE image backbone (BEV) checkpoint')
    parser.add_argument('--raft_model_path', type=str, default=os.path.join(trace_model_dir,'trace_motion_backbone.pth'), help = 'The path of motion backbone, RAFT')

    # not support temporal processing now
    parser.add_argument('-t', '--temporal_optimize', action='store_true', help = 'Whether to use OneEuro filter to smooth the results')
    parser.add_argument('-sc','--smooth_coeff', type=float, default=3., help = 'The smoothness coeff of OneEuro filter, the smaller, the smoother.')
    parser.add_argument('--webcam_id',type=int, default=0, help = 'The Webcam ID.')
    parser.add_argument('--FOV', type=int, default=50, help = 'Field of View of our pre-defined camera system')

    parser.add_argument('--tracker_det_thresh',type = float,default = 0.18)  
    parser.add_argument('--tracker_match_thresh',type = float,default = 1.2)
    parser.add_argument('--first_frame_det_thresh',type = float,default = 0.3)
    parser.add_argument('--accept_new_dets',type=bool, default = False)
    parser.add_argument('--new_subject_det_thresh',type = float,default = 0.8)
    parser.add_argument('--time2forget', type = int,default = 0)
    parser.add_argument('--large_object_thresh',type = float,default = 0.13)
    parser.add_argument('--suppress_duplicate_thresh',type = float,default = 0.05)
    parser.add_argument('--motion_offset3D_norm_limit',type = float,default = 0.06)
    parser.add_argument('--feature_update_thresh',type = float,default = 0.05)
    parser.add_argument('--feature_inherent',type=bool, default = True)
    parser.add_argument('--occlusion_cam_inherent_or_interp',type=bool, default = False)
    parser.add_argument('--subject_num', type = int,default = 1)
    parser.add_argument('--temp_clip_length', type = int,default = 8)

    parser.add_argument('--smooth_pose_shape', type=bool, default = True)
    parser.add_argument('--pose_smooth_coef', type = float,default = 1)
    parser.add_argument('--eval_video', type=bool, default = False)
    parser.add_argument('--val_batch_size', type = int,default = 1)
    parser.add_argument('--image_batch_size', type = int,default = 16)
    
    parser.add_argument('--results_save_dir', type=str, default='/home/yusun/TRACE_results')
    parser.add_argument('--eval_dataset', type=str, default='DynaCam')
    
    args = parser.parse_args(input_args)

    if not torch.cuda.is_available():
        args.GPU = -1
    if not os.path.exists(args.trace_head_model_path):
        os.makedirs(trace_model_dir, exist_ok=True)
        model_url = 'https://github.com/Arthur151/ROMP/releases/download/V3.0/trace_head.pkl'
        download_model(model_url, args.trace_head_model_path, 'TRACE head')
    if not os.path.exists(args.image_backbone_model_path):
        os.makedirs(trace_model_dir, exist_ok=True)
        model_url = 'https://github.com/Arthur151/ROMP/releases/download/V3.0/trace_image_backbone.pkl'
        download_model(model_url, args.image_backbone_model_path, 'TRACE image backbone')
    if not os.path.exists(args.raft_model_path):
        os.makedirs(trace_model_dir, exist_ok=True)
        model_url = 'https://github.com/Arthur151/ROMP/releases/download/V3.0/trace_motion_backbone.pth'
        download_model(model_url, args.raft_model_path, 'TRACE motion backbone')
    if not os.path.exists(args.input):
        demo_url = 'https://github.com/Arthur151/ROMP/releases/download/V3.0/trace_demo.zip'
        download_model(demo_url, args.input+'.zip', 'DEMO')
        print(f'Attension: please unzip the download demo video file at {args.input}.zip and specifiy the path to input')
        demo_url = 'https://github.com/Arthur151/ROMP/releases/download/V3.0/trace_demo2.zip'
        download_model(demo_url, args.input+'.zip', 'DEMO')
        print(f'Attension: please unzip the download demo video file at {args.input}.zip and specifiy the path to input')

    return args

def get_seq_cfgs(args):
    default_cfgs = {
        'tracker_det_thresh': args.tracker_det_thresh, 
        'tracker_match_thresh': args.tracker_match_thresh,
        'first_frame_det_thresh': args.first_frame_det_thresh, #  to find the target in the first frame
        'accept_new_dets': args.accept_new_dets,
        'new_subject_det_thresh': args.new_subject_det_thresh, 
        'time2forget': args.time2forget, # for avoiding forgeting long-term occlusion subjects, 30 per second
        'large_object_thresh': args.large_object_thresh,
        'suppress_duplicate_thresh': args.suppress_duplicate_thresh,
        'motion_offset3D_norm_limit': args.motion_offset3D_norm_limit,
        'feature_update_thresh': args.feature_update_thresh,
        'feature_inherent': args.feature_inherent,
        'occlusion_cam_inherent_or_interp': args.occlusion_cam_inherent_or_interp, # True for directly inherent, False for interpolation
        'subject_num': args.subject_num,
        'axis_times': np.array([1.2, 2.5, 25]), #np.array([1.2, 2.5, 16])
        'smooth_pose_shape': args.smooth_pose_shape, 'pose_smooth_coef':args.pose_smooth_coef, 'smooth_pos_cam': False}  
    return default_cfgs

def update_seq_cfgs(seq_name, default_cfgs):
    seq_cfgs = copy.deepcopy(default_cfgs)

    sequence_cfgs = {

    }

    if seq_name in sequence_cfgs:
        seq_cfgs.update(sequence_cfgs[seq_name])

    return seq_cfgs