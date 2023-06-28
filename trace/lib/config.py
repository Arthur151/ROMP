import os,sys
import argparse
import math
import numpy as np
import torch
import yaml
import logging
import time
import platform 

currentfile = os.path.abspath(__file__)
code_dir = currentfile.replace('config.py','')
project_dir = currentfile.replace(os.path.sep+os.path.join('trace', 'lib', 'config.py'), '')
source_dir = currentfile.replace(os.path.sep+os.path.join('lib', 'config.py'), '')
root_dir = project_dir.replace(project_dir.split(os.path.sep)[-1], '')
data_cache_dir = os.path.join(project_dir, 'data_cacher')
os.makedirs(data_cache_dir, exist_ok=True)

time_stamp = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000))
yaml_timestamp = os.path.abspath(os.path.join(project_dir, "active_configs","active_context_{}.yaml".format(time_stamp).replace(":","_")))

data_dir = os.path.join(project_dir, 'project_data', 'trace_data')
model_dir = os.path.join(data_dir, 'model_data')
trained_model_dir = os.path.join(data_dir, 'trained_models')

#logging.info("yaml_timestamp", yaml_timestamp)

def parse_args(input_args=None):

    parser = argparse.ArgumentParser(description = 'ROMP: Monocular, One-stage, Regression of Multiple 3D People')
    parser.add_argument('--tab', type = str, default = 'ROMP_v1', help = 'additional tabs')
    parser.add_argument('--configs_yml', type = str, default = 'configs/v1.yml', help = 'settings') 

    mode_group = parser.add_argument_group(title='mode options')
    # mode settings
    mode_group.add_argument('--model_return_loss', type=bool, default=False,help = 'wether return loss value from the model for balanced GPU memory usage')
    mode_group.add_argument('--model_version',type = int,default = 1,help = 'model version')
    mode_group.add_argument('--multi_person',type = bool,default = True,help = 'whether to make Multi-person Recovery')
    mode_group.add_argument('--new_training',type = bool,default = False, help='learning centermap only in first few iterations for stable training.')
    mode_group.add_argument('--perspective_proj',type = bool,default = False,help = 'whether to use perspective projection, else use orthentic projection.')
    mode_group.add_argument('--center3d_loss',type = bool,default = False,help = 'whether to use dynamic supervision.')
    mode_group.add_argument('--high_resolution_input',type = bool,default = False,help = 'whether to process the high-resolution input.')
    mode_group.add_argument('--rotation360_aug',type = bool,default = False,help = 'whether to augment the rotation in -180~180 degree.')
    mode_group.add_argument('--relative_depth_scale_aug',type = bool,default = False,help = 'whether to augment scale of image.')
    mode_group.add_argument('--relative_depth_scale_aug_ratio',type = float,default = 0.25,help = 'the ratio of augmenting the scale of image for alleviating the depth uncertainty.')
    mode_group.add_argument('--high_resolution_folder',type = str,default = 'demo/mpii',help = 'path to high-resolution image.')
    mode_group.add_argument('--add_depth_encoding',type = bool,default = True, help = 'whether to add the depth encoding to the feature vector.')
    mode_group.add_argument('--old_trace_implement',type = bool,default = True, help = 'whether to add the depth encoding to the feature vector.')
    mode_group.add_argument('--video', type=bool, default=False,
                            help='whether to use the video model.')
    mode_group.add_argument('--tmodel_type', type=str, default='conv3D',
                            help='the architecture type of temporal model.')
    mode_group.add_argument('--clip_sampling_way', type=str, default='nooverlap',
                            help='the way of sampling n frames from a video given an anchor frame id. 3 way: nooverlap, overlap')
    mode_group.add_argument('--clip_sampling_position', type=str, default='middle',
                            help='the way of sampling n frames from a video given an anchor frame id. 3 way: start, middle, end')
    mode_group.add_argument('--clip_interval', type=int, default=1,
                            help='The number of non-overlapping interval frames between clips while data loading, like conv slides. 1 for overlapping')
    mode_group.add_argument('--video_batch_size', type=int, default=32,
                            help='The input frames of video inpute')
    mode_group.add_argument('--temp_clip_length', type=int, default=7,
                            help='The input frames of video inpute')
    mode_group.add_argument('--random_temp_sample_internal', type=int, default=6,
                            help='sampling the video clip with random interval between frames max=10, like sampling every 5 frames from the video to form a clip')
    mode_group.add_argument('--bev_distillation', type=bool, default=False,
                            help='whether to use the BEV to distillation some prority of .')
    mode_group.add_argument('--eval_hard_seq', type=bool, default=False,
                            help='whether to evaluate the checkpoint on hard sequence only for faster.')
    mode_group.add_argument('--image_datasets', type=str, default='agora',
                            help='image datasets used for learning more attributes.')
    mode_group.add_argument('--learn_cam_with_fbboxes', type=bool, default=False,
                            help='whether to learn camera parameter with full body bounding boxes.')
    mode_group.add_argument('--regressor_type', type=str, default='gru', help='transformer or mlpmixer')
    mode_group.add_argument('--with_gru', type=bool, default=True, help='transformer or mlpmixer')
    mode_group.add_argument('--dynamic_augment', type=bool, default=False, help='transformer or mlpmixer')
    mode_group.add_argument('--dynamic_augment_ratio',type = float,default = 0.6, help = 'possibility of performing dynamic augments')
    mode_group.add_argument('--dynamic_changing_ratio',type = float,default = 0.6, help = 'ratio of dynamic change / cropping width')
    mode_group.add_argument('--dynamic_aug_tracking_ratio',type = float,default = 0.5, help = 'ratio of dynamic augments via tracking a single target')
    mode_group.add_argument('--learn_foot_contact', type=bool, default=True, help='transformer or mlpmixer')
    mode_group.add_argument('--learn_motion_offset3D', type=bool, default=True, help='transformer or mlpmixer')
    mode_group.add_argument('--learn_cam_init', type=bool, default=False, help='transformer or mlpmixer') 
    mode_group.add_argument('--more_param_head_layer', type=bool, default=False, help='transformer or mlpmixer')   
    mode_group.add_argument('--compute_verts_org', type=bool, default=False) 
    mode_group.add_argument('--debug_tracking', type=bool, default=False) 
    mode_group.add_argument('--tracking_target_max_num', type=int, default=100)
    mode_group.add_argument('--video_show_results', type=bool, default=True) 
    mode_group.add_argument('--joint_num', type=int, default=44, help='44 for smpl, 73 for smplx')

    mode_group.add_argument('--render_option_path', type=str, default=os.path.join(source_dir, 'lib', 'visualization','vis_cfgs','render_options.json'), help='default rendering preference for Open3D')
    
    mode_group.add_argument('--using_motion_offsets_tracking', type=bool, default=True) 
    mode_group.add_argument('--tracking_with_kalman_filter', type=bool, default=False) 
    
    mode_group.add_argument('--use_optical_flow', type=bool, default=False)
    mode_group.add_argument('--raft_model_path', type=str, default=os.path.join(trained_model_dir, 'raft-things.pth'))

    mode_group.add_argument('--CGRU_temp_prop', type=bool, default=True)
    mode_group.add_argument('--learn_temp_cam_consist', type=bool, default=False)
    mode_group.add_argument('--learn_temp_globalrot_consist', type=bool, default=False)

    mode_group.add_argument('--learn_image', type=bool, default=True, help='whether to learn from image at the same time.')
    mode_group.add_argument('--image_repeat_time', type=int, default=2, help='whether to learn from image at the same time.')
    mode_group.add_argument('--drop_first_frame_loss', type=bool, default=True, help='drop the loss of the first frame to facilitate more stable loss learning.')
    mode_group.add_argument('--left_first_frame_num', type=int, default=2, help='drop the loss of the first frame to facilitate more stable loss learning.')
    
    mode_group.add_argument('--learn_dense_correspondence', type=bool, default=False,
                            help='whether to learn the dense correspondence between image pixel and IUV map (from densepose).')

    mode_group.add_argument('--learnbev2adjustZ', type=bool, default=False,
                            help='whether to fix the wrong cam_offset_bev adjustment from X to Z.')
    
    mode_group.add_argument('--image_loading_mode', type=str, default='image_relative', help='The Base Class (image, image_relative) used for loading image datasets.')
    mode_group.add_argument('--video_loading_mode', type=str, default='video_relative', help='The Base Class (image, image_relative, video_relative) used for loading video datasets.')
                            
    mode_group.add_argument('--temp_upsampling_layer', type=str, default='trilinear',
                            help='the way of upsampling in decoder of Trajectory3D model. 2 way: trilinear, deconv')
    mode_group.add_argument('--temp_transfomer_layer', type=int, default=3,
                            help='the number of transfomer layers. 2, 3, 4, 5, 6')
    mode_group.add_argument('--calc_smpl_mesh', type=bool, default=True,
                            help='whether to calculate smpl mesh during inference.')
    mode_group.add_argument('--calc_mesh_loss', type=bool, default=True,
                            help='whether to calculate smpl mesh during inference.')
    mode_group.add_argument('--eval_video', type=bool, default=False,
                            help='whether to evaluate on video benchmark.')
    mode_group.add_argument('--mp_tracker', type=str, default='byte',
                            help='Which tracker is employed to retrieve the 3D trjectory of multiple detected person.')
    
    mode_group.add_argument('--inference_video', type=bool, default=False,
                            help='run in inference mode.')
    mode_group.add_argument('--old_temp_model', type=bool, default=False,
                            help='run in inference mode.')
    mode_group.add_argument('--evaluation_gpu', type=int, default=1,
                            help='the gpu device used for evaluating the temporal model, better not using 0 to avoid out of memory during evaluation.')
    mode_group.add_argument('--deform_motion', type=bool, default=False,
                            help='run in inference mode.')
    mode_group.add_argument('--temp_simulator', type=bool, default=False,
                            help='use a simulator to simulate the temporal feature.')
    mode_group.add_argument('--tmodel_version', type=int, default=1,
                            help='the version ID of temporal model.')
                            
    mode_group.add_argument('--separate_smil_betas', type=bool, default=False,
                            help='estimating individual beta for smil baby model.')
    mode_group.add_argument('--no_evaluation', type=bool, default=False,
                            help='focus on training.')
    mode_group.add_argument('--temp_clip_length_eval', type=int, default=64,
                            help='The temp_clip_length during evaluation')
    mode_group.add_argument('--learn_temporal_shape_consistency', type=bool, default=False,
                            help='whether to learn the shape consistency in temporal dim.')
    mode_group.add_argument('--learn_deocclusion', type=bool, default=False,
                            help='focus on training.')
    mode_group.add_argument('--BEV_matching_gts2preds', type=str, default='3D+2D_center',
                            help='the way of properly matching the ground truths to the predictions.')
    mode_group.add_argument('--estimate_camera', type=bool, default=False,
                            help='also estimate the extrinsics and FOV of camera.')   
    mode_group.add_argument('--learn2Dprojection', type=bool, default=True,
                            help='also estimate the extrinsics and FOV of camera.')  
    mode_group.add_argument('--train_backbone', type=bool, default=True,
                            help='also estimate the extrinsics and FOV of camera.')
    mode_group.add_argument('--temp_cam_regression', type=bool, default=True,
                            help='regression of camera parameter in temporal mode.')
    mode_group.add_argument('--learn_cam_motion_composition_yz', type=bool, default=True,   
                            help='regression of camera parameter in temporal mode.')     
    mode_group.add_argument('--learn_cam_motion_composition_xyz', type=bool, default=False,   
                            help='regression of camera parameter in temporal mode.')  
    mode_group.add_argument('--learn_CamState', type=bool, default=False)
    mode_group.add_argument('--tracker_match_thresh',type = float,default = 1.2)
    mode_group.add_argument('--tracker_det_thresh',type = float,default = 0.18)  
    mode_group.add_argument('--feature_update_thresh',type = float,default = 0.3)
                            
    V6_group = parser.add_argument_group(title='V6 options')
    V6_group.add_argument('--bv_with_fv_condition',type = bool,default = True)
    V6_group.add_argument('--add_offsetmap',type = bool,default = True)
    V6_group.add_argument('--fv_conditioned_way',type = str,default = 'attention')
    V6_group.add_argument('--num_depth_level',type = int,default = 8,help = 'number of depth.')
    V6_group.add_argument('--scale_anchor',type = bool,default = True)
    V6_group.add_argument('--sampling_aggregation_way',type = str,default = 'floor')
    V6_group.add_argument('--acquire_pa_trans_scale',type = bool,default =False)
    V6_group.add_argument('--cam_dist_thresh',type = float,default = 0.1)
    
    # focal length: when FOV=50 deg 548 = H/2 * 1/(tan(FOV/2)) = 512/2. * 1./np.tan(np.radians(25))
    # focal length: when FOV=60 deg 443.4 = H/2 * 1/(tan(FOV/2)) = 512/2. * 1./np.tan(np.radians(30))
    # focal length: when FOV=72 deg 352 = H/2 * 1/(tan(FOV/2)) = 512/2. * 1./np.tan(np.radians(36))
    V6_group.add_argument('--focal_length',type=float, default = 443.4, help = 'Default focal length, adopted from JTA dataset')
    V6_group.add_argument('--multi_depth',type = bool,default = False,help = 'whether to use the multi_depth mode')
    V6_group.add_argument('--depth_degree',default=1,type=int,help = 'whether to use the multi_depth mode')
    V6_group.add_argument('--FOV',type=int, default = 60, help = 'Field of View')
    V6_group.add_argument('--matching_pckh_thresh',type=float, default = 0.6, help = 'Threshold to determine the sucess matching based on pckh')
    V6_group.add_argument('--baby_threshold',type = float,default = 0.8)
    
    train_group = parser.add_argument_group(title='training options')
    # basic training settings
    train_group.add_argument('--lr', help='lr',default=3e-4,type=float)
    train_group.add_argument('--adjust_lr_factor',type = float,default = 0.1,help = 'factor for adjusting the lr')
    train_group.add_argument('--weight_decay', help='weight_decay',default=1e-6,type=float)
    train_group.add_argument('--epoch', type = int, default = 80, help = 'training epochs')
    train_group.add_argument('--fine_tune',type = bool,default = True,help = 'whether to run online')
    train_group.add_argument('--gpu',default='0',help='gpus',type=str)
    train_group.add_argument('--batch_size',default=64,help='batch_size',type=int)
    train_group.add_argument('--input_size',default=512,type=int,help = 'size of input image')
    train_group.add_argument('--master_batch_size',default=-1,help='batch_size',type=int)
    train_group.add_argument('--nw',default=4,help='number of workers',type=int)
    train_group.add_argument('--optimizer_type',type = str,default = 'Adam',help = 'choice of optimizer')
    train_group.add_argument('--pretrain', type=str, default='simplebaseline',help='imagenet or spin or simplebaseline')
    train_group.add_argument('--fix_backbone_training_scratch',type = bool,default = False,help = 'whether to fix the backbone features if we train the model from scratch.')
    train_group.add_argument('--large_kernel_size',default=False, help='whether use large centermap kernel size',type=bool)
    model_group = parser.add_argument_group(title='model settings')
    # model settings
    model_group.add_argument('--backbone',type = str,default = 'hrnetv4',help = 'backbone model: resnet50 or hrnet')
    model_group.add_argument('--model_precision', type=str, default='fp16', help='the model precision: fp16/fp32')
    model_group.add_argument('--deconv_num', type=int, default=0)
    model_group.add_argument('--head_block_num',type = int,default = 2,help = 'number of conv block in head')
    model_group.add_argument('--merge_smpl_camera_head',type = bool,default = False)
    model_group.add_argument('--use_coordmaps',type = bool,default = True,help = 'use the coordmaps')
    model_group.add_argument('--hrnet_pretrain', type=str, default= os.path.join(data_dir,'trained_models/pretrain_hrnet.pkl'))
    model_group.add_argument('--resnet_pretrain', type=str, default= os.path.join(data_dir,'trained_models/pretrain_resnet.pkl'))
    model_group.add_argument('--resnet_pretrain_sb', type=str, default= os.path.join(data_dir,"trained_models/single_noeval_bs64_3dpw_106.2_75.0.pkl"))

    loss_group = parser.add_argument_group(title='loss options')
    # loss settings
    loss_group.add_argument('--loss_thresh',default=1000,type=float,help = 'max loss value for a single loss')
    loss_group.add_argument('--max_supervise_num',default=-1,type=int,help = 'max person number supervised in each batch for stable GPU memory usage')
    loss_group.add_argument('--supervise_cam_params',type = bool,default = False)
    loss_group.add_argument('--match_preds_to_gts_for_supervision',type = bool,default = False,help = 'whether to match preds to gts for supervision')
    loss_group.add_argument('--matching_mode', type=str, default='all',help='all | random_one | ')
    loss_group.add_argument('--supervise_global_rot',type = bool,default = False,help = 'whether supervise the global rotation of the estimated SMPL model')
    loss_group.add_argument('--HMloss_type', type=str, default='MSE', help='supervision for 2D pose heatmap: MSE or focal loss')
    loss_group.add_argument('--learn_gmm_prior',type=bool,default = False)

    eval_group = parser.add_argument_group(title='evaluation options')
    # basic evaluation settings
    eval_group.add_argument('--eval',type = bool,default = False,help = 'whether to run evaluation')
    # 'agora',, 'mpiinf' ,'pw3d', 'jta','h36m','pw3d','pw3d_pc','oh','h36m' # 'mupots','oh','h36m','mpiinf_test','oh',
    eval_group.add_argument('--eval_datasets',type = str,default = 'pw3d',help = 'whether to run evaluation')
    eval_group.add_argument('--val_batch_size',default=64,help='valiation batch_size',type=int)
    eval_group.add_argument('--test_interval',default=2000,help='interval iteration between validation',type=int)
    eval_group.add_argument('--fast_eval_iter',type = int,default = -1,help = 'whether to run validation on a few iterations, like 200.')
    eval_group.add_argument('--top_n_error_vis', type = int, default = 6, help = 'visulize the top n results during validation')
    eval_group.add_argument('--eval_2dpose',type = bool,default =False)
    eval_group.add_argument('--calc_pck', type = bool, default = False, help = 'whether calculate PCK during evaluation')
    eval_group.add_argument('--PCK_thresh', type = int, default = 150, help = 'training epochs')
    eval_group.add_argument('--calc_PVE_error',type = bool,default =False)

    maps_group = parser.add_argument_group(title='Maps options')
    maps_group.add_argument('--centermap_size', type=int, default=64, help='the size of center map')
    maps_group.add_argument('--centermap_conf_thresh', type=float, default=0.25, help='the threshold of the centermap confidence for the valid subject')
    maps_group.add_argument('--collision_aware_centermap',type = bool,default = False,help = 'whether to use collision_aware_centermap')
    maps_group.add_argument('--collision_factor',type = float,default = 0.2,help = 'whether to use collision_aware_centermap')
    maps_group.add_argument('--center_def_kp', type=bool, default=True,help = 'center definition: keypoints or bbox')

    distributed_train_group = parser.add_argument_group(title='options for distributed training')
    distributed_train_group.add_argument('--local_rank',type = int,default=0,help = 'local rank for distributed training')
    distributed_train_group.add_argument('--init_method',type = str,default='tcp://127.0.0.1:52468',help = 'URL:port of main server for distributed training')
    distributed_train_group.add_argument('--local_world_size',type = int,default=4,help = 'Number of processes participating in the job')
    distributed_train_group.add_argument('--distributed_training', type=bool, default=False,help = 'wether train model in distributed mode')

    reid_group = parser.add_argument_group(title='options for ReID')
    reid_group.add_argument('--with_reid',type = bool,default=False,help = 'whether estimate reid embedding')
    reid_group.add_argument('--reid_dim',type = int,default=64,help = 'channel number of reid embedding maps')

    relative_group = parser.add_argument_group(title='options for learning relativites')
    relative_group.add_argument('--learn_relative', type = bool,default = False)
    relative_group.add_argument('--learn_relative_age', type = bool,default = False)
    relative_group.add_argument('--learn_relative_depth', type = bool,default = False)
    relative_group.add_argument('--depth_loss_type', type=str, default='Log',help='Log | Piecewise | ')
    relative_group.add_argument('--learn_uncertainty', type = bool,default = False)

    log_group = parser.add_argument_group(title='log options')
    # basic log settings
    log_group.add_argument('--print_freq', type = int, default = 50, help = 'training epochs')
    log_group.add_argument('--model_path',type = str,default = '',help = 'trained model path')
    log_group.add_argument('--temp_model_path',type = str,default = '',help = 'trained model path')
    log_group.add_argument('--log-path', type = str, default = os.path.join(root_dir,'log/'), help = 'Path to save log file')

    hm_ae_group = parser.add_argument_group(title='learning 2D pose/associate embeddings options')
    hm_ae_group.add_argument('--learn_2dpose', type = bool,default = False)
    hm_ae_group.add_argument('--learn_AE', type = bool,default = False)
    hm_ae_group.add_argument('--learn_kp2doffset', type = bool,default = False)

    augmentation_group = parser.add_argument_group(title='augmentation options')
    # augmentation settings
    augmentation_group.add_argument('--shuffle_crop_mode',type = bool,default = True,help = 'whether to shuffle the data loading mode between crop / uncrop for indoor 3D pose datasets only')
    augmentation_group.add_argument('--shuffle_crop_ratio_3d',default=0.9,type=float,help = 'the probability of changing the data loading mode from uncrop multi_person to crop single person')
    augmentation_group.add_argument('--shuffle_crop_ratio_2d',default=0.9,type=float,help = 'the probability of changing the data loading mode from uncrop multi_person to crop single person')
    augmentation_group.add_argument('--Synthetic_occlusion_ratio',default=0,type=float,help = 'whether to use use Synthetic occlusion')
    augmentation_group.add_argument('--color_jittering_ratio',default=0.2,type=float,help = 'whether to use use color jittering')
    augmentation_group.add_argument('--rotate_prob',default=0.2,type=float,help = 'whether to use rotation augmentation')

    dataset_group = parser.add_argument_group(title='datasets options')
    #dataset setting:
    dataset_group.add_argument('--dataset_rootdir',type=str, default='/home/yusun/DataCenter/datasets', help= 'root dir of all datasets') #os.path.join(root_dir,'datasets/')
    dataset_group.add_argument('--datasets',type=str, default='h36m,mpii,coco,aich,up,ochuman,lsp,movi' ,help = 'which datasets are used')
    dataset_group.add_argument('--voc_dir', type = str, default = os.path.join(root_dir, 'datasets/VOC2012/'), help = 'VOC dataset path')
    dataset_group.add_argument('--max_person',default=64,type=int,help = 'max person number of each image')
    dataset_group.add_argument('--homogenize_pose_space',type = bool,default = False,help = 'whether to homogenize the pose space of 3D datasets')
    dataset_group.add_argument('--use_eft', type=bool, default=True,help = 'wether use eft annotations for training')
    
    smpl_group = parser.add_argument_group(title='SMPL options')
    smpl_group.add_argument('--smpl_mesh_root_align', type=bool, default=True)
    mode_group.add_argument('--Rot_type', type=str, default='6D', help='rotation representation type: angular, 6D')
    mode_group.add_argument('--rot_dim', type=int, default=6, help='rotation representation type: 3D angular, 6D')
    smpl_group.add_argument('--cam_dim', type=int, default=3, help = 'the dimention of camera param')
    smpl_group.add_argument('--beta_dim', type=int, default=10, help = 'the dimention of SMPL shape param, beta')
    smpl_group.add_argument('--smpl_joint_num', type=int, default=22, help = 'joint number of SMPL model we estimate')
    
    smpl_group.add_argument('--smpl_model_path', type = str,default = os.path.join(model_dir, 'parameters', 'SMPL_NEUTRAL.pth'),help = 'smpl model path')
    smpl_group.add_argument('--smpla_model_path', type = str,default = os.path.join(model_dir, 'parameters', 'SMPLA_NEUTRAL.pth'),help = 'smpl model path') #SMPLA_FEMALE gets better MPJPE #smpla_packed_info.pth
    smpl_group.add_argument('--smil_model_path', type = str,default = os.path.join(model_dir, 'parameters', 'SMIL_NEUTRAL.pth'),help = 'smpl model path')
    smpl_group.add_argument('--smpl_prior_path', type = str,default = os.path.join(model_dir,'parameters','gmm_08.pkl'),help = 'smpl model path')

    smpl_group.add_argument('--smpl_J_reg_h37m_path',type = str,default = os.path.join(model_dir, 'parameters', 'J_regressor_h36m.npy'),help = 'SMPL regressor for 17 joints from H36M datasets')
    smpl_group.add_argument('--smpl_J_reg_extra_path',type = str,default = os.path.join(model_dir, 'parameters', 'J_regressor_extra.npy'),help = 'SMPL regressor for 9 extra joints from different datasets')

    smpl_group.add_argument('--smplx_model_folder',type = str,default = os.path.join(model_dir, 'parameters'), help = 'folder containing SMPLX folder')
    smpl_group.add_argument('--smplx_model_path',type = str,default = os.path.join(model_dir, 'parameters', 'SMPLX_NEUTRAL.pth'), help = 'folder containing SMPLX folder')
    smpl_group.add_argument('--smplxa_model_path',type = str,default = os.path.join(model_dir, 'parameters', 'SMPLXA_NEUTRAL.pth'), help = 'folder containing SMPLX folder')

    smpl_group.add_argument('--smpl_model_type',type = str,default = 'smpl', help = 'wether to use smpl, SMPL+A, SMPL-X')
    
    smpl_group.add_argument('--smpl_uvmap',type = str,default = os.path.join(model_dir, 'parameters', 'smpl_vt_ft.npz'),help = 'smpl UV Map coordinates for each vertice')
    smpl_group.add_argument('--wardrobe', type = str, default=os.path.join(model_dir, 'wardrobe'), help = 'path of smpl UV textures')
    smpl_group.add_argument('--cloth',type = str,default = 'f1',help = 'pick up cloth from the wardrobe or simplely use a single color')

    debug_group = parser.add_argument_group(title='Debug options')
    debug_group.add_argument('--track_memory_usage',type = bool,default = False)

    parsed_args = parser.parse_args(args=input_args)
    parsed_args.adjust_lr_epoch = []
    parsed_args.kernel_sizes = [5]
    
    with open(parsed_args.configs_yml) as file:
        configs_update = yaml.full_load(file)
    for key, value in configs_update['ARGS'].items():
        if sum(['--{}'.format(key) in input_arg for input_arg in input_args])==0:
            if isinstance(value,str):
                exec("parsed_args.{} = '{}'".format(key, value))
            else:
                exec("parsed_args.{} = {}".format(key, value))
    if 'loss_weight' in configs_update:
        for key, value in configs_update['loss_weight'].items():
            exec("parsed_args.{}_weight = {}".format(key, value))
    if 'sample_prob' in configs_update:
        parsed_args.sample_prob_dict = configs_update['sample_prob']
    if 'image_sample_prob' in configs_update:
        parsed_args.image_sample_prob_dict = configs_update['image_sample_prob']
    
    if parsed_args.large_kernel_size:
        parsed_args.kernel_sizes = [11]

    if parsed_args.video:
        parse_args.tab = '{}_bs{}_tcl{}_{}'.format(parsed_args.tab,
                                        parsed_args.batch_size,
                                        parsed_args.temp_clip_length,
                                        parsed_args.datasets)
    else:
        parsed_args.tab = '{}_cm{}_{}'.format(parsed_args.backbone,
                                          parsed_args.centermap_size,
                                          parsed_args.tab,
                                          parsed_args.datasets)

    if parsed_args.distributed_training:
        parsed_args.local_rank = int(os.environ["LOCAL_RANK"])
    print('kernel sizes:', parsed_args.kernel_sizes)
    return parsed_args


class ConfigContext(object):
    """
    Class to manage the active current configuration, creates temporary `yaml`
    file containing the configuration currently being used so it can be
    accessed anywhere.
    """
    yaml_filename = yaml_timestamp
    parsed_args = parse_args(sys.argv[1:]) 
    def __init__(self, parsed_args=None):
        if parsed_args is not None:
            self.parsed_args = parsed_args

    def __enter__(self):
        # if a yaml is left over here, remove it
        self.clean()
        # store all the parsed_args in a yaml file
        with open(self.yaml_filename, 'w') as f:
            d = self.parsed_args.__dict__
            yaml.dump(d, f)

    def __forceyaml__(self, filepath):
        # if a yaml is left over here, remove it
        self.yaml_filename = filepath
        self.clean()
        # store all the parsed_args in a yaml file
        with open(self.yaml_filename, 'w') as f:
            d = self.parsed_args.__dict__
            yaml.dump(d, f)
            logging.info("----------------------------------------------")
            logging.info("__forceyaml__ DUMPING YAML ")
            logging.info("self.yaml_filename", self.yaml_filename)
            logging.info("----------------------------------------------")
            
    def clean(self):
        if os.path.exists(self.yaml_filename):
            os.remove(self.yaml_filename)

    def __exit__(self, exception_type, exception_value, traceback):
        # delete the yaml file
        self.clean()

def args():
    return ConfigContext.parsed_args