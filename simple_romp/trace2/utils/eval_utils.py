import glob
import os
import numpy as np

def pj2ds_to_bbox(pj2ds, enlarge_xy=np.array([1.1,1.18])): # enlarge_xy=np.array([1.1,1.18]) used for evaluation on MuPoTS
    tracked_bbox = np.array([pj2ds[:,0].min(), pj2ds[:,1].min(), pj2ds[:,0].max(), pj2ds[:,1].max()])
    # left, top, right, down -> left, top, width, height
    center = (tracked_bbox[2:] + tracked_bbox[:2]) / 2
    tracked_bbox[2:] = (center - tracked_bbox[:2]) * enlarge_xy
    tracked_bbox[:2] = center - tracked_bbox[2:]
    tracked_bbox[2:] = tracked_bbox[2:] * 2
    return tracked_bbox

def adjust_tracking_results(results):
    for seq_name in list(results.keys()):
        tracked_frames = sorted(list(results[seq_name].keys()))
        for frame_name in tracked_frames:
            for ind, (bbox, pj2ds) in enumerate(zip(results[seq_name][frame_name]['track_bbox'], results[seq_name][frame_name]['pj2ds'])):
                results[seq_name][frame_name]['track_bbox'][ind] = pj2ds_to_bbox(pj2ds)
    return results

def convert2MOTChallenge_format(results, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for seq_name, predictions in results.items():
        tracked_frames = sorted(list(predictions.keys()))
        with open(os.path.join(save_folder,seq_name+'.txt'), 'w') as f:
            for frame_name in tracked_frames:
                frame_id = int(frame_name.replace('.jpg', '').replace('img_', '').replace('image_', '')) + 1
                for track_id, bbox, pj2ds in zip(predictions[frame_name]['track_ids'], predictions[frame_name]['track_bbox'], predictions[frame_name]['pj2ds']):
                    bbox = bbox[0] if len(bbox) == 1 else bbox
                    left, top, width, height = pj2ds_to_bbox(pj2ds)
                    track_id += 1
                    str_line = '{}, {}, {}, {}, {}, {}, -1, -1, -1, -1\n'\
                        .format(frame_id, track_id, left, top, width, height)
                    f.write(str_line)
    return save_folder

def load_previous_predictions(track_results_save_folder):
    tracking_results, kp3d_results = {}, {}
    for results_path in glob.glob(os.path.join(track_results_save_folder, '*.npz')):
        seq_name = os.path.basename(results_path).replace('.npz', '')
        tracking_results[seq_name] = np.load(results_path, allow_pickle=True)['tracking'][()]
        kp3d_results[seq_name] = np.load(results_path, allow_pickle=True)['kp3ds'][()]
    return tracking_results, kp3d_results

def prepare_video_frame_dict(video_path_list, img_ext='jpg'):
    video_frame_dict = {}
    for video_path in video_path_list:
        frame_list = sorted(glob.glob(os.path.join(video_path,'*.'+img_ext)))
        video_frame_dict[video_path] = frame_list
    return video_frame_dict

Dyna3DPW_seq_names = ['downtown_cafe_00', 'downtown_walkBridge_01', 'downtown_runForBus_01', 'downtown_sitOnStairs_00', \
                    'downtown_bar_00', 'downtown_enterShop_00', 'downtown_rampAndStairs_00', 'downtown_runForBus_00', \
                    'downtown_car_00',  'downtown_walking_00', 'downtown_crossStreets_00', 'office_phoneCall_00', 'downtown_warmWelcome_00',\
                    'downtown_weeklyMarket_00', 'downtown_bus_00', 'downtown_windowShopping_00']
PW3D_test_seq_names = [
    'downtown_enterShop_00', 'flat_packBags_00', 'downtown_walkBridge_01', 'downtown_bus_00', 'downtown_weeklyMarket_00', \
    'downtown_walkUphill_00', 'downtown_warmWelcome_00', 'office_phoneCall_00', 'downtown_crossStreets_00', 'downtown_upstairs_00',\
    'downtown_stairs_00', 'downtown_walking_00', 'downtown_downstairs_00', 'downtown_car_00', 'downtown_windowShopping_00', \
    'flat_guitar_01', 'downtown_arguing_00', 'downtown_runForBus_00', 'downtown_rampAndStairs_00', 'downtown_cafe_00', \
    'downtown_bar_00', 'downtown_sitOnStairs_00', 'downtown_runForBus_01', 'outdoors_fencing_01'] 
del_seqs = ['residence4_swing2', 'residence7_riding', 'office7_entrence-0-0', 'pano-fitness5','pano-fitness6']

def get_evaluation_sequence_dict(datasets=None, dataset_dir=None):
    if datasets == 'DynaCam-Panorama':
        dynacam_pano_test_dir = os.path.join(dataset_dir, 'video_frames', 'panorama_test')
        video_path_list = []
        for seq_name in os.listdir(dynacam_pano_test_dir):
            if sum([seq in seq_name for seq in del_seqs]) == 0:
                video_path_list.append(os.path.join(dynacam_pano_test_dir, seq_name))
        sequence_dict = prepare_video_frame_dict(video_path_list, img_ext='jpg')
    elif datasets == 'DynaCam-Translation':
        dynacam_tran_test_dir = os.path.join(dataset_dir, 'video_frames', 'translation_test')
        video_path_list = glob.glob(os.path.join(dynacam_tran_test_dir, '*'))
        sequence_dict = prepare_video_frame_dict(video_path_list, img_ext='png')
    elif datasets == 'mupots':
        video_path_list = [os.path.join(dataset_dir, f'TS{sid}') for sid in range(1,21)]
        sequence_dict = prepare_video_frame_dict(video_path_list, img_ext='jpg')
    elif datasets == 'Dyna3DPW':
        video_path_list = [os.path.join(dataset_dir, f'{name}') for name in Dyna3DPW_seq_names]
        sequence_dict = prepare_video_frame_dict(video_path_list, img_ext='jpg')
    elif datasets == '3DPW':
        video_path_list = [os.path.join(dataset_dir, 'imageFiles', f'{name}') for name in PW3D_test_seq_names]
        sequence_dict = prepare_video_frame_dict(video_path_list, img_ext='jpg')
    return sequence_dict

eval_sequence_cfgs = {
        'TS2': {'subject_num': 3, 'feature_update_thresh': 0.1}, 'TS14': {'subject_num': 3, 'feature_update_thresh': 0.3},
        'TS6': {'subject_num': 2},  'TS11': {'subject_num': 2}, 'TS4': {'first_frame_det_thresh': 0.05}, 
        'TS5': {'large_object_thresh': 0.2}, 'TS18': {'large_object_thresh': 0.2}, 'TS19': {'large_object_thresh': 0.2}, 

        'downtown_car_00': {'subject_num': 2, 'tracker_match_thresh': 1.6}, 'downtown_crossStreets_00': {'subject_num': 2, 'tracker_match_thresh': 0.8},  #, 'axis_times': np.array([0.8, 2.5, 16])
        'downtown_walkBridge_01': {'subject_num': 1}, 'downtown_weeklyMarket_00': {'subject_num': 1},
        'downtown_bar_00': {'subject_num': 2}, 'downtown_cafe_00': {'subject_num': 2}, 'downtown_downstairs_00': {'subject_num': 2},
        'downtown_enterShop_00': {'subject_num': 1, 'tracker_match_thresh': 1.2}, 'downtown_rampAndStairs_00': {'subject_num': 2},'downtown_runForBus_00': {'subject_num': 2},
        'downtown_runForBus_01': {'subject_num': 2, 'feature_update_thresh':0.54}, 'downtown_sitOnStairs_00': {'subject_num': 2}, 'downtown_upstairs_00': {'subject_num': 1},
        'downtown_walking_00': {'subject_num': 2, 'tracker_match_thresh': 1.4}, 'downtown_warmWelcome_00': {'subject_num': 2},
        'downtown_windowShopping_00': {'subject_num': 1},'office_phoneCall_00': {'subject_num': 2}, 
        'downtown_bus_00': {'subject_num': 2, 'new_subject_det_thresh':0.05, 'tracker_match_thresh': 0.6},
        
        'pexels-group_running': {'subject_num': 5},

        '160422_haggling1-00_16-1': {'subject_num': 3}, '160422_haggling1-00_16-2': {'subject_num': 3}, 
        '160422_haggling1-00_30-1': {'subject_num': 3}, '160422_haggling1-00_30-2': {'subject_num': 3},
        '160422_mafia2-00_16-1': {'subject_num': 8}, '160422_mafia2-00_16-2': {'subject_num': 7}, 
        '160422_mafia2-00_30-1': {'subject_num': 8, 'first_frame_det_thresh':0.2}, '160422_mafia2-00_30-2': {'subject_num': 7},
        '160422_ultimatum1-00_16-1': {'subject_num': 2}, '160422_ultimatum1-00_16-2': {'subject_num': 3}, 
        '160422_ultimatum1-00_30-1': {'subject_num': 2}, '160422_ultimatum1-00_30-2': {'subject_num': 3},
        '160906_pizza1-00_16-1': {'subject_num': 5}, '160906_pizza1-00_16-2': {'subject_num': 6}, 
        '160906_pizza1-00_30-1': {'subject_num': 5, 'first_frame_det_thresh':0.5}, '160906_pizza1-00_30-2': {'subject_num': 6, 'first_frame_det_thresh':0.4},
        
        'mpii-dancing-aerobic,general-049617593-0': {'subject_num': 2}, 'mpii-dancing-aerobic,general-083277907-0': {'subject_num': 2}, 
        'mpii-dancing-aerobic,general-089099537-0': {'subject_num': 2}, 'mpii-dancing-Irishstepdancing-082119828-0': {'subject_num': 3},
        'mpii-occupation-horseracing-067916257-0': {'subject_num': 2}, 'mpii-running-jogging,onamini-tramp-011995294-0': {'subject_num': 2},
        'mpii-running-running-089114311-0': {'subject_num': 2}, 'mpii-sports-ropeskipping,general-000003072-0': {'subject_num': 2}}

dataset_cfgs = {
        'mupots': 
        {'tracker_det_thresh': 0.05, 'tracker_match_thresh': 1.2, 'first_frame_det_thresh': 0.16,
        'accept_new_dets': False, 'new_subject_det_thresh': 0.7,  'time2forget': 2000, 'subject_num':3,
        'large_object_thresh': 0.13, 'suppress_duplicate_thresh': 0.05, 'motion_offset3D_norm_limit': 0.06,
        'feature_update_thresh': 0.2, 'feature_inherent': True, 'occlusion_cam_inherent_or_interp': False, \
        'axis_times': np.array([1.1, 0.9, 20]), 'smooth_pose_shape': False, 'pose_smooth_coef':1., 'smooth_pos_cam': False},
        'Dyna3DPW': 
        {'tracker_det_thresh': 0.05, 'tracker_match_thresh': 0.8, 'first_frame_det_thresh': 0.05,
        'accept_new_dets': False, 'new_subject_det_thresh': 0.8,  'time2forget': 1000, 'subject_num':3,
        'large_object_thresh': 0.13, 'suppress_duplicate_thresh': 0.05, 'motion_offset3D_norm_limit': 0.06,
        'feature_update_thresh': 0.05, 'feature_inherent': True, 'occlusion_cam_inherent_or_interp': False, \
        'axis_times': np.array([1.2, 2.5, 25]), 'smooth_pose_shape': True, 'pose_smooth_coef':1., 'smooth_pos_cam': False},
        '3DPW': 
        {'tracker_det_thresh': 0.05, 'tracker_match_thresh': 1.2, 'first_frame_det_thresh': 0.05,
        'accept_new_dets': True, 'new_subject_det_thresh': 0.8,  'time2forget': 20, 'subject_num':3,
        'large_object_thresh': 0.1, 'suppress_duplicate_thresh': 0.05, 'motion_offset3D_norm_limit': 0.06,
        'feature_update_thresh': 0.05, 'feature_inherent': True, 'occlusion_cam_inherent_or_interp': False, \
        'axis_times': np.array([1.2, 2.5, 25]), 'smooth_pose_shape': False, 'pose_smooth_coef':1., 'smooth_pos_cam': False}} 

def update_eval_seq_cfgs(seq_name, default_cfgs, ds_name='demo'):
    seq_cfgs = dataset_cfgs[ds_name] if ds_name in dataset_cfgs else default_cfgs
    if seq_name in eval_sequence_cfgs:
        seq_cfgs.update(eval_sequence_cfgs[seq_name])
    return seq_cfgs