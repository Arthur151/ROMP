import sys, os
from datasets.image_base import *
from datasets.base import Base_Classes, Test_Funcs
import joblib

set_names = {'all':['train','val','test'],'test':['test'],'val':['train','val','test']}
PW3D_PCsubset = {'courtyard_basketball_00':[200,280], 'courtyard_captureSelfies_00':[500,600],\
                'courtyard_dancing_00':[60,370],  'courtyard_dancing_01':[60,270], 'courtyard_hug_00':[100,500], 'downtown_bus_00':[1620,1900]}
# 'courtyard_basketball_00':[110,160], 'courtyard_captureSelfies_00':[150,270],

PW3D_OCsubset = ['courtyard_backpack','courtyard_basketball','courtyard_bodyScannerMotions','courtyard_box','courtyard_golf','courtyard_jacket',\
'courtyard_laceShoe','downtown_stairs','flat_guitar','flat_packBags','outdoors_climbing','outdoors_crosscountry','outdoors_fencing','outdoors_freestyle',\
'outdoors_golf','outdoors_parcours','outdoors_slalom']
PW3D_NOsubset = {}

# evaluate the ordinal depth
PW3D_ODsubset = ['downtown_arguing_00', 'downtown_bar_00', 'downtown_bus_00', 'downtown_cafe_00', \
                 'downtown_car_00', 'downtown_crossStreets_00','downtown_downstairs_00', 'downtown_rampAndStairs_00',\
                 'downtown_runForBus_00', 'downtown_runForBus_01', 'downtown_sitOnStairs_00', 'downtown_upstairs_00',\
                 'downtown_walking_00', 'downtown_warmWelcome_00', 'office_phoneCall_00']

PW3D_CDsubset = ['courtyard_hug_00', 'courtyard_dancing_00'] # used as 3DPW_Crowd in evaluation of 3DCrowdNet

default_mode = args().video_loading_mode if args().video else args().image_loading_mode
invalida_detection_seqs = ['downtown_arguing_00', 'downtown_bus_00', 'downtown_cafe_00', 'downtown_cafe_01', 'downtown_car_00',\
    'downtown_crossStreets_00', 'downtown_downstairs_00', 'downtown_enterShop_00', 'downtown_rampAndStairs_00', 'downtown_runForBus_00',\
    'downtown_runForBus_01', 'downtown_sitOnStairs_00', 'downtown_upstairs_00', 'downtown_walkBridge_01', 'downtown_walkDownhill_00', \
    'downtown_walking_00', 'downtown_walkUphill_00', 'downtown_warmWelcome_00', 'downtown_weeklyMarket_00', 'downtown_windowShopping_00']


def PW3D(base_class=default_mode):
    class PW3D(Base_Classes[base_class]):
        def __init__(self,train_flag = False, split='train', mode='vibe', regress_smpl=True, load_entire_sequence=False, eval_hard_seq=False, load_vertices=False, **kwargs):
            #if train_flag:
            #    mode, split, regress_smpl = ['normal', 'train', True]
            super(PW3D, self).__init__(train_flag, regress_smpl=regress_smpl, load_vertices=load_vertices,
                                    load_entire_sequence=load_entire_sequence)
            self.data_folder = os.path.join(self.data_folder,'3DPW/').replace('DataCenter', 'DataCenter2')
            self.data3d_dir = os.path.join(self.data_folder,'sequenceFiles')
            self.image_dir = os.path.join(self.data_folder,'imageFiles')
            
            self.mode = mode
            self.split = split
            self.regress_smpl = regress_smpl
            self.eval_hard_seq = eval_hard_seq
            if self.eval_hard_seq:
                print('Evaluating the hard sequence only', constants.pw3d_hard_sequences)
            
            self.val_sample_ratio = 5
            self.scale_range = [1.5,2.2]
            self.dataset_name = {'PC':'pw3d_pc', 'NC':'pw3d_nc','OC':'pw3d_oc','vibe':'pw3d_vibe', 'normal':'pw3d_normal', 'OD':'pw3d_od', 'CD':'pw3d_cd'}[mode]
            self.use_org_annot_modes = ['normal','PC','OD','CD']
            self.use_vibe_annot_modes = ['vibe','NC','OC']

            self.camera_annots_path = os.path.join(self.data_folder,'camera_annots.npz')
            if not os.path.exists(self.camera_annots_path):
                pack_camera_parameters(self.data3d_dir, self.camera_annots_path)
            camera_info = np.load(self.camera_annots_path, allow_pickle=True)
            self.camera_intrinsics = camera_info['intrinsics'][()] 
            self.camera_extrinsics = camera_info['extrinsics'][()]

            logging.info('Start loading 3DPW data.')
            if mode in self.use_org_annot_modes:
                logging.info('Loading 3DPW in {} mode, split {}'.format(self.mode,self.split))
                self.joint_mapper = constants.joint_mapping(constants.COCO_18,constants.SMPL_ALL_44)
                self.joint3d_mapper = constants.joint_mapping(constants.SMPL_24,constants.SMPL_ALL_44)
                self.annots_path = os.path.join(self.data_folder,'annots.npz')
                if not os.path.exists(self.annots_path):
                    pack_data(self.data3d_dir, self.annots_path)
                self.load_annots()
            elif mode in self.use_vibe_annot_modes:
                logging.info('Loading 3DPW in VIBE mode, split {}'.format(self.split))
                self.annots_path = os.path.join(config.data_dir,'data/vibe_db')
                self.joint_mapper = constants.joint_mapping(constants.LSP_14,constants.SMPL_ALL_44)
                self.joint3d_mapper = constants.joint_mapping(constants.LSP_14,constants.SMPL_ALL_44)
                #self.regress_smpl = False
                self.load_vibe_annots()
            else:
                logging.info('3DPW loading mode is not recognized, please use the normal / vibe mode')
                raise NotImplementedError

            if mode in self.use_vibe_annot_modes:
                self.root_inds = [constants.SMPL_ALL_44['R_Hip'], constants.SMPL_ALL_44['L_Hip']]
            elif mode in self.use_org_annot_modes:
                self.root_inds = [constants.SMPL_ALL_44['Pelvis_SMPL']]

            if self.regress_smpl:
                logging.info('loading SMPL regressor for mesh vertex calculation.')
                self.smplr = SMPLR(use_gender=True)
            
            if base_class == 'video_relative':
                self.video_clip_ids = self.prepare_video_clips()
            smpl_neutral_betas_path = os.path.join(self.data_folder, 'smpl_neutral_betas.npz')
            self.smpl_neutral_betas = np.load(smpl_neutral_betas_path, allow_pickle=True)['neutral_betas'][()]
            logging.info('3DPW dataset {} split total {} samples, loading mode {}, containing {} video sequence.'.format(\
                self.split ,self.__len__(), self.mode, len(self.sequence_ids)))

        def load_PC_annots(self):
            annots = np.load(self.annots_path,allow_pickle=True)
            params = annots['params'][()]
            kp3ds = annots['kp3d'][()]
            kp2ds = annots['kp2d'][()]
            self.annots = {}
            video_names = list(params.keys())
            for video_name in video_names:
                for person_id in range(len(kp3ds[video_name])):
                    frame_range = PW3D_PCsubset[video_name.strip('.pkl')]
                    for frame_id in range(frame_range[0],frame_range[1]):
                        name = '{}_{}'.format(video_name.strip('.pkl'),frame_id)
                        kp3d = kp3ds[video_name][person_id][frame_id]
                        kp2d = kp2ds[video_name][person_id][frame_id]
                        pose_param = params[video_name]['poses'][person_id][frame_id]
                        beta_param = params[video_name]['betas'][person_id]
                        if name not in self.annots:
                            self.annots[name] = []
                        self.annots[name].append([video_name.strip('.pkl'), person_id, frame_id, kp2d.T, kp3d, pose_param, beta_param])
            self.file_paths = list(self.annots.keys())

        def reset_dataset_length_to_target_person_number(self):
            single_person_file_paths = []
            for name in self.file_paths:
                for person_id, annot in enumerate(self.annots[name]):
                    single_person_key = '{}-{}'.format(name, person_id)
                    single_person_file_paths.append(single_person_key)
                    self.annots[single_person_key]=[annot]
                #del self.annots[name]
            self.file_paths = single_person_file_paths

        def get_image_info(self, index):
            annots = self.annots[self.file_paths[index%len(self.file_paths)]]
            subject_ids, genders, kp2ds, kp3ds, params, bbox, valid_mask_2d, valid_mask_3d = [[] for i in range(8)]
            # the GT root translation and camera motion are wrong!!!
            video_name = annots[0][0]
            detecting_all_people = video_name not in invalida_detection_seqs
            
            for inds, annot in enumerate(annots):
                vmask2D = [True,True,detecting_all_people]
                vmask3D = [True,True,True,True,True,True] 
                video_name, gender, seq_id, subject_id, frame_id, kp2d, kp3d, pose_param, beta_param, tran = annot
                subject_ids.append(subject_id)
                genders.append(gender)
                kp3d = self.map_kps(kp3d, self.joint3d_mapper)
                kp3ds.append(kp3d)

                theta = pose_param[:66]
                beta = self.smpl_neutral_betas[video_name+'.pkl'][subject_id-subject_ids[0]]

                if video_name == 'courtyard_dancing_01' and frame_id>90 and frame_id<135:
                    # 这部分帧的pose有问题，胸部异常隆起。
                    #TODO: 修复这部分。
                    vmask3D[2] = False
                    vmask3D[3] = False

                params.append(np.concatenate([theta, beta])) #beta_param[:10]
                kp2d[:,2] = kp2d[:,0]>0
                kp2d[kp2d == 0]= -2
                kp2d_gt = self.map_kps(kp2d, self.joint_mapper)
                kp2ds.append(kp2d_gt)
                valid_mask_2d.append(vmask2D)
                valid_mask_3d.append(vmask3D)

            kp2ds, kp3ds, params = np.array(kp2ds), np.array(kp3ds), np.array(params)
            #show_keypoints([kp3ds[0]], [constants.All44_connMat], ['blue'])

            valid_mask_2d, valid_mask_3d = np.array(valid_mask_2d), np.array(valid_mask_3d)
            verts = None
            if self.load_vertices or self.mode in self.use_org_annot_modes:
                if self.regress_smpl:
                    kp3ds = []
                    verts = []
                    poses, betas = np.concatenate([params[:,:-10], np.zeros((len(params),6))], 1),params[:,-10:]
                    for pose, beta, gender in zip(poses, betas, genders):
                        gender = 'n' if gender is None else gender
                        #verts.append(self.smplr(pose, beta, gender)[0])
                        smpl_outs = self.smplr(pose, beta, gender)
                        kp3ds.append(smpl_outs[1])
                        verts.append(smpl_outs[0])
                    kp3ds = np.concatenate(kp3ds, 0)
                    verts = np.concatenate(verts, 0)
            
            imgpath = os.path.join(self.image_dir,video_name,'image_{:05}.jpg'.format(frame_id))
            image = cv2.imread(imgpath)[:,:,::-1].copy()        
            valid_masks = np.array([self._check_kp3d_visible_parts_(kp3d) for kp3d in kp3ds])
            kp3ds[~valid_masks] = -2.

            camMats = self.camera_intrinsics[video_name]
            camPoses = self.camera_extrinsics[video_name][frame_id]
            # the GT root translation and camera motion are wrong!!!
            #global_trans = kp3ds[:,self.root_inds].mean(1) 
            is_static_camera = False # all videos are captured with a dynamic camera.

            #show_keypoints([kp3ds[0,:24+11+5]], [constants.All44_connMat[:21+10+4]], ['blue'])
            # face_foot11 +11, +10 | extra9 +9, +15

            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': subject_ids,\
                    'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d,\
                    'kp3ds': kp3ds, 'params': params, 'verts': verts,\
                    #'root_trans': root_trans, #'global_trans':global_trans,\
                    'camMats': camMats, 'is_static_cam': is_static_camera, #'camPoses': camPoses, 
                    'img_size': image.shape[:2],'ds': self.dataset_name}
            
            if base_class == 'video_relative':
                end_frame_flag = frame_id == (len(self.sequence_ids[seq_id])-1)
                img_info.update({'seq_info':[seq_id, frame_id, end_frame_flag]})
            return img_info

        def load_vibe_annots(self):
            set_names = {'all':['train','val','test'],'train':['train'],'test':['test'],'val':['val']}
            self.split_used = set_names[self.split]
            self.annots = {}
            self.file_paths, self.sequence_ids, self.sid_video_name = [], [], {}
            self.seq_person_ids = {}
            subject_id, seq_id = 0, 0
            for split in self.split_used:
                db_file = os.path.join(self.annots_path,'3dpw_{}_db.pt'.format(split))
                db = joblib.load(db_file)
                vid_names = db['vid_name']
                frame_ids = db['frame_id']
                kp2ds, kp3ds, pose_params, beta_params, valids = db['joints2D'], db['joints3D'], db['pose'], db['shape'], db['valid']
                if split=='train':
                    kp3ds = kp3ds[:,25:39]
                for vid_name, frame_id, kp2d, kp3d, pose_param, beta_param, valid in zip(vid_names, frame_ids, kp2ds, kp3ds, pose_params, beta_params, valids):
                    video_name, person_id = vid_name[:-2], int(vid_name[-1])
                    if (self.eval_hard_seq and video_name not in constants.pw3d_hard_sequences) or  valid!=1:
                        continue
                    
                    name = '{}_{}'.format(video_name,frame_id)
                    if video_name not in self.sid_video_name.values():
                        self.sid_video_name[seq_id] = video_name
                        self.sequence_ids.append([])
                        self.seq_person_ids[video_name] = {}
                        seq_id += 1
                    if person_id not in self.seq_person_ids[video_name]:
                        self.seq_person_ids[video_name][person_id] = subject_id
                        subject_id += 1
                    if name not in self.annots:
                        self.annots[name] = []
                    
                    self.annots[name].append([video_name, None, int(seq_id-1), self.seq_person_ids[video_name][person_id], frame_id, kp2d, kp3d, pose_param, beta_param, None])
            
            if self.mode == 'NC':
                logging.info('Convert to NC subset...')
                annots = {}
                for key, annot in self.annots.items():
                    frame_id = key.split('_')[-1]
                    video_name = key.replace('_'+frame_id,'')
                    if video_name[:-3] not in PW3D_OCsubset:
                        if video_name not in PW3D_PCsubset:
                            annots[key] = annot
                self.annots = annots

            if self.mode == 'OC':
                logging.info('Convert to OC subset...')
                video_used = []
                annots = {}
                for key, annot in self.annots.items():
                    frame_id = key.split('_')[-1]
                    video_name = key.replace('_'+frame_id,'')
                    if video_name[:-3] in PW3D_OCsubset:
                        if video_name not in video_used:
                            video_used.append(video_name)
                        annots[key] = annot
                self.annots = annots
            
            for frame_name in self.annots:
                self.file_paths.append(frame_name)
                self.sequence_ids[self.annots[frame_name][0][2]].append(len(self.file_paths)-1)
            #self.sequence_first_ids = [sids[0] for sids in self.sequence_ids]
            self.ID_num = subject_id

        def load_annots(self):
            set_names = {'train':['train'],'all':['train','validation','test'],'val':['validation'],'test':['test']}
            split_used = set_names[self.split]
            annots = np.load(self.annots_path,allow_pickle=True)
            params = annots['params'][()]

            kp3ds = annots['kp3d'][()]
            kp2ds = annots['kp2d'][()]
            self.annots = {}
            self.file_paths, self.sequence_ids, self.sid_video_name = [], [], {}
            subject_id = 0
            video_names = list(params.keys())
            for seq_id, video_name in enumerate(video_names):
                self.sid_video_name[seq_id] = video_name
                self.sequence_ids.append([])
                valid_indices = params[video_name]['valid_indices']
                genders = params[video_name]['genders']
                for person_id, valid_index in enumerate(valid_indices):
                    for annot_id,frame_id in enumerate(valid_index):
                        split = params[video_name]['split']
                        if split not in split_used:
                            continue
                        name = '{}_{}'.format(video_name.strip('.pkl'),frame_id)
                        kp3d = kp3ds[video_name][person_id][annot_id]
                        kp2d = kp2ds[video_name][person_id][annot_id]

                        if args().smpl_model_type == 'smplx':
                            pose_param = smplx_params[video_name]['poses'][person_id][annot_id]
                            beta_param = smplx_params[video_name]['betas'][person_id]
                        elif args().smpl_model_type == 'smpl':
                            pose_param = params[video_name]['poses'][person_id][annot_id]
                            beta_param = params[video_name]['betas'][person_id]
                        gender = genders[person_id]
                        tran = params[video_name]['trans'][person_id][annot_id]
                        
                        if name not in self.annots:
                            self.annots[name] = []
                        self.annots[name].append([video_name.strip('.pkl'), gender, seq_id, subject_id, frame_id, kp2d.T, kp3d, pose_param, beta_param, tran])
                    subject_id += 1

            # bacause VIBE removed the subject occluded, so we have to use the original gt data.
            if self.mode == 'PC':
                annots = {}
                for key, annot in self.annots.items():
                    frame_id = key.split('_')[-1]
                    video_name = key.replace('_'+frame_id,'')
                    if video_name in PW3D_PCsubset:
                        frame_range = PW3D_PCsubset[video_name]
                        if frame_range[0]<=int(frame_id)<frame_range[1]:
                            annots[key] = annot
                self.annots = annots
            
            if self.mode == 'CD':
                annots = {}
                for key, annot in self.annots.items():
                    frame_id = key.split('_')[-1]
                    video_name = key.replace('_'+frame_id,'')
                    if video_name in PW3D_CDsubset:
                        annots[key] = annot
                self.annots = annots

            if self.mode == 'OD':
                annots = {}
                for key, annot in self.annots.items():
                    frame_id = key.split('_')[-1]
                    video_name = key.replace('_'+frame_id,'')
                    if video_name in PW3D_ODsubset:
                        annots[key] = annot
                self.annots = annots
            
            for frame_name in self.annots:
                self.file_paths.append(frame_name)
                self.sequence_ids[self.annots[frame_name][0][2]].append(len(self.file_paths)-1)
            #self.sequence_first_ids = [sids[0] for sids in self.sequence_ids]
            self.ID_num = subject_id
    return PW3D

def pack_data(data3d_dir, annots_path):
    """
    The function reads all the ground truth and prediction files. And concatenates

    :param paths_gt: all the paths corresponding to the ground truth - list of pkl files
    :param paths_prd: all the paths corresponding to the predictions - list of pkl files
    :return:
        jp_pred: jointPositions Prediction. Shape N x 24 x 3
        jp_gt: jointPositions ground truth. Shape: N x 24 x 3
        mats_pred: Global rotation matrices predictions. Shape N x 24 x 3 x 3
        mats_gt: Global rotation matrices ground truths. Shape N x 24 x 3 x 3
    """
    # all ground truth smpl parameters / joint positions / rotation matrices
    from evaluation.pw3d_eval.SMPL import SMPL

    all_params, all_jp_gts, all_jp2d_gts, all_glob_rot_gts = {}, {}, {}, {}
    seq = 0
    num_jps_pred = 0
    num_ors_pred = 0
    paths_gt = glob.glob(os.path.join(data3d_dir,'*/*.pkl'))

    smpl_model_genders = {'f':SMPL(center_idx=0, gender='f', model_root=os.path.join(config.model_dir,'smpl_original')),\
                        'm':SMPL(center_idx=0, gender='m', model_root=os.path.join(config.model_dir,'smpl_original'))  }

    # construct the data structures -
    for path_gt in paths_gt:
        print('Processing: ', path_gt)
        video_name = os.path.basename(path_gt)
        seq = seq + 1
        # Open pkl files
        data_gt = pickle.load(open(path_gt, 'rb'), encoding='latin1')
        split = path_gt.split('/')[-2]
        
        genders = data_gt['genders']
        all_params[video_name], all_jp_gts[video_name], all_jp2d_gts[video_name], all_glob_rot_gts[video_name] = {}, [], [], []
        all_params[video_name]['split'] = split
        all_params[video_name]['genders'] = genders
        all_params[video_name]['poses'], all_params[video_name]['trans'], all_params[video_name]['valid_indices'] = [], [], []
        all_params[video_name]['betas'] = np.array(data_gt['betas'])
        for i in range(len(genders)):
            # Get valid frames
            # Frame with no zeros in the poses2d file and where campose_valid is True
            poses2d_gt = data_gt['poses2d']
            poses2d_gt_i = poses2d_gt[i]
            camposes_valid = data_gt['campose_valid']
            camposes_valid_i = camposes_valid[i]
            valid_indices = check_valid_inds(poses2d_gt_i, camposes_valid_i)
            all_jp2d_gts[video_name].append(poses2d_gt_i[valid_indices])

            # Get the ground truth SMPL body parameters - poses, betas and translation parameters
            pose_params = np.array(data_gt['poses'])
            pose_params = pose_params[i, valid_indices, :]
            shape_params = np.array(data_gt['betas'][i])
            shape_params = np.expand_dims(shape_params, 0)
            shape_params = shape_params[:, :10]
            shape_params = np.tile(shape_params, (pose_params.shape[0], 1))
            trans_params = np.array(data_gt['trans'])
            trans_params = trans_params[i, valid_indices, :]
            all_params[video_name]['trans'].append(trans_params)
            all_params[video_name]['valid_indices'].append(valid_indices)

            # Get the GT joint and vertex positions and the global rotation matrices
            verts_gt, jp_gt, glb_rot_mats_gt = smpl_model_genders[genders[i]].update(pose_params, shape_params, trans_params)

            # Apply Camera Matrix Transformation to ground truth values
            cam_matrix = data_gt['cam_poses']
            new_cam_poses = np.transpose(cam_matrix, (0, 2, 1))
            new_cam_poses = new_cam_poses[valid_indices, :, :]

            # we don't have the joint regressor for female/male model. So we can't regress all 54 joints from the mesh of female/male model.
            jp_gt, glb_rot_mats_gt = apply_camera_transforms(jp_gt, glb_rot_mats_gt, new_cam_poses)
            root_rotation_cam_tranformed = transform_rot_representation(glb_rot_mats_gt[:,0], input_type='mat',out_type='vec')
            pose_params[:,:3] = root_rotation_cam_tranformed
            all_params[video_name]['poses'].append(pose_params)
            all_jp_gts[video_name].append(jp_gt)
            all_glob_rot_gts[video_name].append(glb_rot_mats_gt)

    np.savez(annots_path, params=all_params, kp3d=all_jp_gts, glob_rot=all_glob_rot_gts, kp2d=all_jp2d_gts)

def pack_camera_parameters(data3d_dir, annots_path):
    intrinsics, extrinsics = {}, {}
    seq = 0
    paths_gt = glob.glob(os.path.join(data3d_dir,'*/*.pkl'))
    for path_gt in paths_gt:
        print('Processing: ', path_gt)
        video_name = os.path.basename(path_gt).replace('.pkl', '')
        seq = seq + 1
        # Open pkl files
        data_gt = pickle.load(open(path_gt, 'rb'), encoding='latin1')
        split = path_gt.split('/')[-2]
        
        intrinsics[video_name] = data_gt['cam_intrinsics']
        extrinsics[video_name] = data_gt['cam_poses']

    np.savez(annots_path, intrinsics=intrinsics, extrinsics=extrinsics)

def with_ones(data):
    """
    Converts an array in 3d coordinates to 4d homogenous coordiantes
    :param data: array of shape A x B x 3
    :return return ret_arr: array of shape A x B x 4 where the extra dimension is filled with ones
    """
    ext_arr = np.ones((data.shape[0], data.shape[1], 1))
    ret_arr = np.concatenate((data, ext_arr), axis=2)
    return ret_arr

def apply_camera_transforms(joints, rotations, camera):
    """
    Applies camera transformations to joint locations and rotations matrices
    :param joints: B x 24 x 3
    :param rotations: B x 24 x 3 x 3
    :param camera: B x 4 x 4 - already transposed
    :return: joints B x 24 x 3 joints after applying camera transformations
             rotations B x 24 x 3 x 3 - rotations matrices after applying camera transformations
    """
    joints = with_ones(joints)  # B x 24 x 4
    joints = np.matmul(joints, camera)[:, :, :3]

    # multiply all rotation matrices with the camera rotation matrix
    # transpose camera coordinates back
    cam_new = np.transpose(camera[:, :3, :3], (0, 2, 1))
    cam_new = np.expand_dims(cam_new, 1)
    cam_new = np.tile(cam_new, (1, 24, 1, 1))
    # B x 24 x 3 x 3
    rotations = np.matmul(cam_new, rotations)

    return joints, rotations


def check_valid_inds(poses2d, camposes_valid):
    """
    Computes the indices where further computations are required
    :param poses2d: N x 18 x 3 array of 2d Poses
    :param camposes_valid: N x 1 array of indices where camera poses are valid
    :return: array of indices indicating frame ids in the sequence which are to be evaluated
    """

    # find all indices in the N sequences where the sum of the 18x3 array is not zero
    # N, numpy array
    poses2d_mean = np.mean(np.mean(np.abs(poses2d), axis=2), axis=1)
    poses2d_bool = poses2d_mean == 0
    poses2d_bool_inv = np.logical_not(poses2d_bool)

    # find all the indices where the camposes are valid
    camposes_valid = np.array(camposes_valid).astype('bool')

    final = np.logical_and(poses2d_bool_inv, camposes_valid)
    indices = np.array(np.where(final == True)[0])

    return indices


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    if not os.path.exists(keypoint_fn):
        return None
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    if len(data['people'])<1:
        return None
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])[:25]
        keypoints.append(body_keypoints)

    return keypoints

if __name__ == '__main__':
    #dataset= PW3D(base_class=default_mode)(train_flag=True) #, split='train', mode='vibe'
    #dataset= PW3D(base_class=default_mode)(train_flag=False, split='val', mode='vibe') #, split='train', mode='vibe'
    #dataset= PW3D(base_class=default_mode)(train_flag=True, split='all', mode='NC')
    dataset= PW3D(base_class=default_mode)(train_flag=False, split='all', mode='PC', load_vertices=True)
    Test_Funcs[default_mode](dataset,with_3d=True,with_smpl=True)
    print('Done')