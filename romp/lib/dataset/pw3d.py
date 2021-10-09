import sys, os

from dataset.image_base import *

set_names = {'all':['train','val','test'],'test':['test'],'val':['train','val','test']}
PW3D_PCsubset = {'courtyard_basketball_00':[200,280], 'courtyard_captureSelfies_00':[500,600],\
                'courtyard_dancing_00':[60,370],  'courtyard_dancing_01':[60,270], 'courtyard_hug_00':[100,500], 'downtown_bus_00':[1620,1900]}

PW3D_OCsubset = ['courtyard_backpack','courtyard_basketball','courtyard_bodyScannerMotions','courtyard_box','courtyard_golf','courtyard_jacket',\
'courtyard_laceShoe','downtown_stairs','flat_guitar','flat_packBags','outdoors_climbing','outdoors_crosscountry','outdoors_fencing','outdoors_freestyle',\
'outdoors_golf','outdoors_parcours','outdoors_slalom']
PW3D_NOsubset = {}

class PW3D(Image_base):
    def __init__(self,train_flag = False, split='train', mode='vibe', regress_smpl=True, **kwargs):
        #if train_flag:
        #    mode, split, regress_smpl = ['normal', 'train', True]
        super(PW3D,self).__init__(train_flag,regress_smpl=regress_smpl)
        self.data_folder = os.path.join(self.data_folder,'3DPW/')
        self.data3d_dir = os.path.join(self.data_folder,'sequenceFiles')
        self.image_dir = os.path.join(self.data_folder,'imageFiles')
        self.mode = mode
        self.split = split
        self.regress_smpl = regress_smpl
        
        self.val_sample_ratio = 5
        self.scale_range = [1.56,1.8]
        self.dataset_name = {'PC':'pw3d_pc', 'NC':'pw3d_nc','OC':'pw3d_oc','vibe':'pw3d_vibe', 'normal':'pw3d_normal'}[mode]

        logging.info('Start loading 3DPW data.')
        if mode in ['normal','PC']:
            logging.info('Loading 3DPW in {} mode, split {}'.format(self.mode,self.split))
            self.joint_mapper = constants.joint_mapping(constants.COCO_18,constants.SMPL_ALL_54)
            self.joint3d_mapper = constants.joint_mapping(constants.SMPL_24,constants.SMPL_ALL_54)
            self.annots_path = os.path.join(self.data_folder,'annots.npz')
            if not os.path.exists(self.annots_path):
                self.pack_data()
            self.load_annots()
        elif mode in ['vibe','NC','OC']:
            logging.info('Loading 3DPW in VIBE mode, split {}'.format(self.split))
            self.annots_path = os.path.join(self.data_folder,'vibe_db')
            self.joint_mapper = constants.joint_mapping(constants.LSP_14,constants.SMPL_ALL_54)
            self.joint3d_mapper = constants.joint_mapping(constants.LSP_14,constants.SMPL_ALL_54)
            self.regress_smpl = False
            self.load_vibe_annots()
        else:
            logging.info('3DPW loading mode is not recognized, please use the normal / vibe mode')
            raise NotImplementedError

        if self.split=='val':
            self.file_paths = self.file_paths[::self.val_sample_ratio]

        if mode in ['vibe','NC','OC']:
            self.root_inds = [constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']]
        elif mode in ['PC', 'normal']:
            self.root_inds = [constants.SMPL_ALL_54['Pelvis_SMPL']]

        if self.regress_smpl:
            self.smplr = SMPLR(use_gender=True)
        
        logging.info('3DPW dataset {} split total {} samples, loading mode {}'.format(self.split ,self.__len__(), self.mode))

    def __len__(self):
        return len(self.file_paths)

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
        for inds, annot in enumerate(annots):
            video_name, gender, person_id, frame_id, kp2d, kp3d, pose_param, beta_param = annot
            subject_ids.append(person_id)
            genders.append(gender)
            if not self.regress_smpl:
                kp3d = self.map_kps(kp3d, self.joint3d_mapper)
                kp3ds.append(kp3d)
            params.append(np.concatenate([pose_param[:66], beta_param[:10]]))
            kp2d_gt = self.map_kps(kp2d, self.joint_mapper)
            kp2ds.append(kp2d_gt)
            valid_mask_2d.append([True,False,False])
            valid_mask_3d.append([True,True,True,True])

        kp2ds, kp3ds, params = np.array(kp2ds), np.array(kp3ds), np.array(params)
        valid_mask_2d, valid_mask_3d = np.array(valid_mask_2d), np.array(valid_mask_3d)
        if self.regress_smpl:
            kp3ds = []
            poses, betas = np.concatenate([params[:,:-10], np.zeros((len(params),6))], 1),params[:,-10:]
            for pose, beta, gender in zip(poses, betas, genders):
                smpl_outs = self.smplr(pose, beta, gender)
                kp3ds.append(smpl_outs['j3d'].numpy())
            kp3ds = np.concatenate(kp3ds, 0)
            
        imgpath = os.path.join(self.image_dir,video_name,'image_{:05}.jpg'.format(frame_id))
        image = cv2.imread(imgpath)[:,:,::-1].copy()

        root_trans = kp3ds[:,self.root_inds].mean(1)
        valid_masks = np.array([self._check_kp3d_visible_parts_(kp3d) for kp3d in kp3ds])
        kp3ds -= root_trans[:,None]
        kp3ds[~valid_masks] = -2.

        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': subject_ids,\
                'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d,\
                'kp3ds': kp3ds, 'params': params, 'img_size': image.shape[:2],'ds': self.dataset_name}
        return img_info

    def load_vibe_annots(self):
        set_names = {'all':['train','val','test'],'train':['train'],'test':['test'],'val':['val']}
        self.split_used = set_names[self.split]
        self.annots = {}
        for split in self.split_used:
            db_file = os.path.join(self.annots_path,'3dpw_{}_db.pt'.format(split))
            db = joblib.load(db_file)
            vid_names = db['vid_name']
            frame_ids = db['frame_id']
            kp2ds, kp3ds, pose_params, beta_params, valids = db['joints2D'], db['joints3D'], db['pose'], db['shape'], db['valid']
            if split=='train':
                kp3ds = kp3ds[:,25:39]
            for vid_name, frame_id, kp2d, kp3d, pose_param, beta_param, valid in zip(vid_names, frame_ids, kp2ds, kp3ds, pose_params, beta_params, valids):
                if valid!=1:
                    continue
                video_name, person_id = vid_name[:-2], int(vid_name[-1])
                name = '{}_{}'.format(video_name,frame_id)

                if name not in self.annots:
                    self.annots[name] = []
                self.annots[name].append([video_name, None, person_id, frame_id, kp2d, kp3d, pose_param, beta_param])
        self.file_paths = list(self.annots.keys())
        
        if self.mode == 'NC':
            logging.info('Convert to NC subset...')
            file_paths = []
            annots = {}
            for key, annot in self.annots.items():
                frame_id = key.split('_')[-1]
                video_name = key.replace('_'+frame_id,'')
                if video_name[:-3] not in PW3D_OCsubset:
                    if video_name not in PW3D_PCsubset:
                        file_paths.append(key)
                        annots[key] = annot
            self.file_paths = file_paths
            self.annots = annots

        if self.mode == 'OC':
            logging.info('Convert to OC subset...')
            video_used = []
            file_paths = []
            annots = {}
            for key, annot in self.annots.items():
                frame_id = key.split('_')[-1]
                video_name = key.replace('_'+frame_id,'')
                if video_name[:-3] in PW3D_OCsubset:
                    if video_name not in video_used:
                        video_used.append(video_name)
                    file_paths.append(key)
                    annots[key] = annot
            self.file_paths = file_paths
            self.annots = annots


    def load_annots(self):
        set_names = {'train':['train'],'all':['train','validation','test'],'val':['validation'],'test':['test']}
        split_used = set_names[self.split]
        annots = np.load(self.annots_path,allow_pickle=True)
        params = annots['params'][()]
        kp3ds = annots['kp3d'][()]
        kp2ds = annots['kp2d'][()]
        self.annots = {}
        video_names = list(params.keys())
        for video_name in video_names:
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
                    pose_param = params[video_name]['poses'][person_id][annot_id]
                    beta_param = params[video_name]['betas'][person_id]
                    gender = genders[person_id]
                    
                    if name not in self.annots:
                        self.annots[name] = []
                    self.annots[name].append([video_name.strip('.pkl'), gender, person_id, frame_id, kp2d.T, kp3d, pose_param, beta_param])
        self.file_paths = list(self.annots.keys())

        # bacause VIBE removed the subject occluded, so we have to use the original gt data.
        if self.mode == 'PC':
            file_paths = []
            annots = {}
            for key, annot in self.annots.items():
                frame_id = key.split('_')[-1]
                video_name = key.replace('_'+frame_id,'')
                if video_name in PW3D_PCsubset:
                    frame_range = PW3D_PCsubset[video_name]
                    if frame_range[0]<=int(frame_id)<frame_range[1]:
                        file_paths.append(key)
                        annots[key] = annot
            self.file_paths = file_paths
            self.annots = annots


    def pack_data(self):
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
        paths_gt = glob.glob(os.path.join(self.data3d_dir,'*/*.pkl'))

        smpl_model_genders = {'f':SMPL(center_idx=0, gender='f', model_root=args().smpl_model_path),\
                              'm':SMPL(center_idx=0, gender='m', model_root=args().smpl_model_path)  }

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

        np.savez(self.annots_path, params=all_params, kp3d=all_jp_gts, glob_rot=all_glob_rot_gts, kp2d=all_jp2d_gts)


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
        '''
        left_hand_keyp = np.array(
            person_data['hand_left_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])
        right_hand_keyp = np.array(
            person_data['hand_right_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])
        hand_kp2d = np.concatenate([left_hand_keyp, right_hand_keyp],0)
        # TODO: Make parameters, 17 is the offset for the eye brows,
        # etc. 51 is the total number of FLAME compatible landmarks
        face_keypoints = np.array(
            person_data['face_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

        contour_keyps = np.array(
            [], dtype=body_keypoints.dtype).reshape(0, 3)
        if use_face_contour:
            contour_keyps = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[:17, :]
       
        keypoints.append([body_keypoints, hand_kp2d, face_keypoints])
        '''

    return keypoints


if __name__ == '__main__':
    #dataset= PW3D(train_flag=False, split='test', mode='vibe')
    dataset= PW3D(train_flag=True)
    test_dataset(dataset,with_3d=True,with_smpl=True)
    print('Done')

'''
    if crop_eval:
                self.reset_dataset_length_to_target_person_number()
                self.multi_mode = Falsec

        self.openpose_dir = os.path.join(self.data_folder,'openpose_json')
        input_cropped_img=False, bbox=None, use_openpose_center=False
        self.input_cropped_img = input_cropped_img
        self.use_bbox = True if bbox is not None else False
        self.use_openpose_center = use_openpose_center

            if self.input_cropped_img:
                self.multi_mode = False
                self.reset_dataset_length_to_target_person_number()
                logging.info('loading 3DPW dataset using cropped image')
            if self.use_bbox:
                self.bboxes = np.load(bbox,allow_pickle=True)['bbox'][()]
                logging.info('using bbox from ', bbox)
        openpose_annot_path = self.openpose_dir.replace('_json', '_body_results.npz')
        if not os.path.exists(openpose_annot_path):
            self.pack_openpose_results(openpose_annot_path)
        self.openpose_kp2ds = np.load(openpose_annot_path,allow_pickle=True)['annots'][()]


    def get_image_info(self,index):
        if not self.input_cropped_img:
            multi_person_annots = self.annots[self.file_paths[index]]
            return self.get_complete_image_info(multi_person_annots)
        if self.input_cropped_img:
            annot_id, person_id = self.file_paths[index].split('-')
            multi_person_annots = self.annots[annot_id]
            target_person_annots = multi_person_annots[int(person_id)]
            video_name, frame_id = target_person_annots[0], target_person_annots[2]
            if video_name in self.openpose_kp2ds:
                if frame_id in self.openpose_kp2ds[video_name]:
                    self.multi_mode = False
                    return self.get_cropped_image_info(target_person_annots)
            self.multi_mode = True
            return self.get_complete_image_info(multi_person_annots)


        def get_complete_image_info(self, multi_person_annots):
            # if self.train_flag and self.train_with_openpose:
            #     video_name, frame_id = multi_person_annots[0][0], multi_person_annots[0][2]
            #     if frame_id in self.openpose_kp2ds[video_name]:
            #         full_kp2d = self.openpose_kp2ds[video_name][frame_id]
            #     else:
            #         return self.get_image_info(random.randint(0,len(self)))
            #     #full_kp2d = [self.map_kps(kp2d,maps=constants.body1352coco25) for kp2d in full_kp2d]
            #     subject_ids = np.arange(len(full_kp2d))
            #     kp3d_monos, params = None, None

            subject_ids, full_kp2d, kp3d_monos, params, bbox = [[] for i in range(5)]
            video_name, frame_id = multi_person_annots[0][0], multi_person_annots[0][2]
            #if self.use_openpose_center:
            #    full_kp2d_op = np.array(self.openpose_kp2ds[video_name][frame_id])
            #    openpose_center = np.array([self._calc_center_(kp2d) for kp2d in full_kp2d_op])
            for subject_id, annots in enumerate(multi_person_annots):
                video_name, person_id, frame_id, kp2d, kp3d, pose_param, beta_param = annots
                subject_ids.append(person_id)
                kp3d_monos.append(kp3d)
                params.append(np.concatenate([pose_param[:66], beta_param]))
                kp2d_gt = self.map_kps(kp2d, self.joint_mapper)
                #if self.use_openpose_center:
                #    kp2d_gt_center = self._calc_center_(kp2d_gt)
                #    min_dist_idx = np.argmin(np.linalg.norm(openpose_center-kp2d_gt_center[None],axis=-1))
                #    full_kp2d.append(full_kp2d_op[min_dist_idx])
                full_kp2d.append(kp2d_gt)
                
            imgpath = os.path.join(self.image_dir,video_name,'image_{:05}.jpg'.format(frame_id))
            image = cv2.imread(imgpath)[:,:,::-1].copy()
            info_2d = ('pw3d', imgpath, image, full_kp2d[np.random.randint(len(full_kp2d))], full_kp2d, None, subject_ids)
            info_3d = ('pw3d', kp3d_monos, params, None)
            return info_2d, info_3d

    def get_cropped_image_info(self, target_person_annots):
        video_name, person_id, frame_id, kp2d, kp3d, pose_param, beta_param = target_person_annots
        kp2d_op = self.openpose_kp2ds[video_name][frame_id]
        kp2d_op_matched = self.match_op_to_gt(kp2d_op,kp2d)
        full_kp2d = [kp2d]
        subject_ids = [person_id]
        kp3d_monos, params = [kp3d], [np.concatenate([pose_param[:66], beta_param])]
            
        imgpath = os.path.join(self.image_dir,video_name,'image_{:05}.jpg'.format(frame_id))
        image = cv2.imread(imgpath)[:,:,::-1].copy()
        info_2d = ('pw3d', imgpath, image, kp2d_op_matched, full_kp2d,None,subject_ids)
        info_3d = ('pw3d', kp3d_monos, params, None)
        return info_2d, info_3d

            if self.use_bbox:
                    bbox_center = self.bboxes[video_name][person_id,frame_id]
                    min_dist_idx = np.argmin(np.linalg.norm(openpose_center[:,:2]-bbox_center[None],axis=-1))
                    center = self._calc_center_(full_kp2d_op[min_dist_idx])
                    centers.append(center)
            if self.use_bbox:
                centers = np.array(centers)
    
    def pack_openpose_results(self, annot_file_path):
        self.openpose_kp2ds = {}
        for key, multi_person_annots in self.annots.items():
            video_name, frame_id = multi_person_annots[0][0], multi_person_annots[0][2]
            openpose_file_path = os.path.join(self.openpose_dir,video_name+'-'+'image_{:05}_keypoints.json'.format(frame_id))
            full_kp2d = read_keypoints(openpose_file_path)
            if full_kp2d is None:
                continue
            if video_name not in self.openpose_kp2ds:
                self.openpose_kp2ds[video_name] = {}
            self.openpose_kp2ds[video_name][frame_id] = full_kp2d
        np.savez(annot_file_path, annots=self.openpose_kp2ds)


    def match_op_to_gt(self, kp2ds_op, kp2d_gt):
        kp2ds_op_dist = {}
        vis_gt = kp2d_gt[self.torso_ids,-1]>0
        center_gt = kp2d_gt[self.torso_ids][vis_gt].mean(0)
        for idx, kp2d_op in enumerate(kp2ds_op):
            vis = kp2d_op[self.torso_ids,-1]>0
            if vis.sum()>1:
                center_point = kp2d_op[self.torso_ids][vis].mean(0)
                dist = np.linalg.norm(center_point-center_gt)
                kp2ds_op_dist[dist] = idx

        kp2d_op_matched_id = kp2ds_op_dist[np.min(list(kp2ds_op_dist.keys()))]

        return kp2ds_op[kp2d_op_matched_id]


    if 'joint_format' in kwargs:
                joint_format=kwargs['joint_format']
            else:
                joint_format='coco25'
            print('joint_format',joint_format)

    #for set_name in set_names[self.phase]:
        #    label_dir = os.path.join(self.data3d_dir,set_name)
        #    self.get_labels(label_dir)

    def get_image_info(self,index):
        annot_3d = self.labels[index]
        imgpath = os.path.join(self.image_dir,annot_3d['name'],'image_{:05}.jpg'.format(annot_3d['ids']))
        subject_ids = annot_3d['subject_ids'].tolist()
        person_num = len(subject_ids)
        #name = os.path.join(self.image_dir,annot_3d['name'],'image_{:05}_{}.jpg'.format(annot_3d['ids'],subject_id))
        image = cv2.imread(imgpath)[:,:,::-1].copy()
        
        #openpose_file_path = os.path.join(self.openpose_dir,annot_3d['name']+'-'+'image_{:05}_keypoints.json'.format(annot_3d['ids']))
        #openpose_result_list = read_keypoints(openpose_file_path)
        #kp2d_body = self.process_openpose(openpose_result_list, kps)

        full_kps = annot_3d['kp2d'].copy()
        thetas,betas,ts,genders = annot_3d['poses'].copy(),annot_3d['betas'].copy(),annot_3d['t'].copy(),annot_3d['gender'].copy()
        full_kp2d,kp3d_monos = [],[]
        for idx in range(person_num):
            joint = self.map_kps(full_kps[idx].T)
            if (joint[:,-1]>-1).sum()<1:
                subject_ids.remove(idx)
                continue

            full_kp2d.append(joint)
            kp3d = self.smplr(thetas[idx], betas[idx], genders[idx])[0]
            kp3d_monos.append(kp3d)
        #kp3d_mono = annot_3d['kp3d'].copy().reshape(24,3)
        #kp3d_mono[:,1:] *= -1
        #kp3d_mono = self.map_kps(kp3d_mono,maps=config.smpl24_2_coco25)
        params = np.concatenate([np.array(thetas)[:,:66], np.array(betas)[:,-10:]],-1)

        info_2d = ('pw3d', imgpath, image, full_kp2d[np.random.randint(len(full_kp2d))], full_kp2d,None,None,subject_ids)
        info_3d = ('pw3d', kp3d_monos, params, None)
        return info_2d, info_3d

    def get_labels(self,label_dir):
        label_paths = glob.glob(label_dir+'/*.pkl')
        for label_path in label_paths:
            raw_labels = self.read_pkl(label_path)

            frame_num = len(raw_labels['img_frame_ids'])
            for j in range(frame_num):
                label = {}
                label['name'] = raw_labels['sequence']
                label['ids'] = j#raw_labels['img_frame_ids'][j]\
                #img_frame_ids: an index-array to down-sample 60 Hz 3D poses to corresponding image frame ids
                label['frame_ids'] = raw_labels['img_frame_ids'][j]
                label['subject_ids'] = np.arange(len(raw_labels['poses']))
                label['kp2d'] = np.array([raw_labels['poses2d'][idx][j] for idx in range(len(raw_labels['poses2d']))])
                if (label['kp2d'][:,:,-1]>-1).sum()<1:
                    continue

                extrinsics = raw_labels['cam_poses'][j,:3,:3]
                poses,shapes,trans = [[] for idx in range(3)]
                for idx in range(len(raw_labels['poses'])):
                    trans.append(raw_labels['trans'][idx][j])
                    shapes.append(raw_labels['betas'][idx][:10])
                    pose=raw_labels['poses'][idx][j]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]
                    poses.append(pose)
                label['poses'],label['betas'],label['t'] = poses,shapes,trans
                
                label['kp3d'] = [raw_labels['jointPositions'][idx][j] for idx in range(len(raw_labels['jointPositions']))]
                label['gender'] = [raw_labels['genders'][idx] for idx in range(len(raw_labels['genders']))]
                #label['cam_poses'] = raw_labels['cam_poses'][i]#Rt矩阵
                label['cam_trans'] = raw_labels['cam_poses'][j,:3,3]
                label['cam_rotation_matrix'] = raw_labels['cam_poses'][j,:3,:3]#Rt矩阵
                #label['campose_valid_mask'] = raw_labels['campose_valid'][i][j]
                self.labels.append(label)

        return True

    def process_openpose(self,result_list, kps_gt):
        if result_list is not None:
            if len(result_list)>1:
                for body_kp2d_op, hand_kp2d_op, face_kp2d_op in result_list:
                    body_kp2d_op = body_kp2d_op[config.body1352coco25]
                    if body_kp2d_op[9,2]>0.05 and body_kp2d_op[12,2]>0.05:
                        body_kp2d_op[8] = (body_kp2d_op[9]+body_kp2d_op[12])/2
                    else:
                        body_kp2d_op[8,2] = -2
                    vis_id = ((body_kp2d_op[:,2]>0.04).astype(np.float32) + (kps_gt[:,2]>0.04).astype(np.float32))>1
                    if vis_id.sum()>4:
                        error = np.linalg.norm((body_kp2d_op[vis_id,:2]-kps_gt[vis_id,:2]), axis=-1).mean()
                    else:
                        error = 1000
                    if error<70:
                        return body_kp2d_op

        return kps_gt

    



    def load_file_list(self):
        self.file_paths = []
        self.annots = np.load(self.annots_file, allow_pickle=True)['annots'][()]

        with open(self.imgs_list_file) as f:
            test_list = f.readlines()
        for test_file in test_list:
            self.file_paths.append(test_file.strip())

        self.kps_op, self.facial_kps2d, self.hand_kps2d = {},{},{}
        with open(self.kps_openpose_json_file,'r') as f:
            openpose_labels = json.load(f)
        empty_count=0
        for idx,img_name in enumerate(self.file_paths):
            img_name = os.path.basename(img_name)
            annot = openpose_labels[img_name]
            if annot is None:
                empty_count += 1
                continue
            kp2d = np.array(annot['pose_keypoints_2d']).reshape(-1,3)
            self.kps_op[img_name] = kp2d.astype(np.float32)
            face_kp2d = np.array(annot['face_keypoints_2d']).reshape(-1,3)[17:68]
            self.facial_kps2d[img_name] = face_kp2d.astype(np.float32)
            hand_kp2d = np.concatenate([np.array(annot['hand_left_keypoints_2d']).reshape(-1,3),\
                np.array(annot['hand_right_keypoints_2d']).reshape(-1,3)],0)
            self.hand_kps2d[img_name] = hand_kp2d.astype(np.float32)

        print('empty_count_op:',empty_count)

    def load_alphapose_mpii(self):
        with open(self.kps_alpha_json_file,'r') as f:
            raw_labels = json.load(f)
        error_num = 0
        for idx,annot_3d in enumerate(self.labels):
            content = raw_labels['{}-image_{:05}.jpg'.format(annot_3d['name'],annot_3d['ids'])]
            poses = []
            for pid in range(len(content)):
                poses.append(np.array(content[pid]['keypoints']).reshape(-1,3)[:,:3])
            poses = np.array(poses)[:,self.mpii_2_lsp14]
            kps_gt = annot_3d['kp2d'].copy().T[self.coco18_2_lsp14][:-2]
            vis = np.where(kps_gt[:,2]>0)[0]
            poses_comp = poses[:,vis,:2]
            kps_gt = kps_gt[vis,:2][None,:,:]
            mis_errors = np.mean(np.linalg.norm(poses_comp-kps_gt,ord=2,axis=-1),-1)
            pose = poses[np.argmin(mis_errors)]

            pose[pose[:,2]<0.01,2] = 0
            pose[pose[:,2]>0.01,2] = 1
            annot_3d['kps_alpha'] = pose

    def load_alphapose_coco(self):
        with open(self.kps_alpha_json_file,'r') as f:
            raw_labels = json.load(f)
        frame_num = len(raw_labels)
        print('frame_num',frame_num)
        error_count=0
        for idx,annot_3d in enumerate(self.labels):
            try:
                content = raw_labels['{}-image_{:05}.jpg'.format(annot_3d['name'],annot_3d['ids'])]['bodies']
                poses = []
                for pid in range(len(content)):
                    poses.append(np.array(content[pid]['joints']).reshape(-1,3))
                poses = np.array(poses)[:,self.coco18_2_lsp14]

                poses[:,-1,2] = 0

                kps_gt = annot_3d['kp2d'].copy().T[self.coco18_2_lsp14][:-2]
                vis = np.where(kps_gt[:,2]>0)[0]
                mis_errors = []
                for i in range(len(poses)):
                    poses_comp = poses[i,vis]
                    vis_pred = poses_comp[:,2]>0
                    poses_comp = poses_comp[vis_pred,:2]
                    kps_gti = kps_gt[vis,:2][vis_pred,:]
                    mis_errors.append(np.mean(np.linalg.norm(poses_comp-kps_gti,ord=2,axis=-1)))
                mis_errors = np.array(mis_errors)
                pose = poses[np.argmin(mis_errors)]

                pose[pose[:,2]<0.1,2] = 0
                pose[pose[:,2]>0.1,2] = 1
                annot_3d['kps_alpha'] = pose
            except :
                print('{}/image_{:05}.jpg'.format(annot_3d['name'],annot_3d['ids']))
                error_count+=1
                pose_gt = annot_3d['kp2d'].copy().T[self.coco18_2_lsp14]
                pose_gt[pose_gt[:,2]<0.1,2] = 0
                pose_gt[pose_gt[:,2]>0.1,2] = 1
                annot_3d['kps_alpha'] = pose_gt
        print('error_count',error_count)

    def get_item_video(self,index):
        label = self.labels[index]
        label_dict_name = '{}_{}'.format(label['name'],label['subject_ids'])
        ids_sequence = list(self.label_dict[label_dict_name].keys())
        current_frame = label['ids']
        current_spawn = int((self.spawn-1)/2)
        features_idx = []
        for index, num in enumerate(list(range(current_frame, current_frame+current_spawn+1))):
            if num not in ids_sequence:
                num=features_idx[index-1]
            features_idx.append(num)
        for index, num in enumerate(list(range(current_frame-1, current_frame-current_spawn-1,-1))):
            if num not in ids_sequence:
                num=features_idx[0]
            features_idx=[num]+features_idx
        labels_idx = []
        for idx in features_idx:
            labels_idx.append(self.label_dict[label_dict_name][idx])
        video = []
        video_input = {}
        for label_idx in labels_idx:
            video.append(self.get_item_single_frame(label_idx))
        for key in video[0].keys():
            if key=='image':
                video_input[key] = torch.cat([video[i][key].unsqueeze(0) for i in range(len(video))])
            elif key=='kps_alpha':
                video_input[key] = torch.cat([video[i][key].unsqueeze(0) for i in range(len(video))])
            else:
                video_input[key] = video[current_spawn][key]
        return video_input
'''