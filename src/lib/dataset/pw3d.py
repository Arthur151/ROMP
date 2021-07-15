import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from dataset.image_base import *
from config import args

set_names = {'all':['train','val','test'],'test':['test'],'val':['train','val','test']}

class PW3D(Image_base):
    def __init__(self,train_flag = False, split='test', mode='vibe', regress_smpl=True, **kwargs):
        super(PW3D,self).__init__(train_flag)
        self.data_folder = args().dataset_rootdir
        self.data3d_dir = os.path.join(self.data_folder,'sequenceFiles')
        self.image_dir = os.path.join(self.data_folder,'imageFiles')
        self.mode = mode
        self.split = split

        logging.info('Loading 3DPW in {} mode, split {}'.format(mode,self.split))
        if mode == 'vibe':
            self.annots_path = args().annot_dir #os.path.join(config.project_dir,'data/vibe_db')
            self.joint_mapper = constants.joint_mapping(constants.LSP_14,constants.SMPL_ALL_54)
            self.joint3d_mapper = constants.joint_mapping(constants.LSP_14,constants.SMPL_ALL_54)
            self.load_vibe_annots()
        elif mode == 'whole':
            self.joint_mapper = constants.joint_mapping(constants.COCO_18,constants.SMPL_ALL_54)
            self.joint3d_mapper = constants.joint_mapping(constants.SMPL_24,constants.SMPL_ALL_54)
            self.annots_path = os.path.join(self.data_folder,'annots.npz')
            if not os.path.exists(self.annots_path):
                self.pack_data()
            self.load_annots()
        else:
            raise NotImplementedError

        self.root_inds = [constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']]
        
        logging.info('3DPW dataset {} split total {} samples, loading mode {}'.format(self.split ,self.__len__(), self.mode))

    def __len__(self):
        return len(self.file_paths)

    def get_image_info(self, index):
        annots = self.annots[self.file_paths[index%len(self.file_paths)]]
        subject_ids, genders, full_kp2d, kp3d_monos, params, bbox = [[] for i in range(6)]
        for inds, annot in enumerate(annots):
            video_name, gender, person_id, frame_id, kp2d, kp3d, pose_param, beta_param = annot
            subject_ids.append(person_id)
            genders.append(gender)
            kp3d = self.map_kps(kp3d, self.joint3d_mapper)
            kp3d_monos.append(kp3d)
            params.append(np.concatenate([pose_param[:66], beta_param[:10]]))
            kp2d_gt = self.map_kps(kp2d, self.joint_mapper)
            full_kp2d.append(kp2d_gt)
            
        imgpath = os.path.join(self.image_dir,video_name,'image_{:05}.jpg'.format(frame_id))
        image = cv2.imread(imgpath)[:,:,::-1].copy()
        info_2d = ('pw3d_vibe', imgpath, image, full_kp2d, None, subject_ids)
        info_3d = ('pw3d_vibe', kp3d_monos, params, None)
        return info_2d, info_3d

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


if __name__ == '__main__':
    dataset= PW3D()
    test_dataset(dataset,with_3d=True,with_smpl=True)
    print('Done')
