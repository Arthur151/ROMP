from config import args
from collections import OrderedDict
from datasets.image_base import *
from datasets.base import Base_Classes, Test_Funcs
from datasets.camera_parameters import h36m_cameras_intrinsic_params

default_mode = args().video_loading_mode if args().video else args().image_loading_mode if args().video else args().image_loading_mode

def H36M(base_class=default_mode):
    class H36M(Base_Classes[base_class]):
        def __init__(self, train_flag=True, split='train', regress_smpl=False, load_entire_sequence=False, **kwargs):
            super(H36M, self).__init__(train_flag, regress_smpl, syn_obj_occlusion=False, load_entire_sequence=load_entire_sequence)
            self.data_folder = os.path.join(self.data_folder,'h36m/').replace('DataCenter', 'DataCenter2')
            self.image_folder = os.path.join(self.data_folder,'images/')
            self.annots_file = os.path.join(self.data_folder,'annots_smplkps.npz') # annots.npz
            self.scale_range = [1.4, 3.0]
            self.train_flag=train_flag
            self.split = split
            self.train_test_subject = {'train':['S1','S5','S6','S7','S8'],'test':['S9','S11']}
            self.track_id = {'S1':1,'S5':2,'S6':3,'S7':4,'S8':5, 'S9':6,'S11':7}
            self.camMat_views = [np.array([[intrinsic['focal_length'][0], 0, intrinsic['center'][0]],\
                                        [0, intrinsic['focal_length'][1], intrinsic['center'][1]],\
                                        [0,0,1]]) \
                                    for intrinsic in h36m_cameras_intrinsic_params]
            # http://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=calibrate
            # k1, k2, p1, p2, k3, k4,k5,k6
            self.cam_distortions = [np.array([*intrinsic['radial_distortion'][:2], *intrinsic['tangential_distortion'], intrinsic['radial_distortion'][2]])
                                    for intrinsic in h36m_cameras_intrinsic_params]

            if self.regress_smpl:
                self.use_gender = False
                self.smplr = SMPLR(use_gender=self.use_gender)
                self.root_inds = None
            
            self.joint_mapper = constants.joint_mapping(constants.H36M_32,constants.SMPL_ALL_44)
            if self.train_flag and self.regress_smpl:
                self.joint3d_mapper = constants.joint_mapping(constants.SMPL_ALL_44,constants.SMPL_ALL_44)
            else:
                self.joint3d_mapper = constants.joint_mapping(constants.H36M_32,constants.SMPL_ALL_44)

            self.kps_vis = (self.joint_mapper!=-1).astype(np.float32)[:,None]
            self.shuffle_mode = args().shuffle_crop_mode
            self.aug_cropping_ratio = args().shuffle_crop_ratio_3d
            self.test2val_sample_ratio = 10
            self.compress_length = 5

            self.subject = self.train_test_subject[self.phase]
            self.openpose_results = os.path.join(self.data_folder,"h36m_openpose_{}.npz".format(self.phase))
            self.imgs_list_file = os.path.join(self.data_folder,"h36m_{}.txt".format(self.phase))

            if base_class == 'video_relative':
                self.load_video_list()
            else:
                self.load_file_list()
            smpl_neutral_betas_path = os.path.join(self.data_folder, 'smpl_neutral_betas.npz')
            self.smpl_neutral_betas = np.load(smpl_neutral_betas_path, allow_pickle=True)['neutral_betas'][()]
            self.subject_gender = {'S1':1, 'S5':1, 'S6':0, 'S7':1, 'S8':0, 'S9':0, 'S11':0}
            if not self.train_flag:
                self.multi_mode=False
                self.scale_range=[1.8,1.8]

            logging.info('Loaded Human3.6M data,total {} samples'.format(self.__len__()))

        def load_file_list(self):
            self.file_paths = []
            self.annots = np.load(self.annots_file, allow_pickle=True)['annots'][()]

            with open(self.imgs_list_file) as f:
                test_list = f.readlines()
            
            self.sequence_dict, self.sequence_ids, self.sid_video_name = [], [], {}
            for test_file in test_list:
                frame_name = test_file.strip()
                self.file_paths.append(frame_name)

            if self.split=='val':
                self.file_paths = self.file_paths[::self.test2val_sample_ratio]

            if self.homogenize_pose_space and self.train_flag:
                cluster_results_file = os.path.join(config.data_dir, 'data/pose_space_optimization', 'data', 'cluster_results_noumap_h36m_kmeans.npz')
                self.cluster_pool = self.parse_cluster_results(cluster_results_file,self.file_paths)
            
            if self.train_flag:
                self.sample_num = len(self.file_paths)//self.compress_length
            else:
                self.sample_num = len(self.file_paths)
        
        def load_video_list(self):
            
            with open(self.imgs_list_file) as f:
                test_list = f.readlines()
            
            self.sequence_dict, self.sequence_ids, self.sid_video_name = {}, [], {}
            for test_file in test_list:
                frame_name = test_file.strip()
                frame_id = int(frame_name.split('_')[-1].replace('.jpg',''))
                seq_name = frame_name.replace('_{}.jpg'.format(frame_id), '')
                if seq_name not in self.sequence_dict:
                    self.sequence_dict[seq_name] = {}
                self.sequence_dict[seq_name][frame_id] = os.path.basename(frame_name)
            self.sequence_dict = OrderedDict(self.sequence_dict)
            self.file_paths, self.seq_info, self.sequence_ids, self.sid_video_name = [], [], [], {}
            for sid, video_name in enumerate(self.sequence_dict):
                self.sid_video_name[sid] = video_name
                self.sequence_ids.append([])
                frame_ids = sorted(self.sequence_dict[video_name].keys())
                for fid in frame_ids:
                    self.file_paths.append(self.sequence_dict[video_name][fid])
                    self.seq_info.append([sid,fid])
                    self.sequence_ids[sid].append(len(self.file_paths)-1)
            self.ID_num = len(self.sequence_ids)

            if self.split=='val':
                self.file_paths = self.file_paths[::self.test2val_sample_ratio]
            
            self.annots = np.load(self.annots_file, allow_pickle=True)['annots'][()]
            if base_class == 'video_relative':
                self.video_clip_ids = self.prepare_video_clips()
                self.sample_num = len(self.video_clip_ids)
            
            self.random_temp_sample_internal = 2 # the video frame is sampled by every 5 frames, setting interval as 2 is enough  

            if self.homogenize_pose_space and self.train_flag:
                cluster_results_file = os.path.join(config.data_dir, 'data/pose_space_optimization', 'data', 'cluster_results_noumap_h36m_kmeans.npz')
                self.cluster_pool = self.parse_cluster_results(cluster_results_file,self.file_paths)

        def get_image_info(self,index,total_frame=None):
            if self.train_flag and not base_class == 'video_relative':
                index = index*self.compress_length + random.randint(0,self.compress_length-1)
                if self.homogenize_pose_space:
                    index = self.homogenize_pose_sample(index)
            img_name = self.file_paths[index%len(self.file_paths)]
            subject_id = img_name.split('_')[0]
            if base_class == 'video_relative':
                sid, frame_id = self.seq_info[index%len(self.file_paths)]
                seq_name = self.sid_video_name[sid]
                end_frame_flag = frame_id == (len(self.sequence_ids[sid])-1)
                track_ids = [sid]
            else:
                track_ids = [self.track_id[subject_id]]
            
            info = self.annots[img_name].copy()

            cam_view_id = int(img_name.split('_')[2])
            camMats = self.camMat_views[cam_view_id]
            camDists = self.cam_distortions[cam_view_id]
            root_trans = info['kp3d_mono'].reshape(-1,3)[[constants.H36M_32['R_Hip'], constants.H36M_32['L_Hip']]].mean(0)[None]

            imgpath = os.path.join(self.image_folder,img_name)
            image = cv2.imread(imgpath)[:,:,::-1]#把BGR转成RGB
            kp2d = self.map_kps(info['kp2d'].reshape(-1,2).copy(),maps=self.joint_mapper)
            kp2ds = np.concatenate([kp2d,self.kps_vis],1)[None]
            
            smpl_randidx = 1#random.randint(0,2)
            root_rotation = np.array(info['cam'])[smpl_randidx]
            pose = np.array(info['poses'])[smpl_randidx]
            pose[:3] = root_rotation
            
            beta = np.array(self.smpl_neutral_betas[subject_id])
            kp3ds = np.array(info['kp3ds_smpl'])[None]      
                
            params = np.concatenate([pose, beta])[None]
            # still 0.0+ error of root trans using joints derived from smpl joints 
            #root_trans_pred = estimate_translation(np.array([kp3d_mono]), kp2ds, proj_mats=camMats[None], cam_dists=camDists[None])
            
            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': track_ids,\
                    'vmask_2d': np.array([[True,True,True]]), 'vmask_3d': np.array([[True,True,True,True,True,True]]),\
                    'kp3ds': kp3ds, 'params': params, 'is_static_cam': True, 
                    'img_size': image.shape[:2], 'ds': 'h36m'}
            #'root_trans': root_trans, 'verts': verts, 'camMats': camMats, 'camDists': camDists, 
            if 'relative' in base_class:
                img_info['depth'] = np.array([[0, self.subject_gender[subject_id], 0, 0]])
                img_info['kid_shape_offsets'] = np.array([0])
            if base_class == 'video_relative':
                img_info.update({'seq_info':[sid, frame_id, end_frame_flag]})
            
            return img_info

        def __len__(self):
            return self.sample_num
    return H36M

if __name__ == '__main__':
    h36m = H36M(base_class=default_mode)(True, regress_smpl=False)
    #h36m = H36M(base_class=default_mode)(False, regress_smpl=False, load_entire_sequence=True)
    #h36m = H36M(True,regress_smpl=True)
    Test_Funcs[default_mode](h36m, with_3d=True, with_smpl=True, vis_global_open3d=False, show_kp3ds=True)
