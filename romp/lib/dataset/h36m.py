from config import args
from collections import OrderedDict
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs
from dataset.camera_parameters import h36m_cameras_intrinsic_params

default_mode = args().image_loading_mode

def H36M(base_class=default_mode):
    class H36M(Base_Classes[base_class]):

        def __init__(self, train_flag=True, split='train',regress_smpl=True,**kwargs):
            super(H36M, self).__init__(train_flag,regress_smpl)
            self.data_folder = os.path.join(self.data_folder,'h36m/')
            self.image_folder = os.path.join(self.data_folder,'images/')
            self.annots_file = os.path.join(self.data_folder,'annots.npz')
            self.scale_range = [1.4,2.0]
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
                self.smplr = SMPLR(use_gender=True)
                self.root_inds = None
            
            self.joint_mapper = constants.joint_mapping(constants.H36M_32,constants.SMPL_ALL_54)
            if self.train_flag and self.regress_smpl:
                self.joint3d_mapper = constants.joint_mapping(constants.SMPL_ALL_54,constants.SMPL_ALL_54)
            else:
                self.joint3d_mapper = constants.joint_mapping(constants.H36M_32,constants.SMPL_ALL_54)

            self.kps_vis = (self.joint_mapper!=-1).astype(np.float32)[:,None]
            self.shuffle_mode = args().shuffle_crop_mode
            self.shuffle_ratio = args().shuffle_crop_ratio_3d
            self.test2val_sample_ratio = 10
            self.compress_length = 5

            self.subject = self.train_test_subject[self.phase]
            self.openpose_results = os.path.join(self.data_folder,"h36m_openpose_{}.npz".format(self.phase))
            self.imgs_list_file = os.path.join(self.data_folder,"h36m_{}.txt".format(self.phase))

            self.load_file_list()
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
            
            for test_file in test_list:
                frame_name = test_file.strip()
                self.file_paths.append(frame_name)

            if self.split=='val':
                self.file_paths = self.file_paths[::self.test2val_sample_ratio]

            if self.homogenize_pose_space and self.train_flag:
                cluster_results_file = os.path.join(self.data_folder, 'cluster_results_noumap_h36m_kmeans.npz')
                self.cluster_pool = self.parse_cluster_results(cluster_results_file,self.file_paths)
            
            if self.train_flag:
                self.sample_num = len(self.file_paths)//self.compress_length
            else:
                self.sample_num = len(self.file_paths)

        def get_image_info(self,index,total_frame=None):
            if self.train_flag:
                index = index*self.compress_length + random.randint(0,self.compress_length-1)
                if self.homogenize_pose_space:
                    index = self.homogenize_pose_sample(index)
            img_name = self.file_paths[index%len(self.file_paths)]
            subject_id = img_name.split('_')[0]
            track_ids = [self.track_id[subject_id]]
            
            info = self.annots[img_name].copy()

            cam_view_id = int(img_name.split('_')[2])
            camMats = self.camMat_views[cam_view_id]
            camDists = self.cam_distortions[cam_view_id]
            root_trans = info['kp3d_mono'].reshape(-1,3)[[constants.H36M_32['R_Hip'], constants.H36M_32['L_Hip']]].mean(0)[None]

            imgpath = os.path.join(self.image_folder,img_name)
            image = cv2.imread(imgpath)[:,:,::-1]
            kp2d = self.map_kps(info['kp2d'].reshape(-1,2).copy(),maps=self.joint_mapper)
            kp2ds = np.concatenate([kp2d,self.kps_vis],1)[None]
            
            smpl_randidx = 1#random.randint(0,2)
            root_rotation = np.array(info['cam'])[smpl_randidx]
            pose = np.array(info['poses'])[smpl_randidx]
            pose[:3] = root_rotation
            beta = np.array(info['betas'])
            
            verts = None
            if self.train_flag and self.regress_smpl:
                gender = ['m','f'][self.subject_gender[subject_id]]
                verts, kp3ds = self.smplr(pose, beta, gender)
            else:
                camkp3d = info['kp3d_mono'].reshape(-1,3).copy()
                camkp3d -= root_trans
                kp3ds = self.map_kps(camkp3d, maps=self.joint3d_mapper)[None]
                
            params = np.concatenate([pose, beta])[None]
            
            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': track_ids,\
                    'vmask_2d': np.array([[True,True,True]]), 'vmask_3d': np.array([[True,True,True,True,True,True]]),\
                    'kp3ds': kp3ds, 'params': params, 'root_trans': root_trans, 'verts': verts,\
                    'camMats': camMats, 'camDists': camDists, 'img_size': image.shape[:2], 'ds': 'h36m'}
            if 'relative' in base_class:
                img_info['depth'] = np.array([[0, self.subject_gender[subject_id], 0, 0]])
                img_info['kid_shape_offsets'] = np.array([0])
            
            return img_info

        def __len__(self):
            if self.train_flag:
                return len(self.file_paths)//self.compress_length
            else:
                return len(self.file_paths)
    return H36M

if __name__ == '__main__':
    h36m = H36M(base_class=default_mode)(True,regress_smpl=True)
    #h36m = H36M(True,regress_smpl=True)
    Test_Funcs[default_mode](h36m, with_3d=True,with_smpl=True,)