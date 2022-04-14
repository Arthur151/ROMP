import sys, os

from config import args
from dataset.image_base import *


class MPI_INF_3DHP(Image_base):
    def __init__(self, train_flag=True, validation=False, **kwargs):
        super(MPI_INF_3DHP,self).__init__(train_flag, regress_smpl=True)
        self.data_folder = os.path.join(self.data_folder,'mpi_inf_3dhp/')
        annots_file_path = os.path.join(self.data_folder, 'annots.npz')
        self.image_folder = os.path.join(self.data_folder, 'images')
        self.scale_range = [1.3,1.9]
        if os.path.exists(annots_file_path):
            self.annots = np.load(annots_file_path,allow_pickle=True)['annots'][()]
        else:
            self.pack_data(annots_file_path)
        if not os.path.exists(self.image_folder):
            frame_info = np.load(annots_file_path,allow_pickle=True)['frame_info'][()]
            self.extract_frames(frame_info)
        self.file_paths = list(self.annots.keys())
        if validation:
            set_name = 'validation'
            removed_subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
        else:
            set_name = 'train'
            removed_subjects = ['S8']
        self.track_id = {'S1':1,'S2':2,'S3':3,'S4':4,'S5':5,'S6':6,'S7':7, 'S8':8}
        self.subject_number = 8
        previous_sample_num = len(self.file_paths)
        self.file_paths = self.exclude_subjects(self.file_paths, subjects=removed_subjects)
        print('From file_paths with {} samples, removing subjects: {}, with {} samples left'.format(previous_sample_num, removed_subjects, len(self.file_paths)))

        self.subject_gender = {'S1':1, 'S2':0, 'S3':0, 'S4':1, 'S5':1, 'S6':1, 'S7':0, 'S8':0}
        self.kp2d_mapper = constants.joint_mapping(constants.MPI_INF_28,constants.SMPL_ALL_54)
        self.kp3d_mapper = constants.joint_mapping(constants.MPI_INF_28,constants.SMPL_ALL_54)
        self.compress_length = 3
        self.shuffle_ratio = args().shuffle_crop_ratio_3d

        if self.homogenize_pose_space:
            cluster_results_file = os.path.join(self.data_folder, 'cluster_results_noumap_mpiinf_kmeans.npz')
            self.cluster_pool = self.parse_cluster_results(cluster_results_file,self.file_paths)
        logging.info('Loaded MPI-INF-3DHP {} set,total {} samples'.format(set_name, self.__len__()))

    def exclude_subjects(self, file_paths, subjects=['S8']):
        file_path_left = []
        for inds, file_path in enumerate(file_paths):
            subject_id = os.path.basename(file_path).split('_')[0]
            if subject_id not in subjects:
                file_path_left.append(file_path)
        return file_path_left

    def __len__(self):
        if self.train_flag:
            return len(self.file_paths)//self.compress_length
        else:
            return len(self.file_paths)

    def get_image_info(self, index):
        if self.train_flag:
            index = index*self.compress_length + random.randint(0,self.compress_length-1)
        if self.homogenize_pose_space:
            index = self.homogenize_pose_sample(index)
        img_name = self.file_paths[index%len(self.file_paths)]
        subject_id = os.path.basename(img_name).split('_')[0]

        imgpath = os.path.join(self.image_folder,img_name)
        while not os.path.exists(imgpath):
            print(imgpath,'not exist!')
            img_name = self.file_paths[np.random.randint(len(self))]
            imgpath = os.path.join(self.image_folder,img_name)
        image = cv2.imread(imgpath)[:,:,::-1]
        R, T = self.annots[img_name]['extrinsics']
        fx, fy, cx, cy = self.annots[img_name]['intrinsics']
        camMats = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])
        kp2ds = self.map_kps(self.annots[img_name]['kp2d'],maps=self.kp2d_mapper)
        kp3ds = self.map_kps(self.annots[img_name]['kp3d'], maps=self.kp3d_mapper)[None]
        vis_mask = _check_visible(kp2ds,get_mask=True)
        kp2ds = np.concatenate([kp2ds, vis_mask[:,None]],1)[None]

        root_trans = kp3ds[:,self.root_inds].mean(1)
        
        kp3ds[:,vis_mask] -= root_trans[:,None]

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': [self.track_id[subject_id]],\
                'vmask_2d': np.array([[True,True,True]]), 'vmask_3d': np.array([[True,False,False,False]]),\
                'kp3ds': kp3ds, 'params': None, 'camMats': camMats, 'img_size': image.shape[:2], 'ds': 'mpiinf'}

        img_info['depth'] = np.array([[0, self.subject_gender[subject_id], 0, 0]])
         
        return img_info


    def pack_data(self,annots_file_path):
        self.annots = {}
        frame_info = {}
        user_list = range(1,9)
        seq_list = range(1,3)
        # view point 11,12,13 is look from ceiling, which is unusual.
        vid_list = list(range(11))
        h, w = 2048, 2048

        for user_i in user_list:
            for seq_i in seq_list:
                seq_path = os.path.join('S' + str(user_i),'Seq' + str(seq_i))
                # mat file with annotations
                annot_file = os.path.join(seq_path, 'annot.mat')
                annot_file_path = os.path.join(self.data_folder, annot_file)
                print('Processing ',annot_file_path)
                annot2 = sio.loadmat(annot_file_path)['annot2']
                annot3 = sio.loadmat(annot_file_path)['annot3']
                # calibration file and camera parameters
                calib_file = os.path.join(self.data_folder, seq_path, 'camera.calibration')
                Ks, Rs, Ts = read_calibration(calib_file, vid_list)
                

                for j, vid_i in enumerate(vid_list):
                    annots_2d = annot2[vid_i][0]
                    annots_3d = annot3[vid_i][0]
                    frame_num = len(annots_3d)
                    video_name = os.path.join(seq_path,'imageSequence','video_' + str(vid_i) + '.avi')
                    frame_info[video_name] = []
                    sellected_frame_ids = []

                    fx, fy, cx, cy = Ks[j][0,0], Ks[j][1,1], Ks[j][0,2], Ks[j][1,2]
                    intrinsics = np.array([fx, fy, cx, cy])
                    R, T = Rs[j], Ts[j]
                    for frame_id in range(frame_num):
                        img_name = self.get_image_name(video_name, frame_id)#'S{}_Seq{}_video{}_F{}.jpg'.format(user_i, seq_i, vid_i, frame_id)
                        kp2d = annots_2d[frame_id].reshape(-1,2)
                        kp3d = annots_3d[frame_id].reshape(-1,3)/1000

                        if _check_visible(kp2d, w=w, h=h) and frame_id%10==1:
                            self.annots[img_name] = {'kp2d':kp2d, 'kp3d':kp3d, 'intrinsics': intrinsics, 'extrinsics':[R, T]}
                            frame_info[video_name].append(frame_id)
        np.savez(annots_file_path, annots=self.annots, frame_info=frame_info)
        print('MPI_INF_3DHP data annotations packed')


    def extract_frames(self,frame_info):
        os.makedirs(self.image_folder,exist_ok=True)
        for video_name, frame_ids in frame_info.items():
            video_path = os.path.join(self.data_folder, video_name)
            print('Extracting {}'.format(video_path))
            vidcap = cv2.VideoCapture(video_path)
            frame_id = 0
            while 1:
                success, image = vidcap.read()
                if not success:
                    break
                
                if frame_id in frame_ids:
                    img_name = self.get_image_name(video_name, frame_id)
                    cv2.imwrite(os.path.join(self.image_folder,img_name), image)
                frame_id += 1

    def get_image_name(self,video_name, frame_id):
        return video_name.strip('.avi').replace('/imageSequence','').replace('/','_')+'_F{:06d}.jpg'.format(frame_id)


class MPI_INF_3DHP_VALIDATION(MPI_INF_3DHP):
    def __init__(self,train_flag=False, validation=True, **kwargs):
        super(MPI_INF_3DHP_VALIDATION,self).__init__(train_flag=train_flag, validation=validation)


def _check_visible(joints, w=2048, h=2048, get_mask=False):
    visibility = True
    # check that all joints are visible
    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
    ok_pts = np.logical_and(x_in, y_in)
    if np.sum(ok_pts) < len(joints):
        visibility=False
    if get_mask:
        return ok_pts
    return visibility

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts

if __name__ == '__main__':
    args().use_fit_smpl_params = False
    dataset=MPI_INF_3DHP(train_flag=True,regress_smpl=True)
    test_dataset(dataset,with_smpl=True)
    print('Done')
