from config import args
from collections import OrderedDict
from datasets.image_base import *
from datasets.base import Base_Classes, Test_Funcs
import scipy.io as sio
from utils.util import transform_rot_representation

default_mode = args().video_loading_mode if args().video else args().image_loading_mode

def MPI_INF_3DHP(base_class=default_mode):
    class MPI_INF_3DHP(Base_Classes[base_class]):
        def __init__(self, train_flag=True, validation=False, **kwargs):
            super(MPI_INF_3DHP,self).__init__(train_flag, regress_smpl=False,syn_obj_occlusion=False) # syn_obj_occlusion 会影响人体中心点位置的定义，导致det loss 崩溃.
            self.data_folder = os.path.join(self.data_folder, 'mpi_inf_3dhp/')
            if args().video and 'DataCenter' in self.data_folder:
                self.data_folder = self.data_folder.replace('DataCenter', 'DataCenter2').replace('mpi_inf_3dhp', 'mpi_inf_3dhp_video')
            
            if args().video:
                annots_file_path = os.path.join(self.data_folder, 'annots_video.npz') #_smplx
                self.image_folder = os.path.join(self.data_folder, 'video_frames')
            else:
                annots_file_path = os.path.join(self.data_folder, 'annots_image_smpl.npz')
                self.image_folder = os.path.join(self.data_folder, 'images')
            self.scale_range = [1.3,1.9]
            if os.path.exists(annots_file_path):
                self.annots = np.load(annots_file_path,allow_pickle=True)['annots'][()]
                self.cam_info = np.load(annots_file_path,allow_pickle=True)['cam_info'][()]
            else:
                self.pack_data(annots_file_path)
            
            for missed_image in ['S3_Seq1_video_7_F012488.jpg']:
                if missed_image in self.annots:
                    del self.annots[missed_image]
            #shutil.rmtree(self.image_folder)
            if not os.path.exists(self.image_folder): # and not self.local_test_mode
                frame_info = np.load(annots_file_path,allow_pickle=True)['frame_info'][()]
                self.extract_frames(frame_info)
            self.file_paths = list(self.annots.keys())
            set_name = 'train'
            self.track_id = {'S1':1,'S2':2,'S3':3,'S4':4,'S5':5,'S6':6,'S7':7, 'S8':8}
            self.ID_num = 8
            previous_sample_num = len(self.file_paths)
            self.subject_gender = {'S1':1, 'S2':0, 'S3':0, 'S4':1, 'S5':1, 'S6':1, 'S7':0, 'S8':0}
            self.kp2d_mapper = constants.joint_mapping(constants.MPI_INF_28,constants.SMPL_ALL_44)
            self.kp3d_mapper = constants.joint_mapping(constants.MPI_INF_28,constants.SMPL_ALL_44)
            self.compress_length = 3
            self.aug_cropping_ratio = args().shuffle_crop_ratio_3d

            if base_class == 'video_relative':
                self.load_video_list()
            else:
                self.sample_num = len(self.file_paths)//self.compress_length if self.train_flag else len(self.file_paths)

            self.random_temp_sample_internal = 1 # the video frame is sampled by every 10 frames, setting interval as 1 is enough           
            self.use_smpl_params = False #args().smpl_model_type == 'smplx'

            logging.info('Loaded MPI-INF-3DHP {} set,total {} samples'.format(set_name, self.__len__()))
        
        def load_video_list(self):
            self.sequence_dict = {}
            for frame_name in self.file_paths:
                frame_id = int(frame_name.split('_F')[-1].replace('.jpg',''))
                seq_name = frame_name.replace('_F{:06}.jpg'.format(frame_id), '')
                if seq_name not in self.sequence_dict:
                    self.sequence_dict[seq_name] = {}
                self.sequence_dict[seq_name][frame_id] = frame_name
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
            #self.sequence_first_ids = [sids[0] for sids in self.sequence_ids]
            self.ID_num = len(self.sequence_ids)
            if base_class == 'video_relative':
                self.video_clip_ids = self.prepare_video_clips()
                self.sample_num = len(self.video_clip_ids)

        def exclude_subjects(self, file_paths, subjects=['S8']):
            file_path_left = []
            for inds, file_path in enumerate(file_paths):
                subject_id = os.path.basename(file_path).split('_')[0]
                if subject_id not in subjects:
                    file_path_left.append(file_path)
            return file_path_left

        def __len__(self):
            return self.sample_num

        def get_image_info(self, index):
            if self.train_flag and not base_class == 'video_relative':
                index = index*self.compress_length + random.randint(0,self.compress_length-1)
                # if self.homogenize_pose_space:
                #     index = self.homogenize_pose_sample(index)
            img_name = self.file_paths[index%len(self.file_paths)]
            subject_id = os.path.basename(img_name).split('_')[0]

            if args().video:
                imgpath = os.path.join(self.image_folder, img_name.split('_F')[0], img_name)
                if not os.path.exists(imgpath):
                    basename = os.path.basename(imgpath)
                    frame_str = basename.split('_F')[1].replace('.jpg','')
                    imgpath = os.path.join(os.path.dirname(imgpath), basename.replace(frame_str, str(int(frame_str))))
            else:
                imgpath = os.path.join(self.image_folder, img_name)
                if not os.path.exists(imgpath):
                    basename = os.path.basename(imgpath)
                    frame_str = basename.split('_F')[1].replace('.jpg','')
                    imgpath = os.path.join(os.path.dirname(imgpath), basename.replace('F'+frame_str, 'F{:06d}'.format(int(frame_str))))
            
            # TODO: check the missed images.
            if not os.path.exists(imgpath):
                print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('MPI-INF-3DHP miss image:', imgpath)
                return self.reget_info()

            if base_class == 'video_relative':
                sid, frame_id = self.seq_info[index%len(self.file_paths)]
                seq_name = self.sid_video_name[sid]
                end_frame_flag = frame_id == (len(self.sequence_ids[sid])-1)
                track_ids = [sid]
            else:
                track_ids = [self.track_id[subject_id]]

            image = cv2.imread(imgpath)[:,:,::-1]
            seq_name = img_name.split('_F')[0]
            R, T = self.cam_info[seq_name]['extrinsics']
            fx, fy, cx, cy = self.cam_info[seq_name]['intrinsics']

            camMats = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])
            camPose = np.concatenate([R, T[:,None]], 1)
            kp2ds = self.map_kps(self.annots[img_name]['kp2d'], maps=self.kp2d_mapper)
            kp3ds = self.map_kps(self.annots[img_name]['kp3d'], maps=self.kp3d_mapper)[None]
            vis_mask = _check_visible(kp2ds, get_mask=True)
            kp2ds = np.concatenate([kp2ds, vis_mask[:,None]], 1)[None]

            root_trans = kp3ds[:,self.root_inds].mean(1)
            # don't sub root_trans here, would affect the -2. 
            #kp3ds -= root_trans[:,None]
            # only 0.00+ error of root_trans pass
            #root_trans_pred = estimate_translation(kp3ds, kp2ds, proj_mats=camMats[None])

            if self.use_smpl_params:
                theta, betas = self.annots[img_name]['smpl_theta'], self.annots[img_name]['smpl_beta']
                params = [np.concatenate([theta, betas])]
                vmask_3d = np.array([[True,True,True,False,False,True]])
            else:
                vmask_3d = np.array([[True,False,False,False,False,True]])
                params = None

            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': track_ids,\
                    'vmask_2d': np.array([[True,True,True]]), 'vmask_3d': vmask_3d,\
                    'kp3ds': kp3ds, 'params': params, 'root_trans': root_trans, 'verts': None, 'is_static_cam': True,\
                    'camMats': camMats, 'camPose':camPose, 'img_size': image.shape[:2], 'ds': 'mpiinf'}

            if 'relative' in base_class:
                img_info['depth'] = np.array([[0, self.subject_gender[subject_id], 0, 0]])
                img_info['kid_shape_offsets'] = np.array([0])

            if base_class == 'video_relative':
                img_info.update({'seq_info':[sid, frame_id, end_frame_flag]})
            
            return img_info


        def pack_data(self,annots_file_path):
            self.annots = {}
            frame_info = {}
            cam_info = {}
            user_list = range(1,9)
            seq_list = range(1,3)
            # view point 11,12,13 is look from ceiling, which is unusual.
            vid_list = list(range(11))
            h, w = 2048, 2048
            if 'data_drive' in self.data_folder:
                gt_folder = self.data_folder.replace('data_drive2', 'data_drive')
            else:
                gt_folder = self.data_folder
            
            if args().video:
                smpl_param_path = os.path.join(self.data_folder, 'MPI-INF-3DHP_SMPLX_NeuralAnnot.json')
            else:
                smpl_param_path = os.path.join(self.data_folder, 'MPI-INF-3DHP_SMPL_NeuralAnnot.json')
            with open(smpl_param_path,'r') as f:
                smpl_params = json.load(f)

            for user_i in user_list:
                for seq_i in seq_list:
                    seq_path = os.path.join('S' + str(user_i),'Seq' + str(seq_i))
                    # mat file with annotations
                    annot_file = os.path.join(seq_path, 'annot.mat')
                    annot_file_path = os.path.join(gt_folder, annot_file)
                    print('Processing ',annot_file_path)
                    annot2 = sio.loadmat(annot_file_path)['annot2']
                    annot3 = sio.loadmat(annot_file_path)['annot3']
                    # calibration file and camera parameters
                    calib_file = os.path.join(gt_folder, seq_path, 'camera.calibration')
                    Ks, Rs, Ts = read_calibration(calib_file, vid_list)
                    
                    for j, vid_i in enumerate(vid_list):
                        annots_2d = annot2[vid_i][0]
                        annots_3d = annot3[vid_i][0]
                        seq_smpl_params = smpl_params[str(user_i)][str(seq_i)]
                        frame_num = len(annots_3d)
                        video_name = os.path.join(seq_path,'imageSequence','video_' + str(vid_i) + '.avi')
                        frame_info[video_name] = []
                        sellected_frame_ids = []

                        fx, fy, cx, cy = Ks[j][0,0], Ks[j][1,1], Ks[j][0,2], Ks[j][1,2]
                        intrinsics = np.array([fx, fy, cx, cy])
                        R, T = Rs[j], Ts[j]
                        for frame_id in range(frame_num):
                            if not args().video and frame_id%10!=1:
                                continue

                            img_name = self.get_image_name(video_name, frame_id)#'S{}_Seq{}_video_{}_F{}.jpg'.format(user_i, seq_i, vid_i, frame_id)
                            seq_name = img_name.split('_F')[0]
                            if seq_name not in cam_info:
                                cam_info[seq_name] = {'intrinsics': intrinsics, 'extrinsics':[R, T]}
                            kp2d = annots_2d[frame_id].reshape(-1,2)
                            kp3d = annots_3d[frame_id].reshape(-1,3)/1000
                            
                            # the video is 25 fps, to faclitate video representation learning, we need the real video frames
                            if _check_visible(kp2d, w=w, h=h, min_kp_num=len(kp2d)):
                                if str(frame_id+1) not in seq_smpl_params:
                                    print(seq_path, vid_i,'smplx params of missing', frame_id+1)
                                    continue
                                self.annots[img_name] = {'kp2d':kp2d, 'kp3d':kp3d}
                                frame_info[video_name].append(frame_id)
                                
                                smpl_param = seq_smpl_params[str(frame_id+1)]
                                if args().video:
                                    global_orient = np.array(smpl_param['root_pose'])
                                    body_pose = np.array(smpl_param['body_pose'])
                                else:
                                    global_orient = np.array(smpl_param['pose'][:3])
                                    body_pose = np.array(smpl_param['pose'][3:])
                                smpl_beta = np.array(smpl_param['shape']) #非常不准，同一个人每一帧都不一样
                                trans = np.array(smpl_param['trans'])

                                global_orient_mat = transform_rot_representation(global_orient, input_type='vec',out_type='mat')
                                global_orient_mat_cam = np.matmul(R, global_orient_mat)
                                grot_cam = transform_rot_representation(global_orient_mat_cam, input_type='mat',out_type='vec')
                                smpl_theta = np.concatenate([grot_cam, body_pose], 0)

                                self.annots[img_name].update({'smpl_theta':smpl_theta, 'smpl_beta':smpl_beta})


            np.savez(annots_file_path, annots=self.annots, frame_info=frame_info, cam_info=cam_info)
            self.cam_info = cam_info
            print('MPI_INF_3DHP data annotations packed')

        def extract_frames(self,frame_info):
            os.makedirs(self.image_folder,exist_ok=True)
            if 'data_drive' in self.data_folder:
                gt_folder = self.data_folder.replace('data_drive2', 'data_drive')
            else:
                gt_folder = self.data_folder
            for video_name, frame_ids in frame_info.items():
                video_path = os.path.join(gt_folder, video_name)
                seq_dir = video_name.strip('.avi').replace('/imageSequence','').replace('/','_')
                seq_save_path = os.path.join(self.image_folder, seq_dir)
                if os.path.exists(seq_save_path):
                    continue
                os.makedirs(seq_save_path,exist_ok=True)
                print('Extracting {}'.format(video_path))
                vidcap = cv2.VideoCapture(video_path)
                frame_id = 0
                while 1:
                    success, image = vidcap.read()
                    if not success:
                        break
                    
                    if frame_id in frame_ids:
                        img_name = self.get_image_name(video_name, frame_id)                        
                        cv2.imwrite(os.path.join(self.image_folder, seq_dir,img_name), image)
                    frame_id += 1

        def get_image_name(self,video_name, frame_id):
            if args().video:
                return video_name.strip('.avi').replace('/imageSequence','').replace('/','_')+'_F{:06d}.jpg'.format(frame_id)
            else:
                return video_name.strip('.avi').replace('/imageSequence','').replace('/','_')+'_F{}.jpg'.format(frame_id)
    return MPI_INF_3DHP


def _check_visible(joints, w=2048, h=2048, get_mask=False, min_kp_num=5):
    visibility = True
    # check that all joints are visible
    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
    ok_pts = np.logical_and(x_in, y_in)
    if np.sum(ok_pts) < min_kp_num:
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
    datasets=MPI_INF_3DHP(base_class=default_mode)(train_flag=True,regress_smpl=False)
    Test_Funcs[default_mode](datasets,with_smpl=False)
    print('Done')

"""
all_joint_names = {'spine3', 'spine4', 'spine2', 'spine', 'pelvis', ...     %5       
        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', ... %11
       'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', ... %17
       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', ...        %23   
       'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe'}; 

TRAINING:
For each subject & sequence there is annot.mat
What is in annot.mat:
  'frames': number of frames, N
  'univ_annot3': (14,) for each camera of N x 84 -> Why is there univ for each camera if it's univ..?
  'annot3': (14,) for each camera of N x 84
  'annot2': (14,) for each camera of N x 56
  'cameras':

  In total there are 28 joints, but H3.6M subsets are used.

  The image frames are unpacked in:
  BASE_DIR/S%d/Seq%d/video_%d/frame_%06.jpg


TESTING:
  'valid_frame': N_frames x 1
  'annot2': N_frames x 1 x 17 x 2
  'annot3': N_frames x 1 x 17 x 3
  'univ_annot3': N_frames x 1 x 17 x 3
  'bb_crop': this is N_frames x 34 (not sure what this is..)
  'activity_annotation': N_frames x 1 (of integer indicating activity type
  The test file_paths are already in jpg.
"""