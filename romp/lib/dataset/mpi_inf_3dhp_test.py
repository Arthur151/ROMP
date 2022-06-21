from config import args
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs

default_mode = args().image_loading_mode

def MPI_INF_3DHP_TEST(base_class=default_mode):
    class MPI_INF_3DHP_TEST(Base_Classes[base_class]):
        def __init__(self,train_flag=False, joint_format='smpl24', **kwargs):
            super(MPI_INF_3DHP_TEST,self).__init__(train_flag)
            self.data_folder = os.path.join(self.data_folder,'mpi_inf_3dhp/mpi_inf_3dhp_test_set')
            annots_file_path = os.path.join(self.data_folder, 'annots.npz')
            self.multi_mode = True
            self.track_id = {'TS1':1,'TS2':2,'TS3':3,'TS4':4,'TS5':5,'TS6':6}
            self.subject_gender = {'TS1':0,'TS2':0,'TS3':0,'TS4':0,'TS5':0,'TS6':1}
            self.focal_lengths = {'TS1':1499.2054687744,'TS2':1499.2054687744,'TS3':1499.2054687744,'TS4':1499.2054687744,\
                                    'TS5':1683.98345952,'TS6':1683.98345952}
            
            self.scale_range = [1.6,2.2]
            if os.path.exists(annots_file_path):
                self.annots = np.load(annots_file_path,allow_pickle=True)['annots'][()]
            else:
                self.pack_data(annots_file_path)
            self.file_paths = list(self.annots.keys())
            self.kp2d_mapper = constants.joint_mapping(constants.MPI_INF_TEST_17,constants.SMPL_ALL_54)
            self.kp3d_mapper = constants.joint_mapping(constants.MPI_INF_TEST_17,constants.SMPL_ALL_54)
            self.root_inds = [constants.SMPL_ALL_54['Pelvis']] #[constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']] 

            logging.info('Loaded MPI-INF-3DHP test data,total {} samples'.format(self.__len__()))

        def get_image_info(self, index):
            img_name = self.file_paths[index]
            imgpath = os.path.join(self.data_folder,img_name)
            subject_id = imgpath.split('/')[-3]
            if not os.path.exists(imgpath):
                print(imgpath, 'missing..')
                img_name = self.file_paths[random.randint(0,len(self))]
                imgpath = os.path.join(self.data_folder,img_name)
            image = cv2.imread(imgpath)[:,:,::-1]
            kp2ds = self.map_kps(self.annots[img_name]['kp2d'],maps=self.kp2d_mapper)
            kp3ds = self.map_kps(self.annots[img_name]['univ_kp3d'], maps=self.kp3d_mapper)[None]
            vis_mask = _check_visible(kp2ds,get_mask=True)
            kp2ds = np.concatenate([kp2ds, vis_mask[:,None]],1)[None]
            fl, h, w = self.focal_lengths[subject_id], *image.shape[:2]
            camMats = np.array([[fl, 0, w/2.], [0, fl, h/2.], [0,0,1]])

            root_trans = kp3ds[:,self.root_inds].mean(1)
            kp3ds -= root_trans[:,None]

            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 2: smpl global orient | 3: smpl body pose | 4: smpl body shape | 5: smpl verts | 6: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': [self.track_id[subject_id]],\
                    'vmask_2d': np.array([[True,True,True]]), 'vmask_3d': np.array([[True,False,False,False,False,True]]),\
                    'kp3ds': kp3ds, 'params': None, 'root_trans': root_trans, 'verts': None,\
                    'camMats': camMats, 'img_size': image.shape[:2], 'ds': 'mpiinf_test'}

            if 'relative' in base_class:
                img_info['depth'] = np.array([[0, self.subject_gender[subject_id], 0, 0]])
            
            return img_info

        def pack_data(self,annots_file_path):
            import mat73
            self.annots = {}
            frame_info = {}
            user_list = range(1,7)
            missing_frame = 0

            for user_i in user_list:
                if user_i<5:
                    h, w = 2048, 2048
                else:
                    h, w = 1080, 1920
                video_name = os.path.join('TS' + str(user_i))
                # mat file with annotations
                annot_file = os.path.join(video_name, 'annot_data.mat')
                annot_file_path = os.path.join(self.data_folder, annot_file)
                print('Processing ',annot_file_path)
                annotation = mat73.loadmat(annot_file_path)
                valid_frame = annotation['valid_frame']
                print(list(annotation.keys()))
                activity = annotation['activity_annotation']
                annots_2d = np.array(annotation['annot2'])
                annots_3d = np.array(annotation['annot3'])
                frame_num = annots_3d.shape[-1]
                univ_annot3 = np.array(annotation['univ_annot3'])
                print('valid video length:', valid_frame.sum())

                frame_info[video_name] = []
                for frame_id in np.where(valid_frame)[0]:
                    img_name = self.get_image_name(video_name, frame_id)
                    kp2d = annots_2d[:,:,frame_id].transpose(1,0)
                    kp3d = annots_3d[:,:,frame_id].transpose(1,0)/1000
                    univ_kp3d = univ_annot3[:,:,frame_id].transpose(1,0)/1000
                    if _check_visible(kp2d, w=w, h=h):
                        self.annots[img_name] = {'kp2d':kp2d, 'kp3d':kp3d, 'univ_kp3d':univ_kp3d}
                        frame_info[video_name].append(frame_id)
                    else:
                        missing_frame += 1
                print('{} frame without all kp visible'.format(missing_frame))
            np.savez(annots_file_path, annots=self.annots, frame_info=frame_info)
            print('MPI_INF_3DHP test set data annotations is packed')

        def get_image_name(self,video_name, frame_id):
            return os.path.join(video_name, 'imageSequence', 'img_{:06d}.jpg'.format(frame_id+1))
    return MPI_INF_3DHP_TEST
    
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
    dataset=MPI_INF_3DHP_TEST(base_class=default_mode)(train_flag=False)
    Test_Funcs[default_mode](dataset)
    print('Done')