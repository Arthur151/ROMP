import sys, os
from collections import OrderedDict
from datasets.image_base import *
from datasets.base import Base_Classes, Test_Funcs

default_mode = args().video_loading_mode if args().video else args().image_loading_mode

def PennAction(base_class=default_mode):
    class PennAction(Base_Classes[base_class]):

        def __init__(self,train_flag=True, regress_smpl=False, **kwargs):
            super(PennAction,self).__init__(train_flag,regress_smpl=regress_smpl)
            self.min_pts_required = 2
            self.prepare_data()
            self.kp2d_mapper = constants.joint_mapping(constants.PennAction_13,constants.SMPL_ALL_44)
            if base_class == 'video_relative':
                self.video_clip_ids = self.prepare_video_clips()
            logging.info('PennAction 2D keypoint data has been loaded, total {} samples, contains {} IDs'.format(len(self), self.ID_num))
        
        def prepare_data(self):
            self.root = os.path.join(self.data_folder,"Penn_Action")
            self.annots_file_path = os.path.join(self.root,'annots_train.npz')
            
            if os.path.exists(self.annots_file_path):
                annotations = np.load(self.annots_file_path, allow_pickle=True)
                self.annots, self.ID_num, self.sequence_dict, seq_frame_names = \
                    annotations['annot'][()], annotations['person_ids'][()]['id_number'], annotations['sequence_dict'][()], annotations['seq_frame_names'][()]
            else:
                self.annots, self.ID_num, self.sequence_dict, seq_frame_names = self.pack_annots()
            
            self.sequence_dict = OrderedDict(self.sequence_dict)
            self.file_paths, self.sequence_ids, self.sid_video_name = [], [], {}
            for sid, video_name in enumerate(self.sequence_dict):
                self.sid_video_name[sid] = video_name
                self.sequence_ids.append([])
                for fid in self.sequence_dict[video_name]:
                    self.file_paths.append([sid,fid,seq_frame_names[video_name][fid]])
                    self.sequence_ids[sid].append(len(self.file_paths)-1)
            
            load_eft_annots_path = os.path.join(self.root,'eft_annots.npz')
            #os.remove(load_eft_annots_path)
            if os.path.exists(load_eft_annots_path):
                self.eft_annots = np.load(load_eft_annots_path,allow_pickle=True)['annots'][()]
            else:
                self.load_eft_annots('/home/yusun/DataCenter2/existing_works/eft/eft_out/pennaction_fitting')
                np.savez(load_eft_annots_path, annots=self.eft_annots)
            
            if self.regress_smpl:
                self.smplr = SMPLR(use_gender=False)
                self.root_inds = None
        
        def load_eft_annots(self, annot_file_dir):
            self.eft_annots = {}
            import quaternion
            for video_annots_file in glob.glob(os.path.join(annot_file_dir,'*.pkl')):
                eft_annots = load_pkl_func(video_annots_file)
                for eft_data in eft_annots.values():          
                    imgFullPath = eft_data['imageName'][0].split('/')[-2]+'-'+eft_data['imageName'][0].split('/')[-1]
                    if eft_data['loss_keypoints_2d']>2e-5:
                        print(eft_data['loss_keypoints_2d'], eft_data['loss'])
                        continue

                    pred_betas = np.reshape(np.array( eft_data['pred_shape'], dtype=np.float32), (10) )     #(10,)
                    pred_pose_rotmat = np.reshape( np.array( eft_data['pred_pose_rotmat'], dtype=np.float32), (24,3,3)  )        #(24,3,3)
                    pred_pose = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(pred_pose_rotmat)).reshape(-1)

                    if imgFullPath not in self.eft_annots:
                        self.eft_annots[imgFullPath] = []
                    self.eft_annots[imgFullPath].append([pred_pose, pred_betas])
            logging.info('EFT pseudo-label contains annotations for {} samples'.format(len(self.eft_annots)))

        def get_image_info(self,index):
            sid, frame_id, img_name = self.file_paths[index%len(self.file_paths)]
            seq_name = self.sid_video_name[sid]
            end_frame_flag = frame_id == (len(self.sequence_ids[sid])-1)

            imgpath = self._get_image_path(seq_name, img_name)
            image = cv2.imread(imgpath)[:,:,::-1]

            kp2ds, valid_mask_2d, valid_mask_3d = [], [], []
            annot_name = seq_name+'-'+img_name
            for idx,joint in enumerate(self.annots[annot_name][0]):
                joint = np.array(joint)
                #joint[:,2] = joint[:,0]>1
                joint = np.array(joint)[self.kp2d_mapper]
                joint[self.kp2d_mapper==-1] = -2
                kp2ds.append(joint)
                valid_mask_2d.append([True,True,False])
                valid_mask_3d.append(copy.deepcopy(self.default_valid_mask_3d))
            
            params = np.ones((len(kp2ds), 66+10))*-10
            kp3ds = None
            if annot_name in self.eft_annots:
                theta, beta = self.eft_annots[annot_name][0]
                params[0] = np.array([np.concatenate([theta[:66], beta])])
                #valid_mask_3d[0][1] = False # the rotation is wrong, seems good in front view, but very wrong in side view
                valid_mask_3d[0][2] = True 
                if self.regress_smpl:
                    _, kp3ds = self.smplr(theta, beta)
                    valid_mask_3d[0][0] = True
                
            kp2ds, track_ids = np.array(kp2ds), np.array(self.annots[annot_name][1])
            valid_mask_2d, valid_mask_3d = np.array(valid_mask_2d), np.array(valid_mask_3d)
            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': track_ids,\
                    'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d, 'dynamic_supervise':True, \
                    'kp3ds': kp3ds, 'params': params, 'root_trans': None, 'verts': None, 'is_static_cam': True,\
                    'img_size': image.shape[:2], 'ds': 'PennAction'}
            if base_class == 'video_relative':
                img_info.update({'seq_info':[sid, frame_id, end_frame_flag]})
            
            return img_info

        def pack_annots(self, ):
            print('Packing annotations of Penn Action datasets')
            annots, sequence_dict, seq_frame_names, person_ids, seq_info = {}, {}, {}, {}, {}
            annots_dir = os.path.join(self.root,'labels')
            annots_file_list = sorted(glob.glob(os.path.join(annots_dir, '*.mat')))
            for annot_path in annots_file_list:
                #print('Processing {}'.format(annot_path))
                seq_annots = scio.loadmat(annot_path)
                video_name = os.path.basename(annot_path).replace('.mat', '')
                seq_id = int(video_name)-1

                action_name = seq_annots['action']
                frame_num = seq_annots['nframes'][0,0]
                train_flag = seq_annots['train'][0,0]
                #bboxes = seq_annots['bbox']
                kp2ds = np.stack([seq_annots['x'], seq_annots['y'], seq_annots['visibility']], 2)
                assert frame_num == len(kp2ds), print(kp2ds.shape, frame_num)

                for frame_id in range(frame_num):
                    img_name = '{:06}.jpg'.format(frame_id+1)
                    annot_name = '{}-{}'.format(video_name,img_name)

                    if video_name not in sequence_dict:
                        sequence_dict[video_name], seq_frame_names[video_name], person_ids[video_name]  = [], [], {0:seq_id}
                        seq_info[video_name] = [action_name,frame_num,train_flag]
                    sequence_dict[video_name].append(len(sequence_dict[video_name]))
                    seq_frame_names[video_name].append(img_name)
                    annots[annot_name] = [[kp2ds[frame_id]], [seq_id]]
                if seq_id%200==0 and video_name in sequence_dict:
                    print('Processing {}/{} seqence'.format(seq_id, len(annots_file_list)))
                    #print(video_name,sequence_dict[video_name])

            ID_num = seq_id + 1
            np.savez(self.annots_file_path, annot = annots, sequence_dict=sequence_dict, seq_frame_names=seq_frame_names, \
                person_ids={'map_dict':person_ids, 'id_number':ID_num}, seq_info=seq_info)
            print('Saving annotations to {}'.format(self.annots_file_path))
            return annots, ID_num, sequence_dict, seq_frame_names

        def _get_image_path(self, seq_name, file_name):
            images_dir = os.path.join(self.root,'frames',seq_name,file_name)
            return images_dir
    return PennAction

def load_pkl_func(path_target):
    with open(path_target, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    datasets = PennAction(base_class=default_mode)(train_flag=True)
    Test_Funcs[default_mode](datasets, with_smpl=True)
    print('Done')
