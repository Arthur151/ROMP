
import sys, os
from collections import OrderedDict
from datasets.image_base import *
from datasets.base import Base_Classes, Test_Funcs

default_mode = args().video_loading_mode if args().video else args().image_loading_mode

def DynaCamTranslation(base_class=default_mode):
    class DynaCamTranslation(Base_Classes[base_class]):
        def __init__(self, train_flag=True, split='train', load_entire_sequence=False, regress_smpl=True, **kwargs):
            super(DynaCamTranslation, self).__init__(train_flag, regress_smpl=regress_smpl,
                                            load_entire_sequence=load_entire_sequence)
            self.split = split
            self.dynamic_aug_tracking_ratio = 0.6
            self.scale_range = [1.6,2.4]
            #self.generate_test_set = generate_test_set
            self.prepare_annots()
            if base_class == 'video_relative':
                self.video_clip_ids = self.prepare_video_clips()  
            self.kp2d_mapper = constants.joint_mapping(constants.OpenPose_25,constants.SMPL_ALL_44)
            # data augmentation, rotation, flip and crop would make the solving world trans wrong !!!!! Must be set to False
            self.train_flag = False 
            self.shuffle_mode = False
            print('DynaCamTranslation has been loaded, total {} samples, contains {} IDs'.format(len(self), self.ID_num))
        
        def prepare_annots(self):
            self.root = os.path.join(self.data_folder,"DynaCam")
            
            annots_file_path = os.path.join(self.root, 'annotations', f'translation_{self.split}.npz')
            annotations = np.load(annots_file_path, allow_pickle=True)
            self.annots = annotations['annots'][()]
            self.ID_num, self.sequence_dict = self.annots['ID_num'], self.annots['sequence_dict']
            self.sequence_dict = OrderedDict(self.sequence_dict)
            self.file_paths, self.sequence_ids, self.sid_video_name = [], [], []
            for sid, video_name in enumerate(self.sequence_dict):
                self.sequence_ids.append([])
                for cid, fid in enumerate(self.sequence_dict[video_name]):
                    self.file_paths.append([sid, cid, fid,os.path.join(video_name,'{:06d}.png'.format(fid))])
                    self.sequence_ids[sid].append(len(self.file_paths)-1)
                self.sid_video_name.append(video_name)

            if self.regress_smpl:
                self.smplr = SMPLR(use_gender=False)
                self.root_inds = None
        
        def get_image_info(self,index):
            sid, cid, fid, img_name = self.file_paths[index%len(self.file_paths)]
            seq_name = self.sid_video_name[sid]
            frame_id = np.where(np.array(self.sequence_dict[seq_name])==fid)[0][0]
            end_frame_flag = frame_id == (len(self.sequence_ids[sid])-1)
            imgpath = self._get_image_path(img_name)
            if not os.path.exists(imgpath):
                print(seq_name, imgpath,'is not existing')
            image = cv2.imread(imgpath)[:,:,::-1]

            track_ids = np.array(self.annots[seq_name]['person_id'])
            subject_num = len(track_ids)
            intrinsics = self.annots[seq_name]['camera_intrinsics'][cid]
            extrinsics = self.annots[seq_name]['camera_extrinsics'][cid]

            lt, rb = self.annots[seq_name]['seq_bboxes'][cid]
            image = image[lt[1]:rb[1], lt[0]:rb[0]]

            kp2ds, valid_mask_2d, valid_mask_3d = [], [], []
            params = np.ones((subject_num, 66+10)) * -10
            kp3ds = np.ones((subject_num, args().joint_num, 3), dtype=np.float32) * -2.
            for subject_id in range(len(track_ids)):
                #joint = self.annots[seq_name]['kp2ds'][subject_id, cid]
                joint = self.annots[seq_name]['kp2ds_crop'][subject_id, cid]
                joint = joint[self.kp2d_mapper]
                joint[self.kp2d_mapper==-1] = -2
                kp2ds.append(joint)
                valid_mask_2d.append([True,True,False])
                valid_mask_3d.append(copy.deepcopy(self.default_valid_mask_3d))

                theta = self.annots[seq_name]['poses'][subject_id, cid].reshape(-1)
                beta = self.annots[seq_name]['betas'][subject_id, cid]
                params[subject_id] = np.array([np.concatenate([theta[:66], beta])])
                valid_mask_3d[subject_id][:4] = True # to supervise the world body rotation
                if self.regress_smpl:
                    _, kp3ds[subject_id] = self.smplr(theta, beta)
            valid_mask_2d, valid_mask_3d = np.array(valid_mask_2d), np.array(valid_mask_3d)
            
            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': track_ids,\
                    'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d, \
                    'kp3ds': kp3ds, 'params': params, 'root_trans': None, 'verts': None, 
                    'camMats': intrinsics, 'camPoses': extrinsics, 'is_static_cam': False,
                    'precise_kp3d_mask':np.zeros((len(kp2ds),1),dtype=np.bool_),\
                    'img_size': image.shape[:2], 'ds': 'DynaCam'}
            if base_class == 'video_relative':
                img_info.update({'seq_info':[sid, frame_id, end_frame_flag]})
            
            world_body_rots = np.array([self.annots[seq_name]['world_grots'][subject_id, cid] for subject_id in range(len(track_ids))])
            world_body_trans = np.array([self.annots[seq_name]['world_trans'][subject_id, cid] for subject_id in range(len(track_ids))])
            img_info.update({'world_grots_trans': [world_body_rots, world_body_trans]})
            
            return img_info

        def _get_image_path(self, file_name):
            images_dir = os.path.join(self.root, 'video_frames', f'translation_{self.split}', file_name)
            return images_dir

    return DynaCamTranslation


if __name__ == '__main__':
    datasets = DynaCamTranslation(base_class=default_mode)(train_flag=True, split='train')
    Test_Funcs[default_mode](datasets, with_smpl=True)
    print('Done')
