import sys, os
from collections import OrderedDict
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs

default_mode = args().image_loading_mode

def Posetrack(base_class=default_mode):
    class Posetrack(Base_Classes[base_class]):
        def __init__(self, train_flag=True, **kwargs):
            super(Posetrack, self).__init__(train_flag)
            self.min_pts_required = 2
            self.init_coco()
            self.kp2d_mapper = constants.joint_mapping(constants.Posetrack_17,constants.SMPL_ALL_54)
            logging.info('Posetrack 2D keypoint data has been loaded, total {} samples, contains {} IDs'.format(len(self), self.ID_num))
        
        def init_coco(self):
            self.root = os.path.join(self.data_folder,"posetrack2018")
            
            if self.train_flag:
                self.split = 'train'
                self.annots_file_path = os.path.join(self.root,'annots_train.npz')
            else:
                self.split = 'val'
                self.annots_file_path = os.path.join(self.root,'annots_val.npz')
                self.shuffle_mode = False
            
            if os.path.exists(self.annots_file_path):
                annotations = np.load(self.annots_file_path, allow_pickle=True)
                self.annots, self.ID_num, self.sequence_dict = annotations['annot'][()], annotations['person_ids'][()]['id_number'], annotations['sequence_dict'][()]
            else:
                self.annots, self.ID_num, self.sequence_dict = self.pack_annots()
            
            self.sequence_dict = OrderedDict(self.sequence_dict)
            self.file_paths = []
            for sid, video_name in enumerate(self.sequence_dict):
                for fid in self.sequence_dict[video_name]:
                    self.file_paths.append([sid,fid,os.path.join('images',self.split,video_name,'{:06d}.jpg'.format(fid))])

        def get_image_info(self,index):
            sid, fid, img_name = self.file_paths[index%len(self.file_paths)]
            imgpath = self._get_image_path(img_name)
            image = cv2.imread(imgpath)[:,:,::-1]

            kp2ds, valid_mask_2d, valid_mask_3d = [], [], []
            for idx,joint in enumerate(self.annots[img_name][0]):
                joint = joint[self.kp2d_mapper]
                joint[self.kp2d_mapper==-1] = -2
                kp2ds.append(joint)
                valid_mask_2d.append([True,True,False])
                valid_mask_3d.append(self.default_valid_mask_3d)

            kp2ds, track_ids = np.array(kp2ds), np.array(self.annots[img_name][1])
            valid_mask_2d, valid_mask_3d = np.array(valid_mask_2d), np.array(valid_mask_3d)
            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 2: smpl global orient | 3: smpl body pose | 4: smpl body shape | 5: smpl verts | 6: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': track_ids,\
                    'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d, \
                    'kp3ds': None, 'params': None, 'root_trans': None, 'verts': None,\
                    'img_size': image.shape[:2], 'ds': 'posetrack'}
            
            return img_info

        def pack_annots(self, ):
            print('Packing annotations of posetrack2021 dataset')
            from pycocotools.coco import COCO
            annots, sequence_dict, person_ids, id_cache = {}, {}, {}, 0
            annots_dir = os.path.join(self.root,'annotations',self.split)
            for annot_path in glob.glob(os.path.join(annots_dir, '*.json')):
                print('Processing {}'.format(annot_path))
                coco = COCO(annot_path)
                img_ids = coco.getImgIds()
                dropped_frame_ids = []
                for index, img_id in enumerate(img_ids):
                    file_name = coco.loadImgs(img_id)[0]['file_name']
                    video_name = file_name.split('/')[2]
                    frame_id = int(file_name.split('/')[3].replace('.jpg',''))
                    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
                    annot = coco.loadAnns(ann_ids)
                    joints, track_ids = self.get_joints_ids(annot)
                    if len(joints)==0: # video_name in sequences_with_duplicted_identites or 
                        dropped_frame_ids.append(frame_id)
                        continue

                    if video_name not in sequence_dict:
                        sequence_dict[video_name], person_ids[video_name] = [], {}
                    sequence_dict[video_name].append(frame_id)
                    img_person_ids = []
                    for track_id in track_ids:
                        if track_id not in person_ids[video_name]:
                            person_ids[video_name][track_id] = id_cache
                            id_cache += 1
                        person_id = person_ids[video_name][track_id]
                        img_person_ids.append(person_id)
                    annots[file_name] = [joints, np.array(img_person_ids)]
                    if index%1000==0:
                        print('Processing {}/{}'.format(index, len(img_ids)))
                print('Dropping sequence {}, frames {}'.format(video_name, dropped_frame_ids))

            for video_name in sequence_dict:
                sequence_dict[video_name] = sorted(sequence_dict[video_name])
                #print(video_name,sequence_dict[video_name])
            np.savez(self.annots_file_path, annot = annots, sequence_dict=sequence_dict, person_ids={'map_dict':person_ids, 'id_number':id_cache})
            print('Saving annotations to {}'.format(self.annots_file_path))
            return annots, id_cache, sequence_dict

        def _get_image_path(self, file_name):
            images_dir = os.path.join(self.root, file_name)
            return images_dir

        def get_annot(self, index):
            coco = self.coco
            img_id = self.file_paths[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)

            file_name = coco.loadImgs(img_id)[0]['file_name']
            img = cv2.imread(self._get_image_path(file_name),
                    cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            img = img[:,:,::-1]

            return img, target, self._get_image_path(file_name)

        def get_joints_ids(self, anno):
            num_people = len(anno)
            joints, track_ids = [], []
            for i, obj in enumerate(anno):
                joint = np.array(obj['keypoints']).reshape([-1, 3])
                track_id = obj['track_id']
                if joint[:, -1].sum()<self.min_pts_required:
                    continue
                joints.append(joint)
                track_ids.append(track_id)
            
            return np.array(joints), track_ids
    return Posetrack


if __name__ == '__main__':
    dataset = Posetrack(base_class=default_mode)(train_flag=True)
    Test_Funcs[default_mode](dataset)
    print('Done')