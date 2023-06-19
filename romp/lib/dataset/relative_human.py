import sys, os
from dataset.image_base import *
from pycocotools.coco import COCO
import pycocotools
from dataset.base import Base_Classes, Test_Funcs

default_mode = args().image_loading_mode

def Relative_human(base_class=default_mode):
    class Relative_human(Base_Classes[base_class]):

        def __init__(self, train_flag=True, split='train', regress_smpl=True, **kwargs):
            super(Relative_human,self).__init__(train_flag,regress_smpl)
            self.min_pts_required = 2
            self.split = split
            self._init_()
            logging.info('Relative_human data has been loaded, total {} samples'.format(len(self)))
        
        def _init_(self):
            self.root = os.path.join(self.data_folder,"Relative_human")
            self.annots_file_path = os.path.join(self.root,'{}_annots.npz'.format(self.split))
            self.annots = np.load(self.annots_file_path, allow_pickle=True)['annots'][()]
            self.file_paths = list(self.annots.keys())
            self.kp2d_mapper_OCH = constants.joint_mapping(constants.OCHuman_19,constants.SMPL_ALL_54)
            self.kp2d_mapper_CP = constants.joint_mapping(constants.Crowdpose_14,constants.SMPL_ALL_54)
            self.kp2d_mapper_BK = constants.joint_mapping(constants.BK_19,constants.SMPL_ALL_54)
            if self.homogenize_pose_space and self.train_flag:
                sample_dict_path = os.path.join(self.root, 'age_balanced_sample_dict.npz')
                age_balanced_sample_dict = np.load(sample_dict_path, allow_pickle=True)
                self.file_paths = age_balanced_sample_dict['file_paths'].tolist()
                age_pools = age_balanced_sample_dict['cluster_pool'].tolist()
                self.cluster_pool = [age_pools[0], age_pools[0]+age_pools[1], age_pools[0]+age_pools[1], age_pools[2], age_pools[3]]

        def get_image_info(self,index):
            if self.homogenize_pose_space and self.train_flag:
                index = self.homogenize_pose_sample(index)
            img_name = self.file_paths[index%len(self.file_paths)]
            imgpath = self._get_image_path(img_name)
            image = cv2.imread(imgpath)[:,:,::-1]
            #mask = self.get_exclude_mask(anno, index)[:,:,np.newaxis].astype(np.float32)
            annots = self.annots[img_name]
            kp2ds, valid_mask_2ds, valid_mask_3ds, depth_info = [], [], [], []
            
            for idx,annot in enumerate(annots):
                valid_mask_2d = [False,False,True]
                vbox = np.array(annot['bbox'])
                #vbox[2:] += vbox[:2]
                #fbox = np.array(annot['bbox_wb']) if 'bbox_wb' in annot else vbox
                fbox = vbox
                joint = np.array([fbox[:2], fbox[2:], vbox[:2], vbox[2:]])
                if 'kp2d' in annot:
                    if annot['kp2d'] is not None:
                        joint = np.array(annot['kp2d']).reshape((-1,3))
                        invalid_kp_mask = joint[:,2]==0
                        joint[invalid_kp_mask] = -2.
                        joint[:,2] = joint[:,2]>0
                        valid_mask_2d[0] = True
                        if len(joint) == 19:
                            is_BK = len(os.path.basename(img_name).replace('.jpg',''))==7
                            if is_BK:
                                joint = self.map_kps(joint,maps=self.kp2d_mapper_BK)
                            else:
                                joint = self.map_kps(joint,maps=self.kp2d_mapper_OCH)
                        elif len(joint) == 14:
                            joint = self.map_kps(joint,maps=self.kp2d_mapper_CP)
                        else:
                            raise NotImplementedError                        

                if annot['body_type']==3:
                    annot['body_type']=0
                if 'depth_id' not in annot:
                    annot['depth_id'] = -1
                    print(img_name, 'depth_id missing!!')
                depth_info.append([annot['age'], annot['gender'], annot['body_type'], annot['depth_id']])

                #if 'segms' in annot:
                #    segms = annot['segms']
                kp2ds.append(joint)
                valid_mask_2ds.append(valid_mask_2d)
                valid_mask_3ds.append(self.default_valid_mask_3d)
            valid_mask_2ds, valid_mask_3ds = np.array(valid_mask_2ds), np.array(valid_mask_3ds)
            depth_info = np.array(depth_info)

            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': np.arange(len(kp2ds)),\
                    'vmask_2d': valid_mask_2ds, 'vmask_3d': valid_mask_3ds,\
                    'kp3ds': None, 'params': None, 'root_trans': None, 'verts': None,\
                    'img_size': image.shape[:2],'ds': 'relativity'}
            if 'relative' in base_class:
                img_info['depth'] = depth_info
            
            return img_info

        def _get_image_path(self, file_name):
            images_dir = os.path.join(self.root, 'images')
            return os.path.join(images_dir, file_name)

        def get_mask(self, anno, idx):
            # mask of crowd or person without annotated keypoint
            coco = self.coco
            img_info = coco.loadImgs(self.file_paths[idx])[0]
            m = np.zeros((img_info['height'], img_info['width']))
            for obj in anno:
                if obj['num_keypoints'] > self.min_pts_required and not obj['iscrowd']:
                    rles = pycocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'], img_info['width'])
                    for rle in rles:
                        m += pycocotools.mask.decode(rle)
            return m>0

        def get_exclude_mask(self, anno, idx):
            # mask of crowd or person without annotated keypoint
            coco = self.coco
            img_info = coco.loadImgs(self.file_paths[idx])[0]
            m = np.zeros((img_info['height'], img_info['width']))

            for obj in anno:
                if obj['iscrowd']:
                    rle = pycocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'], img_info['width'])
                    m += pycocotools.mask.decode(rle)
                elif obj['num_keypoints'] == 0:
                    rles = pycocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'], img_info['width'])
                    for rle in rles:
                        m += pycocotools.mask.decode(rle)

            return m<0.5
    return Relative_human

def prepare_age_balanced_sample_dict():
    annots_file_path = '/home/yusun/Desktop/train_annots.npz'
    annots = np.load(annots_file_path, allow_pickle=True)['annots'][()]
    file_paths = list(annots.keys())
    cluster_pool = [[] for _ in range(4)]
    for img_id, img_name in enumerate(file_paths):
        annot = annots[img_name]
        ages = np.zeros(len(annot))
        for idx,ann in enumerate(annot):
            ages[idx] = ann['age']
        for age_id in range(4):
            if (ages==age_id).sum()>0:
                cluster_pool[age_id].append(img_id)
    np.savez('age_balanced_sample_dict.npz', file_paths=file_paths, cluster_pool=cluster_pool)

if __name__ == '__main__':
    dataset = Relative_human(base_class=default_mode)(train_flag=True, split='train', regress_smpl=False)
    #dataset = Relative_human(base_class=default_mode)(train_flag=False, split='val')
    Test_Funcs[default_mode](dataset, with_smpl=False)
    print('Done')