from pycocotools.coco import COCO
import sys, os
from dataset.image_base import *

class Crowdpose(Image_base):
    def __init__(self,train_flag=True,split='train',**kwargs):
        super(Crowdpose,self).__init__(train_flag)
        self.min_pts_required = 2
        self.split = split
        self.init_coco()
        self.kp2d_mapper = constants.joint_mapping(constants.Crowdpose_14,constants.SMPL_ALL_54)
        logging.info('Crowdpose 2D keypoint data has been loaded, total {} samples'.format(len(self)))
    
    def init_coco(self):
        self.root = os.path.join(self.data_folder,"crowdpose")
        self.dataset_name = self.split
        self.annots_file_path = os.path.join(self.root,'annots_{}.npz'.format(self.dataset_name))
        if not os.path.exists(self.annots_file_path):
            self.coco = COCO(self._get_anno_file_name())
            self.file_paths = list(self.coco.imgs.keys())
            self.pack_annots()
        self.annots = np.load(self.annots_file_path, allow_pickle=True)['annot'][()]
        self.file_paths = list(self.annots.keys())

    def get_image_info(self,index):
        img_name = self.file_paths[index%len(self.file_paths)]
        imgpath = self._get_image_path(img_name)
        image = cv2.imread(imgpath)[:,:,::-1]

        person_num = len(self.annots[img_name])
        kp2ds = np.array([self.map_kps(kp2d, self.kp2d_mapper) for kp2d in self.annots[img_name]])
        valid_mask_2d = np.array([[True,True,True] for _ in range(person_num)])
        valid_mask_3d = np.array([self.default_valid_mask_3d for _ in range(person_num)])

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,\
                'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d,\
                'kp3ds': None, 'params': None, 'img_size': image.shape[:2], 'ds': 'crowdpose'}
         
        return img_info

    def pack_annots(self):
        annots = {}
        for index in range(len(self)):
            img_id = self.file_paths[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            if len(ann_ids) > 0:
                annot = self.coco.loadAnns(ann_ids)
                joints = self.get_joints(annot)
                if len(joints)>0:
                    if np.max(joints[:,:,-1].sum(-1))>self.min_pts_required:
                        idx = np.argmax(joints[:,:,-1].sum(-1))
                        valid_pt, valid_idx = joints[idx,:,1], joints[idx,:,-1]>0
                        valid_pt = valid_pt[valid_idx]
                        if (valid_pt.max()-valid_pt.min())>40:
                            file_name = self.coco.loadImgs(img_id)[0]['file_name']
                            annots[file_name] = joints
            if index%1000==0:
                print(index)
        np.savez(self.annots_file_path, annot = annots)

    def _get_anno_file_name(self):
        return os.path.join(self.root,'json','crowdpose_{}.json'.format(self.dataset_name))

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images', file_name)
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

    def get_joints(self, anno):
        num_people = len(anno)
        joints = []
        for i, obj in enumerate(anno):
            joint = np.array(obj['keypoints']).reshape([-1, 3])
            if joint[:, -1].sum()<1:
                continue
            joints.append(joint)
        
        return np.array(joints)


if __name__ == '__main__':
    dataset = Crowdpose(train_flag=False)
    test_dataset(dataset)
    print('Done')