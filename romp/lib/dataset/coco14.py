import sys, os
from dataset.image_base import *
from pycocotools.coco import COCO
import pycocotools
import lap

class COCO14(Image_base):
    def __init__(self,train_flag=True,high_resolution=False, regress_smpl=True,**kwargs):
        super(COCO14,self).__init__(train_flag,regress_smpl)
        self.min_pts_required = 2
        self.init_coco()
        logging.info('COCO 2D keypoint data has been loaded, total {} samples'.format(len(self)))
    
    def init_coco(self):
        self.name = 'COCO'
        self.root = os.path.join(self.data_folder,"coco")
        self.dataset_name = 'train2014' if self.train_flag else 'val2014'
        self.annots_file_path = os.path.join(self.root,'annots_{}.npz'.format(self.dataset_name))
        if os.path.exists(self.annots_file_path):
            self.annots = np.load(self.annots_file_path, allow_pickle=True)['annot'][()]
        else:
            self.coco = COCO(self._get_anno_file_name())
            self.file_paths = list(self.coco.imgs.keys())
            self.annots = self.pack_annots()
        self.file_paths = list(self.annots.keys())
        self.joint_mapper = constants.joint_mapping(constants.COCO_17, constants.SMPL_ALL_54)
        if self.regress_smpl:
            self.smplr = SMPLR(use_gender=False)
            self.root_inds = None

        load_eft_annots_path = os.path.join(self.root,'eft_annots.npz')
        if os.path.exists(load_eft_annots_path):
            self.eft_annots = np.load(load_eft_annots_path,allow_pickle=True)['annots'][()]
        else:
            self.load_eft_annots(os.path.join(config.project_dir, 'data/eft_fit/COCO2014-All-ver01.json'))
            np.savez(load_eft_annots_path, annots=self.eft_annots)

    def get_image_info(self,index):
        img_name = self.file_paths[index%len(self.file_paths)]
        imgpath = self._get_image_path(img_name)
        image = cv2.imread(imgpath)[:,:,::-1]

        kp2ds, valid_mask_2d, valid_mask_3d, params = [], [], [], None
        
        for idx,joint in enumerate(self.annots[img_name]):
            joint = self.map_kps(joint,maps=self.joint_mapper)
            kp2ds.append(joint)
            valid_mask_2d.append([True,True,True])
            valid_mask_3d.append(self.default_valid_mask_3d)
        valid_mask_2d, valid_mask_3d = np.array(valid_mask_2d), np.array(valid_mask_3d)

        if img_name in self.eft_annots and self.use_eft:
            eft_annot = self.eft_annots[img_name]
            bbox_center_list, pose_list, betas_list = [], [], []
            for bbox_center, pose, betas in eft_annot:
                bbox_center_list.append(bbox_center)
                pose_list.append(pose[:66])
                betas_list.append(betas)
            bbox_center_list = np.array(bbox_center_list)
            cdist = np.array([np.linalg.norm(bbox_center_list-self._calc_center_(kp2d)[:2][None], axis=-1) for kp2d in kp2ds])
            matches = []
            cost, x, y = lap.lapjv(cdist, extend_cost=True)
            for ix, mx in enumerate(x):
                if mx >= 0:
                    matches.append([ix, mx])
            matches = np.asarray(matches)

            params = [None for _ in range(len(kp2ds))]
            for kid, pid in matches:
                matched_param = np.concatenate([pose_list[pid], betas_list[pid]])
                # when comes to crowds, this will lead to a lot duplicated matching to the same smpl parameters.
                params[kid] = matched_param
                valid_mask_3d[kid] = np.array([self.regress_smpl,True,True,False])

        kp3ds = self.regress_kp3d_from_smpl(params)

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,\
                'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d,\
                'kp3ds': kp3ds, 'params': params, 'img_size': image.shape[:2], 'ds': 'coco'}
         
        return img_info

    def load_eft_annots(self, annot_file_path):
        self.eft_annots = {}
        annots = json.load(open(annot_file_path,'r'))['data']
        for eft_data in annots:
            #Get raw image path
            imgFullPath = eft_data['imageName']
            imgName = os.path.basename(imgFullPath)

            #EFT data
            bbox_scale = eft_data['bbox_scale']
            bbox_center = eft_data['bbox_center']

            pred_camera = np.array(eft_data['parm_cam'])
            pred_betas = np.reshape(np.array( eft_data['parm_shape'], dtype=np.float32), (10) )     #(10,)
            pred_pose_rotmat = np.reshape( np.array( eft_data['parm_pose'], dtype=np.float32), (24,3,3)  )        #(24,3,3)
            pred_pose = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(pred_pose_rotmat)).reshape(-1)
            if imgName not in self.eft_annots:
                self.eft_annots[imgName] = []
            self.eft_annots[imgName].append([bbox_center, pred_pose, pred_betas])

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
                        if (valid_pt.max()-valid_pt.min())>128:
                            annot = [ obj for obj in annot if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0]
                            file_name = self.coco.loadImgs(img_id)[0]['file_name']
                            annots[file_name] = joints
                            print(file_name)
            if index%1000==0:
                print(index)
        np.savez(self.annots_file_path, annot = annots)
        return annots

    def _get_anno_file_name(self):
        return os.path.join(self.root,'annotations','person_keypoints_{}.json'.format(self.dataset_name))

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        return os.path.join(images_dir, self.dataset_name, file_name)

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


if __name__ == '__main__':
    args().configs_yml = 'configs/v7.yml'
    args().model_version=7
    dataset = COCO14(train_flag=True, regress_smpl=True)
    test_dataset(dataset, with_smpl=True)
    print('Done')
