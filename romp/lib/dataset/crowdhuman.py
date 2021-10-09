from pycocotools.coco import COCO
from dataset.image_base import *

class CrowdHuman(Image_base):
    def __init__(self,train_flag=True, **kwargs):
        super(CrowdHuman,self).__init__(train_flag)
        self.min_pts_required = 2
        self.init_coco()
        self.kp2d_mapper = constants.joint_mapping(constants.Posetrack_17,constants.SMPL_ALL_54)
        logging.info('CrowdHuman 2D detection data has been loaded, total {} samples'.format(len(self)))
    
    def init_coco(self):
        self.root = os.path.join(self.data_folder,"crowdhuman")
        self.split_name = 'train' if self.train_flag else 'val'
        self.annots_file_path = os.path.join(self.root,'annots_{}.npz'.format(self.split_name))
        if os.path.exists(self.annots_file_path):
            self.annots = np.load(self.annots_file_path, allow_pickle=True)['annots'][()]
        else:
            self.pack_annots()
        self.file_paths = list(self.annots.keys())

    def get_image_info(self,index):
        img_name = self.file_paths[index%len(self.file_paths)]
        imgpath = self._get_image_path(img_name)
        image = cv2.imread(imgpath)[:,:,::-1]
        bboxes = self.annots[img_name]
        person_num = len(bboxes['fbox'])

        fv_bboxes, valid_mask_2d, valid_mask_3d = [], [], []
        for inds in range(person_num):
            (fx, fy, fw, fh), (vx, vy, vw, vh) = bboxes['fbox'][inds], bboxes['vbox'][inds]
            fv_bboxes.append(np.array([[fx, fy], [fx+fw, fy+fh], [vx, vy], [vx+vw, vy+vh]]))
            valid_mask_2d.append([False,False,True])
            valid_mask_3d.append(self.default_valid_mask_3d)

        valid_mask_2d, valid_mask_3d = np.array(valid_mask_2d), np.array(valid_mask_3d)
        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': fv_bboxes, 'track_ids': None,\
                'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d,\
                'kp3ds': None, 'params': None, 'root_trans': None, 'verts': None,\
                'img_size': image.shape[:2], 'ds': 'crowdhuman'}
         
        return img_info

    def pack_annots(self):
        self.annots = {}
        ann_path = os.path.join(self.root,'annotation_{}.odgt'.format(self.split_name))
        anns_data = load_func(ann_path)
        for ann_data in anns_data:
            self.annots['{}.jpg'.format(ann_data['ID'])]={'hbox':[], 'fbox':[],'vbox':[]}
            anns = ann_data['gtboxes']
            for i in range(len(anns)):
                iscrowd = 1 if 'extra' in anns[i] and 'ignore' in anns[i]['extra'] and \
                                         anns[i]['extra']['ignore'] == 1 else 0
                if iscrowd:
                    continue
                self.annots['{}.jpg'.format(ann_data['ID'])]['vbox'].append(anns[i]['vbox'])
                self.annots['{}.jpg'.format(ann_data['ID'])]['fbox'].append(anns[i]['fbox'])
                self.annots['{}.jpg'.format(ann_data['ID'])]['hbox'].append(anns[i]['hbox'])

        np.savez(self.annots_file_path, annots = self.annots)

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images', self.split_name, file_name)
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

def load_func(fpath):
    print('fpath', fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records

if __name__ == '__main__':
    args().configs_yml = 'configs/basic_training_pretrain.yml'
    args().model_version=0
    dataset = CrowdHuman(train_flag=False)
    test_dataset(dataset)
    print('Done')
