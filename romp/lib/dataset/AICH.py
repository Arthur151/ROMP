from dataset.image_base import *

class AICH(Image_base):
    def __init__(self,train_flag=True,high_resolution=False, **kwargs):
        super(AICH,self).__init__(train_flag)
        self.max_intersec_ratio = 0.9
        self.min_pts_required = 3
        self.compress_length=8

        self.data_folder = os.path.join(self.data_folder,"ai_challenger/")
        self.annots_path = os.path.join(self.data_folder,'annots.npz')
        self.img_ext = '.jpg'
        self.joint_mapper = constants.joint_mapping(constants.LSP_14, constants.SMPL_ALL_54)
        if os.path.exists(self.annots_path):
            self.kp2ds = np.load(self.annots_path,allow_pickle=True)['annots'][()]
        else:
            self._load_data_set()
            np.savez(self.annots_path, annots=self.kp2ds)
        self.file_paths = list(self.kp2ds.keys())
        
        logging.info('AICH 2D keypoint data has been loaded, total {} samples'.format(len(self)))

    def _load_data_set(self):
        self.kp2ds = {}
        for imgdir_name, set_dir, anno_file in zip(['keypoint_train_images_20170902', 'keypoint_validation_images_20170911'],\
        ['ai_challenger_keypoint_train_20170909', 'ai_challenger_keypoint_validation_20170911'], ['keypoint_train_annotations_20170909.json', "keypoint_validation_annotations_20170911.json"]):
            anno_file_path = os.path.join(self.data_folder, set_dir, anno_file)
            logging.info('Processing {}'.format(anno_file_path))
            with open(anno_file_path, 'r') as reader:
                anno = json.load(reader)
            for record in anno:
                image_name = record['image_id'] + self.img_ext
                image_path = os.path.join(set_dir, imgdir_name, image_name)
                kp_set = record['keypoint_annotations']
                box_set = record['human_annotations']
                self._handle_image(image_path, kp_set, box_set)

        logging.info('finished load AI CH keypoint data, total {} samples'.format(len(self)))

    def _ai_ch_to_lsp(self, pts):
        kp_map = [8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 13, 12]
        pts = np.array(pts, dtype = np.float).reshape(14, 3).copy()
        pts[:, 2] = (3.0 - pts[:, 2]) / 2.0
        return pts[kp_map].copy()


    def _handle_image(self, image_path, kp_set, box_set):
        assert len(kp_set) == len(box_set)
        for key in kp_set.keys():
            kps = kp_set[key]
            box = box_set[key]
            self._handle_sample(key, image_path, kps, [ [box[0], box[1]], [box[2], box[3]] ], box_set)

    def _handle_sample(self, key, image_path, pts, box, boxs):
        def _collect_box(key, boxs):
            r = []
            for k, v in boxs.items():
                if k == key:
                    continue
                r.append([[v[0],v[1]], [v[2],v[3]]])
            return r

        def _collide_heavily(box, boxs):
            for it in boxs:
                if get_rectangle_intersect_ratio(box[0], box[1], it[0], it[1]) > self.max_intersec_ratio:
                    return True
            return False
        pts = self._ai_ch_to_lsp(pts)[self.joint_mapper]
        pts[self.joint_mapper==-1] = -2.
        valid_pt_cound = np.sum(pts[self.joint_mapper!=-1, 2])
        if valid_pt_cound < self.min_pts_required:
            return

        if image_path in self.kp2ds:
            self.kp2ds[image_path].append(pts)
        else:
            self.kp2ds[image_path] = [pts]

    def __len__(self):
        return len(self.file_paths)//self.compress_length

    def get_image_info(self,index):
        index = index*self.compress_length + random.randint(0,self.compress_length-1)
        img_name = self.file_paths[index%len(self.file_paths)]
        kp2ds = self.kp2ds[img_name].copy()
        imgpath = os.path.join(self.data_folder,img_name)
        image = cv2.imread(imgpath)[:,:,::-1]
        
        valid_mask_2d = np.array([[True,False,False] for _ in range(len(kp2ds))])
        valid_mask_3d = np.array([self.default_valid_mask_3d for _ in range(len(kp2ds))])
        
        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,\
                'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d,\
                'kp3ds': None, 'params': None, 'img_size': image.shape[:2], 'ds': 'aich'}
         
        return img_info



if __name__ == '__main__':
    dataset = AICH(train_flag=True)
    test_dataset(dataset)
    print('Done')
