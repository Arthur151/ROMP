import sys, os

from dataset.image_base import *

class MPII(Image_base):
    def __init__(self, train_flag=True, regress_smpl=True, **kwargs):
        super(MPII,self).__init__(train_flag,regress_smpl)
        self.const_box = [np.array([0,0]),np.array([256,256])]
        self.empty_kps = np.ones((6,3))*-2

        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]
        self.data_folder = os.path.join(self.data_folder,'mpii/')
        self.image_set='train' if self.train_flag else 'valid'
        self._get_db()

        load_eft_annots_path = os.path.join(self.data_folder,'eft_annots.npz')
        if os.path.exists(load_eft_annots_path):
            self.eft_annots = np.load(load_eft_annots_path,allow_pickle=True)['annots'][()]
        else:
            self.load_eft_annots(os.path.join(config.project_dir, 'data/eft_fit/MPII_ver01.json'))
            np.savez(load_eft_annots_path, annots=self.eft_annots)
        self.file_paths = list(self.eft_annots.keys())
        self.joint_mapper = constants.joint_mapping(constants.MPII_16, constants.SMPL_ALL_54)
        if self.regress_smpl:
            self.smplr = SMPLR(use_gender=False)
        logging.info('Loaded MPII data total {} samples'.format(self.__len__()))

    def get_image_info(self,index):
        img_name = self.file_paths[index%len(self.file_paths)]
        infos = self.annots[img_name]
        sellected_id = self.sellect_person(infos)
        info = infos[sellected_id]

        imgpath = os.path.join(self.img_dir, img_name)
        image = cv2.imread(imgpath)[:,:,::-1]

        kp2ds, valid_mask_2d, valid_mask_3d, params = [], [], [], []
        for info in infos:
            kp2ds.append(self.process_single_person_joints(info['joints']))
            valid_mask_2d.append([True,False,True])
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
            for inds, kp2d in enumerate(kp2ds):
                center_i = self._calc_center_(kp2d)
                center_dist = np.linalg.norm(bbox_center_list-center_i[:2][None], axis=-1)
                closet_idx = np.argmin(center_dist)
                matched_param = np.concatenate([pose_list[closet_idx], betas_list[closet_idx]])
                params.append(matched_param)
                valid_mask_3d[inds, :4] = np.array([self.regress_smpl, True, True, False])

        kp2ds, params = np.array(kp2ds), np.array(params)
        kp3ds = self.regress_kp3d_from_smpl(params)

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,\
                'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d,\
                'kp3ds': kp3ds, 'params': params, 'img_size': image.shape[:2], 'ds': 'mpii'}
         
        return img_info


    def load_eft_annots(self, annot_file_path):
        self.eft_annots = {}
        annots = json.load(open(annot_file_path,'r'))['data']
        for eft_data in annots:
            #Get raw image path
            imgFullPath = eft_data['imageName']
            imgName = os.path.basename(imgFullPath)
            bbox_scale = eft_data['bbox_scale']
            bbox_center = eft_data['bbox_center']
            pred_camera = np.array(eft_data['parm_cam'])
            pred_betas = np.reshape(np.array( eft_data['parm_shape'], dtype=np.float32), (10) )     #(10,)
            pred_pose_rotmat = np.reshape( np.array( eft_data['parm_pose'], dtype=np.float32), (24,3,3)  )        #(24,3,3)
            pred_pose = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(pred_pose_rotmat)).reshape(-1)
            if imgName not in self.eft_annots:
                self.eft_annots[imgName] = []
            self.eft_annots[imgName].append([bbox_center, pred_pose, pred_betas])
        logging.info('EFT pseudo-label contains annotations for {} samples'.format(len(self.eft_annots)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.data_folder,'annot',self.image_set+'.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        file_paths, self.annots = [], {}
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float32)
            s = np.array([a['scale'], a['scale']], dtype=np.float32)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1
            joints = np.array(a['joints'])
            joints[:, 0:2] = joints[:, 0:2] - 1
            assert len(joints) == 16, 'joint num diff: {} vs {}'.format(len(joints),16)

            joints_vis = np.zeros((16,3),dtype=np.float32)
            joints_vis[:, 0:2] = joints[:, 0:2]
            joints_vis[:, -1] = np.array(a['joints_vis'])

            imgpath = image_name
            annot = {'center': c,'scale': s,'joints': joints_vis}
            if imgpath in self.annots:
                self.annots[imgpath].append(annot)
            else:
                self.annots[imgpath] = [annot]
                file_paths.append(imgpath)

        self.img_dir = os.path.join(self.data_folder, 'images')
        self.file_paths = file_paths
        print('remove the same {}/{}'.format(len(self.file_paths), len(file_paths)))

    def sellect_person(self, infos):
        if len(infos)==1:
            return 0
        else:
            return np.random.randint(len(infos))
            #return np.argmax(kps[:,:,-1].sum(-1))

    def process_single_person_joints(self,joint_info):
        joints = joint_info[:,0:2]
        joints_vis = joint_info[:,-1]
        joints[joints_vis<0.05] = -2. 
        kp2d = np.concatenate([joints[:,0:2],joints_vis[:,None]],1)[self.joint_mapper]
        kp2d[self.joint_mapper==-1] = -2.
        return kp2d

    def evaluate(self, preds, output_dir=None, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            scio.savemat(pred_file, mdict={'preds': preds})

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(self.data_folder,
                               'annot',
                               'gt_{}.mat'.format('valid'))
        gt_dict = scio.loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

if __name__ == '__main__':
    dataset = MPII(train_flag=True,regress_smpl=True)
    test_dataset(dataset,with_smpl=True)
    print('Done')