import sys, os

from dataset.image_base import *

class LSP(Image_base):
    def __init__(self,train_flag = True, regress_smpl=True, **kwargs):
        super(LSP,self).__init__(train_flag, regress_smpl)
        self.data_folder = os.path.join(self.data_folder,'lsp/')
        self.joint_mapper = constants.joint_mapping(constants.LSP_14, constants.SMPL_ALL_54)
        self.scale_range = [1.6,1.8]
        self.load_data()
        self.file_paths = list(self.eft_annots.keys())
        self.multi_mode = False
        if self.regress_smpl:
            self.smplr = SMPLR(use_gender=False)
        logging.info('LSP dataset total {} samples'.format(len(self)))

    def load_data(self):
        self.file_paths, self.annots = [], {}
        root_dir = os.path.join(self.data_folder, 'hr-lspet') # 'lsp_original'  'lsp_ext'
        self.img_dir = root_dir # os.path.join(root_dir,'images')
        joints = scio.loadmat(os.path.join(root_dir,'joints.mat'))['joints'].transpose(2,0,1).astype(np.float32)
        img_paths = glob.glob(os.path.join(self.img_dir, '*.png'))
        img_number_list = []
        for img_path, joint in zip(img_paths, joints):
            img_name = os.path.basename(img_path)
            img_number_list.append(int(img_name.split('.png')[0][2:]))
            self.file_paths.append(img_name)
        img_number_list.sort()
        for idx, img_number in enumerate(img_number_list):
            img_name = 'im{:05}.png'.format(img_number)
            self.annots[img_name] = joints[idx]
        
        load_eft_annots_path = os.path.join(root_dir,'eft_annots.npz')
        if os.path.exists(load_eft_annots_path):
            self.eft_annots = np.load(load_eft_annots_path,allow_pickle=True)['annots'][()]
        else:
            self.load_eft_annots(os.path.join(config.project_dir, 'data/eft_fit/LSPet_ver01.json'))
            np.savez(load_eft_annots_path, annots=self.eft_annots)

    def load_eft_annots(self, annot_file_path):
        joint_mapper = constants.joint_mapping(constants.SMPL_24, constants.LSP_14)
        self.eft_annots = {}
        annots = json.load(open(annot_file_path,'r'))['data']
        for idx, eft_data in enumerate(annots):            
            imgFullPath = eft_data['imageName']
            imgName = os.path.basename(imgFullPath)
            kp2d_gt = self.annots[imgName]

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


    def get_image_info(self,index):
        img_name = self.file_paths[index%len(self.file_paths)]
        imgpath = os.path.join(self.img_dir, img_name)
        image = cv2.imread(imgpath)[:,:,::-1]

        kp2ds = self.map_kps(self.annots[img_name], self.joint_mapper)[None]

        params, valid_mask_3d = [], np.array([self.default_valid_mask_3d])
        if img_name in self.eft_annots and self.use_eft:
            eft_annot = self.eft_annots[img_name]
            bbox_center, pose, betas = eft_annot[0]
            params = np.array([np.concatenate([pose[:66], betas])])
            valid_mask_3d[0,:4] = np.array([self.regress_smpl, True, True, False])
            
        kp3ds = self.regress_kp3d_from_smpl(params)
        valid_mask_2d = np.array([[True,False,False]])

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,\
                'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d,\
                'kp3ds': kp3ds, 'params': params, 'img_size': image.shape[:2], 'ds': 'lsp'}
         
        return img_info

if __name__ == '__main__':
    dataset=LSP(regress_smpl=True)
    test_dataset(dataset,with_smpl=True)
    print('Done')