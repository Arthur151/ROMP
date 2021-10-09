import sys, os

from dataset.image_base import *

class UP(Image_base):
    def __init__(self,train_flag = True, regress_smpl=True):
        super(UP,self).__init__(train_flag, regress_smpl)
        self.data_folder = os.path.join(self.data_folder,'UP/')
        self.data3d_dir = os.path.join(self.data_folder,'up-3d')
        self.joint_mapper = constants.joint_mapping(constants.LSP_14, constants.SMPL_ALL_54)
        #self.joint3d_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.SMPL_ALL_54)

        self.scale_dir = os.path.join(self.data_folder,'p14_joints/scale_14_500_p14_joints.txt')
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [8, 9], [7, 10]]
        self.multi_mode = False

        self.high_qulity_idx = self.get_high_qulity_idx()
        if self.regress_smpl:
            self.smplr = SMPLR(use_gender=False)
        logging.info('UP dataset total {} samples'.format(len(self)))

    def get_high_qulity_idx(self):
        files = glob.glob(os.path.join(self.data3d_dir,'*_quality_info.txt'))
        high_qulity_idx = []
        for file in files:
            quality = self.read_txt(file)
            data_idx = os.path.basename(file).split('_')[0]
            dataset_info_dir = os.path.join(self.data3d_dir,'{}_dataset_info.txt'.format(data_idx))
            dataset_info = self.read_txt(dataset_info_dir)[0]
            if 'high\n' in quality and dataset_info!='fashionpose':
                high_qulity_idx.append(data_idx)
        return high_qulity_idx

    def read_txt(self,file_path):
        #in 08514_fit_crop_info.txt ,there are 6 number:
        # width height width_crop_start width_crop_end height_crop_start height_crop_end
        f=open(file_path)
        lines = f.readlines()
        if len(lines)!=1:
            print('different crop_fit_info lines of {}:'.format(file_path), len(lines))
        info = lines[0].split(' ')
        return info

    def __len__(self):
        return len(self.high_qulity_idx)

    def get_image_info(self,index):
        index = self.high_qulity_idx[index%len(self.high_qulity_idx)]
        imgpath = os.path.join(self.data3d_dir,'{}_image.png'.format(index))
        image = cv2.imread(imgpath)[:,:,::-1]

        annot_3d_dir = os.path.join(self.data3d_dir,'{}_body.pkl'.format(index))
        annot_3d = self.read_pkl(annot_3d_dir)
        theta,beta,t = annot_3d['pose'][:66],annot_3d['betas'],annot_3d['t']
        params = np.array([np.concatenate([theta, beta])])

        annot_2d_kp_dir = os.path.join(self.data3d_dir,'{}_joints.npy'.format(index))
        kp2ds = self.map_kps(self.read_npy(annot_2d_kp_dir).T,maps=self.joint_mapper)[None]
        kp3ds = self.regress_kp3d_from_smpl(params)

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,\
                'vmask_2d': np.array([[True,False,False]]), 'vmask_3d': np.array([[True,True,True,False]]),\
                'kp3ds': kp3ds, 'params': params, 'img_size': image.shape[:2], 'ds': 'up'}
         
        return img_info

if __name__ == '__main__':
    dataset=UP()
    test_dataset(dataset,with_smpl=True)
    print('Done')