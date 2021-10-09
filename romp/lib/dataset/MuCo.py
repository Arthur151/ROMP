import sys, os

from dataset.image_base import *

class MuCo(Image_base):
    def __init__(self,train_flag=True, mode='augmented', **kwargs):
        super(MuCo,self).__init__(train_flag)
        self.data_folder = os.path.join(self.data_folder,'MuCo/')
        self.min_pts_required = 5
        self.collision_factor = 0.3
        self.scale_range = [1.8,2.6]
        self.compress_length = 5
        self.mode = mode
        if self.mode=='augmented':
            annots_file_path = os.path.join(self.data_folder, 'annots_augmented.npz')
            self.image_folder = self.data_folder
        else:
            annots_file_path = os.path.join(self.data_folder, 'annots.npz')
            self.image_folder = os.path.join(self.data_folder, 'images')
        self.shuffle_mode = args().shuffle_crop_mode
        self.shuffle_ratio = args().shuffle_crop_ratio_3d
        self.scale_range = [1.5,2.0]
        if os.path.exists(annots_file_path):
            self.annots = np.load(annots_file_path,allow_pickle=True)['annots'][()]
        else:
            if self.mode=='augmented':
                self.pack_data_augmented(annots_file_path)
            else:
                self.pack_data(annots_file_path)

        self.file_paths = list(self.annots.keys())
        self.kp2d_mapper = constants.joint_mapping(constants.MuCo_21, constants.SMPL_ALL_54)
        self.kp3d_mapper = constants.joint_mapping(constants.MuCo_21, constants.SMPL_ALL_54)
        self.root_inds = [constants.SMPL_ALL_54['Pelvis']]
        logging.info('MuCo dataset total {} samples, loading mode {}'.format(self.__len__(), self.mode))

    def __len__(self):
        if self.train_flag:
            return len(self.file_paths)//self.compress_length
        else:
            return len(self.file_paths)

    def get_image_info(self, index):
        if self.train_flag:
            index = index*self.compress_length + random.randint(0,self.compress_length-1)
        img_name = self.file_paths[index%len(self.file_paths)]
        imgpath = os.path.join(self.image_folder,img_name)
        while not os.path.exists(imgpath):
            img_name = self.file_paths[np.random.randint(len(self))]
            imgpath = os.path.join(self.image_folder,img_name)
        image = cv2.imread(imgpath)[:,:,::-1]

        kp2ds, valid_mask_2d, valid_mask_3d, kp3ds = [], [], [], []
        for kp2d, kp3d in zip(self.annots[img_name][0], self.annots[img_name][1]):
            kp2ds.append(self.map_kps(kp2d,maps=self.kp2d_mapper))
            kp3ds.append(self.map_kps(kp3d/1000.,maps=self.kp3d_mapper))
            valid_mask_2d.append([True,False,True])
            valid_mask_3d.append([True,False,False,False])

        kp2ds, kp3ds = np.array(kp2ds), np.array(kp3ds)
        root_trans = kp3ds[:,self.root_inds].mean(1)
        valid_masks = np.array([self._check_kp3d_visible_parts_(kp3d) for kp3d in kp3ds])
        kp3ds -= root_trans[:,None]
        kp3ds[~valid_masks] = -2.
        
        f,c = self.annots[img_name][2]
        camMats = np.array([[f[0],0,c[0]],[0,f[1],c[1]],[0,0,1]])

        vis_masks = []
        for kp2d in kp2ds:
            vis_masks.append(_check_visible(kp2d,get_mask=True))
        kp2ds = np.concatenate([kp2ds, np.array(vis_masks)[:,:,None]],2)

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,\
                'vmask_2d': np.array(valid_mask_2d), 'vmask_3d': np.array(valid_mask_3d),\
                'kp3ds': kp3ds, 'params': None, 'camMats': camMats, 'img_size': image.shape[:2], 'ds': 'muco'}
         
        return img_info

    def pack_data(self,annots_file_path):
        self.annots = {}
        annots_files = glob.glob(os.path.join(self.data_folder, 'annotations','*.mat'))
        for annots_file in annots_files:
            annots = scio.loadmat(annots_file)
            image_names = annots['img_names'][0]
            kp3ds = annots['joint_loc3'].transpose((3,2,1,0))
            kp2ds = annots['joint_loc2'].transpose((3,2,1,0))
            for img_name, kp2d, kp3d in zip(image_names, kp2ds, kp3ds):
                self.annots[img_name[0]] = [kp2d,kp3d]
            
        np.savez(annots_file_path, annots=self.annots)
        logging.info('MuCo data annotations packed')

    def pack_data_augmented(self, annots_file_path):
        from pycocotools.coco import COCO
        self.annots = {}
        db = COCO(os.path.join(self.data_folder, 'MuCo-3DHP.json'))
        data = []
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_id = img["id"]
            img_width, img_height = img['width'], img['height']
            imgname = img['file_name']
            if 'unaugmented' in imgname:
                continue
            img_path = os.path.join(self.data_folder, 'augmented_set', imgname)
            f = img["f"]
            c = img["c"]
            intrinsic = np.array([f,c])

            # crop the closest person to the camera
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)
            kp2d = np.array([ann['keypoints_img'] for ann in anns])
            kp3d = np.array([ann['keypoints_cam'] for ann in anns])

            self.annots[imgname] = [kp2d,kp3d,intrinsic]
        np.savez(annots_file_path, annots=self.annots)
        print('MuCo augmented data annotations packed')


    def get_image_name(self,video_name, frame_id):
        return video_name.strip('.avi').replace('/imageSequence','').replace('/','_')+'_F{}.jpg'.format(frame_id)

def _check_visible(joints, w=2048, h=2048, get_mask=False):
    visibility = True
    # check that all joints are visible
    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
    ok_pts = np.logical_and(x_in, y_in)
    if np.sum(ok_pts) < 16:
        visibility=False
    if get_mask:
        return ok_pts
    return visibility

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts

if __name__ == '__main__':
    dataset=MuCo(train_flag=True)
    test_dataset(dataset)
    print('Done')

"""
 ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist',  #5
 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', #10
 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', #15
 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
"""