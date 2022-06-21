from pycocotools.coco import COCO
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs

default_mode = args().image_loading_mode

green_frames = ['160422_haggling1-00_16_00002945.jpg',
'160422_haggling1-00_16_00002946.jpg',
'160422_haggling1-00_16_00002947.jpg',
'160422_haggling1-00_16_00002948.jpg',
'160422_haggling1-00_16_00002949.jpg',
'160422_haggling1-00_16_00002950.jpg',
'160422_haggling1-00_16_00002951.jpg',
'160422_haggling1-00_16_00002952.jpg',
'160422_haggling1-00_16_00002953.jpg',
'160422_haggling1-00_16_00002954.jpg',
'160422_haggling1-00_30_00001402.jpg',
'160422_haggling1-00_30_00001403.jpg',
'160422_haggling1-00_30_00001404.jpg',
'160422_haggling1-00_30_00001405.jpg',
'160422_haggling1-00_30_00001406.jpg',
'160422_haggling1-00_30_00001407.jpg',
'160422_haggling1-00_30_00001408.jpg',
'160422_haggling1-00_30_00001409.jpg',
'160422_haggling1-00_30_00001410.jpg',
'160422_haggling1-00_30_00001411.jpg',
'160422_haggling1-00_30_00001412.jpg',
'160422_haggling1-00_30_00001414.jpg']

hard_seq = []

def CMU_Panoptic_eval(base_class=default_mode):
    class CMU_Panoptic_eval(Base_Classes[base_class]):
        def __init__(self,train_flag=True, split='test',joint_format='h36m', load_entire_sequence=False,**kwargs):
            super(CMU_Panoptic_eval,self).__init__(train_flag, load_entire_sequence=load_entire_sequence)
            self.data_folder = os.path.join(self.data_folder,'cmu_panoptic/')
            if not os.path.isdir(self.data_folder):
                self.data_folder = '/home/yusun/data_drive/dataset/cmu_panoptic_CRMH'
            self.min_pts_required = 5
            self.split = split
            self.test2val_sample_ratio = 10
            self.J24_TO_H36M = np.array([14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6])
            self.H36M_TO_LSP = self.J24_TO_H36M[np.array([6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10])]
            self.annots_folder = os.path.join(self.data_folder,'panoptic_annot')
            self.load_annots()

            self.image_folder = os.path.join(self.data_folder,'images/')            
            self.joint_mapper = constants.joint_mapping(constants.LSP_14, constants.SMPL_ALL_54)
            if joint_format=='lsp14':
                self.kp3d_mapper = self.H36M_TO_LSP
            elif joint_format=='h36m':
                # centerHMR v1 use h36m keypoints
                self.kp3d_mapper = self.J24_TO_H36M
            self.root_inds = None#[constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']]

            logging.info('CMU Panoptic dataset total {} samples, loading {} split'.format(self.__len__(), self.split))
        
        def load_annots(self):
            self.annots = {}
            for annots_file_name in os.listdir(self.annots_folder):
                ann_file = os.path.join(self.annots_folder, annots_file_name)
                with open(ann_file, 'rb') as f:
                    img_infos = pickle.load(f)
                for img_info in img_infos:
                    img_path = img_info['filename'].split('/')
                    img_name = img_path[1]+'-'+img_path[-1].replace('.png', '.jpg')
                    self.annots[img_name] = {}
                    self.annots[img_name] = img_info
            self.file_paths = list(self.annots.keys())

        def determine_visible_person(self, kp2ds, width, height):
            visible_person_id,kp2d_vis = [],[]
            for person_id,kp2d in enumerate(kp2ds):
                visible_kps_mask = np.logical_and(np.logical_and(0<kp2d[:,0],kp2d[:,0]<width),np.logical_and(0<kp2d[:,1],kp2d[:,1]<height,kp2d[:,2]>0))
                if visible_kps_mask.sum()>1:
                    visible_person_id.append(person_id)
                    kp2d_vis.append(np.concatenate([kp2d[:,:2], visible_kps_mask[:,None]],1))
            return np.array(visible_person_id), np.array(kp2d_vis)

        def get_image_info(self, index):
            img_name = self.file_paths[index%len(self.file_paths)]
            imgpath = os.path.join(self.image_folder,img_name)
            image = cv2.imread(imgpath)[:,:,::-1]

            visible_person_id, kp2ds = self.determine_visible_person(self.annots[img_name]['kpts2d'], self.annots[img_name]['width'],self.annots[img_name]['height'])
            kp3ds = self.annots[img_name]['kpts3d'][visible_person_id]
            full_kp2d, kp_3ds, valid_mask_2d, valid_mask_3d = [], [], [], []
            for inds, (kp2d, kp3d) in enumerate(zip(kp2ds, kp3ds)):
                invis_kps = kp2d[:,-1]<0.1
                kp2d *= 1920./832.
                kp2d[invis_kps] = -2.
                kp2d = self.map_kps(kp2d[self.H36M_TO_LSP],maps=self.joint_mapper)
                kp2d[constants.SMPL_ALL_54['Head_top']] = -2.
                full_kp2d.append(kp2d)
                valid_mask_2d.append([True, False, True])
                invis_3dkps = kp3d[:,-1]<0.1
                kp3d = kp3d[:,:3]
                kp3d[invis_3dkps] = -2.
                kp3d = kp3d[self.J24_TO_H36M]
                kp3d[0] -= np.array([0,0.06,0.0])
                kp_3ds.append(kp3d)
                valid_mask_3d.append([True,False,False,False,False,False])

            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': full_kp2d, 'track_ids': None,\
                    'vmask_2d': np.array(valid_mask_2d), 'vmask_3d': np.array(valid_mask_3d),\
                    'kp3ds': kp_3ds, 'params': None, 'root_trans': None, 'verts': None,\
                    'img_size': image.shape[:2], 'ds': 'cmup'}

            return img_info
    return CMU_Panoptic_eval

if __name__ == '__main__':
    dataset=CMU_Panoptic_eval(base_class=default_mode)(train_flag=False)
    Test_Funcs[default_mode](dataset,with_smpl=False)
    print('Done')