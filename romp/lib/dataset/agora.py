from config import args
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs

default_mode = args().image_loading_mode

def AGORA(base_class=default_mode):
    class AGORA(Base_Classes[base_class]):
        def __init__(self, train_flag=True, split='train',**kwargs):
            super(AGORA, self).__init__(train_flag,False)
            self.data_folder = os.path.join(args().dataset_rootdir,'AGORA/')
            if not os.path.isdir(self.data_folder):
                self.data_folder = '/home/yusun/data_drive/dataset/AGORA'
            self.train_flag=train_flag
            self.split = split

            self.annots_path = os.path.join(self.data_folder,'annots_{}.npz'.format(self.split))
            self.vertex_save_dir = os.path.join(self.data_folder, 'image_vertex_{}'.format(self.split))
            if not os.path.exists(self.annots_path):
                print('packing the annotations into a single file')
                self.annots = pack_data(self.vertex_save_dir, self.data_folder, self.split, self.annots_path)
            else:
                self.load_annots()
            self.file_paths = list(self.annots.keys())
            self.shuffle_mode = args().shuffle_crop_mode
            self.shuffle_ratio = args().shuffle_crop_ratio_3d
            self.multi_mode=True
            self.root_inds = [constants.SMPL_ALL_54['Pelvis_SMPL']]

            logging.info('Loaded AGORA,total {} samples'.format(self.__len__()))

        def load_annots(self):
            self.annots = np.load(self.annots_path, allow_pickle=True)['annots'][()]

        def get_image_info(self,index,total_frame=None):
            imgpath = self.file_paths[index%len(self.file_paths)]
            annots = self.annots[imgpath].copy()
            imgpath = os.path.join(self.data_folder, self.split,imgpath)
            image = cv2.imread(imgpath)[:,:,::-1]
            img_name = os.path.basename(imgpath)
            valid_mask = np.where(np.array([annot['isValid'] for annot in annots]))[0]
            if len(valid_mask) ==0:
                print(img_name, 'lack valid person')
                valid_mask = np.array([0])
            verts = np.load(os.path.join(self.vertex_save_dir, img_name.replace('.png', '.npz')), allow_pickle=True)['verts'][valid_mask]
            annots = [annots[ind] for ind in valid_mask]
            params = np.stack([np.concatenate([np.ones(3)*-10, annot['body_pose'].reshape(-1)[:63], annot['betas'].reshape(-1)[:10]]) for annot in annots])
            kp2ds = np.stack([annot['kp2d'] for annot in annots])
            kp2ds = np.concatenate([kp2ds, np.ones((kp2ds.shape[0], kp2ds.shape[1], 1))],2)
            kp3ds = np.stack([annot['kp3d'] for annot in annots])
            root_trans = kp3ds[:,self.root_inds].mean(1)
            track_ids = np.stack([annot['ID'] for annot in annots])
            camMats = annots[0]['camMats']
            person_num = len(kp2ds)
            
            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': track_ids,\
                    'vmask_2d': np.array([[True,True,True] for _ in range(person_num)]), 'vmask_3d': np.array([[True,False,True,True,True,True] for _ in range(person_num)]),\
                    'kp3ds': kp3ds, 'params': params, 'root_trans': root_trans, 'verts': verts,\
                    'camMats': camMats, 'img_size': image.shape[:2],'ds': 'agora'}

            if 'relative' in base_class:
                properties = np.stack([annot['props'] for annot in annots])
                genders = (properties[:,0]=='female').astype(np.int)
                ages = (properties[:,1]=='kid').astype(np.int)*2
                depth_level = body_type = np.ones_like(ages) * -1
                img_info['depth'] = np.stack([ages, genders, depth_level, body_type],1)
                if ages.sum()>0:
                    img_info['kid_shape_offsets'] = np.array([annot['betas'][0,10] if annot['betas'].shape[-1]==11 else 0 for annot in annots])
            
            return img_info

        def __len__(self):
            return len(self.file_paths)
    return AGORA

def pack_data(vertex_save_dir, data_folder, split, annots_path):
    import pandas
    annots = {}
    smpl_subject_dict, subject_id = {}, 0
    os.makedirs(vertex_save_dir, exist_ok=True)
    all_annot_paths = glob.glob(os.path.join(data_folder, 'CAM2', '{}*_withj2.pkl'.format(split)))
    for af_ind, annot_file in enumerate(all_annot_paths):
        annot = pandas.read_pickle(annot_file)
        annot_dicts = annot.to_dict(orient='records')
        for annot_ind, annot_dict in enumerate(annot_dicts):
            print('{}/{} {}/{}'.format(af_ind, len(all_annot_paths), annot_ind, len(annot_dicts), annot_dict['imgPath']))
            img_annot, img_verts, valid_num = [], [], 0
            pimg_annot = {
                    'cam_locs':np.array([annot_dict['camX'],annot_dict['camY'],annot_dict['camZ'],annot_dict['camYaw']]),
                    'trans':np.array([annot_dict['X'],annot_dict['Y'],annot_dict['Z'],annot_dict['Yaw']]).transpose((1,0)),
                    'props':np.array([annot_dict['gender'],annot_dict['kid'],annot_dict['occlusion'],annot_dict['age'],annot_dict['ethnicity']]),
                    'isValid':annot_dict['isValid'], 'gt_path_smpl':annot_dict['gt_path_smpl'],'gt_path_smplx':annot_dict['gt_path_smplx']
                }
            for ind, smpl_annot_path in enumerate(annot_dict['gt_path_smpl']):
                if annot_dict['isValid'][ind]:
                    valid_num += 1
                subj_annot = {}
                smpl_annot = pandas.read_pickle(os.path.join(data_folder,smpl_annot_path.replace('.obj', '.pkl')))
                subj_annot['body_pose'] = smpl_annot['body_pose'].detach().cpu().numpy()
                subj_annot['betas'] = smpl_annot['betas'].detach().cpu().numpy()
                subj_annot['root_rot'] = smpl_annot['root_pose'].detach().cpu().numpy()
                subj_annot['props'] = [annot_dict['gender'][ind],'kid' if annot_dict['kid'][ind] else 'adult',\
                                    annot_dict['age'][ind],annot_dict['ethnicity'][ind]]
                if annot_dict['gt_path_smpl'][ind].replace('.obj','') not in smpl_subject_dict:
                    smpl_subject_dict[annot_dict['gt_path_smpl'][ind].replace('.obj','')] = subject_id
                    subject_id += 1
                subj_annot['ID'] = smpl_subject_dict[annot_dict['gt_path_smpl'][ind].replace('.obj','')]
                subj_annot['occlusion'] = annot_dict['occlusion'][ind]
                subj_annot['isValid'] = annot_dict['isValid'][ind]
                subj_annot['kp2d'] = annot_dict['gt_joints_2d'][ind]
                subj_annot['kp3d'] = annot_dict['gt_joints_3d'][ind]
                subj_annot['cam_locs'] = pimg_annot['cam_locs']
                subj_annot['smpl_trans'] = pimg_annot['trans'][ind]
                subj_annot['camMats'] = annot_dict['camMats'][ind]
                subj_annot['root_rotMats'] = annot_dict['root_rotMats'][ind]
                img_annot.append(subj_annot)
                #img_verts.append(annot_dict['gt_verts'][ind])
            if valid_num!=0:
                annots[annot_dict['imgPath']] = img_annot
            vertex_save_name = os.path.join(vertex_save_dir,os.path.basename(annot_dict['imgPath']).replace('.png', '.npz'))
            np.savez(vertex_save_name,verts=img_verts)
        np.savez(self.annots_path.replace('.npz','_{}.npz'.format(af_ind)),annots=annots)
    np.savez(annots_path,annots=annots)   
    np.savez(os.path.join(self.data_folder,'subject_IDs_dict_{}.npz'.format(self.split)), subject_ids=smpl_subject_dict) 
    return annots

if __name__ == '__main__':
    #agora = AGORA(base_class=default_mode)(False, split='validation')
    agora = AGORA(base_class=default_mode)(True, split='train')
    Test_Funcs[default_mode](agora, with_3d=True,with_smpl=True,)
