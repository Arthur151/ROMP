
import sys, os
from collections import OrderedDict
from datasets.image_base import *
from datasets.base import Base_Classes, Test_Funcs
from utils.rotation_transform import angle_axis_to_rotation_matrix

default_mode = args().video_loading_mode if args().video else args().image_loading_mode

invalida_detection_seqs = ['pano-slam_dunk-0-0-0', 'pano-slam_dunk-1-0-0', 'pano-slam_dunk-2-0-0', 'pano-slam_dunk-4-0-0']

def DynaCamRotation(base_class=default_mode):
    class DynaCamRotation(Base_Classes[base_class]):
        def __init__(self, train_flag=True, split='train', load_entire_sequence=False, regress_smpl=True,**kwargs):
            super(DynaCamRotation, self).__init__(train_flag, regress_smpl=regress_smpl,
                load_entire_sequence=load_entire_sequence, dynamic_augment=False)
            self.split = split
            self.prepare_annots()
            if base_class == 'video_relative':
                self.video_clip_ids = self.prepare_video_clips()
            
            # data augmentation, rotation, flip and crop would make the solving world trans wrong !!!!! Must be set to False
            self.train_flag = False 
            self.shuffle_mode = False
            
            logging.info('DynaCamRotation dataset has been loaded, total {} samples, contains {} IDs'.format(len(self), self.ID_num))
        
        def prepare_annots(self):
            self.root = os.path.join(self.data_folder,"DynaCam")
            annots_file_path = os.path.join(self.root, 'annotations', f'panorama_{self.split}.npz')
            annotations = np.load(annots_file_path, allow_pickle=True)
            self.annots = annotations['annots'][()]
            self.ID_num, self.sequence_dict = self.annots['ID_num'], self.annots['sequence_dict']
            self.sequence_dict = OrderedDict(self.sequence_dict)
            self.file_paths, self.sequence_ids, self.sid_video_name = [], [], []
            for sid, video_name in enumerate(self.sequence_dict):
                self.sequence_ids.append([])
                for cid, fid in enumerate(self.sequence_dict[video_name]):
                    self.file_paths.append([sid, cid, fid,os.path.join(video_name,'{:06d}.jpg'.format(fid))])
                    self.sequence_ids[sid].append(len(self.file_paths)-1)
                self.sid_video_name.append(video_name)

            if self.regress_smpl:
                self.smplr = SMPLR(use_gender=False)
                self.root_inds = None
            
            self.kp2d_mapper = constants.joint_mapping(constants.OpenPose_25,constants.SMPL_ALL_44)

        def get_image_info(self, index):
            sid, cid, fid, img_name = self.file_paths[index%len(self.file_paths)]
            seq_name = self.sid_video_name[sid]
            frame_id = np.where(np.array(self.sequence_dict[seq_name])==fid)[0][0]
            end_frame_flag = frame_id == (len(self.sequence_ids[sid])-1)
            imgpath = self._get_image_path(img_name)
            if not os.path.exists(imgpath):
                print(seq_name, imgpath)
            image = cv2.imread(imgpath)[:,:,::-1]

            track_ids = np.array(self.annots[seq_name]['person_id'])
            subject_num = len(track_ids)
            #print(self.annots[seq_name]['poses'].shape, cid, fid, len(self.sequence_dict[seq_name]), self.sequence_dict[seq_name])
            intrinsics = self.annots[seq_name]['camera_intrinsics'][cid]
            extrinsics = self.annots[seq_name]['camera_extrinsics_aligned'][cid]

            kp2ds, valid_mask_2d, valid_mask_3d = [], [], []
            params = np.ones((subject_num, 66+10)) * -10
            kp3ds = np.ones((subject_num, args().joint_num, 3), dtype=np.float32) * -2. 
            detecting_all_people = seq_name not in invalida_detection_seqs
            for subject_id in range(len(track_ids)):
                joint = self.annots[seq_name]['kp2ds'][subject_id, cid]
                joint[joint[:,2]<0.1] = -2.
                joint = joint[self.kp2d_mapper]
                joint[self.kp2d_mapper==-1] = -2
                kp2ds.append(joint)
                valid_mask_2d.append([True,True,detecting_all_people])
                valid_mask_3d.append(copy.deepcopy(self.default_valid_mask_3d))

                theta = self.annots[seq_name]['poses'][subject_id, cid].reshape(-1)
                beta = self.annots[seq_name]['betas'][subject_id, cid]
                params[subject_id] = np.array([np.concatenate([theta[:66], beta])])
                valid_mask_3d[subject_id][:4] = True # to supervise the world body rotation
                if self.regress_smpl:
                    _, kp3ds[subject_id] = self.smplr(theta, beta)
            valid_mask_2d, valid_mask_3d = np.array(valid_mask_2d), np.array(valid_mask_3d)
            
            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': track_ids,\
                    'vmask_2d': valid_mask_2d, 'vmask_3d': valid_mask_3d, \
                    'kp3ds': kp3ds, 'params': params, 'root_trans': None, 'verts': None, 
                    'camMats': intrinsics, 'camPoses': extrinsics, 'is_static_cam': False,
                    'precise_kp3d_mask':np.zeros((len(kp2ds),1),dtype=np.bool_),\
                    'img_size': image.shape[:2], 'ds': 'DynaCam'}
            if base_class == 'video_relative':
                img_info.update({'seq_info':[sid, frame_id, end_frame_flag]})
            
            world_body_rots = np.array([self.annots[seq_name]['world_grots_aligned'][subject_id, cid] for subject_id in range(len(track_ids))])
            world_body_trans = np.array([self.annots[seq_name]['world_trans_aligned'][subject_id, cid] for subject_id in range(len(track_ids))])
            img_info.update({'world_grots_trans': [world_body_rots, world_body_trans]})
            
            return img_info

        def _get_image_path(self, file_name):
            images_dir = os.path.join(self.root, 'video_frames',f'panorama_{self.split}', file_name)
            return images_dir

    return DynaCamRotation

def _calc_rotation_matrix_from_pitch_yaw(pitch, yaw):
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(pitch))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(yaw))
    R = R2 @ R1
    return R.T

def _derive_world_cam_from_camera_rotation(self, smpl_params, full_kp2ds, kp3ds, cam_mask, first_frame_camera_rotation, camera_rotation):
    world_cam_mask = cam_mask.clone()
    world_cam_params = torch.ones(self.max_person,3,dtype=torch.float32)*-2
    world_trans = torch.ones(self.max_person,3,dtype=torch.float32)*-2
    world_global_rots = torch.ones(self.max_person,3,dtype=torch.float32)*-10
    
    if cam_mask.sum() == 0:
        return world_global_rots, world_cam_params, world_cam_mask, world_trans, None
    
    cf_fov = camera_rotation[0]
    fov = 1./np.tan(np.radians(cf_fov / 2))
    # TODO: dig which one is the correct one.
    #delta_pitch_yaw = first_frame_camera_rotation[1:]-camera_rotation[1:] 
    delta_pitch_yaw_roll = camera_rotation[1:] - first_frame_camera_rotation[1:]

    body_rots_cam = angle_axis_to_rotation_matrix(smpl_params[cam_mask][:,:3]).numpy()
    body_trans_cam, _ = self.solving_trans3D(full_kp2ds[cam_mask], kp3ds[cam_mask], fov)
    body_R_in_world, body_T_in_world, cam_RT = convert_camera2world_RT(body_rots_cam, body_trans_cam, fov, delta_pitch_yaw_roll)

    #self.convert_body_T2default_fov()
    world_cam_params[world_cam_mask] = torch.from_numpy(normalize_trans_to_cam_params(body_T_in_world)).float()
    world_trans[world_cam_mask] = torch.from_numpy(body_T_in_world).float()
    world_global_rots[world_cam_mask] = body_R_in_world

    return world_global_rots, world_cam_params, world_cam_mask, world_trans, cam_RT

def process_world_annots(self, world_annots_file_path, vis_global=False, target_seq=None, least_person_num = 1):
    # least_person_num # to control to show the samples with at least N people labeled. 
    self.world_annots = {}
    seq_inds = np.array(self.sequence_ids)
    if target_seq is not None:
        seq_inds = target_seq
    
    for sid, seq_image_inds in enumerate(seq_inds):
        first_frame_ind = seq_image_inds[0]
        _, _, img_name = self.file_paths[first_frame_ind]
        first_frame_camera_rotation = np.array(self.annots[img_name][3])

        image_list = []
        cam_Ks = []
        cam_Rts = []
        smpl_poses, smpl_betas, global_trans, global_orient = [], [], [], []

        print(sid, img_name, seq_image_inds)
        for frame_id, image_ind in enumerate(seq_image_inds):
            valid_masks = np.zeros((self.max_person, 8), dtype=np.bool_)
            img_name = self.file_paths[image_ind][2]
            info = self.get_image_info(image_ind)

            image_h, image_w = info['image'].shape[:2]

            img_info = process_image(info['image'], info['kp2ds'], is_pose2d=info['vmask_2d'][:,0])
            image, image_wbg, full_kps, offsets = img_info

            centermap, person_centers, full_kp2ds, used_person_inds, valid_masks[:,0], bboxes_hw_norm, heatmap, AE_joints = \
                self.process_kp2ds_bboxes(full_kps, img_shape=image.shape, is_pose2d=info['vmask_2d'][:,0])
            dst_image, org_image = self.prepare_image(image, image_wbg)

            # valid mask of 3D pose, smpl root rot, smpl pose param, smpl shape param, global translation
            kp3d, valid_masks[:,1] = self.process_kp3ds(info['kp3ds'], used_person_inds, info['ds'], \
                valid_mask_kp3ds=info['vmask_3d'][:, 0])
            params, valid_masks[:,3:6] = self.process_smpl_params(info['params'], used_person_inds, \
                valid_mask_smpl=info['vmask_3d'][:, 1:4])

            default_fovs = torch.Tensor(np.array(args().focal_length / 256))
            gt_fovs = torch.Tensor(np.array(max(image_h, image_w) / 2 * 1./np.tan(np.radians(info['camera_rotation'][0]/2))))
            root_trans, cam_params, cam_mask = self._calc_normed_cam_params_(full_kp2ds, kp3d, valid_masks[:, 1], info['ds'], fovs=gt_fovs)
            params = torch.from_numpy(params).float()
            world_global_rots, world_cam_params, world_cam_mask, world_root_trans, cam_RT = self._derive_world_cam_from_camera_rotation(params, full_kp2ds, kp3d, \
                                    cam_mask, first_frame_camera_rotation, info['camera_rotation'])
            
            valid_cam_ids = torch.where(world_cam_mask)[0].numpy()
            used_person_inds = np.array(used_person_inds)
            
            if world_cam_mask.sum()<least_person_num:
                print('lacking valid person')
                continue
            
            used_person_inds = used_person_inds[valid_cam_ids]
            track_ids = info['track_ids'][used_person_inds]

            bodyINworld = (world_global_rots[world_cam_mask], world_root_trans[world_cam_mask])
            self.world_annots[img_name] = [bodyINworld, used_person_inds, track_ids, cam_RT]
            #print(img_name, self.world_annots[img_name])
            
            if vis_global:
                image_list.append(info['imgpath'])
                fy = max(image_h, image_w) / 2 * 1./np.tan(np.radians(info['camera_rotation'][0]/2))
                fx = max(image_h, image_w) / 2 * 1./np.tan(np.radians(info['camera_rotation'][0]/2))
                cam_K = np.array([[fx, 0, image_w / 2, 0], [0, fy, image_h / 2, 0], [0, 0, 1, 0]], dtype=np.float32)
                cam_Ks.append(cam_K)
                cam_Rts.append(cam_RT)
                
                global_trans.append(world_root_trans[world_cam_mask])
                global_orient.append(world_global_rots[world_cam_mask])

                smpl_pose = params[world_cam_mask][:,3:66]
                smpl_beta = params[world_cam_mask][:,-10:].numpy()
                smpl_pose = torch.cat([smpl_pose, torch.zeros_like(smpl_pose)[:,:6]],-1).numpy()
                smpl_poses.append(smpl_pose)
                smpl_betas.append(smpl_beta)
        
        if vis_global:
            max_person_num = np.array([len(i) for i in smpl_poses]).min()
            smpl_poses = [i[:max_person_num] for i in smpl_poses]
            smpl_betas = [i[:max_person_num] for i in smpl_betas]
            global_trans = [i[:max_person_num] for i in global_trans]
            global_orient = [i[:max_person_num] for i in global_orient]

            smpl_poses = np.stack(smpl_poses, 0)
            smpl_betas = np.stack(smpl_betas, 0)
            global_trans = np.stack(global_trans, 0)
            global_orient = np.stack(global_orient, 0)

            cam_Rts = np.stack(cam_Rts, 0)
            cam_Ks = np.stack(cam_Ks, 0)

            vis_global_view(smpl_poses, smpl_betas, global_trans, global_orient, cam_Rts, cam_Ks, image_list)
 
    np.savez(world_annots_file_path, annot=self.world_annots)
    return self.world_annots

def vis_global_view(smpl_poses, smpl_betas, global_trans, global_orient, cam_Rts, cam_Ks, image_list):
    from visualization.call_aitviewer import GlobalViewer
    viewer_cfgs_update = {'fps':25, 'playback_fps':25.0}
    global_viewer = GlobalViewer(viewer_cfgs_update=viewer_cfgs_update)

    subj_num = smpl_poses.shape[1]
    for subj_ind in range(subj_num):
        gtrans, grots = copy.deepcopy(global_trans[:,subj_ind]), copy.deepcopy(global_orient[:,subj_ind])
        global_viewer.add_smpl_sequence2scene(smpl_poses[:,subj_ind], smpl_betas[:,subj_ind], gtrans, grots)
    
    mean_cam_position = cam_Rts[:, :3, 3].mean(0)
    mean_subj_position = global_trans.mean(0).mean(0)
    world2dynamic_div_dynamic2people = 1.2
    dynamic2people_distance = np.linalg.norm(global_trans, ord=2, axis=-1).max()+3
    
    global_viewer.add_camera2scene(cam_Ks, cam_Rts)
    global_viewer.add_dynamic_image2scene(image_list, distance=dynamic2people_distance) #
    
    cam_position = mean_cam_position + world2dynamic_div_dynamic2people * (mean_cam_position - mean_subj_position)
    cam_target = mean_subj_position
    #cam_up = np.array([1,0,0]) #cam_Rts[0, 1, :3] # -self.current_Rt[1, :3]

    # Set initial camera position and target
    # in ait-viewer z is up, x is right
    global_viewer.viewer.scene.camera.position = cam_position[[0,2,1]] # np.array((0.0, 2, 0))
    global_viewer.viewer.scene.camera.target = cam_target[[0,2,1]] # np.array((0, 0, -2))
    #global_viewer.viewer.scene.camera.cam_up= np.array((0, 1, 0)) can't roll camera

    # Viewer settings
    global_viewer.viewer.scene.floor.enabled = False
    global_viewer.viewer.scene.fps = 30.0
    global_viewer.viewer.playback_fps = 30.0
    global_viewer.viewer.shadows_enabled = False
    global_viewer.viewer.auto_set_camera_target = False

    global_viewer.viewer.run() # cam_up=cam_up cam_target=cam_target,  cam_position=cam_position, 


def load_pkl_func(path_target):
    with open(path_target, 'rb') as f:
        data = pickle.load(f)
    return data

test_seq_names = ['pano-gym_0-0-1', 'pano-gym_0-0-2', 'pano-gym_0-1-1', 
    'pano-gym2_0-2-0', 'pano-gym2_0-3-0', 'pano-gym4_0-3-0', 'pano-gym4_0-4-0', 'pano-gym5_0-0-0',
    'pano-boat_0-3-0', 'pano-boat_0-3-1',
    'pano-fitness_360-2-0', 'pano-fitness_360-2-1',
    'pano-dance_0-3-1', 'pano-dance_0-0-0', 'pano-dance_0-0-1', 'pano-dance_0-5-1', 'pano-dance_0-5-0', 'pano-dance_0-5-2', 'pano-dance_0-2-0',
    'pano-home2_0-0-1', 'pano-home2_0-0-0', 'pano-home2_0-2-1', 'pano-home2_0-2-0', 'pano-home_0-0-0',
    'pano-slam_dunk-4-0', 'pano-slam_dunk-4-1',]

def pack_annots(root_dir, seq_source_list, annots_file_path, split='train'):
    print('Packing annotations of Human Trajectories in Dynamic Camera datasets, panorama part')
    annots, sequence_dict, person_ids = {}, {}, {}

    for sid, seq_source in enumerate(seq_source_list):
        id_cache = sid * 10000
        seq_root_dir = os.path.join(root_dir, seq_source)
        for annot_path in glob.glob(os.path.join(seq_root_dir, '2Dposes', '*.npz')):
            seq_annots = np.load(annot_path, allow_pickle=True)['annots'][()]
            org_frame_inds = np.load(annot_path, allow_pickle=True)['org_frame_ids'][()]
            video_name = os.path.splitext(os.path.basename(annot_path))[0]
            if split=='train' and video_name in test_seq_names:
                continue
            if split=='val' and video_name not in test_seq_names:
                continue

            image_paths = sorted(glob.glob(os.path.join(seq_root_dir, 'video_frames', video_name,'*.jpg')))
            camera_motion_path = os.path.join(seq_root_dir, 'org_video_frames', video_name+'.npy')
            camera_motions = np.load(camera_motion_path)

            for frame_id, image_path in enumerate(image_paths):
                annot = seq_annots[frame_id]
                kp2ds, bboxes, track_ids = [], [], []
                for i, obj in enumerate(annot):
                    track_ids.append(obj['track_id'])
                    kp2ds.append(obj['keypoints'])
                    bboxes.append(obj['bbox'])
                if len(kp2ds) == 0:
                    continue
                
                if video_name not in sequence_dict:
                    sequence_dict[video_name], person_ids[video_name] = [], {}
                sequence_dict[video_name].append(frame_id)

                img_person_ids = []
                for track_id in track_ids:
                    if track_id not in person_ids[video_name]:
                        person_ids[video_name][track_id] = id_cache
                        id_cache += 1
                    person_id = person_ids[video_name][track_id]
                    img_person_ids.append(person_id)
                file_name = image_path.replace(image_path.split(video_name)[0], '')

                camera_motion = camera_motions[org_frame_inds[frame_id]-1]
                annots[file_name] = [kp2ds, bboxes, np.array(img_person_ids), camera_motion]

            print('sequence {}, frames {}'.format(video_name, len(annots)))

    for video_name in sequence_dict:
        sequence_dict[video_name] = sorted(sequence_dict[video_name])
        #print(video_name,sequence_dict[video_name])
    np.savez(annots_file_path, annot = annots, sequence_dict=sequence_dict, person_ids={'map_dict':person_ids, 'id_number':id_cache})
    print('Saving annotations to {}'.format(annots_file_path))
    return annots, id_cache, sequence_dict

if __name__ == '__main__':
    datasets = DynaCamRotation(base_class=default_mode)(train_flag=False)
    Test_Funcs[default_mode](datasets, with_smpl=True, vis_global_aitviewer=False)
    print('Done')