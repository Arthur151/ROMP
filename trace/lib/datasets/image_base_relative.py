from datasets.image_base import *
from maps_utils.centermap import _calc_radius_
from utils.rotation_transform import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from utils.projection import perspective_projection_withfovs

class Image_base_relative(Image_base):
    def __init__(self, train_flag=True, regress_smpl = False, **kwargs):
        super(Image_base_relative,self).__init__(train_flag=train_flag, regress_smpl=regress_smpl, **kwargs) #
        self.depth_degree_thresh = [0.36,0.18,0]
        self.regress_smpl = regress_smpl

    def get_item_single_frame(self, index, augment_cfgs=None, return_augcfgs=False, \
                            dynamic_augment_cfgs=None, syn_occlusion=None, first_frame_camera_rotation=None):
        # valid annotation flags for 
        # 0: 2D pose/bounding box(True/False), # 7: detecting all person/front-view person(True/False)
        # 1: 3D pose, 2: subject id, 3: smpl root rot, 4: smpl pose param, 5: smpl shape param, 6: global translation, 7: vertex of SMPL model
        valid_masks = np.zeros((self.max_person, 8), dtype=np.bool_)
        info = self.get_image_info(index)

        if augment_cfgs is None:
            position_augments, pixel_augments = self._calc_augment_confs(info['image'], copy.deepcopy(info['kp2ds']), is_pose2d=copy.deepcopy(info['vmask_2d'][:,0]))
        else:
            position_augments, pixel_augments = augment_cfgs
            if dynamic_augment_cfgs is not None:
                dynamic_position_augments = copy.deepcopy(position_augments)
                position_augments[2] = dynamic_augment_cfgs[0]

                dynamic_img_info = process_image(copy.deepcopy(info['image']), copy.deepcopy(info['kp2ds']), augments=dynamic_position_augments, is_pose2d=info['vmask_2d'][:,0])
                dynamic_image, _ = self.prepare_image(dynamic_img_info[0], dynamic_img_info[1], augments=pixel_augments) 
        
        if args().learn_deocclusion:
            org_kp2ds_dc = copy.deepcopy(info['kp2ds'])
        org_kp2ds_dc = copy.deepcopy(info['kp2ds'])
        
        img_info = process_image(info['image'], info['kp2ds'], augments=position_augments, is_pose2d=info['vmask_2d'][:,0], syn_occlusion=syn_occlusion)
        
        image, image_wbg, full_kps, offsets = img_info
        centermap, person_centers, full_kp2ds, used_person_inds, valid_masks[:,0], bboxes_hw_norm, heatmap, AE_joints = \
                self.process_kp2ds_bboxes(full_kps, img_shape=image.shape, is_pose2d=info['vmask_2d'][:,0])

        if args().learn_cam_with_fbboxes:
            full_body_bboxes = self.process_full_body_bboxes(full_kps, ~info['vmask_2d'][:,0], image.shape, info['ds'])

        if len(info['vmask_2d']) == 0:
            print('empty vmask_2d', info['vmask_2d'], info['imgpath'])
            return self.resample()
        
        all_person_detected_mask = info['vmask_2d'][0,2]

        subject_ids, valid_masks[:,2] = self.process_suject_ids(info['track_ids'], used_person_inds, valid_mask_ids=info['vmask_2d'][:,1])
        dst_image, org_image = self.prepare_image(image, image_wbg, augments=pixel_augments)

        # valid mask of 3D pose, smpl root rot, smpl pose param, smpl shape param, global translation
        kp3ds, valid_masks[:,1] = self.process_kp3ds(info['kp3ds'], used_person_inds, info['ds'], \
            augments=position_augments, valid_mask_kp3ds=info['vmask_3d'][:, 0])

        params, valid_masks[:,3:6] = self.process_smpl_params(info['params'], used_person_inds, \
            augments=position_augments, valid_mask_smpl=info['vmask_3d'][:, 1:4])
        
        rot_flip = np.array([position_augments[0], position_augments[1]]) if position_augments is not None else np.array([0,0])
        
        input_data = {
            'image': torch.from_numpy(dst_image).float(),
            'image_org': torch.from_numpy(org_image),
            'full_kp2d': torch.from_numpy(full_kp2ds).float(),
            'person_centers':torch.from_numpy(person_centers).float(), 
            'subject_ids':torch.from_numpy(subject_ids).long(),
            'centermap': centermap.float(),
            'kp_3d': torch.from_numpy(kp3ds).float(),
            'params': torch.from_numpy(params).float(),
            'valid_masks':torch.from_numpy(valid_masks).bool(),
            'offsets': torch.from_numpy(offsets).float(),
            'rot_flip': torch.from_numpy(rot_flip).float(),
            'all_person_detected_mask':torch.Tensor([all_person_detected_mask]).bool(),
            'imgpath': info['imgpath'],
            'data_set': info['ds']}
        
        if args().dynamic_augment:
            valid_dynamic_aug = info['is_static_cam']
            input_data.update({'valid_dynamic_aug':torch.Tensor([valid_dynamic_aug]).bool()})
            if dynamic_augment_cfgs is None and valid_dynamic_aug:
                # if the original video is captured by a static camera, then the original image is the dynamic image.
                dynamic_image = torch.from_numpy(dst_image).float()
                dynamic_kp2ds = full_kp2ds
            elif dynamic_augment_cfgs is None and not valid_dynamic_aug:
                # if the original video is captured by a dynamic camera, then we don't have the GTs.
                dynamic_image = torch.zeros(*self.input_shape, 3).float()
                dynamic_kp2ds = np.ones((self.max_person, self.joint_number, 2))*-2
            else:
                dynamic_image = torch.from_numpy(dynamic_image).float()
                dynamic_kp2ds = [dynamic_img_info[2][uid] for uid in used_person_inds]
                dynamic_kp2ds_mask = info['vmask_2d'][used_person_inds,0]
                dynamic_kp2ds = self.process_kp2ds_bboxes(dynamic_kp2ds, img_shape=dynamic_img_info[0].shape, is_pose2d=dynamic_kp2ds_mask)[2]
            input_data.update({'dynamic_kp2ds': torch.from_numpy(dynamic_kp2ds).float(), 'dynamic_image': dynamic_image})

        input_data = self.add_cam_parameters(input_data, info, position_augments)
        input_data['global_trans'], valid_masks[:,6] = self.add_global_trans(info, used_person_inds, position_augments)

        if self.load_vertices:
            verts_processed, valid_masks[:,7] = self.process_verts(info['verts'], used_person_inds, \
                augments=position_augments, valid_mask_verts=info['vmask_3d'][:, 4])
            input_data.update({'verts': torch.from_numpy(verts_processed).float()})
        
        if args().learn_cam_with_fbboxes:
            input_data['full_body_bboxes'] = torch.from_numpy(full_body_bboxes).float()

        if self.train_flag:
            img_scale = 1 if position_augments is None else position_augments[3]
            input_data['img_scale'] = torch.Tensor([img_scale]).float()

        if args().learn_2dpose:
            input_data.update({'heatmap':torch.from_numpy(heatmap).float()})
        if args().learn_AE:
            input_data.update({'AE_joints': torch.from_numpy(AE_joints).long()})

        root_trans, cam_params, cam_mask = self._calc_normed_cam_params_(full_kp2ds, kp3ds, valid_masks[:, 1], info['ds'], fovs=input_data['fovs'])
        centermap_3d, valid_centermap3d_mask = self.generate_centermap_3d(person_centers, cam_params, cam_mask, bboxes_hw_norm, all_person_detected_mask)
        #center_coords = self.CM.parse_3dcentermap_heatmap_adaptive_scale_batch(centermap_3d[None])
        input_data.update({'cams': cam_params,'cam_mask': cam_mask, 'root_trans_cam':root_trans,\
            'centermap_3d':centermap_3d.float(),'valid_centermap3d_mask':torch.Tensor([valid_centermap3d_mask]).bool()})
        
        if args().video:
            #'world_grots_trans': [grots_world, trans_world]
            if 'world_grots_trans' in info:
                grots_world, trans_world = info['world_grots_trans']
                world_global_rots, world_cam_params, world_cam_mask, world_root_trans, _ = self._pack_world_grots_trans_(grots_world, trans_world, cam_mask, info['camPoses'])
                input_data.update({'world_global_rots':world_global_rots, 'world_cams': world_cam_params, 'world_cam_mask': world_cam_mask, 'world_root_trans': world_root_trans})
            
            elif 'camPoses' in info:
                world_global_rots, world_cam_params, world_cam_mask, world_root_trans = self._derive_world_cam_from_camera_rotation2(input_data['params'], full_kp2ds, kp3ds, cam_mask, info['camMats'], info['camPoses'])
                world_cam_params[world_cam_mask], world_root_trans[world_cam_mask] = self.convert_worldRT2defaultFOVs2(kp3ds[world_cam_mask], world_root_trans[world_cam_mask], info['camMats'], input_data['fovs'])
                input_data.update({'world_global_rots':world_global_rots, 'world_cams': world_cam_params, 'world_cam_mask': world_cam_mask, 'world_root_trans': world_root_trans})
            
            elif args().dynamic_augment:
                valid_kp3d_mask = valid_masks[:, 1] if valid_dynamic_aug else np.zeros(self.max_person, dtype=np.bool_)
                world_root_trans, world_cam_params, world_cam_mask = self._calc_normed_cam_params_(dynamic_kp2ds, kp3ds, valid_kp3d_mask, info['ds'], fovs=input_data['fovs'])
                world_global_rots = input_data['params'][:,:3]
                input_data.update({'world_global_rots':world_global_rots, 'world_cams': world_cam_params,'world_cam_mask': world_cam_mask, 'world_root_trans': world_root_trans})
        
        if 'precise_kp3d_mask' in info:
            input_data['valid_masks'][:,1] = self.reset_precise_kp3d_mask(input_data['valid_masks'][:,1], input_data['kp_3d'], info['precise_kp3d_mask'], used_person_inds)

        if args().learn_relative:
            # age; gender; depth level; body type
            depth_info = self._organize_depth_info_(info, used_person_inds)
            input_data.update({'depth_info': torch.from_numpy(depth_info).long()})
            input_data['kid_shape_offsets'] = self._organize_kid_shape_offsets_(info, used_person_inds)
        if args().video:
            if 'seq_info' in info:
                input_data.update({'seq_info':torch.Tensor(info['seq_info'])})
            else:
                input_data.update({'seq_info':torch.Tensor([-2,-2,0])})
        
        if args().learn_deocclusion:
            extra_image_data = self.process_extra_image(info, org_kp2ds_dc, used_person_inds, position_augments, pixel_augments)
            input_data.update(extra_image_data)

        if return_augcfgs:
            return input_data, (position_augments, pixel_augments)
        else:
            return input_data
        
    def convert_worldRT2defaultFOVs2(self, kp3ds, world_root_trans, intrinsics, target_fovs):
        world_kp3ds = kp3ds+world_root_trans.unsqueeze(1).numpy()
        fovs = 1./np.tan(np.radians(intrinsics[0, 0]/2))
        kp2ds = perspective_projection_withfovs(world_kp3ds, fovs=fovs).numpy() # , rotation=inv_cam_rotmat

        world_root_trans, world_cam_params = self.solving_trans3D(kp2ds, kp3ds, target_fovs)
        return world_cam_params, world_root_trans
    
    def convert_worldRT2defaultFOVs(self, kp3ds, world_root_trans, first_frame_camera_rotation, camera_rotation, target_fovs):
        # inverse_pitch_yaw = camera_rotation[1:] - first_frame_camera_rotation[1:]
        # inv_cam_rotmat = camera_pitch_yaw_roll2rotation_matrix(*inverse_pitch_yaw)
        # inv_cam_rotmat = torch.from_numpy(inv_cam_rotmat).float()[None].repeat(len(kp3ds),1,1)

        world_kp3ds = kp3ds+world_root_trans.unsqueeze(1).numpy()
        fovs = 1./np.tan(np.radians(camera_rotation[0]/2))
        kp2ds = perspective_projection_withfovs(world_kp3ds, fovs=fovs).numpy() # , rotation=inv_cam_rotmat

        world_root_trans, world_cam_params = self.solving_trans3D(kp2ds, kp3ds, target_fovs)
        return world_cam_params, world_root_trans
        
        
    def add_global_trans(self, info, used_person_inds, position_augments):
        global_trans = torch.zeros(self.max_person, 3).float()
        valid_mask = torch.zeros(self.max_person).bool()
        if 'global_trans' in info:
            global_trans[:len(used_person_inds)] = torch.from_numpy(info['global_trans'][used_person_inds])
            valid_mask[:len(used_person_inds)] = True
            # TODO: how does the position_augments affect the global_trans?
        return global_trans, valid_mask
            
    
    def process_extra_image(self, info, org_kp2ds, used_person_inds, position_augments, pixel_augments, max_person_num=18):
        def apply_transformation(extra_image, info, kp2ds, position_augments, pixel_augments):
            extra_image = process_image(extra_image, kp2ds, augments=position_augments, is_pose2d=info['vmask_2d'][:,0])[0]
            if pixel_augments is not None:
                extra_image = self.aug_image(extra_image, pixel_augments[0], pixel_augments[1])
            extra_image = cv2.resize(extra_image, tuple(self.input_shape), interpolation = cv2.INTER_CUBIC)
            return extra_image
        
        extra_image_data = {'extra_images':['']*max_person_num,\
            'extra_images_mask': torch.zeros(max_person_num).bool()}
        if 'extra_images' not in info:
            return extra_image_data
        
        extra_images = [info['extra_images'][ind] for ind in used_person_inds]
        for ind, extra_image in enumerate(extra_images):
            transformed_image = apply_transformation(extra_image, info, copy.deepcopy(org_kp2ds), position_augments, pixel_augments)
            unique_name = os.path.join(config.data_cache_dir, '{}-{}'.format(int(random.random()*10000000), os.path.basename(info['imgpath'])))
            cv2.imwrite(unique_name, transformed_image)
            extra_image_data['extra_images'][ind] = unique_name
            extra_image_data['extra_images_mask'][ind] = True
        return extra_image_data
    
    def process_full_body_bboxes(self, full_kps, is_bboxes, img_shape, ds):
        full_body_bboxes = np.ones((self.max_person, 4))*-2.
        if is_bboxes.sum()>0:
            if len(is_bboxes) > self.max_person:
                is_bboxes = is_bboxes[:self.max_person]
                full_kps = full_kps[:self.max_person]
            full_bboxes = np.array([self.process_kps(full_kps[ind], img_shape,set_minus=False) for ind in np.where(is_bboxes)[0]])
            if ds in ['crowdhuman', 'DanceTrack']: # requires bounding boxes for the full body, not the visibile part only in most datasets.
                full_body_bboxes[:len(is_bboxes)][is_bboxes] = full_bboxes[:,:2].reshape((-1,4))
        return full_body_bboxes

    def generate_dynamic_depth(self, cam_params, cam_mask, bboxes_hw_norm, all_person_detected_mask):
        radius_list = torch.Tensor(_calc_radius_(bboxes_hw_norm)).long()
        person_scales = torch.zeros(self.max_person).long()
        person_num = min(self.max_person, len(radius_list))
        person_scales[:person_num] = radius_list[:person_num]
        return person_scales

    def _organize_depth_info_(self, info, used_person_inds):
        prepared_info = np.ones((self.max_person, 4))*-1
        if 'depth' in info:
            if len(info['depth']) != 0:
                prepared_info[:len(used_person_inds)] = np.array(info['depth'])[used_person_inds]
        return prepared_info

    def _organize_kid_shape_offsets_(self, info, used_person_inds):
        kid_shape_offsets_processed = torch.ones(self.max_person).float()*-1
        if 'kid_shape_offsets' in info:
            kid_shape_offsets_processed[:len(used_person_inds)] = torch.from_numpy(info['kid_shape_offsets'][used_person_inds]).float()
        return kid_shape_offsets_processed


    def __getitem__(self, index):
        return self.get_item_single_frame(index)
        try:
            return self.get_item_single_frame(index)
        except Exception as error:
            logging.error(error)
            index = np.random.randint(len(self))
            return self.get_item_single_frame(index)
    
    def _derive_world_cam_from_camera_rotation2(self, smpl_params, full_kp2ds, kp3ds, cam_mask, intrinsics, extrinsics):
        world_cam_mask = cam_mask.clone()
        world_cam_params = torch.ones(self.max_person,3,dtype=torch.float32)*-2
        world_trans = torch.ones(self.max_person,3,dtype=torch.float32)*-2
        world_global_rots = torch.ones(self.max_person,3,dtype=torch.float32)*-10
        
        if cam_mask.sum() == 0:
            return world_global_rots, world_cam_params, world_cam_mask, world_trans
        
        cf_fov = intrinsics[0, 0]
        fov = 1./np.tan(np.radians(cf_fov / 2))

        body_rots_cam = angle_axis_to_rotation_matrix(smpl_params[cam_mask][:,:3]).numpy()
        body_trans_cam, _ = self.solving_trans3D(full_kp2ds[cam_mask], kp3ds[cam_mask], fov)
        body_R_in_world, body_T_in_world = convert_camera2world_RT2(body_rots_cam, body_trans_cam, extrinsics)

        #self.convert_body_T2default_fov()
        world_cam_params[world_cam_mask] = torch.from_numpy(normalize_trans_to_cam_params(body_T_in_world)).float()
        world_trans[world_cam_mask] = torch.from_numpy(body_T_in_world).float()
        world_global_rots[world_cam_mask] = body_R_in_world

        return world_global_rots, world_cam_params, world_cam_mask, world_trans

    
    def _derive_world_cam_from_camera_rotation(self, smpl_params, full_kp2ds, kp3ds, cam_mask, first_frame_camera_rotation, camera_rotation):
        world_cam_mask = cam_mask.clone()
        world_cam_params = torch.ones(self.max_person,3,dtype=torch.float32)*-2
        world_trans = torch.ones(self.max_person,3,dtype=torch.float32)*-2
        world_global_rots = torch.ones(self.max_person,3,dtype=torch.float32)*-10
        
        if cam_mask.sum() == 0:
            return world_global_rots, world_cam_params, world_cam_mask, world_trans, None
        
        cf_fov = camera_rotation[0]
        fov = 1./np.tan(np.radians(cf_fov / 2))
        delta_pitch_yaw_roll = camera_rotation[1:] - first_frame_camera_rotation[1:]

        body_rots_cam = angle_axis_to_rotation_matrix(smpl_params[cam_mask][:,:3]).numpy()
        body_trans_cam, _ = self.solving_trans3D(full_kp2ds[cam_mask], kp3ds[cam_mask], fov)
        body_R_in_world, body_T_in_world, cam_RT = convert_camera2world_RT(body_rots_cam, body_trans_cam, fov, delta_pitch_yaw_roll)

        #self.convert_body_T2default_fov()
        world_cam_params[world_cam_mask] = torch.from_numpy(normalize_trans_to_cam_params(body_T_in_world)).float()
        world_trans[world_cam_mask] = torch.from_numpy(body_T_in_world).float()
        world_global_rots[world_cam_mask] = body_R_in_world

        return world_global_rots, world_cam_params, world_cam_mask, world_trans, cam_RT
    
    def _pack_world_grots_trans_(self, grots_world, trans_world, cam_mask, cam_RT):
        world_cam_mask = cam_mask.clone()
        world_cam_params = torch.ones(self.max_person,3,dtype=torch.float32)*-2
        world_trans = torch.ones(self.max_person,3,dtype=torch.float32)*-2
        world_global_rots = torch.ones(self.max_person,3,dtype=torch.float32)*-10

        if cam_mask.sum() == 0:
            return world_global_rots, world_cam_params, world_cam_mask, world_trans, None
        
        world_cam_params[world_cam_mask] = torch.from_numpy(normalize_trans_to_cam_params(trans_world)).float()
        world_global_rots[world_cam_mask] = torch.from_numpy(grots_world).float()
        world_trans[world_cam_mask] = torch.from_numpy(trans_world).float()

        return world_global_rots, world_cam_params, world_cam_mask, world_trans, cam_RT


def convertRT2transform(R, T):
    transform4x4 = np.eye(4)
    transform4x4[:3, :3] = R
    transform4x4[:3, 3] = T
    return transform4x4

def transform_trans(transform_mat, trans):
    trans = np.concatenate((trans, np.ones_like(trans[[0]])), axis=-1)[None, :]
    trans_new = np.matmul(trans, np.transpose(transform_mat, (1,0)))[0, :3]
    return trans_new

def camera_pitch_yaw2rotation_matrix(pitch, yaw, roll=0):
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(pitch))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(yaw))
    R = R2 @ R1
    return R.T

def camera_pitch_yaw_roll2rotation_matrix(pitch, yaw, roll=0):
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)

    Rx, _ = cv2.Rodrigues(x_axis * np.radians(pitch))
    Ry, _ = cv2.Rodrigues(y_axis * np.radians(yaw))
    Rz, _ = cv2.Rodrigues(z_axis * np.radians(roll))
    R = Rz @ Ry @ Rx
    return R

def inverse_transform(transform_mat):
    transform_inv = np.zeros_like(transform_mat)
    transform_inv[:3, :3] = np.transpose(transform_mat[:3, :3], (1,0))
    transform_inv[:3, 3] = -np.matmul(transform_mat[:3, 3][None], transform_mat[:3, :3])
    transform_inv[3, 3] = 1.0
    return transform_inv

def convert_camera2world_RT2(body_rots_cam, body_trans_cam, world2camera):
    camera2world = inverse_transform(world2camera)
    
    # SMPL pose without global rotation or 3D translation
    body_R_in_world = np.stack([np.matmul(camera2world[:3, :3], body_R_in_cam) for body_R_in_cam in body_rots_cam], 0)
    body_R_in_world = rotation_matrix_to_angle_axis(torch.from_numpy(body_R_in_world).float())
    
    body_T_in_world = np.stack([transform_trans(camera2world, body_T_in_cam) for body_T_in_cam in body_trans_cam], 0)
    return body_R_in_world, body_T_in_world

def convert_camera2world_RT(body_rots_cam, body_trans_cam, fov, pitch_yaw_roll):
    camera_R_mat = camera_pitch_yaw_roll2rotation_matrix(*pitch_yaw_roll)
    camera_T = np.zeros(3)
    world2camera = convertRT2transform(camera_R_mat, camera_T)
    camera2world = inverse_transform(world2camera)
    
    # SMPL pose without global rotation or 3D translation
    body_R_in_world = np.stack([np.matmul(camera2world[:3, :3], body_R_in_cam) for body_R_in_cam in body_rots_cam], 0)
    body_R_in_world = rotation_matrix_to_angle_axis(torch.from_numpy(body_R_in_world).float())
    
    body_T_in_world = np.stack([transform_trans(camera2world, body_T_in_cam) for body_T_in_cam in body_trans_cam], 0)
    return body_R_in_world, body_T_in_world, world2camera

name_dict = {
    'depth_id': {0: '最前排', 1: '第二排', 2: '第三排', 3: '第四排', 4: '第五排', 5: '第六排', 6: '第七排', 7: '第八排', 8: '第九排', 9: '第十排', -1: '深度不明'},
    'age': {0: '成年', 1: '青少年', 2: '小孩', 3:'婴幼儿', -1: '年龄不明'},
    'body_type': {0: '正常', 1: '微胖', 2: '胖', 3: '强壮'},
    'occluded_by_others': {0: '无遮挡', 1: '遮挡'},
    'gender': {0: '男', 1: '女', -1:'性别不明'}
}

def visualize_3d_hmap(hmap,save_name):
    if not (type(hmap) is np.ndarray):
        try:
            hmap = hmap.cpu().numpy()
        except:
            hmap = hmap.detach().cpu().numpy()

    hmap[hmap < 0] = 0
    hmap[hmap > 1] = 1
    hmap = (hmap * 255).astype(np.uint8)
    for d, x in enumerate(hmap):
        x = cv2.applyColorMap(x, colormap=cv2.COLORMAP_JET)
        x = cv2.putText(x, f'{d}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 128, 128), 2, cv2.LINE_AA)
        cv2.imwrite(save_name+'_{}.jpg'.format(d), x)

def exclude_wrong_kp2ds2(kp2ds, image):
    h, w = image.shape[:2]
    kp_num = len(kp2ds)
    for ind in range(kp_num):
        x, y = kp2ds[ind]
        if x==-256 and y==-256:
            continue
        valid_mask = kp2ds[:,0]>0
        dists = np.linalg.norm(kp2ds[valid_mask]-kp2ds[ind][None], axis=1, ord=2).mean() / min(h, w)
        if dists > 0.5:
            print(dists, ind, kp2ds[ind], h, w, kp2ds)
            kp2ds[ind] = -2.
            cv2.imshow('image',cv2.circle(image, (x,y), 10, (0,0,255), thickness=-1))
            cv2.waitKey(0)
    return kp2ds

def _calc_bbox_normed2(full_kps):
    bboxes = []
    for kps_i in full_kps:
        if (kps_i[:,0]>-2).sum()>0:
            bboxes.append(calc_aabb(kps_i[kps_i[:,0]>-2]))
        else:
            bboxes.append(np.zeros((2,2)))

    return bboxes


def test_image_relative_dataset(datasets,with_3d=False,with_smpl=False):
    print('testing relative datasets loading')
    print('configs_yml:', args().configs_yml)
    print('model_version:',args().model_version)
    
    from visualization.visualization import Visualizer
    test_projection_part = True if args().model_version in [4,5,6,7] else False
    print('test_projection_part:',test_projection_part)

    save_dir = os.path.join(config.project_dir,'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    print('Initialized datasets')

    batch_size, model_type= 1, 'smpl'
    dataloader = DataLoader(dataset=datasets,batch_size = batch_size,shuffle = True,\
        drop_last = False,pin_memory = False,num_workers = 1)
    visualizer = Visualizer(resolution = (512,512,3), result_img_dir=save_dir,with_renderer=True,use_gpu=True)
    print('Initialized visualizer')

    from visualization.visualization import make_heatmaps, draw_skeleton_multiperson
    from utils.cam_utils import denormalize_cam_params_to_trans
    from utils.util import save_obj
    if with_smpl:
        from smpl_family.smpl import SMPL
        smpl = SMPL(args().smpl_model_path, model_type='smpl')
        from smpl_family.smpla import SMPLA_parser
        smpl_family = SMPLA_parser(args().smpla_model_path, args().smil_model_path, baby_thresh=args().baby_threshold)
    print('Initialized SMPL models')

    img_size = 512
    print(f'Joint number: {args().joint_num}')
    if args().joint_num == 44:
        bones, cm = constants.All44_connMat, constants.cm_All54
    elif args().joint_num == 73:
        bones, cm = constants.All73_connMat, constants.cm_All54
    print('Start loading data.')
    for _,r in enumerate(dataloader):
        if _%100==0:
            for key,value in r.items():
                if isinstance(value,torch.Tensor):
                    print(key,value.shape)
                elif isinstance(value,list):
                    print(key,len(value))
        for inds in range(1):
            img_bsname = os.path.basename(r['imgpath'][inds])
            image = r['image'][inds].numpy().astype(np.uint8)[:,:,::-1]
            valid_person_inds = np.where((r['full_kp2d'][inds]!=-2).sum(-1).sum(-1)>0)[0]

            full_kp2d = (r['full_kp2d'][inds].numpy() + 1) * img_size / 2.0
            person_centers = (r['person_centers'][inds].numpy() + 1) * img_size / 2.0
            subject_ids = r['subject_ids'][inds]
            image_kp2d = visualizer.draw_skeleton_multiperson(image.copy(), full_kp2d, bones=bones, cm=cm, label_kp_order=True)

            if test_projection_part and r['cam_mask'][inds].sum()>0:
                cam_mask = r['cam_mask'][inds]
                kp3d_tp = r['kp_3d'][inds][cam_mask].clone()
                kp2d_tp = r['full_kp2d'][inds][cam_mask].clone()
                pred_cam_t = denormalize_cam_params_to_trans(r['cams'][inds][cam_mask].clone())
                
                pred_keypoints_2d = perspective_projection(kp3d_tp,translation=pred_cam_t,focal_length=args().focal_length, normalize=False)+512//2
                invalid_mask = np.logical_or(kp3d_tp[:,:,-1]==-2., kp2d_tp[:,:,-1]==-2.)
                pred_keypoints_2d[invalid_mask] = -2.
                #print('kp3d_tp', kp3d_tp[:,[39,35,36]], pred_keypoints_2d[:,[39,35,36]])
                image_kp2d_projection = visualizer.draw_skeleton_multiperson(image.copy(), pred_keypoints_2d, bones=bones, cm=cm)
                cv2.imwrite('{}/{}_{}_projection.jpg'.format(save_dir,_,img_bsname), image_kp2d_projection)

            for pinds, (person_center, subject_id) in enumerate(zip(person_centers,subject_ids)):
                y,x = person_center.astype(np.int32)
                if y>0 and x>0:
                    cv2.circle(image_kp2d, (x,y), 6, [0,0,255],-1)
                    text = '{}'.format(subject_id)
                    if 'depth_info' in r:
                        depth_info = r['depth_info'][inds][pinds]
                        text+= '{}'.format(depth_info.numpy().tolist())           
          
            centermap_color = make_heatmaps(image.copy(), r['centermap'][inds])
            image_vis = np.concatenate([image_kp2d, centermap_color],1)
            cv2.imwrite('{}/{}_{}_centermap.jpg'.format(save_dir,_,img_bsname), image_vis)
            if 'heatmap' in r:
                heatmap_color = make_heatmaps(image.copy(), r['heatmap'][inds])
                cv2.imwrite('{}/{}_{}_heatmap.jpg'.format(save_dir,_,img_bsname), heatmap_color)

            person_centers_onmap = ((r['person_centers'][inds].numpy() + 1)/ 2.0 * (args().centermap_size-1)).astype(np.int32)
            positive_position = torch.stack(torch.where(r['centermap'][inds,0]==1)).permute(1,0)

        if with_smpl and r['valid_masks'][0,0,4]:
            params, subject_ids = r['params'][0],  r['subject_ids'][0]
            image = r['image'][0].numpy().astype(np.uint8)[:,:,::-1]
            valid_mask = torch.where(r['valid_masks'][0,:,4])[0]
            subject_ids = subject_ids[valid_mask]
            pose = params[valid_mask][:,:66].float()
            if r['valid_masks'][0,valid_mask,5].sum()>0:
                betas = params[valid_mask][:,-10:].float()
            else:
                betas = torch.zeros(len(pose),10)
            pose = torch.cat([pose, torch.zeros(len(pose),6)],-1).float()
            if 'kid_shape_offsets' in r:
                kso_vmask = r['kid_shape_offsets'][0][valid_mask]!=-1
                betas = torch.cat([betas, torch.zeros(len(betas), 1)], 1)
                if (kso_vmask).sum()>0:
                    betas[kso_vmask,-1] = r['kid_shape_offsets'][0][valid_mask][kso_vmask]

                verts,joints = smpl_family(poses=pose, betas=betas)
                print('using kid shape offset to create mesh')
            else:
                verts,joints = smpl(poses=pose, betas=betas, get_skin = True)

            if test_projection_part and r['cam_mask'][inds].sum()>0:
                if r['cam_mask'][0][valid_mask].sum()>0:
                    trans = denormalize_cam_params_to_trans(r['cams'][0][valid_mask].clone(), positive_constrain=True)
                else:
                    trans = r['root_trans'][0][valid_mask]     

                pred_keypoints_2d = perspective_projection(joints,translation=trans,focal_length=args().focal_length, normalize=False)+512//2
   
                render_img = visualizer.visualize_renderer_verts_list([verts.cuda()], trans=[trans.cuda()], images=image[None])[0]

                rendered_img_bv = visualizer.visualize_renderer_verts_list([verts.cuda()], trans=[trans.cuda()], bird_view=True, auto_cam=True)[0]
                person_centers = (r['person_centers'][0].numpy() + 1) * img_size / 2.0
                image = np.array(image).astype(np.uint8)
                if len(person_centers) == len(trans):
                    for pinds, person_center in enumerate(person_centers):
                        y,x = person_center.astype(np.int32)
                        if y>0 and x>0:
                            cv2.circle(np.array(image), (x,y), 6, [0,0,255],-1)
                            text = '{:.2f}'.format(trans[pinds,2])
                            print((x,y), text)
                            cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)     
                cv2.imwrite('{}/mesh_{}.png'.format(save_dir,_), np.concatenate([image, render_img, rendered_img_bv],1)) #image_kp2d_projection_smpl24, image_kp2d_projection_extra30
            else:
                verts[:,:,2] += 5
                render_img = visualizer.visualize_renderer_verts_list([verts.cuda()], images=image[None])[0]
                cv2.imwrite('{}/mesh_{}.png'.format(save_dir,_), render_img)
        j3ds = r['kp_3d'][0,0]
        image = r['image'][0].numpy().astype(np.uint8)[:,:,::-1]
        if r['valid_masks'][0,0,1]:
            pj2d = (j3ds[:,:2] + 1) * img_size / 2.0
            pj2d[j3ds[:,-1]==-2.] = -2.
            image_pkp3d = visualizer.draw_skeleton(image.copy(), pj2d, bones=bones, cm=cm)
            cv2.imwrite('{}/pkp3d_{}_{}.png'.format(save_dir,_,r['subject_ids'][0, 0]), image_pkp3d)