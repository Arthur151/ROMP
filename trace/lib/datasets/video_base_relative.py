from datasets.image_base_relative import *
from datasets.image_base import get_bounding_bbox
from random import sample
from utils.video_utils import convert_centers_to_trajectory
"""
python -m lib.datasets.pw3d --configs_yml='configs/video_v1.yml'
"""

class Video_base_relative(Image_base_relative):
    def __init__(self, train_flag=True, regress_smpl = False, load_entire_sequence=False, \
                dynamic_augment=args().dynamic_augment,  aligning2first_frame_coordinate=False,**kwargs):
        super(Video_base_relative,self).__init__(train_flag=train_flag, regress_smpl=regress_smpl, **kwargs)
        self.load_entire_sequence = load_entire_sequence
        self.dynamic_augment = dynamic_augment 
        self.aligning2first_frame_coordinate = aligning2first_frame_coordinate
        self.dynamic_augment_ratio = args().dynamic_augment_ratio
        self.dynamic_aug_tracking_ratio = args().dynamic_aug_tracking_ratio
        if not self.load_entire_sequence:
            self.random_temp_sample_internal = args().random_temp_sample_internal
            self.temp_clip_length = args().temp_clip_length
            self.temp_spawn = self.temp_clip_length // 2
        else:
            # TODO: figure out why this need the max limitation ? limited GPU memory?
            self.clip_max_length = 128
            self.prepare_video_clips = self.split_sequence2clips
    
    def split_sequence2clips(self):
        video_clip_ids = []
        seq_end_flag = []
        
        for sid, seq_sample_ids in enumerate(self.sequence_ids):
            seq_length = len(seq_sample_ids)
            clip_num = int(np.ceil(seq_length/self.clip_max_length))
            for clip_id in range(clip_num):
                video_clip_ids.append([sid,seq_sample_ids[self.clip_max_length*clip_id : self.clip_max_length*(clip_id+1)]])
                seq_end_flag.append(clip_id==(clip_num-1))
        self.seq_end_flag = seq_end_flag
        return video_clip_ids
    
    def collect_entire_sequence_inputs(self, index):
        sequence_id, frame_ids = self.video_clip_ids[index]
        frame_data = [None for _ in range(len(frame_ids))]
        augment_cfgs = (None,None)
        for cid, fid in enumerate(frame_ids):
            frame_data[cid] = self.get_item_single_frame(
                fid, augment_cfgs=augment_cfgs)
        seq_data = self.pack_clip_data(frame_data)
        seq_data['seq_end_flag'] = torch.tensor([self.seq_end_flag[index]])
        seq_data['seq_name'] = self.sid_video_name[sequence_id]
        return seq_data
    
    def sample_one_clip_data(self, frame_ids):
        clip_data = [None for _ in range(len(frame_ids))]
        augment_cfgs, dynamic_augment_cfgs, syn_occlusion = self.determine_augment_cfgs(frame_ids)

        #camera_rotation = None
        #if self.with_dynamic_camera_rotation:
        #    camera_rotation = self.get_image_info(frame_ids[0])['camera_rotation']

        for cid, fid in enumerate(frame_ids):
            clip_data[cid] = self.get_item_single_frame(fid, augment_cfgs=copy.deepcopy(augment_cfgs), \
                    dynamic_augment_cfgs=dynamic_augment_cfgs[cid], syn_occlusion=syn_occlusion)
            #, first_frame_camera_rotation=camera_rotation
            #clip_data[cid]['camera_states'] = self.determine_camera_states(dynamic_augment_cfgs[cid], clip_data[cid]['valid_dynamic_aug'])
        clip_data = self.pack_clip_data(clip_data)
        
        return clip_data

    def determine_augment_cfgs(self, frame_ids):
        dynamic_augment_cfgs = [None for _ in range(len(frame_ids))]
        is_static_cam = self.check_static_camera(frame_ids)
        if not self.train_flag:
            augment_cfgs = None
            info = None
            syn_occlusion = None
            return augment_cfgs, dynamic_augment_cfgs, syn_occlusion
        elif self.dynamic_augment and is_static_cam and random.random() < self.dynamic_augment_ratio:
            if random.random() < self.dynamic_aug_tracking_ratio:
                # tracking some subjects with dynamic camera
                infos = [self.get_image_info(frame_id) for frame_id in frame_ids]
                scale = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
                augment_cfgs, dynamic_augment_cfgs = generate_dynamic_tracking_augments(copy.deepcopy(infos), scale) 
                info = infos[0]
            else:
                info = self.get_image_info(frame_ids[0])
                augment_cfgs = self._calc_augment_confs(info['image'], copy.deepcopy(info['kp2ds']), is_pose2d=copy.deepcopy(info['vmask_2d'][:,0]))
                dynamic_augment_cfgs = generate_dynamic_augments(frame_ids, augment_cfgs) 
        else:
            info = self.get_image_info(frame_ids[0])
            augment_cfgs = self._calc_augment_confs(info['image'], copy.deepcopy(info['kp2ds']), is_pose2d=copy.deepcopy(info['vmask_2d'][:,0]))
        syn_occlusion = self.prepared_syn_occlusion(copy.deepcopy(info), augment_cfgs)
        return augment_cfgs, dynamic_augment_cfgs, syn_occlusion
    
    def prepared_syn_occlusion(self, info, augment_cfgs):
        syn_flag = augment_cfgs[1][1]
        if syn_flag and self.syn_obj_occlusion and info is not None:
            occ_image, occluder, center = self.synthetic_occlusion(info['image'])
            return self.synthetic_occlusion, occluder, center
        else:
            return None
    
    def check_static_camera(self, frame_ids):
        info = self.get_image_info(frame_ids[0])
        #dataset_name = info['ds']
        if 'is_static_cam' in info:
            is_static_cam = info['is_static_cam']
        else:
            is_static_cam = True
        return is_static_cam

    def collect_video_clip_inputs(self, index):
        seqence_id, frame_ids, interval = self.video_clip_ids[index % len(self.video_clip_ids)]
        clip_data = self.sample_one_clip_data(frame_ids)

        if not check_person_number(clip_data) or not check_complete_traj(clip_data):
            return self.collect_video_clip_inputs(random.randint(0,len(self)))

        return clip_data
    
    def pack_clip_data(self, clip_data):
        all_data = {}
        for key in clip_data[0].keys():
            if isinstance(clip_data[0][key], torch.Tensor):
                all_data[key] = torch.stack([data[key] for data in clip_data])
            elif isinstance(clip_data[0][key], str):
                all_data[key] = [data[key] for data in clip_data]
                # dataloader will collect list to T x B (7x16), not B x T (16x7) as we espect. 
            elif isinstance(clip_data[0][key], int):
                all_data[key] = torch.Tensor([data[key] for data in clip_data])
        if self.aligning2first_frame_coordinate:
            all_data = align2first_frame_coordinate(all_data)
        
        return all_data

    def __getitem__(self, index):
        if self.load_entire_sequence:
            return self.collect_entire_sequence_inputs(index)
        else:
            return self.collect_video_clip_inputs(index)
        # try:
        #     if self.load_entire_sequence:
        #         return self.collect_entire_sequence_inputs(index)
        #     else:
        #         return self.collect_video_clip_inputs(index)
        # except:
        #     index = np.random.randint(len(self))
        #     return self.__getitem__(index)
    
    def __len__(self):
        return len(self.video_clip_ids)

    def prepare_video_clips(self):
        video_clip_ids = []
        for sid, seq_sample_ids in enumerate(self.sequence_ids):
            # padding the sequence with the beginning / ending sample ids to make sure we use the input video sequence and estimate every frame
            # pad_seq_ids = np.concatenate([np.ones(self.temp_spawn) * seq_sample_ids[0], np.array(seq_sample_ids)], 0)
            # pad_seq_ids = np.concatenate([np.array(pad_seq_ids), np.ones(self.temp_spawn) * seq_sample_ids[-1]], 0).astype(np.int32)
            seq_length = len(seq_sample_ids)
            sampling_interval = random.randint(0, self.random_temp_sample_internal)
            clip_frame_num = self.temp_clip_length + (self.temp_clip_length-1) * sampling_interval
            clip_frame_spawn = clip_frame_num // 2
            clip_num = (seq_length - (clip_frame_num - clip_frame_spawn)) // clip_frame_spawn
            #max((seq_length - (clip_frame_num - clip_frame_spawn)) // clip_frame_spawn, 1)
            clip_sampling_inds = np.arange(self.temp_clip_length) * (sampling_interval+1)
            #print(seq_length, clip_num,sampling_interval, clip_frame_num, clip_sampling_inds)
            for cl_id in range(clip_num):
                sampling_inds = np.array(seq_sample_ids)[cl_id*clip_frame_spawn : cl_id*clip_frame_spawn+clip_frame_num]
                sampling_inds = sampling_inds[clip_sampling_inds]
                video_clip_ids += [[sid, sampling_inds, sampling_interval]]  
        return video_clip_ids

def align2first_frame_coordinate(data):
    camera_extrinsics = data['camPoses']
    world_cam_mask = data['world_cam_mask']
    world_global_rots = data['world_global_rots']
    world_root_trans = data['world_root_trans']

    batch_size = len(camera_extrinsics)
    first_frame_coordinate = camera_extrinsics[[0]].repeat(batch_size, 1, 1)

    #if world_cam_mask.sum()<batch_size:
    #    print(data['data_set'], data['imgpath'][0])
    
    world_grots = world_global_rots[world_cam_mask]
    world_grots = angle_axis_to_rotation_matrix(world_grots)
    world_trans = world_root_trans[world_cam_mask]
    world_grot_trans = torch.cat([world_grots, world_trans[:,:,None]], -1)
    world_homo = torch.cat([world_grot_trans, torch.Tensor([[[0,0,0,1]]]).repeat(len((world_grot_trans)),1,1)], 1)
    world_homo = world_homo.reshape(batch_size, -1, 4, 4)

    first_frame_coordinate_inv = torch.linalg.inv(first_frame_coordinate)
    first_frame_coordinate = first_frame_coordinate.reshape(batch_size, 1, 4, 4).repeat(1, world_homo.shape[1],1,1)
    #print(first_frame_coordinate.shape, world_homo.shape, first_frame_coordinate_inv.shape)

    world_homo_aligned = torch.matmul(first_frame_coordinate, world_homo).reshape(-1,4,4)
    data['world_global_rots'][world_cam_mask] = rotation_matrix_to_angle_axis(world_homo_aligned[:, :3, :3].contiguous())
    data['world_root_trans'][world_cam_mask] = world_homo_aligned[:, :3, 3]
    data['camPoses'] = torch.matmul(first_frame_coordinate_inv, camera_extrinsics)
    return data


dynamic_camera_changing_modes = ['single_direction', 'shaking', 'return']

def single_direction_changing_curves(tcl):
    return [
        np.sin(np.pi / 2 * np.arange(tcl).astype(np.float32) / (tcl-1)), # sin
        -np.sin(np.pi / 2 * np.arange(tcl).astype(np.float32) / (tcl-1)), # -sin
        np.arange(tcl).astype(np.float32) / (tcl-1), # mean
        -np.arange(tcl).astype(np.float32) / (tcl-1), # -mean    
    ] #+ [np.zeros(tcl)]*2 # static camera

def get_fixsize_cut_box(leftTop, rightBottom, ExpandsRatio, Center = None, force_square=False):
    ExpandsRatio = [ExpandsRatio, ExpandsRatio, ExpandsRatio, ExpandsRatio]

    def _expand_crop_box(lt, rb, scale):
        center = (lt + rb) / 2.0
        xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]

        xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
        #expand it
        lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
        lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
        center = (lt + rb) / 2
        return center, lt, rt, rb, lb

    if Center == None:
        Center = (leftTop + rightBottom) // 2

    Center, leftTop, rightTop, rightBottom, leftBottom = _expand_crop_box(leftTop, rightBottom, ExpandsRatio)

    offset = (rightBottom - leftTop) // 2

    cx = offset[0]
    cy = offset[1]

    if force_square:
        r = max(cx, cy)
        cx = r
        cy = r

    x = int(Center[0])
    y = int(Center[1])

    return [x - cx, y - cy], [x + cx, y + cy]

def generate_dynamic_tracking_augments(infos, scale):
    color_jitter = True if random.random()<args().color_jittering_ratio else False
    syn_occlusion = True if random.random()<args().Synthetic_occlusion_ratio else False
    pixel_augments = (color_jitter, syn_occlusion)
    
    # flip will lead to wrong world gts
    flip = False #True if random.random()<0.5 else 
    rot = random.randint(-30,30) if random.random()<args().rotate_prob else 0                 
    
    height, width = infos[0]['image'].shape[0], infos[0]['image'].shape[1]
    kp2ds = [info['kp2ds'] for info in infos]
    track_ids = [info['track_ids'] for info in infos]
    all_subject_ids = np.concatenate(track_ids, 0)
    full_traj_subjects = []
    for subject_id in np.unique(all_subject_ids):
        if (all_subject_ids == subject_id).sum() == len(infos):
            full_traj_subjects.append(subject_id)
    if len(full_traj_subjects) == 0:
        #print('Error@!!! without subjects with complete trajectory for dynamic tracking')
        position_augments = [rot, flip, (0, 0, width, height), 1]
        augment_cfgs = (position_augments, pixel_augments)
        dynamic_augment_cfgs = [None for _ in range(len(infos))]
        return augment_cfgs, dynamic_augment_cfgs
    
    sellected_subject_id = random.sample(full_traj_subjects, 1)

      
    moving_centers, sizes = [], []
    for full_kp2ds, subject_ids in zip(kp2ds, track_ids):
        sample_ids = np.where(subject_ids == sellected_subject_id)[0]

        boxes = np.concatenate([get_bounding_bbox(full_kp2ds[ind]) for ind in sample_ids], 0)
        if flip:
            boxes[:,0] = width - boxes[:,0]
        box = calc_aabb(boxes)
        box[:,0], box[:,1] = np.clip(box[:,0], 0, width), np.clip(box[:,1], 0, height)
        leftTop, rightBottom = box
        moving_centers.append((leftTop+rightBottom)/2)
        sizes.append(max((rightBottom-leftTop)/2))

    dynamic_augment_cfgs = []  
    fixed_size = int(max(sizes) * scale)
    for moving_center in moving_centers:
        crop_bbox = (*(moving_center.astype(np.int32) - fixed_size), *(moving_center.astype(np.int32) + fixed_size))
        dynamic_augment_cfgs.append([crop_bbox])
    
    init_crop_bbox = copy.deepcopy(dynamic_augment_cfgs[0][0])
    img_scale = convert_bbox2scale(init_crop_bbox, (height, width))
    position_augments = [rot, flip, init_crop_bbox, img_scale]
    augment_cfgs = (position_augments, pixel_augments)
    return augment_cfgs, dynamic_augment_cfgs

def gambling_changing_curve(changing_mode, changing_ratio=0.8, tcl=args().temp_clip_length):
    if changing_mode == 'static':
        changing_curve = np.zeros(tcl)
    elif changing_mode == 'single_direction':
        changing_curve = random.sample(single_direction_changing_curves(tcl), 1)[0] * changing_ratio * (0.4+random.random()*0.6) \
                            + (np.random.random(tcl) - 0.5) / 100
    elif changing_mode == 'shaking':
        changing_curve = (np.random.random(tcl) - 0.5) / np.random.randint(10, 20)
    elif changing_mode == 'return':
        return_point = np.random.randint(0, tcl)
        changing_curve = random.sample([
            np.sin(np.pi/2 + np.pi/2 * (np.arange(tcl)-return_point).astype(np.float32) / (tcl-1)) - (random.random() + 0.5),
            -np.sin(np.pi/2 + np.pi/2 * (np.arange(tcl)-return_point).astype(np.float32) / (tcl-1)) + (random.random() + 0.5),
            ], 1)[0] * changing_ratio * random.random() + (np.random.random(tcl) - 0.5) / 100
    return changing_curve


def generate_dynamic_augments(frame_ids, augment_cfgs):
    frame_num = len(frame_ids)
    crop_ltrb = augment_cfgs[0][2]
    l,t,r,b = crop_ltrb
    cx, cy = (l+r)/2, (t+b)/2
    w, h = cx - l, cy - t
    
    xys_changings = []
    for axis in ['x', 'y', 'scale']:
        if axis == 'x':
            changing_mode = random.sample(dynamic_camera_changing_modes + ['single_direction']*6 + ['return']*3 + ['static']*1, 1)[0]
            changing_ratio = args().dynamic_changing_ratio
        elif axis == 'y':
            changing_mode = random.sample(['shaking']*1 + ['return']*2 + ['static']*5 + ['single_direction']*2, 1)[0]
            changing_ratio = args().dynamic_changing_ratio / 2
        elif axis == 'scale':
            changing_mode = random.sample(['static']*6, 1)[0]
            changing_ratio = args().dynamic_changing_ratio / 3
        changing_curve = gambling_changing_curve(changing_mode, changing_ratio=changing_ratio, tcl=frame_num)
        if axis == 'scale':
            changing_curve = changing_curve / 3 + 1
        xys_changings.append(changing_curve)
    
    dynamic_augment_cfgs = []
    for ind in range(frame_num):
        dx = w * xys_changings[0][ind]
        dy = h * xys_changings[1][ind]
        dscale = xys_changings[2][ind]
        d_cx = cx + dx 
        d_cy = cy + dy
        d_w, d_h = w*dscale, h*dscale
        d_crop_ltrb = (d_cx - d_w, d_cy - d_h, d_cx + d_w, d_cy + d_h)
        dynamic_augment_cfgs.append([d_crop_ltrb])
    # to make sure that the first frame of the clip is always the world coordinate (the anchor).
    augment_cfgs[0][2] = copy.deepcopy(dynamic_augment_cfgs[0][0])
    return dynamic_augment_cfgs

def check_complete_traj(clip_data):
    trajectory_info, track_ids_flatten = convert_centers_to_trajectory(
        clip_data['person_centers'][None], clip_data['subject_ids'][None], clip_data['cams'][None], clip_data['cam_mask'][None])
    traj2D_gts = trajectory_info['traj2D_gts']
    
    have_complete_traj = (((traj2D_gts!=-2).sum(-1) > 0).sum(-1) == traj2D_gts.shape[2]).sum() > 0
    return have_complete_traj

def check_person_number(clip_data):
    person_centers = clip_data['person_centers']
    #all_person_detected_mask = clip_data['all_person_detected_mask']
    have_visible_person = (person_centers!=-2).sum() > 0 #and all_person_detected_mask.sum() > 0
    #if not have_visible_person:
    #    print('about centermap person number:', all_person_detected_mask, all_person_detected_mask.sum())
    return have_visible_person

def visualize_motion_offset2D(center_cyxs, track_ids, clip_images):
    all_subj_ids = np.unique(track_ids[track_ids!=-1])
    def get_xy(center_info):
        if len(center_info) == 4:
            l, t, r, d = center_info
            y = (t + d)/2
            x = (l + r)/2
        else:
            y, x = center_info
        return y,x
    
    for subj_id in all_subj_ids:
        for ind in range(1, len(clip_images)):
            if subj_id not in track_ids[ind] or subj_id not in track_ids[ind-1]:
                continue
            y, x = get_xy(center_cyxs[ind][track_ids[ind]==subj_id][0])
            y0, x0 = get_xy(center_cyxs[ind-1][track_ids[ind-1]==subj_id][0])

            color = (0,0,255)
            linewidth = 2
            image_lined = cv2.line(clip_images[ind],
                    (int(x0), int(y0)), (int(x), int(y)),
                    color, linewidth, cv2.LINE_AA)#
            if not isinstance(image_lined, np.ndarray):
                image_lined = cv2.UMat.get(image_lined)
            clip_images[ind] = image_lined
    return clip_images
    
    for ind in range(1, len(clip_images)):
        cv2.imshow('motion offsets', clip_images[ind]) #str(ind)
        cv2.waitKey()

def visualize_motion_offset3D(center_czyxs, track_ids, clip_images):
    for ind in range(len(clip_images)):
        print(center_czyxs[ind], track_ids[ind])


def save_video(frame_save_paths, save_path, frame_rate=24):
    if len(frame_save_paths)== 0:
        return 
    height, width = cv2.imread(frame_save_paths[0]).shape[:2]
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
    for frame_path in frame_save_paths:
        writer.write(cv2.imread(frame_path))
    writer.release()

def save_panning_dynamic_test_set(ind, data,itern=0):
    save_dir = '/home/yusun/data_drive3/datasets/static2dynamic_camera/static_video_frames_test'
    video_save_dir = '/home/yusun/data_drive3/datasets/static2dynamic_camera/static_video_mp4s_test'
    seq_name = os.path.basename(os.path.dirname(data['imgpath'][0][0]))
    seq_save_dir = os.path.join(save_dir, str(itern)+'-'+seq_name+'-'+str(ind))
    dyna_seq_save_dir = os.path.join(save_dir, str(itern)+'-'+seq_name+'-dyna'+str(ind))
    os.makedirs(seq_save_dir, exist_ok=True)
    os.makedirs(dyna_seq_save_dir, exist_ok=True)
    os.makedirs(video_save_dir, exist_ok=True)
    frame_num = len(data['imgpath'])
    print(seq_save_dir)
    frame_save_paths = []
    dyna_frame_save_paths = []
    for fid in range(frame_num):
        frame_save_path = os.path.join(seq_save_dir, '{:06d}.jpg'.format(fid))
        dyna_frame_save_path = os.path.join(dyna_seq_save_dir, '{:06d}.jpg'.format(fid))
        image = data['image'][0,fid].numpy().astype(np.uint8)[:,:,::-1]
        dynamic_image = data['dynamic_image'][0,fid].numpy().astype(np.uint8)[:,:,::-1]
        cv2.imwrite(frame_save_path, image)
        cv2.imwrite(dyna_frame_save_path, dynamic_image)
        frame_save_paths.append(frame_save_path)
        dyna_frame_save_paths.append(dyna_frame_save_path)
    for remove_item in ['image', 'image_org', 'centermap', 'centermap_3d','dynamic_image']:
        del data[remove_item]
    np.savez(seq_save_dir+'.npz', annots=data)
    save_video(frame_save_paths, seq_save_dir.replace('_frames_', '_mp4s_')+'.mp4', frame_rate=24)
    save_video(dyna_frame_save_paths, dyna_seq_save_dir.replace('_frames_', '_mp4s_')+'.mp4', frame_rate=24)


def visualize_motion_offset(r, batch_size):
    plotly_figs = []
    for bid in range(batch_size):
        center_cyxs = [(r['person_centers'][bid,i][r['valid_masks'][bid,i,:,2]].numpy() + 1) * img_size / 2.0 for i in range(tcl)]
        # center_cyxs = [(r['full_body_bboxes'][bid,i][r['valid_masks'][bid,i,:,2]].numpy() + 1) * img_size / 2.0 for i in range(tcl)]
        track_ids = [r['subject_ids'][bid,i][r['valid_masks'][bid,i,:,2]].numpy() for i in range(tcl)]
        clip_images = [r['image'][bid,i].numpy().astype(np.uint8)[:,:,::-1] for i in range(tcl)]
        clip_images = visualize_motion_offset2D(center_cyxs, track_ids, clip_images)
        plotly_figs.append(np.stack(clip_images)[:,:,:,::-1])
        #plotly_fig = play_video_clips(np.stack(clip_images)[:,:,:,::-1])
        #center_czyxs = [denormalize_cam_params_to_trans(r['cams'][bid,i][r['cam_mask'][bid,i]].clone(), fovs=r['fovs'][bid,i]) for i in range(tcl)]
        #visualize_motion_offset3D(center_czyxs, track_ids, clip_images)
        #plotly_figs.append(plotly_fig)
    # (seq, clip, items, 512, 512, 3)
    merge_figs2html(np.array(plotly_figs)[:,:,np.newaxis], 'test/{}.html'.format(_))


def visualize_global_state_aitviewer(r):
    from visualization.call_aitviewer import GlobalViewer
    viewer_cfgs_update = {'fps':16, 'playback_fps':16.0}
    global_viewer = GlobalViewer(viewer_cfgs_update=viewer_cfgs_update)
    tcl = args().temp_clip_length

    smpl_poses = torch.stack([r['params'][0,i,:,3:22*3][r['valid_masks'][0,i,:,4]] for i in range(tcl)], 0)
    smpl_poses = torch.cat([smpl_poses, torch.zeros_like(smpl_poses)[:,:,:6]],-1).numpy()
    smpl_betas = torch.stack([r['params'][0,i,:,-10:][r['valid_masks'][0,i,:,5]] for i in range(tcl)], 0).numpy()
    world_trans = torch.stack([r['world_root_trans'][0,i][r['world_cam_mask'][0,i]] for i in range(tcl)], 0).numpy()
    world_grots = torch.stack([r['world_global_rots'][0,i][r['world_cam_mask'][0,i]] for i in range(tcl)], 0)
    world_grots = rotation_matrix_to_angle_axis(world_grots).numpy()

    cam_Ks = r['camMats'][0].numpy()
    cam_Rts = r['camPoses'][0].numpy()
    image_list = [r['imgpath'][i][0] for i in range(tcl)]

    mean_cam_position = cam_Rts[:, :3, 3].mean(0)
    mean_subj_position = world_trans.mean(0).mean(0)
    world2dynamic_div_dynamic2people = 0.5
    dynamic2people_distance = np.linalg.norm(mean_cam_position - mean_subj_position, ord=2)
    
    for subj_ind in range(smpl_poses.shape[1]):
        global_viewer.add_smpl_sequence2scene(smpl_poses[:,subj_ind], smpl_betas[:,subj_ind], world_trans[:,subj_ind], world_grots[:,subj_ind])
    
    global_viewer.add_camera2scene(cam_Ks, cam_Rts)
    global_viewer.add_dynamic_image2scene(image_list, distance=dynamic2people_distance)
    
    cam_position = mean_cam_position + world2dynamic_div_dynamic2people * (mean_cam_position - mean_subj_position)
    cam_target = mean_subj_position
    cam_up = cam_Rts[0, 1, :3] # -self.current_Rt[1, :3]
    global_viewer.show(cam_position=cam_position, cam_target=cam_target, cam_up=cam_up)

def initialize_smpl_models():
    if args().smpl_model_type == 'smpl':
        from smpl_family.smpl import SMPL
        smpl = SMPL(args().smpl_model_path, model_type='smpl')
        from smpl_family.smpla import SMPLA_parser
        smpl_family = SMPLA_parser(args().smpla_model_path, args().smil_model_path, baby_thresh=args().baby_threshold)
        print('Initialized SMPL models')
    elif args().smpl_model_type == 'smplx':
        from smpl_family.smplx import SMPLX
        smpl = SMPLX(args().smplx_model_path, model_type='smplx')
        smpl_family = SMPLX(args().smplxa_model_path, model_type='smplxa')
        print('Initialized SMPLX models') 
    return smpl, smpl_family

def visualize_global_state_open3d(r, smpl_model):
    from visualization.non_blocking_open3d import OnlineVisualizer
    viewer = OnlineVisualizer(faces=smpl_model.faces_tensor.numpy())

    tcl = r['params'].shape[1]
    smpl_poses = torch.stack([r['params'][0,i,:,3:22*3][r['valid_masks'][0,i,:,4]] for i in range(tcl)], 0)
    smpl_betas = torch.stack([r['params'][0,i,:,-10:][r['valid_masks'][0,i,:,4]] for i in range(tcl)], 0)
    world_trans = torch.stack([r['world_root_trans'][0,i][r['world_cam_mask'][0,i]] for i in range(tcl)], 0)
    world_grots = torch.stack([r['world_global_rots'][0,i][r['world_cam_mask'][0,i]] for i in range(tcl)], 0)
    #world_grots = rotation_matrix_to_angle_axis(world_grots)

    world_smpl_poses = torch.cat([world_grots, smpl_poses, torch.zeros_like(smpl_poses)[:,:,:6]],-1).squeeze(1)
    #print(world_smpl_poses.shape, world_trans.shape, smpl_betas.shape)
    world_verts = smpl_model(betas=smpl_betas.squeeze(1), poses=world_smpl_poses, root_align=False)[0] + world_trans
    world_verts = world_verts.numpy()

    cam_Ks = r['camMats'][0].numpy()
    cam_Rts = r['camPoses'][0].numpy()
    image_list = [r['image'][0][i].numpy().astype(np.uint8)[:,:,::-1] for i in range(tcl)]

    viewer.open_window()
    for _ in range(10):
        for i in range(tcl):
            viewer.run(images_dict={0:image_list[i]}, verts_dict={0:world_verts[i]}, cam_pose=cam_Rts[i])
            time.sleep(0.05)
    viewer.close_window()    

def test_video_relative_dataset(dataset, with_3d=False, with_smpl=False, \
            vis_motion_offsets=False, vis_global_aitviewer=False, vis_global_open3d=False, show_dynamic_kp2ds=False, show_kp3ds=False):
    print('testing video clips loading')
    print('configs_yml:', args().configs_yml)
    print('model_version:',args().model_version)

    from visualization.visualization import Visualizer
    test_projection_part = True if args().model_version in [4,5,6,7] else False
    print('test_projection_part:',test_projection_part)

    save_dir = os.path.join(config.project_dir,'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    print('Initialized dataset')

    batch_size = 1
    dataloader = DataLoader(dataset = dataset,batch_size = batch_size,shuffle = True,\
        drop_last = False,pin_memory = True,num_workers = 1)
    visualizer = Visualizer(resolution = (512,512,3), result_img_dir=save_dir,with_renderer=True)
    print('Initialized visualizer')

    checkboard_image_path = '/home/yusun/Infinity/project_data/staff/checkerboard.jpg'

    from visualization.visualization import make_heatmaps, draw_skeleton_multiperson
    from utils.cam_utils import denormalize_cam_params_to_trans
    from utils.projection import perspective_projection_withfovs 

    if with_smpl:
        smpl, smpl_family = initialize_smpl_models()

    img_size = 512
    tcl = args().temp_clip_length
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
        #save_panning_dynamic_test_set(_, r,itern=5)
        #from visualization.plotly_volume_viewer import plot_3dslice, plot_3D_volume,show_plotly_figure, play_video_clips, merge_figs2html
        #show_plotly_figure(volume=r['centermap_3d'][0,0].numpy(), image=r['image'][0,0].numpy().astype(np.uint8))
        if vis_motion_offsets:
            visualize_motion_offset(r, batch_size)
        
        if vis_global_aitviewer:
            visualize_global_state_aitviewer(r)
        
        if vis_global_open3d:
            visualize_global_state_open3d(r, smpl)
            
        for inds in range(tcl):
            img_bsname = os.path.basename(r['imgpath'][inds][0])
            image = r['image'][0,inds].numpy().astype(np.uint8)[:,:,::-1]
            full_kp2d = (r['full_kp2d'][0,inds].numpy() + 1) * img_size / 2.0

            person_centers = (r['person_centers'][0,inds].numpy() + 1) * img_size / 2.0
            subject_ids = r['subject_ids'][0,inds]
            image_kp2d = visualizer.draw_skeleton_multiperson(image.copy(), full_kp2d, bones=bones, cm=cm, label_kp_order=True)

            show_list = [image_kp2d]

            if args().learn_cam_with_fbboxes:
                full_body_bboxes = (r['full_body_bboxes'][0,inds].numpy() + 1) * img_size / 2.0
                bboxes_edges = []
                for bbox in full_body_bboxes[(full_body_bboxes==-2).sum(-1)==0]:
                    x1, y1, x2, y2 = bbox
                    bboxes_edges.append(np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]]))
                image_kp2d = visualizer.draw_skeleton_multiperson(image_kp2d, np.array(bboxes_edges), bones=np.array([[0,1],[1,2],[2,3],[3,0]]), cm=cm)
            
            if test_projection_part and r['cam_mask'][0,inds].sum()>0 and r['valid_masks'][0,inds,:1].sum()>0:
                cam_mask = r['cam_mask'][0,inds]
                kp3d_tp = r['kp_3d'][0,inds][cam_mask].clone()
                kp2d_tp = r['full_kp2d'][0,inds][cam_mask].clone()
                
                pred_cam_t = denormalize_cam_params_to_trans(r['cams'][0,inds][cam_mask].clone(), fovs=r['fovs'][0,inds])
                pred_keypoints_2d = perspective_projection_withfovs(kp3d_tp, translation=pred_cam_t, fovs=r['fovs'][0,inds])
                pred_keypoints_2d = (pred_keypoints_2d + 1) * img_size / 2.0
                
                invalid_mask = np.logical_or(kp3d_tp[:,:,-1]==-2., kp2d_tp[:,:,-1]==-2.)
                pred_keypoints_2d[invalid_mask] = -2.
                image_kp2d_projection = visualizer.draw_skeleton_multiperson(image.copy(), pred_keypoints_2d, bones=bones, cm=cm)
                show_list.append(image_kp2d_projection)

            for pinds, (person_center, subject_id) in enumerate(zip(person_centers,subject_ids)):
                y,x = person_center.astype(np.int32)
                if y>0 and x>0:
                    cv2.circle(image_kp2d, (x,y), 6, [0,0,255],-1)
                    text = '{}'.format(subject_id)
                    cv2.putText(image_kp2d, text, (x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)            

            valid_mask = torch.where(r['valid_masks'][0,inds,:,4])[0]
            if with_smpl and len(valid_mask)>0:
                params, subject_ids = r['params'][0,inds],  r['subject_ids'][0,inds]
                subject_ids = subject_ids[valid_mask]
                pose = params[valid_mask][:,:66].float()
                if r['valid_masks'][0,inds,valid_mask,5].sum()>0:
                    betas = params[valid_mask][:,-10:].float()
                else:
                    betas = torch.zeros(len(pose),10)
                if 'kid_shape_offsets' in r:
                    kso_vmask = r['kid_shape_offsets'][0,inds][valid_mask]!=-1
                    betas = torch.cat([betas, torch.zeros(len(betas), 1)], 1)
                    if (kso_vmask).sum()>0:
                        betas[kso_vmask,-1] = r['kid_shape_offsets'][0,inds][valid_mask][kso_vmask]
                    verts, joints = smpl_family(poses=pose, betas=betas)
                else:
                    verts, joints = smpl(poses=pose, betas=betas, get_skin = True)
            
                if test_projection_part and r['cam_mask'][0,inds].sum()>0:
                    fovs = None
                    if r['cam_mask'][0,inds][valid_mask].sum()>0:
                        trans = denormalize_cam_params_to_trans(r['cams'][0,inds][valid_mask].clone(), fovs=r['fovs'][0,inds])
                        fovs = [torch.arctan(1 / r['fovs'][0,inds]) * 2 * 180 / np.pi] 
                        last_trans = denormalize_cam_params_to_trans(r['cams'][0,max(0,inds-1)][valid_mask].clone(), fovs=r['fovs'][0,max(0,inds-1)])
                    else:
                        trans = r['root_trans_cam'][0,inds][valid_mask]     

                    mesh_num = len(verts)
                    motion_offset_verts_list = [torch.cat([verts.cuda(), verts.cuda()], 0)]
                    motion_offset_trans_list = [torch.cat([trans.cuda(), last_trans.cuda()], 0)]
                    mesh_colors = [torch.cat([torch.Tensor([[.9, .9, .8]]).repeat(mesh_num,1), torch.Tensor([[.5, .5, .8]]).repeat(mesh_num,1)],0)]
                    motion_offset_render_img = visualizer.visualize_renderer_verts_list(copy.deepcopy(motion_offset_verts_list), trans=copy.deepcopy(motion_offset_trans_list), \
                                                    colors=mesh_colors, images=image[None], cam_FOVs=fovs)[0]
                    show_list.append(motion_offset_render_img) 
                  
            if args().dynamic_augment:
                dynamic_image = r['dynamic_image'][0,inds].numpy().astype(np.uint8)[:,:,::-1]
                if dynamic_image[0,0,0]==0:
                    dynamic_image = cv2.resize(cv2.imread(checkboard_image_path), (512,512))
                if show_dynamic_kp2ds:
                    dynamic_kp2ds = (r['dynamic_kp2ds'][0,inds].numpy() + 1) * img_size / 2.0
                    dynamic_image_kp2d = visualizer.draw_skeleton_multiperson(dynamic_image, dynamic_kp2ds, bones=bones, cm=cm, label_kp_order=False)
                    if not isinstance(dynamic_image_kp2d, np.ndarray):
                        dynamic_image_kp2d = dynamic_image_kp2d.get()
                    show_list.append(dynamic_image_kp2d)

                if with_smpl and r['world_cam_mask'][0,inds][valid_mask].sum()>0:
                    params = r['params'][0,inds][valid_mask]
                    world_rots = r['world_global_rots'][0,inds][valid_mask]
                    pose = params[:,3:66].float()
                    betas = torch.zeros(len(pose),10)
                    pose = torch.cat([world_rots, pose],-1).float()
                    verts, joints = smpl(poses=pose, betas=betas, get_skin = True)

                    #trans = denormalize_cam_params_to_trans(r['world_cams'][0,inds][valid_mask].clone(), fovs=r['fovs'][0,inds])
                    trans = r['world_root_trans'][0,inds][valid_mask].clone()
                    # print(_, inds)
                    # print('world_trans', trans)
                    # print('camera_poses', r['camPoses'][0,inds])

                    fovs = [torch.arctan(1 / r['fovs'][0,inds]) * 2 * 180 / np.pi] 
                    world_render_img = visualizer.visualize_renderer_verts_list(copy.deepcopy([verts.cuda()]), trans=[trans.cuda()], images=dynamic_image[None], cam_FOVs=fovs)[0]
                    show_list.append(world_render_img)

                    cam_Rs, cam_Ts = visualizer.prepare_monitor_cam_RT([verts.cuda()])
                    monitor_render_img = visualizer.visualize_renderer_verts_list(copy.deepcopy([verts.cuda()]), trans=copy.deepcopy([trans.cuda()]), cam_FOVs=fovs, cam_Rs=cam_Rs, cam_Ts=cam_Ts)[0]
                    show_list.append(monitor_render_img) 
            
            if show_kp3ds:
                #print(r['valid_masks'][0,inds,0],r['kp_3d'][0,inds,0])
                if r['valid_masks'][0,inds,0,1]:
                    print('drawing pkp3ds')
                    j3ds = r['kp_3d'][0,inds,0]
                    #image = r['image'][0,inds].numpy().astype(np.uint8)[:,:,::-1]
                    pj2d = (j3ds[:,:2] + 1) * img_size / 2.0
                    pj2d[j3ds[:,-1]==-2.] = -2.
                    image_pkp3d = visualizer.draw_skeleton(image.copy(), pj2d, bones=bones, cm=cm)
                    #cv2.imwrite('{}/pkp3d_{}_{}.png'.format(save_dir,_,r['subject_ids'][0,0, 0]), image_pkp3d)
                    show_list.append(image_pkp3d)   

            image2show = np.concatenate(show_list,1)
            cv2.imwrite('{}/{:03d}_{:03d}.png'.format(save_dir,_,inds), image2show)