import os
import numpy as np
import random
import cv2
import json
import torch
import shutil
import pickle
import copy
import logging
import scipy.io as scio
#import quaternion
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

from config import args
import config
import constants

from smpl_family.smpl_regressor import SMPLR
from utils import estimate_translation, filter_out_incorrect_trans
from utils.augments import get_image_cut_box, process_image, calc_aabb, flip_kps, rot_imgplane, pose_processing, Synthetic_occlusion, convert_bbox2scale
from utils.projection import perspective_projection
from maps_utils.centermap import CenterMap
if args().learn_2dpose: 
    from maps_utils.kp_group import HeatmapParser
if args().learn_AE:
    from maps_utils.target_generators import JointsGenerator

from utils.cam_utils import convert_scale_to_depth_level, normalize_trans_to_cam_params, convert_trans2cam
from utils.center_utils import denormalize_center
from maps_utils.centermap import _calc_radius_
from models.EProPnP6DoFSolver import EProPnP6DoFSolver, prepare_camera_mats

from utils.debug import show_keypoints

class Image_base(Dataset):
    def __init__(self, train_flag=True, regress_smpl = False, vis_mode=False, load_vertices=False, syn_obj_occlusion=False, **kwargs):
        super(Image_base,self).__init__()
        self.data_folder = args().dataset_rootdir
        if not os.path.isdir(self.data_folder):
            self.local_test_mode = True
            self.data_folder = '/media/yusun/DataCenter2/datasets'
        else:
            self.local_test_mode = False
        self.scale_range = [1.4,2.]
        self.half_prob = 0.12
        self.noise = 0.2
        self.vis_thresh = 0.03
        self.channels_mix = False
        self.ID_num = 0
        self.min_vis_pts = 2
        self.vis_mode = vis_mode
        self.load_vertices = load_vertices
        self.syn_obj_occlusion = syn_obj_occlusion
        self.flip_ratio = 0.5 if not args().video else 0
        print('flip prob.', self.flip_ratio, 'rotate prob.', args().rotate_prob)

        self.max_person = args().max_person
        self.multi_mode = args().multi_person
        self.use_eft = args().use_eft

        self.depth_degree_thresh = [0.36,0.18,0]
        self.regress_smpl = regress_smpl
        # 'posetrack' and coco doesn't have the 3D annots for all people, but valid_centermap3d_mask will not use the images without 3D annots for all people for supervision.
        # the quality of the pseudo 3D annots of 'PennAction' is so low that effect the training stability
        self.valid_cam_ds = ['agora', 'h36m', 'mpiinf', 'aist',  'lsp', 'coco', 'posetrack', 'DynaCam', 'egobody', 'sloper4d','bedlam',#  'SMART' doesn't have the pseudo 3D
                'pw3d_vibe', 'pw3d_pc', 'pw3d_nc', 'pw3d_oc', 'pw3d_normal', 'pw3d_od', 'pw3d_cd'] #invalid cam due to wrong shape ['coco', 'lsp', 'mpii', 'up', 'cmup', ]
        self.invalid_kp3ds = ['aist',  'lsp', 'coco', 'posetrack']

        self.homogenize_pose_space = False
        if train_flag:
            self.homogenize_pose_space = args().homogenize_pose_space
            if args().Synthetic_occlusion_ratio>0 and syn_obj_occlusion:
               self.synthetic_occlusion = Synthetic_occlusion(args().voc_dir)
            if args().color_jittering_ratio>0:
                self.color_jittering = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)

        # only shuffle some 3D pose datasets, such as h36m and mpi-inf-3Dhp
        self.shuffle_mode = args().shuffle_crop_mode
        self.aug_cropping_ratio = args().shuffle_crop_ratio_2d
        self.train_flag=train_flag

        self.input_shape = [args().input_size, args().input_size]
        self.vis_size = args().input_size
        self.labels, self.images, self.file_paths = [],[],[]

        self.fov_tan_itw = np.tan(np.radians(args().FOV/2.))
        self.focal_length = args().focal_length

        self.root_inds = [constants.SMPL_ALL_44['R_Hip'], constants.SMPL_ALL_44['L_Hip']]
        self.neck_idx, self.pelvis_idx = 1,8
        self.torso_ids = [constants.SMPL_ALL_44[part] for part in \
            ['Neck', 'Neck_LSP', 'R_Shoulder', 'L_Shoulder','Pelvis', 'R_Hip', 'L_Hip', 'Pelvis_SMPL', 'L_Hip_SMPL', 'R_Hip_SMPL', 'L_Collar', 'R_Collar']]
        self.phase = 'train' if self.train_flag else 'test'
        self.default_valid_mask_3d = np.array([False for _ in range(6)])
        self.heatmap_res = 128
        self.joint_number = len(list(constants.SMPL_ALL_44.keys()))

        if args().learn_2dpose:
            self.heatmap_mapper = constants.joint_mapping(constants.SMPL_ALL_44, constants.COCO_17)
            self.heatmap_generator = HeatmapGenerator(self.heatmap_res,len(self.heatmap_mapper))
        if args().learn_AE:
            self.heatmap_mapper = constants.joint_mapping(constants.SMPL_ALL_44, constants.COCO_17)
            self.joint_generator = JointsGenerator(self.max_person,len(self.heatmap_mapper),128,True)
        self.CM = CenterMap()

        #if args().estimate_camera:
        self.Solver6DoF = EProPnP6DoFSolver() 

    def get_item_single_frame(self,index):
        # valid annotation flags for 
        # 0: 2D pose/bounding box(True/False), # detecting all person/front-view person(True/False)
        # 1: 3D pose, 2: subject id, 3: smpl root rot, 4: smpl pose param, 5: smpl shape param, 6: global translation, 7:verts
        valid_masks = np.zeros((self.max_person, 8), dtype=np.bool_)
        info = self.get_image_info(index)

        position_augments, pixel_augments = self._calc_augment_confs(info['image'], copy.deepcopy(info['kp2ds']), is_pose2d=copy.deepcopy(info['vmask_2d'][:,0]))

        img_info = process_image(info['image'], info['kp2ds'], augments=position_augments, is_pose2d=info['vmask_2d'][:,0])
        image, image_wbg, full_kps, offsets = img_info
        
        centermap, person_centers, full_kp2ds, used_person_inds, valid_masks[:,0], bboxes_hw_norm, heatmap, AE_joints = \
            self.process_kp2ds_bboxes(full_kps, img_shape=image.shape, is_pose2d=info['vmask_2d'][:,0])

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
        
        #input_data = self.add_cam_parameters(input_data, info, position_augments)
        
        if self.load_vertices:
            verts_processed, valid_masks[:,7] = self.process_verts(info['verts'], used_person_inds, \
                augments=position_augments, valid_mask_verts=info['vmask_3d'][:, 4])
            input_data.update({'verts': torch.from_numpy(verts_processed).float()})

        if args().learn_2dpose:
            input_data.update({'heatmap':torch.from_numpy(heatmap).float()})
        if args().learn_AE:
            input_data.update({'AE_joints': torch.from_numpy(AE_joints).long()})

        if args().perspective_proj:
            root_trans, cam_params, cam_mask = self._calc_normed_cam_params_(full_kp2ds, kp3ds, valid_masks[:, 1], info['ds'], fovs=input_data['fovs'])
            centermap_3d, valid_centermap3d_mask = self.generate_centermap_3d(person_centers, cam_params, cam_mask, bboxes_hw_norm, all_person_detected_mask)
            input_data.update({'cams': cam_params,'cam_mask': cam_mask, 'root_trans_cam':root_trans,\
                'centermap_3d':centermap_3d.float(),'valid_centermap3d_mask':torch.Tensor([valid_centermap3d_mask]).bool()})
        
        if 'precise_kp3d_mask' in info:
            input_data['valid_masks'][:,1] = self.reset_precise_kp3d_mask(input_data['valid_masks'][:,1], info['precise_kp3d_mask'], used_person_inds)

        if args().learn_relative:
            input_data.update({'depth_info': torch.ones(self.max_person, 4).long()*-1,\
                'kid_shape_offsets': torch.zeros(self.max_person).float()})
        #print_data_shape(input_data)

        if 'seq_info' in info:
            input_data.update({'seq_info':torch.Tensor(info['seq_info'])})
        return input_data
    
    def _calc_augment_confs(self, image, full_kp2ds, is_pose2d=None):
        if not self.train_flag:
            return None, None
        
        color_jitter = True if random.random()<args().color_jittering_ratio else False
        syn_occlusion = True if random.random()<args().Synthetic_occlusion_ratio and self.syn_obj_occlusion else False
        pixel_augments = (color_jitter, syn_occlusion)
        
        flip = True if random.random()<self.flip_ratio else False
        if args().rotation360_aug:
            rot = random.randint(-180,180) if random.random()<args().rotate_prob else 0
        else:
            rot = random.randint(-30,30) if random.random()<args().rotate_prob else 0                 
        
        height, width = image.shape[0], image.shape[1]
        crop_bbox, img_scale = (0, 0, width, height), 1
        if random.random()<self.aug_cropping_ratio:
            scale = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            
            crop_person_number = np.random.randint(len(full_kp2ds)) + 1
            sample_ids = np.array(random.sample(list(range(len(full_kp2ds))), crop_person_number))
            boxes = np.concatenate([get_bounding_bbox(full_kp2ds[ind]) for ind in sample_ids], 0)

            if len(boxes)>0:
                if flip:
                    boxes[:,0] = width - boxes[:,0]
                box = calc_aabb(boxes)
                box[:,0], box[:,1] = np.clip(box[:,0], 0, width), np.clip(box[:,1], 0, height)
                leftTop, rightBottom = box
                [l, t], [r, b] = get_image_cut_box(leftTop, rightBottom, scale, force_square=True)
                crop_bbox = (l,t,r,b)
                img_scale = convert_bbox2scale(crop_bbox, (height, width))
        
        position_augments = [rot, flip, crop_bbox, img_scale]
        return position_augments, pixel_augments

    def process_kps(self,kps,img_size,set_minus=True):
        kps = kps.astype(np.float32)
        kps[:,0] = kps[:,0]/ float(img_size[1])
        kps[:,1] = kps[:,1]/ float(img_size[0])
        kps[:,:2] = 2.0 * kps[:,:2] - 1.0

        if set_minus:
            if kps.shape[1]>2:
                kps[kps[:,2]<=self.vis_thresh] = -2.
            kps=kps[:,:2]
            for inds, kp in enumerate(kps):
                # set the out-of-region points to -2.
                if not _check_upper_bound_lower_bound_(kp, ub=1, lb=-1):
                    kps[inds] = -2.

        return kps

    def generate_centermap_3d(self, person_centers, cam_params, cam_mask, bboxes_hw_norm, all_person_detected_mask):
        valid_pc_yx, valid_scale = person_centers[cam_mask], cam_params[cam_mask, 0]
        person_num = len(bboxes_hw_norm)
        # if args().model_version>=7:
        #     if args().center3d_loss == 'static' and all_person_detected_mask and cam_mask.sum()!=person_num:
        #         valid_pc_yx, valid_scale = person_centers[:person_num], cam_params[:person_num, 0].clone()
        #         invalid_cp_mask = ~cam_mask[:person_num]
        #         valid_scale[invalid_cp_mask] = torch.Tensor([np.sqrt(h**2+w**2) for h,w in np.array(bboxes_hw_norm)[invalid_cp_mask.numpy()]]).float()
        valid_pc_z = convert_scale_to_depth_level(valid_scale).numpy()
        valid_pc_yx = denormalize_center(valid_pc_yx.copy(), args().centermap_size)
        valid_pc_zyx = np.concatenate([valid_pc_z[:,None], valid_pc_yx], 1)
        centermap_3d, valid_centermap3d_mask = self.CM.generate_centermap_3dheatmap_adaptive_scale(valid_pc_zyx[:,[2,1,0]])
        
        # the center map 3D should have all people with 2D centers, to avoid the wrong supervision caused by only having part of pseudo 3D SMPL annots.
        valid_centermap3d_mask = np.logical_and(valid_centermap3d_mask, person_num==len(valid_pc_zyx))
        return centermap_3d, valid_centermap3d_mask

    def map_kps(self,joint_org,maps=None):
        kps = joint_org[maps].copy()
        kps[maps==-1] = -2.
        return kps

    def parse_multiperson_kp2ds(self, full_kps):
        bboxes_normalized = _calc_bbox_normed(full_kps)
        if args().learn_2dpose or args().learn_AE:
            heatmap, AE_joints = self.generate_heatmap_AEmap(full_kps)
        else:
            heatmap, AE_joints = np.zeros((17, 128, 128)), np.zeros((self.max_person, 17, 2))
        person_centers, full_kp2ds, bboxes_hw_norm, used_person_inds = [], [], [], []
        for inds in range(min(len(full_kps),self.max_person)):
            center = self._calc_center_(full_kps[inds])
            if center is None or len(center)==0:
                continue
            person_centers.append(center)
            full_kp2ds.append(full_kps[inds])
            bboxes_hw_norm.append((bboxes_normalized[inds][1]-bboxes_normalized[inds][0])[::-1])
            used_person_inds.append(inds)
        
        person_centers, full_kp2ds = np.array(person_centers), np.array(full_kp2ds)
        occluded_by_who = detect_occluded_person(person_centers,full_kp2ds) if args().collision_aware_centermap else None
        return person_centers, full_kp2ds, used_person_inds, bboxes_hw_norm, occluded_by_who, heatmap, AE_joints

    def parse_bboxes(self, full_kps, hw_ratio_thresh=0.8):
        person_centers, bboxes_hw_norm, used_person_inds = [], [], []
        fbox, vbox = full_kps[:,:2, :2], full_kps[:, 2:, :2]
        fwh, vwh = fbox[:,1]-fbox[:,0], vbox[:,1]-vbox[:,0]
        fhw_ratios, vhw_ratios = fwh[:,1]/(fwh[:,0]+1e-4), vwh[:,1]/(vwh[:,0]+1e-4)
        fb_centers = np.stack([0.5*fbox[:,0,0]+0.5*fbox[:,1,0], 0.7*fbox[:,0,1]+0.3*fbox[:,1,1]], 1)
        vb_centers = np.stack([0.5*vbox[:,0,0]+0.5*vbox[:,1,0], 0.6*vbox[:,0,1]+0.4*vbox[:,1,1]], 1)
        for inds, (fhw_ratio, vhw_ratio) in enumerate(zip(fhw_ratios, vhw_ratios)):
            if len(used_person_inds)>=self.max_person:
                continue
            if vhw_ratio>1. and _check_upper_bound_lower_bound_(fb_centers[inds], ub=1, lb=-1):
                person_centers.append(fb_centers[inds])
                bboxes_hw_norm.append(fwh[inds][::-1])
                used_person_inds.append(inds)
            elif _check_upper_bound_lower_bound_(vb_centers[inds], ub=1, lb=-1):
                person_centers.append(vb_centers[inds])
                bboxes_hw_norm.append(vwh[inds][::-1])
                used_person_inds.append(inds)

        return person_centers, bboxes_hw_norm, used_person_inds

    def process_kp2ds_bboxes(self, full_kps, img_shape=None, is_pose2d=None):
        person_centers = np.ones((self.max_person, 2))*-2.
        full_kp2ds = np.ones((self.max_person, self.joint_number, 2))*-2.
        valid_mask_kp2ds = np.zeros(self.max_person, dtype=np.bool_)
        used_person_inds, bboxes_hw_norm, occluded_by_who = [], [], None
        
        if is_pose2d.sum()>0:
            # mask out the out-of-range subject because of getting cropped out from image area.
            full_kp2d = [self.process_kps(full_kps[ind], img_shape,set_minus=True) for ind in np.where(is_pose2d)[0]]
            
            person_centers_kp2d, full_kp2ds_kp2d, mask_kp2d, bboxes_hw_norm_kp2d, occluded_by_who, heatmap, AE_joints = self.parse_multiperson_kp2ds(full_kp2d)
            mask_kp2d = np.where(is_pose2d)[0][np.array(mask_kp2d,dtype=np.int32)].tolist()
            
            if len(mask_kp2d)>0:
                person_centers[:len(mask_kp2d)], full_kp2ds[:len(mask_kp2d)], valid_mask_kp2ds[:len(mask_kp2d)] = person_centers_kp2d, full_kp2ds_kp2d, True
                used_person_inds += mask_kp2d
                bboxes_hw_norm += bboxes_hw_norm_kp2d
        
        if (~is_pose2d).sum()>0:
            full_bboxes = np.array([self.process_kps(full_kps[ind], img_shape,set_minus=False) for ind in np.where(~is_pose2d)[0]])
            person_centers_bbox, bboxes_hw_norm_bbox, mask_bbox = self.parse_bboxes(full_bboxes)
            mask_bbox = np.where(~is_pose2d)[0][np.array(mask_bbox,dtype=np.int32)].tolist()
            left_num = max(0, min(self.max_person - len(used_person_inds), len(mask_bbox)))
            if left_num != len(mask_bbox):
                person_centers_bbox, bboxes_hw_norm_bbox, mask_bbox = person_centers_bbox[:left_num], bboxes_hw_norm_bbox[:left_num], mask_bbox[:left_num]
            if len(mask_bbox)>0:
                person_centers[len(used_person_inds):len(used_person_inds)+left_num] = person_centers_bbox
                used_person_inds += mask_bbox
                bboxes_hw_norm += bboxes_hw_norm_bbox

        if is_pose2d.sum() == 0:
            heatmap, AE_joints = np.zeros((17, 128, 128)), np.zeros((self.max_person, 17, 2))
        # person_centers changed after CAR processing
        centermap = self.CM.generate_centermap(person_centers, bboxes_hw_norm=bboxes_hw_norm, occluded_by_who=occluded_by_who)
        # rectify the x, y order, from x-y to y-x, because the image is in H x W, so the results map is in H,W shape, we sample at (y,x)
        person_centers = person_centers[:,::-1].copy()
        return centermap, person_centers, full_kp2ds, used_person_inds, valid_mask_kp2ds, bboxes_hw_norm, heatmap, AE_joints

    def process_suject_ids(self, subject_ids, used_person_inds, valid_mask_ids=False):
        person_ids, valid_id_mask = np.ones(self.max_person)*-1, np.zeros(self.max_person,dtype=np.bool_)
        if subject_ids is None:
            return person_ids, valid_id_mask
        for inds, s_inds in enumerate(used_person_inds):
            person_ids[inds] = subject_ids[s_inds]
            valid_id_mask[inds] = valid_mask_ids[s_inds]
        return person_ids, valid_id_mask

    def prepare_image(self, image, image_wbg, augments=None):
        if augments is not None:
            image = self.aug_image(image, augments[0]) #, augments[1]
        try:
            dst_image = cv2.resize(image, tuple(self.input_shape), interpolation = cv2.INTER_CUBIC)
            org_image = cv2.resize(image_wbg, (self.vis_size, self.vis_size), interpolation=cv2.INTER_CUBIC)
        except:
            dst_image = np.zeros((self.input_shape[0], self.input_shape[1], 3))
            org_image = np.zeros((self.vis_size, self.vis_size, 3))
        return dst_image, org_image
    
    def reset_precise_kp3d_mask(self, valid_kp3d_masks, kp3ds, precise_kp3d_mask, used_person_inds):
        for ind, used_id in enumerate(used_person_inds):
            valid_kp3d_masks[ind] = bool(precise_kp3d_mask[used_id][0])
            if not valid_kp3d_masks[ind]:
                kp3ds[ind] = -2.
        return valid_kp3d_masks

    def process_kp3ds(self, kp3ds, used_person_inds, ds, augments=None, valid_mask_kp3ds=None):
        kp3d_flag = np.zeros(self.max_person, dtype=np.bool_)
        joint_num = self.joint_number if self.train_flag or kp3ds is None else kp3ds[0].shape[0]
        kp3d_processed = np.ones((self.max_person, joint_num, 3), dtype=np.float32)*-2. # -2 serves as an invisible flag

        if ds in self.invalid_kp3ds:
           return kp3d_processed, kp3d_flag

        for inds, used_id in enumerate(used_person_inds):
            if valid_mask_kp3ds[used_id]:
                #print(kp3ds, valid_mask_kp3ds, used_id)
                kp3d, kp3d_flag[inds] = kp3ds[used_id], valid_mask_kp3ds[used_id]
                valid_mask = self._check_kp3d_visible_parts_(kp3d)
                if self.root_inds is not None:
                    kp3d -= kp3d[self.root_inds].mean(0)[None]
                if augments is not None:
                    if augments[0]!=0:
                        kp3d = rot_imgplane(kp3d, augments[0])
                    if augments[1]:
                        kp3d = flip_kps(kp3d,flipped_parts=constants.All44_flip)
                        valid_mask = valid_mask[constants.All44_flip]
                kp3d[~valid_mask] = -2.
                kp3d_processed[inds] = kp3d
             
        return kp3d_processed, kp3d_flag

    def process_smpl_params(self, params, used_person_inds, augments=None, valid_mask_smpl=None):
        params_processed = np.ones((self.max_person,76), dtype=np.float32)*-10
        smpl_flag = np.zeros((self.max_person, 3), dtype=np.bool_)

        for inds, used_id in enumerate(used_person_inds):
            if valid_mask_smpl[used_id].sum()>0:
                param, smpl_flag[inds] = params[used_id], valid_mask_smpl[used_id]
                theta, beta = param[:66], param[-10:]
                if augments is not None:
                    theta = pose_processing(theta,augments[0],augments[1],valid_grot=smpl_flag[inds,0],valid_pose=smpl_flag[inds,1])
                params_processed[inds] = np.concatenate([theta, beta])

        return params_processed, smpl_flag

    def process_verts(self, verts, used_person_inds, augments=None, valid_mask_verts=None):
        # root_trans, , valid_mask_depth=None , root_trans_processed, depth_flag
        verts_processed = np.ones((self.max_person,6890, 3), dtype=np.float32)*-10
        verts_flag = np.zeros(self.max_person, dtype=np.bool_)
        # root_trans_processed = np.ones((self.max_person, 3), dtype=np.float32)*-2
        # depth_flag = np.zeros(self.max_person, dtype=np.bool_)
        # if root_trans is not None:
        #     root_trans_processed[:len(used_person_inds)] = root_trans[used_person_inds]
        #     depth_flag[:len(used_person_inds)] = valid_mask_depth[used_person_inds]
        if verts is not None:
            for inds, used_id in enumerate(used_person_inds):
                if valid_mask_verts[used_id].sum()>0:
                    vert = verts[used_id]# - np.expand_dims(root_trans[used_id], axis=0)
                    if augments is not None:
                        if augments[0]!=0:
                            vert = rot_imgplane(vert, augments[0])
                        if augments[1]:
                            vert[:,0] *= -1
                    verts_processed[inds] = vert
                    verts_flag[inds] = valid_mask_verts[used_id]

        return verts_processed, verts_flag
    
    def solving_trans3D(self, full_kp2ds, kp_3ds, fovs):
        x3ds = torch.from_numpy(kp_3ds).float()
        x2ds = torch.from_numpy(full_kp2ds).float()
        w2ds = ((x2ds != -2).sum(-1) * (x3ds != -2).sum(-1) > 0).float()[:,:,None].repeat(1,1,2)
        pose_opt = self.Solver6DoF.solve(x3ds, x2ds, w2ds, fovs)
        root_trans = pose_opt[:,:3]
        cam_params = convert_trans2cam(root_trans, fovs)
        #cam_mask = torch.ones(len(cam_params), dtype=torch.bool)
        return root_trans, cam_params

    def _calc_normed_cam_params_(self, full_kp2ds, kp_3ds, kp3d_flag, ds, fovs=None, thresh=50):
        cam_params = torch.ones(self.max_person,3,dtype=torch.float32)*-2
        cam_mask = torch.zeros(self.max_person,dtype=torch.bool)
        root_trans  = torch.ones(self.max_person,3,dtype=torch.float32)*-2

        valid_kp2d_mask = ((full_kp2ds!=-2).sum(-1)>1).sum(-1)>4
        valid_mask = np.logical_and(valid_kp2d_mask, kp3d_flag)

        if valid_mask.sum()==0 or ds not in self.valid_cam_ds and not self.vis_mode:
            return root_trans, cam_params, cam_mask
        
        # TODO: make error when the fovs not given 0
        assert (fovs==0).sum() == 0, print('the fovs used to solving_trans3D is', fovs)

        root_trans[valid_mask], cam_params[valid_mask]= self.solving_trans3D(full_kp2ds[valid_mask], copy.deepcopy(kp_3ds[valid_mask]), fovs)
        cam_mask[valid_mask] = True
        #print(root_trans[valid_mask], fovs)
        return root_trans, cam_params, cam_mask

        try:
            root_trans[valid_mask], cam_params[valid_mask]= self.solving_trans3D(full_kp2ds[valid_mask], kp_3ds[valid_mask], fovs)
            cam_mask[valid_mask] = True
        except:
            print('solving_trans3D failed')
            print(full_kp2ds[valid_mask])
            pass
        # if args().estimate_camera:
        #     root_trans[kp3d_flag], cam_params[kp3d_flag]= self.solving_trans3D(full_kp2ds[kp3d_flag], kp_3ds[kp3d_flag], fovs)
        #     cam_mask[kp3d_flag] = True
        # else:
        #     kp_3ds_2dformat = kp_3ds[kp3d_flag]
        #     kp2ds = denormalize_kp2ds(full_kp2ds[kp3d_flag])
        #     trans = estimate_translation(kp_3ds_2dformat, kp2ds, \
        #             focal_length=self.focal_length, img_size=np.array(self.input_shape))
        #     trans, valid_mask = filter_out_incorrect_trans(kp_3ds_2dformat, trans, kp2ds, thresh=thresh, \
        #             focal_length=self.focal_length, center_offset=torch.Tensor(self.input_shape)/2.)
        #     kp3d_flag[kp3d_flag] = valid_mask
        #     # rectify the root displacement between SMPL 24 and OpenPose 25 body joints
        #     normed_cams = normalize_trans_to_cam_params(trans)
    
        #     if valid_mask.sum()>0:
        #         cam_params[kp3d_flag] = torch.from_numpy(normed_cams).float()
        #         root_trans[kp3d_flag] = trans.float()
        #         cam_mask[kp3d_flag] = True
        return root_trans, cam_params, cam_mask

    def regress_kp3d_from_smpl(self, params, maps=None, genders=None):
        kp3ds = None
        if params is not None and self.regress_smpl:
            kp3ds = []
            for inds, param in enumerate(params):
                if param is not None:
                    pose, beta = np.concatenate([param[:-10], np.zeros(6)]),param[-10:]
                    gender = 'n' if genders is None else genders[inds]
                    verts, kp3d = self.smplr(pose, beta, gender=gender)
                    kp3d = kp3d[0]
                    if maps is not None:
                        kp3d = self.map_kps(kp3d,maps=maps)
                    kp3ds.append(kp3d)
                else:
                    kp3ds.append(None)
        return kp3ds

    def generate_heatmap_AEmap(self, full_kps):
        heatmaps, AE_joints = None, None
        if args().learn_2dpose or args().learn_AE:
            full_kps_hm = [(kps_i+1.)/2.*self.heatmap_res for kps_i in full_kps]
            full_kps_hm = [np.concatenate([kps_i,(kps_i[:,0]>0)[:,None]],-1) for kps_i in full_kps_hm]
            heatmap_kps = []
            for kps in full_kps_hm:
                heatmap_kps.append(kps[self.heatmap_mapper])
        
        if args().learn_2dpose:
            heatmaps = self.heatmap_generator.single_process(heatmap_kps)
        if args().learn_AE:
            AE_joints = self.joint_generator.single_process(heatmap_kps)
        return heatmaps, AE_joints

    def _check_kp3d_visible_parts_(self, kp3ds, invisible_flag=-2.):
        visible_parts_mask = (kp3ds!=invisible_flag).sum(-1) == kp3ds.shape[-1]
        return visible_parts_mask

    def parse_cluster_results(self, cluster_results_file, file_paths):
        annots = np.load(cluster_results_file, allow_pickle=True)
        cluster_results, img_names = annots['kp3ds'], annots['img_names']
        cluster_dict = {os.path.basename(img_name): cluster_id for img_name, cluster_id in zip(img_names, cluster_results)}
        cluster_num = max(cluster_results)+1
        cluster_pool = [[] for _ in range(cluster_num)]
        for inds, img_name in enumerate(file_paths):
            cluster_pool[cluster_dict[os.path.basename(img_name)]].append(inds)
        return cluster_pool

    def homogenize_pose_sample(self, index):
        cluster_num = len(self.cluster_pool)
        return random.sample(self.cluster_pool[index%cluster_num],1)[0]

    def get_image_info(self,index):
        raise NotImplementedError

    def resample(self):
        return self.__getitem__(random.randint(0,len(self)))

    def reget_info(self):
        return self.get_image_info(random.randint(0,len(self)))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return self.get_item_single_frame(index)
        try:
            return self.get_item_single_frame(index)
        except:
            print('eror!!!! datasets length:',len(self))
            index = np.random.randint(len(self))
            return self.get_item_single_frame(index)
    
    def add_syn_occlusion(self, image):
        if syn_occlusion and self.syn_obj_occlusion:
            image = self.synthetic_occlusion(image)[0]
        return image

    def aug_image(self, image, color_jitter):      
        if color_jitter:
            image = np.array(self.color_jittering(Image.fromarray(image)))

        return image

    def read_pkl(self,file_path):
        return pickle.load(open(file_path,'rb'),encoding='iso-8859-1')

    def read_json(self,file_path):
        with open(file_path,'r') as f:
            file = json.load(f)
        return file

    def read_npy(self,file_path):
        return np.load(file_path)

    def _calc_center_(self, kps):
        center = None
        if args().center_def_kp:
            vis = kps[self.torso_ids,0]>-1
            if vis.sum()>1:
                center = kps[self.torso_ids][vis].mean(0)
            elif (kps[:,0]>-1).sum()>2:
                center = kps[kps[:,0]>-1].mean(0)
        else:
            vis = kps[:,0]>-1
            if vis.sum()>self.min_vis_pts:
                center = kps[vis].mean(0)
        return center

    def add_cam_parameters(self, input_data, info, position_augments):
        img_scale = 1 if position_augments is None else position_augments[3]
        input_data['img_scale'] = torch.Tensor([img_scale]).float()

        if 'camMats' in info and args().estimate_camera: # 
            input_data['camMats'] = torch.from_numpy(info['camMats']).float()
            input_data['fovs'] = torch.Tensor(input_data['camMats'][0,0] / input_data['camMats'][0,2]).float()
            # TODO: discussed with MJB, he said it is better to stay the same .
            #input_data['fovs'] = input_data['camMats'][0,0] / input_data['camMats'][0,2] / input_data['img_scale'][0]
        else:
            input_data['camMats'] = torch.tensor([[args().focal_length, 0, self.input_shape[1]/2.],\
                                                  [0, args().focal_length, self.input_shape[0]/2.],\
                                                  [0, 0, 0]])
            input_data['fovs'] = 0 if args().estimate_camera else torch.Tensor([args().focal_length / (self.input_shape[0]/2.)]).float()
        if 'camPoses' in info:
            input_data['camPoses'] = torch.from_numpy(info['camPoses']).float()
            if len(input_data['camPoses']) == 3:
                input_data['camPoses'] = torch.cat([input_data['camPoses'], torch.Tensor([[0,0,0,1]])], 0)
        else:
            input_data['camPoses'] = torch.eye(4)
        
        # if 'camDists' in info:
        #     input_data['camDists'] = torch.from_numpy(info['camDists']).float()
        # else:
        #     input_data['camDists'] = torch.zeros(5).float()
        return input_data

def get_bounding_bbox(full_kp2d):
    full_kp2d = full_kp2d[(full_kp2d!=-2).sum(-1)>=2]

    if len(full_kp2d)>0:
        box = calc_aabb(full_kp2d)
        return box
    else:
        return np.array([[0,0],[512,512]])

def detect_occluded_person(person_centers, full_kp2ds, thresh=2*64/512.):
    person_num = len(person_centers)
    # index of the person at the front of an occlusion
    occluded_by_who = np.ones(person_num)*-1
    if person_num>1:
        for inds, (person_center, kp2d) in enumerate(zip(person_centers, full_kp2ds)):
            dist = np.sqrt(((person_centers-person_center)**2).sum(-1))
            if (dist>0).sum()>0:
                if (dist[dist>0]<thresh).sum()>0:
                    # Comparing the visible keypoint number to justify whether the person is occluded by the others
                    #if (full_kp2ds[inds,:,0]>0).sum() >= (kp2d[:,0]>0).sum():
                    closet_idx = np.where(dist==np.min(dist[dist>0]))[0][0]
                    if occluded_by_who[closet_idx]<0:
                        occluded_by_who[inds] = closet_idx

    return occluded_by_who.astype(np.int32)

def _calc_bbox_normed(full_kps):
    bboxes = []
    for kps_i in full_kps:
        if (kps_i[:,0]>-2).sum()>0:
            bboxes.append(calc_aabb(kps_i[kps_i[:,0]>-2]))
        else:
            bboxes.append(np.zeros((2,2)))

    return bboxes

        
def _check_minus2_error_(kp3ds, acceptable_list=[-2., 0.]):
    kp3ds_flatten = kp3ds[:,1:].reshape(-1, 3)
    for kp3d in kp3ds_flatten:
        if kp3d[0] in acceptable_list and kp3d[1] in acceptable_list and kp3d[2] in acceptable_list:
            continue

        equal_kp_value = (kp3ds_flatten[:,0] == kp3d[0]).long() + (kp3ds_flatten[:,1] == kp3d[1]).long() + (kp3ds_flatten[:,2] == kp3d[2]).long()
        equal_mask = equal_kp_value==3
        if equal_mask.sum()>3:
            print(torch.where((kp3ds[:,:] == kp3d).sum(-1)==3))
            print('there are incorrect process that may break the state of invisible flag -2., and make the value becomes {}'.format(kp3ds_flatten[equal_mask]))
            #raise Exception

def _check_upper_bound_lower_bound_(kps, ub=1, lb=-1):
    for k in kps:
        if k >= ub or k <= lb:
            return False
    return True

def check_and_mkdir(dir):
    os.makedirs(dir,exist_ok=True)

def test_projection_depth(pred_joints, trans_xyz, depth_pred, fov_tan):
    from utils.projection import batch_persp_depth
    projected_kp2d = batch_persp_depth(pred_joints, trans_xyz, depth_pred, fov_tan)
    return projected_kp2d

def denormalize_kp2ds(mat, img_size=args().input_size):
    return (mat+1)/2*img_size

def print_data_shape(data):
    for key,value in data.items():
        if isinstance(value,torch.Tensor):
            print(key,value.shape)
        elif isinstance(value,list):
            print(key,len(value))
        elif isinstance(value,str):
            print(key,value)

def test_image_dataset(datasets,with_3d=False,with_smpl=False):
    #args().configs_yml='configs/basic_training_resnet.yml'
    print('configs_yml:', args().configs_yml)
    print('model_version:',args().model_version)
    
    from visualization.visualization import Visualizer, draw_skeleton_multiperson
    test_projection_part = True if args().model_version in [4,5,6,7] else False
    print('test_projection_part:',test_projection_part)

    save_dir = os.path.join(config.project_dir,'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    print('Initialized datasets')

    batch_size, model_type= 2, 'smpl'
    dataloader = DataLoader(datasets = datasets,batch_size = batch_size,shuffle = True,\
        drop_last = False,pin_memory = True,num_workers = 1)
    visualizer = Visualizer(resolution = (512,512,3), result_img_dir=save_dir,with_renderer=True)

    from visualization.visualization import make_heatmaps
    from utils.cam_utils import denormalize_cam_params_to_trans
    if with_smpl:
        from smpl_family.smpl import SMPL
        smpl = SMPL(args().smpl_model_path, model_type='smpl')
    print('Initialized SMPL models')

    img_size = 512
    if args().joint_num == 44:
        bones, cm = constants.All44_connMat, constants.cm_All54
    elif args().joint_num == 73:
        bones, cm = constants.All73_connMat, constants.cm_All54

    for _,r in enumerate(dataloader):
        if _%100==0:
            print(_)
            print_data_shape(r)
        _check_minus2_error_(r['kp_3d'])

        for inds in range(2):
            img_bsname = os.path.basename(r['imgpath'][inds])
            image = r['image'][inds].numpy().astype(np.uint8)[:,:,::-1]
            full_kp2d = (r['full_kp2d'][inds].numpy() + 1) * img_size / 2.0
            person_centers = (r['person_centers'][inds].numpy() + 1) * img_size / 2.0
            subject_ids = r['subject_ids'][inds]
            image_kp2d = draw_skeleton_multiperson(image.copy(), full_kp2d, bones=bones, cm=cm)
            
            if test_projection_part and r['cam_mask'][inds].sum()>0:
                cam_mask = r['cam_mask'][inds]
                kp3d_tp = r['kp_3d'][inds][cam_mask].clone()
                kp2d_tp = r['full_kp2d'][inds][cam_mask].clone()
                pred_cam_t = denormalize_cam_params_to_trans(r['cams'][inds][cam_mask].clone(), positive_constrain=True)
                print('pred_cam_t:',pred_cam_t.half())
                pred_keypoints_2d = perspective_projection(kp3d_tp,translation=pred_cam_t,focal_length=args().focal_length, normalize=False)+512//2
                invalid_mask = np.logical_or(kp3d_tp[:,:,-1]==-2., kp2d_tp[:,:,-1]==-2.)
                pred_keypoints_2d[invalid_mask] = -2.
                image_kp2d_projection = draw_skeleton_multiperson(image.copy(), pred_keypoints_2d, bones=bones, cm=cm)
                cv2.imwrite('{}/{}_{}_projection.jpg'.format(save_dir,_,img_bsname), image_kp2d_projection)

            for person_center, subject_id in zip(person_centers,subject_ids):
                y,x = person_center.astype(np.int32)
                if y>0 and x>0:
                    image_kp2d[y-10:y+10, x-10:x+10] = [0,0,255]
                    cv2.putText(image_kp2d,'id:{}'.format(subject_id), (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
          
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
            image = r['image_org'][0].numpy().astype(np.uint8)[:,:,::-1]
            valid_mask = torch.where(r['valid_masks'][0,:,4])[0]
            subject_ids = subject_ids[valid_mask]
            pose, betas = params[valid_mask][:,:66].float(), params[valid_mask][:,-10:].float()
            pose = torch.cat([pose, torch.zeros(len(pose),6)],-1).float()
            print(pose[:,12*3:13*3])
            verts, joints = smpl(poses=pose, betas=betas, get_skin = True)
            if r['valid_masks'][0,0,6]:
                verts = verts[0][r['valid_masks'][0,:,6]]

            if test_projection_part:
                if r['cam_mask'][0][valid_mask].sum()>0:
                    trans = denormalize_cam_params_to_trans(r['cams'][0][valid_mask].clone(), positive_constrain=True)
                else:
                    trans = r['root_trans'][0][valid_mask]
                pred_keypoints_2d = perspective_projection(joints,translation=trans,focal_length=args().focal_length, normalize=False)+512//2
   
                render_img = visualizer.visualize_renderer_verts_list([verts.cuda()], trans=[trans.cuda()], images=image[None])[0]
                image_kp2d_projection_smpl24 = draw_skeleton_multiperson(render_img.copy(), pred_keypoints_2d, bones=bones[:23], cm=cm[:23], label_kp_order=True)
                image_kp2d_projection_extra30 = draw_skeleton_multiperson(render_img.copy(), pred_keypoints_2d, bones=bones[23:], cm=cm[23:], label_kp_order=True)
                #cv2.imwrite('{}/op25_regressed_{}.jpg'.format(save_dir,_), image_kp2d_projection)
                rendered_img_bv = visualizer.visualize_renderer_verts_list([verts.cuda()], trans=[trans.cuda()], bird_view=True, auto_cam=True)[0]
                cv2.imwrite('{}/mesh_{}.png'.format(save_dir,_), np.concatenate([render_img, rendered_img_bv, image_kp2d_projection_smpl24, image_kp2d_projection_extra30],1))
            else:
                verts[:,:,2] += 2
                render_img = visualizer.visualize_renderer_verts_list([verts.cuda()], images=image[None])[0]
                cv2.imwrite('{}/mesh_{}.png'.format(save_dir,_), render_img)
        j3ds = r['kp_3d'][0,0]
        image = r['image_org'][0].numpy().astype(np.uint8)[:,:,::-1]
        if r['valid_masks'][0,0,1]:
            pj2d = (j3ds[:,:2] + 1) * img_size / 2.0
            pj2d[j3ds[:,-1]==-2.] = -2.
            image_pkp3d = visualizer.draw_skeleton(image.copy(), pj2d, bones=bones, cm=cm)
            cv2.imwrite('{}/pkp3d_{}_{}.png'.format(save_dir,_,r['subject_ids'][0, 0]), image_pkp3d)

def test_dataset2(datasets,**kwargs):
    scales_dict = {}
    length = len(datasets)
    print(length)
    for inds in range(length):
        r = datasets.__getitem__(inds)
        scales_dict[r['imgpath']] = r['scales']
        name = r['data_set']
    np.savez('../statistic_scales/{}_scales.npz'.format(name),scales=scales_dict)
