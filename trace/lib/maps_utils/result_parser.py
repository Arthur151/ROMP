import os,sys
import torch
import torch.nn as nn
import numpy as np 
import logging

import config
from config import args
import constants
if not args().learn_relative:
    from smpl_family.smpl_wrapper import SMPLWrapper
else:
    from smpl_family.smpl_wrapper_relative import SMPLWrapper

from maps_utils.centermap import CenterMap
from maps_utils.debug_utils import print_dict
from maps_utils.matching import matching_gts2preds
from maps_utils.suppress_duplication import suppressing_duplicate_mesh

from utils.video_utils import match_trajectory_gts
from utils.projection import convert_kp2ds2org_images
from utils.center_utils import process_gt_center

class ResultParser(nn.Module):
    def __init__(self, with_smpl_parser=True):
        super(ResultParser,self).__init__()
        self.map_size = args().centermap_size
        self.with_smpl_parser = with_smpl_parser
        if args().calc_smpl_mesh and with_smpl_parser:
            self.params_map_parser = SMPLWrapper()
        
        self.centermap_parser = CenterMap()
        self.match_preds_to_gts_for_supervision = args().match_preds_to_gts_for_supervision

    def matching_forward(self, outputs, meta_data, cfg):
        if args().BEV_matching_gts2preds == 'kp2ds':
            outputs = self.params_map_parser(outputs,meta_data,calc_pj2d_org=False)
            #print_dict(outputs)
            if args().model_version in [6,8,9]:
                outputs,meta_data = self.match_params_new(outputs, meta_data, cfg)
            else:
                outputs,meta_data = self.match_params(outputs, meta_data, cfg)
        else:
            if args().model_version in [6,8,9]:
                outputs,meta_data = self.match_params_new(outputs, meta_data, cfg)
            else:
                outputs,meta_data = self.match_params(outputs, meta_data, cfg)
            if 'params_pred' in outputs and self.with_smpl_parser and args().calc_smpl_mesh:
                outputs = self.params_map_parser(outputs,meta_data)
                
        if 'detection_flag' not in outputs:
            outputs['detection_flag'] = self.determine_detection_flag(outputs, meta_data)
        # DEBUG TypeError: zip argument #1 must support iteration
        # print_dict(outputs)
        
        return outputs,meta_data

    @torch.no_grad()
    def parsing_forward(self, outputs, meta_data, cfg):
        outputs, meta_data = self.parse_maps(outputs, meta_data, cfg)
        if 'params_pred' in outputs and self.with_smpl_parser:
            outputs = self.params_map_parser(outputs,meta_data)
        if 'detection_flag' not in outputs:
            outputs['detection_flag'] = self.determine_detection_flag(outputs, meta_data)
        return outputs, meta_data

    def determine_detection_flag(self, outputs, meta_data):
        detected_ids = torch.unique(outputs['reorganize_idx'])
        detection_flag = torch.Tensor([batch_id in detected_ids for batch_id in meta_data['batch_ids']]).cuda()
        return detection_flag

    def match_params_new(self, outputs, meta_data, cfg):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'valid_masks','subject_ids', 'verts','cam_mask', 'kid_shape_offsets', 'root_trans_cam', 'cams']
        if args().learn_relative:
            gt_keys+=['depth_info']
        if args().learn_cam_with_fbboxes:
            gt_keys+=['full_body_bboxes']
        exclude_keys = ['heatmap','centermap','AE_joints','person_centers','params_pred','all_person_detected_mask', "person_scales", "dynamic_supervise"]

        if cfg['with_nms']:
            outputs = suppressing_duplicate_mesh(outputs)
        
        batch_ids, person_ids, matched_pred_ids, center_confs = matching_gts2preds[args().BEV_matching_gts2preds](meta_data, outputs, cfg)
        
        outputs['params_pred'] = outputs['params_pred'][matched_pred_ids]
        if args().video:
            #print('matching with old image matching.')
            outputs['motion_offsets'] = outputs['motion_offsets'][matched_pred_ids]
            exclude_keys += ['traj3D_gts', 'traj2D_gts', 'Tj_flag', 'traj_gt_ids']

        for center_key in ['pred_batch_ids', 'pred_czyxs', 'top_score']:
            outputs[center_key] = outputs[center_key][matched_pred_ids]
        
        # convert current batch id (0,1,2,3,..) on single gpu to the global id on all gpu (16,17,18,19,...)
        outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        
        exclude_keys+=['centermap_3d', 'valid_centermap3d_mask']
        
        outputs, meta_data = reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
        outputs['center_confs'] = center_confs

        if args().BEV_matching_gts2preds == 'kp2ds':
            output_keys = ['verts', 'j3d', 'joints_h36m17', 'cam', 'global_orient', 'body_pose', 'smpl_betas', 'smpl_thetas', \
                'Age_preds', 'kid_offsets_pred', 'cam_trans', 'pj2d', 'pj2d_h36m17', 'verts_camed']
            for key in output_keys:
                outputs[key] = outputs[key][matched_pred_ids]
            outputs = convert_kp2ds2org_images(outputs, meta_data['offsets'])

        if 'traj3D_gts' in meta_data:
            meta_data['traj3D_gts'], meta_data['traj2D_gts'] = match_trajectory_gts(
                meta_data['traj3D_gts'], meta_data['traj2D_gts'], meta_data['traj_gt_ids'], meta_data['subject_ids'], meta_data['batch_ids'])
        return outputs,meta_data

    def match_params(self, outputs, meta_data, cfg):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'subject_ids', 'valid_masks', 'verts', 'cam_mask', 'kid_shape_offsets', 'root_trans_cam', 'cams']
        if args().learn_relative:
            gt_keys+=['depth_info']
        exclude_keys = ['heatmap','centermap','AE_joints','person_centers','all_person_detected_mask']

        center_gts_info = process_gt_center(meta_data['person_centers'])
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        mc_centers = matching_gts2preds['2D_center'](center_gts_info, center_preds_info, outputs['center_map'].device, cfg['is_training'])
        batch_ids, flat_inds, person_ids = mc_centers['batch_ids'], mc_centers['flat_inds'], mc_centers['person_ids']
        
        if 'params_maps' in outputs and 'params_pred' not in outputs:
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)

        if 'uncertainty_map' in outputs:
            outputs['uncertainty_pred'] = torch.sqrt(self.parameter_sampling(outputs['uncertainty_map'], batch_ids, flat_inds, use_transform=True)**2) + 1
        
        if 'reid_map' in outputs:
            outputs['reid_embeds'] = self.parameter_sampling(outputs['reid_map'], batch_ids, flat_inds, use_transform=True)

        if 'joint_sampler_maps' in outputs:
            outputs['joint_sampler_pred'] = self.parameter_sampling(outputs['joint_sampler_maps'], batch_ids, flat_inds, use_transform=True)
            outputs['joint_sampler'] = self.parameter_sampling(outputs['joint_sampler_maps_filtered'], batch_ids, flat_inds, use_transform=True)
            if 'params_pred' in outputs:
                _check_params_pred_(outputs['params_pred'].shape,len(batch_ids))
                outputs['params_pred'] = self.adjust_to_joint_level_sampling(outputs['params_pred'], outputs['joint_sampler'], outputs['params_maps'], batch_ids)

        outputs, meta_data = reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['center_preds'] = torch.stack([flat_inds%args().centermap_size, flat_inds//args().centermap_size],1) * args().input_size / args().centermap_size
        #outputs['center_confs'] = mc_centers['conf']
        return outputs, meta_data
    
    def adjust_to_joint_level_sampling(self, param_preds, joint_sampler, param_maps, batch_ids):        
        sampler_flat_inds = self.process_joint_sampler(joint_sampler)
        batch, channel = param_maps.shape[:2]
        param_maps = param_maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        for inds, joint_inds in enumerate(constants.joint_sampler_relationship):
            start_inds = joint_inds*args().rot_dim + args().cam_dim
            end_inds = start_inds + args().rot_dim
            _check_params_sampling_(param_maps.shape, start_inds, end_inds, batch_ids, sampler_flat_inds[inds])
            param_preds[...,start_inds:end_inds] = param_maps[...,start_inds:end_inds][batch_ids, sampler_flat_inds[inds]].contiguous()
        return param_preds

    def process_joint_sampler(self, joint_sampler, thresh=0.999):
        # restrain the value within -1~1
        joint_sampler = torch.clamp(joint_sampler, -1*thresh,thresh)
        # convert normalized (-1~1) offsets to x/y coordinates
        joint_sampler = (joint_sampler+1)*self.map_size//2
        xs, ys = joint_sampler[:, ::2], joint_sampler[:, 1::2]
        # convert to 1-dim H*W flattened coords
        sampler_flat_inds = (ys*self.map_size + xs).permute((1,0)).long().contiguous()

        return sampler_flat_inds
        
    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        device = maps.device
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        results = maps[batch_ids,flat_inds].contiguous()
        return results

    @torch.no_grad()
    def parse_maps(self,outputs, meta_data, cfg):
        if 'pred_batch_ids' in outputs:
            if cfg['with_nms'] and args().model_version in [6,9]:
                outputs = suppressing_duplicate_mesh(outputs)
            batch_ids = outputs['pred_batch_ids'].long()
            
            outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
            outputs['center_confs'] = outputs['top_score']
        else:
            batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'])
        
            if len(batch_ids)==0:
                batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'], top_n_people=1)
                outputs['detection_flag'] = torch.Tensor([False for _ in range(len(batch_ids))]).cuda()

        if 'params_pred' not in outputs and 'params_maps' in outputs:
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        if 'center_preds' not in outputs:
            outputs['center_preds'] = torch.stack([flat_inds%args().centermap_size, flat_inds//args().centermap_size],1) * args().input_size / args().centermap_size
            outputs['center_confs'] = self.parameter_sampling(outputs['center_map'], batch_ids, flat_inds, use_transform=True)
        if 'joint_sampler_maps_filtered' in outputs:
            outputs['joint_sampler'] = self.parameter_sampling(outputs['joint_sampler_maps_filtered'], batch_ids, flat_inds, use_transform=True)
            if 'params_pred' in outputs:
                _check_params_pred_(outputs['params_pred'].shape,len(batch_ids))
                self.adjust_to_joint_level_sampling(outputs['params_pred'], outputs['joint_sampler'], outputs['params_maps'], batch_ids)
        if 'reid_map' in outputs:
            outputs['reid_embeds'] = self.parameter_sampling(outputs['reid_map'], batch_ids, flat_inds, use_transform=True)
        if 'uncertainty_map' in outputs:
            outputs['uncertainty_pred'] = torch.sqrt(self.parameter_sampling(outputs['uncertainty_map'], batch_ids, flat_inds, use_transform=True)**2) + 1
        #torch.cuda.empty_cache()
        
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image', 'offsets','imgpath','camMats']
        if len(args().gpu) == 1:
            meta_data = reorganize_gts_cpu(meta_data, info_vis, batch_ids)
        else:
            meta_data = reorganize_gts(meta_data, info_vis, batch_ids)

        if 'pred_batch_ids' in outputs:
            # convert current batch id (0,1,2,3,..) on single gpu to the global id on all gpu (16,17,18,19,...)
            outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        return outputs,meta_data

def reorganize_gts_cpu(meta_data, key_list, batch_ids):
    for key in key_list:
        if key in meta_data:
            if isinstance(meta_data[key], torch.Tensor):
                #print(key, meta_data[key].shape, batch_ids)
                meta_data[key] = meta_data[key].cpu()[batch_ids.cpu()]
            elif isinstance(meta_data[key], list):
                meta_data[key] = [meta_data[key][ind] for ind in batch_ids] #np.array(meta_data[key])[batch_ids.cpu().numpy()]
    return meta_data

def reorganize_gts(meta_data, key_list, batch_ids):
    for key in key_list:
        if key in meta_data:
            if isinstance(meta_data[key], torch.Tensor):
                #print(key, meta_data[key].shape, batch_ids)
                meta_data[key] = meta_data[key][batch_ids]
            elif isinstance(meta_data[key], list):
                meta_data[key] = [meta_data[key][ind] for ind in batch_ids] #np.array(meta_data[key])[batch_ids.cpu().numpy()]
    return meta_data

def reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids):
    exclude_keys += gt_keys
    outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
    info_vis = []
    for key, item in meta_data.items():
        if key not in exclude_keys:
            info_vis.append(key)

    meta_data = reorganize_gts(meta_data, info_vis, batch_ids)
    for gt_key in gt_keys:
        if gt_key in meta_data:
            try:
                meta_data[gt_key] = meta_data[gt_key][batch_ids,person_ids]
            except Exception as error:
                print(gt_key,'meets error: ',error)
    return outputs,meta_data


def flatten_inds(coords):
    coords = torch.clamp(coords, 0, args().centermap_size-1)
    return coords[:,0].long()*args().centermap_size+coords[:,1].long()


def _check_params_pred_(params_pred_shape, batch_length):
    assert len(params_pred_shape)==2, logging.error('outputs[params_pred] dimension less than 2, is {}'.format(len(params_pred_shape)))
    assert params_pred_shape[0]==batch_length, logging.error('sampled length not equal.')

def _check_params_sampling_(param_maps_shape, dim_start, dim_end, batch_ids, sampler_flat_inds_i):
    assert len(param_maps_shape)==3, logging.error('During parameter sampling, param_maps dimension is not equal 3, is {}'.format(len(param_maps_shape)))
    assert param_maps_shape[2]>dim_end>=dim_start, \
    logging.error('During parameter sampling, param_maps dimension -1 is not larger than dim_end and dim_start, they are {},{},{}'.format(param_maps_shape[-1],dim_end,dim_start))
    assert (batch_ids>=param_maps_shape[0]).sum()==0, \
    logging.error('During parameter sampling, batch_ids {} out of boundary, param_maps_shape[0] is {}'.format(batch_ids,param_maps_shape[0]))
    assert (sampler_flat_inds_i>=param_maps_shape[1]).sum()==0, \
    logging.error('During parameter sampling, sampler_flat_inds_i {} out of boundary, param_maps_shape[1] is {}'.format(sampler_flat_inds_i,param_maps_shape[1]))

def _check_centermap3d_(meta_data):
    cam_mask = meta_data['cam_mask']
    if cam_mask.sum()>0:
        from utils.cam_utils import convert_cam_params_to_centermap_coords
        cam_params = meta_data['cams'][cam_mask]
        centermap_coords = convert_cam_params_to_centermap_coords(cam_params)
        print('GT center: {}, mask_inds: {}'.format(centermap_coords, cam_mask))
