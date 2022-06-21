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
from maps_utils.kp_group import HeatmapParser
from maps_utils.debug_utils import print_dict
from utils.center_utils import process_gt_center, parse_gt_center3d
from utils.rot_6D import rot6D_to_angular

class ResultParser(nn.Module):
    def __init__(self, with_smpl_parser=True):
        super(ResultParser,self).__init__()
        self.map_size = args().centermap_size
        self.with_smpl_parser = with_smpl_parser
        if args().calc_smpl_mesh and with_smpl_parser:
            self.params_map_parser = SMPLWrapper()
            
        self.heatmap_parser = HeatmapParser()
        self.centermap_parser = CenterMap()
        self.match_preds_to_gts_for_supervision = args().match_preds_to_gts_for_supervision

    def matching_forward(self, outputs, meta_data, cfg):
        if args().model_version in [6,8,9]:
            outputs,meta_data = self.match_params_new(outputs, meta_data, cfg)
        else:
            outputs,meta_data = self.match_params(outputs, meta_data, cfg)
        if 'params_pred' in outputs and self.with_smpl_parser and args().calc_smpl_mesh:
            outputs = self.params_map_parser(outputs,meta_data)
            
        if 'detection_flag' not in outputs:
            outputs['detection_flag'] = self.determine_detection_flag(outputs, meta_data)
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

    def process_reorganize_idx_data_parallel(self,outputs):
        gpu_num = torch.cuda.device_count()
        current_device_id = outputs['params_maps'].device.index
        data_size = outputs['params_maps'].shape[0]
        outputs['reorganize_idx'] += data_size*current_device_id
        return outputs

    def suppressing_silimar_mesh_and_2D_center(self, params_preds, pred_batch_ids, pred_czyxs, top_score, center2D_thresh=5, pose_thresh=2.5): # center2D_thresh=5, pose_thresh=2.5 center2D_thresh=2, pose_thresh=1.2
        pose_params_preds = params_preds[:, args().cam_dim:args().cam_dim+22*args().rot_dim]

        N = len(pred_czyxs)
        center2D_similarity = torch.norm((pred_czyxs[:,1:].unsqueeze(1).repeat(1,N,1) - pred_czyxs[:,1:].unsqueeze(0).repeat(N,1,1)).float(), p=2, dim=-1)
        same_batch_id_mask = pred_batch_ids.unsqueeze(1).repeat(1,N) == pred_batch_ids.unsqueeze(0).repeat(N,1)
        center2D_similarity[~same_batch_id_mask] = center2D_thresh + 1
        similarity = center2D_similarity <= center2D_thresh
        center_similar_inds = torch.where(similarity.sum(-1)>1)[0]

        for s_inds in center_similar_inds:
            pose_angulars = rot6D_to_angular(pose_params_preds[similarity[s_inds]])
            pose_angular_base = rot6D_to_angular(pose_params_preds[s_inds].unsqueeze(0)).repeat(len(pose_angulars), 1)
            pose_similarity = batch_smpl_pose_l2_error(pose_angulars,pose_angular_base)
            sim_past = similarity[s_inds].clone()
            similarity[s_inds,sim_past] = (pose_similarity<pose_thresh)

        score_map = similarity * top_score.unsqueeze(0).repeat(N,1)
        nms_inds = torch.argmax(score_map,1) == torch.arange(N).to(score_map.device)
        return [item[nms_inds] for item in [pred_batch_ids, pred_czyxs, top_score]], nms_inds

    def suppressing_duplicate_mesh(self, outputs):
        # During training, do not use nms to facilitate more thourough learning
        (pred_batch_ids, pred_czyxs, top_score), nms_inds = self.suppressing_silimar_mesh_and_2D_center(
            outputs['params_pred'], outputs['pred_batch_ids'], outputs['pred_czyxs'], outputs['top_score'])
        outputs['params_pred'], outputs['cam_czyx'] = outputs['params_pred'][nms_inds], outputs['cam_czyx'][nms_inds]
        if 'motion_offsets' in outputs:
            outputs['motion_offsets'] = outputs['motion_offsets'][nms_inds]
        outputs.update({'pred_batch_ids': pred_batch_ids, 'pred_czyxs': pred_czyxs, 'top_score': top_score})
        return outputs

    def match_params_new(self, outputs, meta_data, cfg):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'valid_masks','subject_ids', 'verts','cam_mask', 'kid_shape_offsets', 'root_trans', 'cams']
        if args().learn_relative:
            gt_keys+=['depth_info']
        
        exclude_keys = ['heatmap','centermap','AE_joints','person_centers','params_pred','all_person_detected_mask', "person_scales"]

        if cfg['with_nms']:
            outputs = self.suppressing_duplicate_mesh(outputs)
        cam_mask = meta_data['cam_mask']
        center_gts_info_3d = parse_gt_center3d(cam_mask, meta_data['cams'])
        person_centers = meta_data['person_centers'].clone()
        # exclude the 2D body centers with precise 3D center locations.
        person_centers[cam_mask] = -2.
        center_gts_info_2d = process_gt_center(person_centers)    
        
        mc = self.match_gt_pred_3d_2d(center_gts_info_2d, center_gts_info_3d, \
            outputs['pred_batch_ids'], outputs['pred_czyxs'], outputs['top_score'], outputs['cam_czyx'], outputs['center_map_3d'].device, cfg['is_training'], \
            batch_size=len(cam_mask), with_2d_matching=cfg['with_2d_matching'])
        batch_ids, person_ids, matched_pred_ids, center_confs = mc['batch_ids'], mc['person_ids'], mc['matched_ids'], mc['conf']
        
        outputs['params_pred'] = outputs['params_pred'][matched_pred_ids]

        for center_key in ['pred_batch_ids', 'pred_czyxs', 'top_score']:
            outputs[center_key] = outputs[center_key][matched_pred_ids]
        
        # convert current batch id (0,1,2,3,..) on single gpu to the global id on all gpu (16,17,18,19,...)
        outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        
        exclude_keys+=['centermap_3d', 'valid_centermap3d_mask']
        
        outputs,meta_data = reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
        outputs['center_confs'] = center_confs

        return outputs,meta_data

    def match_gt_pred_3d_2d(self, center_gts_info_2d, center_gts_info_3d, pred_batch_ids, pred_czyxs, top_score, cam_czyx, device, is_training,batch_size=1, with_2d_matching=True):
        vgt_batch_ids, vgt_person_ids, vgt_centers = center_gts_info_2d
        vgt_batch_ids_3d, vgt_person_ids_3d, vgt_czyxs = center_gts_info_3d
        mc = {key:[] for key in ['batch_ids', 'matched_ids', 'person_ids', 'conf']}

        # 3D center matching
        for match_ind in torch.arange(len(vgt_batch_ids_3d)):
            batch_id, person_id, center_gt = vgt_batch_ids_3d[match_ind], vgt_person_ids_3d[match_ind], vgt_czyxs[match_ind]
            pids = torch.where(pred_batch_ids==batch_id)[0]
            if len(pids) == 0:
                continue
            center_dist_3d = torch.norm(pred_czyxs[pids].float()-center_gt[None].float().to(device),dim=-1)

            matched_pred_id = pids[torch.argmin(center_dist_3d)]
            mc['batch_ids'].append(batch_id)
            mc['matched_ids'].append(matched_pred_id)
            mc['person_ids'].append(person_id)
            mc['conf'].append(top_score[matched_pred_id])

        # 2D center matching
        for match_ind in torch.arange(len(vgt_batch_ids)):
            batch_id, person_id, center_gt = vgt_batch_ids[match_ind], vgt_person_ids[match_ind], vgt_centers[match_ind]
            pids = torch.where(pred_batch_ids==batch_id)[0]
            if len(pids) == 0:
                continue

            matched_pred_id = pids[torch.argmin(torch.norm(pred_czyxs[pids,1:].float()-center_gt[None].float().to(device),dim=-1))]
            center_matched = pred_czyxs[matched_pred_id].long()
            mc['batch_ids'].append(batch_id)
            mc['matched_ids'].append(matched_pred_id)
            mc['person_ids'].append(person_id)
            mc['conf'].append(top_score[matched_pred_id])

        if args().eval_2dpose:
            for inds, (batch_id, person_id, center_gt) in enumerate(zip(vgt_batch_ids, vgt_person_ids, vgt_centers)):
                if batch_id in pred_batch_ids:
                    center_pred = pred_czyxs[pred_batch_ids==batch_id]
                    matched_id = torch.argmin(torch.norm(center_pred[:,1:].float()-center_gt[None].float().to(device),dim=-1))
                    matched_pred_id = np.where((pred_batch_ids==batch_id).cpu())[0][matched_id]
                    mc['matched_ids'].append(matched_pred_id)
                    mc['batch_ids'].append(batch_id); mc['person_ids'].append(person_id)

        if len(mc['matched_ids'])==0:
            #print('matching failed in match_gt_pred_3d_2d of result_parser.py')
            mc.update({'batch_ids':[0], 'matched_ids':[0], 'person_ids':[0], 'conf':[0]})
        keys_list = list(mc.keys())
        for key in keys_list:
            if key == 'conf':
                mc[key] = torch.Tensor(mc[key]).to(device)
            else:
                mc[key] = torch.Tensor(mc[key]).long().to(device)
            if args().max_supervise_num!=-1 and is_training:
                mc[key] = mc[key][:args().max_supervise_num]
        
        return mc

    def match_params(self, outputs, meta_data, cfg):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'subject_ids', 'valid_masks']
        exclude_keys = ['heatmap','centermap','AE_joints','person_centers','all_person_detected_mask']

        center_gts_info = process_gt_center(meta_data['person_centers'])
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        mc_centers = self.match_gt_pred(center_gts_info, center_preds_info, outputs['center_map'].device, cfg['is_training'])
        batch_ids, flat_inds, person_ids = mc_centers['batch_ids'], mc_centers['flat_inds'], mc_centers['person_ids']
        if len(batch_ids)==0:
            if 'new_training' in cfg:
                if cfg['new_training']:
                    outputs['detection_flag'] = torch.Tensor([False for _ in range(len(meta_data['batch_ids']))]).cuda()
                    outputs['reorganize_idx'] = meta_data['batch_ids'].cuda()
                    return outputs, meta_data
            batch_ids, flat_inds = torch.zeros(1).long().to(outputs['center_map'].device), (torch.ones(1)*self.map_size**2/2.).to(outputs['center_map'].device).long()
            person_ids = batch_ids.clone()
        outputs['detection_flag'] = torch.Tensor([True for _ in range(len(batch_ids))]).cuda()
        
        if 'params_maps' in outputs and 'params_pred' not in outputs:
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)

        outputs, meta_data = reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['centers_pred'] = torch.stack([flat_inds%args().centermap_size, flat_inds//args().centermap_size],1)
        return outputs, meta_data

    def match_gt_pred(self,center_gts_info, center_preds_info, device, is_training):
        vgt_batch_ids, vgt_person_ids, vgt_centers = center_gts_info
        vpred_batch_ids, flat_inds, cyxs, top_score = center_preds_info
        mc = {key:[] for key in ['batch_ids', 'flat_inds', 'person_ids', 'conf']}

        if self.match_preds_to_gts_for_supervision:
            for match_ind in torch.arange(len(vgt_batch_ids)):
                batch_id, person_id, center_gt = vgt_batch_ids[match_ind], vgt_person_ids[match_ind], vgt_centers[match_ind]
                pids = torch.where(vpred_batch_ids==batch_id)[0]
                if len(pids) == 0:
                    continue

                closet_center_ind = pids[torch.argmin(torch.norm(cyxs[pids].float()-center_gt[None].float().to(device),dim=-1))]
                center_matched = cyxs[closet_center_ind].long()
                cy, cx = torch.clamp(center_matched, 0, self.map_size-1)
                flat_ind = cy*args().centermap_size+cx
                mc['batch_ids'].append(batch_id)
                mc['flat_inds'].append(flat_ind)
                mc['person_ids'].append(person_id)
                mc['conf'].append(top_score[closet_center_ind])
            
            keys_list = list(mc.keys())
            for key in keys_list:
                if key != 'conf':
                    mc[key] = torch.Tensor(mc[key]).long().to(device)
                if args().max_supervise_num!=-1 and is_training:
                    mc[key] = mc[key][:args().max_supervise_num]
        else:
            mc['batch_ids'] = vgt_batch_ids.long().to(device)
            mc['flat_inds'] = flatten_inds(vgt_centers.long()).to(device)
            mc['person_ids'] = vgt_person_ids.long().to(device)
            mc['conf'] = torch.zeros(len(vgt_person_ids)).to(device)
        return mc
        
    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        device = maps.device
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        results = maps[batch_ids,flat_inds].contiguous()
        return results

    @torch.no_grad()
    def parse_maps(self, outputs, meta_data, cfg):
        if args().model_version in [6]:
            if cfg['with_nms']:
                outputs = self.suppressing_duplicate_mesh(outputs)
            batch_ids = outputs['pred_batch_ids'].long()
            outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
            outputs['center_confs'] = outputs['top_score']
        else:
            batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'])
        
            if len(batch_ids)==0:
                batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'], top_n_people=1)
                outputs['detection_flag'] = torch.Tensor([False for _ in range(len(batch_ids))]).cuda()

            outputs['centers_pred'] = torch.stack([flat_inds%args().centermap_size, torch.div(flat_inds, args().centermap_size, rounding_mode='floor')], 1)
            outputs['centers_conf'] = self.parameter_sampling(outputs['center_map'], batch_ids, flat_inds, use_transform=True)
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image', 'offsets','imgpath','camMats']
        meta_data = reorganize_gts(meta_data, info_vis, batch_ids)

        if 'pred_batch_ids' in outputs:
            # convert current batch id (0,1,2,3,..) on single gpu to the global id on all gpu (16,17,18,19,...)
            outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        
        return outputs,meta_data

    def parse_kps(self, heatmap_AEs, kp2d_thresh=0.1):
        kps = []
        heatmap_AE_results = self.heatmap_parser.batch_parse(heatmap_AEs.detach())
        for batch_id in range(len(heatmap_AE_results)):
            kp2d, kp2d_conf = heatmap_AE_results[batch_id]
            kps.append(kp2d[np.array(kp2d_conf)>kp2d_thresh])
        return kps

def reorganize_gts_cpu(meta_data, key_list, batch_ids):
    for key in key_list:
        if key in meta_data:
            if isinstance(meta_data[key], torch.Tensor):
                #print(key, meta_data[key].shape, batch_ids)
                meta_data[key] = meta_data[key].cpu()[batch_ids]
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