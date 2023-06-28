from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import torch
import torch.nn as nn
import numpy as np

import config
from config import args
from utils import print_dict
if args().model_precision=='fp16':
    from torch.cuda.amp import autocast
from models.base import Base
from loss_funcs.matching import match_traj_to_3D_2D_gts
from maps_utils.result_parser import reorganize_data, reorganize_gts
from utils.video_utils import reorganize_trajectory_info
from tracker.basetrack import BaseTrack

BN_MOMENTUM = 0.1
default_cfg = {'mode':'val', 'calc_loss': False}#'calc_loss':False, 


class VideoBase(Base):
    def forward(self, feat_inputs, meta_data=None, **cfg):   
        #return self.pure_forward(feat_inputs, meta_data, **cfg) for calculating FLOPs only.
        if cfg['mode'] == 'matching_gts':
            meta_data = reorganize_trajectory_info(meta_data)
            return self.matching_forward(feat_inputs, meta_data, **cfg)
        elif cfg['mode'] == 'parsing':
            return self.parsing_forward(feat_inputs, meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(feat_inputs, meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (parsing/matching_gts)')

    def matching_forward(self, feat_inputs, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.train_forward(feat_inputs, traj2D_gts=meta_data['traj2D_gts'])
                outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        else:
            outputs = self.train_forward(feat_inputs, traj2D_gts=meta_data['traj2D_gts'])
            outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)

        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))
        #print_dict(outputs)
        return outputs

    @torch.no_grad()
    def parsing_forward(self, feat_inputs, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                sequence_length = feat_inputs['image_feature_maps'].shape[0]
                outputs = {'meta_data':{}, 'params':{}}
                    
                memory5D, hidden_state, tracker, init_world_cams, init_world_grots = None, None, None, None, None
                track_id_start = 0
                for iter_num in range(int(np.ceil(sequence_length/float(args().temp_clip_length_eval)))):
                    start, end = iter_num * args().temp_clip_length_eval, (iter_num+1)*args().temp_clip_length_eval
                    seq_inds = feat_inputs['seq_inds'][start:end]
                    seq_inds[:,:3] = seq_inds[:,:3] - seq_inds[[0],:3]
                    # TODO: how to associate the track ids to connect the init_world_cams and init_world_grots
                    split_outputs, hidden_state, memory5D, tracker, init_world_cams, init_world_grots = self.inference_forward(\
                        {'image_feature_maps': feat_inputs['image_feature_maps'][start:end], 'seq_inds': seq_inds, 'optical_flows':feat_inputs['optical_flows'][start:end]}, \
                        hidden_state=hidden_state, memory5D=memory5D, temp_clip_length=args().temp_clip_length_eval, \
                        track_id_start=track_id_start, tracker=tracker, init_world_cams=init_world_cams, init_world_grots=init_world_grots, seq_cfgs = cfg['seq_cfgs'])
                    if split_outputs is None:
                        continue
                    #hidden_state = None
                    split_meta_data = {k:v[start:end] for k, v in meta_data.items()}
                    split_outputs, split_meta_data = self._result_parser.parsing_forward(split_outputs, split_meta_data, cfg)
                    merge_output(split_outputs, split_meta_data, outputs)
                torch.cuda.empty_cache()
        else:
            outputs, hidden_state, tracker = self.inference_forward(feat_inputs)
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
            outputs['meta_data'] = meta_data
            #outputs, meta_data = self.parsing_trajectory(outputs, meta_data, cfg)
        BaseTrack._count = 0 # to refresh the ID back to start from 1
        return outputs

    @torch.no_grad()
    def pure_forward(self, feat_inputs, meta_data, **cfg):
        default_cfgs = {
        'tracker_det_thresh': args().tracker_det_thresh, 
        'tracker_match_thresh': args().tracker_match_thresh,
        'first_frame_det_thresh': args().first_frame_det_thresh, #  to find the target in the first frame
        'accept_new_dets': args().accept_new_dets,
        'new_subject_det_thresh': args().new_subject_det_thresh, 
        'time2forget': args().time2forget, # for avoiding forgeting long-term occlusion subjects, 30 per second
        'large_object_thresh': args().large_object_thresh,
        'suppress_duplicate_thresh': args().suppress_duplicate_thresh,
        'motion_offset3D_norm_limit': args().motion_offset3D_norm_limit,
        'feature_update_thresh': args().feature_update_thresh,
        'feature_inherent': args().feature_inherent,
        'occlusion_cam_inherent_or_interp': args().occlusion_cam_inherent_or_interp, # True for directly inherent, False for interpolation
        'tracking_target_max_num': args().tracking_target_max_num,
        'axis_times': np.array([1.2, 2.5, 25]), #np.array([1.2, 2.5, 16])
        'smooth_pose_shape': args().smooth_pose_shape, 'pose_smooth_coef':args().pose_smooth_coef, 'smooth_pos_cam': False
        }  
        if args().model_precision=='fp16':
            with autocast():
                outputs, hidden_state, memory5D, tracker, init_world_cams = self.inference_forward(
                        {'image_feature_maps': feat_inputs['image_feature_maps'], \
                            'seq_inds': feat_inputs['seq_inds'], 'optical_flows':feat_inputs['optical_flows']},seq_cfgs=default_cfgs)
        else:
            outputs, hidden_state, tracker = self.feed_forward(feat_inputs)
        return outputs
    
    def head_forward(self,x):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

def merge_item(source, target, key):
    if key not in target:
        target[key] = source[key].cpu()
    else:
        target[key] = torch.cat([target[key], source[key].cpu()], 0)


def merge_output(outs, meta_data, outputs):
    keys = ['meta_data', 'params_pred', 'reorganize_idx', 'j3d', 'verts', 'verts_camed_org', \
        'world_cams', 'world_trans', 'world_global_rots',  'world_verts', 'world_j3d', 'world_verts_camed_org',\
        'pj2d_org', 'pj2d','cam_trans','detection_flag', 'pj2d_org_h36m17','joints_h36m17', 'center_confs',\
        'track_ids', 'smpl_thetas', 'smpl_betas']
    for key in keys:
        if key =='meta_data':
            for key1 in meta_data:
                merge_item(meta_data, outputs['meta_data'], key1)
        else:
            if key in outs:
                merge_item(outs, outputs, key)