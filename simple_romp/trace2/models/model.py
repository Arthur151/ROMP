from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


from .basic_modules import BasicBlock, BasicBlock_1D, BasicBlock_3D, ConvGRU, TemporalEncoder, get_coord_maps, get_3Dcoord_maps_zeroz
from .TempTracker import perform_tracking, prepare_complete_trajectory_features_withmemory, infilling_cams_of_low_quality_dets
from .deform_conv import DeformConv

from ..utils.visualize_maps import convert_heatmap, flow2img, plot3DHeatmap, convert_motionmap3D2motionline
from ..utils.utils import rotation_matrix_to_angle_axis, rotation_6d_to_matrix, rot6D_to_angular, OneEuroFilter

BN_MOMENTUM = 0.1
keypoints_select = np.array([4, 5, 7, 8,  16, 17, 18, 19,  20, 21, 24, 35, 36, 12]) # Neck_LSP is not avaialbe in video datasets
def get_cam3dmap_anchor(FOV, centermap_size):
    depth_level = np.array([1, 10, 20, 100], dtype=np.float32)
    map_coord_range_each_level = (np.array([2/64., 25/64., 3/64., 2/64.], dtype=np.float32) * centermap_size).astype(np.int32)
    scale_level = 1/np.tan(np.radians(FOV/2.))/depth_level
    cam3dmap_anchor = []
    scale_cache = 8
    for scale, coord_range in zip(scale_level, map_coord_range_each_level):
        cam3dmap_anchor.append(scale_cache-np.arange(1,coord_range+1)/coord_range*(scale_cache-scale))
        scale_cache = scale
    cam3dmap_anchor = np.concatenate(cam3dmap_anchor)
    return cam3dmap_anchor

def progressive_multiply_global_rotation(grots_offsets, cam_rots, clip_length, init_world_grots=None, accum_way='multiply'):
    grots_offsets = grots_offsets.reshape(-1, clip_length, 6)
    cam_grots = cam_rots.detach().reshape(-1, clip_length, 6)
    clip_num = len(grots_offsets)

    accum_offsets = torch.cumsum(cam_grots, -2)
    world_grots = cam_grots + accum_offsets
    if init_world_grots is not None:
        world_grots = world_grots + init_world_grots
    init_world_grots = accum_offsets[:,[-1]]
    world_grots = torch.stack([rot6D_to_angular(world_grots[ind]) for ind in range(clip_num)], 0).reshape(-1,3)
    return world_grots, None, init_world_grots
    
class TRACE_head(nn.Module):
    def __init__(self, _result_parser, temp_clip_length=8, **kwargs):
        super(TRACE_head, self).__init__()
        self.frame_spawn = temp_clip_length
        self.mesh_feature_dim = 128
        self.backbone_channels = 32
        self.outmap_size =128
        self._result_parser = _result_parser

        self.cam_dim = 3
        self.smpl_pose_dim = 22 * 6
        self.smpl_shape_dim = 21
        self.params_num = self.smpl_pose_dim + self.smpl_shape_dim

        self.keypoint_num = len(keypoints_select)
        self.keypoint_ndim = 2
        self.keypoints_dim = self.keypoint_num * self.keypoint_ndim
        self.weight_ndim = 2
        self.weight_dim = self.keypoint_num * self.weight_ndim

        self.map_size = (self.backbone_channels, self.outmap_size, self.outmap_size)
        self._build_head()
    
    def _build_head(self):
        self.output_cfg = {'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':self.cam_dim, 'NUM_MOTION_MAP':3}
        self.hc = 128
        self.head_cfg = {'NUM_BASIC_BLOCKS':1, 'NUM_CHANNELS': 128}
        self.bv_center_cfg = {'NUM_DEPTH_LEVEL': self.outmap_size//2, 'NUM_BLOCK': 2}
        self.position_embeddings = nn.Embedding(self.outmap_size, self.hc, padding_idx=0)
        self.coordmaps = get_coord_maps(128)
        self.cam3dmap_anchor = torch.from_numpy(get_cam3dmap_anchor(50, 128)).float()
        self.register_buffer('coordmap_3d', get_3Dcoord_maps_zeroz(128, zsize=64)) 
        self._make_final_layers(self.backbone_channels)

        self.temp_model = ConvGRU(input_dim=self.backbone_channels, hidden_dim=self.backbone_channels, kernel_size=3, num_layers=2) # bidirectional ConvGRU

        with_gru = False
        self.temp_smplpose_regressor = TemporalEncoder(with_gru=with_gru, input_size=self.hc, out_size=[6*21], n_gru_layers=1, hidden_size=256)
        self.temp_globalrot_regressor = TemporalEncoder(with_gru=with_gru, input_size=self.hc, out_size=[6 for _ in range(2)], n_gru_layers=1, hidden_size=256)
        self.temp_trans_regressor = TemporalEncoder(with_gru=with_gru, input_size=self.hc+3, out_size=[3], n_gru_layers=1, hidden_size=256) 
        self.temp_smplshape_regressor = TemporalEncoder(with_gru=with_gru, input_size=self.hc, out_size=[self.smpl_shape_dim], n_gru_layers=1, hidden_size=256)
    
        self.deform_motion = self._build_deformable_motion_feature_module_()
            
    def _make_final_layers(self, input_channels):
        self.det_head = self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP']*2, block_num=self.head_cfg['NUM_BASIC_BLOCKS'])
        self.motion_head = self._make_head_layers(input_channels+2, self.output_cfg['NUM_MOTION_MAP'], block_num=2)
        self.param_head = self._make_head_layers(input_channels+2, None, num_channels = self.hc, with_outlayer=False, block_num=2) #self.head_cfg['NUM_BASIC_BLOCKS']
        
        self.cam_motion_head = self._make_head_layers(input_channels+2, 3, block_num=6)
        self.cam_rot_head = nn.Sequential(BasicBlock(self.hc, self.hc, downsample=nn.Conv2d(in_channels=self.hc, out_channels=self.hc, kernel_size=1, stride=1, padding=0)),
                                nn.Conv2d(in_channels=self.hc, out_channels=6,kernel_size=1,stride=1,padding=0))
        
        self._make_bv_center_layers(input_channels,self.bv_center_cfg['NUM_DEPTH_LEVEL']*3)
        self._make_bv_motion_layers(input_channels+2, self.bv_center_cfg['NUM_DEPTH_LEVEL'])
        self._make_3D_map_refiner()
    
    def _make_head_layers(self, input_channels, output_channels, block_num=1, num_channels=None, with_outlayer=True):
        head_layers = []
        if num_channels is None:
            num_channels = self.hc

        for _ in range(block_num):
            head_layers.append(nn.Sequential(
                    BasicBlock(input_channels, num_channels,downsample=nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=1,stride=1,padding=0))))
            input_channels = num_channels
        if with_outlayer:
            head_layers.append(nn.Conv2d(in_channels=num_channels,\
                out_channels=output_channels,kernel_size=1,stride=1,padding=0))

        return nn.Sequential(*head_layers)

    def _make_bv_center_layers(self, input_channels, output_channels):
        num_channels = self.outmap_size // 8
        self.bv_pre_layers = nn.Sequential(
                    nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=1,stride=1,padding=0),\
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),\
                    nn.ReLU(inplace=True),\
                    nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,stride=1,padding=1),\
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),\
                    nn.ReLU(inplace=True),\
                    nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=1,stride=1,padding=0),\
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),\
                    nn.ReLU(inplace=True))

        input_channels = (num_channels + self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP']+self.output_cfg['NUM_MOTION_MAP'])*self.outmap_size
        inter_channels = 512
        self.bv_out_layers = nn.Sequential(
                    BasicBlock_1D(input_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, output_channels))
    
    def _make_bv_motion_layers(self, input_channels, output_channels):
        num_channels = self.outmap_size // 8
        self.bv_motion_pre_layers = nn.Sequential(
                    nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=1,stride=1,padding=0),\
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),\
                    nn.ReLU(inplace=True),\
                    nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,stride=1,padding=1),\
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),\
                    nn.ReLU(inplace=True),\
                    nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=1,stride=1,padding=0),\
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),\
                    nn.ReLU(inplace=True))

        input_channels = (num_channels + self.output_cfg['NUM_MOTION_MAP'])*self.outmap_size
        inter_channels = 512
        self.bv_motion_out_layers = nn.Sequential(
                    BasicBlock_1D(input_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, output_channels))

    def _make_3D_map_refiner(self):
        self.center_map_refiner2 = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CENTER_MAP'], self.output_cfg['NUM_CENTER_MAP']))
        self.cam_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CAM_MAP'], self.output_cfg['NUM_CAM_MAP']))
        self.motion_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_MOTION_MAP'], self.output_cfg['NUM_MOTION_MAP']))

    def _build_deformable_motion_feature_module_(self):
        self.offset_feature_convert = nn.Sequential(BasicBlock(self.backbone_channels, self.backbone_channels),\
            BasicBlock(self.backbone_channels, self.backbone_channels), BasicBlock(self.backbone_channels, self.backbone_channels))
        
        nc = self.backbone_channels
        kh = kw = 3
        dilation, deform_group = 1, 8
        self.feature_offset_predictor = nn.Conv2d(nc, deform_group * 2 * kh * kw,kernel_size=(3, 3),stride=(1, 1), dilation=(dilation, dilation),padding=(1*dilation, 1*dilation),bias=False)
        self.deform_warper = DeformConv(nc, nc, (kh, kw), stride=1, padding=int(kh/2)*dilation, dilation=dilation, deformable_groups=deform_group)
    
    def extract_temporal_features(self, input_maps, hidden_state=None, temp_clip_length=None):
        assert len(input_maps.shape) == 4, print('the dimension of input_maps is supposed to be 4, while we get shape', input_maps.shape)

        if temp_clip_length is None:
            temp_clip_length = self.frame_spawn
        clip_length = min(temp_clip_length, input_maps.shape[0]-1)

        temp_input_maps = input_maps[1:].reshape(-1, clip_length, *self.map_size)
        batch_size = temp_input_maps.shape[0]
        #print(temp_input_maps.shape, [h.shape for h in hidden_state] if hidden_state is not None else '0')
        temp_feature_maps, hidden_state = self.temp_model(temp_input_maps, hidden_state=hidden_state)
        temp_feature_maps = temp_feature_maps.reshape(batch_size*clip_length, *self.map_size)

        # we have to add the zeros position to let every frame get a chance to add the deformable feature, the first frame with zero offset is fine. 
        temporal_feature_difference = (input_maps[1:] - input_maps[:-1]).reshape(-1, clip_length, *self.map_size)
        motion_offset_features = self.offset_feature_convert(temporal_feature_difference.reshape(-1, *self.map_size))
        feature_offsets = self.feature_offset_predictor(motion_offset_features)
        # deformable conv only supports float32 inputs.
        warped_feature_maps = self.deform_warper(input_maps[1:].float(), feature_offsets.float())#.half() # len(input_maps)
        temp_feature_maps = temp_feature_maps + warped_feature_maps
        
        feature_maps = temp_feature_maps + input_maps[1:]

        return feature_maps, hidden_state
    
    def fv_conditioned_bv_estimation(self, x, center_maps_fv, cam_maps_offset, local_res_features):
        img_feats = self.bv_pre_layers(x)
        summon_feats = torch.cat([center_maps_fv, cam_maps_offset, local_res_features, img_feats], 1).reshape(img_feats.size(0), -1, self.outmap_size)
        
        outputs_bv = self.bv_out_layers(summon_feats)
        center_maps_bv = outputs_bv[:, :self.bv_center_cfg['NUM_DEPTH_LEVEL']]
        cam_maps_offset_bv = outputs_bv[:, self.bv_center_cfg['NUM_DEPTH_LEVEL']:self.bv_center_cfg['NUM_DEPTH_LEVEL']*2]
        
        center_map_3d = center_maps_fv.repeat(1,self.bv_center_cfg['NUM_DEPTH_LEVEL'],1,1) * \
                        center_maps_bv.unsqueeze(2).repeat(1,1,self.outmap_size,1)
        return center_map_3d, cam_maps_offset_bv
    
    def coarse2fine_localization(self, x):
        maps_fv = self.det_head(x)
        center_maps_fv = maps_fv[:,:self.output_cfg['NUM_CENTER_MAP']]
        # predict the small offset from each anchor at 128 map to meet the real 2D image map: map from 0~1 to 0~4 image coordinates
        cam_maps_offset = maps_fv[:,self.output_cfg['NUM_CENTER_MAP']:self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP']]
        local_res_features = maps_fv[:,self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP']:self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP']*2]
        # TODO: test whether detech center_maps_fv helps the converge of center_maps_3d
        center_maps_3d, cam_maps_offset_bv = self.fv_conditioned_bv_estimation(x, center_maps_fv.detach(), cam_maps_offset, local_res_features) #cam_maps_offset_bv

        center_maps_3d = self.center_map_refiner2(center_maps_3d.unsqueeze(1)).squeeze(1)

        cam_maps_3d = self.coordmap_3d + cam_maps_offset.unsqueeze(-1).transpose(4,1).contiguous()
        #cam_maps_3d = cam_maps_offset.unsqueeze(-1).transpose(4,1).repeat(1,64,1,1,1).contiguous()
        cam_maps_3d[:,:,:,:,0] = cam_maps_3d[:,:,:,:,0] + cam_maps_offset_bv.unsqueeze(2).repeat(1,1,128,1).contiguous()
        cam_maps_3d = self.cam_map_refiner(cam_maps_3d.unsqueeze(1).transpose(5,1).squeeze(-1))

        return center_maps_3d, center_maps_fv, cam_maps_3d
    
    def fv_conditioned_motion_bv_estimation(self, x, motion_maps_fv):
        img_feats = self.bv_motion_pre_layers(x)
        summon_feats = torch.cat([motion_maps_fv, img_feats], 1).reshape(img_feats.size(0), -1, self.outmap_size)
        motion_map_bv = self.bv_motion_out_layers(summon_feats)
        return motion_map_bv
    
    def motion_regression(self, x):
        motion_maps_fv = self.motion_head(x)
        motion_map_bv = self.fv_conditioned_motion_bv_estimation(x, motion_maps_fv) #cam_maps_offset_bv
        motion_maps_3d = motion_maps_fv.unsqueeze(-1).transpose(4,1).contiguous() + motion_map_bv.unsqueeze(2).unsqueeze(-1).contiguous()
        motion_maps_3d = self.motion_map_refiner(motion_maps_3d.unsqueeze(1).transpose(5,1).squeeze(-1))
        return motion_maps_3d, motion_maps_fv
    
    def separate_regression(self, traj_features, masks=None):
        # learning from the https://github.com/HRNet/DEKR/blob/main/lib/models/hrnet_dekr.py to use separate head for regression of each outputs. 
        # regression considering the temporal information, like mixed fc, or posebert. 
        #root_trans = self.temp_trans_regressor(traj_features, masks)
        world_cam_6Drot = self.temp_globalrot_regressor(traj_features.clone(), masks)
        #world_6Drot, cam_6Drot = world_cam_6Drot[...,:6], world_cam_6Drot[...,6:12]

        smpl_thetas = self.temp_smplpose_regressor(traj_features.clone(), masks)
        smpl_betas = self.temp_smplshape_regressor(traj_features.clone(), masks)
        smpl_params = torch.cat([world_cam_6Drot, smpl_thetas, smpl_betas], -1)
        
        batch_size, clip_length, param_dim = smpl_params.shape
        #print(batch_size, clip_length, param_dim)
        smpl_params = smpl_params.reshape(batch_size * clip_length, param_dim)
        return smpl_params
    
    def image_feature_sampling(self, feature, pred_czyxs, pred_batch_ids):
        cz, cy, cx = pred_czyxs[:,0], pred_czyxs[:,1], pred_czyxs[:,2]
        feature_sampled = feature[pred_batch_ids, :, cy, cx]
        #position_encoding = self.position_embeddings(cz.to(feature.device))
        #input_features = feature_sampled + position_encoding
        return feature_sampled
    
    def parsing_trans2D(self, center_maps_fv):
        center2D_preds_info = self._result_parser.centermap_parser.parse_centermap(center_maps_fv)
        pred_batch_ids = center2D_preds_info[0]
        center_yxs = center2D_preds_info[2].long()
        top_score = center2D_preds_info[3]
        return pred_batch_ids, center_yxs, top_score

    def parsing_trans3D_with2D(self, center_maps_3d, pred_batch_ids, center_yxs, only_max=False):
        pred_batch_inds, czyxs, top_scores = self._result_parser.centermap_parser.parse_local_centermap3D(center_maps_3d, pred_batch_ids, center_yxs, only_max=only_max)
        return pred_batch_inds, czyxs, top_scores
    
    def parsing_trans3D(self, center_maps_3d):
        pred_batch_ids, pred_czyxs, top_score = self._result_parser.centermap_parser.parse_3dcentermap_heatmap_adaptive_scale_batch(center_maps_3d)
        
        return pred_batch_ids, pred_czyxs, top_score

    def localization3D_inference(self, cam_maps_3d, cam_motion_maps, feature_maps, pred_batch_ids, pred_czyxs, all_traj_features, \
                            all_traj_masks=None, init_world_cams=None, traj_track_ids=None,xs=1,ys=1,\
                            seq_inherent_flags=None, memory5D=None, inherent_previous=False,smooth_cam=True, pose_smooth_coef=1.):
        cams_init = cam_maps_3d[pred_batch_ids,:,pred_czyxs[:,0],pred_czyxs[:,1],pred_czyxs[:,2]]
        seq_trackIDs = traj_track_ids[:,0]

        seq_num, clip_length, dim = all_traj_features.shape
        trans_features = torch.cat([cams_init.reshape(seq_num, clip_length, 3), all_traj_features], -1)
        normed_cams = self.temp_trans_regressor(trans_features, all_traj_masks)

        normed_cams, memory5D = infilling_cams_of_low_quality_dets(normed_cams, seq_trackIDs, \
                    memory5D, seq_inherent_flags, direct_inherent=inherent_previous, smooth_cam=smooth_cam,pose_smooth_coef=pose_smooth_coef)
        
        # correct camera in the world should be the initial camera parameters + body motion offsets in the world space
        # human motion in the world is supposed to be the relative human body motion compared to the scene. 
        cam_motions = cam_motion_maps[pred_batch_ids,:,pred_czyxs[:,1],pred_czyxs[:,2]].reshape(seq_num, clip_length, 3)
        if xs!=1:
            cam_motions[..., 2] = cam_motions[..., 2] * xs
        if ys!=1:
            cam_motions[..., 1] = cam_motions[..., 1] * ys
        
        if init_world_cams is None:
            init_world_cams = normed_cams[:,[0]].detach().repeat(1,clip_length,1)
        else:
            init_world_cams = torch.stack([init_world_cams[track_id.item()] if track_id.item() in init_world_cams else normed_cams[ind,[0]] for ind, track_id in enumerate(traj_track_ids[:,0])])
            init_world_cams = init_world_cams.repeat(1,clip_length,1)
        
        world_cams = init_world_cams + torch.cumsum(cam_motions, -2)
        if traj_track_ids is not None:
            init_world_cams = {track_id.item(): world_cams[ind,[-1]] for ind, track_id in enumerate(seq_trackIDs)}
        
        if smooth_cam:
            world_cams = self.smooth_world_cams(world_cams, seq_trackIDs, memory5D, seq_inherent_flags, pose_smooth_coef=pose_smooth_coef)

        world_cams = world_cams.reshape(-1, 3)
        normed_cams = normed_cams.reshape(-1, 3)
        
        return normed_cams, cams_init, world_cams, init_world_cams, memory5D
    
    def smooth_world_cams(self, world_cams, seq_trackIDs, memory5D, seq_inherent_flags, pose_smooth_coef=1.):
        for ind, track_id in enumerate(seq_trackIDs):
            track_id = track_id.item()
            clip_cams = world_cams[ind]
            infilling_clip_ids = torch.where(seq_inherent_flags[0][track_id])[0]
            good_clip_ids = torch.where(~seq_inherent_flags[0][track_id])[0]

            if 'world_cams' not in memory5D[0][track_id]:
                memory5D[0][track_id]['world_cams'] = OneEuroFilter(pose_smooth_coef, 0.7)
            
            if len(infilling_clip_ids) > 0:
                for clip_id in infilling_clip_ids:
                    fore_clips_ids = torch.where(~seq_inherent_flags[0][track_id][:clip_id])[0]
                    if len(fore_clips_ids) == 0:
                        if memory5D[0][track_id]['world_cams'].x_filter.prev_raw_value is not None:
                            world_cams[ind,clip_id] = memory5D[0][track_id]['world_cams'].x_filter.prev_raw_value
                        continue
                    after_clips_ids = torch.where(~seq_inherent_flags[0][track_id][clip_id:])[0]
                    if len(after_clips_ids) == 0:
                        world_cams[ind,clip_id] = clip_cams[good_clip_ids[-1]]
                        continue
                    valid_fore_ind = fore_clips_ids[-1]
                    valid_after_ind = after_clips_ids[0] + clip_id
                    world_cams[ind,clip_id] = (valid_after_ind - clip_id) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_fore_ind] + \
                        (clip_id - valid_fore_ind) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_after_ind]
            
            for clip_id in range(len(clip_cams)):
                world_cams[ind,clip_id] = memory5D[0][track_id]['world_cams'].process(clip_cams[clip_id])
        return world_cams

    def smooth_grots(self, params_pred, memory5D, traj_track_ids, seq_inherent_flags, seq_num, pose_smooth_coef=3., rot_angle_thresh=140):
        seq_trackIDs = traj_track_ids[:,0]
        seq_params_pred = params_pred[:,6:].reshape(seq_num, int(params_pred.shape[0]/seq_num), -1).clone()
        param_dim = seq_params_pred.shape[-1]
        for ind, track_id in enumerate(seq_trackIDs):
            track_id = track_id.item()
            grots_pred = seq_params_pred[ind,:,:6]
            pose_shape_pred = seq_params_pred[ind,:,6:]
            infilling_clip_ids = torch.where(seq_inherent_flags[0][track_id])[0]
            good_clip_ids = torch.where(~seq_inherent_flags[0][track_id])[0]
            
            if 'grots' not in memory5D[0][track_id]:
                # memorized grots, filter out jitter times
                memory5D[0][track_id]['grots'] = [grots_pred[good_clip_ids[0]], 0] if len(good_clip_ids)>0 else None
            if 'pose_rots' not in memory5D[0][track_id]:
                memory5D[0][track_id]['pose_rots'] = OneEuroFilter(pose_smooth_coef, 0.7)
            
            for clip_id in range(len(pose_shape_pred)):
                if seq_inherent_flags[0][track_id][clip_id]:
                    if memory5D[0][track_id]['grots'] is not None:
                        grots_pred[clip_id] = memory5D[0][track_id]['grots'][0]
                        memory5D[0][track_id]['grots'][1] = 0
                    if memory5D[0][track_id]['pose_rots'].x_filter.prev_raw_value is not None:
                        pose_shape_pred[clip_id] = memory5D[0][track_id]['pose_rots'].x_filter.prev_raw_value
                else:
                    if memory5D[0][track_id]['grots'] is not None:
                        axis_angle_diff = angle_between(grots_pred[clip_id], memory5D[0][track_id]['grots'][0])
                        #print(clip_id, track_id, axis_angle_diff)
                        if axis_angle_diff>rot_angle_thresh:
                            print(clip_id,track_id, 'error:','axis_angle_diff:',axis_angle_diff)
                            if memory5D[0][track_id]['grots'][1]<=10:
                                grots_pred[clip_id] = memory5D[0][track_id]['grots'][0]
                                memory5D[0][track_id]['grots'][1] += 1
                        else:
                            memory5D[0][track_id]['grots'][0] = grots_pred[clip_id]
                            memory5D[0][track_id]['grots'][1] = 0  
                    pose_shape_pred[clip_id] = memory5D[0][track_id]['pose_rots'].process(pose_shape_pred[clip_id])
            
            seq_params_pred[ind] = torch.cat([grots_pred, pose_shape_pred], -1).clone()
        params_pred[:,6:] = seq_params_pred.reshape(-1, param_dim)
        return params_pred
    
    def smooth_world_grots(self, params_pred, memory5D, traj_track_ids, seq_inherent_flags, seq_num, pose_smooth_coef=3.):
        seq_trackIDs = traj_track_ids[:,0]
        seq_params_pred = params_pred.reshape(seq_num, int(params_pred.shape[0]/seq_num), -1)
        param_dim = seq_params_pred.shape[-1]
        for ind, track_id in enumerate(seq_trackIDs):
            track_id = track_id.item()
            pose_shape_pred = seq_params_pred[ind]
            infilling_clip_ids = torch.where(seq_inherent_flags[0][track_id])[0]
            good_clip_ids = torch.where(~seq_inherent_flags[0][track_id])[0]
            
            if 'world_grots' not in memory5D[0][track_id]:
                memory5D[0][track_id]['world_grots'] = OneEuroFilter(pose_smooth_coef, 0.7)
            
            for clip_id in range(len(pose_shape_pred)):
                if seq_inherent_flags[0][track_id][clip_id]:
                    if memory5D[0][track_id]['world_grots'].x_filter.prev_raw_value is not None:
                        pose_shape_pred[clip_id] = memory5D[0][track_id]['world_grots'].x_filter.prev_raw_value
                else:
                    pose_shape_pred[clip_id] = memory5D[0][track_id]['world_grots'].process(pose_shape_pred[clip_id])
            
            seq_params_pred[ind] = pose_shape_pred #torch.cat([grots_pred, pose_shape_pred], -1)
        params_pred = seq_params_pred.reshape(-1, param_dim)
        return params_pred

    def inference_regression(self, feature_maps, flow=None, memory5D=None, tracker=None, 
                            init_world_cams=None, init_world_grots=None, seq_cfgs=None, xs=1, ys=1):
        # DONETODO: remove the estimation of cam offset front, change the architecture
        center_maps_3d, center_maps_fv, cam_maps_3d = self.coarse2fine_localization(feature_maps)
        
        if flow is not None:
            combined_feature_maps = torch.cat([flow, feature_maps], 1)
        else:
            combined_feature_maps = feature_maps
        mesh_feature_maps = self.param_head(combined_feature_maps)
        motion_map_3d, motion_maps_fv = self.motion_regression(combined_feature_maps)

        motion_feature_maps = self.cam_motion_head[:-2](combined_feature_maps)
        cam_motion_maps = self.cam_motion_head[-2:](motion_feature_maps)
        cam_rot_motion_maps = self.cam_rot_head(motion_feature_maps)

        # drop the motion offset 3D of the 0-th frame in each clip, cause it is not supervised and wrong.
        motion_map_3d[0] = 0.

        clip_length = len(feature_maps)
        # front-view 2D center map is much more stable because it gets better trained. 
        #pred_batch_ids, center_yxs, top_scores = self.parsing_trans2D(center_maps_fv)
        #pred_batch_ids, pred_czyxs, top_scores = self.parsing_trans3D_with2D(center_maps_3d, pred_batch_ids, center_yxs, only_max=False)
        pred_batch_ids, pred_czyxs, top_scores = self.parsing_trans3D(center_maps_3d) 
        pred_batch_ids, pred_czyxs, top_scores, detection_flag = assert_detection_flag(center_maps_3d, pred_batch_ids, pred_czyxs, top_scores, self.outmap_size)

        motion_offsets = motion_map_3d[pred_batch_ids,:,pred_czyxs[:,0],pred_czyxs[:,1],pred_czyxs[:,2]]   
        cams_init = cam_maps_3d[pred_batch_ids,:,pred_czyxs[:,0],pred_czyxs[:,1],pred_czyxs[:,2]]
        sample_trans_features = mesh_feature_maps[pred_batch_ids, :, pred_czyxs[:, 1], pred_czyxs[:, 2]]
        trans_features = torch.cat([cams_init, sample_trans_features], -1)
        init_normed_cams = self.temp_trans_regressor.image_forward(trans_features)

        seq_tracking_results, tracker = perform_tracking(motion_offsets, pred_batch_ids, init_normed_cams, pred_czyxs, top_scores, \
                                                        batch_num=1, clip_length=clip_length, seq_cfgs=seq_cfgs, tracker=tracker)
        if len(seq_tracking_results) == 0:
            print('Oh, No! empty seq_tracking_results!!!!!! nothing detected!!!!!!')
            return None, memory5D, tracker, init_world_cams, init_world_grots
        if len(seq_tracking_results[0][0]) == 0:
            print('Oh, No! empty 1-st seq_tracking_results!!!!!! nothing detected!!!!!!')
            return None, memory5D, tracker, init_world_cams, init_world_grots
        # pred_batch_ids pred_czyxs只是每帧检测的结果， 通过轨迹完整填充，导致真正最后获得的结果并不是 pred_batch_ids对应的，应该重新获得填充后的轨迹对应的traj_batch_inds, traj_czyxs
        traj_features, traj_czyxs, traj_batch_inds, traj_masks, traj_seq_masks, sample_seq_masks, traj_track_ids, seq_inherent_flags, memory5D = \
                        prepare_complete_trajectory_features_withmemory(self, seq_tracking_results, mesh_feature_maps, \
                            memory5D=memory5D, det_conf_thresh=seq_cfgs['feature_update_thresh'], inherent_previous=seq_cfgs['feature_inherent'])
        traj_track_ids = traj_track_ids[0]

        seq_feat_inds = torch.where(sample_seq_masks)[0]
        seq_traj_features = torch.cat([traj_features[i] for i in seq_feat_inds], 0)
        seq_traj_masks = torch.cat([traj_masks[i] for i in seq_feat_inds], 0)
        params_pred = self.separate_regression(seq_traj_features, masks=seq_traj_masks)

        if seq_cfgs['smooth_pose_shape']:
            params_pred = self.smooth_grots(params_pred, memory5D, traj_track_ids, seq_inherent_flags, len(seq_traj_features), pose_smooth_coef=seq_cfgs['pose_smooth_coef'])

        # pred_batch_ids pred_czyxs只是每帧检测的结果， 通过轨迹完整填充，导致真正最后获得的结果并不是 pred_batch_ids对应的，应该重新获得填充后的轨迹对应的traj_batch_inds, traj_czyxs
        pred_batch_ids = torch.cat([traj_batch_inds[i] for i in seq_feat_inds], 0).reshape(-1)
        pred_czyxs = torch.cat([traj_czyxs[i] for i in seq_feat_inds], 0).reshape(-1, 3)

        normed_cams, cams_init, world_cams, init_world_cams, memory5D = self.localization3D_inference(cam_maps_3d, cam_motion_maps, feature_maps, pred_batch_ids, pred_czyxs, \
                                        seq_traj_features, seq_traj_masks, init_world_cams=init_world_cams, traj_track_ids=traj_track_ids, xs=xs, ys=ys,\
                                        seq_inherent_flags=seq_inherent_flags, memory5D=memory5D, \
                                        inherent_previous=seq_cfgs['occlusion_cam_inherent_or_interp'], smooth_cam=seq_cfgs['smooth_pos_cam'], pose_smooth_coef=seq_cfgs['pose_smooth_coef']) # xs=2.5
        
        world_global_rots6D = cam_rot_motion_maps[pred_batch_ids,:,pred_czyxs[:,1],pred_czyxs[:,2]] + params_pred[:,6:12].detach()
        world_global_rots = rot6D_to_angular(world_global_rots6D)

        traj_track_ids = traj_track_ids.reshape(-1).to(pred_czyxs.device)
        valid_results_mask = seq_traj_masks.reshape(-1)
        params_pred = params_pred[valid_results_mask]
        pred_batch_ids = pred_batch_ids[valid_results_mask]
        pred_czyxs = pred_czyxs[valid_results_mask]
        normed_cams = normed_cams[valid_results_mask]
        cams_init = cams_init[valid_results_mask]
        world_cams = world_cams[valid_results_mask]
        traj_track_ids = traj_track_ids[valid_results_mask]        
        world_global_rots = world_global_rots[valid_results_mask]

        motion_offsets3D = motion_map_3d[pred_batch_ids,:,pred_czyxs[:,0],pred_czyxs[:,1],pred_czyxs[:,2]]

        params_pred = torch.cat([normed_cams, params_pred[:,6:]],1)
        
        motion_offsets2D = motion_maps_fv[pred_batch_ids, :, pred_czyxs[:, 1], pred_czyxs[:, 2]]
        top_scores = center_maps_fv[pred_batch_ids, :, pred_czyxs[:, 1], pred_czyxs[:, 2]]

        output = {'params_pred': params_pred, 
                  'cam': normed_cams.float(), 'cams_init': cams_init.float(), 
                  'world_cams': world_cams.float(), 'world_global_rots':world_global_rots, 
                  'motion_offsets3D': motion_offsets3D.float(), 'motion_offsets2D': motion_offsets2D.float(), 
                  'pred_batch_ids': pred_batch_ids.float().to(params_pred.device), 'pred_czyxs': pred_czyxs.float(), 'top_score': top_scores.float(), 
                  'track_ids': traj_track_ids, 
                  'center_map': center_maps_fv, 'center_map_3d': center_maps_3d.float().squeeze(1), 
                  'motion_map_3d': motion_map_3d.float(), 'mesh_feature_map': mesh_feature_maps.float(),
                  'detection_flag': detection_flag}
        
        return output, memory5D, tracker, init_world_cams, init_world_grots
    
    def forward(self, feat_inputs, traj2D_gts=None, temp_states=[None]*5, \
                temp_clip_length=None, track_id_start=0, seq_cfgs=None, xs=1, ys=1):
        image_feature_maps = feat_inputs['image_feature_maps']
        hidden_state, memory5D, tracker, init_world_cams, init_world_grots = temp_states
        feature_maps, hidden_state = self.extract_temporal_features(image_feature_maps, temp_clip_length=temp_clip_length, hidden_state=hidden_state)
        outputs, memory5D, tracker, init_world_cams, init_world_grots = self.inference_regression(feature_maps, flow=feat_inputs['optical_flows'],\
                                                                memory5D=memory5D, tracker=tracker, init_world_cams=init_world_cams, \
                                                                init_world_grots=init_world_grots, seq_cfgs=seq_cfgs, xs=xs, ys=ys)
        if outputs is not None:
            outputs['track_ids'] = track_id_start + outputs['track_ids']
        temp_states = [hidden_state, memory5D, tracker, init_world_cams, init_world_grots]
        return outputs, temp_states

def angle_between(rot1: torch.Tensor, rot2: torch.Tensor):
    r"""
    Calculate the angle in radians between two rotations. (torch, batch)
    :param rot1: Rotation tensor 1 that can reshape to [batch_size, rep_dim].
    :param rot2: Rotation tensor 2 that can reshape to [batch_size, rep_dim].
    :param rep: The rotation representation used in the input.
    :return: Tensor in shape [batch_size] for angles in radians.
    """
    rot_mat1 = rotation_6d_to_matrix(rot1[None])
    rot_mat2 = rotation_6d_to_matrix(rot2[None])
    #print(rot_mat1[:5], rot_mat2[:5])
    offsets = rot_mat1.transpose(1, 2).bmm(rot_mat2)
    angles = rotation_matrix_to_angle_axis(offsets).norm(dim=1)
    return angles[0].item() * 180 / np.pi

def normalize_center(center_on_map):
    return (center_on_map.float() / 128) * 2 - 1

def assert_detection_flag(center_maps, pred_batch_ids, pred_czyxs, top_scores, outmap_size):
    detection_flag = torch.Tensor([False for _ in range(len(center_maps))]).to(center_maps.device)
    if len(pred_czyxs)==0:
        #logging.Warning('no center prediciton')
        device = center_maps.device
        pred_batch_ids = torch.arange(1).to(device)
        pred_czyxs = torch.Tensor([[outmap_size//4,outmap_size//2,outmap_size//2]]).long().to(device)
        top_scores = torch.ones(1).to(device)
    else:
        detection_flag[pred_batch_ids] = True
    return pred_batch_ids, pred_czyxs, top_scores, detection_flag

def visualize_prediction_maps(images, center_maps_3d, center_maps_fv, motion_maps_fv, motion_map_3d):
    images = images.reshape(-1,512,512,3)
    for ind, cm3D in enumerate(center_maps_3d):
        hm2D_fv = convert_heatmap(center_maps_fv[ind, 0])
        flow_map2D = motion_maps_fv[ind, [2,1]].permute(1,2,0).detach().cpu().numpy() * 64
        mm2D_fv = flow2img(flow_map2D)
        
        motion3D_lines = convert_motionmap3D2motionline(motion_map_3d[ind], center_maps_fv[ind, 0])
        plot3DHeatmap(cm3D, images[ind], hm2D_fv, mm2D_fv, motion3D_lines)


if __name__ == '__main__':
    import os
    from models.debug_utils import copy_state_dict, prepare_video_clip_inputs, prepare_bev_model
    video_frame_path = '/home/yusun/DataCenter2/demo_results/temporal_attempts/parkour'
    clip_length = 8
    batch_size = 2
    image_inputs = prepare_video_clip_inputs(clip_length=clip_length, batch_size=batch_size, video_path=video_frame_path)

    trace_model_dir = '/home/yusun/TRACE_models'
    bev_checkpoint = os.path.join(trace_model_dir, 'trace_image_backbone.pkl')
    image_model = prepare_bev_model(bev_checkpoint).cuda()
    
    image_outputs = image_model(image_inputs, **{'mode':'extract_img_feature_maps', 'temp_clip_length_eval':8, 'xs':2, 'ys':2})      
    
    temp_inputs = { 'image': image_inputs['image'], 
                    'image_feature_maps': image_outputs['image_feature_maps'].detach(), \
                    'sequence_mask': image_inputs['sequence_mask']}
    seq_num = 1
    clip_length = len(temp_inputs['image'])
    temp_inputs['seq_inds'] = torch.stack([torch.arange(seq_num).unsqueeze(1).repeat(1,clip_length).reshape(-1),\
                                        torch.arange(clip_length).unsqueeze(0).repeat(seq_num,1).reshape(-1),\
                                        torch.arange(seq_num*clip_length),\
                                        torch.ones(seq_num*clip_length)], 1).long()  
    
    model_dir = os.path.join('/home/yusun/Infinity', 'project_data', 'trace_data', 'model_data')
    smpl_model_path = os.path.join(model_dir, 'parameters', 'SMPL_NEUTRAL.pth')
    model = TRACE(temp_clip_length=8, smpl_model_path=smpl_model_path, centermap_conf_thresh=0.05).cuda()
    checkpoint_path = os.path.join(trace_model_dir, 'trace_head.pkl')
    state_dict = torch.load(checkpoint_path)
    copy_state_dict(model.state_dict(), state_dict)
    outputs, hidden_state = model.inference_forward(temp_inputs)
    for key, value in outputs.items():
        print(key)
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
