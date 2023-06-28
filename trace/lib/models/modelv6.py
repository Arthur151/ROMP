from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
    
from models.base import Base
from models.CoordConv import get_coord_maps, get_3Dcoord_maps, get_3Dcoord_maps_halfz
from models.basic_modules import BasicBlock,Bottleneck,BasicBlock_1D,BasicBlock_3D
import logging
import config
from config import args
import constants
from maps_utils.result_parser import ResultParser
from utils.cam_utils import denormalize_cam_params_to_trans, convert_cam_params_to_centermap_coords
from utils.center_utils import denormalize_center
if args().model_return_loss:
    from loss_funcs import Loss

BN_MOMENTUM = 0.1

class BEV(Base):
    def __init__(self, backbone=None, with_loss=True,**kwargs):
        super(BEV, self).__init__()
        logging.info('Using BEV, 3D center map + forehead sampling + parameter regression')
        self.backbone = backbone
        self._result_parser = ResultParser()
        self._build_head()
        if args().model_return_loss and with_loss:
            self._calc_loss = Loss()

        #self.init_weights()
        #self.backbone.load_pretrain_params()

    def _build_head(self):
        params_num = 146
        self.NUM_JOINTS = 17
        self.fv_conditioned_way = args().fv_conditioned_way
        self.outmap_size = 128 #args().centermap_size
        self.cam_dim = args().cam_dim
        self.output_cfg = {'NUM_PARAMS_MAP':params_num-self.cam_dim, 'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':self.cam_dim}
        
        self.head_cfg = {'NUM_BASIC_BLOCKS':args().head_block_num, 'NUM_CHANNELS': 128}
        self.bv_center_cfg = {'NUM_DEPTH_LEVEL': self.outmap_size//2, 'NUM_BLOCK': 2}
        
        self.backbone_channels = self.backbone.backbone_channels
        self.transformer_cfg = {'INPUT_C':self.head_cfg['NUM_CHANNELS'], 'NUM_CHANNELS': 512}
        self._make_transformer()
        
        self.coordmaps = get_coord_maps(128)
        self.cam3dmap_anchor = torch.from_numpy(constants.get_cam3dmap_anchor(args().FOV, 128)).float() # args().centermap_size
        self.register_buffer('coordmap_3d', get_3Dcoord_maps_halfz(128, z_base=self.cam3dmap_anchor)) # args().centermap_size
        self._make_final_layers(self.backbone_channels)
    
    def _make_transformer(self, drop_ratio=0.2):
        self.position_embeddings = nn.Embedding(self.outmap_size, self.transformer_cfg['INPUT_C'], padding_idx=0)
        self.transformer = nn.Sequential(
            nn.Linear(self.transformer_cfg['INPUT_C'],self.transformer_cfg['NUM_CHANNELS']),
            #nn.BatchNorm1d(self.transformer_cfg['NUM_CHANNELS']),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(self.transformer_cfg['NUM_CHANNELS'],self.transformer_cfg['NUM_CHANNELS']),
            #nn.BatchNorm1d(self.transformer_cfg['NUM_CHANNELS']),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(self.transformer_cfg['NUM_CHANNELS'],self.output_cfg['NUM_PARAMS_MAP']))

    def _make_final_layers(self, input_channels):
        self.det_head = self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP'])
        self.param_head = self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP'], with_outlayer=False)
        if args().learn_dense_correspondence:
            self.dense_iuv_head = self._make_deconv_layer(1,self.head_cfg['NUM_CHANNELS'],self.head_cfg['NUM_CHANNELS'],[4])
            self.iuv_predict = IUV_predict_layer(self.head_cfg['NUM_CHANNELS'])
        
        self._make_bv_center_layers(input_channels,self.bv_center_cfg['NUM_DEPTH_LEVEL']*2)
        self._make_3D_map_refiner()
    
    def _make_head_layers(self, input_channels, output_channels, num_channels=None, with_outlayer=True):
        head_layers = []
        if num_channels is None:
            num_channels = self.head_cfg['NUM_CHANNELS']

        for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
            head_layers.append(nn.Sequential(
                    BasicBlock(input_channels, num_channels,downsample=nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=1,stride=1,padding=0))))
            input_channels = num_channels
        if with_outlayer:
            head_layers.append(nn.Conv2d(in_channels=num_channels,\
                out_channels=output_channels,kernel_size=1,stride=1,padding=0))

        return nn.Sequential(*head_layers)
    
    def _make_deconv_layer(self, num_layers, input_channels, output_channels, num_kernels):
        def _get_deconv_cfg(deconv_kernel):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0
            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i])
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

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
        
        if args().bv_with_fv_condition:
            input_channels = (num_channels + self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP'])*self.outmap_size
        else:
            input_channels = num_channels * self.outmap_size
        inter_channels = 512
        self.bv_out_layers = nn.Sequential(
                    BasicBlock_1D(input_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, output_channels))

    def _make_3D_map_refiner(self):
        self.center_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CENTER_MAP'], self.output_cfg['NUM_CENTER_MAP']))
        self.cam_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CAM_MAP'], self.output_cfg['NUM_CAM_MAP']))
    
    def coarse2fine_localization(self, x):
        maps_fv = self.det_head(x)
        center_maps_fv = maps_fv[:,:self.output_cfg['NUM_CENTER_MAP']]
        # predict the small offset from each anchor at 128 map to meet the real 2D image map: map from 0~1 to 0~4 image coordinates
        cam_maps_offset = maps_fv[:,self.output_cfg['NUM_CENTER_MAP']:self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP']]
        if args().bv_with_fv_condition:
            center_maps_3d, cam_maps_offset_bv = self.fv_conditioned_bv_estimation(x, center_maps_fv, cam_maps_offset)
        else:
            center_maps_3d, cam_maps_offset_bv = self.direct_bv_estimation(x, center_maps_fv)

        center_maps_3d = self.center_map_refiner(center_maps_3d.unsqueeze(1)).squeeze(1)
        #print(cam_maps_offset.unsqueeze(-1).transpose(4,1).shape, self.coordmap_3d.shape)
        # B x 3 x H x W -> B x 1 x H x W x 3  |  B x 3 x D x W -> B x D x 1 x W x 3
        # B x D x H x W x 3 + B x 1 x H x W x 3 + B x D x 1- x W x 3  .to(cam_maps_offset.device)
        cam_maps_3d = self.coordmap_3d + cam_maps_offset.unsqueeze(-1).transpose(4,1).contiguous()
        #print(cam_maps_offset_bv.shape, cam_maps_offset_bv.unsqueeze(2).shape, cam_maps_3d.shape)
        if args().learnbev2adjustZ:
            # cam_maps_offset_bv adjust z-wise only, the right order of cam is Z-Y-X
            cam_maps_3d[:,:,:,:,0] = cam_maps_3d[:,:,:,:,0] + cam_maps_offset_bv.unsqueeze(2).contiguous()
        else:
            cam_maps_3d[:,:,:,:,2] = cam_maps_3d[:,:,:,:,2] + cam_maps_offset_bv.unsqueeze(2).contiguous()
        cam_maps_3d = self.cam_map_refiner(cam_maps_3d.unsqueeze(1).transpose(5,1).squeeze(-1))
        
        return center_maps_3d, cam_maps_3d, center_maps_fv
    
    def parsing_trans3D(self, center_maps_3d, cam_maps_3d):
        detection_flag = torch.Tensor([False for _ in range(len(center_maps_3d))]).cuda()
        #integrated_features = torch.cat([fv_f, x], 1) #F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        center_preds_info_3d = self._result_parser.centermap_parser.parse_3dcentermap_heatmap_adaptive_scale_batch(center_maps_3d)
        pred_batch_ids, pred_czyxs, top_score = center_preds_info_3d
        detection_flag[pred_batch_ids] = True
        
        if len(pred_czyxs)==0:
            center_preds_info_3d = self._result_parser.centermap_parser.parse_3dcentermap_heatmap_adaptive_scale_batch(center_maps_3d, top_n_people=1)
            pred_batch_ids, pred_czyxs, top_score = center_preds_info_3d
            #logging.Warning('no center prediciton')
            #device = center_maps_3d.device
            #center_preds_info_3d = [torch.arange(1).to(device),\
            #torch.Tensor([[self.bv_center_cfg['NUM_DEPTH_LEVEL']//2,self.outmap_size//2,self.outmap_size//2]]).long().to(device), torch.ones(1).to(device)]
        
        return center_preds_info_3d, detection_flag
    
    def mesh_parameter_regression(self, fv_f, cams_preds, pred_batch_ids):
        cam_czyx = denormalize_center(convert_cam_params_to_centermap_coords(cams_preds.clone()), size=self.outmap_size)
        feature_sampled = self.differentiable_person_feature_sampling(fv_f, cam_czyx, pred_batch_ids)
        params_preds = self.transformer(feature_sampled)
        params_preds = torch.cat([cams_preds, params_preds], 1)
        return params_preds, cam_czyx

    def head_forward(self, x):
        center_maps_3d, cam_maps_3d, center_maps_fv = self.coarse2fine_localization(x)
        
        center_preds_info_3d, detection_flag = self.parsing_trans3D(center_maps_3d, cam_maps_3d)
        pred_batch_ids, pred_czyxs, top_score = center_preds_info_3d
        if args().add_offsetmap:
            cams_preds = cam_maps_3d[pred_batch_ids,:,pred_czyxs[:,0],pred_czyxs[:,1],pred_czyxs[:,2]]
        else:
            cams_preds = self.coordmap_3d[0,pred_czyxs[:,0],pred_czyxs[:,1],pred_czyxs[:,2]]

        fv_f = self.param_head(x)
        assert fv_f.shape[2] == self.outmap_size, print('feature map must match the size of output maps.')
        
        params_preds, cam_czyx = self.mesh_parameter_regression(fv_f, cams_preds, pred_batch_ids)
        output = {'params_pred':params_preds.float(), 'cam_czyx':cam_czyx.float(), 'mesh_feature_map':fv_f, 
                'center_map':center_maps_fv.float(),'center_map_3d':center_maps_3d.float().squeeze(),'detection_flag':detection_flag,\
                    'pred_batch_ids': pred_batch_ids.float(), 'pred_czyxs': pred_czyxs.float(), 'top_score': top_score.float()} 
        
        if args().learn_dense_correspondence:
            output.update(self.iuv_predict(self.dense_iuv_head(fv_f)))
        return output
    
    def direct_bv_estimation(self, x, center_maps_fv):
        img_feats = self.bv_pre_layers(x)
        outputs_bv = self.bv_out_layers(img_feats.view(img_feats.size(0), -1, self.outmap_size))
        center_maps_bv = outputs_bv[:, :self.bv_center_cfg['NUM_DEPTH_LEVEL']]
        cam_maps_offset_bv = outputs_bv[:, self.bv_center_cfg['NUM_DEPTH_LEVEL']:]
        center_map_3d = center_maps_fv.repeat(1,self.bv_center_cfg['NUM_DEPTH_LEVEL'],1,1) * \
                        center_maps_bv.unsqueeze(2).repeat(1,1,self.outmap_size,1)
        return center_map_3d, cam_maps_offset_bv

    def fv_conditioned_bv_estimation(self, x, center_maps_fv, cam_maps_offset):
        img_feats = self.bv_pre_layers(x)
        summon_feats = torch.cat([center_maps_fv, cam_maps_offset, img_feats], 1).view(img_feats.size(0), -1, self.outmap_size)
        
        outputs_bv = self.bv_out_layers(summon_feats)
        center_maps_bv = outputs_bv[:, :self.bv_center_cfg['NUM_DEPTH_LEVEL']]
        cam_maps_offset_bv = outputs_bv[:, self.bv_center_cfg['NUM_DEPTH_LEVEL']:]
        center_map_3d = center_maps_fv.repeat(1,self.bv_center_cfg['NUM_DEPTH_LEVEL'],1,1) * \
                        center_maps_bv.unsqueeze(2).repeat(1,1,self.outmap_size,1)
        return center_map_3d, cam_maps_offset_bv

    def differentiable_person_feature_sampling(self, feature, pred_czyxs, pred_batch_ids):
        cz, cy, cx = pred_czyxs[:,0], pred_czyxs[:,1], pred_czyxs[:,2]
        position_encoding = self.position_embeddings(cz)
        feature_sampled = feature[pred_batch_ids, :, cy, cx]

        if args().add_depth_encoding:
            input_features = feature_sampled + position_encoding
        else:
            input_features = feature_sampled
        return input_features
    
    def acquire_maps(self, x):
        center_maps_3d, cam_maps_3d, center_maps_fv = self.coarse2fine_localization(x)
        
        center_preds_info_3d, detection_flag = self.parsing_trans3D(center_maps_3d, cam_maps_3d)
        pred_batch_ids, pred_czyxs, top_score = center_preds_info_3d

        cams_preds = cam_maps_3d[pred_batch_ids,:,pred_czyxs[:,0],pred_czyxs[:,1],pred_czyxs[:,2]]

        fv_f = self.param_head(x)
        params_preds, cam_czyx = self.mesh_parameter_regression(fv_f, cams_preds, pred_batch_ids)
        
        output = {'params_pred': params_preds.float(), \
                'cams_preds':cams_preds.float(), 'cam_czyx':cam_czyx.float(),\
                'center_map':center_maps_fv.float(), 'image_feature_maps':x.float(), 'mesh_feature_map':fv_f, \
                'cam_maps_3d': cam_maps_3d.float(), 'center_map_3d':center_maps_3d.float().squeeze(),\
                'pred_batch_ids': pred_batch_ids.float(), 'pred_czyxs': pred_czyxs.float(), 'top_score': top_score.float(), }
        #torch.cuda.empty_cache()
        return output
    
    def mesh_regression_from_features(self, feature_sampled, pred_czyxs, cams_preds):
        cz = pred_czyxs[:,0]
        position_encoding = self.position_embeddings(cz)
        input_features = feature_sampled + position_encoding

        params_preds = self.transformer(input_features)
        params_preds = torch.cat([cams_preds, params_preds], 1)
        return params_preds

if __name__ == '__main__':
    args().configs_yml = 'configs/v6.yml'
    args().model_version=6
    args().learn_dense_correspondence = True
    from models.build import build_model
    from utils import print_dict
    model = build_model(args().backbone, args().model_version).cuda()
    outputs = model.feed_forward({'image':torch.zeros(2,512,512,3).cuda()})
    print_dict(outputs)


'''

    def direct_sampling(self, param_maps, cams, pred_batch_ids):
        scale, ty, tx = cams[:,0],cams[:,1],cams[:,2]

        H = torch.clamp((ty + 1.) / 2. * self.outmap_size, 0, self.outmap_size-1)
        H_floor = torch.floor(H).long()
        W = torch.clamp((tx + 1.) / 2. * self.outmap_size, 0, self.outmap_size-1)
        W_floor = torch.floor(W).long()

        params_sampled = param_maps[pred_batch_ids, :, H_floor, W_floor]
        return params_sampled

        if args().differentiable_sampling:
            
        else:
            param_maps = self.final_layers[1](x)
            params_preds = self.direct_sampling(param_maps, cams_preds.clone().detach(), pred_batch_ids)


        if args().sampling_aggregation_way == 'round':
            feature_sampled = feature[pred_batch_ids, :, torch.round(H).long(), torch.round(W).long()]
            input_features = feature_sampled + position_encoding
        elif 'interpolation' in args().sampling_aggregation_way:
            H_floor, H_ceil = torch.floor(H).long(), torch.ceil(H).long()
            W_floor, W_ceil = torch.floor(W).long(), torch.ceil(W).long()

            if args().sampling_aggregation_way == 'L1 interpolation':
                feature_sampled = (feature[pred_batch_ids, :, H_floor, W_floor] * ((1.-(H-H_floor)) * (1.-(W-W_floor))).unsqueeze(1) + \
                                    feature[pred_batch_ids, :, H_floor, W_ceil] * ((1.-(H-H_floor)) * (1.-(W_ceil-W))).unsqueeze(1) +\
                                    feature[pred_batch_ids, :, H_ceil, W_floor] * ((1.-(H_ceil-H)) * (1.-(W-W_floor))).unsqueeze(1) +\
                                    feature[pred_batch_ids, :, H_ceil, W_ceil] * ((1.-(H_ceil-H)) * (1.-(W_ceil-W))).unsqueeze(1) ) / 4.
            elif args().sampling_aggregation_way == 'L2 interpolation':
                ffd = torch.sqrt((H-H_floor)**2 + (W-W_floor)**2)
                fcd = torch.sqrt((H-H_floor)**2 + (W_ceil-W)**2)
                cfd = torch.sqrt((H_ceil-H)**2 + (W-W_floor)**2)
                ccd = torch.sqrt((H_ceil-H)**2 + (W_ceil-W)**2)
                sum_dist = ffd + fcd + cfd + ccd

                feature_sampled = (feature[pred_batch_ids, :, H_floor, W_floor] * (1.-(ffd/sum_dist)).unsqueeze(1) + \
                                    feature[pred_batch_ids, :, H_floor, W_ceil] * (1.-(fcd/sum_dist)).unsqueeze(1) +\
                                    feature[pred_batch_ids, :, H_ceil, W_floor] * (1.-(cfd/sum_dist)).unsqueeze(1) +\
                                    feature[pred_batch_ids, :, H_ceil, W_ceil] * (1.-(ccd/sum_dist)).unsqueeze(1) ) / 4.
            input_features = feature_sampled + position_encoding
        elif args().sampling_aggregation_way =='old_interp':
            H_floor, H_floor_weight = torch.floor(H).long(), (1.-(H-torch.floor(H))).unsqueeze(1)
            H_ceil, H_ceil_weight = torch.ceil(H).long(), (1.-(torch.ceil(H)-H)).unsqueeze(1)
            W_floor, W_floor_weight = torch.floor(W).long(), (1.-(W-torch.floor(W))).unsqueeze(1)
            W_ceil, W_ceil_weight = torch.ceil(W).long(), (1.-(torch.ceil(W)-W)).unsqueeze(1)

            input_features = (feature[pred_batch_ids, :, H_floor, W_floor] * H_floor_weight * W_floor_weight +\
                            feature[pred_batch_ids, :, H_floor, W_ceil] * H_floor_weight * W_ceil_weight +\
                            feature[pred_batch_ids, :, H_ceil, W_floor] * H_ceil_weight * W_floor_weight +\
                            feature[pred_batch_ids, :, H_ceil, W_ceil] * H_ceil_weight * W_ceil_weight) / 4.
        else:
            raise NotImplementedError
        
'''
