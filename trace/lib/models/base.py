from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import torch
import torch.nn as nn

import config
from config import args
from maps_utils.debug_utils import print_dict
if args().model_precision=='fp16':
    from torch.cuda.amp import autocast

BN_MOMENTUM = 0.1
default_cfg = {'mode':'val', 'calc_loss': False}#'calc_loss':False, 

class Base(nn.Module):
    def forward(self, meta_data, **cfg):        
        if cfg['mode'] == 'matching_gts':
            return self.matching_forward(meta_data, **cfg)
        elif cfg['mode'] == 'parsing':
            return self.parsing_forward(meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(meta_data, **cfg)
        elif cfg['mode'] == 'extract_img_feature_maps':
            return self.extract_img_feature_maps(meta_data, **cfg)
        elif cfg['mode'] == 'extract_mesh_feature_maps':
            return self.extract_mesh_feature_maps(meta_data, **cfg)
        elif cfg['mode'] == 'mesh_regression':
            return self.regress_mesh_from_sampled_features(meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (parsing/matching_gts)')

    def matching_forward(self, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)

        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))
        #print_dict(outputs)
        return outputs

    @torch.no_grad()
    def parsing_forward(self, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)

        outputs['meta_data'] = meta_data
        return outputs

    def feed_forward(self, meta_data):
        x = self.backbone(meta_data['image'].contiguous())
        outputs = self.head_forward(x)
        return outputs

    @torch.no_grad()
    def pure_forward(self, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
        else:
            outputs = self.feed_forward(meta_data)
        return outputs
    
    def extract_feature_maps(self, image):
        x = self.backbone(image.contiguous())
        if args().learn_deocclusion:
            outputs = self.acquire_maps(x)
        else:
            outputs = {'image_feature_maps': x.float()}
        return outputs
    
    def extract_img_feature_maps(self, image_inputs, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.extract_feature_maps(image_inputs['image'])
        else:
            outputs = self.extract_feature_maps(image_inputs['image'])
        #outputs['pred_batch_ids']+=image_inputs['batch_ids'][0]
        return outputs
    
    @torch.no_grad()
    def extract_mesh_feature_maps(self, image_inputs, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                mesh_feature_maps = self.param_head(self.backbone(image_inputs['image'].contiguous()))
        else:
            mesh_feature_maps = self.param_head(self.backbone(image_inputs['image'].contiguous()))
        return mesh_feature_maps
    
    def regress_mesh_from_sampled_features(self, packed_data, **cfg):
        features_sampled, cam_czyx, cam_preds, outputs = packed_data
        if args().model_precision=='fp16':
            with autocast():
                outputs['params_pred'] = self.mesh_regression_from_features(features_sampled, cam_czyx, cam_preds)
                outputs = self._result_parser.params_map_parser(outputs,outputs['meta_data'])
        else:
            outputs['params_pred'] = self.mesh_regression_from_features(features_sampled, cam_czyx, cam_preds)
            outputs = self._result_parser.params_map_parser(outputs,outputs['meta_data'])
        
        if 'detection_flag' not in outputs:
            outputs['detection_flag'] = self.determine_detection_flag(outputs, outputs['meta_data'])

        return outputs

    def head_forward(self,x):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def _build_gpu_tracker(self):
        self.gpu_tracker = MemTracker()

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