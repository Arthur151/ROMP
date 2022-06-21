from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from models.base import Base
from models.CoordConv import get_coord_maps
from models.basic_modules import BasicBlock,Bottleneck

import config
from config import args
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser

BN_MOMENTUM = 0.1

class ROMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(ROMP, self).__init__()
        print('Using ROMP v1')
        self.backbone = backbone
        self._result_parser = ResultParser()
        self._build_head()
        if args().model_return_loss:
            self._calc_loss = Loss()
        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

    def head_forward(self,x):
        #generate heatmap (K) + associate embedding (K)
        #kp_heatmap_ae = self.final_layers[0](x)
        x = torch.cat((x, self.coordmaps.to(x.device).repeat(x.shape[0],1,1,1)), 1)

        params_maps = self.final_layers[1](x)
        center_maps = self.final_layers[2](x)
        if args().merge_smpl_camera_head:
            cam_maps, params_maps = params_maps[:,:3], params_maps[:,3:]
        else:
            cam_maps = self.final_layers[3](x)
        # to make sure that scale is always a positive value
        cam_maps[:, 0] = torch.pow(1.1,cam_maps[:, 0])
        params_maps = torch.cat([cam_maps, params_maps], 1)
        output = {'params_maps':params_maps.float(), 'center_map':center_maps.float()} #, 'kp_ae_maps':kp_heatmap_ae.float()
        return output

    def _build_head(self):
        self.outmap_size = args().centermap_size
        params_num, cam_dim = self._result_parser.params_map_parser.params_num, 3
        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': args().head_block_num}
        self.output_cfg = {'NUM_PARAMS_MAP':params_num-cam_dim, 'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':cam_dim}

        self.final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.coordmaps = get_coord_maps(128)

    def _make_final_layers(self, input_channels):
        final_layers = []
        final_layers.append(None)
        #output_channels = self.NUM_JOINTS + self.NUM_JOINTS
        #final_layers.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,\
        #    kernel_size=1,stride=1,padding=0))

        input_channels += 2
        if args().merge_smpl_camera_head:
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']+self.output_cfg['NUM_CAM_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
        else:
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CAM_MAP']))

        return nn.ModuleList(final_layers)
    
    def _make_head_layers(self, input_channels, output_channels):
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']

        kernel_sizes, strides, paddings = self._get_trans_cfg()
        for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
            head_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))
        
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        head_layers.append(nn.Conv2d(in_channels=num_channels,out_channels=output_channels,\
            kernel_size=1,stride=1,padding=0))

        return nn.Sequential(*head_layers)

    def _get_trans_cfg(self):
        if self.outmap_size == 32:
            kernel_sizes = [3,3]
            paddings = [1,1]
            strides = [2,2]
        elif self.outmap_size == 64:
            kernel_sizes = [3]
            paddings = [1]
            strides = [2]
        elif self.outmap_size == 128:
            kernel_sizes = [3]
            paddings = [1]
            strides = [1]

        return kernel_sizes, strides, paddings

if __name__ == '__main__':
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({'image':torch.rand(4,512,512,3).cuda()})
    for key, value in outputs.items():
        if isinstance(value,tuple):
            print(key, value)
        elif isinstance(value,list):
            print(key, value)
        else:
            print(key, value.shape)