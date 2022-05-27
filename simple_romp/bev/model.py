import torch
import torch.nn as nn
import numpy as np
from romp.model import HigherResolutionNet, BasicBlock
from .post_parser import CenterMap3D

BN_MOMENTUM = 0.1

def get_3Dcoord_maps_halfz(size, z_base):
    range_arr = torch.arange(size, dtype=torch.float32)
    z_len = len(z_base)
    Z_map = z_base.reshape(1,z_len,1,1,1).repeat(1,1,size,size,1)
    Y_map = range_arr.reshape(1,1,size,1,1).repeat(1,z_len,1,size,1) / size * 2 -1
    X_map = range_arr.reshape(1,1,1,size,1).repeat(1,z_len,size,1,1) / size * 2 -1

    out = torch.cat([Z_map,Y_map,X_map], dim=-1)
    return out

def conv3x3_1D(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock_1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = conv3x3_1D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1D(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

def conv3x3_3D(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock_3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_3D, self).__init__()
        self.conv1 = conv3x3_3D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_3D(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        return out

def get_cam3dmap_anchor(FOV, centermap_size):
    depth_level = np.array([1, 10, 20, 100], dtype=np.float32)
    map_coord_range_each_level = (np.array([2/64., 25/64., 3/64., 2/64.], dtype=np.float32) * centermap_size).astype(np.int)
    scale_level = 1/np.tan(np.radians(FOV/2.))/depth_level
    cam3dmap_anchor = []
    scale_cache = 8
    for scale, coord_range in zip(scale_level, map_coord_range_each_level):
        cam3dmap_anchor.append(scale_cache-np.arange(1,coord_range+1)/coord_range*(scale_cache-scale))
        scale_cache = scale
    cam3dmap_anchor = np.concatenate(cam3dmap_anchor)
    return cam3dmap_anchor

def convert_cam_params_to_centermap_coords(cam_params, cam3dmap_anchor):
    center_coords = torch.ones_like(cam_params)
    center_coords[:,1:] = cam_params[:,1:].clone()
    cam3dmap_anchors = cam3dmap_anchor.to(cam_params.device)[None]
    scale_num = len(cam3dmap_anchor)
    if len(cam_params) != 0:
        center_coords[:,0] = torch.argmin(torch.abs(cam_params[:,[0]].repeat(1, scale_num) - cam3dmap_anchors), dim=1).float()/128 * 2. - 1.
    
    return center_coords

def denormalize_center(center, size=128):
    center = (center+1)/2*size
    center = torch.clamp(center, 1, size-1).long()
    return center

class BEVv1(nn.Module):
    def __init__(self, **kwargs):
        super(BEVv1, self).__init__()
        print('Using BEV.')
        self.backbone = HigherResolutionNet()
        self._build_head()
        self._build_parser(conf_thresh=kwargs.get('center_thresh', 0.1))
    
    def _build_parser(self, conf_thresh=0.12):
        self.centermap_parser = CenterMap3D(conf_thresh=conf_thresh)

    def _build_head(self):
        params_num, cam_dim = 3+22*6+11, 3
        self.outmap_size = 128 
        self.output_cfg = {'NUM_PARAMS_MAP':params_num-cam_dim, 'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':cam_dim}
        
        self.head_cfg = {'NUM_BASIC_BLOCKS':1, 'NUM_CHANNELS': 128}
        self.bv_center_cfg = {'NUM_DEPTH_LEVEL': self.outmap_size//2, 'NUM_BLOCK': 2}
        
        self.backbone_channels = self.backbone.backbone_channels
        self.transformer_cfg = {'INPUT_C':self.head_cfg['NUM_CHANNELS'], 'NUM_CHANNELS': 512}
        self._make_transformer()
        
        self.cam3dmap_anchor = torch.from_numpy(get_cam3dmap_anchor(60, self.outmap_size)).float()
        self.register_buffer('coordmap_3d', get_3Dcoord_maps_halfz(self.outmap_size, z_base=self.cam3dmap_anchor))
        self._make_final_layers(self.backbone_channels)
    
    def _make_transformer(self, drop_ratio=0.2):
        self.position_embeddings = nn.Embedding(self.outmap_size, self.transformer_cfg['INPUT_C'], padding_idx=0)
        self.transformer = nn.Sequential(
            nn.Linear(self.transformer_cfg['INPUT_C'],self.transformer_cfg['NUM_CHANNELS']),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(self.transformer_cfg['NUM_CHANNELS'],self.transformer_cfg['NUM_CHANNELS']),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(self.transformer_cfg['NUM_CHANNELS'],self.output_cfg['NUM_PARAMS_MAP']))

    def _make_final_layers(self, input_channels):
        self.det_head = self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP'])
        self.param_head = self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP'], with_outlayer=False)
        
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
        
        input_channels = (num_channels + self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP'])*self.outmap_size
        inter_channels = 512
        self.bv_out_layers = nn.Sequential(
                    BasicBlock_1D(input_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, output_channels))

    def _make_3D_map_refiner(self):
        self.center_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CENTER_MAP'], self.output_cfg['NUM_CENTER_MAP']))
        self.cam_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CAM_MAP'], self.output_cfg['NUM_CAM_MAP']))
    
    def fv_conditioned_bv_estimation(self, x, center_maps_fv, cam_maps_offset):
        img_feats = self.bv_pre_layers(x)
        summon_feats = torch.cat([center_maps_fv, cam_maps_offset, img_feats], 1).view(img_feats.size(0), -1, self.outmap_size)
        
        outputs_bv = self.bv_out_layers(summon_feats)
        center_maps_bv = outputs_bv[:, :self.bv_center_cfg['NUM_DEPTH_LEVEL']]
        cam_maps_offset_bv = outputs_bv[:, self.bv_center_cfg['NUM_DEPTH_LEVEL']:]
        center_map_3d = center_maps_fv.repeat(1,self.bv_center_cfg['NUM_DEPTH_LEVEL'],1,1) * \
                        center_maps_bv.unsqueeze(2).repeat(1,1,self.outmap_size,1)
        return center_map_3d, cam_maps_offset_bv
    
    def coarse2fine_localization(self, x):
        maps_fv = self.det_head(x)
        center_maps_fv = maps_fv[:,:self.output_cfg['NUM_CENTER_MAP']]
        # predict the small offset from each anchor at 128 map to meet the real 2D image map: map from 0~1 to 0~4 image coordinates
        cam_maps_offset = maps_fv[:,self.output_cfg['NUM_CENTER_MAP']:self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP']]
        center_maps_3d, cam_maps_offset_bv = self.fv_conditioned_bv_estimation(x, center_maps_fv, cam_maps_offset)

        center_maps_3d = self.center_map_refiner(center_maps_3d.unsqueeze(1)).squeeze(1)
        # B x 3 x H x W -> B x 1 x H x W x 3  |  B x 3 x D x W -> B x D x 1 x W x 3
        # B x D x H x W x 3 + B x 1 x H x W x 3 + B x D x 1- x W x 3  .to(cam_maps_offset.device)
        cam_maps_3d = self.coordmap_3d + \
                        cam_maps_offset.unsqueeze(-1).transpose(4,1).contiguous()
        # cam_maps_offset_bv adjust z-wise only
        cam_maps_3d[:,:,:,:,2] = cam_maps_3d[:,:,:,:,2] + cam_maps_offset_bv.unsqueeze(2).contiguous()
        cam_maps_3d = self.cam_map_refiner(cam_maps_3d.unsqueeze(1).transpose(5,1).squeeze(-1))
        
        return center_maps_3d, cam_maps_3d, center_maps_fv
    
    def differentiable_person_feature_sampling(self, feature, pred_czyxs, pred_batch_ids):
        cz, cy, cx = pred_czyxs[:,0], pred_czyxs[:,1], pred_czyxs[:,2]
        position_encoding = self.position_embeddings(cz)
        feature_sampled = feature[pred_batch_ids, :, cy, cx]

        input_features = feature_sampled + position_encoding
        return input_features
    
    def mesh_parameter_regression(self, fv_f, cams_preds, pred_batch_ids):
        cam_czyx = denormalize_center(convert_cam_params_to_centermap_coords(cams_preds.clone(), self.cam3dmap_anchor), size=self.outmap_size)
        feature_sampled = self.differentiable_person_feature_sampling(fv_f, cam_czyx, pred_batch_ids)
        params_preds = self.transformer(feature_sampled)
        params_preds = torch.cat([cams_preds, params_preds], 1)
        return params_preds, cam_czyx
    
    @torch.no_grad()
    def forward(self, x):
        x = self.backbone(x)
        center_maps_3d, cam_maps_3d, center_maps_fv = self.coarse2fine_localization(x)
        
        center_preds_info_3d = self.centermap_parser.parse_3dcentermap(center_maps_3d)
        if len(center_preds_info_3d[0])==0:
            print('No person detected!')
            return None
        pred_batch_ids, pred_czyxs, center_confs = center_preds_info_3d
        cams_preds = cam_maps_3d[pred_batch_ids,:,pred_czyxs[:,0],pred_czyxs[:,1],pred_czyxs[:,2]]

        front_view_features = self.param_head(x)
        params_preds, cam_czyx = self.mesh_parameter_regression(front_view_features, cams_preds, pred_batch_ids)
        
        output = {'params_pred':params_preds.float(), 'cam_czyx':cam_czyx.float(), 
                'center_map':center_maps_fv.float(),'center_map_3d':center_maps_3d.float().squeeze(),
                'pred_batch_ids':pred_batch_ids, 'pred_czyxs':pred_czyxs, 'center_confs':center_confs} 
        return output

def export_model_to_onnx_static():
    model = BEVv1().cuda()
    state_dict = torch.load('/home/yusun/CenterMesh/trained_models/BEV_review.pth')
    model.load_state_dict(state_dict, strict=False)
    save_file = '/home/yusun/ROMP/trained_models/BEV.onnx'

    import cv2
    image = cv2.imread('/home/yusun/CenterMesh/simple_romp/test/ages.png')[400:]
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (512,512))
    image = torch.from_numpy(image)[None].cuda().float()
    torch.onnx.export(model, (image),
                      save_file, 
                      input_names=['image'],
                      output_names=['center_maps', 'params_maps'],
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True)
    print('ROMP onnx saved into: ', save_file)


if __name__ == '__main__':
    export_model_to_onnx_static()
    """
    model = BEVv1().cuda()
    model_path = '/home/yusun/CenterMesh/trained_models/BEV_review.pth'
    model_path = '/home/yusun/CenterMesh/trained_models/BEV_rebuttal.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    import cv2
    image = cv2.imread('/home/yusun/CenterMesh/simple_romp/test/ages.png')[400:]
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (512,512))
    cv2.imwrite('/home/yusun/CenterMesh/simple_romp/test/ages_croped.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    input_image = torch.from_numpy(image).cuda()[None]
    outputs = model(input_image)
    for key, value in outputs.items():
        if isinstance(value,tuple):
            print(key, value)
        elif isinstance(value,list):
            print(key, value)
        else:
            print(key, value.shape)
    """