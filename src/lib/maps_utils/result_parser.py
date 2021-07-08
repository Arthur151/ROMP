import os,sys
import torch
import torch.nn as nn
import numpy as np 
import logging
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import config
from config import args
import constants
from models.smpl_wrapper import SMPLWrapper
from maps_utils import CenterMap
from utils.center_utils import process_gt_center
from utils.rot_6D import rot6D_to_angular

class ResultParser(nn.Module):
    def __init__(self):
        super(ResultParser,self).__init__()
        self.map_size = args().centermap_size
        self.params_map_parser = SMPLWrapper()
        self.centermap_parser = CenterMap()

    def train_forward(self, outputs, meta_data, cfg):
        outputs,meta_data = self.match_params(outputs, meta_data)
        outputs = self.params_map_parser(outputs,meta_data)
        return outputs,meta_data

    @torch.no_grad()
    def val_forward(self, outputs, meta_data, cfg):
        outputs, meta_data = self.parse_maps(outputs, meta_data, cfg)
        outputs = self.params_map_parser(outputs,meta_data)
        return outputs, meta_data

    def match_params(self, outputs, meta_data):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'smpl_flag', 'kp3d_flag', 'subject_ids']
        exclude_keys = ['person_centers','offsets']

        center_gts_info = process_gt_center(meta_data['person_centers'])
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        mc_centers = self.match_gt_pred(center_gts_info, center_preds_info, outputs['center_map'].device)
        batch_ids, flat_inds, person_ids = mc_centers['batch_ids'], mc_centers['flat_inds'], mc_centers['person_ids']
        if len(batch_ids)==0:
            logging.error('number of predicted center is {}'.format(batch_ids))
            batch_ids, flat_inds = torch.zeros(2).long().to(outputs['center_map'].device), (torch.ones(2)*self.map_size**2/2.).to(outputs['center_map'].device).long()
            person_ids = batch_ids.clone()
        
        params_pred = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        outputs,meta_data = self.reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['params_pred'] = params_pred
        return outputs,meta_data

    def match_gt_pred(self,center_gts_info, center_preds_info, device):
        vgt_batch_ids, vgt_person_ids, center_gts = center_gts_info
        vpred_batch_ids, flat_inds, cyxs, top_score = center_preds_info
        mc = {key:[] for key in ['batch_ids', 'flat_inds', 'person_ids']}

        for batch_id, person_id, center_gt in zip(vgt_batch_ids, vgt_person_ids, center_gts):
            if batch_id in vpred_batch_ids:
                center_pred = cyxs[vpred_batch_ids==batch_id]
                center_gt = center_pred[torch.argmin(torch.norm(center_pred.float()-center_gt[None].float().to(device),dim=-1))].long()
                cy, cx = torch.clamp(center_gt, 0, self.map_size-1)
                flat_ind = cy*args().centermap_size+cx
                mc['batch_ids'].append(batch_id); mc['flat_inds'].append(flat_ind); mc['person_ids'].append(person_id)
        keys_list = list(mc.keys())
        for key in keys_list:
            mc[key] = torch.Tensor(mc[key]).long().to(device)

        return mc
        
    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        device = maps.device
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        results = maps[batch_ids,flat_inds].contiguous()
        return results

    def reorganize_gts(self, meta_data, key_list, batch_ids):
        for key in key_list:
            if isinstance(meta_data[key], torch.Tensor):
                meta_data[key] = meta_data[key][batch_ids]
            elif isinstance(meta_data[key], list):
                meta_data[key] = np.array(meta_data[key])[batch_ids.cpu().numpy()]
        return meta_data

    def reorganize_data(self, outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids):
        exclude_keys += gt_keys
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = []
        for key, item in meta_data.items():
            if key not in exclude_keys:
                info_vis.append(key)

        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        for gt_key in gt_keys:
            meta_data[gt_key] = meta_data[gt_key][batch_ids,person_ids]
        #meta_data['kp_2d'] = meta_data['full_kp2d']
        return outputs,meta_data

    @torch.no_grad()
    def parse_maps(self,outputs, meta_data, cfg):
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        batch_ids, flat_inds, cyxs, top_score = center_preds_info
        if len(batch_ids)==0:
            #logging.error('number of predicted center is {}'.format(batch_ids))
            batch_ids, flat_inds = torch.zeros(2).long().to(outputs['center_map'].device), (torch.ones(2)*self.map_size**2/2.).to(outputs['center_map'].device).long()
            person_ids = batch_ids.clone()
            outputs['detection_flag'] = False
        else:
            outputs['detection_flag'] = True
        outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image_org', 'offsets']
        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        
        return outputs,meta_data

def flatten_inds(coords):
    coords = torch.clamp(coords, 0, args().centermap_size-1)
    return coords[:,0].long()*args().centermap_size+coords[:,1].long()
