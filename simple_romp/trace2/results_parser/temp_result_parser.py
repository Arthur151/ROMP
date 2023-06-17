import torch
import torch.nn as nn
import logging
import numpy as np
from .smpl_wrapper_relative_temp import SMPLWrapper
from .centermap import CenterMap

SMPL_24 = {'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'SMPL_Head':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23}
joint_sampler_target_name = ['L_Hip_SMPL', 'R_Hip_SMPL', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Toe_SMPL', 'R_Toe_SMPL', \
                            'Neck', 'L_Collar', 'R_Collar', 'SMPL_Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
joint_sampler_relationship = np.array([SMPL_24[joint_name] for joint_name in joint_sampler_target_name])

class TempResultParser(nn.Module):
    def __init__(self, smpl_model_path, centermap_conf_thresh, **kwargs):
        super(TempResultParser, self).__init__()
        self.map_size = 128
        self.params_map_parser = SMPLWrapper(smpl_model_path)
        self.centermap_parser = CenterMap(centermap_conf_thresh)

    @torch.no_grad()
    def parsing_forward(self, outputs, meta_data, cfg):
        outputs, meta_data = self.parse_maps(outputs, meta_data, cfg)
        outputs = self.params_map_parser(outputs, meta_data)
        return outputs, meta_data

    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -
                             1).permute((0, 2, 1)).contiguous()
        results = maps[batch_ids, flat_inds].contiguous()
        return results

    @torch.no_grad()
    def parse_maps(self, outputs, meta_data, cfg):
        if 'pred_batch_ids' in outputs:
            batch_ids = outputs['pred_batch_ids'].long()

            outputs['center_preds'] = outputs['pred_czyxs'] * 512 / 128
            outputs['center_confs'] = outputs['top_score']
        else:
            batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(
                outputs['center_map'])

            if len(batch_ids) == 0:
                batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(
                    outputs['center_map'], top_n_people=1)
                outputs['detection_flag'] = torch.Tensor(
                    [False for _ in range(len(batch_ids))]).cuda()

        if 'params_pred' not in outputs and 'params_maps' in outputs:
            outputs['params_pred'] = self.parameter_sampling(
                outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        if 'center_preds' not in outputs:
            outputs['center_preds'] = torch.stack([flat_inds % 128, flat_inds//128], 1) * 512 / 128
            outputs['center_confs'] = self.parameter_sampling(
                outputs['center_map'], batch_ids, flat_inds, use_transform=True)

        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image', 'offsets', 'imgpath', 'camMats']
        meta_data = reorganize_gts_cpu(meta_data, info_vis, batch_ids)

        if 'pred_batch_ids' in outputs:
            # convert current batch id (0,1,2,3,..) on single gpu to the global id on all gpu (16,17,18,19,...)
            outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        return outputs, meta_data

def reorganize_gts_cpu(meta_data, key_list, batch_ids):
    for key in key_list:
        if key in meta_data:
            if isinstance(meta_data[key], torch.Tensor):
                #print(key, meta_data[key].shape, batch_ids)
                meta_data[key] = meta_data[key].cpu()[batch_ids.cpu()]
            elif isinstance(meta_data[key], list):
                # np.array(meta_data[key])[batch_ids.cpu().numpy()]
                meta_data[key] = [meta_data[key][ind] for ind in batch_ids]
    return meta_data


def reorganize_gts(meta_data, key_list, batch_ids):
    for key in key_list:
        if key in meta_data:
            try:
                if isinstance(meta_data[key], torch.Tensor):
                    #print(key, meta_data[key].shape, batch_ids)
                    meta_data[key] = meta_data[key][batch_ids]
                elif isinstance(meta_data[key], list):
                    # np.array(meta_data[key])[batch_ids.cpu().numpy()]
                    meta_data[key] = [meta_data[key][ind] for ind in batch_ids]
            except:
                print(key, 'reorganize_gts out range: ', len(meta_data[key]), batch_ids)
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
                #print(gt_key, meta_data[gt_key].shape, batch_ids, person_ids)
                meta_data[gt_key] = meta_data[gt_key][batch_ids, person_ids]
            except:
                print(gt_key, 'reorganize_gts out range: ', meta_data[gt_key].shape, batch_ids)
    return outputs, meta_data

def flatten_inds(coords):
    coords = torch.clamp(coords, 0, 128-1)
    return coords[:, 0].long()*128+coords[:, 1].long()