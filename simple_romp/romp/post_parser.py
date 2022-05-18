import torch
from torch import nn
import sys,os
import numpy as np
from .smpl import SMPL
from .utils import rot6D_to_angular, batch_orth_proj, estimate_translation

class CenterMap(object):
    def __init__(self, conf_thresh):
        self.size = 64
        self.max_person = 64
        self.sigma = 1
        self.conf_thresh= conf_thresh
        self.gk_group, self.pool_group = self.generate_kernels([5])

    def generate_kernels(self, kernel_size_list):
        gk_group, pool_group = {}, {}
        for kernel_size in set(kernel_size_list):
            x = np.arange(0, kernel_size, 1, float)
            y = x[:, np.newaxis]
            x0, y0 = (kernel_size-1)//2,(kernel_size-1)//2
            gaussian_distribution = - ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
            gk_group[kernel_size] = np.exp(gaussian_distribution)
            pool_group[kernel_size] = torch.nn.MaxPool2d(kernel_size, 1, (kernel_size-1)//2)
        return gk_group, pool_group

    def parse_centermap(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[5])
        b, c, h, w = center_map_nms.shape
        K = self.max_person

        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = (topk_inds.long() // w).float()
        topk_xs = (topk_inds % w).int().float()
        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = index.long() // K
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)

        mask = topk_score>self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_yxs = torch.stack([topk_ys[mask], topk_xs[mask]]).permute((1,0))
        return batch_ids, topk_inds[mask], center_yxs, topk_score[mask]


def nms(det, pool_func=None):
    maxm = pool_func(det)
    maxm = torch.eq(maxm, det).float()
    det = det * maxm
    return det

def gather_feature(fmap, index, mask=None):
    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def pack_params_dict(params_pred):
    idx_list, params_dict = [0], {}
    part_name = ['cam', 'global_orient', 'body_pose', 'smpl_betas']
    part_idx = [3, 6, 21*6, 10]
    for i,  (idx, name) in enumerate(zip(part_idx, part_name)):
        idx_list.append(idx_list[i] + idx)
        params_dict[name] = params_pred[:, idx_list[i]: idx_list[i+1]].contiguous()
    params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
    params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
    N = params_dict['body_pose'].shape[0]
    params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N,6).to(params_dict['body_pose'].device)],1)
    params_dict['smpl_thetas'] = torch.cat([params_dict['global_orient'], params_dict['body_pose']], 1)

    return params_dict

def convert_proejection_from_input_to_orgimg(kps, offsets):
    top, bottom, left, right, h, w = offsets
    img_pad_size = max(h,w)
    kps[:, :, 0] = (kps[:,:,0] + 1) * img_pad_size / 2 - left
    kps[:, :, 1] = (kps[:,:,1] + 1) * img_pad_size / 2 - top
    if kps.shape[-1] == 3:
        kps[:, :, 2] = (kps[:,:,2] + 1) * img_pad_size / 2
    return kps

def convert_cam_to_3d_trans(cams, weight=2.):
    (s, tx, ty) = cams[:,0], cams[:,1], cams[:,2]
    depth, dx, dy = 1./s, tx/s, ty/s
    trans3d = torch.stack([dx, dy, depth], 1)*weight
    return trans3d

def convert_cam_to_3d_trans2(j3ds, pj3d): 
    predicts_j3ds = j3ds[:,:24].contiguous().detach().cpu().numpy()
    predicts_pj2ds = (pj3d[:,:,:2][:,:24].detach().cpu().numpy()+1)*256
    cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, \
                                focal_length=443.4, img_size=np.array([512,512])).to(j3ds.device)
    return cam_trans
    

def body_mesh_projection2image(j3d_preds, cam_preds, vertices=None, input2org_offsets=None):
    pj3d = batch_orth_proj(j3d_preds, cam_preds, mode='2d')
    pred_cam_t = convert_cam_to_3d_trans2(j3d_preds, pj3d)
    projected_outputs = {'pj2d': pj3d[:,:,:2], 'cam_trans':pred_cam_t}
    if vertices is not None:
        projected_outputs['verts_camed'] = batch_orth_proj(vertices, cam_preds, mode='3d',keep_dim=True)

    if input2org_offsets is not None:
        projected_outputs['pj2d_org'] = convert_proejection_from_input_to_orgimg(projected_outputs['pj2d'], input2org_offsets)
        projected_outputs['verts_camed_org'] = convert_proejection_from_input_to_orgimg(projected_outputs['verts_camed'], input2org_offsets)
    return projected_outputs

class SMPL_parser(nn.Module):
    def __init__(self, model_path):
        super(SMPL_parser, self).__init__()
        self.smpl_model = SMPL(model_path)
    
    def forward(self, outputs, root_align=False):
        verts, joints, face = self.smpl_model(outputs['smpl_betas'], outputs['smpl_thetas'], root_align=root_align)
        outputs.update({'verts': verts, 'joints': joints, 'smpl_face':face})
        
        return outputs


def parameter_sampling(maps, batch_ids, flat_inds, use_transform=True):
    if use_transform:
        batch, channel = maps.shape[:2]
        maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
    results = maps[batch_ids,flat_inds].contiguous()
    return results

def parsing_outputs(center_maps, params_maps, centermap_parser):
    center_preds_info = centermap_parser.parse_centermap(center_maps)
    batch_ids, flat_inds, cyxs, center_confs = center_preds_info
    if len(batch_ids)==0:
        print('None person detected')
        return None

    params_pred = parameter_sampling(params_maps, batch_ids, flat_inds, use_transform=True)
    parsed_results = pack_params_dict(params_pred)
    parsed_results['center_preds'] = torch.stack([flat_inds%64, flat_inds//64],1) * 512 // 64
    parsed_results['center_confs'] = parameter_sampling(center_maps, batch_ids, flat_inds, use_transform=True)
    return parsed_results

