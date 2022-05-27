import torch
from torch import nn
import numpy as np
from romp.smpl import SMPL
from romp.utils import rot6D_to_angular

def get_3Dcoord_maps(size=128, z_base=None):
    range_arr = torch.arange(size, dtype=torch.float32)
    if z_base is None:
        Z_map = range_arr.reshape(1,size,1,1,1).repeat(1,1,size,size,1) / size * 2 -1
    else:
        Z_map = z_base.reshape(1,size,1,1,1).repeat(1,1,size,size,1)
    Y_map = range_arr.reshape(1,1,size,1,1).repeat(1,size,1,size,1) / size * 2 -1
    X_map = range_arr.reshape(1,1,1,size,1).repeat(1,size,size,1,1) / size * 2 -1

    out = torch.cat([Z_map,Y_map,X_map], dim=-1)
    return out

class CenterMap3D(object):
    def __init__(self, conf_thresh):
        print('Threshold for positive center detection:', conf_thresh)
        self.size = 128
        self.max_person = 64
        self.sigma = 1
        self.conf_thresh= conf_thresh
        self.gk_group, self.pool_group = self.generate_kernels([5])
        self.prepare_parsing()
    
    def prepare_parsing(self):
        self.coordmap_3d = get_3Dcoord_maps(size=self.size)
        self.maxpool3d = torch.nn.MaxPool3d(5, 1, (5-1)//2)

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

    def parse_3dcentermap(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.maxpool3d).squeeze(1)
        b, c, h, w = center_map_nms.shape
        K = self.max_person

        # acquire top k value/index at each depth
        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = (topk_inds.long()// w).float()
        topk_xs = (topk_inds % w).int().float()
        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        # div by K because index is grouped by K(C x K shape)
        topk_zs = index.long() // K
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)

        mask = topk_score>self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_zyxs = torch.stack([topk_zs[mask].long(), topk_ys[mask].long(), topk_xs[mask].long()]).permute((1,0)).long()

        return [batch_ids, center_zyxs, topk_score[mask]]

def perspective_projection(points, translation=None,rotation=None, 
                           focal_length=443.4, camera_center=None, img_size=512, normalize=True):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    if isinstance(points,np.ndarray):
        points = torch.from_numpy(points).float()
    if isinstance(translation,np.ndarray):
        translation = torch.from_numpy(translation).float()
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    if camera_center is not None:
        K[:,-1, :-1] = camera_center

    # Transform points
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)
    if translation is not None:
        points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / (points[:,:,-1].unsqueeze(-1)+1e-6)
    # Apply camera intrinsics
    # projected_points = torch.einsum('bij,bkj->bki', K, projected_points)[:, :, :-1]
    projected_points = torch.matmul(projected_points.contiguous(), K.contiguous())
    projected_points = projected_points[:, :, :-1].contiguous()

    if normalize:
        projected_points /= float(img_size)/2.

    return projected_points

tan_fov = np.tan(np.radians(60/2.))

def convert_scale_to_depth(scale):
    return 1 / (scale * tan_fov + 1e-3)

def denormalize_cam_params_to_trans(normed_cams, positive_constrain=False):
    #convert the predicted camera parameters to 3D translation in camera space.
    scale = normed_cams[:, 0]
    if positive_constrain:
        positive_mask = (normed_cams[:, 0] > 0).float()
        scale = scale * positive_mask

    trans_XY_normed = torch.flip(normed_cams[:, 1:],[1])
    # convert from predicted scale to depth
    depth = convert_scale_to_depth(scale).unsqueeze(1)
    # convert from predicted X-Y translation on image plane to X-Y coordinates on camera space.
    trans_XY = trans_XY_normed * depth * tan_fov
    trans = torch.cat([trans_XY, depth], 1)

    return trans

def convert_proejection_from_input_to_orgimg(kps, offsets):
    top, bottom, left, right, h, w = offsets
    img_pad_size = max(h,w)
    kps[:, :, 0] = (kps[:,:,0] + 1) * img_pad_size / 2 - left
    kps[:, :, 1] = (kps[:,:,1] + 1) * img_pad_size / 2 - top
    if kps.shape[-1] == 3:
        kps[:, :, 2] = (kps[:,:,2] + 1) * img_pad_size / 2
    return kps


def body_mesh_projection2image(j3d_preds, cam_preds, vertices=None, input2org_offsets=None):
    pred_cam_t = denormalize_cam_params_to_trans(cam_preds, positive_constrain=False)
    pj2d = perspective_projection(j3d_preds,translation=pred_cam_t,focal_length=443.4, normalize=True)
    projected_outputs = {'cam_trans':pred_cam_t, 'pj2d': pj2d.float()}
    if vertices is not None:
        projected_outputs['verts_camed'] = perspective_projection(vertices.clone().detach(),translation=pred_cam_t,focal_length=443.4, normalize=True)
        projected_outputs['verts_camed'] = torch.cat([projected_outputs['verts_camed'], vertices[:,:,[2]]], -1)

    if input2org_offsets is not None:
        projected_outputs['pj2d_org'] = convert_proejection_from_input_to_orgimg(projected_outputs['pj2d'], input2org_offsets)
        if 'verts_camed' in projected_outputs:
            projected_outputs['verts_camed_org'] = convert_proejection_from_input_to_orgimg(projected_outputs['verts_camed'], input2org_offsets)
    return projected_outputs

def remove_subjects(outputs, removed_subj_inds):
    N = len(outputs['params_pred'])
    remove_mask = torch.ones(N).bool()
    remove_mask[removed_subj_inds] = False
    left_subj_inds = torch.where(remove_mask)[0].tolist()
    
    keys = list(outputs.keys())
    for key in keys:
        if key in ['smpl_face', 'center_map', 'center_map_3d']:
            continue
        outputs[key] = outputs[key][left_subj_inds]
    return outputs

def suppressing_redundant_prediction_via_projection(outputs, img_shape, thresh=16, conf_based=False):
    pj2ds = outputs['pj2d']
    N = len(pj2ds)
    if N == 1:
        return outputs

    pj2d_diff = pj2ds.unsqueeze(1).repeat(1,N,1,1) - pj2ds.unsqueeze(0).repeat(N,1,1,1)
    pj2d_dist_mat = torch.norm(pj2d_diff, p=2, dim=-1).mean(-1)
    
    person_scales = outputs['cam'][:,0] * 2
    ps1, ps2 = person_scales.unsqueeze(1).repeat(1,N), person_scales.unsqueeze(0).repeat(N, 1)
    max_scale_mat = torch.where(ps1>ps2, ps1, ps2)

    pj2d_dist_mat_normalized = pj2d_dist_mat / max_scale_mat

    triu_mask = torch.triu(torch.ones_like(pj2d_dist_mat), diagonal=1)<0.5
    pj2d_dist_mat_normalized[triu_mask] = 10000.
    # print('pj2d_dist_mat_normalized', sorted(pj2d_dist_mat_normalized[pj2d_dist_mat_normalized<100].cpu().numpy()))

    max_length = max(img_shape)
    thresh = thresh * max_length / 640
    repeat_subj_inds = torch.where(pj2d_dist_mat_normalized<thresh)
    if len(repeat_subj_inds)>0:
        # exclude the subject behind the duplicated one, larger depth, smaller scale value
        if conf_based:
            center_confs = outputs['center_confs']
            removed_subj_inds = torch.where(center_confs[repeat_subj_inds[0]]<center_confs[repeat_subj_inds[1]], repeat_subj_inds[0], repeat_subj_inds[1])
        else:
            removed_subj_inds = torch.where(person_scales[repeat_subj_inds[0]]<person_scales[repeat_subj_inds[1]], repeat_subj_inds[0], repeat_subj_inds[1])
        # print('removed:', removed_subj_inds, )
        outputs = remove_subjects(outputs, removed_subj_inds)
    return outputs

def remove_outlier(outputs, relative_scale_thresh=3, scale_thresh=0.25): #0.3
    # remove the isolate remote outliers
    cam_trans = outputs['cam_trans']
    N = len(cam_trans)
    if N<3:
        return outputs
    trans_diff = cam_trans.unsqueeze(1).repeat(1,N,1) - cam_trans.unsqueeze(0).repeat(N,1,1)
    trans_dist_mat = torch.norm(trans_diff, p=2, dim=-1)
    # drop the least and the largest dist 
    trans_dist_mat = torch.sort(trans_dist_mat).values[:,1:-1]
    mean_dist = trans_dist_mat.mean(1)
    
    relative_scale = mean_dist / ((mean_dist.sum() - mean_dist)/(N-1))
    #print('relative_scale', torch.sort(relative_scale).values)
    outlier_mask = relative_scale > relative_scale_thresh
    #print('cam', outputs['cam'][:,0][outlier_mask])
    # false positive predictions are usually in small scale
    outlier_mask *= outputs['cam'][:,0] < scale_thresh
    
    removed_subj_inds = torch.where(outlier_mask)[0]
    if len(removed_subj_inds)>0:
        outputs = remove_subjects(outputs, removed_subj_inds)
    return outputs

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
    part_idx = [3, 6, 21*6, 11]
    for i,  (idx, name) in enumerate(zip(part_idx, part_name)):
        idx_list.append(idx_list[i] + idx)
        params_dict[name] = params_pred[:, idx_list[i]: idx_list[i+1]].contiguous()
    params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
    params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
    N = params_dict['body_pose'].shape[0]
    params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N,6).to(params_dict['body_pose'].device)],1)
    params_dict['smpl_thetas'] = torch.cat([params_dict['global_orient'], params_dict['body_pose']], 1)

    return {'cam': params_dict['cam'], 'smpl_thetas': params_dict['smpl_thetas'], 'smpl_betas': params_dict['smpl_betas']}

class SMPLA_parser(nn.Module):
    def __init__(self, smpla_path, smil_path):
        super(SMPLA_parser, self).__init__()
        self.smil_model = SMPL(smil_path, model_type='smpl')
        self.smpl_model = SMPL(smpla_path, model_type='smpla')
        self.baby_thresh= 0.8
    
    def forward(self, betas=None, thetas=None, root_align=True):
        baby_mask = betas[:,10] > self.baby_thresh
        if baby_mask.sum()>0:
            adult_mask = ~baby_mask
            person_num = len(thetas)
            verts, joints = torch.zeros(person_num, 6890, 3, device=thetas.device).float(), torch.zeros(person_num, 54+17, 3, device=thetas.device).float()
            verts[baby_mask], joints[baby_mask], face = self.smil_model(betas[baby_mask,:10], thetas[baby_mask])
            if adult_mask.sum()>0:
                verts[adult_mask], joints[adult_mask], face = self.smpl_model(betas[adult_mask], thetas[adult_mask])
        else:
            verts, joints, face = self.smpl_model(betas, thetas)
        if root_align:
            # use the Pelvis of most 2D image, not the original Pelvis
            root_trans = joints[:,[45,46]].mean(1).unsqueeze(1)
            joints = joints - root_trans
            verts =  verts - root_trans
        return verts, joints, face