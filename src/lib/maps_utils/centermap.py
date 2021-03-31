import torch
import sys,os
import numpy as np
sys.path.append(os.path.abspath(__file__).replace('maps_utils/centermap.py',''))
from config import args

class CenterMap(object):
    def __init__(self,style='heatmap_adaptive_scale'):
        self.style=style
        self.size = args.centermap_size
        self.max_person = args.max_person
        self.shrink_scale = float(args.input_size//self.size)
        self.dims = 1
        self.sigma = 1
        self.conf_thresh= args.centermap_conf_thresh
        self.gk_group, self.pool_group = self.generate_kernels(args.kernel_sizes)

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

    def process_gt_CAM(self, center_normed):
        center_list = []
        valid_mask = center_normed[:,:,0]>-1
        valid_inds = torch.where(valid_mask)
        valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
        center_gt = ((center_normed+1)/2*self.size).long()
        center_gt_valid = center_gt[valid_mask]
        return (valid_batch_inds, valid_person_ids, center_gt_valid)

    def parse_centermap(self, center_map):
        return self.parse_centermap_heatmap_adaptive_scale_batch(center_map)

    def parse_centermap_heatmap_adaptive_scale_batch(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[args.kernel_sizes[-1]])
        b, c, h, w = center_map_nms.shape
        K = self.max_person

        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = (topk_inds // w).int().float()
        topk_xs = (topk_inds % w).int().float()
        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index // K).int()
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

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def process_center(center_gt, centermap):
    center_list = []
    center_locs = torch.stack(torch.where(centermap[0]>0.25)).transpose(1,0)
    dists = []
    for center in center_gt:
        dists.append(torch.norm(center_locs.float()-center[None].float(),dim=1))
    dists = torch.stack(dists)
    assign_id = torch.argmin(dists,0)
    for center_id in range(len(center_gt)):
        center_list.append(center_locs[assign_id==center_id])

    return center_list

def print_matrix(matrix):
    for k in matrix:
        print_item = ''
        for i in k:
            print_item+='{:.2f} '.format(i)
        print(print_item)
