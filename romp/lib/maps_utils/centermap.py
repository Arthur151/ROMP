import torch
import sys,os
import numpy as np

from config import args


class CenterMap(object):
    def __init__(self,style='heatmap_adaptive_scale'):
        self.style=style
        self.size = args().centermap_size
        self.max_person = args().max_person
        self.shrink_scale = float(args().input_size//self.size)
        self.dims = 1
        self.sigma = 1
        self.conf_thresh= args().centermap_conf_thresh
        print('Confidence:', self.conf_thresh)
        self.gk_group, self.pool_group = self.generate_kernels(args().kernel_sizes)

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
    
    def generate_centermap(self, center_locs, **kwargs):
        return self.generate_centermap_heatmap_adaptive_scale(center_locs, **kwargs)

    def parse_centermap(self, center_map):
        return self.parse_centermap_heatmap_adaptive_scale_batch(center_map)

    def generate_centermap_heatmap_adaptive_scale(self, center_locs, bboxes_hw_norm, occluded_by_who=None,**kwargs):
        '''
           center_locs is in the order of (y,x), corresponding to (w,h), while in the loading data, we have rectified it to the correct (x, y) order
        '''
        radius_list = _calc_radius_(bboxes_hw_norm, map_size=self.size)

        if args().collision_aware_centermap and occluded_by_who is not None:
            # CAR : Collision-Aware Represenation
            for cur_idx, occluded_idx in enumerate(occluded_by_who):
                if occluded_idx>-1:
                    dist_onmap = np.sqrt(((center_locs[occluded_idx]-center_locs[cur_idx])**2).sum()) + 1e-4
                    least_dist = (radius_list[occluded_idx]+radius_list[cur_idx]+1)/self.size*2
                    if dist_onmap<least_dist:
                        offset = np.abs(((radius_list[occluded_idx]+radius_list[cur_idx]+1)/self.size*2-dist_onmap)/dist_onmap) \
                        * (center_locs[occluded_idx]-center_locs[cur_idx]+ 1e-4) * args().collision_factor

                        center_locs[cur_idx] -= offset/2
                        center_locs[occluded_idx] += offset/2

            # restrcit the range from -1 to 1
            center_locs = np.clip(center_locs, -1, 1)
            center_locs[center_locs==-1] = -0.96
            center_locs[center_locs==1] = 0.96

        heatmap = self.generate_heatmap_adaptive_scale(center_locs, radius_list)
        heatmap = torch.from_numpy(heatmap)
        return heatmap

    def generate_heatmap_adaptive_scale(self,center_locs, radius_list,k=1):
        heatmap = np.zeros((1, self.size, self.size),dtype=np.float32)
        for center, radius in zip(center_locs,radius_list):
            diameter = 2 * radius + 1
            gaussian = gaussian2D((diameter, diameter), sigma=float(diameter) / 6)

            x, y = int((center[0]+1)/2*self.size), int((center[1]+1)/2*self.size)
            if x < 0 or y < 0 or x >= self.size or y >= self.size:
                continue
            height, width = heatmap.shape[1:]

            left, right = min(x, radius), min(width - x, radius + 1)
            top, bottom = min(y, radius), min(height - y, radius + 1)

            masked_heatmap  = heatmap[0,y - top:y + bottom, x - left:x + right]
            masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
            if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
                np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
            heatmap[0, y, x]=1
        return heatmap


    def parse_centermap_heatmap_adaptive_scale(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[args().kernel_sizes[-1]])[0]
        h, w = center_map_nms.shape

        centermap = center_map_nms.view(-1)
        confidence, index = centermap.topk(self.max_person)
        x = index%w
        y = (index/float(w)).long()
        idx_topk = torch.stack((y,x),dim=1)
        centers_pred, conf_pred = idx_topk[confidence>self.conf_thresh], confidence[confidence>self.conf_thresh]
        return centers_pred, conf_pred

    def parse_centermap_heatmap_adaptive_scale_batch(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[args().kernel_sizes[-1]])
        b, c, h, w = center_map_nms.shape
        K = self.max_person

        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = torch.div(topk_inds, w, rounding_mode='floor').int().float()
        topk_xs = (topk_inds % w).int().float()
        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = torch.div(index, K, rounding_mode='floor').int()
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

def _calc_radius_(bboxes_hw_norm, map_size=64):
    if len(bboxes_hw_norm) == 0:
        return []
    minimum_radius = map_size / 32.
    scale_factor = map_size / 16.
    scales = np.linalg.norm(np.array(bboxes_hw_norm)/2, ord=2, axis=1)
    radius = (scales * scale_factor + minimum_radius).astype(np.uint8)
    return radius

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

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


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


def test_centermaps():
    batch_size = 2
    CM = CenterMap()
    CM.size=16
    center_locs = np.array([[0,0],[-0.3,-0.7]])
    bboxes = [np.array([0.2,0.3]),np.array([0.5,0.4])]
    centermaps = []
    for i in range(batch_size):
        centermaps.append(torch.from_numpy(CM.generate_centermap(center_locs,bboxes_hw_norm=bboxes)))
    centermaps = torch.stack(centermaps).cuda()
    print_matrix(centermaps[0,0])
    print('__'*10)
    results = CM.parse_centermap_heatmap_adaptive_scale_batch(centermaps)
    print(results)
    #5CM.print_matrix(torch.nn.functional.softmax(centermap,1)[0])
    for i in range(batch_size):
        result = CM.parse_centermap(centermaps[i])
        print(result)
        center_list = process_center(result[0], centermaps[i])
        print(center_list)


if __name__ == '__main__':
    test_centermaps()
