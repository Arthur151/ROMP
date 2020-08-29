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
        self.conf_thresh=0.25
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


    def parse_centermap(self, center_map):
        if self.style =='heatmap':
            return self.parse_centermap_heatmap(center_map)
        elif self.style == 'heatmap_adaptive_scale':
            return self.parse_centermap_heatmap_adaptive_scale(center_map)
        else:
            raise NotImplementedError


    def nms(self, det, pool_func=None):
        maxm = pool_func(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def multi_channel_nms(self,center_maps):
        center_map_pooled = []
        for depth_idx, center_map in enumerate(center_maps):
            center_map_pooled.append(self.nms(center_map[None], pool_func=self.pool_group[args.kernel_sizes[depth_idx]]))
        center_maps_max = torch.max(torch.cat(center_map_pooled,0),0).values
        center_map_nms = self.nms(center_maps_max[None], pool_func=self.pool_group[args.kernel_sizes[-1]])[0]
        return center_map_nms

    def parse_centermap_mask(self,center_map):
        center_map_bool = torch.argmax(center_map,1).bool()
        center_idx = torch.stack(torch.where(center_map_bool)).transpose(1,0)
        return center_idx

    def parse_centermap_heatmap(self,center_maps):
        if center_maps.shape[0]>1:
            center_map_nms = self.multi_channel_nms(center_maps)
        else:
            center_map_nms = self.nms(center_maps, pool_func=self.pool_group[args.kernel_sizes[-1]])[0]
        h, w = center_map_nms.shape

        centermap = center_map_nms.view(-1)
        confidence, index = centermap.topk(self.max_person)
        x = index%w
        y = (index/w).long()
        idx_topk = torch.stack((y,x),dim=1)
        centers_pred, conf_pred = idx_topk[confidence>self.conf_thresh], confidence[confidence>self.conf_thresh]
        return centers_pred, conf_pred

    def parse_centermap_heatmap_adaptive_scale(self, center_maps):
        center_map_nms = self.nms(center_maps, pool_func=self.pool_group[args.kernel_sizes[-1]])[0]
        h, w = center_map_nms.shape

        centermap = center_map_nms.view(-1)
        confidence, index = centermap.topk(self.max_person)
        x = index%w
        y = (index/float(w)).long()
        idx_topk = torch.stack((y,x),dim=1)
        centers_pred, conf_pred = idx_topk[confidence>self.conf_thresh], confidence[confidence>self.conf_thresh]
        return centers_pred, conf_pred

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


if __name__ == '__main__':
    batch_size = 2
    CM = CenterMap()
    center_locs = np.array([[0,0],[-0.3,-0.7]])
    bboxes = [np.array([225,164]),np.array([225,164])]
    centermaps = []
    for i in range(batch_size):
        centermaps.append(torch.from_numpy(CM.generate_centermap(center_locs,bboxes_hw=bboxes)))
    centermaps = torch.stack(centermaps).cuda()
    CM.print_matrix(centermaps[0,0])
    print('__'*10)
    #5CM.print_matrix(torch.nn.functional.softmax(centermap,1)[0])
    for i in range(batch_size):
        result = CM.parse_centermap(centermaps[i])
        print(result)