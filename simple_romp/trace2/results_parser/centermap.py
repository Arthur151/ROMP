import torch
import numpy as np

class CenterMap(object):
    def __init__(self,centermap_conf_thresh=0.05,style='heatmap_adaptive_scale'):
        self.style=style
        self.size = 128
        self.max_person = 64
        self.shrink_scale = float(512//self.size)
        self.kernel_size = 5
        self.dims = 1
        self.sigma = 1
        self.conf_thresh = centermap_conf_thresh
        print('self.conf_thresh', self.conf_thresh)
        self.gk_group, self.pool_group = self.generate_kernels([self.kernel_size])
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

    def process_gt_CAM(self, center_normed):
        center_list = []
        valid_mask = center_normed[:,:,0]>-1
        valid_inds = torch.where(valid_mask)
        valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
        center_gt = ((center_normed+1)/2*self.size).long()
        center_gt_valid = center_gt[valid_mask]
        return (valid_batch_inds, valid_person_ids, center_gt_valid)
    
    def generate_centermap(self, center_locs, **kwargs):
        if self.style =='heatmap':
            return self.generate_centermap_heatmap(center_locs, **kwargs)
        elif self.style == 'heatmap_adaptive_scale':
            return self.generate_centermap_heatmap_adaptive_scale(center_locs, **kwargs)
        else:
            raise NotImplementedError

    def parse_centermap(self, center_map):
        if self.style =='heatmap':
            return self.parse_centermap_heatmap(center_map)
        elif self.style == 'heatmap_adaptive_scale' and center_map.shape[1]==1:
            return self.parse_centermap_heatmap_adaptive_scale_batch(center_map)
        elif self.style == 'heatmap_adaptive_scale' and center_map.shape[1]==self.size:
            return self.parse_3dcentermap_heatmap_adaptive_scale_batch(center_map)
        else:
            raise NotImplementedError

    def generate_centermap_mask(self,center_locs):
        centermap = np.ones((self.dims,self.size,self.size))
        centermap[-1] = 0
        for center_loc in center_locs:
            map_coord = ((center_loc+1)/2 * self.size).astype(np.int32)-1
            centermap[0,map_coord[0],map_coord[1]] = 0
            centermap[1,map_coord[0],map_coord[1]] = 1
        return centermap

    def generate_centermap_heatmap(self,center_locs, kernel_size=5,**kwargs):
        hms = np.zeros((self.dims, self.size, self.size),dtype=np.float32)
        offset = (kernel_size-1)//2
        for idx, pt in enumerate(center_locs):
            x, y = int((pt[0]+1)/2*self.size), int((pt[1]+1)/2*self.size)
            if x < 0 or y < 0 or \
               x >= self.size or y >= self.size:
                continue

            ul = int(np.round(x - offset)), int(np.round(y - offset))
            br = int(np.round(x + offset+1)), int(np.round(y + offset+1))

            c, d = max(0, -ul[0]), min(br[0], self.size) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.size) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.size)
            aa, bb = max(0, ul[1]), min(br[1], self.size)
            hms[0,aa:bb, cc:dd] = np.maximum(
                hms[0,aa:bb, cc:dd], self.gk_group[kernel_size][a:b, c:d])
        return hms

    def generate_centermap_heatmap_adaptive_scale(self, center_locs, bboxes_hw_norm, occluded_by_who=None,**kwargs):
        '''
           center_locs is in the order of (y,x), corresponding to (w,h), while in the loading data, we have rectified it to the correct (x, y) order
        '''
        radius_list = _calc_radius_(bboxes_hw_norm, map_size=self.size)
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

    def generate_centermap_3dheatmap_adaptive_scale_batch(self, batch_center_locs, radius=3, depth_num=None, device='cuda:0'):
        if depth_num is None:
            depth_num = int(self.size // 2)
        heatmap = torch.zeros((len(batch_center_locs), depth_num, self.size, self.size), device=device)
        
        for bid, center_locs in enumerate(batch_center_locs):
            for cid, center in enumerate(center_locs):
                diameter = int(2 * radius + 1)
                gaussian_patch = gaussian3D(w=diameter, h=diameter, d=diameter,\
                center=(diameter // 2, diameter // 2, diameter // 2), s=float(diameter) / 6, device=device)

                xa, ya, za = int(max(0, center[0] - diameter // 2)), int(max(0, center[1] - diameter // 2)), int(max(0, center[2] - diameter // 2))
                xb, yb, zb = int(min(center[0]+diameter//2, self.size-1)), int(min(center[1]+diameter//2, self.size-1)), int(min(center[2]+diameter//2, depth_num-1))

                gxa = xa - int(center[0] - diameter // 2)
                gya = ya - int(center[1] - diameter // 2)
                gza = za - int(center[2] - diameter // 2)

                gxb = xb + 1 - xa + gxa
                gyb = yb + 1 - ya + gya
                gzb = zb + 1 - za + gza

                heatmap[bid, za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(
                    torch.cat(tuple([
                        heatmap[bid, za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0),
                        gaussian_patch[gza:gzb, gya:gyb, gxa:gxb].unsqueeze(0)
                    ])), 0)[0]
        return heatmap

    def generate_centermap_3dheatmap_adaptive_scale(self, center_locs, depth_num=None, device='cpu'):
        '''
        center_locs: center locations (X,Y,Z) on 3D center map (BxDxHxW)
        '''
        if depth_num is None:
            depth_num = int(self.size // 2)
        heatmap = torch.zeros((depth_num, self.size, self.size)).to(device)
        if len(center_locs)==0:
            return heatmap, False
        
        adaptive_depth_uncertainty = np.array(center_locs)[:,2].astype(np.float16) / depth_num
        depth_uncertainty = ((4 + adaptive_depth_uncertainty * 4).astype(np.int32) // 2) * 2 + 1

        adaptive_image_scale = (1 - adaptive_depth_uncertainty) / 2.
        uv_radius = (_calc_uv_radius_(adaptive_image_scale, map_size=self.size) * 2 + 1).astype(np.int32)
        
        for cid, center in enumerate(center_locs):
            width, height = uv_radius[cid], uv_radius[cid]
            depth = depth_uncertainty[cid]
            diameter = np.linalg.norm([width/2., height/2., depth/2.], ord=2, axis=0) * 2
            
            gaussian_patch = gaussian3D(w=width, h=height, d=depth,\
                center=(width // 2, height // 2, depth // 2), s=float(diameter) / 6, device=device)

            xa, ya, za = int(max(0, center[0] - width // 2)), int(max(0, center[1] - height // 2)), int(max(0, center[2] - depth // 2))
            xb, yb, zb = int(min(center[0] + width // 2, self.size-1)), int(min(center[1] + height // 2, self.size-1)), int(min(center[2] + depth // 2, depth_num-1))

            gxa = xa - int(center[0] - width // 2)
            gya = ya - int(center[1] - height // 2)
            gza = za - int(center[2] - depth // 2)

            gxb = xb + 1 - xa + gxa
            gyb = yb + 1 - ya + gya
            gzb = zb + 1 - za + gza

            heatmap[za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(
                torch.cat(tuple([
                    heatmap[za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0),
                    gaussian_patch[gza:gzb, gya:gyb, gxa:gxb].unsqueeze(0)
                ])), 0)[0]
        return heatmap, True
    
    def generate_centermap_3dheatmap_adaptive_scale_org(self, center_locs, radius=3, depth_num=None, device='cpu'):
        '''
        center_locs: center locations (X,Y,Z) on 3D center map (BxDxHxW)
        '''
        if depth_num is None:
            depth_num = int(self.size // 2)
        heatmap = torch.zeros((depth_num, self.size, self.size)).to(device)
        if len(center_locs)==0:
            return heatmap, False
        
        for cid, center in enumerate(center_locs):
            
            diameter = int(2 * radius + 1)
            
            gaussian_patch = gaussian3D(w=diameter, h=diameter, d=diameter,\
            center=(diameter // 2, diameter // 2, diameter // 2), s=float(diameter) / 6, device=device)

            xa, ya, za = int(max(0, center[0] - diameter // 2)), int(max(0, center[1] - diameter // 2)), int(max(0, center[2] - diameter // 2))
            xb, yb, zb = int(min(center[0]+diameter//2, self.size-1)), int(min(center[1]+diameter//2, self.size-1)), int(min(center[2]+diameter//2, depth_num-1))

            gxa = xa - int(center[0] - diameter // 2)
            gya = ya - int(center[1] - diameter // 2)
            gza = za - int(center[2] - diameter // 2)

            gxb = xb + 1 - xa + gxa
            gyb = yb + 1 - ya + gya
            gzb = zb + 1 - za + gza

            heatmap[za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(
                torch.cat(tuple([
                    heatmap[za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0),
                    gaussian_patch[gza:gzb, gya:gyb, gxa:gxb].unsqueeze(0)
                ])), 0)[0]
        return heatmap, True


    def multi_channel_nms(self,center_maps):
        center_map_pooled = []
        for depth_idx, center_map in enumerate(center_maps):
            center_map_pooled.append(nms(center_map[None], pool_func=self.pool_group[self.kernel_size]))
        center_maps_max = torch.max(torch.cat(center_map_pooled,0),0).values
        center_map_nms = nms(center_maps_max[None], pool_func=self.pool_group[self.kernel_size])[0]
        return center_map_nms

    def parse_centermap_mask(self,center_map):
        center_map_bool = torch.argmax(center_map,1).bool()
        center_idx = torch.stack(torch.where(center_map_bool)).transpose(1,0)
        return center_idx

    def parse_centermap_heatmap(self,center_maps):
        if center_maps.shape[0]>1:
            center_map_nms = self.multi_channel_nms(center_maps)
        else:
            center_map_nms = nms(center_maps, pool_func=self.pool_group[self.kernel_size])[0]
        h, w = center_map_nms.shape

        centermap = center_map_nms.view(-1)
        confidence, index = centermap.topk(self.max_person)
        x = index%w
        y = (index/w).long()
        idx_topk = torch.stack((y,x),dim=1)
        center_preds, conf_pred = idx_topk[confidence>self.conf_thresh], confidence[confidence>self.conf_thresh]
        return center_preds, conf_pred

    def parse_centermap_heatmap_adaptive_scale(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[self.kernel_size])[0]
        h, w = center_map_nms.shape

        centermap = center_map_nms.view(-1)
        confidence, index = centermap.topk(self.max_person)
        x = index%w
        y = (index/float(w)).long()
        idx_topk = torch.stack((y,x),dim=1)
        center_preds, conf_pred = idx_topk[confidence>self.conf_thresh], confidence[confidence>self.conf_thresh]
        return center_preds, conf_pred

    def parse_centermap_heatmap_adaptive_scale_batch(self, center_maps, top_n_people=None):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[self.kernel_size])
        b, c, h, w = center_map_nms.shape
        K = self.max_person if top_n_people is None else top_n_people

        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = torch.div(topk_inds.long(), w).float()
        topk_xs = (topk_inds % w).int().float()
        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = torch.div(index.long(), K)
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)

        if top_n_people is not None:
            mask = topk_score>0
            mask[:] = True
        else:
            mask = topk_score>self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_yxs = torch.stack([topk_ys[mask], topk_xs[mask]]).permute((1,0))
        return batch_ids, topk_inds[mask], center_yxs, topk_score[mask]

    def parse_3dcentermap_heatmap_adaptive_scale_batch(self, center_maps, top_n_people=None):
        center_map_nms = nms(center_maps, pool_func=self.maxpool3d).squeeze(1)
        b, c, h, w = center_map_nms.shape

        K = self.max_person if top_n_people is None else top_n_people

        # acquire top k value/index at each depth
        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = torch.div(topk_inds.long(), w).float()
        topk_xs = (topk_inds % w).int().float()
        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        # div by K because index is grouped by K(C x K shape)
        topk_zs = torch.div(index.long(), K)
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)

        if top_n_people is not None:
            mask = topk_score>0
            mask[:] = True
        else:
            mask = topk_score>self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_zyxs = torch.stack([topk_zs[mask], topk_ys[mask], topk_xs[mask]]).permute((1,0)).long()

        return [batch_ids, center_zyxs, topk_score[mask]]
    
    def parse_local_centermap3D(self, center_maps, pred_batch_ids, center_yxs, only_max=False):
        if len(center_yxs) == 0:
            return [], [], []
        cys = center_yxs[:, 0]
        cxs = center_yxs[:, 1]
        center_map_nms = nms(center_maps, pool_func=self.maxpool3d).squeeze(1)
        b, c, h, w = center_map_nms.shape
        cys = torch.clip(cys, 0, h-1)
        cxs = torch.clip(cxs, 0, w-1)
        device = center_maps.device
        local_K = 16
        
        czyxs = []
        new_pred_batch_inds = []
        top_scores = []
        # TODO: select the surrounding volume of the center points for depth localization
        for batch_id, cy, cx in zip(pred_batch_ids, cys, cxs):
            local_vec = center_map_nms[batch_id, :, cy, cx]
            topk_scores, topk_zs = torch.topk(local_vec, local_K)

            if only_max:
                mask = torch.zeros(len(topk_scores)).bool()
                mask[0] = True
            else:
                mask = topk_scores > self.conf_thresh
                if mask.sum() == 0:
                    mask[0] = True
                    
            for cz, score in zip(topk_zs[mask], topk_scores[mask]):
                czyxs.append(torch.Tensor([cz, cy, cx]))
                new_pred_batch_inds.append(batch_id)
                top_scores.append(score)
        czyxs = torch.stack(czyxs).long().to(device)
        new_pred_batch_inds = torch.Tensor(new_pred_batch_inds).long().to(device)
        top_scores = torch.Tensor(top_scores).to(device)
        return new_pred_batch_inds, czyxs, top_scores
        

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
    scales = np.linalg.norm(np.stack(bboxes_hw_norm, 0)/2, ord=2, axis=1)
    radius = (scales * scale_factor + minimum_radius).astype(np.uint8)
    return radius

def _calc_uv_radius_(scales, map_size=64):
    minimum_radius = map_size / 32.
    scale_factor = map_size / 16.
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

def gaussian3D(d, h, w, center, s=2, device='cuda'):
    """
    :param d: hmap depth
    :param h: hmap height
    :param w: hmap width
    :param center: center of the Gaussian | ORDER: (x, y, z)
    :param s: sigma of the Gaussian
    :return: heatmap (shape torch.Size([d, h, w])) with a gaussian centered in `center`
    """
    x = torch.arange(0, w, 1).float().to(device)
    y = torch.arange(0, h, 1).float().to(device)
    y = y.unsqueeze(1)
    z = torch.arange(0, d, 1).float().to(device)
    z = z.unsqueeze(1).unsqueeze(1)

    x0 = center[0]
    y0 = center[1]
    z0 = center[2]

    return torch.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / s ** 2)

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