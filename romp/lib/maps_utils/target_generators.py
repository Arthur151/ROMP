#Brought from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/lib/dataset/target_generators/target_generators.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        # gaussian kernel with size.
        gaussian_distribution = - ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
        self.g = np.exp(gaussian_distribution)
        #for k in self.g:
        #    print_item = ''
        #    for i in k:
        #        print_item+='{:.4f} '.format(i)
        #    print(print_item)

    def single_process(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

    def batch_process(self,batch_joints):
        vis = ((batch_joints>-1.).sum(-1)==batch_joints.shape[-1]).unsqueeze(-1).float()
        batch_joints = (torch.cat([batch_joints,vis],-1).unsqueeze(1)+1)/2 * self.output_res 
        heatmaps = []
        for joints in batch_joints:
            heatmaps.append(torch.from_numpy(self.single_process(joints)))
        return torch.stack(heatmaps).cuda()


class ScaleAwareHeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints

    def get_gaussian_kernel(self, sigma):
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        for p in joints:
            sigma = p[0, 3]
            g = self.get_gaussian_kernel(sigma)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], g[a:b, c:d])
        return hms


class JointsGenerator():
    def __init__(self, max_num_people, num_joints, output_res, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.output_res = output_res
        self.tag_per_joint = tag_per_joint

    def single_process(self, joints):
        visible_nodes = np.zeros((self.max_num_people, self.num_joints, 2))
        output_res = self.output_res
        for i in range(min(len(joints),self.max_num_people)):
            tot = 0
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                if pt[2] > 0 and x >= 0 and y >= 0 \
                   and x < self.output_res and y < self.output_res:
                    if self.tag_per_joint:
                        visible_nodes[i][tot] = \
                            (idx * output_res**2 + y * output_res + x, 1)
                    else:
                        visible_nodes[i][tot] = \
                            (y * output_res + x, 1)
                    tot += 1

        return visible_nodes

    def batch_process(self,batch_joints):
        vis = ((batch_joints>-1.).sum(-1)==batch_joints.shape[-1]).unsqueeze(-1).float()
        batch_joints = (torch.cat([batch_joints,vis],-1).unsqueeze(1)+1)/2 * self.output_res 
        joints_processed = []
        for joints in batch_joints:
            joints_processed.append(self.single_process(joints))
        return torch.from_numpy(np.array(joints_processed)).long().cuda()

if __name__ == '__main__':
    num_joints = 17
    output_res = 128
    if 1:
        hg = HeatmapGenerator(output_res,num_joints)

        x = torch.rand(1,num_joints,2).cuda()*2-1
        x[0,:2] = -2.
        heatmaps = hg.batch_process(x)
        imgs = heatmaps[0].cpu().numpy()

        import cv2
        for idx,img in enumerate(imgs):
            cv2.imwrite('test_heatmaps{}.png'.format(idx), (img[:,:,np.newaxis]*255).astype(np.uint8))
    else:
        jg = JointsGenerator(1,num_joints,output_res,True)
        x = torch.rand(1,num_joints,2).cuda()*2-1
        x[0,:10] = -2.
        print(x)
        results = jg.batch_process(x)
        print(results[0,:,:,-1].sum())
        print(results)