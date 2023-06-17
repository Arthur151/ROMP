
import cv2
import numpy as np
import torch
import cv2
from torch import nn
import torch.nn.functional as F

from ..raft.raft import RAFT
from ..raft.utils import flow_viz

class FlowExtract(nn.Module):
    def __init__(self, model_path, device='cuda'):
        super(FlowExtract, self).__init__()
        model = torch.nn.DataParallel(RAFT())
        model.load_state_dict(torch.load(model_path))
        self.device = device
        self.model = model.module.to(self.device).eval()
    
    @torch.no_grad()
    def forward(self, images, source_img_inds, target_img_inds):
        input_images = images.permute(0, 3, 1, 2).to(self.device)
        # flow in low resolution, flow in input resolution
        flows_low, flows_high = self.model(input_images[source_img_inds].contiguous(), input_images[target_img_inds].contiguous(), iters=20, upsample=False, test_mode=True)
        flows = F.interpolate(flows_high, size=(128,128), mode='bilinear', align_corners=True) / 8
        return flows
    
def show_seq_flow(images, flows):
    for img, flo in zip(images, flows):
        img = img.cpu().numpy()
        flo = flo.permute(1,2,0).cpu().numpy()
        
        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        flo = cv2.resize(flo, img.shape[:2])
        img_flo = np.concatenate([img, flo], axis=1)

        img2show = img_flo[:, :, [2,1,0]]/255.0
        h, w = img2show.shape[:2]
        #img2show = cv2.resize(img2show, (w//2, h//2))
        cv2.imshow('image', img2show)
        cv2.waitKey()


def load_image(imfile):
    img = cv2.imread(imfile)[:,:,[2,1,0]]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to('cuda')


def show_flow(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    img2show = img_flo[:, :, [2,1,0]]/255.0
    h, w = img2show.shape[:2]
    img2show = cv2.resize(img2show, (w//2, h//2))
    cv2.imshow('image', img2show)
    cv2.waitKey()

if __name__ == '__main__':
    demo('/home/yusun/data_drive3/datasets/DAVIS-data/DAVIS/JPEGImages/480p/motocross-jump')
