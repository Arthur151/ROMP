import numpy as np
import cv2
import torch
from .post_parser import remove_subjects

def padding_image_overlap(image, overlap_ratio=0.46):
    h, w = image.shape[:2]
    pad_length = int(h* overlap_ratio)
    pad_w = w+2*pad_length
    pad_image = np.zeros((h, pad_w, 3), dtype=np.uint8)
    top, left = 0, pad_length
    bottom, right = h, w+pad_length
    pad_image[top:bottom, left:right] = image
    
    # due to BEV takes square input, so we convert top, bottom to the state that assuming square padding
    pad_height = (w - h)//2
    top = pad_height
    bottom = w - top
    left = 0
    right = w
    image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
    return pad_image, image_pad_info, pad_length

def get_image_split_plan(image, overlap_ratio=0.46):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    slide_time = int(np.ceil((aspect_ratio - 1) / (1 - overlap_ratio))) + 1

    crop_box = [] # left, right, top, bottom
    move_step = (1 - overlap_ratio) * h 
    for ind in range(slide_time):
        if ind == (slide_time-1):
            left = w-h
        else:
            left = move_step * ind
            right = left+h
        crop_box.append([left, right, 0, h])

    return np.array(crop_box).astype(np.int32)

def exclude_boudary_subjects(outputs, drop_boundary_ratio, ptype='left', torlerance=0.05):
    if ptype=='left':
        drop_mask = outputs['cam'][:, 2] > (1 - drop_boundary_ratio + torlerance)
    elif ptype=='right':
        drop_mask = outputs['cam'][:, 2] < (drop_boundary_ratio - 1 - torlerance)
    remove_subjects(outputs, torch.where(drop_mask)[0])

def convert_crop_cam_params2full_image(cam_params, crop_bbox, image_shape):
    h, w = image_shape
    # adjust scale, cam 3: depth, y, x
    scale_adjust = (crop_bbox[[1,3]]-crop_bbox[[0,2]]).max() / max(h, w)
    cam_params *= scale_adjust

    # adjust x
    # crop_bbox[:2] -= pad_length
    bbox_mean_x = crop_bbox[:2].mean()
    cam_params[:,2] += bbox_mean_x / (w /2) - 1
    return cam_params

def collect_outputs(outputs, all_outputs):
    keys = list(outputs.keys())
    for key in keys:
        if key not in all_outputs:
            all_outputs[key] = outputs[key]
        else:
            if key in ['smpl_face']:
                continue
            if key in ['center_map']:
                all_outputs[key] = torch.cat([all_outputs[key], outputs[key]],3)
                continue
            if key in ['center_map_3d']:
                all_outputs[key] = torch.cat([all_outputs[key], outputs[key]],2)
                continue
            all_outputs[key] = torch.cat([all_outputs[key], outputs[key]],0)
