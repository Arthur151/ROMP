import torch
import constants
from config import args
import numpy as np
from .cam_utils import convert_cam_params_to_centermap_coords

def denormalize_center(center, size=args().centermap_size):
    center = (center+1)/2*size

    center[center<1] = 1
    center[center>size - 1] = size - 1
    if isinstance(center, np.ndarray):
        center = center.astype(np.int32)
    elif isinstance(center, torch.Tensor):
        center = center.long()
    return center

def process_gt_center(center_normed):
    valid_mask = center_normed[:,:,0]>-1
    valid_inds = torch.where(valid_mask)
    valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
    # restrain the normalized center >= -1, so that the denormalized center > 0
    center_normed[valid_inds] = torch.max(center_normed[valid_inds], torch.ones_like(center_normed[valid_inds])*-1)
    center_gt = ((center_normed+1)/2*(args().centermap_size-1)).long()
    # restrain the denormalized center <= centermap_size-1
    center_gt = torch.min(center_gt, torch.ones_like(center_gt)*(args().centermap_size-1))
    center_gt_valid = center_gt[valid_mask]
    return (valid_batch_inds, valid_person_ids, center_gt_valid)


def parse_gt_center3d(cam_mask, cams, size=args().centermap_size):
    batch_ids, person_ids = torch.where(cam_mask)
    cam_params = cams[batch_ids, person_ids]
    centermap_coords = convert_cam_params_to_centermap_coords(cam_params)
    czyxs = denormalize_center(centermap_coords, size=size)
    #sample_view_ids = determine_sample_view(batch_ids,czyxs)
    return batch_ids, person_ids, czyxs


def determine_sample_view(batch_ids,czyxs,thresh=3.):
    batch_ids_unique = torch.unique(batch_ids)
    sample_view_ids = torch.zeros_like(batch_ids).long()
    for batch_id in batch_ids_unique:
        person_mask = batch_ids == batch_id
        if person_mask.sum()==1:
            continue
        sample_czyxs = czyxs[person_mask]
        sample_view_id = torch.zeros(len(sample_czyxs)).to(czyxs.device)
        for inds, czyx in enumerate(sample_czyxs):
            dist = torch.norm(sample_czyxs[:,1:] - czyx[1:][None].float(), dim=-1, p=2)
            sample_view_id[inds] = (dist<thresh).sum() > 0 
        sample_view_ids[person_mask] = sample_view_id.long()
    return sample_view_ids

if __name__ == '__main__':
    test_projection_depth()