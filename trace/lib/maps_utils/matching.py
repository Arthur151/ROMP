import torch
import numpy as np
from config import args
from utils.center_utils import process_gt_center, parse_gt_center3d
from maps_utils.matching_utils import greedy_matching_kp2ds

def match_with_kp2ds(meta_data, outputs, cfg): 
    #print(outputs['pj2d'].shape, meta_data['full_kp2d'].shape, meta_data['valid_masks'][:,:,0].shape)
    pred_kp2ds = outputs['pj2d']
    device = pred_kp2ds.device
    pred_batch_ids = outputs['pred_batch_ids'].long()
    body_center_confs = outputs['top_score']

    # get valid kp2d gts
    gt_kp2ds = meta_data['full_kp2d']
    valid_mask = meta_data['valid_masks'][:,:,0]
    vgt_batch_ids, vgt_person_ids = torch.where(valid_mask)
    vgt_kp2ds = gt_kp2ds[vgt_batch_ids, vgt_person_ids].to(device)
    vgt_valid_mask = ((vgt_kp2ds!=-2.).sum(-1)==2).sum(-1)>0
    if vgt_valid_mask.sum()!=len(vgt_valid_mask):
        vgt_kp2ds, vgt_batch_ids, vgt_person_ids = vgt_kp2ds[vgt_valid_mask], vgt_batch_ids[vgt_valid_mask], vgt_person_ids[vgt_valid_mask]

    mc = {key:[] for key in ['batch_ids', 'matched_ids', 'person_ids', 'conf']}

    # matching each prediction with a gt for supervision
    matching_batch_ids = torch.unique(pred_batch_ids)
    for batch_id in matching_batch_ids:
        gt_ids = torch.where(vgt_batch_ids==batch_id)[0]
        if len(gt_ids) == 0:
            continue
        
        pred_ids = torch.where(pred_batch_ids==batch_id)[0]
        # ensure a gt is only matched to a single predictions, avoiding matching the same easier gt to multiple predictions. 
        # Force model to learn the hard occluded subjects.
        pred_kp2ds_matching = pred_kp2ds[pred_ids].detach().cpu().numpy()
        gt_kp2ds_matching = vgt_kp2ds[gt_ids].cpu().numpy()
        gt_valid_mask_matching = (gt_kp2ds_matching == -2.).sum(-1) == 0
        #print('matching inputs', len(pred_kp2ds_matching), len(gt_kp2ds_matching), len(gt_valid_mask_matching))
        bestMatch, falsePositives, misses = greedy_matching_kp2ds(pred_kp2ds_matching, gt_kp2ds_matching, gt_valid_mask_matching)
        #print('matching outputs', bestMatch, falsePositives, misses)
        for pid, gtid in bestMatch:
            matched_gt_id = gt_ids[gtid]
            gt_batch_id = vgt_batch_ids[matched_gt_id]
            gt_person_id = vgt_person_ids[matched_gt_id]

            pred_batch_id = pred_ids[pid]

            mc['batch_ids'].append(gt_batch_id)
            mc['person_ids'].append(gt_person_id)
            mc['matched_ids'].append(pred_batch_id)
            mc['conf'].append(body_center_confs[int(pred_batch_id)])

    if len(mc['matched_ids'])==0:
        print('matching failed in match_gt_pred_3d_2d of result_parser.py')
        mc.update({'batch_ids':[0], 'matched_ids':[0], 'person_ids':[0], 'conf':[0]})
    
    keys_list = list(mc.keys())
    for key in keys_list:
        if key == 'conf':
            mc[key] = torch.Tensor(mc[key]).to(device)
        else:
            mc[key] = torch.Tensor(mc[key]).long().to(device)
        if args().max_supervise_num!=-1 and cfg['is_training']:
            mc[key] = mc[key][:args().max_supervise_num]
    return mc['batch_ids'], mc['person_ids'], mc['matched_ids'], mc['conf']


def match_with_2d_centers(center_gts_info, center_preds_info, device, is_training):
    vgt_batch_ids, vgt_person_ids, vgt_centers = center_gts_info
    vpred_batch_ids, flat_inds, cyxs, top_score = center_preds_info
    mc = {key:[] for key in ['batch_ids', 'flat_inds', 'person_ids', 'conf']}

    if args().match_preds_to_gts_for_supervision:
        for match_ind in torch.arange(len(vgt_batch_ids)):
            batch_id, person_id, center_gt = vgt_batch_ids[match_ind], vgt_person_ids[match_ind], vgt_centers[match_ind]
            pids = torch.where(vpred_batch_ids==batch_id)[0]
            if len(pids) == 0:
                continue

            closet_center_ind = pids[torch.argmin(torch.norm(cyxs[pids].float()-center_gt[None].float().to(device),dim=-1))]
            center_matched = cyxs[closet_center_ind].long()
            cy, cx = torch.clamp(center_matched, 0, args().centermap_size-1)
            flat_ind = cy*args().centermap_size+cx
            mc['batch_ids'].append(batch_id)
            mc['flat_inds'].append(flat_ind)
            mc['person_ids'].append(person_id)
            mc['conf'].append(top_score[closet_center_ind])
        
        keys_list = list(mc.keys())
        for key in keys_list:
            if key != 'conf':
                mc[key] = torch.Tensor(mc[key]).long().to(device)
            if args().max_supervise_num!=-1 and is_training:
                mc[key] = mc[key][:args().max_supervise_num]
    if not args().match_preds_to_gts_for_supervision or len(mc['batch_ids'])==0:
        mc['batch_ids'] = vgt_batch_ids.long().to(device)
        mc['flat_inds'] = flatten_inds(vgt_centers.long()).to(device)
        mc['person_ids'] = vgt_person_ids.long().to(device)
        mc['conf'] = torch.zeros(len(vgt_person_ids)).to(device)
    return mc

def match_with_3d_2d_centers(meta_data, outputs, cfg):  
    with_2d_matching=cfg['with_2d_matching']
    is_training = cfg['is_training']
    cam_mask = meta_data['cam_mask']
    batch_size = len(cam_mask)
    pred_batch_ids = outputs['pred_batch_ids']
    pred_czyxs = outputs['pred_czyxs']
    top_score = outputs['top_score']
    device = outputs['center_map_3d'].device

    center_gts_info_3d = parse_gt_center3d(cam_mask, meta_data['cams'])
    person_centers = meta_data['person_centers'].clone()
    # exclude the 2D body centers with precise 3D center locations.
    person_centers[cam_mask] = -2.
    center_gts_info_2d = process_gt_center(person_centers)    

    vgt_batch_ids, vgt_person_ids, vgt_centers = center_gts_info_2d
    vgt_batch_ids_3d, vgt_person_ids_3d, vgt_czyxs = center_gts_info_3d
    mc = {key:[] for key in ['batch_ids', 'matched_ids', 'person_ids', 'conf']}

    # 3D center matching
    for match_ind in torch.arange(len(vgt_batch_ids_3d)):
        batch_id, person_id, center_gt = vgt_batch_ids_3d[match_ind], vgt_person_ids_3d[match_ind], vgt_czyxs[match_ind]
        pids = torch.where(pred_batch_ids==batch_id)[0]
        if len(pids) == 0:
            continue
        center_dist_3d = torch.norm(pred_czyxs[pids].float()-center_gt[None].float().to(device),dim=-1)
        #if torch.min(center_dist_3d).item()<args().center_dist_thresh_3d:
        matched_pred_id = pids[torch.argmin(center_dist_3d)]
        mc['batch_ids'].append(batch_id)
        mc['matched_ids'].append(matched_pred_id)
        mc['person_ids'].append(person_id)
        mc['conf'].append(top_score[matched_pred_id])

    # 2D center matching
    for match_ind in torch.arange(len(vgt_batch_ids)):
        batch_id, person_id, center_gt = vgt_batch_ids[match_ind], vgt_person_ids[match_ind], vgt_centers[match_ind]
        pids = torch.where(pred_batch_ids==batch_id)[0]
        if len(pids) == 0:
            continue

        matched_pred_id = pids[torch.argmin(torch.norm(pred_czyxs[pids,1:].float()-center_gt[None].float().to(device),dim=-1))]
        center_matched = pred_czyxs[matched_pred_id].long()
        mc['batch_ids'].append(batch_id)
        mc['matched_ids'].append(matched_pred_id)
        mc['person_ids'].append(person_id)
        mc['conf'].append(top_score[matched_pred_id])

    if args().eval_2dpose:
        for inds, (batch_id, person_id, center_gt) in enumerate(zip(vgt_batch_ids, vgt_person_ids, vgt_centers)):
            if batch_id in pred_batch_ids:
                center_pred = pred_czyxs[pred_batch_ids==batch_id]
                matched_id = torch.argmin(torch.norm(center_pred[:,1:].float()-center_gt[None].float().to(device),dim=-1))
                matched_pred_id = np.where((pred_batch_ids==batch_id).cpu())[0][matched_id]
                mc['matched_ids'].append(matched_pred_id)
                mc['batch_ids'].append(batch_id); mc['person_ids'].append(person_id)

    if len(mc['matched_ids'])==0:
        #print('matching failed in match_gt_pred_3d_2d of result_parser.py')
        mc.update({'batch_ids':[0], 'matched_ids':[0], 'person_ids':[0], 'conf':[0]})
    keys_list = list(mc.keys())
    for key in keys_list:
        if key == 'conf':
            mc[key] = torch.Tensor(mc[key]).to(device)
        else:
            mc[key] = torch.Tensor(mc[key]).long().to(device)
        if args().max_supervise_num!=-1 and is_training:
            mc[key] = mc[key][:args().max_supervise_num]
    
    return mc['batch_ids'], mc['person_ids'], mc['matched_ids'], mc['conf']


matching_gts2preds = {
    'kp2ds': match_with_kp2ds,
    '3D+2D_center': match_with_3d_2d_centers,
    '2D_center': match_with_2d_centers,
}