import torch
import numpy as np
import lap

def linear_assignment(cost_matrix, thresh=100.):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    matches = np.asarray(matches)
    #unmatched_a = np.where(x < 0)[0]
    #unmatched_b = np.where(y < 0)[0]
    
    return matches #, unmatched_a, unmatched_b

def calc_dist_matix(pred_trajs, gt_trajs):
    valid_mask = (gt_trajs!=-2).sum(-1)!=0
    valid_mask = valid_mask.unsqueeze(1).repeat(1,len(pred_trajs),1).float()
    dist_matix = torch.norm(gt_trajs.unsqueeze(1).repeat(1,len(pred_trajs),1,1)-pred_trajs.unsqueeze(0).repeat(len(gt_trajs),1,1,1),p=2, dim=-1)
    dist_matix = (dist_matix * valid_mask).sum(-1) / (valid_mask.sum(-1)+1e-4)

    return dist_matix
    
def match_traj_to_3D_2D_gts(traj3D_gts, traj2D_gts, Tj_flag, traj_preds, pred_batch_ids):
    mc = {key:[] for key in ['batch_ids', 'matched_ids', 'person_ids']}

    unique_batch_ids = torch.unique(pred_batch_ids)
    # matching 3D trajectory
    for batch_id in unique_batch_ids:
        pred_mask = pred_batch_ids == batch_id
        batch_ids = pred_batch_ids[pred_mask]
        pred_ids = torch.where(pred_mask)[0]
        pred_trajs = traj_preds[pred_mask]

        if Tj_flag[batch_id,1]: # have 3D traj gt
            gt_trajs = traj3D_gts[batch_id] # max_person_num, args().temp_clip_length, 3
        elif Tj_flag[batch_id,0]: # have 2D traj gt
            gt_trajs = traj2D_gts[batch_id] # max_person_num, args().temp_clip_length, 2
            pred_trajs = pred_trajs[:,:,[2,1]]
        else:
            continue
        
        gt_mask = (gt_trajs!=-2).sum(-1).sum(-1)>0
        gt_trajs = gt_trajs[gt_mask]
        person_ids = torch.where(gt_mask)[0]

        dist_matrix = calc_dist_matix(pred_trajs.detach(), gt_trajs)
        matches = linear_assignment(dist_matrix.cpu().numpy())
        if len(matches) == 0:
            continue
        person_ids = person_ids[matches[:,0]]
        pred_ids = pred_ids[matches[:,1]]
        batch_ids = batch_ids[matches[:,1]]

        mc['batch_ids'].append(batch_ids)
        mc['matched_ids'].append(pred_ids)
        mc['person_ids'].append(person_ids)
    
    if len(mc['matched_ids'])==0:
        mc.update({'batch_ids':[0], 'matched_ids':[0], 'person_ids':[0]})
    keys_list = list(mc.keys())
    for key in keys_list:
        mc[key] = torch.cat(mc[key], 0).long().to(traj_preds.device)
    return mc
