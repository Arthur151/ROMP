import torch
import numpy as np
from ..tracker.tracker3D import Tracker
from ..utils.utils import denormalize_cam_params_to_trans, OneEuroFilter

def convert_traj2D2center_yxs(traj2D_gts, outmap_size, seq_mask):
    """
    Flattening N valuable trajectories in 2D body centers, traj2D_gts,
        from shape (batch, 64, 8, 2) to (N, 8, 2).
    Inputs:
        traj2D_gts: torch.Tensor, shape [batch, person_num, clip, 2] the ground truth 2D body centers
    Return:
        batch_inds: torch.Tensor, shape [N]  
        traj2D_cyxs: torch.Tensor, shape [N, clip, 2]  
        top_scores: torch.Tensor, shape [N]  
        gt_inds: torch.Tensor, shape [N, 2]  the (batch, subject) Index in gts matrix.
    """
    batch_size, max_person_num, clip_length, dim = traj2D_gts.shape
    device = traj2D_gts.device

    batch_inds = []
    gt_inds = []
    traj2D_cyxs = []
    seq_masks = []
    for batch_id in range(batch_size):
        for person_id in range(max_person_num):
            valid_mask = traj2D_gts[batch_id,person_id][:,0] != -2
            if valid_mask.sum() == 0:
                break
            # remove the subjects without complete trajectory and pose annotations
            if valid_mask.sum()!=clip_length and seq_mask[batch_id*clip_length]:
                continue

            cyxs = traj2D_gts[batch_id,person_id].clone()
            cyxs[valid_mask] = (cyxs[valid_mask] + 1) / 2 * outmap_size
            if not seq_mask[batch_id*clip_length]:
                cyxs = cyxs[valid_mask]
                valid_clip_inds = torch.where(valid_mask)[0].cpu()
                batch_ind = torch.Tensor([batch_id*clip_length+i for i in valid_clip_inds]).long()
                valid_mask = valid_mask[valid_mask]
                seq_masks.append(torch.zeros(len(batch_ind)).bool())
            else:
                batch_ind = torch.Tensor([batch_id*clip_length+i for i in range(clip_length)]).long()
                seq_masks.append(torch.ones(len(batch_ind)).bool())
                valid_clip_inds = torch.arange(clip_length)
            batch_ind[~valid_mask] = -1
            traj2D_cyxs.append(cyxs)
            batch_inds.append(batch_ind)
            gt_inds.append(torch.stack([torch.ones(len(valid_clip_inds))*batch_id, \
                                            torch.ones(len(valid_clip_inds))*person_id, valid_clip_inds], 1).long())
            
    traj2D_cyxs = torch.cat(traj2D_cyxs, 0).long().to(device)
    batch_inds = torch.cat(batch_inds, 0).to(device)
    gt_inds = torch.cat(gt_inds, 0).to(device)
    seq_masks = torch.cat(seq_masks, 0)

    top_scores = torch.ones(len(batch_inds)).to(device)
    return batch_inds, traj2D_cyxs, top_scores, gt_inds, seq_masks

def seach_clip_id(batch_id, seq_inds):
    batch_ids = seq_inds[:,2]
    clip_id = seq_inds[torch.where(batch_ids==batch_id)[0],1]
    return clip_id

def prepare_complete_trajectory_features(self, seq_tracking_results, mesh_feature_maps, seq_inds):
    seq_traj_features = {}
    seq_traj_czyxs = {}
    seq_traj_batch_inds = {}
    seq_traj_valid_masks = {}
    seq_masks = {}
    seq_track_ids = {}
    for seq_id, (center_traj3Ds, batch_inds, seq_flags) in seq_tracking_results.items():
        seq_traj_features[seq_id] = []
        seq_traj_czyxs[seq_id] = []
        seq_traj_batch_inds[seq_id] = []
        seq_traj_valid_masks[seq_id] = []
        seq_masks[seq_id] = []
        seq_track_ids[seq_id] = []
        for track_id in center_traj3Ds:
            subj_center3D_traj = center_traj3Ds[track_id]
            subj_batch_inds = batch_inds[track_id]
            seq_flag = seq_flags[track_id]
            valid_mask = torch.ones(len(subj_batch_inds)).bool()
            #print(seq_id, track_id, subj_center3D_traj, subj_batch_inds)

            # in-filling the missed part in trajectories via interpolating the neighborhoods (the fore and the after)
            fore_subj_center3D_traj = subj_center3D_traj.clone().detach()
            fore_subj_center3D_traj_weight = torch.zeros(len(subj_center3D_traj)).to(mesh_feature_maps.device) + 0.5
            fore_subj_batch_inds = subj_batch_inds.clone().detach()
            after_subj_center3D_traj = subj_center3D_traj.clone().detach()
            after_subj_center3D_traj_weight = torch.zeros(len(subj_center3D_traj)).to(mesh_feature_maps.device) + 0.5
            after_subj_batch_inds = subj_batch_inds.clone().detach()

            for sbid in range(len(subj_batch_inds)):
                subj_batch_ind = subj_batch_inds[sbid]
                
                # TODO: proper method to infilling the features of missed detection in trajectory.
                if sbid == 0 and subj_batch_ind == -1:
                    valid_ind = torch.where(subj_batch_inds != -1)[0]
                    if len(valid_ind) == 0:
                        print('subj_batch_inds', subj_batch_inds, subj_center3D_traj)
                    valid_ind = 0 if len(valid_ind) == 0 else valid_ind[0] # TODO: fix this outrange error.
                    
                    fore_subj_center3D_traj[sbid] = fore_subj_center3D_traj[valid_ind]
                    after_subj_center3D_traj[sbid] = after_subj_center3D_traj[valid_ind]
                    fore_subj_batch_inds[sbid] = subj_batch_inds[valid_ind]
                    after_subj_batch_inds[sbid] = subj_batch_inds[valid_ind]
                    subj_batch_inds[0] = subj_batch_inds[valid_ind] - seach_clip_id(subj_batch_inds[valid_ind], seq_inds)
                    valid_mask[0] = False
                if subj_batch_ind == -1:
                    valid_fore_ind = torch.where(subj_batch_inds[:sbid] != -1)[0].max()
                    if len(torch.where(subj_batch_inds[(sbid+1):] != -1)[0])>0:
                        valid_after_ind = sbid + 1 + torch.where(subj_batch_inds[(sbid+1):] != -1)[0].min()
                        if valid_after_ind == valid_fore_ind:
                            print('Error!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            print('____'*20)
                            print('valid_after_ind == valid_fore_ind would cause nan for division','at prepare_complete_trajectory_features')
                            print('____'*20)
                        fore_subj_center3D_traj_weight[sbid] = (valid_after_ind - sbid) / (valid_after_ind - valid_fore_ind)
                        after_subj_center3D_traj_weight[sbid] = (sbid - valid_fore_ind) / (valid_after_ind - valid_fore_ind)
                    else:
                        valid_after_ind = valid_fore_ind
                    fore_subj_center3D_traj[sbid] = fore_subj_center3D_traj[valid_fore_ind]
                    after_subj_center3D_traj[sbid] = after_subj_center3D_traj[valid_after_ind]
                    
                    fore_subj_batch_inds[sbid] = subj_batch_inds[valid_fore_ind]
                    after_subj_batch_inds[sbid] = subj_batch_inds[valid_after_ind]
                    #print(fore_subj_center3D_traj_weight[sbid].item(), fore_subj_batch_inds[sbid], after_subj_batch_inds[sbid], after_subj_center3D_traj_weight[sbid].item())
                    #print(sbid, subj_batch_inds, valid_fore_ind, seq_inds)
                    subj_batch_inds[sbid] = sbid + subj_batch_inds[valid_fore_ind] - seach_clip_id(subj_batch_inds[valid_fore_ind], seq_inds)
                    valid_mask[sbid] = False
            fore_subj_features = self.image_feature_sampling(mesh_feature_maps, fore_subj_center3D_traj, fore_subj_batch_inds)
            after_subj_features = self.image_feature_sampling(mesh_feature_maps, after_subj_center3D_traj, after_subj_batch_inds)
            subj_features = fore_subj_features * fore_subj_center3D_traj_weight.unsqueeze(1) + \
                            after_subj_center3D_traj_weight.unsqueeze(1) * after_subj_features
            
            subj_czyxs = fore_subj_center3D_traj * fore_subj_center3D_traj_weight.unsqueeze(1) + \
                        after_subj_center3D_traj * after_subj_center3D_traj_weight.unsqueeze(1)
            seq_traj_features[seq_id].append(subj_features)
            seq_traj_czyxs[seq_id].append(subj_czyxs.long())
            seq_traj_batch_inds[seq_id].append(subj_batch_inds)
            seq_traj_valid_masks[seq_id].append(valid_mask)
            seq_masks[seq_id].append(seq_flag)
            seq_track_ids[seq_id].append(track_id*torch.ones(len(subj_features)))
    
    traj_batch_inds = []
    traj_czyxs = []
    traj_features = []
    traj_masks = []
    traj_seq_masks = []
    sample_seq_masks = []
    traj_track_ids = []

    for seq_id, subj_features_list in seq_traj_features.items():
        traj_features.append(torch.stack(subj_features_list))
        traj_czyxs.append(torch.stack(seq_traj_czyxs[seq_id]))
        traj_batch_inds.append(torch.stack(seq_traj_batch_inds[seq_id]))
        traj_masks.append(torch.stack(seq_traj_valid_masks[seq_id]).to(mesh_feature_maps.device))
        traj_seq_masks = traj_seq_masks + seq_masks[seq_id]
        sample_seq_masks.append(torch.Tensor(seq_masks[seq_id]).sum()>0)
        traj_track_ids.append(torch.stack(seq_track_ids[seq_id]).long())
    
    #traj_features = torch.cat(traj_features, 0)
    #traj_batch_inds = torch.cat(traj_batch_inds, 0)
    #traj_czyxs = torch.cat(traj_czyxs, 0)
    #traj_masks = torch.cat(traj_masks, 0).to(mesh_feature_maps.device)
    traj_seq_masks = torch.Tensor(traj_seq_masks).bool().reshape(-1)
    sample_seq_masks = torch.Tensor(sample_seq_masks).bool().reshape(-1)
    return traj_features, traj_czyxs, traj_batch_inds, traj_masks, traj_seq_masks, sample_seq_masks, traj_track_ids


def infilling_cams_of_low_quality_dets(normed_cams, seq_trackIDs, memory5D, seq_inherent_flags, direct_inherent=False, smooth_cam=True, pose_smooth_coef=1.):
    #print('memory5D', memory5D, seq_trackIDs,seq_inherent_flags)
    for ind, track_id in enumerate(seq_trackIDs):
        track_id = track_id.item()
        clip_cams = normed_cams[ind]
        infilling_clip_ids = torch.where(seq_inherent_flags[0][track_id])[0]
        good_clip_ids = torch.where(~seq_inherent_flags[0][track_id])[0]

        if smooth_cam:
            if 'cams' not in memory5D[0][track_id]:
                memory5D[0][track_id]['cams'] = OneEuroFilter(pose_smooth_coef, 0.7)
            
            if len(infilling_clip_ids) > 0:
                for clip_id in infilling_clip_ids:
                    fore_clips_ids = torch.where(~seq_inherent_flags[0][track_id][:clip_id])[0]
                    if len(fore_clips_ids) == 0:
                        if memory5D[0][track_id]['cams'].x_filter.prev_raw_value is not None:
                            normed_cams[ind,clip_id] = memory5D[0][track_id]['cams'].x_filter.prev_raw_value
                        continue
                    after_clips_ids = torch.where(~seq_inherent_flags[0][track_id][clip_id:])[0]
                    if len(after_clips_ids) == 0:
                        normed_cams[ind,clip_id] = clip_cams[good_clip_ids[-1]]
                        continue
                    valid_fore_ind = fore_clips_ids[-1]
                    valid_after_ind = after_clips_ids[0] + clip_id
                    normed_cams[ind,clip_id] = (valid_after_ind - clip_id) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_fore_ind] + \
                        (clip_id - valid_fore_ind) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_after_ind]
            
            for clip_id in range(len(clip_cams)):
                normed_cams[ind,clip_id] = memory5D[0][track_id]['cams'].process(clip_cams[clip_id])

        else:
            if 'cams' not in memory5D[0][track_id]:
                memory5D[0][track_id]['cams'] = clip_cams[good_clip_ids[0]] if len(good_clip_ids)>0 else None
            
            if direct_inherent:
                # simply inherent the previous cam positions.
                for clip_id in range(normed_cams.shape[1]):
                    if seq_inherent_flags[0][track_id][clip_id] and memory5D[0][track_id]['cams'] is not None:
                        # print('infilling cam at', ind, track_id, clip_id)
                        normed_cams[ind,clip_id] = memory5D[0][track_id]['cams']
                    elif not seq_inherent_flags[0][track_id][clip_id]:
                        memory5D[0][track_id]['cams'] = normed_cams[ind,clip_id]
            else:
                if len(infilling_clip_ids) > 0:
                    for clip_id in infilling_clip_ids:
                        fore_clips_ids = torch.where(~seq_inherent_flags[0][track_id][:clip_id])[0]
                        if len(fore_clips_ids) == 0:
                            if memory5D[0][track_id]['cams'] is not None:
                                normed_cams[ind,clip_id] = memory5D[0][track_id]['cams']
                            continue
                        after_clips_ids = torch.where(~seq_inherent_flags[0][track_id][clip_id:])[0]
                        if len(after_clips_ids) == 0:
                            normed_cams[ind,clip_id] = clip_cams[good_clip_ids[-1]]
                            continue
                        valid_fore_ind = fore_clips_ids[-1]
                        valid_after_ind = after_clips_ids[0] + clip_id
                        normed_cams[ind,clip_id] = (valid_after_ind - clip_id) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_fore_ind] + \
                            (clip_id - valid_fore_ind) / (valid_after_ind - valid_fore_ind) * clip_cams[valid_after_ind]
                        #print(ind, clip_id.item(), valid_fore_ind.item(), valid_after_ind.item(), normed_cams[ind,clip_id])
                if len(good_clip_ids)>0:
                    memory5D[0][track_id]['cams'] = clip_cams[good_clip_ids[-1]]


    return normed_cams, memory5D

def prepare_complete_trajectory_features_withmemory(self, seq_tracking_results, mesh_feature_maps, memory5D=None, det_conf_thresh=0.2, inherent_previous=True):
    if memory5D is None:
        memory5D = {seq_id: None for seq_id in seq_tracking_results}
    seq_traj_features = {}
    seq_traj_czyxs = {}
    seq_traj_batch_inds = {}
    seq_traj_valid_masks = {}
    seq_masks = {}
    seq_track_ids = {}
    seq_inherent_flags = {}
    for seq_id, (center_traj3Ds, batch_inds, seq_flags, track_quality) in seq_tracking_results.items():
        seq_traj_features[seq_id] = []
        seq_traj_czyxs[seq_id] = []
        seq_traj_batch_inds[seq_id] = []
        seq_traj_valid_masks[seq_id] = []
        seq_masks[seq_id] = []
        seq_track_ids[seq_id] = []
        seq_inherent_flags[seq_id] = {}

        if memory5D[seq_id] is None:
            memory5D[seq_id] = {track_id:{'feature':None, 'inherent_flag':{}} for track_id in center_traj3Ds}

        for track_id in center_traj3Ds:
            subj_center3D_traj = center_traj3Ds[track_id]
            subj_batch_inds = batch_inds[track_id]
            seq_flag = seq_flags[track_id]
            det_confs = track_quality[track_id][:,0]
            tracked_states = track_quality[track_id][:,1]
            valid_mask = torch.ones(len(subj_batch_inds)).bool()

            if track_id not in memory5D[seq_id]:
                memory5D[seq_id][track_id] = {'feature':None, 'inherent_flag':{}}

            for sbid in range(len(subj_batch_inds)):
                subj_batch_ind = subj_batch_inds[sbid]
                if subj_batch_ind == -1:
                    valid_mask[sbid] = False
            
            subj_features = self.image_feature_sampling(mesh_feature_maps, subj_center3D_traj, subj_batch_inds)
            inherent_flags = torch.ones(len(subj_features)).bool()

            if inherent_previous:
                for clip_id in range(len(subj_features)):
                    inherent_flag = True
                    if valid_mask[clip_id]:
                        # if this is a high-quality detection and the tracker successfully tracked the corresponding subject, 
                        # then sampling feature from estimated feature map for mesh regression.
                        if det_confs[clip_id]>det_conf_thresh and tracked_states[clip_id]>0.99:
                            memory5D[seq_id][track_id]['feature'] = subj_features[clip_id]
                            inherent_flag = False
                        
                        # if this is a low-quality detection, 
                        # then inherenting feature from the memory5D for this subject.
                        elif det_confs[clip_id]<=det_conf_thresh and memory5D[seq_id][track_id]['feature'] is not None:
                            subj_features[clip_id] = memory5D[seq_id][track_id]['feature']
                        
                        # if the tracker failed to find the right detection for this subject, 
                        # then inherenting feature from the memory5D for this subject.
                        elif tracked_states[clip_id]<0.99 and memory5D[seq_id][track_id]['feature'] is not None:
                            subj_features[clip_id] = memory5D[seq_id][track_id]['feature']
                    memory5D[seq_id][track_id]['inherent_flag'][clip_id] = inherent_flag
                    inherent_flags[clip_id] = inherent_flag
            
            subj_czyxs = subj_center3D_traj
            seq_traj_features[seq_id].append(subj_features)
            seq_traj_czyxs[seq_id].append(subj_czyxs.long())
            seq_traj_batch_inds[seq_id].append(subj_batch_inds)
            seq_traj_valid_masks[seq_id].append(valid_mask)
            seq_masks[seq_id].append(seq_flag)
            seq_track_ids[seq_id].append(track_id*torch.ones(len(subj_features)))
            seq_inherent_flags[seq_id][track_id] = inherent_flags
    
    traj_batch_inds = []
    traj_czyxs = []
    traj_features = []
    traj_masks = []
    traj_seq_masks = []
    sample_seq_masks = []
    traj_track_ids = []

    for seq_id, subj_features_list in seq_traj_features.items():
        traj_features.append(torch.stack(subj_features_list))
        traj_czyxs.append(torch.stack(seq_traj_czyxs[seq_id]))
        traj_batch_inds.append(torch.stack(seq_traj_batch_inds[seq_id]))
        traj_masks.append(torch.stack(seq_traj_valid_masks[seq_id]).to(mesh_feature_maps.device))
        traj_seq_masks = traj_seq_masks + seq_masks[seq_id]
        sample_seq_masks.append(torch.Tensor(seq_masks[seq_id]).sum()>0)
        traj_track_ids.append(torch.stack(seq_track_ids[seq_id]).long())
    
    #traj_features = torch.cat(traj_features, 0)
    #traj_batch_inds = torch.cat(traj_batch_inds, 0)
    #traj_czyxs = torch.cat(traj_czyxs, 0)
    #traj_masks = torch.cat(traj_masks, 0).to(mesh_feature_maps.device)
    traj_seq_masks = torch.Tensor(traj_seq_masks).bool().reshape(-1)
    sample_seq_masks = torch.Tensor(sample_seq_masks).bool().reshape(-1)
    return traj_features, traj_czyxs, traj_batch_inds, traj_masks, traj_seq_masks, sample_seq_masks, traj_track_ids, seq_inherent_flags, memory5D

def get_org_batch_inds(trans_tracked, batch_ids, org_trans):
    batch_inds = torch.zeros(len(trans_tracked))
    matched_inds = torch.zeros(len(trans_tracked)).long()
    for ind in range(len(trans_tracked)):
        matched_ind = torch.argmin(torch.norm(org_trans - trans_tracked[ind][None], p=2, dim=-1))
        matched_inds[ind] = matched_ind
        batch_inds[ind] = batch_ids[matched_ind]
    return matched_inds, batch_inds

def parse_tracking_ids(seq_tracking_ids,pred_batch_ids, clip_length):
    # seq_tracking_ids is dict {seq_id: Tensor N x 4, (z,y,x, trackID)}
    all_track_results = torch.cat([v[0] for k, v in seq_tracking_ids.items()],0)
    all_track_ids = torch.unique(all_track_results[:,3].long())
    seq_tracking_results = {tid.item(): torch.zeros(clip_length,3).to(all_track_results.device).long() for tid in all_track_ids} 
    seq_tracking_batch_inds = {tid.item(): torch.ones(clip_length).to(all_track_results.device).long()*-1 for tid in all_track_ids} 
    seq_tracking_quality = {tid.item(): torch.zeros(clip_length,2).to(all_track_results.device).float() for tid in all_track_ids} 
    seq_masks = {tid.item(): True for tid in all_track_ids}
    valid_tracked_num = {tid.item(): 0 for tid in all_track_ids}
    
    frame_ids = sorted(seq_tracking_ids.keys())
    subject_traj_dict = {}
    for frame_id in frame_ids:
        trans_track_ids, batch_ids, org_trans, czyxs = seq_tracking_ids[frame_id]
        trans_tracked = trans_track_ids[:,:3]

        detected_track_ids = trans_track_ids[:,3]
        tracked_det_conf = trans_track_ids[:,4]
        tracked_flag = trans_track_ids[:,5]
        tracked_czyx = trans_track_ids[:,6:9]

        for track_id in seq_tracking_results:
            this_subject_mask = detected_track_ids==track_id
            if this_subject_mask.sum() == 0:
                continue
            seq_tracking_results[track_id][frame_id] = tracked_czyx[this_subject_mask]
            seq_tracking_quality[track_id][frame_id, 0] = tracked_det_conf[this_subject_mask]
            seq_tracking_quality[track_id][frame_id, 1] = tracked_flag[this_subject_mask]
            seq_tracking_batch_inds[track_id][frame_id] = batch_ids[0]
            valid_tracked_num[track_id] += 1

    # To drop the detection shows up less than 4 times in long sequence. 
    for tid, valid_num in valid_tracked_num.items():
        if valid_num < min(len(frame_ids), 6):
            del seq_tracking_results[tid], seq_tracking_batch_inds[tid], seq_masks[tid], seq_tracking_quality[tid]
    
    return seq_tracking_results, seq_tracking_batch_inds, seq_masks, seq_tracking_quality

def MemoryUnit(object):
    def __init__(self, tracked_ID, tracked_3Dposition, update_feature_thresh=0.16):
        self.ID = tracked_ID
        self.position3D = tracked_3Dposition

        self.last_det_confidence = 0.
        self.feature = None
        self.update_feature_thresh=update_feature_thresh
    
    def update_last_det_confidence(self, conf):
        self.last_det_confidence = conf
    
    def update_feature(self, new_feature):
        if self.last_det_confidence > self.update_feature_thresh:
            self.feature = new_feature
            return self.feature, True
        else:
            return self.feature, False


def perform_tracking(motion_offsets, pred_batch_ids, pred_cams, pred_czyxs, top_score, \
            batch_num=1, clip_length=8, seq_cfgs=None, tracker=None, debug=False):
    # TODO: add seq_confinues flag to determine whether the input is a long | continuous sequence to track, instead of short clips.
    pred_seq_ids = torch.div(pred_batch_ids, clip_length, rounding_mode='floor')
    clip_ids = pred_batch_ids % clip_length

    seq_tracking_results = {}
    for seq_id in torch.unique(pred_seq_ids):
        seq_clip_ids = torch.sort(torch.unique(clip_ids[pred_seq_ids==seq_id])).values
        if tracker is None:
            # set high det_thresh to avoid affected by low-quality detection / localization.
            # match_thresh 1, set to .5 improving the ability to hold the unchanged state under long-term occlusion. But will make the fast movement stucking...
            tracker = Tracker(det_thresh=seq_cfgs['tracker_det_thresh'], first_frame_det_thresh=seq_cfgs['first_frame_det_thresh'], \
                            accept_new_dets=seq_cfgs['accept_new_dets'], new_subject_det_thresh=seq_cfgs['new_subject_det_thresh'], \
                            track_buffer=seq_cfgs['time2forget'], match_thresh=seq_cfgs['tracker_match_thresh'], axis_times=seq_cfgs['axis_times'],frame_rate=30)
        seq_tracking_ids = {}
        for clip_id in seq_clip_ids:
            frame_inds = torch.where(torch.logical_and(pred_seq_ids==seq_id, clip_ids==clip_id))[0]
            if len(frame_inds) == 0:
                continue
            
            large_object_mask = filter_out_small_objects(pred_cams[frame_inds], thresh=seq_cfgs['large_object_thresh'])
            #print(clip_id, pred_cams[frame_inds],'filtered', pred_cams[frame_inds][large_object_mask])
            frame_inds = frame_inds[large_object_mask]
            if len(frame_inds) == 0:
                continue

            sdd_mask = suppress_duplicate_dets(pred_cams[frame_inds], top_score[frame_inds], thresh=seq_cfgs['suppress_duplicate_thresh'])
            # if sdd_mask.sum()!=len(sdd_mask):
            #     camera_cam_info = torch.cat([pred_cams[frame_inds], top_score[frame_inds].unsqueeze(1)], -1)
            #     print('duplicate dets before and after', camera_cam_info, camera_cam_info[~sdd_mask])
            frame_inds = frame_inds[sdd_mask]
            if len(frame_inds) == 0:
                continue
            
            normed_cam = pred_cams[frame_inds]
            scores = top_score[frame_inds]
            l2c_motion_offset = motion_offsets[frame_inds].float()
            # limit the norm of motion offset to avoid sudden peak.
            large_offset_mask = torch.norm(l2c_motion_offset,p=2,dim=-1)>seq_cfgs['motion_offset3D_norm_limit']
            if large_offset_mask.sum()>0:
                if debug:
                    print('weird l2c_motion_offset', l2c_motion_offset[large_offset_mask])
                l2c_motion_offset[large_offset_mask] = 0
                #l2c_motion_offset[large_offset_mask] = l2c_motion_offset[large_offset_mask] / \
                #    torch.norm(l2c_motion_offset[large_offset_mask],p=2,dim=-1).unsqueeze(-1) * seq_cfgs['motion_offset3D_norm_limit']
            
            batch_ids = pred_batch_ids[frame_inds]
            czyxs = pred_czyxs[frame_inds].detach().cpu().numpy()
            
            trans3D_current_frame = denormalize_cam_params_to_trans(normed_cam, positive_constrain=False)
            trans3D_last_frame = denormalize_cam_params_to_trans(normed_cam-l2c_motion_offset, positive_constrain=False)

            # TODO: use cam - offset作为上一帧关联目标，而不是只是输入offset，在tracker里算。
            trans_track_ids = tracker.update(trans3D_current_frame.detach().cpu().numpy(), \
                    scores.detach().cpu().numpy(), trans3D_last_frame.detach().cpu().numpy(), czyxs, \
                    debug=debug, never_forget=False, tracking_target_max_num=seq_cfgs['subject_num'],\
                    using_motion_offsets=True)

            if len(trans_track_ids) > 0:
                seq_tracking_ids[clip_id.item()] = [torch.from_numpy(trans_track_ids).to(pred_cams.device), batch_ids, trans3D_current_frame, czyxs]
        if len(seq_tracking_ids)>0:
            seq_tracking_results[seq_id.item()] = parse_tracking_ids(seq_tracking_ids, pred_batch_ids, clip_length)
    return seq_tracking_results, tracker

def suppress_duplicate_dets(cams, det_confs, thresh=0.1):
    sdd_mask = torch.ones(len(cams)).bool()
    if len(cams)>1:
        del_index = []
        #trans = denormalize_cam_params_to_trans(cams, positive_constrain=False)
        for ind in range(len(cams)):
            dists = torch.sort(torch.norm(cams - cams[[ind]], p=2, dim=-1), descending=False)
            dup_inds = dists.indices[dists.values < thresh][1:].tolist()
            if len(dup_inds)>0:
                dup_inds = dup_inds+[ind]
                # only keep the one with maximum conf
                dup_inds_inds = torch.sort(det_confs[dup_inds], descending=True).indices[1:].tolist()
                det_ind_with_nonmaximum_conf = [dup_inds[i] for i in dup_inds_inds]
                # print(dup_inds, det_confs[dup_inds], det_ind_with_nonmaximum_conf, dup_inds_inds)
                sdd_mask[det_ind_with_nonmaximum_conf] = False

    return sdd_mask

def filter_out_small_objects(cams, thresh=0.1):
    large_object_mask = cams[:,0] > thresh
    return large_object_mask