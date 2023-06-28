import torch
import numpy as np
from config import args
import copy

def match_trajectory_gts(traj3D_gts, traj2D_gts, traj_sids, subject_ids, batch_ids):
    subject_num = len(subject_ids)
    traj3D_gts_matched = torch.ones(
        subject_num, args().temp_clip_length, 3).float().to(traj3D_gts.device) * -2.
    traj2D_gts_matched = torch.ones(
        subject_num, args().temp_clip_length, 2).float().to(traj2D_gts.device) * -2.
    
    parallel_start_id = int(batch_ids[0].item())
    for ind, (sid, bid) in enumerate(zip(subject_ids, batch_ids)):
        if sid == -1:
            continue
        input_batch_ind = int(bid.item()) // args().temp_clip_length - parallel_start_id // args().temp_clip_length
        input_subject_ind = torch.where(traj_sids[input_batch_ind,:,0,3] == sid)[0]
        
        try:
            if len(input_subject_ind)>0:
                traj2D_gts_matched[ind] = traj2D_gts[input_batch_ind, input_subject_ind]
                traj3D_gts_matched[ind] = traj3D_gts[input_batch_ind, input_subject_ind]
        except:
            print('matching traj error:', ind, sid, input_batch_ind, input_subject_ind, traj2D_gts.shape)
    
    return traj3D_gts_matched, traj2D_gts_matched

def convert2BDsplit(tensor):
    """
    B batch size, N person number, T temp_clip_length, {} means might have this dimension for some input tensor but not all.
    Convert the input tensor from shape (B,...) to (BxT, ...), expanded [0,1,2,...] as [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, ...] if T=5.
    The shape of image feature maps is (BxT) x C x H x W
    In this way, we can use the balanced dataparallel to split data of shape (BxT) to multiple GPUs.
    """
    shape = list(tensor.shape) # B x N x T {x 2/3}
    shape[0] = shape[0] * args().temp_clip_length # (BxT) x N x T {x 2/3}

    repeat_times = [1 for _ in range(len(shape)+1)] # 1, 1, 1, 1{,1}
    repeat_times[1] = args().temp_clip_length # 1, T, 1, 1{,1}
    
    # B x N x T {x 2/3} -> B x 1 x N x T {x 2/3} -> B x T x N x T {x 2/3} -> (BxT) x N x T {x 2/3}
    return tensor.unsqueeze(1).repeat(*repeat_times).reshape(*shape)

def convertback2batch(tensor):
    return tensor[::args().temp_clip_length].contiguous()

def reorganize_trajectory_info(meta_data):
    for item in ['traj3D_gts', 'traj2D_gts', 'Tj_flag', 'traj_gt_ids']:
        if item in meta_data:
            meta_data[item] = convertback2batch(meta_data[item])
    return meta_data


def convert_centers_to_trajectory(person_centers, track_ids, cam_params, cam_masks, seq_inds=None, image_repeat=args().image_repeat_time):
    """ 
    Convert the 2D/3D body centers into trajectory format. 
    Please note that we need to memorize the sampled ID to re-organize the other GTs for supervision. 
    """
    #print(person_centers.shape, track_ids.shape, cam_params.shape, cam_masks.shape)
    batch_size, _, max_person_num, _ = person_centers.shape
    device = person_centers.device
    clip_length = args().temp_clip_length
    # max_person_num in clip could be larger than max_person_num of each frames
    max_person_num *= 2
    trajectory2D = torch.ones(batch_size, max_person_num, clip_length, 2).float().to(device) * -2.
    trajectory3D = torch.ones(batch_size, max_person_num, clip_length, 3).float().to(device) * -2.
    trajectory_gt_ids = torch.ones(batch_size, max_person_num, clip_length, 4).to(device).long() * -1. 
    Tj_flag = torch.zeros(batch_size, 2).bool().to(device)
    
    pc_masks = (person_centers!=-2).sum(-1)>0

    for bid in range(len(track_ids)):
        if seq_inds is not None:
            if not seq_inds[bid*clip_length, 3]:
                for ri in range(image_repeat):
                    half_mask = torch.zeros_like(pc_masks[bid]).bool()
                    half_mask[ri::image_repeat] = pc_masks[bid, ri::image_repeat]
                    track_ids[bid][half_mask] = torch.arange(half_mask.sum())

        unique_ids = torch.unique(track_ids[bid][track_ids[bid]!=-1.])
        for ind, sid in enumerate(unique_ids):
            if ind >= max_person_num:
                continue
            mask = track_ids[bid] == sid
            pc_mask = mask * pc_masks[bid]
            cam_mask = mask * cam_masks[bid]
            clip_valid_mask = pc_mask.sum(-1) > 0
            if (~clip_valid_mask).sum()>0:
                clip_inds = torch.ones(len(pc_mask), dtype=torch.long, device=device) * -1
                subj_inds = torch.ones(len(pc_mask), dtype=torch.long, device=device) * -1
                clip_inds[clip_valid_mask] = torch.where(pc_mask)[0]
                subj_inds[clip_valid_mask] = torch.where(pc_mask)[1]
            else:
                clip_inds, subj_inds = torch.where(pc_mask)

            ## last dim (batch, clip, subject, ID)
            trajectory_gt_ids[bid,ind,:,0] = bid
            trajectory_gt_ids[bid,ind,:,1] = clip_inds.long()
            trajectory_gt_ids[bid,ind,:,2] = subj_inds.long()
            trajectory_gt_ids[bid,ind,:,3] = sid
            
            trajectory2D[bid,ind,clip_valid_mask] = person_centers[bid][pc_mask]
            trajectory3D[bid,ind,cam_mask.sum(-1) > 0] = cam_params[bid][cam_mask]
            
        Tj_flag[bid, 0] = (trajectory2D[bid] != -2).sum() > 0
        Tj_flag[bid, 1] = (trajectory3D[bid] != -2).sum() > 0

    #if args().learn_image:
    #    image_batch_mask = ~seq_inds[::clip_length, 3].bool()
    #print(trajectory2D[-1][:16])
    trajectory_info = {'traj3D_gts': convert2BDsplit(trajectory3D), 'traj2D_gts': convert2BDsplit(trajectory2D), \
        'Tj_flag': convert2BDsplit(Tj_flag), 'traj_gt_ids': convert2BDsplit(trajectory_gt_ids)}
    return trajectory_info, track_ids.reshape(-1, args().max_supervise_num)

def full_body_bboxes2person_centers(full_body_bboxes):
    valid_mask = (full_body_bboxes!=-2).sum(-1) == 0
    person_centers = (full_body_bboxes[:,:,:2] + full_body_bboxes[:,:,2:]) / 2
    person_centers[~valid_mask] = -2.
    return person_centers

def ordered_organize_frame_outputs_to_clip(seq_inds, person_centers=None, track_ids=None, cam_params=None, cam_mask=None, full_body_bboxes=None, ):
    # temp_clip_length = args().temp_clip_length
    # if temp_clip_length % 2 == 1:
    #     span_len = (temp_clip_length-1)//2
    #     sampled_ids = np.arange(len(track_ids)//temp_clip_length) * temp_clip_length + span_len
    #     seq_sampling_index = np.array([np.arange(bi-span_len,bi+span_len+1) for bi in sampled_ids])
    # else:
    #     span_len = temp_clip_length//2
    #     sampled_ids = np.arange(len(track_ids)//temp_clip_length) * temp_clip_length + span_len
    #     seq_sampling_index = np.array([np.arange(bi-span_len,bi+span_len) for bi in sampled_ids])
    seq_sampling_index = seq_inds[:,2].reshape(-1, args().temp_clip_length).numpy()

    if full_body_bboxes is not None:
        person_centers = full_body_bboxes2person_centers(full_body_bboxes)
    
    person_centers_inputs, track_ids_inputs, cam_params_inputs, cam_mask_inputs = \
        [torch.stack([item[ids] for ids in seq_sampling_index], 0).contiguous(
        ) for item in [person_centers, track_ids, cam_params, cam_mask]]
    #print(person_centers_inputs[-1,:,:2], track_ids_inputs[-1,:,:2])
    trajectory_info, track_ids_flatten = convert_centers_to_trajectory(
        person_centers_inputs, track_ids_inputs, cam_params_inputs, cam_mask_inputs, seq_inds=seq_inds)
    return trajectory_info, track_ids_flatten

if __name__ == '__main__':
    test_temp_inputs_channel_convert()

"""

def temp_inputs_channel_convert(image_trajectory3D_inputs):
    image_trajectory3D_inputs = image_trajectory3D_inputs.transpose(2,1).transpose(3,2)
    b,c,d,n,h,w = image_trajectory3D_inputs.shape
    image_trajectory3D_inputs = image_trajectory3D_inputs.reshape(b,c,(d*n),h,w)
    return image_trajectory3D_inputs

def test_temp_inputs_channel_convert():
    image_trajectory3D_inputs = torch.arange(1*2*3*4*1*1).view(1,2,3,4,1,1) # 
    image_trajectory3D_inputs = temp_inputs_channel_convert(image_trajectory3D_inputs)
    image_trajectory3D_inputs = torch.arange(1*2*2*2*1*1).view(1,2,2,2,1,1)
    print(image_trajectory3D_inputs)
    image_trajectory3D_inputs = image_trajectory3D_inputs.repeat(1,1,1,1,2,2)
    print(image_trajectory3D_inputs)
    image_trajectory3D_inputs = temp_inputs_channel_convert(image_trajectory3D_inputs)
    print(image_trajectory3D_inputs)

def prepare_trajectory_candidates(pred_batch_ids, top_score, cams_preds, cam_czyxs, image_feature_maps, sampled_ids, seq_sampling_index, match_thresh=0.2, match_num=2):
    '''
    以中心帧为核心想两边展开，首先只关心高质信度匹配的结果，然后再处理剩下的轨迹，处理方式就两种，一种是在7帧里复制出现的位置
    '''
    top_score = top_score.cpu()
    trajectory_dict = {}
    temp_clip_length = args().temp_clip_length
    spawn = temp_clip_length // 2
    feature_dim = 3+1+32

    for bid, (center_sid, sids) in enumerate(zip(sampled_ids, seq_sampling_index)):
        clip_dets = [[] for i in range(temp_clip_length)]
        for fid, sid in enumerate(sids):
            sample_mask = pred_batch_ids == sid
            if sample_mask.sum()>0:
                clip_dets[fid] = [cams_preds[sample_mask], cam_czyxs[sample_mask], top_score[sample_mask]]

        if len(clip_dets[spawn]) == 0:
            continue
        trajectory_dict[bid] = []
        
        dets, czyxs, confs = clip_dets[spawn]
        for det, czyx, conf in zip(dets, czyxs, confs):
            trajs = [torch.zeros(temp_clip_length, feature_dim)]
            trajs[0][spawn] = torch.cat([det, conf[None], image_feature_maps[center_sid,:,czyx[1],czyx[2]]])

            for spwan_ind in range(1,spawn+1):
                tid = spawn - spwan_ind
                if len(clip_dets[tid])>0:
                    cand_dets, cand_czyxs, cand_confs = clip_dets[tid]
                    dets_cand = [traj[tid+1, :3] for traj in trajs]
                    for trajID, det in enumerate(dets_cand):
                        dists = torch.norm(cand_dets - det[None], p=2, dim=-1)
                        dists = dists[dists < match_thresh]
                        close_inds = torch.topk(dists, min(len(dists), match_num), largest=False, sorted=True).indices
                        if len(close_inds) > 0:
                            cand_info = (cand_dets[close_inds], cand_czyxs[close_inds], cand_confs[close_inds])
                            for mid, (on_det, on_czyx, on_conf) in enumerate(zip(*cand_info)):
                                if mid == 0:
                                    trajs[trajID][tid] = torch.cat([on_det, on_conf[None], image_feature_maps[sids[tid],:,on_czyx[1],on_czyx[2]]])
                                else:
                                    another_trajs = copy.deepcopy([trajs[trajID]])
                                    another_trajs[0][tid] = torch.cat([on_det, on_conf[None], image_feature_maps[sids[tid],:,on_czyx[1],on_czyx[2]]])
                                    trajs += another_trajs
                else:
                    for traj in trajs:
                        traj[tid] = traj[tid+1]
            
            for spwan_ind in range(1,spawn+1):
                tid = spawn + spwan_ind
                if len(clip_dets[tid])>0:
                    cand_dets, cand_czyxs, cand_confs = clip_dets[tid]
                    dets_cand = [traj[tid-1, :3] for traj in trajs]
                    for trajID, det in enumerate(dets_cand):
                        dists = torch.norm(cand_dets - det[None], p=2, dim=-1)
                        dists = dists[dists < match_thresh]
                        close_inds = torch.topk(dists, min(len(dists), match_num), largest=False, sorted=True).indices
                        if len(close_inds) > 0:
                            cand_info = (cand_dets[close_inds], cand_czyxs[close_inds], cand_confs[close_inds])
                            for mid, (on_det, on_czyx, on_conf) in enumerate(zip(*cand_info)):
                                if mid == 0:
                                    trajs[trajID][tid] = torch.cat([on_det, on_conf[None], image_feature_maps[sids[tid],:,on_czyx[1],on_czyx[2]]])
                                else:
                                    another_trajs = copy.deepcopy([trajs[trajID]])
                                    another_trajs[0][tid] = torch.cat([on_det, on_conf[None], image_feature_maps[sids[tid],:,on_czyx[1],on_czyx[2]]])
                                    trajs += another_trajs
                else:
                    for traj in trajs:
                        traj[tid] = traj[tid-1]
            
            '''
            on_det, on_czyx, on_conf = det, czyx, conf
            for spwan_ind in range(1,spawn+1):
                tid = spawn + spwan_ind
                if len(clip_dets[tid])>0:
                    cand_dets, cand_czyxs, cand_confs = clip_dets[tid]
                    dists = torch.norm(cand_dets-det[None], p=2, dim=-1)
                    close_ind = torch.argmin(dists)
                    if dists[close_ind]<match_thresh:
                        on_det, on_czyx, on_conf = cand_dets[close_ind], cand_czyxs[close_ind], cand_confs[close_ind]
                traj[tid] = torch.cat([on_det, on_conf[None], image_feature_maps[sids[tid],:,on_czyx[1],on_czyx[2]]])
            trajectory_dict[bid] += trajs
            '''
            trajectory_dict[bid] += trajs
    #print(trajectory_dict[3])
    max_traj_num = max([len(traj_list) for traj_list in trajectory_dict.values()])
    # B x N x 924
    trajectory_candidates = torch.zeros(len(sampled_ids),max_traj_num,temp_clip_length*feature_dim)
    traj_padding_masks = torch.ones(len(sampled_ids),max_traj_num).bool()
    for bid, traj_list in trajectory_dict.items():
        traj_padding_masks[bid, :len(traj_list)] = False
        trajectory_candidates[bid, :len(traj_list)] = torch.stack(traj_list, 0).reshape(len(traj_list), -1)
    return trajectory_candidates, traj_padding_masks




def convert_centers_to_trajectory_center_frame(person_centers, track_ids, pc_masks, cam_params, cam_masks):
    batch_size, _, max_person_num, _ = person_centers.shape
    trajectory2D = torch.ones(
        batch_size, max_person_num, args().temp_clip_length, 2).float() * -2.
    trajectory3D = torch.ones(
        batch_size, max_person_num, args().temp_clip_length, 3).float() * -2.
    Tj_flag = torch.zeros(batch_size, 2).bool()
    middle_id = args().temp_clip_length//2
    for bid, subject_ids in enumerate(track_ids):
        # only take the subjects shown up in the middle frame 3-th of 7 frames
        clip_subject_mask = torch.where(subject_ids[middle_id] != -1)[0]
        for sid in clip_subject_mask:
            mask = subject_ids == subject_ids[middle_id, sid]
            pc_mask = mask * pc_masks[bid]
            cam_mask = mask * cam_masks[bid]
            trajectory2D[bid][sid][pc_mask.sum(-1)
                                   > 0] = person_centers[bid][pc_mask]
            trajectory3D[bid][sid][cam_mask.sum(-1)
                                   > 0] = cam_params[bid][cam_mask]
        Tj_flag[bid, 0] = (trajectory2D[bid] != -2).sum() > 0
        Tj_flag[bid, 1] = (trajectory3D[bid] != -2).sum() > 0
    # track_ids meets bugs about repeated last ID
    #print('check track_ids:', len(track_ids), track_ids[0])
    device = person_centers.device
    print(trajectory2D.shape, trajectory3D.shape)
    print(trajectory2D, trajectory3D)
    trajectory_info = {'trajectory3D_gts': trajectory3D.to(device), 'trajectory2D_gts': trajectory2D.to(
        device), 'Tj_flag': Tj_flag.to(device), 'track_ids': track_ids}
    return trajectory_info


def ordered_organize_frame_outputs_to_seq(image_outputs, seq_info, person_centers=None, track_ids=None, pc_mask=None, cam_params=None, cam_mask=None, with_traj_info=True):
    output_device = image_outputs['image_feature_maps'].device
    batch_size = len(image_outputs['image_feature_maps'])
    cams_preds = image_outputs['cams_preds'].detach().cpu()
    cam_czyx = image_outputs['cam_czyx'].detach().cpu().long()
    # 32 x 128 x 128
    image_featrure_maps = image_outputs['image_feature_maps'].detach().cpu()

    seq_info = np.array(seq_info.cpu(),dtype=np.int32)
    span_len = (args().temp_clip_length-1)//2
    sampled_ids = np.arange(span_len, batch_size-span_len)
    seq_sampling_index = np.array([np.arange(bi-span_len,bi+span_len+1) for bi in sampled_ids])

    trajectory_candidates, traj_padding_masks = \
        prepare_trajectory_candidates(image_outputs['pred_batch_ids'], image_outputs['top_score'], cams_preds, cam_czyx, image_featrure_maps, sampled_ids, seq_sampling_index)
    
    temp_inputs = {'trajectory_candidates': trajectory_candidates.to(output_device),\
                    'traj_padding_masks': traj_padding_masks.to(output_device)}
    
    if with_traj_info:
        person_centers_inputs, track_ids_inputs, pc_mask_inputs, cam_params_inputs, cam_mask_inputs = \
            [torch.stack([item[ids] for ids in seq_sampling_index], 0).contiguous() for item in [person_centers, track_ids, pc_mask, cam_params, cam_mask]]
        
        trajectory_info = convert_centers_to_trajectory_center_frame(
            person_centers_inputs, track_ids_inputs, pc_mask_inputs, cam_params_inputs, cam_mask_inputs)
        return temp_inputs, sampled_ids, trajectory_info
    return temp_inputs, sampled_ids


def ordered_organize_frame_outputs_to_seq_conv3D(image_outputs, seq_info, person_centers=None, track_ids=None, pc_mask=None, cam_params=None, cam_mask=None, with_traj_info=True):
    output_device = image_outputs['mesh_feature_map'].device
    Mesh_featrure_maps = image_outputs['mesh_feature_map'].detach().cpu()
    image_trajectory3D = torch.cat([image_outputs['center_map_3d'].detach().cpu().unsqueeze(1), image_outputs['cam_maps_3d'].detach().cpu()], 1)
    seq_info = np.array(seq_info.cpu(),dtype=np.int32)
    span_len = (args().temp_clip_length-1)//2

    update_items = [image_trajectory3D, Mesh_featrure_maps]
    if with_traj_info:
        update_items += [person_centers, track_ids, pc_mask, cam_params, cam_mask]
    
    sampled_ids = np.arange(span_len, len(image_trajectory3D)-span_len)
    seq_sampling_index = np.array([np.arange(bi-span_len,bi+span_len+1) for bi in sampled_ids])

    image_trajectory3D_inputs, Mesh_featrure_maps_inputs = [torch.stack([item[ids] for ids in seq_sampling_index], 0).contiguous() for item in update_items[:2]]
    image_trajectory3D_inputs = temp_inputs_channel_convert(image_trajectory3D_inputs)
    
    temp_inputs = {'image_trajectory3D': image_trajectory3D_inputs.to(output_device), \
                    'mesh_feature_maps': Mesh_featrure_maps_inputs.to(output_device)}
    
    if with_traj_info:
        person_centers_inputs, track_ids_inputs, pc_mask_inputs, cam_params_inputs, cam_mask_inputs = \
            [torch.stack([item[ids] for ids in seq_sampling_index], 0).contiguous() for item in update_items[2:]]
        
        trajectory_info = convert_centers_to_trajectory_center_frame(
            person_centers_inputs, track_ids_inputs, pc_mask_inputs, cam_params_inputs, cam_mask_inputs)
        return temp_inputs, sampled_ids, trajectory_info
    return temp_inputs, sampled_ids


def ordered_organize_frame_outputs_to_clip_old(image_outputs, seq_info, person_centers=None, track_ids=None, pc_mask=None, cam_params=None, cam_mask=None, with_traj_info=True):
    batch_size = len(image_outputs['image_feature_maps'])
    seq_info = np.array(seq_info.cpu(),dtype=np.int32)
    temp_clip_length = args().temp_clip_length
    span_len = (temp_clip_length-1)//2
    sampled_ids = np.arange(batch_size//temp_clip_length) * temp_clip_length + span_len
    seq_sampling_index = np.array([np.arange(bi-span_len,bi+span_len+1) for bi in sampled_ids])
    temp_inputs = {'image_feature_maps': torch.cat(
        [image_outputs['image_feature_maps'][ids] for ids in seq_sampling_index], 0).contiguous().detach()}

    trajectory_info = None
    if with_traj_info:
        person_centers_inputs, track_ids_inputs, pc_mask_inputs, cam_params_inputs, cam_mask_inputs = \
            [torch.stack([item[ids] for ids in seq_sampling_index], 0).contiguous() for item in [person_centers, track_ids, pc_mask, cam_params, cam_mask]]
        
        trajectory_info = convert_centers_to_trajectory(person_centers_inputs, track_ids_inputs, pc_mask_inputs, cam_params_inputs, cam_mask_inputs)
    return temp_inputs, seq_sampling_index.reshape(-1), trajectory_info

    output_device = image_outputs['image_feature_maps'].device
    cams_preds = image_outputs['cams_preds'].detach().cpu()
    cam_czyx = image_outputs['cam_czyx'].detach().cpu().long()
    image_featrure_maps = image_outputs['image_feature_maps'].detach().cpu()

    trajectory_candidates, traj_padding_masks = \
        prepare_trajectory_candidates(image_outputs['pred_batch_ids'], image_outputs['top_score'], cams_preds, cam_czyx, image_featrure_maps, sampled_ids, seq_sampling_index)
    
    temp_inputs = {'trajectory_candidates': trajectory_candidates.to(output_device),\
                    'traj_padding_masks': traj_padding_masks.to(output_device)}
    
    if with_traj_info:
        person_centers_inputs, track_ids_inputs, pc_mask_inputs, cam_params_inputs, cam_mask_inputs = \
            [torch.stack([item[ids] for ids in seq_sampling_index], 0).contiguous() for item in [person_centers, track_ids, pc_mask, cam_params, cam_mask]]
        
        trajectory_info = convert_centers_to_trajectory(person_centers_inputs, track_ids_inputs, pc_mask_inputs, cam_params_inputs, cam_mask_inputs)
        return temp_inputs, sampled_ids, trajectory_info
    return temp_inputs, sampled_ids



    #seq_ids, frame_ids, seq_end_flag = seq_info[:,0], seq_info[:,1], seq_info[:,2]
    #frame_seq_info = [[ds_name, seqid, fid] for ds_name, seqid, fid in zip(ds_org,seq_ids,frame_ids)] # '{}-{}-{}'.format(ds_name, seqid, fid)
    # frame_seq_info_str = ['{}-{}-{}'.format(ds_name, seqid, fid) for ds_name, seqid, fid in frame_seq_info]


        if mode == 'ordered':
            print('starting making up frames', '-'*20)
            # append the left span of the first frame
            for ci in range(span_len):
                frame_name = '{}-{}-{}'.format(frame_seq_info[0][0], frame_seq_info[0][1], frame_seq_info[0][2]-ci)
                if frame_name in self.seq_cacher:
                    update_items = [torch.cat([self.seq_cacher[frame_name][ui_id][None], item], 0) for ui_id, item in enumerate(update_items)]
                    frame_seq_info = [[frame_seq_info[0][0], frame_seq_info[0][1], frame_seq_info[0][2]-ci]] + frame_seq_info
                else:
                    update_items = [torch.cat([item[[0]], item], 0) for ui_id, item in enumerate(update_items)]
                    frame_seq_info = [frame_seq_info[0]] + frame_seq_info
                print('Appending the first frame:', [item.shape for item in update_items], frame_seq_info)
            
            # should keep the batch size constent! Better to sample the span_len frames of last sequence via batch_sampler
            for ci in range(span_len, self.temp_clip_length):
                frame_name = '{}-{}-{}'.format(frame_seq_info[0][0], frame_seq_info[0][1], frame_seq_info[0][2]-ci)
                if frame_name in self.seq_cacher:
                    update_items = [torch.cat([self.seq_cacher[frame_name][ui_id][None], item], 0) for ui_id, item in enumerate(update_items)]
                    frame_seq_info = [[frame_seq_info[0][0], frame_seq_info[0][1], frame_seq_info[0][2]-ci]] + frame_seq_info
                print('Appending the frame of last sequence:', [item.shape for item in update_items], frame_seq_info)
            
            #if seq_end_flag.sum()>0:
            #    update_items = [torch.cat([item, item[[-1]].repeat(span_len)], 0) for ui_id, item in enumerate(update_items)]
            #    frame_seq_info = frame_seq_info + [frame_seq_info[-1]]*span_len
            #    print('making up the last frame:', frame_seq_info)
            print('Ending up making up frames', len(image_trajectory3D), '-'*20)
        

if mode == 'ordered':
            # save the end of last clip for prediciton with complete info.
            self.seq_cacher = {'{}-{}-{}'.format(*frame_seq_info[ci]): \
                [item[ci] for item in update_items] for ci in range(-self.temp_clip_length,0)}
"""
