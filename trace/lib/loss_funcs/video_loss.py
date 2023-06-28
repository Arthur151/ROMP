from numpy import result_type
import torch
import numpy as np
import itertools

from config import args
from utils.cam_utils import denormalize_cam_params_to_trans
from utils.projection import perspective_projection, perspective_projection_withfovs
from loss_funcs.keypoints_loss import batch_kp_2d_l2_loss
from utils.rotation_transform import angle_axis_to_quaternion, quaternion_to_rotation_matrix
from utils.quaternion_operations import quaternion_difference

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c

def clip_frame_pairs_indes(N,device):
    # index of pairs of frames in each sequence, to calculate the loss between them 
    first_index, second_index = torch.meshgrid(torch.arange(N), torch.arange(N))
    first_index = first_index.reshape(-1).to(device)
    second_index = second_index.reshape(-1).to(device)
    # remove the index pairs that points to the same frame
    k = first_index != second_index
    first_index = first_index[k]
    second_index = second_index[k]
    return first_index, second_index

def quaternion_loss(predictions, targets):
    """
    Computes the quaternion loss between predicted and target quaternions.
    Args:
        predictions (torch.Tensor): predicted quaternions of shape (batch_size, 4)
        targets (torch.Tensor): target quaternions of shape (batch_size, 4)
    Returns:
        quaternion_loss (torch.Tensor): quaternion loss value
    """
    # Normalize the predictions and targets
    predictions_norm = predictions / torch.norm(predictions, dim=1, keepdim=True)
    targets_norm = targets / torch.norm(targets, dim=1, keepdim=True)
    
    # Compute the dot product between the normalized predictions and targets
    dot_product = torch.sum(predictions_norm * targets_norm, dim=1)
    
    # Compute the angle between the predictions and targets
    angle = 2 * torch.acos(torch.clamp(dot_product, min=-1, max=1))
    
    # Compute the quaternion loss
    quaternion_loss = torch.mean(angle)
    
    return quaternion_loss

    # Calculate the difference in rotation matrices
    # diff_rot = torch.matmul(quaternion_to_rotation_matrix(delta_pred), quaternion_to_rotation_matrix(delta_gt).transpose(1, 2)) - torch.eye(3).unsqueeze(0)

    # # Convert rotation matrix to axis-angle representation
    # diff_aa = rotation_matrix_to_axis_angle(diff_rot)

    # # Calculate the loss as the L2 norm of the axis-angle difference
    # loss = torch.norm(diff_aa, p=2)

def _calc_world_gros_loss_(preds, gts, vmasks, sequence_inds):
    # the world trans is supposed to be in shape (B, N, 3), B - batch size, N - temporal length (frame number in a video clip)
    loss = []
    device = preds.device
    gts, vmasks = gts.to(device), vmasks.to(device)
    for seq_inds in sequence_inds:
        pred, gt, vmask = preds[seq_inds], gts[seq_inds], vmasks[seq_inds]
        if vmask.sum()==0:
            continue

        N = pred.shape[0]
        first_index, second_index = clip_frame_pairs_indes(N, device)
        
        quaternion_pred = angle_axis_to_quaternion(pred)
        quaternion_gt = angle_axis_to_quaternion(gt)

        # refer to DPVO to 
        # inv 和 log 都是 lietorch的操作，转换成SE3，trans3+rots3
        # delta_pred = (pred[first_index]).inv() * pred[second_index]
        # delta_gt = (gt[first_index]).inv() * gt[second_index]
        # error = (delta_pred * delta_gt.inv()).log()
        # rots_error = error[...,3:6].norm(dim=-1)

        delta_pred = quaternion_difference(quaternion_pred[first_index], quaternion_pred[second_index])
        delta_gt = quaternion_difference(quaternion_gt[first_index], quaternion_gt[second_index])

        error = quaternion_loss(delta_pred, delta_gt)
        #error = torch.norm(delta_pred - delta_gt, dim=-1).mean()
        loss.append(error)
    loss = torch.stack(loss) if len(loss)>0 else torch.zeros(1,device=device)
    return loss

def _calc_world_trans_loss_(preds, gts, vmasks, sequence_inds):
    # the world trans is supposed to be in shape (B, N, 3), B - batch size, N - temporal length (frame number in a video clip)
    loss = []
    device = preds.device
    gts, vmasks = gts.to(device), vmasks.to(device)
    for seq_inds in sequence_inds:
        pred, gt, vmask = preds[seq_inds], gts[seq_inds], vmasks[seq_inds]
        if vmask.sum()==0:
            continue

        N = pred.shape[0]
        first_index, second_index = clip_frame_pairs_indes(N, device)
        
        scale2align = kabsch_umeyama(gt, pred).detach().clamp(max=10.0)
        #print('scale2align:',scale2align)
        pred_aligned = pred * scale2align

        delta_pred_aligned = pred_aligned[first_index] - pred_aligned[second_index]
        delta_gt = gt[first_index] - gt[second_index]
        error = torch.norm(delta_pred_aligned - delta_gt, dim=-1).mean()

        if torch.isnan(error):
            #print('_calc_world_trans_loss_ is nan', delta_pred_aligned)
            continue
        loss.append(error)
    loss = torch.stack(loss) if len(loss)>0 else torch.zeros(1,device=device)
    return loss


def get_3Dbbox(verts):
    # get 3D bounding box from predicted 3D vertex of body surface
    minimum = verts.min(1).values
    maximum = verts.max(1).values
    min_x, min_y, min_z = minimum[:,0], minimum[:,1], minimum[:,2]
    max_x, max_y, max_z = maximum[:,0], maximum[:,1], maximum[:,2]
    bbox3D = torch.stack([
        minimum,
        torch.stack([max_x, min_y, min_z],1),
        torch.stack([min_x, max_y, min_z],1),
        torch.stack([min_x, min_y, max_z],1),
        torch.stack([max_x, max_y, min_z],1),
        torch.stack([min_x, max_y, max_z],1),
        torch.stack([max_x, min_y, max_z],1),
        maximum],1)
    return bbox3D

def get_valid_offset_mask(traj_gts, clip_frame_ids):
    dims = traj_gts.shape[-1]
    current_position = traj_gts[torch.arange(len(clip_frame_ids)), clip_frame_ids]
    previous_position = traj_gts[torch.arange(len(clip_frame_ids)), clip_frame_ids-1]

    valid_offset_mask = (clip_frame_ids != 0).to(current_position.device) * \
                        ((current_position != -2.).sum(-1) == dims) * \
                        ((previous_position != -2.).sum(-1) == dims)
    offset_gts = current_position[valid_offset_mask] - previous_position[valid_offset_mask]
    return valid_offset_mask, offset_gts

def get_valid_offset_maskV2(traj_gts, clip_frame_ids):
    dims = traj_gts.shape[-1]
    current_position = traj_gts.clone()
    previous_position = traj_gts[torch.arange(len(traj_gts))-1].clone()

    valid_offset_mask = (clip_frame_ids.clone() != 0).to(current_position.device) * \
                        ((current_position != -2.).sum(-1) == dims) * \
                        ((previous_position != -2.).sum(-1) == dims)
    offset_gts = current_position[valid_offset_mask] - previous_position[valid_offset_mask]
    return valid_offset_mask, offset_gts

def extract_sequence_inds(subject_ids, video_seq_ids, clip_frame_ids):
    sequence_inds = []
    for seq_id in torch.unique(video_seq_ids):
        seq_inds = torch.where(video_seq_ids==seq_id)[0]
        for subj_id in torch.unique(subject_ids[seq_inds]):
            if subj_id == -1:
                continue
            subj_mask = subject_ids[seq_inds]==subj_id
            sequence_inds.append(seq_inds[subj_mask][torch.argsort(clip_frame_ids[seq_inds][subj_mask])])
    return sequence_inds

def match_previous_frame_inds(subject_ids, video_seq_ids, clip_frame_ids):
    previous_frame_result_inds = torch.ones(len(video_seq_ids)).long() * -1
    previous_frame_clip_ids = clip_frame_ids - 1
    for rid, (sid, vid, cid) in enumerate(zip(subject_ids, video_seq_ids, previous_frame_clip_ids)):
        if sid == -1 or cid == -1:
            continue
        match_mask = (subject_ids == sid) * (video_seq_ids == vid) * (clip_frame_ids == cid)
        if match_mask.sum()==1:
            matched_ind = torch.where(match_mask)[0]
            previous_frame_result_inds[rid] = matched_ind
    return previous_frame_result_inds


def calc_temporal_shape_consistency_loss(pred_betas, sequence_inds, weights=None):
    temp_shape_consist_loss = 0

    seq_losses = []
    for seq_inds in sequence_inds:
        seq_pred_betas = pred_betas[seq_inds]
        average_shape = seq_pred_betas.mean(0).unsqueeze(0).detach()
        diff = seq_pred_betas - average_shape
        if weights is not None:
            diff = diff * weights.unsqueeze(0).to(seq_pred_betas.device)
        #(seq_pred_betas.unsqueeze(1) - seq_pred_betas.unsqueeze(0)) * weights.unsqueeze(0).unsqueeze(0).to(seq_pred_betas.device)
        seq_losses.append(torch.norm(diff, p=2, dim=1))
    if len(seq_losses)>0:
        temp_shape_consist_loss = torch.cat(seq_losses, 0)

    return temp_shape_consist_loss

def calc_vel_acc_error(seq_gts, seq_preds, seq_mask):
    velocity_mask = (seq_mask[1:]*seq_mask[:-1]).float()
    vel_gts = seq_gts[1:] - seq_gts[:-1]
    vel_preds = seq_preds[1:] - seq_preds[:-1]
    velocity_loss = (vel_gts - vel_preds) * velocity_mask[:,None]
    velocity_loss = torch.norm(velocity_loss, p=2, dim=1)

    acc_mask = (velocity_mask[1:]*velocity_mask[:-1]).float()
    acc_gts = vel_gts[1:] - vel_gts[:-1]
    acc_preds = vel_preds[1:] - vel_preds[:-1]
    acc_loss = (acc_gts - acc_preds) * acc_mask[:,None]
    acc_loss = torch.norm(acc_loss, p=2, dim=1)
    # to be equal length with the vel loss
    acc_loss = torch.cat([torch.zeros_like(acc_loss)[[0]], acc_loss],0)

    return velocity_loss, acc_loss

def calc_temporal_world_kp3ds_consistency_loss(world_cams_gt, world_cam_masks, world_cams_pred, kp3d_gts, kp3d_preds, sequence_inds):
    temp_foot_consist_loss = 0
    seq_losses = []
    for seq_inds in sequence_inds:
        pred_seq_trans = denormalize_cam_params_to_trans(world_cams_pred[seq_inds], positive_constrain=False)
        gt_seq_cams = denormalize_cam_params_to_trans(world_cams_gt[seq_inds], positive_constrain=False)
        seq_loss = 0
        for kp_ind in range(kp3d_gts.shape[1]):
            seq_kp3d_gts = kp3d_gts[seq_inds, kp_ind]
            seq_kp3d_preds = kp3d_preds[seq_inds, kp_ind]
            seq_mask = world_cam_masks[seq_inds] * ((seq_kp3d_gts !=-2).sum(-1) ==3)

            world_kp3d_gts = seq_kp3d_gts + gt_seq_cams
            world_kp3d_preds = seq_kp3d_preds + pred_seq_trans
            velocity_loss, acc_loss = calc_vel_acc_error(world_kp3d_gts, world_kp3d_preds, seq_mask)
            seq_loss = seq_loss + velocity_loss + acc_loss

        if (seq_loss>0).sum()>0:
            seq_losses.append(seq_loss[seq_loss>0])
    if len(seq_losses)>0:
        temp_foot_consist_loss = torch.cat(seq_losses, 0)

    return temp_foot_consist_loss

def calc_temporal_camtrans_consistency_loss(cams_gts, gt_masks, cam_preds, sequence_inds):
    temp_camtrans_consist_loss = 0
    seq_losses = []
    for seq_inds in sequence_inds:
        pred_seq_cams = cam_preds[seq_inds]
        gt_seq_cams = cams_gts[seq_inds]
        seq_mask = gt_masks[seq_inds]

        velocity_loss, acc_loss = calc_vel_acc_error(gt_seq_cams, pred_seq_cams, seq_mask)
        seq_loss = velocity_loss + acc_loss
        if (seq_loss>0).sum()>0:
            seq_losses.append(seq_loss[seq_loss>0])
    if len(seq_losses)>0:
        temp_camtrans_consist_loss = torch.cat(seq_losses, 0)

    return temp_camtrans_consist_loss

def calc_temporal_globalrot_consistency_loss(grot_gts, gt_masks, grot_preds, sequence_inds):
    temp_globalrot_consist_loss = 0

    seq_losses = []
    for seq_inds in sequence_inds:
        pred_seqs = grot_preds[seq_inds]
        gt_seqs = grot_gts[seq_inds]
        seq_mask = gt_masks[seq_inds]
        
        velocity_loss, acc_loss = calc_vel_acc_error(gt_seqs, pred_seqs, seq_mask)
        seq_loss = velocity_loss + acc_loss
        if (seq_loss>0).sum()>0:
            seq_losses.append(seq_loss[seq_loss>0])
    if len(seq_losses)>0:
        temp_globalrot_consist_loss = torch.cat(seq_losses, 0)

    return temp_globalrot_consist_loss


def calc_cam_loss_with_full_body_bboxes(all_pred_cams, all_pred_verts, full_body_bboxes, focal_length):
    valid_mask = (full_body_bboxes!=-2).sum(-1) == 0
    if valid_mask.sum() == 0:
        return 0
    pred_verts = all_pred_verts[valid_mask]
    fb_bboxes = full_body_bboxes[valid_mask]
    
    bbox3D = get_3Dbbox(pred_verts)

    if args().estimate_camera:
        pred_trans = all_pred_cams[valid_mask]
        pred_fovs = focal_length[valid_mask]
        bbox_2Dprojection = perspective_projection_withfovs(
                        bbox3D, translation=pred_trans, fovs=pred_fovs)
    else:
        pred_cams = all_pred_cams[valid_mask]
        trans_preds = denormalize_cam_params_to_trans(pred_cams, positive_constrain=False).to(bbox3D.device)
        bbox_2Dprojection = perspective_projection(bbox3D, translation=trans_preds, focal_length=focal_length, normalize=True)[:,:,:2].float()
    
    bbox3D_2Dcenter_projection_preds = (bbox_2Dprojection.min(1).values + bbox_2Dprojection.max(1).values) / 2
    bbox2Dcenter_gts = (fb_bboxes[:, :2] + fb_bboxes[:, 2:]) / 2
    
    fbbox_cam_loss = torch.norm(bbox3D_2Dcenter_projection_preds - bbox2Dcenter_gts, p=2, dim=1)
    return fbbox_cam_loss


def calc_motion_offsets2D_loss(pred_motion_offsets, pred_trans, pred_verts, full_body_bboxes, pre_frame_inds, traj2D_gts, clip_frame_ids, focal_length, pred_cams, pred_fovs):
    pre_frame_inds = pre_frame_inds.to(full_body_bboxes.device)
    valid_mask = (pre_frame_inds != -1) * ((full_body_bboxes!=-2).sum(-1) == 0)
    if valid_mask.sum() == 0:
        return 0
    
    if len(traj2D_gts.shape) == 3:
        valid_2Doffset_masks, offset2D_gts = get_valid_offset_mask(traj2D_gts[valid_mask], clip_frame_ids[valid_mask])
    elif len(traj2D_gts.shape) == 2:
        valid_2Doffset_masks, offset2D_gts = get_valid_offset_maskV2(traj2D_gts[valid_mask], clip_frame_ids[valid_mask])
    #valid_2Doffset_masks, offset2D_gts = get_valid_offset_mask(traj2D_gts[valid_mask], clip_frame_ids[valid_mask])
    if valid_2Doffset_masks.sum() == 0:
        return 0
    
    # TODO: check the projection format, like previous normalize
    bbox3D = get_3Dbbox(pred_verts[valid_mask])
    if args().estimate_camera:
        cf_bbox_2Dprojection_preds = perspective_projection_withfovs(
                        bbox3D, translation=pred_trans[valid_mask], fovs=pred_fovs[valid_mask])
    else:
        #trans_preds = denormalize_cam_params_to_trans(pred_cams[valid_mask], positive_constrain=False).to(bbox3D.device)
        cf_bbox_2Dprojection_preds = perspective_projection(bbox3D, translation=pred_trans[valid_mask], focal_length=focal_length, normalize=True)[:,:,:2].float()
    cf_bbox3D_2Dcenter_projection_preds = (cf_bbox_2Dprojection_preds.min(1).values + cf_bbox_2Dprojection_preds.max(1).values) / 2

    previous_frame_cams_preds = pred_cams[valid_mask].detach() - pred_motion_offsets[valid_mask]
    prev_bbox3D = get_3Dbbox(pred_verts[pre_frame_inds[valid_mask]])
    if args().estimate_camera:
        prev_trans_preds = denormalize_cam_params_to_trans(previous_frame_cams_preds, fovs=pred_fovs[valid_mask],  positive_constrain=False)
        pf_bbox_2Dprojection_preds = perspective_projection_withfovs(
                        prev_bbox3D, translation=prev_trans_preds, fovs=pred_fovs[valid_mask])[:,:,:2].float()
    else:
        prev_trans_preds = denormalize_cam_params_to_trans(previous_frame_cams_preds, positive_constrain=False).to(prev_bbox3D.device)
        pf_bbox_2Dprojection_preds = perspective_projection(prev_bbox3D, translation=prev_trans_preds, focal_length=focal_length, normalize=True)[:,:,:2].float()
    pf_bbox3D_2Dcenter_projection_preds = (pf_bbox_2Dprojection_preds.min(1).values + pf_bbox_2Dprojection_preds.max(1).values) / 2

    offset2D_preds = (cf_bbox3D_2Dcenter_projection_preds - pf_bbox3D_2Dcenter_projection_preds)[valid_2Doffset_masks]
    offset2D_gts = offset2D_gts.to(offset2D_preds.device)
    offset2D_loss = torch.norm(offset2D_preds - offset2D_gts, p=2, dim=-1)
    """
    current_frame_center2d_preds = pj2d_preds.mean(1)[valid_mask]
    #previous_frame_center2d_gts = traj2D_gts[torch.arange(len(clip_frame_ids)), clip_frame_ids-1][valid_mask]
    previous_frame_center2d_preds = previous_frame_pj2d_preds.mean(1)
    offset2D_preds = (current_frame_center2d_preds - previous_frame_center2d_preds)[valid_2Doffset_masks]
    offset2D_gts = offset2D_gts.to(offset2D_preds.device)
    #print('previous_frame_center2d_gts, previous_frame_center2d_preds', current_frame_center2d_preds, previous_frame_center2d_preds)
    #center2d_gts_valid_mask = (previous_frame_center2d_gts>-1.99).sum(-1)==previous_frame_center2d_gts.shape[-1]
    if valid_2Doffset_masks.sum()>0:
        offset2D_loss = torch.norm(offset2D_preds - offset2D_gts, p=2, dim=-1)
        if (offset2D_loss>50).sum()>0:
            error_mask = offset2D_loss > 50
            print('offset2D_loss:', offset2D_loss[error_mask])
            print((offset2D_preds - offset2D_gts)[error_mask],
                  offset2D_preds[error_mask], offset2D_gts[error_mask])
        #offset2D_loss = torch.norm(previous_frame_center2d_gts[center2d_gts_valid_mask] - previous_frame_center2d_preds[center2d_gts_valid_mask], p=2, dim=-1)
    """
    return offset2D_loss

def calc_motion_2Dprojection_loss(motion_offsets, cam_preds, kp3d_preds, kp2d_gts, pre_frame_inds, focal_length, pred_fovs):
    valid_mask = pre_frame_inds != -1

    previous_frame_cams_preds = (cam_preds.detach() - motion_offsets)[valid_mask]
    previous_frame_kp2d_gts = kp2d_gts[pre_frame_inds][valid_mask]
    previous_frame_kp3d_preds = kp3d_preds[pre_frame_inds][valid_mask].detach()
    
    if args().estimate_camera:
        previous_frame_trans_preds = denormalize_cam_params_to_trans(previous_frame_cams_preds, fovs=pred_fovs[valid_mask],  positive_constrain=False).to(previous_frame_kp3d_preds.device)
        previous_frame_pj2d_preds = perspective_projection_withfovs(
                        previous_frame_kp3d_preds, translation=previous_frame_trans_preds, fovs=pred_fovs[valid_mask])
    else:
        previous_frame_trans_preds = denormalize_cam_params_to_trans(previous_frame_cams_preds, positive_constrain=False).to(previous_frame_kp3d_preds.device)
        previous_frame_pj2d_preds = perspective_projection(previous_frame_kp3d_preds, translation=previous_frame_trans_preds, focal_length=focal_length, normalize=True)[:,:,:2].float()

    pj2d_loss = batch_kp_2d_l2_loss(previous_frame_kp2d_gts,previous_frame_pj2d_preds)

    return pj2d_loss

def calc_motion_offsets3D_loss(motion_offsets, clip_frame_ids, traj3D_gts):
    if len(traj3D_gts.shape) == 3:
        valid_3Doffset_masks, offset3D_gts = get_valid_offset_mask(traj3D_gts, clip_frame_ids)
    elif len(traj3D_gts.shape) == 2:
        valid_3Doffset_masks, offset3D_gts = get_valid_offset_maskV2(traj3D_gts, clip_frame_ids)
        
    if valid_3Doffset_masks.sum()>0:
        valid_3Doffset_masks = valid_3Doffset_masks.detach()
        offset3D_loss = torch.norm(motion_offsets[valid_3Doffset_masks] - offset3D_gts.to(motion_offsets.device), p=2, dim=-1)
    else:
        offset3D_loss = 0
    return offset3D_loss

def calc_associate_offsets_loss(motion_offsets, cam_preds, subject_ids, video_seq_ids, clip_frame_ids, temp_clip_length):
    traj3D_preds = torch.ones(
        len(motion_offsets), temp_clip_length, 3).float().to(cam_preds.device) * -2.
    for mid in range(len(motion_offsets)):
        if subject_ids[mid] == -1.:
            continue
        traj_mask = (video_seq_ids == video_seq_ids[mid]) & (
            subject_ids == subject_ids[mid])
        traj3D_preds[mid, clip_frame_ids[traj_mask]] = cam_preds[traj_mask]
    return calc_motion_offsets3D_loss(motion_offsets, clip_frame_ids, traj3D_preds)

#trajectory_weight = torch.Tensor([[0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4]]).float()

def _calc_trajectory_visibility_loss_(trajectory_visibility, trajectory2D_gts, Tj_flag):
    if Tj_flag.sum() == 0:
        return 0
    vis_gts = ((trajectory2D_gts[Tj_flag] == -2.).sum(-1) == 0)
    # if person show once, we assume that it might be occluded by objects/other people \
    # if it don't have person 2Ds between first_show_time and last_show_time 
    if (~vis_gts).sum()>0:
        bs, ts = torch.where(~vis_gts)
        for b,t in zip(bs, ts):
            if t == 0:
                continue
            if vis_gts[b,:t].sum()>0:
                show_frame_ids = torch.where(vis_gts[b])[0]
                first_show_time = show_frame_ids[0]
                last_show_time = show_frame_ids[-1]
                vis_gts[b,first_show_time:last_show_time+1] = True
    vis_loss = ((trajectory_visibility[Tj_flag] - vis_gts.float())**2).sum(-1)
    return vis_loss

def _calc_trajectory_probs_loss_(trajectory_probs, matched_pred_ids):
    assert len(trajectory_probs) > matched_pred_ids.max(), \
        "_calc_trajectory_probs_loss_ faile, trajectory_probs shape {}, matched_pred_ids {}".format(trajectory_probs, matched_pred_ids)
    set_match_gts = torch.zeros_like(trajectory_probs)
    set_match_gts[:,0] = 1
    set_match_gts[matched_pred_ids, 0] = 0
    set_match_gts[matched_pred_ids, 1] = 1
    set_prediction_loss = ((trajectory_probs - set_match_gts)**2).sum(-1)
    return set_prediction_loss

def _calc_trajectory2D_movement_loss_(center_trajectory2D, trajectory_cam_preds):
    valid_gt_mask = (center_trajectory2D[:,1:] != -2.).float() * (center_trajectory2D[:,:-1] != -2.).float()
    valid_gt_mask = (valid_gt_mask[...,0] * valid_gt_mask[...,1]).bool()
    if valid_gt_mask.sum()==0:
        return 0
    gt_move2D = center_trajectory2D[:,1:] - center_trajectory2D[:,:-1]
    # extract X-Y scale on image from D-Y-X
    pred_move2D = trajectory_cam_preds[:,1:,[2,1]] - trajectory_cam_preds[:,:-1,[2,1]]
    move2D_loss = ((gt_move2D - pred_move2D)**2).sum(-1)[valid_gt_mask].mean()
    return move2D_loss

def _calc_trajectory3D_loss_(center_trajectory3D, trajectory_cam_preds, trans_vflag):
    trajectory_loss = 0
    if trans_vflag.sum()>0:
        assert (trans_vflag.float() - ((center_trajectory3D!=-2).sum(-1).sum(-1)>0).float()).sum()==0, 'the valid mask of center_trajectory3D and trans_vflag mismatch'
        gt_mask = (center_trajectory3D!=-2).sum(-1) > 0
        trajectory_loss = torch.sqrt(((center_trajectory3D-trajectory_cam_preds)**2).sum(-1)[gt_mask])
        #trajectory_loss = (trajectory_loss * trajectory_weight.to(trajectory_loss.device)).mean(-1)
        trajectory_loss = trajectory_loss.mean()
    return trajectory_loss

def _calc_trajectory_temporal_consistency_loss_(trajectory_preds, reorganize_inds, subject_ids):
    """
    Supervising the temporal consistency of the same subject at each time stamp (reorganize_ind at here)
    """
    loss_list = []
    _, frame_spawn, cam_dim = trajectory_preds.shape
    identity_ids = torch.unique(subject_ids)

    for identity_id in identity_ids:
        subject_mask = torch.where(subject_ids==identity_id)[0]
        subject_batch_ids = reorganize_inds[subject_mask]
        for bid_pairs in itertools.combinations(subject_batch_ids, 2):
            bid_s, bid_l = min(bid_pairs), max(bid_pairs)
            overlapping_number = max(frame_spawn - (bid_l - bid_s), 0)
            if overlapping_number == 0:
                continue
            pid_s, pid_l = [subject_mask[subject_batch_ids==bid] for bid in [bid_s, bid_l]]
            #print(bid_s, bid_l, overlapping_number)
            #print(pid_s, trajectory_preds[pid_s, -overlapping_number:], pid_l, trajectory_preds[pid_l, :overlapping_number])
            loss = ((trajectory_preds[pid_s, -overlapping_number:] - trajectory_preds[pid_l, :overlapping_number])**2).sum(-1).mean()
            loss_list.append(loss)
    if len(loss_list) == 0:
        return 0
    else:
        return sum(loss_list)/len(loss_list)

def test_trajectory_temporal_consistency_loss(frame_spawn=7):
    reorganize_inds = torch.Tensor([0,0,0,2,2,3,3,3,4,4]).long().cuda()
    subject_ids = torch.Tensor([0,1,2,0,1,0,1,2,1,2]).long().cuda()
    trajectory_preds = torch.zeros(len(reorganize_inds),frame_spawn,3)
    for ind, (bid, sid) in enumerate(zip(reorganize_inds, subject_ids)):
        for fid in range(frame_spawn):
            trajectory_preds[ind,fid] = fid + bid + sid * 100
    print(trajectory_preds)
    loss = _calc_trajectory_temporal_consistency_loss_(trajectory_preds, reorganize_inds, subject_ids)
    assert loss==0

if __name__ == '__main__':
    test_trajectory_temporal_consistency_loss()
