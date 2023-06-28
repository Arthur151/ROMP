import torch
from config import args
from utils.rot_6D import rot6D_to_angular
from loss_funcs.params_loss import batch_smpl_pose_l2_error


def suppressing_silimar_mesh_and_2D_center(params_preds, pred_batch_ids, pred_czyxs, top_score,rot_dim=6, center2D_thresh=5, pose_thresh=2.5): # center2D_thresh=5, pose_thresh=2.5 center2D_thresh=2, pose_thresh=1.2
    pose_params_preds = params_preds[:, args().cam_dim:args().cam_dim+22*rot_dim]

    N = len(pred_czyxs)
    center2D_similarity = torch.norm((pred_czyxs[:,1:].unsqueeze(1).repeat(1,N,1) - pred_czyxs[:,1:].unsqueeze(0).repeat(N,1,1)).float(), p=2, dim=-1)
    same_batch_id_mask = pred_batch_ids.unsqueeze(1).repeat(1,N) == pred_batch_ids.unsqueeze(0).repeat(N,1)
    center2D_similarity[~same_batch_id_mask] = center2D_thresh + 1
    similarity = center2D_similarity <= center2D_thresh
    center_similar_inds = torch.where(similarity.sum(-1)>1)[0]

    #print(params_preds.shape, pred_batch_ids.shape, pred_czyxs.shape, top_score.shape, similarity.shape)
    #print('suppressing_silimar', similarity)

    for s_inds in center_similar_inds:
        if rot_dim==6:
            pose_angulars = rot6D_to_angular(pose_params_preds[similarity[s_inds]])
            pose_angular_base = rot6D_to_angular(pose_params_preds[s_inds].unsqueeze(0)).repeat(len(pose_angulars), 1)
        elif rot_dim==3:
            pose_angulars = pose_params_preds[similarity[s_inds]]
            pose_angular_base = pose_params_preds[s_inds].unsqueeze(0).repeat(len(pose_angulars))
        pose_similarity = batch_smpl_pose_l2_error(pose_angulars,pose_angular_base)
        sim_past = similarity[s_inds].clone()
        similarity[s_inds,sim_past] = (pose_similarity<pose_thresh)

    score_map = similarity * top_score.unsqueeze(0).repeat(N,1)
    nms_inds = torch.argmax(score_map,1) == torch.arange(N).to(score_map.device)
    return [item[nms_inds] for item in [pred_batch_ids, pred_czyxs, top_score]], nms_inds

def suppressing_duplicate_mesh(outputs, rot_dim=args().rot_dim):
    # During training, do not use nms to facilitate more thourough learning
    (pred_batch_ids, pred_czyxs, top_score), nms_inds = suppressing_silimar_mesh_and_2D_center(
        outputs['params_pred'], outputs['pred_batch_ids'], outputs['pred_czyxs'], outputs['top_score'], rot_dim=rot_dim)
    outputs['params_pred']= outputs['params_pred'][nms_inds]
    if 'cam_czyx' in outputs:
        outputs['cam_czyx'] = outputs['cam_czyx'][nms_inds]
    if 'world_cams' in outputs:
        outputs['world_cams'] = outputs['world_cams'][nms_inds]
    outputs.update({'pred_batch_ids': pred_batch_ids, 'pred_czyxs': pred_czyxs, 'top_score': top_score})
    return outputs
