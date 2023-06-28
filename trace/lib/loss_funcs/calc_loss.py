from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from config import args
import constants

from utils.center_utils import denormalize_center
from loss_funcs.params_loss import batch_smpl_pose_l2_error, batch_l2_loss
from loss_funcs.keypoints_loss import batch_kp_2d_l2_loss, calc_mpjpe, calc_pampjpe, calc_pj2d_error, batch_kp_2d_l2_loss_old
from loss_funcs.maps_loss import focal_loss, focal_loss_3D
from loss_funcs.prior_loss import MaxMixturePrior
from loss_funcs.relative_loss import relative_depth_loss, relative_shape_loss, relative_age_loss, kid_offset_loss, relative_depth_scale_loss
from loss_funcs.video_loss import calc_motion_offsets3D_loss, calc_motion_offsets2D_loss, match_previous_frame_inds, extract_sequence_inds,\
                                calc_associate_offsets_loss, calc_motion_2Dprojection_loss, calc_cam_loss_with_full_body_bboxes,\
                                calc_temporal_shape_consistency_loss, calc_temporal_camtrans_consistency_loss, \
                                calc_temporal_globalrot_consistency_loss, calc_temporal_world_kp3ds_consistency_loss,\
                                _calc_world_trans_loss_, _calc_world_gros_loss_

from evaluation.evaluation_matrix import _calc_matched_PCKh_
from maps_utils.centermap import CenterMap

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.gmm_prior = MaxMixturePrior(smpl_prior_path=args().smpl_prior_path,num_gaussians=8,dtype=torch.float32) #.cuda()
        if args().HMloss_type=='focal':
            args().heatmap_weight /=1000
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        self.align_inds_MPJPE = np.array([constants.SMPL_ALL_44['L_Hip'], constants.SMPL_ALL_44['R_Hip']])
        self.shape_pca_weight = torch.Tensor([1, 0.64, 0.32, 0.32, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16]).unsqueeze(0).float()
        
        if args().center3d_loss == 'dynamic':
            self.CM = CenterMap()

    def forward(self, outputs, **kwargs):
        meta_data = outputs['meta_data']

        detect_loss_dict = self._calc_detection_loss(outputs, meta_data)
        detection_flag = outputs['detection_flag'].sum()#  if args().model_return_loss else outputs['detection_flag']

        loss_dict = detect_loss_dict
        kp_error = None
        if (detection_flag or args().model_return_loss) and args().calc_mesh_loss:
            mPCKh = _calc_matched_PCKh_(outputs['meta_data']['full_kp2d'].float(), outputs['pj2d'].float(), outputs['meta_data']['valid_masks'][:,0])
            matched_mask = mPCKh > args().matching_pckh_thresh
            kp_loss_dict, kp_error = self._calc_keypoints_loss(outputs, meta_data, matched_mask)
            loss_dict = dict(loss_dict, **kp_loss_dict)

            params_loss_dict = self._calc_param_loss(outputs, meta_data, matched_mask)
            loss_dict = dict(loss_dict, **params_loss_dict)

            if args().video:
                temp_loss_dict = self._calc_temp_loss(outputs, meta_data)
                loss_dict = dict(loss_dict, **temp_loss_dict)
            
            if args().estimate_camera:
                camera_loss_dict = self._calc_camera_loss(outputs, meta_data)
                loss_dict = dict(loss_dict, **camera_loss_dict)

        loss_names = list(loss_dict.keys())
        for name in loss_names:
            if isinstance(loss_dict[name],tuple):
                loss_dict[name] = loss_dict[name][0]
            elif isinstance(loss_dict[name],int):
                loss_dict[name] = torch.zeros(1,device=outputs[list(outputs.keys())[0]].device)
            loss_dict[name] = loss_dict[name].mean() * eval('args().{}_weight'.format(name))

        return {'loss_dict':loss_dict, 'kp_error':kp_error}
    
    def _calc_camera_loss(self, outputs, meta_data):
        camera_loss_dict = {'fovs': 0} #, 'cam_move3D':0, 'cam_rot3D':0
        camera_loss_dict['fovs'] = calc_fov_loss(outputs['fovs'], meta_data['fovs'].squeeze())
        return camera_loss_dict
    
    def _calc_temp_loss(self, outputs, meta_data):
        temp_loss_dict = {item: 0 for item in ['world_foot', 'world_grots','temp_shape_consist']} # 'world_cams_consist', 'world_cams', 'world_pj2D', 'temp_cam_consist', 'temp_rot_consist'
        if args().learn_motion_offset3D:
            temp_loss_dict.update({item: 0 for item in ['motion_offsets3D']}) #, 'associate_offsets3D'
        sequence_mask = outputs['pred_seq_mask'] #meta_data['seq_inds'][:,3].bool()
        if sequence_mask.sum() == 0:
            return temp_loss_dict
            #print('Error!!! no sequence data has been used for calculating temp loss')
        
        #pred_trans = outputs['cam_trans'][sequence_mask]
        #pred_kp3ds = outputs['j3d'][sequence_mask].detach()
        #pred_verts = outputs['verts'][sequence_mask].detach()
        pred_batch_ids = outputs['pred_batch_ids'][sequence_mask].detach().long() - meta_data['batch_ids'][0]
        subject_ids = meta_data['subject_ids'][sequence_mask]
        pred_cams = outputs['cam'][sequence_mask].float()
        #full_kp2ds = meta_data['full_kp2d'][sequence_mask].to(pred_kp3ds.device)
        #full_body_bboxes = meta_data['full_body_bboxes'][sequence_mask]
        
        clip_frame_ids = meta_data['seq_inds'][pred_batch_ids,1]
        video_seq_ids = meta_data['seq_inds'][pred_batch_ids,0]
        sequence_inds = extract_sequence_inds(subject_ids, video_seq_ids, clip_frame_ids)

        if args().dynamic_augment:
            # loss 应该直接监督 delta offset / rotation 而不是最终值，导致误差累计。
            world_cam_masks = meta_data['world_cam_mask'][sequence_mask]
            world_cams_gt = meta_data['world_cams'][sequence_mask]
            world_trans_gts = meta_data['world_root_trans'][sequence_mask]

            pred_world_cams = outputs['world_cams'][sequence_mask].float()
            world_trans_preds = outputs['world_trans'][sequence_mask].float()

            world_global_rots_gt = meta_data['world_global_rots'][sequence_mask]
            world_global_rots_pred = outputs['world_global_rots'][sequence_mask]

            grot_masks = meta_data['valid_masks'][sequence_mask][:,3]
            valid_world_global_rots_mask = torch.logical_and(grot_masks, world_cam_masks)

            temp_loss_dict['world_grots'] = _calc_world_gros_loss_(world_global_rots_pred, world_global_rots_gt, valid_world_global_rots_mask, sequence_inds)
            temp_loss_dict['wrotsL2'] = 0
            if valid_world_global_rots_mask.sum()>0:
                temp_loss_dict['wrotsL2'] = batch_smpl_pose_l2_error(world_global_rots_gt[valid_world_global_rots_mask].contiguous(), world_global_rots_pred[valid_world_global_rots_mask].contiguous()).mean()
            
            #temp_loss_dict['world_trans'] = _calc_world_trans_loss_(world_trans_preds, world_trans_gts, world_cam_masks, sequence_inds)
            #temp_loss_dict['world_cams_consist'] = calc_temporal_camtrans_consistency_loss(world_cams_gt, world_cam_masks, pred_world_cams, sequence_inds)
            #temp_loss_dict['world_cams'] = temp_loss_dict['world_cams'] = batch_l2_loss(world_cams_gt[world_cam_masks], pred_world_cams[world_cam_masks])

            temp_loss_dict['world_pj2D'] = batch_kp_2d_l2_loss_old(meta_data['dynamic_kp2ds'], outputs['world_pj2d'])
            #if 'init_world_pj2d' in outputs:
            #    temp_loss_dict['init_world_pj2d'] = batch_kp_2d_l2_loss_old(meta_data['dynamic_kp2ds'], outputs['init_world_pj2d'])
        
            # if args().learn_foot_contact:
            #     foot_inds = [constants.SMPL_ALL_44['L_Ankle'], constants.SMPL_ALL_44['R_Ankle']]
            #     foot_kp3d_gts = meta_data['kp_3d'][sequence_mask][:,foot_inds].contiguous()
            #     foot_kp3d_preds = outputs['j3d'][sequence_mask][:,foot_inds].detach().contiguous()

            #     temp_loss_dict['world_foot'] = calc_temporal_world_kp3ds_consistency_loss(world_cams_gt, world_cam_masks, pred_world_cams, foot_kp3d_gts, foot_kp3d_preds, sequence_inds)

        if args().learn_temporal_shape_consistency:
            pred_betas = outputs['smpl_betas'][sequence_mask]
            temp_loss_dict['temp_shape_consist'] = calc_temporal_shape_consistency_loss(pred_betas, sequence_inds)

        if args().learn_motion_offset3D:
            pred_motion_offsets = outputs['motion_offsets3D'][sequence_mask]
            traj3D_gts = meta_data['traj3D_gts'][sequence_mask]
            traj2D_gts = meta_data['traj2D_gts'][sequence_mask]
            temp_loss_dict['motion_offsets3D'] = calc_motion_offsets3D_loss(pred_motion_offsets, clip_frame_ids, traj3D_gts)
            #temp_loss_dict['associate_offsets3D'] = calc_associate_offsets_loss(
            #    pred_motion_offsets, pred_cams.detach(), subject_ids, video_seq_ids, clip_frame_ids, args().temp_clip_length)
        
        return temp_loss_dict

    def _calc_detection_loss(self, outputs, meta_data):
        detect_loss_dict = {}
        all_person_mask = meta_data['all_person_detected_mask'].to(outputs['center_map'].device)
        if args().calc_mesh_loss and 'center_map' in outputs:
            
            if all_person_mask.sum()>0:
                detect_loss_dict['CenterMap'] = focal_loss(outputs['center_map'][all_person_mask], meta_data['centermap'][all_person_mask].to(
                    outputs['center_map'].device))  # ((centermaps-centermaps_gt)**2).sum(-1).sum(-1).mean(-1) #
            else:
                detect_loss_dict['CenterMap'] = 0

        reorganize_idx_on_each_gpu = outputs['reorganize_idx']-outputs['meta_data']['batch_ids'][0]

        if 'center_map_3d' in outputs:
            valid_mask_c3d = meta_data['valid_centermap3d_mask'].squeeze().to(outputs['center_map_3d'].device)
            # 必须要都可，才能用于监督，不然loss很容易nan，然后centermap3D很容易学没了。
            valid_mask_c3d = torch.logical_and(valid_mask_c3d, all_person_mask.squeeze())
            detect_loss_dict['CenterMap_3D'] = 0
            valid_mask_c3d = valid_mask_c3d.reshape(-1)
            #print(meta_data['valid_centermap3d_mask'].sum(), meta_data['centermap_3d'][valid_mask_c3d].sum())
            if valid_mask_c3d.sum()>0:
                detect_loss_dict['CenterMap_3D'] = focal_loss_3D(outputs['center_map_3d'][valid_mask_c3d], meta_data['centermap_3d'][valid_mask_c3d].to(outputs['center_map_3d'].device))
            
        return detect_loss_dict

    def _calc_keypoints_loss(self, outputs, meta_data, matched_mask):
        kp_loss_dict, error = {'P_KP2D':0, 'MPJPE':0, 'PAMPJPE':0}, {'3d':{'error':[], 'idx':[]},'2d':{'error':[], 'idx':[]}}
        #if 'kp_ae_maps' in outputs:
        #    kp_loss_dict['KP2D'],kp_loss_dict['AE'] = self.heatmap_AE_loss(real_2d, outputs['kp_ae_maps'], meta_data['heatmap'].to(device), meta_data['AE_joints'])
        real_2d = meta_data['full_kp2d']
        if 'pj2d' in outputs and args().learn2Dprojection:
            if args().model_version == 3:
                kp_loss_dict['joint_sampler'] = self.joint_sampler_loss(real_2d, outputs['joint_sampler_pred'])
            kp_loss_dict['P_KP2D'] = batch_kp_2d_l2_loss_old(real_2d, outputs['pj2d'])
            #batch_kp_2d_l2_loss(real_2d.clone(), outputs['pj2d'].clone(), meta_data['image'])
        # if 'init_pj2d' in outputs and args().learn_cam_init:
        #     kp_loss_dict['init_pj2d'] = batch_kp_2d_l2_loss_old(real_2d, outputs['init_pj2d'])
        
        # if 'pred_kp2ds' in outputs:
        #     real_2d_regression = meta_data['full_kp2d'].to(outputs['pred_kp2ds'].device)[:, constants.keypoints_select].clone()
        #     kp_loss_dict['KP2D_Reg'] = batch_kp_2d_l2_loss_old(real_2d_regression, outputs['pred_kp2ds'])
        #     # some datasets (dancetrack) doesn't have kp2ds
        #     if torch.isnan(kp_loss_dict['KP2D_Reg']).sum() > 0:
        #         kp_loss_dict['KP2D_Reg'] = 0
        
        kp3d_mask = meta_data['valid_masks'][:,1]#.to(outputs['j3d'].device)
        if kp3d_mask.sum()>1 and 'j3d' in outputs:
            kp3d_gt = meta_data['kp_3d'].contiguous().to(outputs['j3d'].device)
            preds_kp3d = outputs['j3d'][:, :kp3d_gt.shape[1]].contiguous()

            if not args().model_return_loss and args().PAMPJPE_weight>0:
                try:
                    pampjpe_each = calc_pampjpe(kp3d_gt[kp3d_mask].contiguous(), preds_kp3d[kp3d_mask].contiguous())
                    kp_loss_dict['PAMPJPE'] = pampjpe_each
                except Exception as exp_error:
                    print('PA_MPJPE calculation failed!', exp_error)
            
            if args().MPJPE_weight>0:
                fit_mask = kp3d_mask.bool()
                if fit_mask.sum()>0:
                    mpjpe_each = calc_mpjpe(kp3d_gt[fit_mask].contiguous(), preds_kp3d[fit_mask].contiguous(), align_inds=self.align_inds_MPJPE)
                    kp_loss_dict['MPJPE'] = mpjpe_each
                    error['3d']['error'].append(mpjpe_each.detach()*1000)
                    error['3d']['idx'].append(torch.where(fit_mask)[0])

        return kp_loss_dict, error

    def _calc_param_loss(self, outputs, meta_data, matched_mask):
        params_loss_dict = {'Pose': 0, 'Shape':0}
        
        _check_params_(meta_data['params'])
        device = outputs['body_pose'].device
        grot_masks, smpl_pose_masks, smpl_shape_masks = meta_data['valid_masks'][:,3].to(device), meta_data['valid_masks'][:,4].to(device), meta_data['valid_masks'][:,5].to(device)

        if grot_masks.sum()>0:
            params_loss_dict['Pose'] += batch_smpl_pose_l2_error(meta_data['params'][grot_masks,:3].to(device).contiguous(), outputs['global_orient'][grot_masks].contiguous()).mean()

        if smpl_pose_masks.sum()>0:
            params_loss_dict['Pose'] += batch_smpl_pose_l2_error(meta_data['params'][smpl_pose_masks,3:22*3].to(device).contiguous(), outputs['body_pose'][smpl_pose_masks,:21*3].contiguous()).mean()
            # the smpl pose annotation of h36m for neck (12) are not zeros 
            #params_loss_dict['Pose'] += batch_smpl_pose_l2_error(meta_data['params'][smpl_pose_masks,3:12*3].to(device).contiguous(), outputs['body_pose'][smpl_pose_masks,:11*3].contiguous()).mean() +\
            #    batch_smpl_pose_l2_error(meta_data['params'][smpl_pose_masks,13*3:22*3].to(device).contiguous(), outputs['body_pose'][smpl_pose_masks,12*3:21*3].contiguous()).mean()

        if smpl_shape_masks.sum()>0:
            # beta annots in datasets are for each gender (male/female), not for our neutral. 
            smpl_shape_diff = meta_data['params'][smpl_shape_masks,-10:].to(device).contiguous() - outputs['smpl_betas'][smpl_shape_masks,:10].contiguous()
            params_loss_dict['Shape'] += torch.norm(smpl_shape_diff*self.shape_pca_weight.to(device), p=2, dim=-1).mean() / 20.
        
        if args().separate_smil_betas:
            # TODO: separate loss for separate_smil_betas 10 + 11
            pass

        if (~smpl_shape_masks).sum()>0:
           params_loss_dict['Shape'] += (outputs['smpl_betas'][~smpl_shape_masks,:10]**2).mean() / 80.
        
        if args().supervise_cam_params:
            params_loss_dict.update({'Cam':0})
            cam_mask = meta_data['cam_mask']
            if cam_mask.sum()>0:
                params_loss_dict['Cam'] += batch_l2_loss(meta_data['cams'][cam_mask], outputs['cam'][cam_mask])

        if args().learn_relative:
            if args().learn_relative_age:
                params_loss_dict['R_Age'] = relative_age_loss(outputs['kid_offsets_pred'], meta_data['depth_info'][:,0], matched_mask=matched_mask) + \
                                            kid_offset_loss(outputs['kid_offsets_pred'], meta_data['kid_shape_offsets'], matched_mask=matched_mask) * 2
            if args().learn_relative_depth:
                params_loss_dict['R_Depth'] = relative_depth_loss(outputs['cam_trans'][:,2], meta_data['depth_info'][:,3], outputs['reorganize_idx'], matched_mask=matched_mask)
            
            if args().relative_depth_scale_aug and not args().model_return_loss:
                torso_pj2d_errors = calc_pj2d_error(meta_data['full_kp2d'].to(outputs['pj2d'].device).clone(), outputs['pj2d'].float().clone(), \
                        joint_inds=constants.torso_joint_inds)
                params_loss_dict['R_Depth_scale'] = relative_depth_scale_loss(
                    outputs['cam_trans'][:, 2], meta_data['img_scale'], meta_data['subject_ids'], outputs['reorganize_idx'], torso_pj2d_errors)

        if args().learn_gmm_prior:
            gmm_prior_loss = self.gmm_prior(outputs['body_pose']).mean()/100.
            # remove the low loss, only punish the high loss
            valuable_prior_loss_thresh=8.
            gmm_prior_loss[gmm_prior_loss<valuable_prior_loss_thresh] = 0
            #angle_prior_loss = angle_prior(outputs['body_pose']).mean()/5.
            params_loss_dict['Prior'] = gmm_prior_loss# + angle_prior_loss

        return params_loss_dict

    def joint_sampler_loss(self, real_2d, joint_sampler):
        batch_size = joint_sampler.shape[0]
        joint_sampler = joint_sampler.view(batch_size, -1, 2)
        joint_gt = real_2d[:,constants.joint_sampler_mapper]
        loss = batch_kp_2d_l2_loss(joint_gt, joint_sampler)
        return loss


def _check_params_(params):
    assert params.shape[0]>0, logging.error('meta_data[params] dim 0 is empty, params: {}'.format(params))
    assert params.shape[1]>0, logging.error('meta_data[params] dim 1 is empty, params shape: {}, params: {}'.format(params.shape, params))
