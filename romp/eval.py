
from .base import *
from loss_funcs import _calc_MPJAE, calc_mpjpe, calc_pampjpe, calc_pck, align_by_parts
from evaluation import h36m_evaluation_act_wise, cmup_evaluation_act_wise
from evaluation.evaluation_matrix import _calc_relative_age_error_weak_, _calc_absolute_depth_error,\
                                    _calc_relative_depth_error_weak_, _calc_relative_depth_error_withgts_, _calc_matched_PCKh_


def calc_outputs_evaluation_matrix(self, outputs, ED):
    for ds in set(outputs['meta_data']['data_set']):
        val_idx = np.where(np.array(outputs['meta_data']['data_set']) == ds)[0]
        real_3d = outputs['meta_data']['kp_3d'][val_idx].contiguous().cuda()
        if ds in constants.dataset_smpl2lsp:
            real_3d = real_3d[:, self.All54_to_LSP14_mapper].contiguous()
            if (self.All54_to_LSP14_mapper == -1).sum() > 0:
                real_3d[:, self.All54_to_LSP14_mapper == -1] = -2.

            predicts = outputs['joints_h36m17'].contiguous()
            align_inds = [constants.LSP_14['R_Hip'],
                              constants.LSP_14['L_Hip']]
            bones, colors, kp_colors = constants.lsp14_connMat, constants.cm_body14, constants.lsp14_kpcm
        else:
            predicts = outputs['j3d'][val_idx, :24].contiguous()
            real_3d = real_3d[:, :24].contiguous()
            align_inds = [constants.SMPL_24['Pelvis_SMPL']]
            bones, colors, kp_colors = constants.smpl24_connMat, constants.cm_smpl24, constants.smpl24_kpcm

        mPCKh = _calc_matched_PCKh_(outputs['meta_data']['full_kp2d'].float(), outputs['pj2d'].float(), outputs['meta_data']['valid_masks'][:, 0])
        ED['mPCKh'][ds].append(mPCKh)
        matched_mask = mPCKh > args().matching_pckh_thresh
        if ds in constants.dataset_depth:
            predicts_j3ds = outputs['j3d'][val_idx].contiguous().detach().cpu().numpy()
            predicts_pj2ds = outputs['pj2d_org'].detach().cpu().numpy()
            if ds in ['agora', 'mini']:
                predicts_j3ds = predicts_j3ds[:,:24] - predicts_j3ds[:, [0]]
                predicts_pj2ds = predicts_pj2ds[:, :24]
            #trans_preds = estimate_translation(predicts_j3ds, predicts_pj2ds, \
            #    proj_mats=outputs['meta_data']['camMats'].cpu().numpy(), cam_dists=outputs['meta_data']['camDists'].cpu().numpy(),pnp_algorithm='cv2')
            trans_preds = outputs['cam_trans'].detach().cpu()
            trans_gts = outputs['meta_data']['root_trans']

            # _calc_absolute_depth_error(trans_preds.numpy(), trans_gts.cpu().numpy())
            ED['root_depth'][ds].append(np.concatenate([trans_preds.numpy()[None], trans_gts.cpu().numpy()[None]]))
            age_gts = outputs['meta_data']['depth_info'][:,0] if 'depth_info' in outputs['meta_data'] else None
            relative_depth_errors = _calc_relative_depth_error_withgts_(trans_preds[:, 2], trans_gts[:, 2], outputs['reorganize_idx'],
                                                                            age_gts=age_gts, matched_mask=matched_mask)
            for dr_type in constants.relative_depth_types:
                ED['depth_relative'][ds][dr_type] += relative_depth_errors[dr_type]
                ED['depth_relative'][ds][dr_type +'_age'] += relative_depth_errors[dr_type+'_age']

        if ds in ED['depth_relative']:
            age_gts = outputs['meta_data']['depth_info'][:,0] if 'depth_info' in outputs['meta_data'] else None
            relative_depth_errors = _calc_relative_depth_error_weak_(outputs['cam_trans'][:, 2], outputs['meta_data']['depth_info'][:, 3],
                                                                         outputs['reorganize_idx'], age_gts=age_gts, matched_mask=matched_mask)
            for dr_type in constants.relative_depth_types:
                ED['depth_relative'][ds][dr_type] += relative_depth_errors[dr_type]
                ED['depth_relative'][ds][dr_type +'_age'] += relative_depth_errors[dr_type+'_age']
        if ds in ED['age_relative'] and args().learn_relative:
            relative_age_errors = _calc_relative_age_error_weak_(outputs['Age_preds'], outputs['meta_data']['depth_info'][:, 0], matched_mask=matched_mask)
            for age_type in constants.relative_age_types:
                ED['age_relative'][ds][age_type] += relative_age_errors[age_type]

        if ds not in constants.dataset_nokp3ds:
            if args().calc_PVE_error and ds in constants.PVE_ds:
                batch_PVE = torch.norm(
                    outputs['meta_data']['verts'][val_idx]-outputs['verts'][val_idx], p=2, dim=-1).mean(-1)
                ED['PVE_new'][ds].append(batch_PVE)

            abs_error, aligned_poses = calc_mpjpe(real_3d, predicts, align_inds=align_inds, return_org=True)
            abs_error = abs_error.float().cpu().numpy()*1000
            rt_error = calc_pampjpe(real_3d, predicts).float().cpu().numpy()*1000
            kp3d_vis = (*aligned_poses, bones)

            if self.calc_pck:
                pck_joints_sampled = constants.SMPL_MAJOR_JOINTS if real_3d.shape[1] == 24 else np.arange(12)
                mpjpe_pck_batch = calc_pck(
                        real_3d, predicts, lrhip=lrhip, pck_joints=pck_joints_sampled).cpu().numpy()*1000
                ED['PCK3D'][ds].append((mpjpe_pck_batch.reshape(-1) < self.PCK_thresh).astype(np.float32)*100)
                if ds in constants.MPJAE_ds:
                    rel_pose_pred = torch.cat([outputs['params']['global_orient'][val_idx], outputs['params']['body_pose'][val_idx]], 1)[:, :22*3].contiguous()
                    rel_pose_real = outputs['meta_data']['params'][val_idx, :22*3].cuda()
                    MPJAE_error = _calc_MPJAE(rel_pose_pred, rel_pose_real)
                    ED['MPJAE'][ds].append(MPJAE_error)

            ED['MPJPE'][ds].append(abs_error.astype(np.float32))
            ED['PA_MPJPE'][ds].append(rt_error.astype(np.float32))
            ED['imgpaths'][ds].append(np.array(outputs['meta_data']['imgpath'])[val_idx])
        else:
            kp3d_vis = None
    return ED, kp3d_vis

@torch.no_grad()
def val_result(self, loader_val, evaluation = False, vis_results=False):
    eval_model = nn.DataParallel(self.model.module).eval()
    ED = _init_error_dict()

    for iter_num, meta_data in enumerate(loader_val):
        if meta_data is None:
            continue

        meta_data_org = meta_data.copy()
        try:
            outputs = self.network_forward(eval_model, meta_data, self.eval_cfg)
        except:
            continue

        if outputs['detection_flag'].sum()==0:
            print('Detection failure!!! {}'.format(outputs['meta_data']['imgpath']))
            continue

        ED, kp3d_vis = calc_outputs_evaluation_matrix(
            self, outputs, ED)

        if iter_num % (self.val_batch_size*2) == 0:
            print('{}/{}'.format(iter_num, len(loader_val)))
            #eval_results = print_results(ED.copy())
            if not evaluation:
                outputs = self.network_forward(eval_model, meta_data_org, self.val_cfg)
            vis_ids = np.arange(max(min(self.val_batch_size, len(outputs['reorganize_idx'])), 8)//4), None
            save_name = '{}_{}'.format(self.global_count,iter_num)
            for ds_name in set(outputs['meta_data']['data_set']):
                save_name += '_{}'.format(ds_name)
            show_items = ['mesh', 'joint_sampler', 'pj2d', 'classify']
            if kp3d_vis is not None:
                show_items.append('j3d')
            self.visualizer.visulize_result(outputs, outputs['meta_data'], show_items=show_items,\
                vis_cfg={'settings': ['save_img'], 'vids': vis_ids, 'save_dir':self.result_img_dir, 'save_name':save_name}, kp3ds=kp3d_vis) #'org_img', 

    print('{} on local_rank {}'.format(['Evaluation' if evaluation else 'Validation'], self.local_rank))
    eval_results = print_results(ED)

    return eval_results


def print_results(ED):
    eval_results = {}
    for key, results in ED['root_depth'].items():
        if len(results)>0:
            results_all = np.concatenate(results,axis=1)
            axis_error = np.abs(results_all[0] - results_all[1]).mean(0)
            root_error = np.sqrt(np.sum((results_all[0] - results_all[1]) ** 2, axis=1)).mean()
            print('Root trans error of {}: {:.4f} | axis-wise (x,y,z) error: {}'.format(key, root_error, axis_error))

    for ds, results in ED['depth_relative'].items():
        result_length = sum([len(ED['depth_relative'][ds][dr_type]) for dr_type in constants.relative_depth_types])
        if result_length>0:
            eq_dists = torch.cat(ED['depth_relative'][ds]['eq'], 0)
            cd_dists = torch.cat(ED['depth_relative'][ds]['cd'], 0)
            fd_dists = torch.cat(ED['depth_relative'][ds]['fd'], 0)
            age_flag = len(ED['depth_relative'][ds]['eq_age'])>0
            if age_flag:
                eq_age_ids = torch.cat(ED['depth_relative'][ds]['eq_age'], 0)
                cd_age_ids = torch.cat(ED['depth_relative'][ds]['cd_age'], 0)
                fd_age_ids = torch.cat(ED['depth_relative'][ds]['fd_age'], 0)
                dr_age_ids = torch.cat([eq_age_ids, cd_age_ids, fd_age_ids], 0)
            dr_all = np.array([len(eq_dists), len(cd_dists), len(fd_dists)])
            for dr_thresh in [0.2]: #[0.1,0.15,0.2,0.25,0.3]:
                dr_corrects = [torch.abs(eq_dists)<dr_thresh, cd_dists<-dr_thresh, fd_dists>dr_thresh]
                print('Thresh: {} | Equal {} close {} far {}'.format(dr_thresh, dr_corrects[0].sum().item() / dr_all[0], \
                                                    dr_corrects[1].sum().item() / dr_all[1], dr_corrects[2].sum().item() / dr_all[2]))
                dr_corrects = torch.cat(dr_corrects,0)
                eval_results['{}-PCRD_{}'.format(ds, dr_thresh)] = dr_corrects.sum() / dr_all.sum()
                if age_flag:
                    for age_ind, age_name in enumerate(constants.relative_age_types):
                        age_mask = (dr_age_ids == age_ind).sum(-1).bool()
                        if age_mask.sum()>0:
                            eval_results['{}-PCRD_{}_{}'.format(ds, dr_thresh, age_name)] = dr_corrects[age_mask].sum() / age_mask.sum()
    
    for ds, results in ED['age_relative'].items():
        result_length = sum([len(ED['age_relative'][ds][age_type]) for age_type in constants.relative_age_types])
        if result_length>0:
            print('Relative age evaluation results:')
            age_error_results = {}
            for age_id, age_type in enumerate(constants.relative_age_types):
                age_pred_ids = torch.cat(ED['age_relative'][ds][age_type], 0)
                age_error_results[age_type] = (age_pred_ids==age_id).float()
                if age_id== 0:
                    near_error_results = (age_pred_ids==1).float()
                elif age_id == 1:
                    near_error_results = (age_pred_ids==0).float() + (age_pred_ids==2).float()
                elif age_id == 2:
                    near_error_results = (age_pred_ids==1).float() + (age_pred_ids==3).float()
                elif age_id == 3:
                    near_error_results = (age_pred_ids==2).float()
                age_error_results[age_type] += near_error_results.float() * 0.667
                eval_results['{}-acc_{}'.format(ds, age_type)] = age_error_results[age_type].sum() / len(age_error_results[age_type])
                
            age_all_results = torch.cat(list(age_error_results.values()), 0)
            eval_results['{}-age_acc'.format(ds)] = age_all_results.sum() / len(age_all_results)

    for ds, results in ED['mPCKh'].items():
        if len(ED['mPCKh'][ds])>0:
            mPCKh = torch.cat(ED['mPCKh'][ds], 0)
            mPCKh = mPCKh[mPCKh!=-1]
            for thresh in range(6,7): #range(1,11):
                thresh = thresh / 10.
                eval_results['{}-mPCKh_{}'.format(ds, thresh)] = (mPCKh >= thresh).sum() / len(mPCKh)

    eval_results.update(process_matrix(ED['MPJPE'],'MPJPE'))
    eval_results.update(process_matrix(ED['PA_MPJPE'],'PA_MPJPE'))
    if args().calc_pck:
        eval_results.update(process_matrix(ED['PCK3D'],'PCK3D'))

    if args().calc_PVE_error:
        for ds_name in constants.PVE_ds:
            if len(ED['PVE_new'][ds_name])>0:
                eval_results['{}-PVE'.format(ds_name)] = torch.cat(ED['PVE_new'][ds_name], 0).mean() * 1000

    for ds_name in constants.MPJAE_ds:
        if ds_name in ED['MPJAE']:
            if len(ED['MPJAE'][ds_name])>0:
                eval_results['{}-MPJAE'.format(ds_name)] = np.concatenate(ED['MPJAE'][ds_name],axis=0).mean()

    print_table(eval_results)
    
    if len(ED['MPJPE']['h36m'])>0:
        print('Detail results on Human3.6M dataset:')
        PA_MPJPE_acts = h36m_evaluation_act_wise(np.concatenate(ED['PA_MPJPE']['h36m'],axis=0),np.concatenate(np.array(ED['imgpaths']['h36m']),axis=0),constants.h36m_action_names)
        MPJPE_acts = h36m_evaluation_act_wise(np.concatenate(ED['MPJPE']['h36m'],axis=0),np.concatenate(np.array(ED['imgpaths']['h36m']),axis=0),constants.h36m_action_names)
        table = PrettyTable(['Protocol']+constants.h36m_action_names)
        table.add_row(['1']+MPJPE_acts)
        table.add_row(['2']+PA_MPJPE_acts)
        print(table)

    return eval_results

def process_matrix(matrix, name, times=1.):
    eval_results = {}
    for ds, error_list in matrix.items():
        if len(error_list)>0:
            result = np.concatenate(error_list,axis=0)
            result = result[~np.isnan(result)].mean()
            eval_results['{}-{}'.format(ds,name)] = result*times
    return eval_results

def _init_error_dict():
    ED = {}
    ED['MPJPE'], ED['PA_MPJPE'], ED['PCK3D'], ED['imgpaths'] = [{ds:[] for ds in constants.dataset_involved} for _ in range(4)]
    ED['MPJAE'] = {ds:[] for ds in constants.MPJAE_ds}
    ED['PVE_new'] = {ds:[] for ds in constants.PVE_ds}
    ED['PVE'] = {ds:{'target_theta':[], 'pred_theta':[]} for ds in constants.PVE_ds}
    ED['ds_bias'] = {ds:{'scale':[], 'trans':[]} for ds in constants.dataset_involved}
    ED['root_depth'] = {ds:[] for ds in constants.dataset_depth}
    ED['mPCKh'] = {ds:[] for ds in constants.dataset_kp2ds}
    ED['depth_relative'] = {ds:{'eq': [], 'cd': [], 'fd':[], 'eq_age': [], 'cd_age': [], 'fd_age':[]} for ds in constants.dataset_relative_depth+constants.dataset_depth}
    ED['age_relative'] = {ds:{age_name:[] for age_name in constants.relative_age_types} for ds in constants.dataset_relative_age}
    return ED

def print_table(eval_results):
    matrix_dict = {}
    em_col_id = 0
    matrix_list = []
    for name in eval_results:
        ds,em = name.split('-')
        if em not in matrix_dict:
            matrix_dict[em] = em_col_id
            matrix_list.append(em)
            em_col_id += 1
    
    raw_dict = {}
    for name, result in eval_results.items():
        ds,em = name.split('-')
        if ds not in raw_dict:
            raw_dict[ds] = np.zeros(em_col_id).tolist()
        raw_dict[ds][matrix_dict[em]] = '{:.3f}'.format(result)

    table = PrettyTable(['DS/EM']+matrix_list)
    for idx, (ds, mat_list) in enumerate(raw_dict.items()):
        table.add_row([ds]+mat_list)
    print(table)
    print('-'*20)


if __name__ == '__main__':
    test_depth_error()
