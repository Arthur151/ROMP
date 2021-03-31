
from base import *

dataset_involved, PVE_ds = ['pw3d_vibe'],['pw3d_vibe']

@torch.no_grad()
def val_result(self, loader_val, evaluation = False):
    self.model.eval()
    self.All54_to_LSP14_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.LSP_14)
    ED = _init_error_dict(dataset_involved, PVE_ds)

    for iter_num, meta_data in enumerate(loader_val):
        outputs = self.net_forward(meta_data, cfg=self.test_cfg)        
        error_dict= {'3d':{'error':[], 'idx':[]},'2d':{'error':[], 'idx':[]}}
        for ds in set(outputs['meta_data']['data_set']):
            val_idx = np.where(np.array(outputs['meta_data']['data_set'])==ds)[0]
            real_3d = outputs['meta_data']['kp_3d'][val_idx].contiguous().cuda()
            real_3d = real_3d[:,self.All54_to_LSP14_mapper].contiguous()
            if (self.All54_to_LSP14_mapper==-1).sum()>0:
                real_3d[:,self.All54_to_LSP14_mapper==-1] = -2.
            
            predicts = outputs['joints_h36m17'][:, :14].contiguous()
            align_inds = [constants.LSP_14['R_Hip'], constants.LSP_14['L_Hip']]

            if args.calc_PVE_error:
               target_theta = torch.cat([outputs['meta_data']['params'][val_idx,:66].cpu(), torch.zeros(len(val_idx),6), outputs['meta_data']['params'][val_idx,66:].cpu()],1)
               pred_theta = torch.cat([outputs['params']['global_orient'], outputs['params']['body_pose'], outputs['params']['betas']],1)
               ED['PVE'][ds]['target_theta'].append(target_theta)
               ED['PVE'][ds]['pred_theta'].append(pred_theta[val_idx].float().detach().cpu())

            abs_error = calc_mpjpe(real_3d, predicts, align_inds=align_inds).float().cpu().numpy()*1000
            rt_error = calc_pampjpe(real_3d, predicts).float().cpu().numpy()*1000
            
            ED['MPJPE'][ds].append(abs_error.astype(np.float32))
            ED['PA_MPJPE'][ds].append(rt_error.astype(np.float32))
            ED['imgpaths'][ds].append(np.array(outputs['meta_data']['imgpath'])[val_idx])
            error_dict['3d']['error'].append(abs_error); error_dict['3d']['idx'].append(val_idx)

        if evaluation and iter_num % self.val_batch_size==0:
            print('{}/{}'.format(iter_num, len(loader_val)))
            MPJPE_result, PA_MPJPE_result, eval_matrix = print_results(self, ED)

    print('Validation on local_rank {}'.format(self.local_rank))
    MPJPE_result, PA_MPJPE_result, eval_matrix = print_results(self, ED, calc_PVE_error=args.calc_PVE_error)

    return MPJPE_result, PA_MPJPE_result, eval_matrix

def print_results(self, ED, calc_PVE_error=False):
    MPJPE_list, PA_MPJPE_list = [], []
    for key, results in ED['MPJPE'].items():
        if len(results)>0:
            MPJPE_list += results
            PA_MPJPE_list += ED['PA_MPJPE'][key]
    MPJPE_result = np.concatenate(MPJPE_list,axis=0).mean()
    PA_MPJPE_result = np.concatenate(PA_MPJPE_list,axis=0).mean()

    eval_matrix = {}
    eval_matrix.update(process_matrix(ED['MPJPE'],'MPJPE'))
    eval_matrix.update(process_matrix(ED['PA_MPJPE'],'PA_MPJPE'))
    print_table(eval_matrix)

    if calc_PVE_error:
        for ds_name in PVE_ds:
            if len(ED['MPJPE'][ds_name])>0:
                eval_matrix['{}-PVE'.format(ds_name)] = np.mean(compute_error_verts(target_theta=torch.cat(ED['PVE'][ds_name]['target_theta'],0), pred_theta=torch.cat(ED['PVE'][ds_name]['pred_theta'],0), smpl_path=os.path.join(self.smpl_model_path, 'smpl'))) * 1000
        print_table(eval_matrix)

    return MPJPE_result, PA_MPJPE_result, eval_matrix

def process_matrix(matrix, name, times=1.):
    eval_matrix = {}
    for ds, error_list in matrix.items():
        if len(error_list)>0:
            result = np.concatenate(error_list,axis=0)
            result = result[~np.isnan(result)].mean()
            eval_matrix['{}-{}'.format(ds,name)] = result*times
    return eval_matrix

def _init_error_dict(dataset_involved, PVE_ds):
    ED = {}
    ED['MPJPE'], ED['PA_MPJPE'], ED['PCK3D'], ED['imgpaths'] = [{ds:[] for ds in dataset_involved} for _ in range(4)]
    ED['PVE'] = {ds:{'target_theta':[], 'pred_theta':[]} for ds in PVE_ds}
    return ED

def print_table(eval_matrix):
    matrix_dict = {}
    em_col_id = 0
    matrix_list = []
    for name in eval_matrix:
        ds,em = name.split('-')
        if em not in matrix_dict:
            matrix_dict[em] = em_col_id
            matrix_list.append(em)
            em_col_id += 1
    
    raw_dict = {}
    for name, result in eval_matrix.items():
        ds,em = name.split('-')
        if ds not in raw_dict:
            raw_dict[ds] = np.zeros(em_col_id).tolist()
        raw_dict[ds][matrix_dict[em]] = '{:.2f}'.format(result)

    table = PrettyTable(['DS/EM']+matrix_list)
    for idx, (ds, mat_list) in enumerate(raw_dict.items()):
        table.add_row([ds]+mat_list)
    print(table)
    print('-'*20)

def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)

def calc_mpjpe(real, pred, align_inds=None, sample_wise=True):
    vis_mask = real[:,:,0] != -2.
    pred_aligned = align_by_parts(pred,align_inds=align_inds)
    real_aligned = align_by_parts(real,align_inds=align_inds)
    mpjpe_each = compute_mpjpe(pred_aligned, real_aligned, vis_mask, sample_wise=sample_wise)
    return mpjpe_each

def calc_pampjpe(real, pred, sample_wise=True,return_transform_mat=False):
    real, pred = real.float(), pred.float()
    # extracting the keypoints that all samples have the annotations
    vis_mask = (real[:,:,0] != -2.).sum(0)==len(real)
    pred_tranformed, PA_transform = batch_compute_similarity_transform_torch(pred[:,vis_mask], real[:,vis_mask])
    pa_mpjpe_each = compute_mpjpe(pred_tranformed, real[:,vis_mask], sample_wise=sample_wise)
    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each