import numpy as np
np.set_printoptions(precision=3, suppress=True)
import os
import pickle
import torch
from .utils import glamr_mapping2D

def process_idx(reorganize_idx, vids=None):
    result_size = reorganize_idx.shape[0]
    reorganize_idx = reorganize_idx
    used_org_inds = np.unique(reorganize_idx)
    per_img_inds = [np.where(reorganize_idx==org_idx)[0] for org_idx in used_org_inds]

    return used_org_inds, per_img_inds

def load_gts(world_annots_path):
    annots = np.load(world_annots_path, allow_pickle=True)['annots'][()]
    seq_names = list(annots.keys())
    seq_names.remove('sequence_dict')
    seq_names.remove('ID_num')
    return annots, seq_names

def load_glamr_seq_results(folder, results_folder):
    out_file = os.path.join(results_folder, folder+'_seed1.pkl')
    if not os.path.exists(out_file):
        return None
    results = pickle.load(open(out_file, 'rb'))
    person_num = len(results['person_data'])
    person_ids = list(results['person_data'].keys())
    #print(os.path.basename(out_file), 'has subjects', person_ids)
    glamr_results = {}
    for person_id in person_ids:
        frame2ind = results['person_data'][person_id]['frame2ind']
        kp_2d_pred = results['person_data'][person_id]['kp_2d_pred']
        root_trans_world = results['person_data'][person_id]['root_trans_world']
        smpl_orient_world = results['person_data'][person_id]['smpl_orient_world']
        glamr_results[person_id] = (frame2ind, kp_2d_pred, root_trans_world, smpl_orient_world)
    return glamr_results

def load_single_bev_dpvo_resutls(seq_names, predicts_folder):
    results = {}
    for seq_name in seq_names:
        results[seq_name] = {}
        results_path = os.path.join(predicts_folder, seq_name+'.npz')
        if not os.path.exists(results_path):
            results[seq_name] = None
            continue
        outputs = np.load(results_path, allow_pickle=True)['results'][()]
        frame2ind = {fid:ind+1 for ind, fid in enumerate(np.where(outputs['frame2ind']!=0)[0])}
        frame2ind[0] = 0
        results[seq_name][0] = [frame2ind, outputs['pj2d_org'], outputs['root_trans_world'], outputs['smpl_orient_world']]
    return results

def load_glamr_eval_resutls(seq_names, results_folder):
    results = {seq_name: load_glamr_seq_results(seq_name, results_folder) for seq_name in seq_names}
    return results      


def load_eval_resutls(predicts_folder, seq_names):
    results = {}
    for seq_name in seq_names:
        results[seq_name] = {}
        outputs = np.load(os.path.join(predicts_folder, seq_name+'.npz'), allow_pickle=True)['outputs'][()]

        used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])

        frame2ind = {fid:inds[0] for fid, inds in zip(used_org_inds, per_img_inds)}
        fovs = np.ones(len(outputs['world_trans'])) * 50
        if torch.is_tensor(outputs['pj2d_org']):
            results[seq_name][0] = [frame2ind, outputs['pj2d_org'].numpy(), outputs['world_trans'].numpy(), outputs['world_global_rots'].numpy()]
        else:
            results[seq_name][0] = [frame2ind, outputs['pj2d_org'], outputs['world_trans'], outputs['world_global_rots']]
        
    return results

def load_mutli_eval_resutls(results_folder, seq_names):
    results = {}
    for seq_name in seq_names:
        results[seq_name] = {}
        outputs = np.load(os.path.join(results_folder, seq_name+'.npz'), allow_pickle=True)['outputs'][()]
        used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])
        
        frame2ind = {fid:np.array(inds) for fid, inds in zip(used_org_inds, per_img_inds)}
        results[seq_name] = [frame2ind, outputs['pj2d_org'], outputs['world_trans'], outputs['world_global_rots']]
        
    return results

def load_mutli_eval_bev_dpvo_resutls(seq_names, predicts_folder):
    results = {}
    for seq_name in seq_names:
        results[seq_name] = {}
        results_path = os.path.join(predicts_folder, seq_name+'.npz')
        if not os.path.exists(results_path):
            results[seq_name] = None
            continue
        outputs = np.load(results_path, allow_pickle=True)['results'][()]
        frame2ind = {fid:ind+1 for ind, fid in enumerate(np.where(outputs['frame2ind']!=0)[0])}
        frame2ind[0] = 0
        results[seq_name] = [frame2ind, outputs['pj2d_org'], outputs['root_trans_world'], outputs['smpl_orient_world']]
    return results

def load_glamr_seq_multiperson_results(seq_name, results_folder):
    out_file = os.path.join(results_folder, seq_name+'_seed1.pkl')
    if not os.path.exists(out_file):
        return None
    results = pickle.load(open(out_file, 'rb'))
    person_num = len(results['person_data'])
    person_ids = list(results['person_data'].keys())
    #print(os.path.basename(out_file), 'has subjects', person_ids)
    glamr_results = {}
    for person_id in person_ids:
        frame2ind = results['person_data'][person_id]['frame2ind']
        kp_2d_pred = results['person_data'][person_id]['kp_2d_pred'][:,glamr_mapping2D]
        kp_2d_pred[:,glamr_mapping2D==-1] = -2.
        #root_trans_cam = results['person_data'][person_id]['root_trans_cam']
        root_trans_world = results['person_data'][person_id]['root_trans_world']
        smpl_orient_world = results['person_data'][person_id]['smpl_orient_world']
        #print(person_id, kp_2d_pred.shape, root_trans_world.shape,  smpl_orient_world.shape)
        glamr_results[person_id] = (frame2ind, kp_2d_pred, root_trans_world, smpl_orient_world)

    frame2ind = {}
    new_ind = 0
    kp2ds, world_trans, world_grots = [], [], []
    for person_id in glamr_results:
        for key, org_ind in glamr_results[person_id][0].items():
            if key not in frame2ind:
                frame2ind[key] = []
            kp2ds.append(glamr_results[person_id][1][org_ind])
            world_trans.append(glamr_results[person_id][2][org_ind])
            world_grots.append(glamr_results[person_id][3][org_ind])
            frame2ind[key].append(new_ind)
            new_ind += 1
    return [frame2ind, np.array(kp2ds), np.array(world_trans), np.array(world_grots)]

def load_glamr_multiperson_results(seq_names, results_folder):
    results = {seq_name: load_glamr_seq_multiperson_results(seq_name, results_folder) for seq_name in seq_names}
    return results     