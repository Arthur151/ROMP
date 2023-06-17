import torch 
import copy
import os

delete_output_keys = ['params_pred', 'verts', 'verts_camed_org', 'world_verts', 'world_j3d', 'world_verts_camed_org', 'detection_flag']
def remove_large_keys(outputs, del_keys=delete_output_keys):
    save_outputs = copy.deepcopy(outputs)
    for key in del_keys:
        del save_outputs[key]
    rest_keys = list(save_outputs.keys())
    for key in rest_keys:
        if torch.is_tensor(save_outputs[key]):
            save_outputs[key] = save_outputs[key].detach().cpu().numpy()

    return save_outputs
    
def collect_kp_results(outputs, img_paths):
    seq_kp3d_results = {}
    for ind, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        if img_name not in seq_kp3d_results:
            seq_kp3d_results[img_name] = []
        subject_results = [outputs['pj2d_org'][ind].cpu().numpy(),outputs['j3d'][ind].cpu().numpy(), outputs['pj2d_org_h36m17'][ind].cpu().numpy(),outputs['joints_h36m17'][ind].cpu().numpy(),\
                outputs['smpl_thetas'][ind].cpu().numpy(), outputs['smpl_betas'][ind].cpu().numpy(), outputs['cam_trans'][ind].cpu().numpy()]
        seq_kp3d_results[img_name].append(subject_results)
    return seq_kp3d_results

def insert_last_human_state(current, last_state, key, init=None):
    if key in last_state:
        return torch.cat([last_state[key], current], 0).contiguous()
    if key not in last_state:
        return torch.cat([current[[0]], current], 0).contiguous()

def save_last_human_state(cacher, last_state, key):
    if key not in cacher:
        cacher = {}
    cacher[key] = last_state
    return cacher

def merge_item(source, target, key):
    if key not in target:
        target[key] = source[key].cpu()
    else:
        target[key] = torch.cat([target[key], source[key].cpu()], 0)

def merge_output(split_outputs, seq_outputs):
    keys = ['params_pred', 'reorganize_idx', 'j3d', 'verts', 'verts_camed_org', \
        'world_cams', 'world_trans', 'world_global_rots',  'world_verts', 'world_j3d', 'world_verts_camed_org',\
        'pj2d_org', 'pj2d','cam_trans','detection_flag', 'pj2d_org_h36m17','joints_h36m17', 'center_confs',\
        'track_ids', 'smpl_thetas', 'smpl_betas']
    for key in keys:
        if key in split_outputs:
                merge_item(split_outputs, seq_outputs, key)
    return seq_outputs