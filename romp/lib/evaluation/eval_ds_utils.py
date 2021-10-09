import sys,os
import torch
import numpy as np

def cmup_evaluation_act_wise(results,imgpaths,action_names):
    actions = []
    action_results = []
    for imgpath in imgpaths:
        actions.append(os.path.basename(imgpath).split('-')[0].split('_')[1])

    for action_name in action_names:
        action_idx = np.where(np.array(actions)==action_name)[0]
        action_results.append('{:.2f}'.format(results[action_idx].mean()))
    return action_results

def h36m_evaluation_act_wise(results,imgpaths,action_names):
    actions = []
    action_results = []
    for imgpath in imgpaths:
        actions.append(os.path.basename(imgpath).split('.jpg')[0].split('_')[1].split(' ')[0])

    for action_name in action_names:
        action_idx = np.where(np.array(actions)==action_name)[0]
        action_results.append('{:.2f}'.format(results[action_idx].mean()))
    return action_results

def pp_evaluation_cam_wise(results,imgpaths):
    cam_ids = []
    cam_results = []
    for imgpath in imgpaths:
        cam_ids.append(int(os.path.basename(imgpath).split('_')[1]))
    #22 is missing
    for camid in list(range(21))+list(range(22,31)):
        cam_idx = np.where(np.array(cam_ids)==camid)[0]
        cam_results.append('{:.2f}'.format(results[cam_idx].mean()))
    return cam_results

def determ_worst_best(VIS_IDX,top_n=2):
    sellected_ids, sellected_errors = [], []
    if VIS_IDX is not None:
        for ds_type in VIS_IDX:
            for error, idx in zip(VIS_IDX[ds_type]['error'], VIS_IDX[ds_type]['idx']):
                if torch.is_tensor(error):
                    error, idx = error.cpu().numpy(), idx.cpu().numpy()
                worst_id = np.argsort(error)[-top_n:]
                sellected_ids.append(idx[worst_id]); sellected_errors.append(error[worst_id])
                best_id = np.argsort(error)[:top_n]
                sellected_ids.append(idx[best_id]); sellected_errors.append(error[best_id])
    if len(sellected_ids)>0 and len(sellected_errors)>0:
        sellected_ids = np.concatenate(sellected_ids).tolist()
        sellected_errors = np.concatenate(sellected_errors).tolist()
    else:
        sellected_ids, sellected_errors = [0], [0]
    return sellected_ids, sellected_errors

def reorganize_vis_info(vis_ids, vis_errors, org_imgpath, new_imgpath):
    vis_ids_new, vis_errors_new = [], []
    org_imgpath_dict = {}
    for vis_id, vis_error in zip(vis_ids, vis_errors):
        imgpath = org_imgpath[vis_id]
        if imgpath not in org_imgpath_dict:
            org_imgpath_dict[imgpath] = []
        org_imgpath_dict[imgpath].append(vis_error)

    new_imgpath = np.array(new_imgpath)
    for imgpath, errors in org_imgpath_dict.items():
        for new_idx in np.where(new_imgpath==imgpath)[0]:
            vis_ids_new.append(new_idx)
            if len(errors) == 0:
                vis_errors_new.append(0)
            else:
                vis_errors_new.append(max(errors))
                errors.remove(max(errors))
    return vis_ids_new, vis_errors_new