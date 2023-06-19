import numpy as np
np.set_printoptions(precision=3, suppress=True)
import os
import glob

from .dynacam_evaluation.loading_data import load_gts, load_eval_resutls, load_mutli_eval_resutls
from .dynacam_evaluation.evalute_ate import evaluate_ate
from .dynacam_evaluation.utils import search_valid_frame, mat2angle, angle2mat, angle2quaternion

def eval_single(preds, gts, pano_frame_dir, seq_names, vis=True, vis_folder='traj_vis', missing_punish=[2,4]):
    errors = {'ate':{}, 'ape':{}}
    for seq_name in seq_names:
        if preds[seq_name] is None:
            errors['ate'][seq_name] = missing_punish[0]
            errors['ape'][seq_name] = missing_punish[1]
            continue
        frame2ind, kp_2d_pred, root_trans_world, root_rot_world = preds[seq_name][0]
        world_annots = gts[seq_name]
        gtran_gts, grot_gts = [], []
        gtran_preds, grot_preds = [], []
        
        frame_ids = world_annots['frame_ids']
        frame_names = sorted([os.path.basename(path) for path in glob.glob(os.path.join(pano_frame_dir, seq_name, '*.jpg'))])
        clip_frames = np.array([int(name.replace('.jpg', '')) for name in frame_names])
        clip_frame_ids = np.array([np.where(clip_frames==fid)[0][0] for fid in frame_ids])
        #print(clip_frame_ids, frame2ind)
        used_frame_ids = []
        for gid, frame_id in enumerate(clip_frame_ids):
            grot_gt, gtran_gt = world_annots['world_grots'][0, gid], world_annots['world_trans'][0, gid]
            grot_gts.append(grot_gt)
            gtran_gts.append(gtran_gt)
            if frame_id not in frame2ind:
                frame_id = search_valid_frame(frame2ind, frame_id)
            rid = frame2ind[frame_id]
            grot_preds.append(root_rot_world[rid])
            pred_world_trans = root_trans_world[rid]
            gtran_preds.append(pred_world_trans)
            used_frame_ids.append(frame_id)
        used_frame_ids = np.array(used_frame_ids)
        gtran_gts = np.array(gtran_gts)
        gtran_preds = np.array(gtran_preds)

        # aligning to the first-frame coordinates.
        extrinsic = world_annots['camera_extrinsics'][0]
        gtran_gts = np.matmul(extrinsic[:3,:3][None], gtran_gts[:,:,None])[:,:,0]
        grot_gts = np.array([mat2angle(np.matmul(extrinsic[:3,:3], angle2mat(grot))) for grot in grot_gts])
        
        #gtran_preds = gtran_preds[:,[0,2,1]]
        gtran_gts = gtran_gts - gtran_gts[[0]]
        gtran_preds = gtran_preds - gtran_preds[[0]]
        grot_gts = np.array([angle2quaternion(grot) for grot in grot_gts])
        grot_preds = np.array([angle2quaternion(grot) for grot in grot_preds])

        traj_est = np.concatenate([gtran_preds, grot_preds], 1)
        traj_ref = np.concatenate([gtran_gts, grot_gts], 1)
        timestamps = used_frame_ids.astype(np.float32) / 30
        ate, ape = evaluate_ate(traj_est, traj_ref, timestamps, seq_name, show_results=vis, vis_folder=vis_folder)
        errors['ate'][seq_name] = ate
        errors['ape'][seq_name] = ape

    print('ATE:', np.array(list(errors['ate'].values())).mean())
    print('APE:', np.array(list(errors['ape'].values())).mean())

def match_kp2ds(preds, gts):
    match_ids = []
    for gt in gts:
        valid_mask = gt[:,0]>0
        dists = np.linalg.norm(preds[:,valid_mask] - gt[valid_mask][None], ord=2, axis=-1).mean(-1)
        match_id = np.argmin(dists)
        #if dists[match_id] > 15:
        #    print(min(dists))
        match_ids.append(match_id)
    return np.array(match_ids)

def eval_multi(preds, allgts, seq_names, vis=True,vis_folder='traj_vis',  missing_punish=[2,4]):
    errors = {'ate':{}, 'ape':{}}
    for seq_name in seq_names:
        if preds[seq_name] is None:
            errors['ate'][seq_name] = missing_punish[0]
            errors['ape'][seq_name] = missing_punish[1]
            continue
        frame2ind, kp_2d_pred, root_trans_world, root_rot_world = preds[seq_name]
        gts = allgts[seq_name]
        gtran_gts, grot_gts = [], []
        gtran_preds, grot_preds = [], []
        
        frame_ids = gts['frame_ids']
        frame_names = sorted([os.path.basename(path) for path in glob.glob(os.path.join(tran_frame_dir, seq_name, '*.png'))])
        clip_frames = np.array([int(name.replace('.png', '').replace('.jpg', '')) for name in frame_names])
        clip_frame_ids = np.array([np.where(clip_frames==fid)[0][0] for fid in frame_ids])
        used_frame_ids = []
        for gid, frame_id in enumerate(clip_frame_ids):
            kp2d_gts = gts['kp2ds'][:,gid, :, :2]
            grot_gt, gtran_gt = gts['world_grots'][:,gid], gts['world_trans'][:,gid]

            grot_gts.append(grot_gt)
            gtran_gts.append(gtran_gt)
            if frame_id not in frame2ind:
                #print(frame_id,'missing')
                while frame_id > 0:
                    frame_id = frame_id-1
                    if frame_id in frame2ind:
                        break
            rid = frame2ind[frame_id]
            if isinstance(rid, int):
                pred_kp2ds = kp_2d_pred[rid]
                if len(pred_kp2ds.shape)==2:
                    pred_kp2ds = pred_kp2ds[None]
                if len(root_rot_world[rid].shape)<2:
                    root_rot_world[rid] = root_rot_world[rid].reshape((1,3))
                if len(root_trans_world[rid].shape)<2:
                    root_trans_world[rid] = root_trans_world[rid].reshape((1,3))
                match_ids = match_kp2ds(pred_kp2ds[:,:24], kp2d_gts[:,:24]) #args().joint_num
                grot_preds.append(root_rot_world[rid][match_ids])
                pred_world_trans = root_trans_world[rid][match_ids]
            else:
                rid = np.array(frame2ind[frame_id])
                if len(rid) > 1:
                    match_ids = match_kp2ds(kp_2d_pred[rid], kp2d_gts)
                    rid = rid[match_ids]
                grot_preds.append(root_rot_world[rid])
                pred_world_trans = root_trans_world[rid]

            gtran_preds.append(pred_world_trans)
            used_frame_ids.append(frame_id)
        used_frame_ids = np.array(used_frame_ids)
        gtran_gts = np.stack(gtran_gts)
        #print([pred.shape for pred in gtran_preds])
        gtran_preds = np.stack(gtran_preds)
        if gtran_preds.shape[1] > gtran_preds.shape[0]:
            gtran_preds = gtran_preds.transpose((1,0))
        
        grot_gts = np.array([angle2quaternion(grot) for grot in grot_gts])
        grot_preds = np.array([angle2quaternion(grot) for grot in grot_preds])
        person_num = gtran_gts.shape[1]
        
        for sid in range(person_num):
            traj_est = np.concatenate([gtran_preds[:,sid], grot_preds[:,sid]], 1)
            traj_ref = np.concatenate([gtran_gts[:,sid], grot_gts[:,sid]], 1)
            timestamps = used_frame_ids.astype(np.float32) / 30
            try:
                vis_seq_name = seq_name+str(sid)
                ate, ape = evaluate_ate(traj_est, traj_ref, timestamps, vis_seq_name, align=True, show_results=vis, vis_folder=vis_folder)
                errors['ate'][vis_seq_name] = ate
                errors['ape'][vis_seq_name] = ape
            except:
                vis_seq_name = seq_name+str(sid)
                #print(vis_seq_name, 'Failed during alingment, evaluate this without scale alignment')
                ate, ape = evaluate_ate(traj_est, traj_ref, timestamps, vis_seq_name, align=False, show_results=vis, vis_folder=vis_folder)
                errors['ate'][vis_seq_name] = ate
                errors['ape'][vis_seq_name] = ape
    
    print('ATE:', np.array(list(errors['ate'].values())).mean())
    print('APE:', np.array(list(errors['ape'].values())).mean())
        
def evaluate_panorama(results_folder, root_dir, vis=True):
    world_annots_path = os.path.join(root_dir, 'annotations', 'panorama_test.npz')
    pano_frame_dir = os.path.join(root_dir, 'video_frames', 'panorama_test')
    gts, seq_names = load_gts(world_annots_path)
    results = load_eval_resutls(results_folder, seq_names)
    eval_single(results, gts, pano_frame_dir, seq_names, vis=vis, vis_folder='pano_traj_vis')

def evaluate_translation(results_folder, root_dir, vis=True):
    world_annots_path = os.path.join(root_dir, 'annotations', 'translation_test.npz')
    gts, seq_names = load_gts(world_annots_path)
    results = load_mutli_eval_resutls(results_folder, seq_names)
    eval_multi(results, gts, seq_names, vis=vis, vis_folder='trans_traj_vis')

