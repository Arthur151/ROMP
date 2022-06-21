from unittest import result
from bev import BEV
import argparse
import os, sys
import os.path as osp
import numpy as np
import cv2
import torch
import pickle
from romp import ResultSaver
from romp.utils import progress_bar
from itertools import product

model_id = 2
model_dict = {
    1: '/home/yusun/CenterMesh/trained_models/BEV_Tabs/BEV_ft_agora.pth',
    2: '/home/yusun/CenterMesh/trained_models/BEV.pth',
}
conf_dict = {1:[0.25, 40, 2], 2:[0.1, 40, 1.6]}

visualize_results = False

dataset_dir = '/home/yusun/data_drive/dataset/cmu_panoptic'
model_name = osp.splitext(osp.basename(model_dict[model_id]))[0]
output_save_dir = '/home/yusun/data_drive/evaluation_results/CMU_Panoptic_results/evaluation_{}'.format(model_name)
#if osp.isdir(output_save_dir):
#    import shutil
#    shutil.rmtree(output_save_dir)
os.makedirs(output_save_dir,exist_ok=True)

default_eval_settings = argparse.Namespace(GPU=0, calc_smpl=True, center_thresh=conf_dict[model_id][0], nms_thresh=conf_dict[model_id][1],\
    render_mesh = visualize_results, renderer = 'pyrender', show = False, show_largest = False, \
    input=None, frame_rate=24, temporal_optimize=False, smooth_coeff=3.0, \
    mode='image', model_path = model_dict[model_id], onnx=False, \
    save_path = osp.join(output_save_dir,'visualization'), save_video=False, show_items='mesh', relative_scale_thresh=conf_dict[model_id][2],\
    smpl_path='/home/yusun/.romp/smpla_packed_info.pth', smil_path='/home/yusun/.romp/smil_packed_info.pth')

@torch.no_grad()
def get_results():
    image_folder = osp.join(dataset_dir, 'images')
    file_list = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
    
    model = BEV(default_eval_settings)
    #J_regressor_h36m = torch.load(default_eval_settings.smpl_path)['J_regressor_h36m17']
    results = {}
    if visualize_results:
        saver = ResultSaver(default_eval_settings.mode, default_eval_settings.save_path, save_npz=False)
    for image_path in progress_bar(file_list):
        image = cv2.imread(image_path)
        outputs = model(image)
        if outputs is None:
            continue
        #pred_vertices = outputs['verts'].float()
        #J_regressor_batch = J_regressor_h36m[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        #pred_kp3ds = torch.matmul(J_regressor_batch, pred_vertices)
        results[osp.basename(image_path)] = [outputs['pj2d_org'][:,54:],outputs['joints'][:,54:]]
        if visualize_results:
            saver(outputs, image_path)
    np.savez(osp.join(output_save_dir, 'predictions.npz'), results=results)

def determine_visible_person(kp2ds, width, height):
    visible_person_id,kp2d_vis = [],[]
    for person_id,kp2d in enumerate(kp2ds):
        visible_kps_mask = np.logical_and(np.logical_and(0<kp2d[:,0],kp2d[:,0]<width),np.logical_and(0<kp2d[:,1],kp2d[:,1]<height,kp2d[:,2]>0.2))
        if visible_kps_mask.sum()>5:
            visible_person_id.append(person_id)
            kp2d_vis.append(np.concatenate([kp2d[:,:2], visible_kps_mask[:,None]],1))
    return np.array(visible_person_id), np.array(kp2d_vis)

def load_gts():
    annots_path = osp.join(dataset_dir,'annots.npz')
    annots_folder = osp.join(dataset_dir,'panoptic_annot')
    annots = {}
    for annots_file_name in os.listdir(annots_folder):
        ann_file = os.path.join(annots_folder, annots_file_name)
        with open(ann_file, 'rb') as f:
            img_infos = pickle.load(f)
        for img_info in img_infos:
            img_path = img_info['filename'].split('/')
            img_name = img_path[1]+'-'+img_path[-1].replace('.png', '.jpg')
            annots[img_name] = {}
            annots[img_name] = img_info
    
    new_annots = {}
    J24_TO_H36M = np.array([14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6])
    H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 0] #, 10, 0, 7, 9
    for img_name in annots:
        kp2ds = annots[img_name]['kpts2d'][:,J24_TO_H36M][:,H36M_TO_J14]
        visible_person_id, kp2ds = determine_visible_person(kp2ds, annots[img_name]['width'],annots[img_name]['height'])
        kp3ds = annots[img_name]['kpts3d'][:,J24_TO_H36M][:,H36M_TO_J14][visible_person_id]
        N = len(kp3ds)
        full_kp2d, kp_3ds = np.zeros((N,14,2)), np.zeros((N,14,3))
        for inds, (kp2d, kp3d) in enumerate(zip(kp2ds, kp3ds)):
            kp2d *= 1920./832.
            full_kp2d[inds] = kp2d[:,:2]

            invis_3dkps = kp3d[:,-1]<0.2
            kp3d = kp3d[:,:3]
            kp3d[invis_3dkps] = -2.
            kp3d[:13] += np.array([0,0.06,0.03])
            kp_3ds[inds] = kp3d
        new_annots[img_name] = [full_kp2d, kp_3ds]
    np.savez(annots_path, annots=new_annots)

def get_bbx_overlap(p1, p2, imgpath, baseline=None):
    min_p1 = np.min(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p1 = np.max(p1, axis=0)
    max_p2 = np.max(p2, axis=0)

    bb1 = {}
    bb2 = {}

    bb1['x1'] = min_p1[0]
    bb1['x2'] = max_p1[0]
    bb1['y1'] = min_p1[1]
    bb1['y2'] = max_p1[1]
    bb2['x1'] = min_p2[0]
    bb2['x2'] = max_p2[0]
    bb2['y1'] = min_p2[1]
    bb2['y2'] = max_p2[1]

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou

def l2_error(j1, j2):
    return np.linalg.norm(j1 - j2, 2)

def match_2d_greedy(
        pred_kps,
        gtkp,
        valid_mask,
        imgPath=None,
        baseline=None,
        iou_thresh=0.05,
        valid=None,
        ind=-1):
    '''
    matches groundtruth keypoints to the detection by considering all possible matchings.
    :return: best possible matching, a list of tuples, where each tuple corresponds to one match of pred_person.to gt_person.
            the order within one tuple is as follows (idx_pred_kps, idx_gt_kps)
    '''
    predList = np.arange(len(pred_kps))
    gtList = np.arange(len(gtkp))
    # get all pairs of elements in pred_kps, gtkp
    # all combinations of 2 elements from l1 and l2
    combs = list(product(predList, gtList))

    errors_per_pair = {}
    errors_per_pair_list = []
    for comb in combs:
        vmask = valid_mask[comb[1]]
        assert vmask.sum()>0, print('no valid points')
        errors_per_pair[str(comb)] = l2_error(
            pred_kps[comb[0]][vmask, :2], gtkp[comb[1]][vmask, :2])
        errors_per_pair_list.append(errors_per_pair[str(comb)])

    gtAssigned = np.zeros((len(gtkp),), dtype=bool)
    opAssigned = np.zeros((len(pred_kps),), dtype=bool)
    errors_per_pair_list = np.array(errors_per_pair_list)

    bestMatch = []
    excludedGtBecauseInvalid = []
    falsePositiveCounter = 0
    while np.sum(gtAssigned) < len(gtAssigned) and np.sum(
            opAssigned) + falsePositiveCounter < len(pred_kps):
        found = False
        falsePositive = False
        while not(found):
            if sum(np.inf == errors_per_pair_list) == len(
                    errors_per_pair_list):
                print('something went wrong here')

            minIdx = np.argmin(errors_per_pair_list)
            minComb = combs[minIdx]
            # compute IOU
            iou = get_bbx_overlap(
                pred_kps[minComb[0]], gtkp[minComb[1]], imgPath, baseline)
            # if neither prediction nor ground truth has been matched before and iou
            # is larger than threshold
            if not(opAssigned[minComb[0]]) and not(
                    gtAssigned[minComb[1]]) and iou >= iou_thresh:
                #print(imgPath + ': found matching')
                found = True
                errors_per_pair_list[minIdx] = np.inf
            else:
                errors_per_pair_list[minIdx] = np.inf
                # if errors_per_pair_list[minIdx] >
                # matching_threshold*headBboxs[combs[minIdx][1]]:
                if iou < iou_thresh:
                    #print(
                    #   imgPath + ': false positive detected using threshold')
                    found = True
                    falsePositive = True
                    falsePositiveCounter += 1

        # if ground truth of combination is valid keep the match, else exclude
        # gt from matching
        if not(valid is None):
            if valid[minComb[1]]:
                if not falsePositive:
                    bestMatch.append(minComb)
                    opAssigned[minComb[0]] = True
                    gtAssigned[minComb[1]] = True
            else:
                gtAssigned[minComb[1]] = True
                excludedGtBecauseInvalid.append(minComb[1])

        elif not falsePositive:
            # same as above but without checking for valid
            bestMatch.append(minComb)
            opAssigned[minComb[0]] = True
            gtAssigned[minComb[1]] = True

    bestMatch = np.array(bestMatch)
    # add false positives and false negatives to the matching
    # find which elements have been successfully assigned
    opAssigned = []
    gtAssigned = []
    for pair in bestMatch:
        opAssigned.append(pair[0])
        gtAssigned.append(pair[1])
    opAssigned.sort()
    gtAssigned.sort()

    falsePositives = []
    misses = []

    # handle false positives
    opIds = np.arange(len(pred_kps))
    # returns values of oIds that are not in opAssigned
    notAssignedIds = np.setdiff1d(opIds, opAssigned)
    for notAssignedId in notAssignedIds:
        falsePositives.append(notAssignedId)
    gtIds = np.arange(len(gtList))
    # returns values of gtIds that are not in gtAssigned
    notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned)

    # handle false negatives/misses
    for notAssignedIdGt in notAssignedIdsGt:
        if not(valid is None):  # if using the new matching
            if valid[notAssignedIdGt]:
                #print(imgPath + ': miss')
                misses.append(notAssignedIdGt)
            else:
                excludedGtBecauseInvalid.append(notAssignedIdGt)
        else:
            #print(imgPath + ': miss')
            misses.append(notAssignedIdGt)

    return bestMatch, falsePositives, misses  # tuples are (idx_pred_kps, idx_gt_kps)

# following the code of coherece reconstruction of multiperson Jiang et. al.
# Brought from https://github.com/JiangWenPL/multiperson/blob/4d3dbae945e22bb1e270521b061a837976699685/mmdetection/mmdet/core/utils/eval_utils.py#L265

def evaluation_results():
    annots = np.load(osp.join(dataset_dir,'annots.npz'), allow_pickle=True)['annots'][()]
    results = np.load(osp.join(output_save_dir,'predictions.npz'), allow_pickle=True)['results'][()]
    action_name = ['haggling', 'mafia', 'ultimatum', 'pizza']
    mpjpe_cacher = {aname:[] for aname in action_name}
    
    H36M17_TO_J14 = [0,1,2,3, 4,5,6,7,8 ,9,10,11,12,14]
    missing_punish = 150

    for img_name in progress_bar(annots):
        kp2d_gts, kp3d_gts = annots[img_name]
        root_gts = kp3d_gts[:,[13]]
        visible_kpts = kp3d_gts[:,:,0]>-2.
        valid_mask = kp2d_gts[:,:,0]>-2.
        valid_ids =  valid_mask.sum(-1) != 0
        kp2d_gts, kp3d_gts = kp2d_gts[valid_ids], kp3d_gts[valid_ids] - root_gts[valid_ids]
        valid_mask, visible_kpts = valid_mask[valid_ids], visible_kpts[valid_ids]
        if img_name in results:
            kp2d_preds, kp3d_preds = results[img_name]
            kp2d_preds, kp3d_preds = kp2d_preds[:,H36M17_TO_J14], kp3d_preds[:,H36M17_TO_J14] - kp3d_preds[:,[14]]
            
            bestMatch, falsePositives, misses = match_2d_greedy(kp2d_preds, kp2d_gts, valid_mask)
            bestMatch = np.array(bestMatch)
            if len(bestMatch)>0:
                pids, gids = bestMatch[:,0], bestMatch[:,1]
                #kp3d_preds[pids] -= root_gts[gids]
                #kp3d_gts -= root_gts
                mpjpes = (np.sqrt(((kp3d_preds[pids] - kp3d_gts[gids]) ** 2).sum(-1)) * visible_kpts[gids]) *1000
                mpjpes = np.concatenate([mpjpes.mean(-1), np.ones(len(misses)) * missing_punish])
            else:
                mpjpes = np.ones(len(kp3d_gts)) * missing_punish
        else:
            mpjpes = np.ones(len(kp3d_gts)) * missing_punish

        for mpjpe in mpjpes:
            for aname in action_name:
                if aname in os.path.basename(img_name):
                    mpjpe_cacher[aname].append(float(mpjpe.item()))
        
    print('Final results:')
    avg_all = []
    for key,value in mpjpe_cacher.items():
        mean = sum(value)/len(value)
        print(key, mean)
        avg_all.append(value)
    print('MPJPE results:', np.concatenate(avg_all).mean())

if __name__ == '__main__':
    get_results()
    #load_gts()
    evaluation_results()
