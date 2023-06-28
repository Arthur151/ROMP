import os
import numpy as np
import cv2
import torch
from itertools import product
from evaluation.smpl import SMPL, SMPLA_parser
from evaluation.evaluation_matrix import compute_error_accel_np
import glob
import json
import tqdm
from utils.rotation_transform import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

dataset_dir = '/home/yusun/DataCenter/datasets/3DMPB_full'

annots_save_path = os.path.join(dataset_dir, 'annots.npz')

LSP_14 = {
    'R_Ankle':0, 'R_Knee':1, 'R_Hip':2, 'L_Hip':3, 'L_Knee':4, 'L_Ankle':5, 'R_Wrist':6, 'R_Elbow':7, \
    'R_Shoulder':8, 'L_Shoulder':9, 'L_Elbow':10, 'L_Wrist':11, 'Neck_LSP':12, 'Head_top':13}

SMPL_24 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'SMPL_Head':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23}
SMPL_Face_Foot_11 = {
    'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28, \
    'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34}
SMPL_EXTRA_9 = {
    'R_Hip': 35, 'L_Hip':36, 'Neck_LSP':37, 'Head_top':38, 'Pelvis':39, 'Thorax_MPII':40, \
    'Spine_H36M':41, 'Jaw_H36M':42, 'Head':43}
SMPL_ALL_44 = {**SMPL_24, **SMPL_Face_Foot_11, **SMPL_EXTRA_9}

lsp14_connMat = np.array([[ 0, 1 ],[ 1, 2 ],[ 3, 4 ],[ 4, 5 ],[ 6, 7 ],[ 7, 8 ],[ 8, 2 ],[ 8, 9 ],[ 9, 3 ],[ 2, 3 ],[ 8, 12],[ 9, 10],[12, 9 ],[10, 11],[12, 13]])

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format),dtype=np.int32)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)

all2lsp = joint_mapping(SMPL_ALL_44, LSP_14)

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
        iou_thresh=0.15,
        valid=None,
        ind=-1):
    '''
    matches groundtruth keypoints to the detection by considering all possible matchings.
    :return: best possible matching, a list of tuples, where each tuple corresponds to one match of pred_person.to gt_person.
            the order within one tuple is as follows (idx_pred_kps, idx_gt_kps)
    '''
    # get all pairs of elements in pred_kps, gtkp
    # all combinations of 2 elements from l1 and l2
    combs = list(product(np.arange(len(pred_kps)), np.arange(len(gtkp))))

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
    gtIds = np.arange(len(gtkp))
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

def _calc_MPJPE_(kp3d_preds, kp3d_gts):
    mpjpes = (np.sqrt(((kp3d_preds - kp3d_gts) ** 2).sum(-1))).mean(-1) *1000
    return mpjpes

def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)
    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))
    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = torch.linalg.svd(K)
    V = Vh.transpose(-2,-1)
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))
    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))
    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))
    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat, (scale, R, t)

def _calc_PAMPJPE_(kp3d_preds, kp3d_gts):
    kp3d_preds_tensor, kp3d_gts_tensor = torch.from_numpy(kp3d_preds).float(), torch.from_numpy(kp3d_gts).float()
    pred_tranformed, PA_transform = batch_compute_similarity_transform_torch(kp3d_preds_tensor, kp3d_gts_tensor)
    pa_mpjpe_each = _calc_MPJPE_(pred_tranformed.numpy(), kp3d_gts)
    return pa_mpjpe_each

def _calc_PVE_(verts_gts,verts_preds):
    batch_PVE = torch.norm(verts_gts-verts_preds, p=2, dim=-1).mean(-1) *1000
    return batch_PVE

def collect_sequence_gts():
    annots = np.load(annots_save_path, allow_pickle=True)['annots'][()]
    smpl_male = SMPL(os.path.join(os.path.expanduser("~"),'.romp','SMPL_MALE.pth'))
    smpl_female = SMPL(os.path.join(os.path.expanduser("~"),'.romp','SMPL_FEMALE.pth'))

def convert2kp3d_sequence(sequence_kp3ds):
    gt_kp3d_sequence, pred_kp3d_sequence = [], []
    video_names = list(sequence_kp3ds.keys())
    for video_name in video_names:
        frame_names = sorted(list(sequence_kp3ds[video_name].keys()))
        frame_ids = np.array([int(frame_name.replace('image_', '').replace('.jpg', '')) for frame_name in frame_names])
        subject_ids = [sequence_kp3ds[video_name][frame_name]['subject_id'] for frame_name in frame_names]
        subject_num = len(np.unique(np.concatenate(subject_ids, 0)))
        sequence_length = frame_ids.max() + 1
        gt_kp3d_seq, pred_kp3d_seq = np.zeros((subject_num, sequence_length, 14, 3))-2, np.zeros((subject_num, sequence_length, 14, 3))-2
        #gt_kp3d_seq[subject_ids, frame_ids] = np.array([sequence_kp3ds[video_name][frame_name]['gts'] for frame_name in frame_names])
        #pred_kp3d_seq[subject_ids, frame_ids] = np.array([sequence_kp3ds[video_name][frame_name]['preds'] for frame_name in frame_names])
        for sids, frame_name, fid in zip(subject_ids, frame_names, frame_ids):
            gt_kp3d_seq[sids, fid] = sequence_kp3ds[video_name][frame_name]['gts']
            pred_kp3d_seq[sids, fid] = sequence_kp3ds[video_name][frame_name]['preds']
        gt_kp3d_sequence.append(gt_kp3d_seq)
        pred_kp3d_sequence.append(pred_kp3d_seq)

    return gt_kp3d_sequence, pred_kp3d_sequence

def rectify_grot_with_fov_and_pitch(grots, rx=8):
    rx_transform_axis_angle = torch.Tensor([[np.radians(rx),0,0]])
    R = angle_axis_to_rotation_matrix(rx_transform_axis_angle).repeat(len(grots),1,1)
    grots_mat = angle_axis_to_rotation_matrix(torch.from_numpy(grots))
    grots_r = R.bmm(grots_mat)
    grots_new = rotation_matrix_to_angle_axis(grots_r)
    return grots_new.numpy()

hard_seq = ['downtown_stairs_00','downtown_weeklyMarket_00', 'downtown_sitOnStairs_00', 'downtown_runForBus_01', 'downtown_warmWelcome_00', 'flat_guitar_01']

class Evaluator(object):
    def __init__(self):
        self.smpl_male = SMPL(os.path.join(os.path.expanduser("~"),'.romp','SMPL_MALE.pth'))
        self.smpl_female = SMPL(os.path.join(os.path.expanduser("~"),'.romp','SMPL_FEMALE.pth'))
        self.smpl_neutral = SMPL(os.path.join(os.path.expanduser("~"),'.romp','SMPL_NEUTRAL.pth'))
        self.smpla_model = SMPLA_parser(os.path.join(os.path.expanduser("~"),'.romp','SMPLA_NEUTRAL.pth'), os.path.join(os.path.expanduser("~"),'.romp','smil_packed_info.pth'))

        self.J_regressor = torch.load(os.path.join(os.path.expanduser("~"),'.romp','SMPL_MALE.pth'))['J_regressor_h36m17'][None, :]
        #joint_mapper_h36m = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9][:14]
    
    def acquire_verts_kp3ds_with_gender(self, theta_gts, beta_gts, genders):
        # Brought from https://github.com/syguan96/DynaBOA/blob/2ce049c70f27de8aa488cb2b2c04366b6b332141/dynaboa_benchmark.py#L220
        J_regressor_batch = self.J_regressor.expand(theta_gts.shape[0], -1, -1)
        verts_gts, kp3d_gts, _ = self.smpl_male(betas=beta_gts, poses=theta_gts)
        verts_gts_female, kp3d_gts_female, _ = self.smpl_female(betas=beta_gts, poses=theta_gts)
        verts_gts[genders == 1, :, :] = verts_gts_female[genders == 1, :, :]
        kp3d_gts[genders == 1, :, :] = kp3d_gts_female[genders == 1, :, :]

        #kp3d_gts = kp3d_gts[:,:24] - kp3d_gts[:, [0], :]
        kp3d_gts = torch.matmul(J_regressor_batch.to(verts_gts.device), verts_gts)
        root_gts = kp3d_gts[:, [14], :]#.mean(1,keepdim=True) # 14 is pelvis
        kp3d_gts = kp3d_gts[:,:14] - root_gts #
        kp3d_gts = kp3d_gts.numpy()
        return verts_gts, kp3d_gts
    
    def acquire_verts_kp3ds_natural(self, theta_preds, beta_preds):
        if beta_preds.shape[-1]>10:
            verts_preds, _, _ = self.smpla_model(thetas=theta_preds, betas=beta_preds, root_align=False)
        else:
            verts_preds, _, _ = self.smpl_neutral(poses=theta_preds, betas=beta_preds)
        
        kp3d_preds = torch.matmul(self.J_regressor.expand(verts_preds.shape[0], -1, -1).to(verts_preds.device), verts_preds)
        root_preds = kp3d_preds[:, [14], :]#.mean(1,keepdim=True) # 14 is pelvis
        kp3d_preds = kp3d_preds[:,:14] - root_preds
        kp3d_preds = kp3d_preds.numpy()

        return verts_preds, kp3d_preds

def make_axis_angle_between_0_to_2pi(axis_angle, axis=np.array([0])):
    # TODO: current assumpts is wrong!!! not to make it right, this is important.
    axis_angle[:,axis] = axis_angle[:,axis] % (2*np.pi)
    axis_angle[:,axis][axis_angle[:,axis]<0] = axis_angle[:,axis][axis_angle[:,axis]<0] + 2*np.pi
    return axis_angle


def load_gts():
    annot_path = os.path.join(dataset_dir, 'annot.json')
    with open(annot_path, 'r') as f: 
        org_annots = json.load(f)
    annots = {}

    for ind, org_annot in enumerate(org_annots):
        image_info = org_annot['img_file']
        _, seq1, seq2, image_name = image_info.split("\\")
        seq_name = f"{seq1}-{seq2}"
        if seq_name not in annots:
            annots[seq_name] = {}
        bboxes, kp2ds, kp_vis, kp3ds, params, trans = [], [], [], [], [], []
        for annot in org_annot['annotations']:
            bboxes.append(annot['bbox'])
            kp2ds.append(annot['lsp_joints_2d'])
            kp_vis.append(annot['vis'])
            kp3ds.append(annot['lsp_joints_3d'])
            params.append(np.concatenate([np.array(annot['pose']), np.array(annot['betas'])],0))
            trans.append(annot['trans'])
        bboxes, kp2ds, kp_vis, kp3ds, params, trans = \
            np.array(bboxes), np.array(kp2ds), np.array(kp_vis), np.array(kp3ds), np.array(params), np.array(trans)
        annots[seq_name][image_name] = [ind, bboxes, kp2ds, kp_vis, kp3ds, params, trans, org_annot['intri']]
    np.savez(annots_save_path, annots=annots)
    return annots


def evaluate_3dmpb_results(kp3d_results, debug=False, seq_wise_results=True, eval_hard_seq=False, grot_rx_rectify=False):
    annots = np.load(annots_save_path, allow_pickle=True)['annots'][()]

    matrix_results = {'3dmpb-MPJPE': [], '3dmpb-PA_MPJPE': []}
    missing_punish = 150
    missed_frames = {}
    missed_subjects = []

    evaluator = Evaluator()

    sequence_kp3ds = {}
    sequence_errors = {}
    for video_name in tqdm.tqdm(annots):
        for frame_name in sorted(annots[video_name]):
            if video_name not in sequence_kp3ds:
                sequence_kp3ds[video_name] = {}
                sequence_errors[video_name] = {'MPJPE': [], 'PA_MPJPE': []}

            kp2d_gts = annots[video_name][frame_name][2]
            kp3d_gts = annots[video_name][frame_name][4]
            kp2d_vis = annots[video_name][frame_name][3]
            trans = annots[video_name][frame_name][6]
            kp3d_gts = kp3d_gts - kp3d_gts[:,[2,3]].mean(1)[:,None]
            
            if frame_name not in kp3d_results[video_name]:
                MPJPEs = PA_MPJPEs = np.ones(len(kp3d_gts)) * missing_punish
                bestMatch = []
            else:
                kp2d_preds = np.stack([result[0] for result in kp3d_results[video_name][frame_name]])
                kp3d_preds = np.stack([result[1] for result in kp3d_results[video_name][frame_name]])
                kp2d_preds = kp2d_preds[:,all2lsp]
                kp3d_preds = kp3d_preds[:,all2lsp]
                
                valid_mask = kp2d_gts[:, :, 2].astype(np.bool_) #kp2d_vis.astype(np.bool_)
                bestMatch, falsePositives, misses = match_2d_greedy(kp2d_preds, kp2d_gts, valid_mask)
                bestMatch = np.array(bestMatch)

            if len(bestMatch)>0:
                pids, gids = bestMatch[:,0], bestMatch[:,1]
                if debug:
                    from vis_human.vis_utils import Plotter3dPoses
                    plot3d = Plotter3dPoses()
                    pose3d_plot = plot3d.plot([kp3d_preds[pids[0]], kp3d_gts[gids[0]]], bones=lsp14_connMat, colors=[(255, 0, 0), (0,0,255)])
                    cv2.imshow('3D skeleton', pose3d_plot)
                    cv2.waitKey(10)
                
                sequence_kp3ds[video_name][frame_name] = {'preds':kp3d_preds[pids], 'gts':kp3d_gts[gids], 'subject_id':np.array(gids)}
                
                MPJPEs = _calc_MPJPE_(kp3d_preds[pids], kp3d_gts[gids])
                PA_MPJPEs = _calc_PAMPJPE_(kp3d_preds[pids], kp3d_gts[gids])
            else:
                continue
                if video_name not in missed_frames:
                    missed_frames[video_name] = []
                missed_frames[video_name].append(frame_name)
                MPJPEs = PA_MPJPEs = PVEs = np.ones(len(kp3d_gts)) * missing_punish
            
            matrix_results['3dmpb-MPJPE'].append(MPJPEs); matrix_results['3dmpb-PA_MPJPE'].append(PA_MPJPEs)
            sequence_errors[video_name]['MPJPE'].append(MPJPEs); sequence_errors[video_name]['PA_MPJPE'].append(PA_MPJPEs)

    keys = list(matrix_results.keys())
    for key in keys:
        matrix_results[key] = np.concatenate(matrix_results[key]).mean()
    
    if seq_wise_results:
        video_names = list(sequence_errors.keys())
        for video_name in video_names:
            for key in sequence_errors[video_name]:
                sequence_errors[video_name][key] = np.concatenate(sequence_errors[video_name][key]).mean()

        MPJPEs = np.array([sequence_errors[video_name]['MPJPE'] for video_name in video_names])
        PA_MPJPEs = np.array([sequence_errors[video_name]['PA_MPJPE'] for video_name in video_names])

    gt_kp3d_sequence, pred_kp3d_sequence = convert2kp3d_sequence(sequence_kp3ds)

    seq_acceleration_errors = []
    for gt_kp3d_seq, pred_kp3d_seq in zip(gt_kp3d_sequence, pred_kp3d_sequence):
        person_num = len(gt_kp3d_seq)
        for pid in range(person_num):
            acc_error = compute_error_accel_np(gt_kp3d_seq[pid], pred_kp3d_seq[pid]) 
            seq_acceleration_errors.append(acc_error)
    acc_error_mean = np.concatenate(seq_acceleration_errors, 0).mean() * 1000.
    matrix_results['3dmpb-mACC'] = acc_error_mean

    print('Missed frames:', missed_frames)
    print('Missing subjects:', sorted(missed_subjects))
    print('3DPW:', matrix_results)
    return matrix_results

def collect_results():
    kp3d_result_folder = '/home/yusun/DataCenter/demo_results/tracking_results/3dmpb-demo/TRACE_tracking_results'
    kp3d_results = {}
    
    for results_path in glob.glob(os.path.join(kp3d_result_folder, '*.npz')):
        seq_name = os.path.basename(results_path).replace('.npz', '')
        kp3d_results[seq_name] = np.load(results_path, allow_pickle=True)['kp3ds'][()]
    return kp3d_results

if __name__ == '__main__':
    import sys
    #load_gts()
    kp3d_results = collect_results()
    evaluate_3dmpb_results(kp3d_results,debug=False,seq_wise_results=True, grot_rx_rectify=True)
