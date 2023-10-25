import argparse
import os, sys
import os.path as osp
import numpy as np
import cv2
import torch
from itertools import product
import joblib
from .smpl import SMPL, SMPLA_parser
import glob
import tqdm
from ..utils.utils import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

LSP_14 = {
    'R_Ankle':0, 'R_Knee':1, 'R_Hip':2, 'L_Hip':3, 'L_Knee':4, 'L_Ankle':5, 'R_Wrist':6, 'R_Elbow':7, \
    'R_Shoulder':8, 'L_Shoulder':9, 'L_Elbow':10, 'L_Wrist':11, 'Neck_LSP':12, 'Head_top':13
    }

lsp14_connMat = np.array([[ 0, 1 ],[ 1, 2 ],[ 3, 4 ],[ 4, 5 ],[ 6, 7 ],[ 7, 8 ],[ 8, 2 ],[ 8, 9 ],[ 9, 3 ],[ 2, 3 ],[ 8, 12],[ 9, 10],[12, 9 ],[10, 11],[12, 13]])

JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',      # 25, 0
'Right Knee',       # 26, 1
'Right Hip',        # 27, 2
'Left Hip',         # 28, 3
'Left Knee',        # 29, 4
'Left Ankle',       # 30, 5
'Right Wrist',      # 31, 6
'Right Elbow',      # 32, 7
'Right Shoulder',   # 33, 8
'Left Shoulder',    # 34, 9
'Left Elbow',       # 35, 10
'Left Wrist',       # 36, 11
'Neck (LSP)',       # 37, 12 ###
'Top of Head (LSP)',# 38, 13
'Pelvis (MPII)',    # 39, 14 #
'Thorax (MPII)',    # 40, 15
'Spine (H36M)',     # 41, 16 #
'Jaw (H36M)',       # 42, 17 #
'Head (H36M)',      # 43, 18 #
'Nose',             # 44, 19
'Left Eye',         # 45, 20
'Right Eye',        # 46, 21
'Left Ear',         # 47, 22
'Right Ear'         # 48, 23
]

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

joints = [JOINT_MAP[i] for i in JOINT_NAMES]
DBOA_LSP14_inds = np.arange(14)+25

def compute_error_accel_np(joints_gt, joints_pred):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    vis_mask = (joints_gt != -2.).sum(-1) >2
    vis = vis_mask[:-2] * vis_mask[1:-1] * vis_mask[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, ord=2, axis=2)
    return normed[vis]

def get_rotate_x_mat(angle):
    angle = np.radians(angle)
    rot_mat = torch.Tensor([
        [1, 0, 0], 
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]])
    return rot_mat

def get_rotate_y_mat(angle):
    angle = np.radians(angle)
    rot_mat = torch.Tensor([
        [np.cos(angle), 0, np.sin(angle)], 
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]])
    return rot_mat

def rotate_view_perspective(verts, rx=30, ry=0, FOV=60, bbox3D_center=None, depth=None):
    device, dtype = verts.device, verts.dtype

    # front2birdview: rx=90, ry=0 ; front2sideview: rx=0, ry=90
    Rx_mat = get_rotate_x_mat(rx).type(dtype).to(device)
    Ry_mat = get_rotate_y_mat(ry).type(dtype).to(device)
    verts_rot = torch.einsum('bij,kj->bik', verts, Rx_mat)
    verts_rot = torch.einsum('bij,kj->bik', verts_rot, Ry_mat)
    
    if bbox3D_center is None:
        flatten_verts = verts_rot.view(-1, 3)
        # To move the vertices to the center of view, we get the bounding box of vertices and its center location 
        bbox3D_center = 0.5 * (flatten_verts.min(0).values + flatten_verts.max(0).values)[None, None]
    verts_aligned = verts_rot - bbox3D_center
    
    if depth is None:
        # To ensure all vertices are visible, we need to move them further.
        # get the least / the greatest distance between the center of 3D bbox and all vertices
        dist_min = torch.abs(verts_aligned.view(-1, 3).min(0).values)
        dist_max = torch.abs(verts_aligned.view(-1, 3).max(0).values)
        z = dist_max[:2].max() / np.tan(np.radians(FOV/2)) + dist_min[2]
        depth = torch.tensor([[[0, 0, z]]], device=device)    
    verts_aligned = verts_aligned + depth

    return verts_aligned, bbox3D_center, depth

def key_3dpw(elem):
    elem = os.path.basename(elem)
    vid = elem.split('_')[1]
    pid = elem.split('_')[2][:-4]
    return int(vid)*10+int(pid)

def parse_3dpw(dataname):
    data = np.load(dataname)
    imgnames = data['imgname']
    scales = data['scale']
    centers = data['center']
    pose = data['pose'].astype(np.float32)
    betas = data['shape'].astype(np.float32)
    smpl_j2ds = data['j2d']
    op_j2ds = data['op_j2d']
    # Get gender data, if available
    gender = data['gender']
    genders = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)

    return imgnames, scales, centers, pose, betas, smpl_j2ds, op_j2ds, genders

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
    if S1.shape[1] != 3 and S1.shape[1] != 2:
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


def evaluate_3dpw_results(results_dir, dataset_dir, seq_wise_results=True, grot_rx_rectify=True, debug=False, **kwargs):
    annots_save_path = os.path.join(dataset_dir, '3dpw_dboa_test_annots.npz')
    annots = np.load(annots_save_path, allow_pickle=True)['annots'][()]

    kp3d_results = {}
    for results_path in glob.glob(os.path.join(results_dir, '*_tracking.npz')):
        seq_tracking_results = np.load(results_path, allow_pickle=True)['kp3ds'][()]
        seq_name = os.path.basename(results_path).replace('_tracking.npz', '')
        kp3d_results[seq_name] = seq_tracking_results

    matrix_results = {'pw3d-MPJPE': [], 'pw3d-PA_MPJPE': [], 'pw3d-PVE': []}
    missing_punish = 150
    missed_frames = {}
    missed_subjects = []

    evaluator = Evaluator()

    sequence_kp3ds = {}
    sequence_errors = {}
    image_names = np.load(os.path.join(dataset_dir,'3dpw_test_annots.npz'), allow_pickle=True)['annots'][()]
    video2frame = {}
    for img_ind, img_name in enumerate(image_names):
        video_name, frame_name = img_name.split('-')
        if video_name not in video2frame:
            video2frame[video_name] = []
        video2frame[video_name].append(frame_name)
    
    grot_x_bias = []
    #for video_name in ['downtown_bar_00']: #, 'flat_guitar_01'
    for video_name in tqdm.tqdm(video2frame):
        for frame_name in sorted(video2frame[video_name]):
            img_name = f'{video_name}-{frame_name}'
            if video_name not in sequence_kp3ds:
                sequence_kp3ds[video_name] = {}
                sequence_errors[video_name] = {'MPJPE': [], 'PA_MPJPE': [], 'PVE': []}
            # genders, smpl_j2ds, pose, betas = annots[img_name]
            genders = np.array([annot[0] for annot in annots[img_name]])
            kp2d_gts = np.stack([annot[1][DBOA_LSP14_inds] for annot in annots[img_name]], 0)
            theta_gts = np.stack([annot[2] for annot in annots[img_name]], 0)
            #theta_gts[:,:3] = make_axis_angle_between_0_to_2pi(theta_gts[:,:3])
            beta_gts = np.stack([annot[3] for annot in annots[img_name]], 0)
            
            verts_gts, kp3d_gts = evaluator.acquire_verts_kp3ds_with_gender(theta_gts, beta_gts, genders)
            
            if frame_name not in kp3d_results[video_name]:
                MPJPEs = PA_MPJPEs = PVEs = np.ones(len(kp3d_gts)) * missing_punish
                bestMatch = []
            else:
                kp2d_preds = np.stack([result[2] for result in kp3d_results[video_name][frame_name]])
                theta_preds = np.stack([result[4] for result in kp3d_results[video_name][frame_name]])
                beta_preds = np.stack([result[5] for result in kp3d_results[video_name][frame_name]])
                #if grot_rx_rectify and video_name in sequence2adjust_grots:
                #    theta_preds[:,:3] = rectify_grot_with_fov_and_pitch(theta_preds[:,:3], rx=sequence2adjust_grots[video_name])
                kp2d_preds = kp2d_preds[:,:14]

                verts_preds, kp3d_preds = evaluator.acquire_verts_kp3ds_natural(theta_preds, beta_preds)
                
                valid_mask = np.ones_like(kp2d_gts[:,:,0]).astype(np.bool_)
                bestMatch, falsePositives, misses = match_2d_greedy(kp2d_preds, kp2d_gts, valid_mask)
                bestMatch = np.array(bestMatch)

            if len(bestMatch)>0:
                pids, gids = bestMatch[:,0], bestMatch[:,1]
                grot_x_bias.append( np.degrees(theta_gts[gids,0])-np.degrees(theta_preds[pids,0]))
                if debug:
                    from vis_human.pyrenderer import Py3DR
                    renderer = Py3DR()
                    verts_vis = np.stack([verts_gts[gids[0]].cpu().numpy(),verts_preds[pids[0]].cpu().numpy()],0)
                    mesh_colors = np.array([[0,0,255],[255,0,0]])
                    background = np.zeros((512,512,3),dtype=np.uint8)
                    rendered_image, rend_depth = renderer(verts_vis+np.array([[[0,0,3]]]), \
                        evaluator.smpl_male.faces_tensor, background, mesh_colors=mesh_colors)
                    verts_side_view, bbox3D_center, move_depth = rotate_view_perspective(torch.from_numpy(verts_vis), rx=0, ry=90)
                    rendered_sv_image, rend_depth = renderer(verts_side_view.cpu().numpy(), evaluator.smpl_male.faces_tensor, background, mesh_colors=mesh_colors)
                    cv2.imshow('mesh', np.concatenate([rendered_image, rendered_sv_image],1))
                    cv2.waitKey(10)
                
                sequence_kp3ds[video_name][frame_name] = {'preds':kp3d_preds[pids], 'gts':kp3d_gts[gids], 'subject_id':np.array(gids)}
                
                PVEs = _calc_PVE_(verts_gts[gids],verts_preds[pids]).numpy()
                MPJPEs = _calc_MPJPE_(kp3d_preds[pids], kp3d_gts[gids])
                PA_MPJPEs = _calc_PAMPJPE_(kp3d_preds[pids], kp3d_gts[gids])
                # MPJPEs = np.concatenate([MPJPEs, np.ones(len(misses)) * missing_punish])
                # PA_MPJPEs = np.concatenate([PA_MPJPEs, np.ones(len(misses)) * missing_punish])
                # PVEs = np.concatenate([PVEs, np.ones(len(misses)) * missing_punish])
                # if len(misses)>0:
                #     missed_subjects.append(frame_name)
            else:
                continue
                if video_name not in missed_frames:
                    missed_frames[video_name] = []
                missed_frames[video_name].append(frame_name)
                MPJPEs= PA_MPJPEs = PVEs = np.ones(len(kp3d_gts)) * missing_punish
            
            matrix_results['pw3d-MPJPE'].append(MPJPEs); matrix_results['pw3d-PA_MPJPE'].append(PA_MPJPEs); matrix_results['pw3d-PVE'].append(PVEs)
            sequence_errors[video_name]['MPJPE'].append(MPJPEs); sequence_errors[video_name]['PA_MPJPE'].append(PA_MPJPEs); sequence_errors[video_name]['PVE'].append(PVEs)

    grot_x_bias = np.concatenate(grot_x_bias)
    grot_x_bias = grot_x_bias[np.abs(grot_x_bias)<50]
    print('bias grot in x :', grot_x_bias.mean())

    keys = list(matrix_results.keys())
    for key in keys:
        matrix_results[key] = np.concatenate(matrix_results[key]).mean()
    
    if seq_wise_results:
        video_names = list(sequence_errors.keys())
        for video_name in video_names:
            for key in sequence_errors[video_name]:
                sequence_errors[video_name][key] = np.concatenate(sequence_errors[video_name][key]).mean()
        #print('Sequence errors:', sequence_errors)
        PVEs = np.array([sequence_errors[video_name]['PVE'] for video_name in video_names])
        MPJPEs = np.array([sequence_errors[video_name]['MPJPE'] for video_name in video_names])
        PA_MPJPEs = np.array([sequence_errors[video_name]['PA_MPJPE'] for video_name in video_names])
        order_small2large = np.argsort(PVEs)
        for order in order_small2large:
            print(video_names[order], PVEs[order], MPJPEs[order], PA_MPJPEs[order])

    gt_kp3d_sequence, pred_kp3d_sequence = convert2kp3d_sequence(sequence_kp3ds)

    seq_acceleration_errors = []
    for gt_kp3d_seq, pred_kp3d_seq in zip(gt_kp3d_sequence, pred_kp3d_sequence):
        person_num = len(gt_kp3d_seq)
        for pid in range(person_num):
            acc_error = compute_error_accel_np(gt_kp3d_seq[pid], pred_kp3d_seq[pid]) 
            seq_acceleration_errors.append(acc_error)
    acc_error_mean = np.concatenate(seq_acceleration_errors, 0).mean() * 1000.
    matrix_results['pw3d-mACC'] = acc_error_mean

    print('Missed frames:', missed_frames)
    print('Missing subjects:', sorted(missed_subjects))
    print('3DPW:', matrix_results)
    return matrix_results

if __name__ == '__main__':
    import sys
    #kp3d_results_path = sys.argv[1]
    #kp3d_results = np.load(kp3d_results_path, allow_pickle=True)['results'][()]
    kp3d_results = {}
    track_results_save_folder = '/home/yusun/data_drive3/tracking_results/pw3d-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000-smooth/TROMP_v6_tracking_results'
    #'/home/yusun/data_drive3/TRACE_paper_results/pw3d-DTROMP_v5_GRU_TC_OF_SC_MO_SF_SO_PDC_CS-lr1e-5_val_4000-Challenge/TROMP_v6_tracking_results'
    for results_path in glob.glob(os.path.join(track_results_save_folder, '*.npz')):
        seq_name = os.path.basename(results_path).replace('.npz', '')
        #tracking_results[seq_name] = np.load(results_path, allow_pickle=True)['tracking'][()]
        kp3d_results[seq_name] = np.load(results_path, allow_pickle=True)['kp3ds'][()]
    evaluate_3dpw_results(kp3d_results,debug=False,seq_wise_results=True, grot_rx_rectify=True)
