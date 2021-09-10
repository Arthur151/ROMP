import os,sys
import numpy as np
import pickle as pkl
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from pw3d_eval.SMPL import SMPL
import glob
import cv2

PCK_THRESH = 50.0
AUC_MIN = 0.0
AUC_MAX = 200.0
NUM_SEQS = 87
NUM_OCSEQS = 24
SMPL_NR_JOINTS = 24

SMPL_MAJOR_JOINTS = np.array([1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21])

SMPL_OR_JOINTS = np.array([0, 1, 2, 4, 5, 16, 17, 18, 19])

smpl_model_path = os.path.join(os.path.abspath(__file__).replace('romp/lib/evaluation/pw3d_eval/evaluate.py',''), 'model_data', 'parameters')
print('loading SMPL statistical parameters from ', smpl_model_path)

def joint_angle_error(pred_mat, gt_mat):
    """
    Compute the geodesic distance between the two input matrices.
    :param pred_mat: predicted rotation matrices. Shape: ( Seq, 9g, 3, 3)
    :param gt_mat: ground truth rotation matrices. Shape: ( Seq, 24, 3, 3)
    :return: Mean geodesic distance between input matrices.
    """

    gt_mat = gt_mat[:, SMPL_OR_JOINTS, :, :]

    # Reshape the matrices into B x 3 x 3 arrays
    r1 = np.reshape(pred_mat, [-1, 3, 3])
    r2 = np.reshape(gt_mat, [-1, 3, 3])

    # Transpose gt matrices
    r2t = np.transpose(r2, [0, 2, 1])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(r1, r2t)

    angles = []
    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r1.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))

    return np.mean(np.array(angles))


def compute_auc(xpts, ypts):
    """
    Calculates the AUC.
    :param xpts: Points on the X axis - the threshold values
    :param ypts: Points on the Y axis - the pck value for that threshold
    :return: The AUC value computed by integrating over pck values for all thresholds
    """
    a = np.min(xpts)
    b = np.max(xpts)
    from scipy import integrate
    myfun = lambda x: np.interp(x, xpts, ypts)
    auc = integrate.quad(myfun, a, b)[0]
    return auc


def compute_pck(errors, THRESHOLD):
    """
    Computes Percentage-Correct Keypoints
    :param errors: N x 12 x 1
    :param THRESHOLD: Threshold value used for PCK
    :return: the final PCK value
    """
    errors_pck = errors <= THRESHOLD
    errors_pck = np.mean(errors_pck, axis=1)
    return np.mean(errors_pck)


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Ensure that the first argument is the prediction

    Source: https://en.wikipedia.org/wiki/Kabsch_algorithm

    :param S1 predicted joint positions array 24 x 3
    :param S2 ground truth joint positions array 24 x 3
    :return S1_hat: the predicted joint positions after apply similarity transform
            R : the rotation matrix computed in procrustes analysis
    '''
    # If all the values in pred3d are zero then procrustes analysis produces nan values
    # Instead we assume the mean of the GT joint positions is the transformed joint value

    if not (np.sum(np.abs(S1)) == 0):
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert (S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1 ** 2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        # 7. Error:
        S1_hat = scale * R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat, R
    else:
        S1_hat = np.tile(np.mean(S2, axis=0), (SMPL_NR_JOINTS, 1))
        R = np.identity(3)

        return S1_hat, R


def align_by_root(joints):
    """
    Assumes joints is 24 x 3 in SMPL order.
    Subtracts the location of the root joint from all the other joints
    """
    root = joints[0, :]

    return joints - root


def compute_errors(preds3d, gt3ds):
    """
    Gets MPJPE after root alignment + MPJPE after Procrustes.
    Evaluates on all the 24 joints joints.
    Inputs:
    :param gt3ds: N x 24 x 3
    :param preds: N x 24 x 3
    :returns
        MPJPE : scalar - mean of all MPJPE errors
        MPJPE_PA : scalar- mean of all MPJPE_PA errors
        errors_pck : N x 24 - stores the error b/w GT and prediction for each joint separate
        proc_mats : N x 3 x 3 - for each frame, stores the 3 x 3 rotation matrix that best aligns the prediction and GT
    """
    errors, errors_pa, errors_pck = [], [], []

    proc_rot = []

    for i, (gt3d, pred3d) in enumerate(zip(gt3ds, preds3d)):
        # gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_root(gt3d)
        pred3d = align_by_root(pred3d)

        # Compute MPJPE
        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        errors.append(np.mean(joint_error))

        # Joint errors for PCK Calculation
        joint_error_maj = joint_error[SMPL_MAJOR_JOINTS]
        errors_pck.append(joint_error_maj)

        # Compute MPJPE_PA and also store similiarity matrices to apply them later to rotation matrices for MPJAE_PA
        pred3d_sym, R = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_pa.append(np.mean(pa_error))

        proc_rot.append(R)

    return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), \
           np.stack(errors_pck, 0), np.stack(proc_rot, 0)


def with_ones(data):
    """
    Converts an array in 3d coordinates to 4d homogenous coordiantes
    :param data: array of shape A x B x 3
    :return return ret_arr: array of shape A x B x 4 where the extra dimension is filled with ones
    """
    ext_arr = np.ones((data.shape[0], data.shape[1], 1))
    ret_arr = np.concatenate((data, ext_arr), axis=2)
    return ret_arr


def apply_camera_transforms(joints, rotations, camera):
    """
    Applies camera transformations to joint locations and rotations matrices
    :param joints: B x 24 x 3
    :param rotations: B x 24 x 3 x 3
    :param camera: B x 4 x 4 - already transposed
    :return: joints B x 24 x 3 joints after applying camera transformations
             rotations B x 24 x 3 x 3 - rotations matrices after applying camera transformations
    """
    joints = with_ones(joints)  # B x 4 x 4
    joints = np.matmul(joints, camera)

    # multiply all rotation matrices with the camera rotation matrix
    # transpose camera coordinates back
    cam_new = np.transpose(camera[:, :3, :3], (0, 2, 1))
    cam_new = np.expand_dims(cam_new, 1)
    cam_new = np.tile(cam_new, (1, 24, 1, 1))
    # B x 24 x 3 x 3
    rotations = np.matmul(cam_new, rotations)

    return joints[:, :, :3], rotations


def check_valid_inds(poses2d, camposes_valid):
    """
    Computes the indices where further computations are required
    :param poses2d: N x 18 x 3 array of 2d Poses
    :param camposes_valid: N x 1 array of indices where camera poses are valid
    :return: array of indices indicating frame ids in the sequence which are to be evaluated
    """

    # find all indices in the N sequences where the sum of the 18x3 array is not zero
    # N, numpy array
    poses2d_mean = np.mean(np.mean(np.abs(poses2d), axis=2), axis=1)
    poses2d_bool = poses2d_mean == 0
    poses2d_bool_inv = np.logical_not(poses2d_bool)

    # find all the indices where the camposes are valid
    camposes_valid = np.array(camposes_valid).astype('bool')

    final = np.logical_and(poses2d_bool_inv, camposes_valid)
    indices = np.array(np.where(final == True)[0])

    return indices


def get_data(paths_gt, paths_pred, truth_dir):
    """
    The function reads all the ground truth and prediction files. And concatenates

    :param paths_gt: all the paths corresponding to the ground truth - list of pkl files
    :param paths_prd: all the paths corresponding to the predictions - list of pkl files
    :return:
        jp_pred: jointPositions Prediction. Shape N x 24 x 3
        jp_gt: jointPositions ground truth. Shape: N x 24 x 3
        mats_pred: Global rotation matrices predictions. Shape N x 24 x 3 x 3
        mats_gt: Global rotation matrices ground truths. Shape N x 24 x 3 x 3
    """

    # all predicted joint positions
    all_jp_preds = []
    # all ground truth joint positions
    all_jp_gts = []

    # all ground-truth rotation matrices
    all_glob_rot_gts = []
    # all predicted rotations matrices
    all_glob_rot_preds = []

    seq = 0
    num_jps_pred = 0
    num_ors_pred = 0
    num_params_pred = 0

    smpl_model_dict = {'f':SMPL(center_idx=0, gender='f', model_root=smpl_model_path),\
                        'm':SMPL(center_idx=0, gender='m', model_root=smpl_model_path),\
                        'n':SMPL(center_idx=0, gender='neutral', model_root=smpl_model_path),}

    # construct the data structures -
    for path_pred, path_gt in zip(paths_pred, paths_gt):
        seq = seq + 1
        # Open pkl files
        data_gt = pkl.load(open(path_gt, 'rb'), encoding='latin1')
        data_pred = pkl.load(open(path_pred, 'rb'), encoding='latin1')

        genders = data_gt['genders']

        for i in range(len(genders)):

            # Get valid frames
            # Frame with no zeros in the poses2d file and where campose_valid is True
            poses2d_gt = data_gt['poses2d']
            poses2d_gt_i = poses2d_gt[i]

            camposes_valid = data_gt['campose_valid']
            camposes_valid_i = camposes_valid[i]

            valid_indices = check_valid_inds(poses2d_gt_i, camposes_valid_i)

            keys_pred = data_pred.keys()

            # Get prediction joints
            if 'jointPositions' in keys_pred:
                num_jps_pred = num_jps_pred + 1
                jp_pred = np.array(data_pred['jointPositions'])
                # select a subset of prediction data
                jp_pred_i = jp_pred[i, valid_indices, :, :]
                all_jp_preds.append(jp_pred_i)

            # Get predicted orientations
            if 'orientations' in keys_pred:
                num_ors_pred = num_ors_pred + 1
                glob_rot_pred = np.array(data_pred['orientations'])
                # select a subset of prediction data
                glob_rot_pred_i = glob_rot_pred[i, valid_indices, :, :]
                all_glob_rot_preds.append(glob_rot_pred_i)

            if 'smpl_params' in keys_pred:
                num_params_pred = num_params_pred + 1
                params_pred = np.array(data_pred['smpl_params'])
                params_pred_i = params_pred[i, valid_indices, :]

            # Get the ground truth SMPL body parameters - poses, betas and translation parameters
            pose_params = np.array(data_gt['poses'])
            pose_params_i = pose_params[i, valid_indices, :]

            shape_params = np.array(data_gt['betas'][i])
            shape_params = np.expand_dims(shape_params, 0)
            shape_params = shape_params[:, :10]
            shape_params = np.tile(shape_params, (pose_params_i.shape[0], 1))

            trans_params = np.array(data_gt['trans'])
            
            trans_params_i = trans_params[i, valid_indices, :]

            # Get the GT joint and vertex positions and the global rotation matrices
            verts_gt, jp_gt, glb_rot_mats_gt = smpl_model_dict[genders[i]].update(pose_params_i, shape_params, trans_params_i)

            # Apply Camera Matrix Transformation to ground truth values
            cam_matrix = data_gt['cam_poses']
            new_cam_poses = np.transpose(cam_matrix, (0, 2, 1))
            new_cam_poses_i = new_cam_poses[valid_indices, :, :]

            jp_gt, glb_rot_mats_gt = apply_camera_transforms(jp_gt, glb_rot_mats_gt, new_cam_poses_i)
            all_jp_gts.append(jp_gt)
            all_glob_rot_gts.append(glb_rot_mats_gt)
            del jp_gt, verts_gt, glb_rot_mats_gt

    print('number of sequences: ', num_jps_pred)
    all_jp_preds = np.concatenate(all_jp_preds, 0)
    all_jp_gts = np.concatenate(all_jp_gts, 0)

    all_glob_rot_preds = np.concatenate(all_glob_rot_preds, 0)
    all_glob_rot_gts = np.concatenate(all_glob_rot_gts, 0)

    return (all_jp_preds, all_jp_gts, all_glob_rot_preds, all_glob_rot_gts)


def get_paths(submit_dir, truth_dir):
    """
    submit_dir: The location of the submit directory
    truth_dir: The location of the truth directory
    Return: two lists
            fnames_gt : the list of all files in ground truth folder
            fnames_pred : the list of all files in the predicted folder
    """
    fnames_gt = []
    fnames_pred = []

    keys = ['train', 'validation', 'test']

    for key in keys:
        fnames_gt_temp = sorted(glob.glob(os.path.join(truth_dir, key, "") + "*.pkl"))
        fnames_pred_temp = sorted(glob.glob(os.path.join(submit_dir, key, "") + "*.pkl"))
        fnames_gt = fnames_gt + fnames_gt_temp
        fnames_pred = fnames_pred + fnames_pred_temp

    assert len(fnames_gt) == len(fnames_pred), print('len(fnames_gt),len(fnames_pred): ',len(fnames_gt),len(fnames_pred))
    return sorted(fnames_gt), sorted(fnames_pred)


def main(submit_dir, truth_dir, output_filename):
    """
    :param submit_dir: The location of the submission files
    :param truth_dir: The location of the GT files
    :return output_filename: The location of the output txt file
    """

    # Get all the GT and submission paths in paired list form
    fnames_gt, fnames_pred = get_paths(submit_dir, truth_dir)

    print('geting data')
    # Get all the ground-truth and submission joint positions
    # Get all the ground-truth and submission Global rotation matrices
    jp_pred, jp_gt, mats_pred, mats_gt = get_data(fnames_gt, fnames_pred, truth_dir)

    # Check if the predicted and GT joints have the same number
    assert jp_pred.shape == jp_gt.shape
    assert mats_pred.shape[0] == mats_gt.shape[0]
    PVE = np.inf

    print('start evaluation')
    # If there are submitted joint predictions
    if not jp_pred.shape == (0,):

        # Joint errors and procrustes matrices
        MPJPE_final, MPJPE_PA_final, errors_pck, mat_procs = compute_errors(jp_pred * 1000., jp_gt * 1000.)

        # PCK value
        pck_final = compute_pck(errors_pck, PCK_THRESH) * 100.

        # AUC value
        auc_range = np.arange(AUC_MIN, AUC_MAX)
        pck_aucs = []
        for pck_thresh_ in auc_range:
            err_pck_tem = compute_pck(errors_pck, pck_thresh_)
            pck_aucs.append(err_pck_tem)

        auc_final = compute_auc(auc_range / auc_range.max(), pck_aucs)

        # If orientation are submitted, compute orientation errors
        if not (mats_pred.shape == (0,)):
            # Apply procrustus rotation to the global rotation matrices
            mats_procs_exp = np.expand_dims(mat_procs, 1)
            mats_procs_exp = np.tile(mats_procs_exp, (1, len(SMPL_OR_JOINTS), 1, 1))
            mats_pred_prc = np.matmul(mats_procs_exp, mats_pred)

            # Compute differences between the predicted matrices after procrustes and GT matrices
            error_rot_procruster = np.degrees(joint_angle_error(mats_pred_prc, mats_gt))
        else:
            error_rot_procruster = np.inf
    else:
        MPJPE_final = np.inf
        MPJPE_PA_final = np.inf

        pck_final = np.inf
        auc_final = np.inf

        error_rot_procruster = np.inf

    # If only orientation are provided, only compute MPJAE
    # MPJAE_PA requires procrustes analysis which requires joint positions to be submitted as well
    if not (mats_pred.shape == (0,)):
        error_rot = np.degrees(joint_angle_error(mats_pred, mats_gt))
    else:
        error_rot = np.inf

    errs = {
        'MPJPE': MPJPE_final,
        'MPJPE_PA': MPJPE_PA_final,
        'PCK': pck_final,
        'AUC': auc_final,
        'MPJAE': error_rot,
        'MPJAE_PA': error_rot_procruster,
        'PVE': PVE,
    }

    str = ''
    for err in errs.keys():
        if not errs[err] == np.inf:
            str = str + err + ': {}\n'.format(errs[err])

    print(str)
    with open(output_filename, 'w') as f:
        f.write(str)
    f.close()


if __name__ == "__main__":

    # Process reference and results directory
    submit_dir = sys.argv[1]
    truth_dir = sys.argv[2]

    # Make output directory
    output_filename = submit_dir+'_scores.txt'
    print('~~start evaluation~~')
    print('submit_dir', submit_dir)
    print('truth_dir', truth_dir)
    # Execute main program
    main(submit_dir, truth_dir, output_filename)