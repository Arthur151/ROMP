import numpy as np
import sys
sys.path.append('..')
import config
"""
The canonical part stick order:
0 Head
1 Torso
2 Right Upper Arm
3 Right Lower Arm
4 Right Upper Leg
5 Right Lower Leg
6 Left Upper Arm
7 Left Lower Arm
8 Left Upper Leg
9 Left Lower Leg
"""

CANONICAL_STICK_NAMES = ['Head', 'Torso', 'RU Arm', 'RL Arm', 'RU Leg',
                         'RL Leg', 'LU Arm', 'LL Arm', 'LU Leg', 'LL Leg']

"""
Person-Centric (PC) Annotations.
The canonical joint order for LSP dataset:
0 Head top
1 Neck
2 Right shoulder (from person's perspective)
3 Right elbow
4 Right wrist
5 Right hip
6 Right knee
7 Right ankle
8 Left shoulder
9 Left elbow
10 Left wrist
11 Left hip
12 Left knee
13 Left ankle
"""

CANONICAL_JOINT_NAMES = ['Head', 'Neck', 'R Shoulder',
                         'R elbow', 'R wrist',
                         'R hip', 'R knee', 'R ankle',
                         'L shoulder', 'L elbow',
                         'L wrist', 'L hip',
                         'L knee', 'L ankle']


def joints2sticks(joints,dataset_name):
    """
    Args:
        joints: array of joints in the canonical order.
      The canonical joint order:
        0 Head top
        1 Neck
        2 Right shoulder (from person's perspective)
        3 Right elbow
        4 Right wrist
        5 Right hip
        6 Right knee
        7 Right ankle
        8 Left shoulder
        9 Left elbow
        10 Left wrist
        11 Left hip
        12 Left knee
        13 Left ankle
    Returns:
        sticks: array of sticks in the canonical order.
      The canonical part stick order:
        0 Head
        1 Torso
        2 Right Upper Arm
        3 Right Lower Arm
        4 Right Upper Leg
        5 Right Lower Leg
        6 Left Upper Arm
        7 Left Lower Arm
        8 Left Upper Leg
        9 Left Lower Leg
    """
    assert joints.shape == (14, 2)
    stick_n = 10  # number of stick
    sticks = np.zeros((stick_n, 4), dtype=np.float32)
    sticks[0, :] = np.hstack([joints[0, :], joints[1, :]])  # Head
    sticks[1, :] = np.hstack([(joints[2, :] + joints[8, :]) / 2.0,
                             (joints[5, :] + joints[11, :]) / 2.0])  # Torso
    sticks[2, :] = np.hstack([joints[2, :], joints[3, :]])  # Left U.arms
    sticks[3, :] = np.hstack([joints[3, :], joints[4, :]])  # Left L.arms
    sticks[4, :] = np.hstack([joints[5, :], joints[6, :]])  # Left U.legs
    sticks[5, :] = np.hstack([joints[6, :], joints[7, :]])  # Left L.legs
    sticks[6, :] = np.hstack([joints[8, :], joints[9, :]])  # Right U.arms
    sticks[7, :] = np.hstack([joints[9, :], joints[10, :]])  # Right L.arms
    sticks[8, :] = np.hstack([joints[11, :], joints[12, :]])  # Right U.legs
    sticks[9, :] = np.hstack([joints[12, :], joints[13, :]])  # Right L.legs
    return sticks


def convert2canonical(joints,dataset_name):
    """
    Convert joints to evaluation structure.
    Permute joints according to the canonical joint order.
    """
    if dataset_name=='lsp':
      assert joints.shape[1:] == (14, 2), 'LSP must contain 14 joints per person'
      # convert to the canonical joint order
      joint_order = [13,  # Head top
                     12,  # Neck
                     8,   # Right shoulder
                     7,   # Right elbow
                     6,   # Right wrist
                     2,   # Right hip
                     1,   # Right knee
                     0,   # Right ankle
                     9,   # Left shoulder
                     10,  # Left elbow
                     11,  # Left wrist
                     3,   # Left hip
                     4,   # Left knee
                     5]   # Left ankle


    canonical = [dict() for _ in range(joints.shape[0])]
    for i in range(joints.shape[0]):
        canonical[i]['joints'] = joints[i, joint_order, :]
        canonical[i]['sticks'] = joints2sticks(canonical[i]['joints'],dataset_name)
    return canonical

def eval_pck(dataset_name, gt, pred,vis, thresh=0.05):
    """
    Compute average PCKh per joint.
    Matching threshold is 50% (thresh) of the head segment box size by default
    Args:
      gt_joints, predicted_joints: arrays of gt and predicted joints in the canonical order
      thresh: fraction of the torso height: || left_shoulder - right hip || segment length. This is the maximal deviation of the
        predicted joint from the gt joint position.
    Returns:
        pckh_per_joint: array of PCK scores. i-th element is the PCK score for the i-th joint
    """
    num_joints = 19
    assert pred.shape[1:] == (num_joints, 2), 'COCO19 must contain 19 joints per person'

    num_examples = len(gt)
    vis = np.where(vis.sum(-1)!=0)

    is_matched = np.zeros((num_examples, num_joints), dtype=int)

    for i in range(num_examples):
        left_shoulder_id = 5
        right_hip_id = 9
        gt_torso_len = np.linalg.norm(gt[i][left_shoulder_id] -
                                     gt[i][right_hip_id])
        for joint_id in range(num_joints):
            delta = np.linalg.norm(pred[i][joint_id] -
                                   gt[i][joint_id]) / gt_torso_len

            is_matched[i, joint_id] = delta <= thresh
    pckh_per_joint = []
    for i in range(num_joints):
        vis_joint_idx = vis[0][vis[1]==i]
        if len(vis_joint_idx)==0:
            pckh_per_joint.append(0)
        pckh_per_joint.append(np.mean(is_matched[vis_joint_idx,i]))
    pckh_per_joint = np.array(pckh_per_joint)

    return pckh_per_joint

def eval_pckh(dataset_name, gt, pred,vis, thresh=0.5):
    """
    Compute average PCKh per joint.
    Matching threshold is 50% (thresh) of the head segment box size by default
    Args:
      gt_joints, predicted_joints: arrays of gt and predicted joints in the canonical order
      thresh: fraction of the head segment length. This is the maximal deviation of the
        predicted joint from the gt joint position.
    Returns:
        pckh_per_joint: array of PCKh scores. i-th element is the PCKh score for the i-th joint
    """
    gt_joints, predicted_joints = convert2canonical(gt,dataset_name), convert2canonical(pred,dataset_name)

    num_joints = 14
    num_examples = len(gt_joints)
    vis = np.where(vis.sum(-1)!=0)

    is_matched = np.zeros((num_examples, num_joints), dtype=int)

    for i in range(num_examples):
        head_id = 0
        gt_head_len = np.linalg.norm(gt_joints[i]['sticks'][head_id, :2] -
                                     gt_joints[i]['sticks'][head_id, 2:])
        for joint_id in range(num_joints):
            delta = np.linalg.norm(predicted_joints[i]['joints'][joint_id] -
                                   gt_joints[i]['joints'][joint_id]) / gt_head_len

            is_matched[i, joint_id] = delta <= thresh
    pckh_per_joint = []
    for i in range(num_joints):
        vis_joint_idx = vis[0][vis[1]==i]
        if len(vis_joint_idx)==0:
            pckh_per_joint.append(0)
        pckh_per_joint.append(np.mean(is_matched[vis_joint_idx,i]))
    pckh_per_joint = np.array(pckh_per_joint)

    return pckh_per_joint


def average_pckh_symmetric_joints(dataset_name, pckh_per_joint):
    if dataset_name not in ['mpii', 'lsp']:
        raise ValueError('Unknown dataset {}'.format(dataset_name))

    joint_names = ['Head', 'Neck', 'Shoulder',
                   'Elbow', 'Wrist',
                   'Hip', 'Knee', 'Ankle',
                   'Thorax', 'Pelvis']
    if dataset_name == 'lsp':
        joint_names = joint_names[:-2]
    pckh_symmetric_joints = pckh_per_joint[:2].tolist()
    for i in range(2, 8):
        pckh_symmetric_joints.append((pckh_per_joint[i] + pckh_per_joint[i + 6]) / 2.0)
    pckh_symmetric_joints += pckh_per_joint[14:].tolist()
    return pckh_symmetric_joints, joint_names

def compute_pckh_lsp(gt,pred,vis, thresh=0.5):
    pckh_per_joint = eval_pckh('lsp', convert2canonical(gt), convert2canonical(pred),vis, thresh=thresh)
    #pckh_symmetric_joints, joint_names = average_pckh_symmetric_joints('lsp', pckh_per_joint)
    #return pckh_symmetric_joints, joint_names, pckh_per_joint.mean()
    return pckh_per_joint.mean()

def main():
    a=np.ones((10,14,2))
    b=np.ones((10,14,2))
    a[:,-2] = 0
    b[:,-2] = 0
    print(compute_pckh_lsp(a,b))

if __name__ == '__main__':
    main()