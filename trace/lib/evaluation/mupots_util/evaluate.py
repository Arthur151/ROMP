import numpy as np 
from . import mpii_get_joints

def mean(l):
    return sum(l) / len(l)

def mpii_compute_3d_pck(seq_err):
    pck_curve_array = []
    pck_array = []
    auc_array = []
    thresh = np.arange(0, 200, 5)
    pck_thresh = 150
    joint_groups, all_joints = mpii_get_joints.mpii_joint_groups()
    for seq_idx in range(len(seq_err)):
        pck_curve = []
        pck_seq = []
        auc_seq = []
        err = np.array(seq_err[seq_idx]).astype(np.float32)
        for j in range(len(joint_groups)):
            err_selected = err[:,joint_groups[j][1]]
            buff = []
            for t in thresh:
                pck = np.float32(err_selected < t).sum() / len(joint_groups[j][1]) / len(err)
                buff.append(pck) #[Num_thresholds]
            pck_curve.append(buff)
            auc_seq.append(mean(buff))
            pck = np.float32(err_selected < pck_thresh).sum() / len(joint_groups[j][1]) / len(err)
            pck_seq.append(pck)
        
        buff = []
        for t in thresh:
            pck = np.float32(err[:, all_joints] < t).sum() / len(err) / len(all_joints)
            buff.append(pck) #[Num_thresholds]
        pck_curve.append(buff)

        pck = np.float32(err[:, all_joints] < pck_thresh).sum() / len(err) / len(all_joints)
        pck_seq.append(pck)
        
        pck_curve_array.append(pck_curve)   # [num_seq: [Num_grpups+1: [Num_thresholds]]]
        pck_array.append(pck_seq) # [num_seq: [Num_grpups+1]]
        auc_array.append(auc_seq) # [num_seq: [Num_grpups]]

    return pck_curve_array, pck_array, auc_array

def calculate_multiperson_errors(seq_err):
    return mpii_compute_3d_pck(seq_err)
