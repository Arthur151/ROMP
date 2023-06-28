def mpii_joint_groups():
    joint_groups = [
        ['Head', [0]],
        ['Neck', [1]],
        ['Shou', [2,5]],
        ['Elbow', [3,6]],
        ['Wrist', [4,7]],
        ['Hip', [8,11]],
        ['Knee', [9,12]],
        ['Ankle', [10,13]],
    ]
    all_joints = []
    for i in joint_groups:
        all_joints += i[1]
    return joint_groups, all_joints


def mpii_get_joints(set_name):
    original_joint_names = ['spine3', 'spine4', 'spine2', 'spine1', 'spine',         
                        'neck', 'head', 'head_top', 'left_shoulder', 'left_arm', 'left_forearm', 
                       'left_hand', 'left_hand_ee',  'right_shoulder', 'right_arm', 'right_forearm', 'right_hand', 
                       'right_hand_ee', 'left_leg_up', 'left_leg', 'left_foot', 'left_toe', 'left_ee',         
                       'right_leg_up' , 'right_leg', 'right_foot', 'right_toe', 'right_ee']
    
    all_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',   
        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', 
       'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', 
       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',   
       'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe']

    if set_name=='relavant':
        joint_idx = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]
        joint_parents_o1 = [ 2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2]
        joint_parents_o2 = [ 16, 15, 16, 2, 3, 16, 2, 6, 16, 15, 9, 16, 15, 12, 15, 15, 16]
        joint_idx = [i-1 for i in joint_idx]
        joint_parents_o1 = [i-1 for i in joint_parents_o1]
        joint_parents_o2 = [i-1 for i in joint_parents_o2]
        joint_names = [all_joint_names[i] for i in joint_idx]
        return joint_idx, joint_parents_o1, joint_parents_o2, joint_names
    else:
        raise NotImplementedError('Not implemented yet.')
