import pickle
import numpy as np
import os
import torch

gender = 'neutral'.upper()

import pickle
import numpy as np
import os
import torch

root_folder = "/home/yusun/Infinity/project_data/romp_data/model_data/parameters/"
with open("/home/yusun/Infinity/project_data/romp_data/model_data/parameters/smil/smil_web.pkl", 'rb') as smpl_file:
    model_info = pickle.load(smpl_file, encoding='latin1')

np_model_info = {}
kintree_table = np.array(model_info['kintree_table'][0]).astype(np.int32)
kintree_table[0] = -1
np_model_info['kintree_table'] = kintree_table

np_model_info['J_regressor_extra9'] = np.array(np.load(root_folder+'J_regressor_extra.npy'), dtype=np.float32)
J_regressor_h36m = np.load(root_folder+'J_regressor_h36m.npy')
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
J_regressor_h36m17 = J_regressor_h36m[H36M_TO_J17]
np_model_info['J_regressor_h36m17'] = np.array(J_regressor_h36m17, dtype=np.float32)

# take the top 10 PCA componence of shape prior
np_model_info['shapedirs'] = np.array(model_info['shapedirs'][:, :, :10])
num_pose_basis = model_info['posedirs'].shape[-1]
posedirs = np.reshape(model_info['posedirs'], [-1, num_pose_basis]).T
np_model_info['posedirs'] = posedirs

VERTEX_IDS = {
    'smplh': {
        'nose':         332,
        'reye':         6260,
        'leye':         2800,
        'rear':         4071,
        'lear':         583,
        'rthumb':       6191,
        'rindex':       5782,
        'rmiddle':      5905,
        'rring':        6016,
        'rpinky':       6133,
        'lthumb':       2746,
        'lindex':       2319,
        'lmiddle':      2445,
        'lring':        2556,
        'lpinky':       2673,
        'LBigToe':      3216,
        'LSmallToe':    3226,
        'LHeel':        3387,
        'RBigToe':      6617,
        'RSmallToe':    6624,
        'RHeel':        6787
    }
}
np_model_info['extra_joints_index'] = extra_joints_idxs = np.array([
            # facial 5 joints
            VERTEX_IDS['smplh']['nose'], VERTEX_IDS['smplh']['reye'], VERTEX_IDS['smplh']['leye'], VERTEX_IDS['smplh']['rear'], VERTEX_IDS['smplh']['lear'], 
            # feet 6 joints
            VERTEX_IDS['smplh']['LBigToe'], VERTEX_IDS['smplh']['LSmallToe'], VERTEX_IDS['smplh']['LHeel'], VERTEX_IDS['smplh']['RBigToe'], VERTEX_IDS['smplh']['RSmallToe'], VERTEX_IDS['smplh']['RHeel'],
            # hand 10 joints
            #VERTEX_IDS['smplh']['lthumb'], VERTEX_IDS['smplh']['lindex'], VERTEX_IDS['smplh']['lmiddle'], VERTEX_IDS['smplh']['lring'], VERTEX_IDS['smplh']['lpinky'], 
            #VERTEX_IDS['smplh']['rthumb'], VERTEX_IDS['smplh']['rindex'], VERTEX_IDS['smplh']['rmiddle'], VERTEX_IDS['smplh']['rring'], VERTEX_IDS['smplh']['rpinky'],
            ], dtype=np.int64)

np_model_info['f'] = np.array(model_info['f'], dtype=np.int64)
np_model_info['v_template'] = np.array(model_info['v_template'], dtype=np.float32)
np_model_info['J_regressor'] = np.array(model_info['J_regressor'].todense(), dtype=np.float32)
np_model_info['v_template'] = np.array(model_info['v_template'], dtype=np.float32)
np_model_info['weights'] = np.array(model_info['weights'], dtype=np.float32)

np.savez('smil_packed_info.npz', annots=np_model_info)
tensor_model_info = {}
for k, v in np_model_info.items():
    print(k)
    if k in ['kintree_table', 'extra_joints_index']:
        tensor_model_info[k] = torch.from_numpy(v).long()
    else:
        tensor_model_info[k] = torch.from_numpy(v).float()
torch.save(tensor_model_info, f'SMIL_{gender}.pth')