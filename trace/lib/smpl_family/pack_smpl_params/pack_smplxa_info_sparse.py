import pickle
import numpy as np
import os
import torch

gender = 'neutral'.upper()
#gender = 'female'.upper()
# gender = 'male'.upper()

root_folder = "/home/yusun/Infinity/project_data/romp_data/model_data/parameters/"
with open(root_folder+"smplx/SMPLX_{}.pkl".format(gender.upper()), 'rb') as smpl_file:
    model_info = pickle.load(smpl_file, encoding='latin1')

print(list(model_info.keys()))
#print(model_info['joint2num'])

np_model_info = {}
kintree_table = np.array(model_info['kintree_table'][0]).astype(np.int32)
kintree_table[0] = -1
np_model_info['kintree_table'] = kintree_table

np_model_info['J_regressor_extra9'] = np.array(np.load(root_folder+'J_regressor_extra.npy'), dtype=np.float32)
J_regressor_h36m = np.load(root_folder+'J_regressor_h36m.npy')
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
J_regressor_h36m17 = J_regressor_h36m[H36M_TO_J17]
np_model_info['J_regressor_h36m17'] = np.array(J_regressor_h36m17, dtype=np.float32)

smplx2smpl = pickle.load(open('/home/yusun/DataCenter2/smpl_models/model_transfer/smplx_to_smpl.pkl', 'rb'))['matrix'][()]
np_model_info['J_regressor_extra9'] = np.matmul(np_model_info['J_regressor_extra9'], smplx2smpl)
np_model_info['J_regressor_h36m17'] = np.matmul(np_model_info['J_regressor_h36m17'], smplx2smpl)
print(np_model_info['J_regressor_extra9'].shape, smplx2smpl.shape)
#print(np.where(np_model_info['J_regressor_extra9']>0)[0])

# take the top 10 PCA componence of shape prior
# shape space  400 =  SHAPE_SPACE_DIM 300 + EXPRESSION_SPACE_DIM 100
np_model_info['shapedirs'] = np.array(model_info['shapedirs'])[:,:,:10]
np_model_info['expr_dirs'] = np.array(model_info['shapedirs'])[:,:,300:]

num_pose_basis = model_info['posedirs'].shape[-1]
posedirs = np.reshape(model_info['posedirs'], [-1, num_pose_basis]).T
np_model_info['posedirs'] = posedirs

np_model_info['hands_componentsr'] = model_info['hands_componentsr']
np_model_info['hands_componentsl'] = model_info['hands_componentsl']
np_model_info['hands_meanr'] = model_info['hands_meanr']
np_model_info['hands_meanl'] = model_info['hands_meanl']

# the commented keypoints we directly use the default smplx joint regressed by J_regressor
VERTEX_IDS = {
    'smplx': {
        'nose':         9120,
        #'reye':         9929,
        #'leye':         9448,
        'rear':         616,
        'lear':         6,
        # 'rthumb':       8079,
        # 'rindex':       7669,
        # 'rmiddle':      7794,
        # 'rring':        7905,
        # 'rpinky':       8022,
        # 'lthumb':       5361,
        # 'lindex':       4933,
        # 'lmiddle':      5058,
        # 'lring':        5169,
        # 'lpinky':       5286,
        'LBigToe':      5770,
        'LSmallToe':    5780,
        'LHeel':        8846,
        'RBigToe':      8463,
        'RSmallToe':    8474,
        'RHeel':        8635
    },
    'mano': {
            'thumb':	744,
            'index':	320,
            'middle':	443,
            'ring':		554,
            'pinky':	671}}

# the commented keypoints we directly use the default smplx joint regressed by J_regressor
np_model_info['extra_joints_index'] = extra_joints_idxs = np.array([
            # facial 5 joints
            VERTEX_IDS['smplx']['nose'], VERTEX_IDS['smplx']['rear'], VERTEX_IDS['smplx']['lear'], 
            # feet 6 joints
            VERTEX_IDS['smplx']['LBigToe'], VERTEX_IDS['smplx']['LSmallToe'], VERTEX_IDS['smplx']['LHeel'], VERTEX_IDS['smplx']['RBigToe'], VERTEX_IDS['smplx']['RSmallToe'], VERTEX_IDS['smplx']['RHeel'],
            ], dtype=np.int64)
np_model_info['mano_joints_index'] = np.array([
    VERTEX_IDS['mano']['thumb'], VERTEX_IDS['mano']['index'], VERTEX_IDS['mano']['middle'], VERTEX_IDS['mano']['ring'], VERTEX_IDS['mano']['pinky']])

np_model_info['f'] = np.array(model_info['f'], dtype=np.int64)
np_model_info['v_template'] = np.array(model_info['v_template'], dtype=np.float32)

np_model_info['v_template'] = np.array(model_info['v_template'], dtype=np.float32)
np_model_info['weights'] = np.array(model_info['weights'], dtype=np.float32)

np_model_info['J_regressor'] = np.array(model_info['J_regressor'], dtype=np.float32)
np_model_info['J_regressor'] = np_model_info['J_regressor']#[SMPLX55_to_SMPL24]

#np.savez('smpl_{}.npz'.format(gender), annots=np_model_info)
tensor_model_info = {}
for k, v in np_model_info.items():
    print(k, v.shape)
    if k in ['kintree_table', 'extra_joints_index', 'mano_joints_index']:
        tensor_model_info[k] = torch.from_numpy(v).long()
    else:
        tensor_model_info[k] = torch.from_numpy(v).float()

for item_name in ['J_regressor', 'J_regressor_extra9', 'J_regressor_h36m17']:
    indices = torch.where(tensor_model_info[item_name]>0)
    matrix_shape = tensor_model_info[item_name].shape
    print(item_name, matrix_shape, indices[0].shape, matrix_shape[0]*matrix_shape[1])
    indices = torch.stack(indices, 0)
    values = tensor_model_info[item_name][indices[0], indices[1]]
    tensor_model_info[item_name] = torch.sparse_coo_tensor(indices, values, matrix_shape)

#torch.save(tensor_model_info, 'project_data/romp_data/model_data/parameters/SMPLX_{}.pth'.format(gender.upper()))

kid_template_path = os.path.join(root_folder, 'smplx_kid_template.npy')
v_template_smil = np.load(kid_template_path)
v_template_smil -= np.mean(v_template_smil, axis=0)
# Recommanded to use gender= 'male' for male kids and gender='neutral' for female kids.
kid_shape_diff = np.array(v_template_smil - np_model_info['v_template'], dtype=np.float32)
np_model_info['shapedirs'] = np.concatenate([np_model_info['shapedirs'], kid_shape_diff[:,:,None]], -1)
tensor_model_info['shapedirs'] = torch.from_numpy(np_model_info['shapedirs']).float()

torch.save(tensor_model_info, 'project_data/romp_data/model_data/parameters/SMPLXA_{}.pth'.format(gender.upper()))