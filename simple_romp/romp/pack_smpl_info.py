import pickle
import numpy as np
import os
import torch
import argparse

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
        },
        'smplx': {
            'nose':         9120,
            'reye':         9929,
            'leye':         9448,
            'rear':         616,
            'lear':         6,
            'rthumb':       8079,
            'rindex':       7669,
            'rmiddle':      7794,
            'rring':        7905,
            'rpinky':       8022,
            'lthumb':       5361,
            'lindex':       4933,
            'lmiddle':      5058,
            'lring':        5169,
            'lpinky':       5286,
            'LBigToe':      5770,
            'LSmallToe':    5780,
            'LHeel':        8846,
            'RBigToe':      8463,
            'RSmallToe':    8474,
            'RHeel':        8635
        }
}

def main():
    parser = argparse.ArgumentParser(description='Convert to our format.')
    parser.add_argument('-source_dir', type=str, help = 'Where you put the download SMPL model files and other related meta data')
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.path.expanduser("~"), '.romp'), help = 'Path to save processed SMPL model files')
    parser.add_argument('--gender', type=str, default='neutral')
    args = parser.parse_args()

    source_dir = args.source_dir
    save_dir = args.save_dir
    gender = args.gender.upper()

    with open(os.path.join(source_dir, "SMPL_{}.pkl".format(gender.upper())), 'rb') as smpl_file:
        model_info = pickle.load(smpl_file, encoding='latin1')

    np_model_info = {}
    kintree_table = np.array(model_info['kintree_table'][0]).astype(np.int32)
    kintree_table[0] = -1
    np_model_info['kintree_table'] = kintree_table

    np_model_info['J_regressor_extra9'] = np.array(np.load(os.path.join(source_dir, 'J_regressor_extra.npy')), dtype=np.float32)
    J_regressor_h36m = np.load(os.path.join(source_dir, 'J_regressor_h36m.npy'))
    H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
    J_regressor_h36m17 = J_regressor_h36m[H36M_TO_J17]
    np_model_info['J_regressor_h36m17'] = np.array(J_regressor_h36m17, dtype=np.float32)

    # take the top 10 PCA componence of shape prior
    np_model_info['shapedirs'] = np.array(model_info['shapedirs'][:, :, :10])
    num_pose_basis = model_info['posedirs'].shape[-1]
    posedirs = np.reshape(model_info['posedirs'], [-1, num_pose_basis]).T
    np_model_info['posedirs'] = posedirs

    np_model_info['extra_joints_index'] = extra_joints_idxs = np.array([
                # facial 5 joints
                VERTEX_IDS['smplh']['nose'], VERTEX_IDS['smplh']['reye'], VERTEX_IDS['smplh']['leye'], VERTEX_IDS['smplh']['rear'], VERTEX_IDS['smplh']['lear'], 
                # feet 6 joints
                VERTEX_IDS['smplh']['LBigToe'], VERTEX_IDS['smplh']['LSmallToe'], VERTEX_IDS['smplh']['LHeel'], VERTEX_IDS['smplh']['RBigToe'], VERTEX_IDS['smplh']['RSmallToe'], VERTEX_IDS['smplh']['RHeel'],
                # hand 10 joints
                VERTEX_IDS['smplh']['lthumb'], VERTEX_IDS['smplh']['lindex'], VERTEX_IDS['smplh']['lmiddle'], VERTEX_IDS['smplh']['lring'], VERTEX_IDS['smplh']['lpinky'], 
                VERTEX_IDS['smplh']['rthumb'], VERTEX_IDS['smplh']['rindex'], VERTEX_IDS['smplh']['rmiddle'], VERTEX_IDS['smplh']['rring'], VERTEX_IDS['smplh']['rpinky'],
                ], dtype=np.int64)

    np_model_info['f'] = np.array(model_info['f'], dtype=np.int64)
    np_model_info['v_template'] = np.array(model_info['v_template'], dtype=np.float32)
    np_model_info['J_regressor'] = np.array(model_info['J_regressor'].todense(), dtype=np.float32)
    np_model_info['v_template'] = np.array(model_info['v_template'], dtype=np.float32)
    np_model_info['weights'] = np.array(model_info['weights'], dtype=np.float32)

    #np.savez('smpl_{}.npz'.format(gender), annots=np_model_info)
    tensor_model_info = {}
    for k, v in np_model_info.items():
        print(k)
        if k in ['kintree_table', 'extra_joints_index']:
            tensor_model_info[k] = torch.from_numpy(v).long()
        else:
            tensor_model_info[k] = torch.from_numpy(v).float()
    torch.save(tensor_model_info, os.path.join(save_dir, 'SMPL_{}.pth'.format(gender.upper())))


    kid_template_path = os.path.join(source_dir, 'smpl_kid_template.npy')
    v_template_smil = np.load(kid_template_path)
    v_template_smil -= np.mean(v_template_smil, axis=0)
    # Recommanded to use gender= 'male' for male kids and gender='neutral' for female kids.
    kid_shape_diff = np.array(v_template_smil - np_model_info['v_template'], dtype=np.float32)
    np_model_info['smpla_shapedirs'] = np.concatenate([np_model_info['shapedirs'], kid_shape_diff[:,:,None]], -1)

    tensor_model_info = {}
    for k, v in np_model_info.items():
        print(k)
        if k in ['kintree_table', 'extra_joints_index']:
            tensor_model_info[k] = torch.from_numpy(v).long()
        else:
            tensor_model_info[k] = torch.from_numpy(v).float()
    torch.save(tensor_model_info, os.path.join(save_dir, 'SMPLA_{}.pth'.format(gender.upper())))