import numpy as np 
import pickle 
import os,sys

file_path = os.path.dirname(__file__)
gt_dict = {}
def get_pred_gt(seq_idx, inst_idx, frame_idx):
    if not seq_idx in gt_dict:
        gt = pickle.load(open(os.path.join(file_path,'mupots_depths','%02d_%02d.pkl'%(seq_idx, inst_idx)), 'rb'))
        gt_dict[(seq_idx, inst_idx)] = np.float32(gt)
    gt = gt_dict[(seq_idx, inst_idx)]
    return gt[frame_idx]
