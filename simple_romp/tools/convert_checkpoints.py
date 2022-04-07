from os import remove
from sklearn.model_selection import PredefinedSplit
import torch
import sys

def remove_prefix(state_dict, prefix='module.', remove_keys=['_result_parser', '_calc_loss']):
    keys = list(state_dict.keys())
    print('orginal keys:', keys)
    for key in keys:
        exist_flag = True
        for rkey in remove_keys:
            if rkey in key:
                del state_dict[key]
                exist_flag = False
        if not exist_flag:
            continue
        if prefix in key:
            state_dict[key.replace(prefix, '')] = state_dict[key]
            del state_dict[key]
    
    keys = list(state_dict.keys())
    print('new keys:', keys)
    return state_dict

if __name__ == '__main__':
    model_path = sys.argv[1]
    save_path = sys.argv[2]
    state_dict = remove_prefix(torch.load(model_path), prefix='module.')
    torch.save(state_dict, save_path)
