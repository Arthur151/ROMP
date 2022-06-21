import torch
import numpy as np

def print_dict(td):
    keys = collect_keyname(td)
    print(keys)

def get_size(item):
    if isinstance(item, list) or isinstance(item, tuple):
        return len(item)
    elif isinstance(item, torch.Tensor):
        return (item.shape, item.device)
    elif isinstance(item, np.np.ndarray):
        return item.shape
    else:
        return item

def collect_keyname(td):
    keys = []
    for key in td:
        if isinstance(td[key], dict):
            keys.append([key, collect_keyname(td[key])])
        else:
            keys.append([key, get_size(td[key])])
    return keys