import sys,os
import random
import torch
import numpy as np
import logging

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = 'module.', drop_prefix='', \
    ignore_layer='_result_parser.params_map_parser.smpl_model',fix_loaded=False):
    success_layers, failed_layers = [], []
    def _get_params(key):
        key = key.replace(drop_prefix,'')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                if ignore_layer not in k:
                   failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix!='':
                k=k.split(prefix)[1]
            success_layers.append(k)
        except:
            logging.info('copy param {} failed, mismatched'.format(k))
            continue
    if len(failed_layers)>0:
        logging.info('missing parameters of layers:{}'.format(failed_layers))

    if fix_loaded and len(failed_layers)>0:
        print('fixing the layers that were loaded successfully, while train the layers that failed,')
        for k in cur_state_dict.keys():
            try:
                if k in success_layers:
                    cur_state_dict[k].requires_grad=False
            except:
                print('fixing the layer {} failed'.format(k))

    return success_layers

def load_model(path, model, prefix = 'module.', drop_prefix='',optimizer=None, **kwargs):
    logging.info('using fine_tune model: {}'.format(path))
    if os.path.exists(path):
        pretrained_model = torch.load(path, map_location=torch.device('cpu'))
        current_model = model.state_dict()
        if isinstance(pretrained_model, dict):
            if 'model_state_dict' in pretrained_model:
                pretrained_model = pretrained_model['model_state_dict']
        copy_state_dict(current_model, pretrained_model, prefix = prefix, drop_prefix=drop_prefix, **kwargs)
    else:
        logging.warning('model {} not exist!'.format(path))
    return model

def print_dict(dt):
    for key, value in dt.items():
        if isinstance(value, dict):
            print('Dict {}'.format(key))
            print_dict(value)
            print('_______________')
        elif isinstance(value, list):
            print('List {}, length {}'.format(key, len(value)))
        elif isinstance(value, tuple):
            print('Tuple {}, length {}'.format(key, len(value)))
        elif isinstance(value, np.ndarray):
            print('Np {}, shape {}, dtype {}'.format(key, value.shape, value.dtype))
        elif torch.is_tensor(value):
            print('Torch Tensor {}, shape {}, on {}'.format(key, value.shape, value.device))
        else:
            print(key, value)

def get_remove_keys(dt, keys=[]):
    targets = []
    for key in keys:
        targets.append(dt[key])
    for key in keys:
        del dt[key]
    return targets


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True