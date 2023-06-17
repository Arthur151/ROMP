import torch
import glob
import numpy as np
import cv2
import os

def prepare_bare_temporal_inputs(clip_length=8, batch_size=2):
    inputs = {'image':torch.rand(batch_size*clip_length,512,512,3).float().cuda(), 
                'sequence_mask':torch.ones(batch_size*clip_length).bool().cuda()}
    return inputs

def prepare_bare_temporal_feature_maps(clip_length=8, batch_size=2):
    inputs = {'image_feature_maps':torch.rand(batch_size*clip_length,32,128,128).float().cuda(), 
                'sequence_mask':torch.ones(batch_size*clip_length).bool().cuda()}
    return inputs

def pad_image2square(image):
    img_size = image.shape
    # padding to square image
    h, w = img_size[:2]
    max_edge = max(img_size)
    pad_size = np.abs((h-w)//2)
    if max_edge == h:
        pad_image = np.zeros((h, pad_size, 3), dtype=np.uint8)
        image = np.concatenate([pad_image, image, pad_image], 1)
    elif max_edge == w:
        pad_image = np.zeros((pad_size, w, 3), dtype=np.uint8)
        image = np.concatenate([pad_image, image, pad_image], 0)
    return image

def prepare_video_clip_inputs(clip_length=8, batch_size=2, video_path=None, input_size=(512, 512)):
    if video_path is None:
        return prepare_bare_temporal_inputs(clip_length=clip_length, batch_size=batch_size)
    
    image_num = batch_size*clip_length
    image_tensor = torch.zeros(image_num, *input_size, 3).float()
    image_list = sorted(glob.glob(os.path.join(video_path, '*')))
    for ind in range(image_num):
        image_path = image_list[ind % len(image_list)]
        image = cv2.imread(image_path)
        image = pad_image2square(image)
        image_tensor[ind] = torch.from_numpy(cv2.resize(image, input_size, interpolation=cv2.INTER_CUBIC))

    inputs = {'image':image_tensor.float().cuda(), 'sequence_mask':torch.ones(image_num).bool().cuda()}
    return inputs

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = '', drop_prefix='', fix_loaded=False):
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
                failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix!='':
                k=k.split(prefix)[1]
            success_layers.append(k)
        except:
            print('copy param {} failed, mismatched'.format(k)) # logging.info
            continue
    print('missing parameters of layers:{}'.format(failed_layers))

    return success_layers

def prepare_bev_model(checkpoint):
    from models.hrnet_32 import HigherResolutionNet
    from models.bev import BEV
    image_model = BEV(backbone=HigherResolutionNet())
    copy_state_dict(image_model.state_dict(), torch.load(checkpoint), prefix='module.')
    return image_model