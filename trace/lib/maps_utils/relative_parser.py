from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
import sys, os
import constants
from config import args

def parse_age_cls_results(age_probs):
    age_preds = torch.ones_like(age_probs).long()*-1
    age_preds[(age_probs<=constants.age_threshold['adult'][2])&(age_probs>constants.age_threshold['adult'][0])] = 0
    age_preds[(age_probs<=constants.age_threshold['teen'][2])&(age_probs>constants.age_threshold['teen'][0])] = 1
    age_preds[(age_probs<=constants.age_threshold['kid'][2])&(age_probs>constants.age_threshold['kid'][0])] = 2
    age_preds[(age_probs<=constants.age_threshold['baby'][2])&(age_probs>constants.age_threshold['baby'][0])] = 3
    return age_preds

def parse_classfication_results(betas_pred, valid_gender_thresh=0.6):
    if betas_pred.shape[1]==13:
        age_probs = betas_pred[:,10]
    elif betas_pred.shape[1]==3:
        age_probs = betas_pred[:,0]
    age_preds = torch.ones_like(age_probs).long()*-1
    age_preds[(age_probs<=constants.age_threshold['adult'][2])&(age_probs>constants.age_threshold['adult'][0])] = 0
    age_preds[(age_probs<=constants.age_threshold['teen'][2])&(age_probs>constants.age_threshold['teen'][0])] = 1
    age_preds[(age_probs<=constants.age_threshold['kid'][2])&(age_probs>constants.age_threshold['kid'][0])] = 2
    age_preds[(age_probs<=constants.age_threshold['baby'][2])&(age_probs>constants.age_threshold['baby'][0])] = 3

    if betas_pred.shape[1]==13:
        gender_results = betas_pred[:,11:13].max(1)
    elif betas_pred.shape[1]==3:
        gender_results = betas_pred[:,1:3].max(1)
    gender_preds, gender_probs = gender_results.indices, gender_results.values
    invalid_gender_preds_mask = gender_probs<valid_gender_thresh
    gender_preds[invalid_gender_preds_mask] = 2

    class_preds = torch.stack([age_preds, gender_preds], 1)
    class_probs = torch.stack([age_probs, gender_probs], 1)
    return class_preds, class_probs

if __name__ == '__main__':
    betas = torch.rand(8,13)
    print(betas)
    class_preds = parse_classfication_results(betas)
    print(betas,betas[:,10:], class_preds)