import numpy as np
import cv2
import os,sys
import torch
import quaternion

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format),dtype=np.int32)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)

SMPL_24 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'SMPL_Head':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23}
SMPL_Face_Foot_11 = {
    'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28, \
    'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34}
SMPL_EXTRA_9 = {
    'R_Hip': 35, 'L_Hip':36, 'Neck_LSP':37, 'Head_top':38, 'Pelvis':39, 'Thorax_MPII':40, \
    'Spine_H36M':41, 'Jaw_H36M':42, 'Head':43}
SMPL_ALL_44 = {**SMPL_24, **SMPL_Face_Foot_11, **SMPL_EXTRA_9}

COCO_17 = {
    'Nose':0, 'L_Eye':1, 'R_Eye':2, 'L_Ear':3, 'R_Ear':4, 'L_Shoulder':5, 'R_Shoulder':6, 'L_Elbow':7, 'R_Elbow':8, \
    'L_Wrist': 9, 'R_Wrist':10, 'L_Hip':11, 'R_Hip':12, 'L_Knee':13, 'R_Knee':14, 'L_Ankle':15, 'R_Ankle':16}
GLAMR_26 = {
    'L_Hip': 1, 'R_Hip':2, 'L_Knee':4, 'R_Knee':5, 'L_Ankle':7, 'R_Ankle':8, 'L_Shoulder':20, 'R_Shoulder':21,'L_Elbow':22, 'R_Elbow':23}
glamr_mapping2D = joint_mapping(GLAMR_26, SMPL_ALL_44)

def rotation_matrix_to_angle_axis(rotmats):
    rotmats = rotmats.numpy()
    aas = np.array([cv2.Rodrigues(rotmat)[0][:,0] for rotmat in rotmats])
    print(aas.shape)
    return torch.from_numpy(aas).float()

def angle_axis_to_rotation_matrix(aas):
    aas = aas.numpy()
    rotmats = np.array([cv2.Rodrigues(aa)[0] for aa in aas])
    print(rotmats.shape)
    return torch.from_numpy(rotmats).float()

def angle2mat(angle):
    return quaternion.as_rotation_matrix(quaternion.from_rotation_vector(angle))
def mat2angle(mat):
    return quaternion.as_rotation_vector(quaternion.from_rotation_matrix(mat))
def angle2quaternion(angle):
    return quaternion.as_float_array(quaternion.from_rotation_vector(angle))

def search_valid_frame(frame2ind, frame_id):
    start_id = sorted(list(frame2ind.keys()))[0]
    if frame_id < start_id:
        #print('smaller than start_id', start_id)
        while 1:
            frame_id = frame_id+1
            if frame_id in frame2ind:
                break
    else:
        while frame_id > 0:
            frame_id = frame_id-1
            if frame_id in frame2ind:
                break
    return frame_id