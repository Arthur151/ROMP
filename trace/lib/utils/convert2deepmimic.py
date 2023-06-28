import sys, getopt
import pickle as pk
import os
import argparse

import numpy as np
import numpy.linalg as LA
import chumpy as ch
import os.path as OP
from pyquaternion import Quaternion

import math
import pickle
import json
from transformation import *
import os.path as OP

from pyquaternion import Quaternion

from utils_pybullet import displayBulletFrames

def get_hmr_outputs(tool="scythe", videoname="scythe_0001"):
    """
    Returns the HMR output for a given motion and video reference
    """
    tool_path = os.path.join(HMR_OUTPUT_PATH, tool)

    assert videoname in os.listdir(tool_path)
    
    hmr_filepath = os.path.join(tool_path, videoname, "hmr", "hmr.pkl")

    with open(hmr_filepath, 'r') as f:
        data = pk.load(f)

    return data

def get_angle(vec1, vec2):
    cos_theta = np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return math.acos(cos_theta)


def get_quaternion(ox, oy, oz, x, y, z):
    # given transformed axis in x-y-z order return a quaternion
    ox /= np.linalg.norm(ox)
    oy /= np.linalg.norm(oy)
    oz /= np.linalg.norm(oz)

    set1 = np.vstack((ox,oy,oz))

    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)

    set2 = np.vstack((x,y,z))
    rot_mat = superimposition_matrix(set1, set2, scale=False, usesvd=True)
    rot_qua = quaternion_from_matrix(rot_mat)

    return rot_qua


def tr_vect(x):
    return np.array([x[0], x[2], -x[1]])

# 3D coord to deepmimic rotations
def coord_to_rot(frame, frame_duration):
    """
    Converts a 3D description of the 
    """
    # eps = 0.001
    # axis_rotate_rate = 0.3
    # print "frame is:"
    # print frame

    frameCorr = np.array([ tr_vect(frame[i, :]) for i in range(frame.shape[0])])

    video32SMPL = {
        0: np.copy(frame[0]),
        1: np.copy(frame[2]),
        2: np.copy(frame[5]),
        3: np.copy(frame[8]),
        4: np.copy(frame[1]),
        5: np.copy(frame[4]),
        6: np.copy(frame[7]),
        7: np.copy(frame[6]),
        8: 0.5*np.copy(frame[13]) + 0.5*np.copy(frame[14]),
        9: np.copy(frame[12]),
        10: np.copy(frame[15]),
        11: np.copy(frame[16]),
        12: np.copy(frame[18]),
        13: np.copy(frame[20]),
        14: np.copy(frame[17]),
        15: np.copy(frame[19]),
        16: np.copy(frame[21])
    }

    frame = np.array(frame)
    tmp = [[] for i in range(15)]
    # duration of frame in seconds (1D),
    tmp[0] = [frame_duration]

    # root position (3D),
    tmp[1] = tr_vect(frame[0])
    tmp[1][0] += 0.5

    # root rotation (4D),
    x = np.array([1.0,0,0])
    y = np.array([0,1.0,0])
    z = np.array([0,0,1.0])
    
    root_y = frameCorr[6] - frameCorr[0]
    root_z = frameCorr[2] - frameCorr[1]
    root_x = np.cross(root_y, root_z)

    rot_qua = get_quaternion(root_x, root_y, root_z, x, y, z)
    tmp[2] = list(rot_qua)
    # tmp[2] = [1,0,0,0]

    # chest rotation (4D),
    chest_y = frameCorr[9] - frameCorr[6]
    chest_z = frameCorr[17] - frameCorr[16]
    chest_x = np.cross(chest_y, chest_z)
    rot_qua = get_quaternion(chest_x, chest_y, chest_z, root_x, root_y, root_z)
    tmp[3] = list(rot_qua)

    # neck rotation (4D),
    neck_y = (frameCorr[15] - frameCorr[9])
    neck_z = np.cross(frameCorr[15]-frameCorr[12], frameCorr[9]-frameCorr[12]) 
    neck_x = np.cross(neck_y, neck_z)
    rot_qua = get_quaternion(neck_x, neck_y, neck_z, chest_x, chest_y, chest_z)
    tmp[4] = list(rot_qua)
    # tmp[4] = [1,0,0,0]

    # right hip rotation (4D),
    r_hip_y = frameCorr[2] - frameCorr[5]
    r_hip_z = np.cross( frameCorr[2]-frameCorr[5], frameCorr[8]-frameCorr[5]) 
    r_hip_x = np.cross(r_hip_y, r_hip_z)
    rot_qua = get_quaternion(r_hip_x, r_hip_y, r_hip_z, root_x, root_y, root_z)
    tmp[5] = list(rot_qua)

    # right knee rotation (1D),
    vec1 = frameCorr[2] - frameCorr[5]
    vec2 = frameCorr[8] - frameCorr[5]
    angle1 = get_angle(vec1, vec2)
    tmp[6] = [angle1-pi]

    # right ankle rotation (4D),
    # r_knee_y = - (frameCorr[8] - frameCorr[5])
    # r_knee_z = np.cross( frameCorr[2]-frameCorr[5], frameCorr[8]-frameCorr[5]) 
    # r_knee_x = np.cross(r_hip_y, r_hip_z)

    # r_ankle_y = - (frameCorr[11] - frameCorr[8])
    # r_ankle_z = np.cross( frameCorr[5]-frameCorr[8], frameCorr[11]-frameCorr[8]) 
    # r_ankle_x = np.cross(r_ankle_y, r_ankle_z)
    # rot_qua = get_quaternion(r_ankle_x, r_ankle_y, r_ankle_z, r_knee_x, r_knee_y, r_knee_z)
    # tmp[7] = rot_qua
    # rot_qua = get_quaternion(r_knee_x, r_knee_y, r_knee_z, root_x, root_y, root_z)
    # tmp[5] = list(rot_qua)
    # tmp[6] = [0.]
    tmp[7] = [1,0,0,0]

    #  right shoulder rotation (4D),
    r_shou_y = (frameCorr[17] - frameCorr[19])
    r_shou_z = np.cross(frameCorr[21]-frameCorr[19], frameCorr[17]-frameCorr[19]) 
    r_shou_x = np.cross(r_shou_y, r_shou_z)
    rot_qua = get_quaternion(r_shou_x, r_shou_y, r_shou_z, chest_x, chest_y, chest_z)
    tmp[8] = list(rot_qua)
    # tmp[8] = [1,0,0,0]

    # right elbow rotation (1D),
    vec1 = frameCorr[17] - frameCorr[19]
    vec2 = frameCorr[21] - frameCorr[19]
    angle1 = get_angle(vec1, vec2)
    tmp[9] = [pi-angle1]
    # tmp[9] = [0.]

    # left hip rotation (4D),
    l_hip_y = (frameCorr[1] - frameCorr[4])
    l_hip_z = np.cross(frameCorr[1]-frameCorr[4], frameCorr[7]-frameCorr[4]) 
    l_hip_x = np.cross(l_hip_y, l_hip_z)
    rot_qua = get_quaternion(l_hip_x, l_hip_y, l_hip_z, root_x, root_y, root_z)    
    tmp[10] = list(rot_qua)
    
    # left knee rotation (1D),
    vec1 = frameCorr[1] - frameCorr[4]
    vec2 = frameCorr[7] - frameCorr[4]
    angle1 = get_angle(vec1, vec2)
    tmp[11] = [angle1-pi]
    
    # left ankle rotation (4D),
    tmp[12] = [1., 0., 0., 0.]

    # left shoulder rotation (4D),
    l_shou_y = (frameCorr[16] - frameCorr[18])
    l_shou_z = np.cross(frameCorr[20]-frameCorr[18], frameCorr[16]-frameCorr[18]) 
    l_shou_x = np.cross(l_shou_y, l_shou_z)
    rot_qua = get_quaternion(l_shou_x, l_shou_y, l_shou_z, chest_x, chest_y, chest_z)
    tmp[13] = list(rot_qua)
    # tmp[13] = [1,0,0,0]

    # left elbow rotation (1D)
    vec1 = frameCorr[16] - frameCorr[18]
    vec2 = frameCorr[20] - frameCorr[18]
    angle1 = get_angle(vec1, vec2)
    tmp[14] = [pi-angle1]
    # tmp[14] = [0.]

    ret = []
    for i in tmp:
        ret += list(i)
    return np.array(ret)
    

def SMPLcoords_to_rots(joint_3d_positions, frame_duration):
    nFrames = joint_3d_positions.shape[0]
    frames = np.zeros((nFrames, 44))

    for i in range(nFrames):
        frames[i, :] = coord_to_rot(joint_3d_positions[i], frame_duration)

    return frames

def hmroutput_to_joints3d(hmrdata):
    # Theta is the 85D vector holding [camera, pose, shape] where:
    # -3        : camera is 3D [s, tx, ty]
    # -72 = 3*24: pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # -10       : shape is 10D shape coefficients of SMPL
    thetas = hmrdata['thetas']
    nFrames = len(thetas)

    joint_3d_positions = np.zeros((nFrames, 24, 3))

    for ind in range(nFrames):
        theta = thetas[ind][3:75]
        beta = thetas[ind][75:]

        cur_joint3d = getJointPositionsFromSMPL(beta, theta)
        joint_3d_positions[ind, :] = SMPLPose_to_Hmu3dSpace(cur_joint3d)

    return joint_3d_positions



def wrapAll():

    hmrdata = get_hmr_outputs()

    # Build SMPL pose for each frame
    joint_filename = "SMPL_joints.pkl"
    if not OP.isfile(joint_filename):
        print("Building SMPL poses...")

        joint_3d_positions = hmroutput_to_joints3d(hmrdata)
        with open(joint_filename, 'wb') as handle:
            pickle.dump(joint_3d_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading SMPL poses...")

        with open(joint_filename, 'rb') as handle:
            joint_3d_positions = pickle.load(handle)


    # Build humanoid3 frame descr from SMPL pose
    print("Converting to DeepMimic frames...")
    frames = SMPLcoords_to_rots(joint_3d_positions, 0.1)

    deepmimicdict = {
        "Loop": "none",
        "Frames": frames.tolist()
    }

    with open("output.json", "w") as f:
        json.dump(deepmimicdict, f)
    
    if False:
        displayBulletFrames(frames, actualSMPLPoses=joint_3d_positions)

    return 0


if __name__ == "__main__":
    print("")

    wrapAll()
    assert False

    hmrdata = get_hmr_outputs()
    thetas = hmrdata['thetas']

    ind = 20
    theta = thetas[ind][3:75]
    beta = thetas[ind][75:]

    joint_3d_position = getJointPositionsFromSMPL(beta, theta)
    joint_3d_position_corr = SMPLPose_to_Hmu3dSpace(joint_3d_position)

    bulletFrame = coord_to_rot(joint_3d_position_corr, 1)
    displayBulletFrames([bulletFrame, bulletFrame], actualSMPLPoses=[joint_3d_position_corr, joint_3d_position_corr])


def SMPLPose_to_Hmu3dSpace(joint_3d_positions):
    joint_3d_normalized = np.zeros(joint_3d_positions.shape)

    ## Rotation
    q_rot = Quaternion(axis=[1, 0, 0], radians=-np.pi / 2)

    for ind, vec3 in enumerate(joint_3d_positions):
        joint_3d_normalized[ind, :] = q_rot.rotate(vec3)


    ## Translation s.t. feets touch fround
    minz = joint_3d_normalized[:, 2].min()
    translation = np.array([0, 0, -minz])

    for ind, vec3 in enumerate(joint_3d_positions):
        joint_3d_normalized[ind, :] += translation

    return joint_3d_normalized  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hmrpkl", type=str,
                        help="Location of the hmr pkl file")
    parser.add_argument("outjson", type=str,
                        help="Location where output json file will be written")
    parser.add_argument("-da", "--doall", action="store_true",
                        help="Process all videos", default=False)
    args = parser.parse_args()


    if not args.doall:
        processOneVideo(args.hmrpkl, args.outjson)
    else:
        processVideos(args.hmrpkl, args.outjson)


def processVideos(inputfolder, outputfolder):

    for tool in ['hammer', 'scythe']:
        tool_path = os.path.join(inputfolder, tool)

        for subdir in os.listdir(tool_path):
            if tool in subdir:
                if not os.path.isfile(os.path.join(outputfolder, subdir+".json")):
                    hmr_filepath = os.path.join(tool_path, subdir, "hmr", "hmr.pkl")
                    processOneVideo(hmr_filepath, os.path.join(outputfolder, subdir+".json"))

    
def processOneVideo(pklfile, outjsonfile):
    """
    Turn the HMR pose dict for a sequence of videos into a DeepMimic compatible
    motion reference file
    """

    print "Loading hmr output...", pklfile
    with open(pklfile, 'r') as f:
        hmrdata = pk.load(f)

    # Build SMPL pose for each frame
    print "Building SMPL pose..."
    joint_3d_positions = hmroutput_to_joints3d(hmrdata)
    
    # Build humanoid3 frame descr from SMPL pose
    print "Converting to DeepMimic frames..."
    frames = SMPLcoords_to_rots(joint_3d_positions, 0.1)

    print "\nWriting frames json"
    deepmimicdict = {
        "Loop": "none",
        "Frames": frames.tolist()
    }

    with open(outjsonfile, "w") as f:
        json.dump(deepmimicdict, f)

    print '\nOutput file written at ', outjsonfile
    return

if __name__ == "__main__":
   main()