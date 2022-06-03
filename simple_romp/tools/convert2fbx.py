# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#
# Author: Joachim Tesch, Max Planck Institute for Intelligent Systems, Perceiving Systems
#
# Create keyframed animated skinned SMPL mesh from .pkl pose description
#
# Generated mesh will be exported in FBX or glTF format
#
# Notes:
#  + Male and female gender models only
#  + Script can be run from command line or in Blender Editor (Text Editor>Run Script)
#  + Command line: Install mathutils module in your bpy virtualenv with 'pip install mathutils==2.81.2'

import os
import sys
import time
import argparse
import numpy as np
from math import radians

try:
    import bpy
except:
    print('Missing bpy, install via pip, please install bpy by yourself if failed.')
    os.system('pip install future-fstrings')
    os.system('pip install tools/bpy-2.82.1 && post_install')
    import bpy
try:
    from mathutils import Matrix, Vector, Quaternion, Euler
except:
    os.system('pip install mathutils==2.81.2')
    from mathutils import Matrix, Vector, Quaternion, Euler


# Globals
# Add your UNIX paths here!
male_model_path = '/home/yusun/ROMP/model_data/SMPL_unity_v.1.0.0/smpl/Models/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'
female_model_path = '/home/yusun/ROMP/model_data/SMPL_unity_v.1.0.0/smpl/Models/SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'
character_model_path = None

'''
python tools/convert2fbx.py --input=/home/yusun/BEV_results/video_results.npz --output=/home/yusun/BEV_results/dance.fbx --gender=female
'''

fps_source = 24
fps_target = 24

gender = 'male' #female

support_formats = ['.fbx', '.glb', '.bvh']

bone_name_from_index = {
    0 : 'Pelvis',
    1 : 'L_Hip',
    2 : 'R_Hip',
    3 : 'Spine1',
    4 : 'L_Knee',
    5 : 'R_Knee',
    6 : 'Spine2',
    7 : 'L_Ankle',
    8: 'R_Ankle',
    9: 'Spine3',
    10: 'L_Foot',
    11: 'R_Foot',
    12: 'Neck',
    13: 'L_Collar',
    14: 'R_Collar',
    15: 'Head',
    16: 'L_Shoulder',
    17: 'R_Shoulder',
    18: 'L_Elbow',
    19: 'R_Elbow',
    20: 'L_Wrist',
    21: 'R_Wrist',
    22: 'L_Hand',
    23: 'R_Hand'
}

# To use other avatar for animation, please define the corresponding 3D skeleton like this.
bone_name_from_index_character = {
    0 : 'Hips',
    1 : 'RightUpLeg',
    2 : 'LeftUpLeg',
    3 : 'Spine',
    4 : 'RightLeg',
    5 : 'LeftLeg',
    6 : 'Spine1',
    7 : 'RightFoot',
    8: 'LeftFoot',
    9: 'Spine2',
    10: 'LeftToeBase',
    11: 'RightToeBase',
    12: 'Neck',
    13: 'LeftHandIndex1',
    14: 'RightHandIndex1',
    15: 'Head',
    16: 'LeftShoulder',
    17: 'RightShoulder',
    18: 'LeftArm',
    19: 'RightArm',
    20: 'LeftForeArm',
    21: 'RightForeArm',
    22: 'LeftHand',
    23: 'RightHand'
}

# Helper functions

# Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
# Source: smpl/plugins/blender/corrective_bpy_sh.py
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)


# Setup scene
def setup_scene(model_path, fps_target):
    scene = bpy.data.scenes['Scene']

    ###########################
    # Engine independent setup
    ###########################

    scene.render.fps = fps_target

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    # Import gender specific .fbx template file
    bpy.ops.import_scene.fbx(filepath=model_path)


# Process single pose into keyframed bone orientations
def process_pose(current_frame, pose, trans, pelvis_position):

    if pose.shape[0] == 72:
        rod_rots = pose.reshape(24, 3)
    else:
        rod_rots = pose.reshape(26, 3)

    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]

    # Set the location of the Pelvis bone to the translation parameter
    armature = bpy.data.objects['Armature']
    bones = armature.pose.bones

    # Pelvis: X-Right, Y-Up, Z-Forward (Blender -Y)
    root_location = Vector(
        (100*trans[1], 100*trans[2], 100*trans[0])) - pelvis_position
    # Set absolute pelvis location relative to Pelvis bone head
    bones[bone_name_from_index[0]].location = root_location
            
    # bones['Root'].location = Vector(trans)
    bones[bone_name_from_index[0]].keyframe_insert('location', frame=current_frame)

    for index, mat_rot in enumerate(mat_rots, 0):
        if index >= 24:
            continue

        bone = bones[bone_name_from_index[index]]

        bone_rotation = Matrix(mat_rot).to_quaternion()
        quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-90))
        #quat_x_n135_cw = Quaternion((1.0, 0.0, 0.0), radians(-135))
        #quat_x_p45_cw = Quaternion((1.0, 0.0, 0.0), radians(45))
        #quat_y_90_cw = Quaternion((0.0, 1.0, 0.0), radians(-90))
        quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))

        if index == 0:
            # Rotate pelvis so that avatar stands upright and looks along negative Y avis
            bone.rotation_quaternion = (quat_x_90_cw @ quat_z_90_cw) @ bone_rotation
        else:
           bone.rotation_quaternion = bone_rotation

        bone.keyframe_insert('rotation_quaternion', frame=current_frame)

    return


# Process all the poses from the pose file
def process_poses(
        input_path,
        gender,
        fps_source,
        fps_target,
        subject_id=-1):

    print('Processing: ' + input_path)

    frame_results = np.load(input_path, allow_pickle=True)['results'][()]
    sequence_results = np.load(input_path, allow_pickle=True)['sequence_results'][()] 

    poses, trans = [], []

    if len(sequence_results)>0:
        subject_ids = list(sequence_results.keys())
        if subject_id == -1 or subject_id not in subject_ids:
            print('Get motion sequence with subject IDs:', subject_ids)
            subject_id = int(input('Please select one subject ID (int):'))
        poses = np.array(sequence_results[subject_id]['smpl_thetas'])
        trans = np.array(sequence_results[subject_id]['cam_trans'])
    else:
        print('Missing tracking IDs in results. Using the first pose results for animation.')
        print('To get the tracking IDs, please use temporal optimization during inference.')
        frame_names = sorted(list(frame_results.keys()))
        poses, trans = np.zeros((len(frame_names), 72)), np.zeros((len(frame_names), 3))
        for inds, frame_name in enumerate(frame_names):
            poses[inds] = frame_results[frame_name]['smpl_thetas'][0]
            trans[inds] = frame_results[frame_name]['cam_trans'][0]

    if gender == 'female':
        model_path = female_model_path
        for k,v in bone_name_from_index.items():
            bone_name_from_index[k] = 'f_avg_' + v
    elif gender == 'male':
        model_path = male_model_path
        for k,v in bone_name_from_index.items():
            bone_name_from_index[k] = 'm_avg_' + v
    elif gender == 'character':
        model_path = character_model_path
        for k, v in bone_name_from_index_character.items():
            bone_name_from_index[k] = 'mixamorig1:' + v
    else:
        print('ERROR: Unsupported gender: ' + gender)
        sys.exit(1)

    # Limit target fps to source fps
    if fps_target > fps_source:
        fps_target = fps_source

    print('Gender:',gender)
    print('Number of source poses: ',poses.shape[0])
    print('Source frames-per-second: ', fps_source)
    print('Target frames-per-second: ', fps_target)
    print('--------------------------------------------------')

    setup_scene(model_path, fps_target)

    scene = bpy.data.scenes['Scene']
    sample_rate = int(fps_source/fps_target)
    scene.frame_end = (int)(poses.shape[0]/sample_rate)

    # Retrieve pelvis world position.
    # Unit is [cm] due to Armature scaling.
    # Need to make copy since reference will change when bone location is modified.
    armaturee = bpy.data.armatures[0]
    ob = bpy.data.objects['Armature']
    armature = ob.data

    bpy.ops.object.mode_set(mode='EDIT')
    # get specific bone name 'Bone'
    pelvis_bone = armature.edit_bones[bone_name_from_index[0]]
    # pelvis_bone = armature.edit_bones['f_avg_Pelvis']
    pelvis_position = Vector(pelvis_bone.head)
    bpy.ops.object.mode_set(mode='OBJECT')

    source_index = 0
    frame = 1

    offset = np.array([0.0, 0.0, 0.0])

    while source_index < poses.shape[0]:
        print('Adding pose: ' + str(source_index))

        # Go to new frame
        scene.frame_set(frame)

        process_pose(frame, poses[source_index], (trans[source_index] - offset), pelvis_position)
        source_index += sample_rate
        frame += 1

    return frame

def rotate_armature(use):
    if use == True:
        # Switch to Pose Mode
        bpy.ops.object.posemode_toggle()
        
        # Find the Armature & Bones
        ob = bpy.data.objects['Armature']
        armature = ob.data
        bones = armature.bones
        rootbone = bones[0]
        
        # Find the Root bone
        for bone in bones:
            if "avg_root" in bone.name:
                rootbone = bone
                
        rootbone.select = True
        
        # Rotate the Root bone by 90 euler degrees on the Y axis. Set --rotate_Y=False if the rotation is not needed.
        bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        # Revert back to Object Mode
        bpy.ops.object.posemode_toggle()
        


def export_animated_mesh(output_path):
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Fix Rotation
    rotate_armature(args.rotate_y)

    # Select only skinned mesh and rig
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Armature'].select_set(True)
    bpy.data.objects['Armature'].children[0].select_set(True)

    if output_path.endswith('.glb'):
        print('Exporting to glTF binary (.glb)')
        # Currently exporting without shape/pose shapes for smaller file sizes
        bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', export_selected=True, export_morph=False)
    elif output_path.endswith('.fbx'):
        print('Exporting to FBX binary (.fbx)')
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, add_leaf_bones=False)
    elif output_path.endswith('.bvh'):
        bpy.ops.export_anim.bvh(filepath=output_path, root_transform_only=False)
    else:
        print('ERROR: Unsupported export format: ' + output_path)
        sys.exit(1)

    return


if __name__ == '__main__':
    if bpy.app.background:

        parser = argparse.ArgumentParser(description='Create keyframed animated skinned SMPL mesh from VIBE output')
        parser.add_argument('--input', dest='input_path', type=str, default='demo/juntiquan_results/juntiquan_frames_ts_results.npz', #'../demo/videos/sample_video2_results.npz',
                            help='Input file or directory')
        parser.add_argument('--output', dest='output_path', type=str, default='demo/videos/jtq.fbx', #'../demo/videos/sample_video2.fbx',
                            help='Output file or directory')
        parser.add_argument('--fps_source', type=int, default=fps_source,
                            help='Source framerate')
        parser.add_argument('--fps_target', type=int, default=fps_target,
                            help='Target framerate')
        parser.add_argument('--gender', type=str, default=gender,
                            help='Always use specified gender')
        parser.add_argument('--subject_id', type=int, default=-1,
                            help='Detected person ID to use for fbx animation')
        parser.add_argument('--rotate_y',type = bool,default=True,help = 'whether to rotate the root bone on the Y axis by -90 on export. Otherwise it may be rotated incorrectly')
        
        args = parser.parse_args()

        input_path = args.input_path
        output_path = args.output_path
        
        print('Input path: ' + input_path)
        print('Output path: ' + output_path)

        if not os.path.exists(input_path):
            print('ERROR: Invalid input path')
            sys.exit(1)

        fps_source = args.fps_source
        fps_target = args.fps_target
        gender = args.gender

    startTime = time.perf_counter()
    cwd = os.getcwd()
    # Turn relative input/output paths into absolute paths
    if not input_path.startswith(os.path.sep):
        input_path = os.path.join(cwd, input_path)
    if not output_path.startswith(os.path.sep):
        output_path = os.path.join(cwd, output_path)
    
    if os.path.splitext(output_path)[1] not in support_formats:
        print('ERROR: Invalid output format, we only support', support_formats)
        sys.exit(1)

    # Process pose file
    poses_processed = process_poses(
            input_path=input_path,
            gender=gender,
            fps_source=fps_source,
            fps_target=fps_target,
            subject_id=args.subject_id
        )
    export_animated_mesh(output_path)

    print('--------------------------------------------------')
    print('Animation export finished, save to ', output_path)
    print('Poses processed: ', poses_processed)
    print('Processing time : ', time.perf_counter() - startTime)
    print('--------------------------------------------------')
