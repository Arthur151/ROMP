import numpy as np 

Dyna3DPW_sequences = ['downtown_walkBridge_01', 'downtown_runForBus_01', 'downtown_sitOnStairs_00', \
                    'downtown_bar_00', 'downtown_cafe_00', 'downtown_enterShop_00', \
                    'downtown_rampAndStairs_00', 'downtown_runForBus_00', 'downtown_windowShopping_00', \
                    'downtown_car_00',  'downtown_walking_00', 'downtown_crossStreets_00',\
                    'office_phoneCall_00', 'downtown_warmWelcome_00', 'downtown_weeklyMarket_00', 'downtown_bus_00']

CMU_Panoptic_video_seqs = [
    '160422_haggling1-00_16-1', '160422_haggling1-00_16-2', '160422_haggling1-00_30-1', '160422_haggling1-00_30-2',
    '160422_mafia2-00_16-1', '160422_mafia2-00_16-2', '160422_mafia2-00_30-1', '160422_mafia2-00_30-2',
    '160906_pizza1-00_30-1', '160906_pizza1-00_30-2'
]

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format),dtype=np.int32)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)

joint2D_tree = {
    'L_Shoulder':['SMPL_Head', 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear'],\
    # not all skeleton from different datasets have Neck, while they all have Shoulder
    #'Neck':['SMPL_Head', 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear'],\
    #'Neck_LSP':['SMPL_Head', 'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear'],\
    'L_Ankle':['L_BigToe', 'L_SmallToe', 'L_Heel', 'L_Toe_SMPL'],\
    'R_Ankle':['R_BigToe', 'R_SmallToe', 'R_Heel', 'R_Toe_SMPL'],\
}

SMPL_24 = {
    'Pelvis_SMPL':50, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':0, 'R_Toe_SMPL':0, 'Neck': 0, 'L_Collar':13, 'R_Collar':14, 'SMPL_Head':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23}

SMPL_24_names = ['Pelvis_SMPL', 'L_Hip_SMPL', 'R_Hip_SMPL', 'Spine_SMPL', 'L_Knee', 'R_Knee', 'Thorax_SMPL', 'L_Ankle', 'R_Ankle','Thorax_up_SMPL', \
    'L_Toe_SMPL', 'R_Toe_SMPL', 'Neck', 'L_Collar', 'R_Collar', 'SMPL_Head', 'L_Shoulder', 'R_Shoulder',\
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']

op25_to_smpl_mapper = np.array([8, 12, 9, 1, 13, 10, 1, 14, 11, 1, 20, 23, 1, 5, 2, 1, 5, 2, 6, 3, 7, 4])

SMPL_Face_Foot_11 = {
    'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28, \
    'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34
}
SMPL_EXTRA_9 = {
    'R_Hip': 35, 'L_Hip':36, 'Neck_LSP':37, 'Head_top':38, 'Pelvis':39, 'Thorax_MPII':40, \
    'Spine_H36M':41, 'Jaw_H36M':42, 'Head':43}

#                            0  1  2  3    4   5   6   7   8    9  10  11  12  13
keypoints_select = np.array([4, 5, 7, 8,  16, 17, 18, 19,  20, 21, 24, 35, 36, 12]) # Neck_LSP is not avaialbe in video datasets
keypoints_select_connMat = np.array([[0,2], [1,3],  [13,4], [13,5], [4,6], [5,7], [6,8], [7,9], [10,13], [11,1], [12,0]])

SMPL_ALL_44 = {**SMPL_24, **SMPL_Face_Foot_11, **SMPL_EXTRA_9}
torso_joint_inds = np.concatenate([np.arange(22), np.arange(35,44)], 0)

smpl24_flip = np.array([0,2,1,3,5,4,6,8,7,9,11,10,12,14,13,15,17,16,19,18,21,20,23,22]).astype(np.int32)
smpl_extra30_flip = np.array([24, 26,25, 28,27, 32,33,34, 29,30,31, 40,41,42,43,44, 35,36,37,38,39, 46,45, 47, 48,49,50,51,52,53]).astype(np.int32)
All54_flip = np.concatenate([smpl24_flip, smpl_extra30_flip],0)

smpl_extra20_flip = np.array([24, 26,25, 28,27, 32,33,34, 29,30,31, 36,35, 37, 38,39,40,41,42,43]).astype(np.int32)
All44_flip = np.concatenate([smpl24_flip, smpl_extra20_flip],0)

smpl24_connMat = np.array([0,1, 0,2, 0,3, 1,4,4,7,7,10, 2,5,5,8,8,11, 3,6,6,9,9,12,12,15, 12,13, 13,16, 16,18, 18,20, 12,14,14,17,17,19,19,21]).reshape(21, 2) # 21, 2
cm_smpl24 = np.array([[255,0,85],   [255,0,0],[255,85,0],[255,170,0],   [255,255,0],[170,255,0],[85,255,0],  [0,255,0],  [255,0,0],[0,255,85],[0,255,170],  \
 [0,255,255],[0,170,255],[0,85,255],  [0,0,255],[255,0,170],  [170,0,255],[255,0,255],  [85,0,255], [0,0,255],[0,0,255],[0,0,255],   [0,255,255]])[:,::-1]
# joint connection relationship for two hands, two feet, face, tow lsp hips, neck and head
All44_connMat = np.concatenate([smpl24_connMat, np.array([
    [7, 29], [7, 31], [29, 30], [8, 32], [8, 34], [32, 33], \
    [24, 25], [25, 27], [24, 26], [26, 28], \
    [35, 39], [35, 5], [36, 39], [36, 4], \
    [37, 16], [37, 17], [37, 38], [37, 40], [41, 39], [41, 40], [12, 40], [42, 37], [42, 12], [43, 37], [43, 12] 
    ]) ], 0)


SMPLX_55 = {
    'Pelvis_SMPL': 0, 'L_Hip_SMPL': 1, 'R_Hip_SMPL': 2, 'Spine_SMPL': 3, 'L_Knee': 4, 'R_Knee': 5, 'Thorax_SMPL': 6, 'L_Ankle': 7, 'R_Ankle': 8, 'Thorax_up_SMPL': 9, \
    'L_Toe_SMPL': 10, 'R_Toe_SMPL': 11, 'Neck': 12, 'L_Collar': 13, 'R_Collar': 14, 'SMPL_Head': 15, 'L_Shoulder': 16, 'R_Shoulder': 17, 'L_Elbow': 18, 'R_Elbow': 19, \
    'L_Wrist': 20, 'R_Wrist': 21, 
    'Jaw': 22, 'L_Eye': 23, 'R_Eye': 24, 
    'L_Index1': 25, 'L_Index2': 26, 'L_Index3': 27, 'L_Middle1': 28, 'L_Middle2': 29, 'L_Middle3': 30,
    'L_Pinky1': 31, 'L_Pinky2': 32, 'L_Pinky3': 33, 'L_Ring1': 34, 'L_Ring2': 35, 'L_Ring3': 36, 'L_Thumb1': 37, 'L_Thumb2': 38, 'L_Thumb3': 39,    
    'R_Index1': 40, 'R_Index2': 41, 'R_Index3': 42, 'R_Middle1': 43,  'R_Middle2': 44, 'R_Middle3': 45, 'R_Pinky1': 46, 'R_Pinky2': 47, 'R_Pinky3': 48,
    'R_Ring1': 49, 'R_Ring2': 50, 'R_Ring3': 51, 'R_Thumb1': 52, 'R_Thumb2': 53, 'R_Thumb3': 54}

SMPLX_EXTRA_18 = {
    'Nose':55, 'R_Ear': 56, 'L_Ear': 57, 
    'L_BigToe': 58, 'L_SmallToe': 59, 'L_Heel': 60, 'R_BigToe': 61, 'R_SmallToe': 62, 'R_Heel': 63,
    'R_Hip': 64, 'L_Hip': 65, 'Neck_LSP': 66, 'Head_top': 67, 'Pelvis': 68, 'Thorax_MPII': 69, 
    'Spine_H36M': 70, 'Jaw_H36M': 71, 'Head': 72}

smplx_body_inds = np.arange(22)
smplx_left_hand_inds = list(range(25, 40))
smplx_right_hand_inds = list(range(40, 55))
smplx_face_foot_inds = list(range(22, 25)) + list(range(55,58))
smplx_foot_inds = list(range(58,64))
SMPLX_ALL_73 = {**SMPLX_55, **SMPLX_EXTRA_18}

smplx25_flip = np.array([0,2,1,3,5,4,6,8,7,9,11,10,12,14,13,15,17,16,19,18,21,20,22,24,23]).astype(np.int32)
smplx_hand_flip = np.array(list(range(40,55)) + list(range(25,40)))
smplx_extra18_flip = np.array([55,57,56, 61,62,63, 58,59,60, 65,64,66,67,68,69,70,71,72]).astype(np.int32)
All73_flip = np.concatenate([smplx25_flip, smplx_hand_flip, smplx_extra18_flip],0)

All73_connMat = np.concatenate([smpl24_connMat, np.array([
    [7, 58], [7, 60], [58, 59], [8, 61], [8, 63], [61, 62], \
    [55, 24], [24, 56], [55, 23], [23, 57], \
    [64, 68], [64, 5], [65, 68], [65, 4], \
    [66, 16], [66, 17], [66, 67], [66, 69], [70, 68], [70, 69], [12, 69], [71, 66], [71, 12], [72, 66], [72, 12] 
    ]) ], 0)

# root 0/spin 3/thorax 6/Thorax_up 9 rotation is still at body center;  
# L_Hip -> L_Hip_SMPL; R_Hip -> R_Hip_SMPL
joint_sampler_source_name = ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Ankle', 'R_Ankle', \
                            'Neck', 'L_Shoulder', 'R_Shoulder', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
joint_sampler_target_name = ['L_Hip_SMPL', 'R_Hip_SMPL', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Toe_SMPL', 'R_Toe_SMPL', \
                            'Neck', 'L_Collar', 'R_Collar', 'SMPL_Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
joint_sampler_relationship = np.array([SMPL_24[joint_name] for joint_name in joint_sampler_target_name])
joint_sampler_mapper = np.array([SMPLX_ALL_73[joint_name] for joint_name in joint_sampler_source_name])
joint_sampler_connMat = np.array([
    [0, 1], [0, 2], [2, 4], [4, 6], [1, 3], [3, 5], [5, 7], \
    [11, 12], [11, 13], [12, 14], [14, 16], [13, 15], [15, 17]])


# Other datasets___________________________________________________________________________________________


OpenPose_25 = {
    'Nose':0, 'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, \
    'L_Wrist':7, 'Pelvis':8, 'R_Hip': 9, 'R_Knee':10, 'R_Ankle':11, 'L_Hip':12, 'L_Knee':13, 'L_Ankle':14, \
    'R_Eye':15, 'L_Eye':16, 'R_Ear':17, 'L_Ear':18, 'L_BigToe':19, 'L_SmallToe':20, 'L_Heel':21, 'R_BigToe':22, 'R_SmallToe':23, 'R_Heel':24}

#OpenPose_25_kploss_weight = np.array([0.6, 1., 1., 1.16, 1.28, 1, 1.16, 1.28, 0.8, 0.6, 1.2, 1.4, 0.6, 1.2, 1.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,0.4, 0.4,0.4, 0.4])

BK_19 = {
    'Head_top': 0, 'Nose': 1, 'Neck': 2, 'L_Eye': 3, 'R_Eye': 4, 'L_Shoulder': 5, 'R_Shoulder': 6, 'L_Elbow': 7, 'R_Elbow': 8, 'L_Wrist': 9, 'R_Wrist': 10,\
    'L_Hip': 11, 'R_Hip': 12, 'L_Knee':13, 'R_Knee':14,'L_Ankle':15, 'R_Ankle':16,'L_BigToe':17, 'R_BigToe':18}

JRDB_17 = {
    'Head_top': 0, 'R_Eye': 1, 'L_Eye': 2, 'R_Shoulder': 3, 'Neck': 4, 'L_Shoulder': 5,  'R_Elbow': 6, 'L_Elbow': 7, 'Pelvis_SMPL': 8, 'R_Wrist': 9, 'R_Hip_SMPL': 10,\
    'L_Hip_SMPL': 11, 'L_Wrist': 12, 'R_Knee':13, 'L_Knee':14,'R_Ankle':15, 'L_Ankle':16}

SPIN_24 = {
    'R_Ankle':0, 'R_Knee':1, 'R_Hip':2, 'L_Hip': 3, 'L_Knee':4, 'L_Ankle':5, 'R_Wrist': 6, 'R_Elbow':7, 'R_Shoulder':8,\
    'L_Shoulder':9, 'L_Elbow':10, 'L_Wrist':11, 'Neck_LSP': 12, 'Head_top':13, 'Pelvis':14, 'Thorax_MPII':15, 'Spine_H36M':16, 'Jaw_H36M': 17, 'Head':18,\
    'Nose':19, 'L_Eye':20, 'R_Eye':21, 'L_Ear': 22, 'R_Ear': 23}

GLAMR_26 = {
    'L_Hip': 1, 'R_Hip':2, 'L_Knee':4, 'R_Knee':5, 'L_Ankle':7, 'R_Ankle':8, 'L_Shoulder':20, 'R_Shoulder':21,'L_Elbow':22, 'R_Elbow':23}

SMART_25 = {
    'Nose':0, 'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, \
    'L_Wrist':7, 'Pelvis':8, 'R_Hip': 9, 'R_Knee':10, 'R_Ankle':11, 'L_Hip':12, 'L_Knee':13, 'L_Ankle':14, \
    'R_Eye':15, 'L_Eye':16, 'R_Ear':17, 'L_Ear':18, 'L_BigToe':19, 'L_SmallToe': 20, 'L_Heel':21, 'R_BigToe':22,'R_SmallToe':23, 'R_Heel':24, }

PennAction_13 = {
    'Head_top':0, 'R_Shoulder':1, 'L_Shoulder':2, 'R_Elbow':3, 'L_Elbow':4, 'R_Wrist':5, 'L_Wrist':6,\
    'R_Hip':7, 'L_Hip':8, 'R_Knee':9, 'L_Knee':10, 'R_Ankle':11, 'L_Ankle':12}

# LSP
LSP_14 = {
    'R_Ankle':0, 'R_Knee':1, 'R_Hip':2, 'L_Hip':3, 'L_Knee':4, 'L_Ankle':5, 'R_Wrist':6, 'R_Elbow':7, \
    'R_Shoulder':8, 'L_Shoulder':9, 'L_Elbow':10, 'L_Wrist':11, 'Neck_LSP':12, 'Head_top':13}

LSP_14_names = ['R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', \
    'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'Neck_LSP', 'Head_top']

#Human3.6M
H36M_32 = {
    'Unkown_part0':0, 'R_Hip':1, 'R_Knee':2, 'R_Ankle':3, 'R_BigToe':4, 'R_SmallToe':5, 'L_Hip':6, 'L_Knee':7, 'L_Ankle':8, 'L_BigToe':9, 'L_SmallToe':10,\
    'Pelvis':11, 'Spine_H36M':12, 'Unkown_part1':13, 'Jaw_H36M':14, 'Head':15, 'Unkown_part2':16, 'L_Shoulder':17, 'L_Elbow':18, 'L_Wrist':19, 'Unkown_part3':20, \
    'Unkown_part4':21, 'Unkown_part5':22, 'Unkown_part6':23, 'Neck':24, 'R_Shoulder':25, 'R_Elbow':26, 'R_Wrist':27,\
    'Unkown_part7':28, 'Unkown_part8':29, 'Unkown_part9':30, 'Unkown_part10':31}

H36M_17 = {
    'Pelvis': 0, 'L_Hip': 1, 'L_Knee': 2, 'L_Ankle': 3, 'R_Hip': 4, 'R_Knee': 5, 'R_Ankle': 6, 'Spine_H36M': 7, 'Neck': 8, 
    'Jaw_H36M': 9, 'Head': 10, 'L_Shoulder': 11, 'L_Elbow': 12, 'L_Wrist': 13, 'R_Shoulder': 14, 'R_Elbow': 15, 'R_Wrist':16 }

h36m17_connMat = np.array([[0,1], [1,2], [2,3], [0, 4], [4, 5], [5, 6], [0,7], [7,8], [8,9], [8,10], [8,11], [11,12], [12,13], [8,14], [14,15], [15,16]])

# the joint defination of each datasets:
# MuCo-3DHP
MuCo_21 = {
    'Head_top':0, 'Thorax_unkown':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, 'L_Wrist':7, 'R_Hip':8, 'R_Knee':9,\
    'R_Ankle':10, 'L_Hip':11, 'L_Knee':12, 'L_Ankle':13, 'Pelvis':14, 'Spine_unkown':15, 'Head':16, 'R_Hand':17, 'L_Hand':18, 'R_BigToe':19, 'L_BigToe':20}

# MuPoTS
MuPoTS_17 = {
    'Head_top':0,'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, 'L_Wrist':7, 'R_Hip':8, 'R_Knee':9, 'R_Ankle':10,\
    'L_Hip':11, 'L_Knee':12, 'L_Ankle':13, 'Pelvis':14, 'Thorax_MPII':15, 'Head':16
    }
MuPoTS_17_connMat = np.array([[0,16],[0,1],[1,2],[1,5],[2,3],[3,4],[2,5],[5,6],[6,7],[2,8],[8,9],[9,10],[5,11],[11,12],[12,13],[11,14],[8,14],[15,14],[15,1]])

# COCO
COCO_17 = {
    'Nose':0, 'L_Eye':1, 'R_Eye':2, 'L_Ear':3, 'R_Ear':4, 'L_Shoulder':5, 'R_Shoulder':6, 'L_Elbow':7, 'R_Elbow':8, \
    'L_Wrist': 9, 'R_Wrist':10, 'L_Hip':11, 'R_Hip':12, 'L_Knee':13, 'R_Knee':14, 'L_Ankle':15, 'R_Ankle':16}
OCPose_16 = {
    'R_Ankle':0, 'R_Knee':1, 'R_Hip':2, 'L_Hip':3, 'L_Knee':4, 'L_Ankle':5,  "Pelvis": 6, "Thorax_MPII": 7, "Neck": 8, \
    "Head_top": 9, "R_Wrist": 10, "R_Elbow": 11, "R_Shoulder": 12, "L_Shoulder": 13, "L_Elbow": 14, "L_Wrist": 15}

COCO_18 = {
    'Nose':0, 'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, \
    'L_Wrist':7, 'R_Hip': 8, 'R_Knee':9, 'R_Ankle':10, 'L_Hip':11, 'L_Knee':12, 'L_Ankle':13, \
    'R_Eye':14, 'L_Eye':15, 'R_Ear':16, 'L_Ear':17}

Panoptic_19 = {
    'Neck':0, 'Nose':1, 'Pelvis':2, 'L_Shoulder':3, 'L_Elbow':4, 'L_Wrist':5, 'L_Hip':6, \
    'L_Knee':7, 'L_Ankle':8, 'R_Shoulder': 9, 'R_Elbow':10, 'R_Wrist':11, 'R_Hip':12, 'R_Knee':13, 'R_Ankle':14, \
    'L_Eye':15, 'L_Ear':16, 'R_Eye':17, 'R_Ear':18}

Panoptic_15 = {
    'Neck':0, 'Head_top':1, 'Pelvis':2, 'L_Shoulder':3, 'L_Elbow':4, 'L_Wrist':5, 'L_Hip':6, \
    'L_Knee':7, 'L_Ankle':8, 'R_Shoulder': 9, 'R_Elbow':10, 'R_Wrist':11, 'R_Hip':12, 'R_Knee':13, 'R_Ankle':14}

MINI_25 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23, 'Nose':24}

Crowdpose_14 = {"L_Shoulder":0, "R_Shoulder":1, "L_Elbow":2, "R_Elbow":3, "L_Wrist":4, "R_Wrist":5,\
     "L_Hip":6, "R_Hip":7, "L_Knee":8, "R_Knee":9, "L_Ankle":10, "R_Ankle":11, "Head_top":12, "Neck_LSP":13}

#MPII
MPII_16 = {
    'R_Ankle':0, 'R_Knee':1, 'R_Hip':2, 'L_Hip':3, 'L_Knee':4, 'L_Ankle':5, 'Pelvis':6, 'Thorax_MPII':7,\
    'Neck':8, 'Head_top':9, 'R_Wrist':10, 'R_Elbow':11, 'R_Shoulder':12, 'L_Shoulder':13, 'L_Elbow':14, 'L_Wrist':15,}

SMAP_15 = {
    'Neck_LSP':0, 'Head_top':1, 'Pelvis':2, 'L_Shoulder':3, 'L_Elbow':4, 'L_Wrist':5, 'L_Hip':6, 'L_Knee':7,\
    'L_Ankle':8, 'R_Shoulder':9, 'R_Elbow':10, 'R_Wrist':11, 'R_Hip':12, 'R_Knee':13, 'R_Ankle':14, }

#Posetrack
Posetrack_17 = {
    'Nose':0, 'empty':1, 'empty':2, 'empty':3, 'empty':4, 'L_Shoulder':5, 'R_Shoulder':6, 'L_Elbow':7, 'R_Elbow':8, \
    'L_Wrist': 9, 'R_Wrist':10, 'L_Hip':11, 'R_Hip':12, 'L_Knee':13, 'R_Knee':14, 'L_Ankle':15, 'R_Ankle':16
    }

#OCHuman
OCHuman_19 = {
    'R_Shoulder':0, 'R_Elbow':1, 'R_Wrist':2, 'L_Shoulder':3, 'L_Elbow':4, 'L_Wrist':5, \
    'R_Hip': 6, 'R_Knee':7, 'R_Ankle':8, 'L_Hip':9, 'L_Knee':10, 'L_Ankle':11, 'Head_top':12, 'Neck':13,\
    'R_Ear':14, 'L_Ear':15, 'Nose':16, 'R_Eye':17, 'L_Eye':18
    }

# MPI-INF-3DHP
MPI_INF_28 = {
    'Spine4_unkown': 0, 'Spine3_unkown': 1, 'Spine2_unkown': 2, 'Spine1_unkown': 3, 'Pelvis': 4,\
    'Neck':5, 'Head':6, 'Head_top':7, 'L_Collar_MPI':8, 'L_Shoulder':9, 'L_Elbow':10,\
    'L_Wrist':11, 'L_Hand':12, 'R_Collar_MPI':13, 'R_Shoulder':14, 'R_Elbow':15, 'R_Wrist':16,\
    'R_Hand':17, 'L_Hip':18, 'L_Knee':19, 'L_Ankle':20, 'L_SmallToe':21, 'L_BigToe':22,\
    'R_Hip':23, 'R_Knee':24, 'R_Ankle':25, 'R_SmallToe':26, 'R_BigToe':27
    }

MPI_INF_TEST_17 = {
    'Misalinged_Head_top': 0, 'Neck_LSP': 1, 'R_Shoulder': 2, 'R_Elbow': 3, 'R_Wrist': 4,\
    'L_Shoulder':5, 'L_Elbow':6, 'L_Wrist':7, 'R_Hip':8, 'R_Knee':9, 'R_Ankle':10,\
    'L_Hip':11, 'L_Knee':12, 'L_Ankle':13, 'Pelvis':14, 'Unknown_Thorax_MPII':15, 'Unknown_Head':16
    }

#NTU RGB+D 
NTU_25 = {
    'Pelvis':0, 'Thorax_unkown':1, 'Neck':2, 'Head':3, 'L_Shoulder':4, 'L_Elbow':5, 'L_Wrist':6, \
    'L_Hand':7, 'R_Shoulder':8, 'R_Elbow':9, 'R_Wrist':10, 'R_Hand':11, 'L_Hip':12, 'L_Knee':13, 'L_Ankle':14,\
    'L_SmallToe_unkown':15,'R_Hip':16, 'R_Knee':17, 'R_Ankle':18, 'R_SmallToe_unkown':19, 'Spine_neck_unkown':20, \
    'L_Hand_tip':21,'L_Hand_thumb':22,'R_Hand_tip':23,'R_Hand_thumb':24\
}

JTA_22 = {'Head_top':0, 'Head':1, 'Neck':2, 'right_clavicle':3, 'R_Shoulder':4, 'R_Elbow':5,\
        'R_Wrist':6, 'left_clavicle':7, 'L_Shoulder':8, 'L_Elbow':9, 'L_Wrist':10,\
        'spine0':11, 'spine1':12, 'spine2':13, 'spine3':14, 'spine4':15, \
        'R_Hip':16,'R_Knee':17,'R_Ankle':18,'L_Hip':19,'L_Knee':20,'L_Ankle':21,}

EFT_49_names = [
'OP_Nose', 'OP_Neck', 'OP_RShoulder',           #0,1,2
'OP_RElbow', 'OP_RWrist', 'OP_LShoulder',       #3,4,5
'OP_LElbow', 'OP_LWrist', 'OP_MidHip',          #6, 7,8
'OP_RHip', 'OP_RKnee', 'OP_RAnkle',             #9,10,11
'OP_LHip', 'OP_LKnee', 'OP_LAnkle',             #12,13,14
'OP_REye', 'OP_LEye', 'OP_REar',                #15,16,17
'OP_LEar', 'OP_LBigToe', 'OP_LSmallToe',        #18,19,20
'OP_LHeel', 'OP_RBigToe', 'OP_RSmallToe', 'OP_RHeel',  #21, 22, 23, 24  ##Total 25 joints  for openpose

'R_Ankle', 'R_Knee', 'R_Hip',               #0,1,2
'L_Hip', 'L_Knee', 'L_Ankle',                  #3, 4, 5
'R_Wrist', 'R_Elbow', 'R_Shoulder',     #6
'L_Shoulder', 'L_Elbow', 'L_Wrist',            #9
'Neck', 'Head_top',                      #12, 13
'Pelvis', 'Thorax_MPII',                       #14, 15
'Spine_H36M', 'Jaw_H36M',                           #16, 17
'Head', 'Nose', 'L_Eye',                      #18, 19, 20
'R_Eye', 'L_Ear', 'R_Ear'                    #21,22,23 (Total 24 joints)
]
EFT_49 = {name:ind for ind, name in enumerate(EFT_49_names)}

lsp14_kpcm = np.array([[255,0,85],[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,85],[0,255,170],[255,0,170],\
    [255,0,255],[0,255,255],[0,85,255],[0,170,255]])
smpl24_kpcm = np.array([[255,0,85],[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,85],[0,255,170],[255,0,170],\
    [255,0,255],[0,255,255],[0,85,255],[0,170,255],[255,0,85],[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,85],[0,255,170],[255,0,170]])
VIBE_COMMON_EVAL_14 = LSP_14

muco21_2_openpose25 = joint_mapping(MuCo_21, OpenPose_25)
muco21_2_smpl24 = joint_mapping(MuCo_21, SMPL_24)
mupots17_2_openpose25 = joint_mapping(MuPoTS_17, OpenPose_25)
mupots17_2_smpl24 = joint_mapping(MuPoTS_17, SMPL_24)
lsp_2_coco25 = joint_mapping(LSP_14, OpenPose_25)
smpl24_2_coco25 = joint_mapping(SMPL_24, OpenPose_25)
coco18_2_openpose25 = joint_mapping(COCO_18,OpenPose_25)
posetrack_2_coco25 = joint_mapping(Posetrack_17,OpenPose_25)
ochuman19_2_coco25 = joint_mapping(OCHuman_19,OpenPose_25)
mpiinf28_2_coco25 = joint_mapping(MPI_INF_28,OpenPose_25)
mpiinf28_2_smpl24 = joint_mapping(MPI_INF_28,SMPL_24)
h36m32_2_coco25 = joint_mapping(H36M_32,OpenPose_25)
h36m32_2_smpl24 = joint_mapping(H36M_32,SMPL_24)
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

move52_2_coco25 = np.array([-1, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,1, 4, 7, -1,-1,-1,-1, 10,-1,-1, 11,-1,-1])
body1352coco25 = np.array([0,17,6,8,10,5,7,9,-1,12,14,16,11,13,15,2,1,4,3, 19,20,21, 22,23,24])
ps_2_openpose25 = np.array([1,0, 9,10,11, 3,4,5, 2, 12,13,14, 6,7,8, 17,15, 18,16, -1,-1,-1, -1,-1,-1])
openpose19_2_ps = np.array([1,0,8, 5,6,7, 12,13,14, 2,3,4, 9,10,11, 15,16,17,18])

valid_kp_mask_smpl24 = {'h36m':h36m32_2_smpl24!=-1 , 'mpiinf':mpiinf28_2_smpl24!=-1 , 'pw3d':np.ones(24).astype(np.bool_), 'mupots': mupots17_2_smpl24!=-1, 'muco':muco21_2_smpl24!=-1}

global_orient_nocam = np.array([0,0,np.pi])

hand_rot_joint_idx = np.array([1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19])

joints_group_face = np.array([[17,19,21],[26,24,22], [36,38,39],[45,43,42], [27,29,30], [31,33,35], [48,51,54],[59,57,55], [60,62,64], [67,66,65]])-17
joints_group_hand = [[0,1,2,3,4], [0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20], [1,5,9,13,17], [2,6,10,14,18], [3,7,11,15,19], [4,8,12,16,20]]
bhf_kps_num = [25,21,21,51]
bhf_connect_kps_num = np.concatenate([25+np.array([6-25,0,2,5,9,13,17]), 25+21+np.array([3-25-21,0,2,5,9,13,17]), np.array([2,5,1,0,15,16,17,18])])
connector_connMat = np.array([0,1, 1,2,1,3,1,4,1,5,1,6, 7,8, 8,9,8,10,8,11,8,12,8,13, 14,16, 15,16, 16,17, 17,18,17,19, 18,20,19,21 ]).reshape(-1, 2)

face51_connMat = np.array([17,18, 18,19, 19,20, 20,21, 22,23,23,24,24,25,25,26, 27,28,28,29,29,30, 31,32,32,33,33,34,34,35, 36,37,37,38,38,39,39,40,40,41,41,36, \
    42,43,43,44,44,45,45,46,46,47,47,42, 48,49,49,50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,48, 60,61,61,62,62,63,63,64,64,65,65,66,66,67,67,60]).reshape(-1, 2)-17
face70_connMat = np.array([0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,\
    17,18, 18,19, 19,20, 20,21, 22,23,23,24,24,25,25,26, 27,28,28,29,29,30, 31,32,32,33,33,34,34,35, 36,37,37,38,38,39,39,40,40,41,41,36, \
    42,43,43,44,44,45,45,46,46,47,47,42, 48,49,49,50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,48, 60,61,61,62,62,63,63,64,64,65,65,66,66,67,67,60]).reshape(-1, 2)

body_connMat = np.array([0, 1, 0, 3, 3, 4, 4, 5, 0, 9, 9, 10, 10, 11, 0, 2, 2, 6, 6, 7, 7, 8, 2, 12, 12, 13, 13, 14, 1, 15, 15, 16, 1, 17, 17, 18]).reshape(-1, 2)

body17_connMat = np.array([[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],\
            [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]])-1
body18_connMat = np.array([0,1, 1,2, 2,3, 3,4, 1,5, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 0,14,14,16, 0,15,15,17]).reshape(-1, 2)
body25_connMat = np.array([0,1, 1,2, 2,3, 3,4, 1,5, 5,6, 6,7, 1,8, 8,9, 9,10, 10,11, 8,12, 12,13, 13,14, 0,15, 15,17, 0,16, 16,18, 14,19, 19,20, 14,21, 11,22, 22,23, 11,24]).reshape(-1, 2)

# exclude 2 hand joints
smpl22_connMat = np.array([0,1, 0,2, 0,3, 1,4,4,7,7,10, 2,5,5,8,8,11, 3,6,6,9,9,12,12,15, 12,13,13,16,16,18,18,20, 12,14, 14,17, 17,19, 19,21]).reshape(-1, 2)
lsp14_connMat = np.array([[ 0, 1 ],[ 1, 2 ],[ 3, 4 ],[ 4, 5 ],[ 6, 7 ],[ 7, 8 ],[ 8, 2 ],[ 8, 9 ],[ 9, 3 ],[ 2, 3 ],[ 8, 12],[ 9, 10],[12, 9 ],[10, 11],[12, 13]])
crowdpose_connMat = joint_mapping(Crowdpose_14,LSP_14)[lsp14_connMat] 
mpii15_connMat = np.array([0,1, 0,2, 0,3, 3,4, 4,5, 2,6, 6,7, 7,8, 0,9, 9,10, 10,11, 2,12, 12,13, 13,14]).reshape(-1, 2)
cmup19_connMat = np.array([0,1, 0,2, 0,3, 3,4, 4,5, 2,6, 6,7, 7,8, 0,9, 9,10, 10,11, 2,12, 12,13, 13,14, 1,15, 15,16, 1,17, 17,18]).reshape(-1, 2)
hand_connMat = np.array([0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 0, 9, 9, 10, 10, 11, 11, 12, 0, 13, 13, 14, 14, 15, 15, 16, 0, 17, 17, 18, 18, 19, 19, 20]).reshape(-1, 2)
connMat = np.concatenate([body25_connMat,hand_connMat+25, hand_connMat+25+21],0)
face51_connMat = np.array([17,18, 18,19, 19,20, 20,21,  22,23, 23,24, 24,25,  27,28, 28,29, 29,30,  31,32, 32,33, 33,34, 34,35,  36,37, 37,38, 38,39, 39,40, 40,41, 41,36, 42,43, 43,44, 44,45, 45,46, 46,47, 47,42,\
 48,49, 49,50, 50,51, 51,52, 52,53, 53,54, 54,55, 55,56, 56,57, 57,58, 58,59, 59,48, 60,61, 61,62, 62,63, 63,64, 64,65, 65,66, 66,67, 67,60]).reshape(-1, 2) - 17

cm_body14 = np.array([[255,0,85],[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,85],[0,255,170],[255,0,170],\
    [255,0,255],[0,255,255],[0,85,255],[0,170,255],[170,0,255]])
cm_body17 = np.array([[255,0,85],   [255,0,0],[255,85,0],[255,170,0],   [255,255,0],[170,255,0],[85,255,0],  [255,0,0],[0,255,85],[0,255,170],  \
 [0,255,255],[0,170,255],[0,85,255],  [0,0,255],[255,0,170],  [170,0,255],[255,0,255], [0,255,255],[0,255,255]])[:,::-1]
cm_body18 = np.array([[255,0,85],   [255,0,0],[255,85,0],[255,170,0],   [255,255,0],[170,255,0],[85,255,0],  [255,0,0],[0,255,85],[0,255,170],  \
 [0,255,255],[0,170,255],[0,85,255],  [0,0,255],[255,0,170],  [170,0,255],[255,0,255]])[:,::-1]
cm_body25 = np.array([[255,0,85],   [255,0,0],[255,85,0],[255,170,0],   [255,255,0],[170,255,0],[85,255,0],  [0,255,0],  [255,0,0],[0,255,85],[0,255,170],  \
 [0,255,255],[0,170,255],[0,85,255],  [0,0,255],[255,0,170],  [170,0,255],[255,0,255],  [85,0,255], [0,0,255],[0,0,255],[0,0,255],   [0,255,255],[0,255,255],[0,255,255]])[:,::-1]
cm_hand21 = np.array([[205,0,0],[205,0,0],[205,0,0],[205,0,0],  [0,43,226],[0,43,226],[0,43,226],[0,43,226], [65,105,225],[65,105,225],[65,105,225],[65,105,225], \
 [139,0,139],[139,0,139],[139,0,139],[139,0,139], [220,20,60],[220,20,60],[220,20,60],[220,20,60] ])
# right eyebow
cm_face51 = np.concatenate([np.array([[255,0,85] for i in range(4)]), np.array([[85,255,0] for i in range(4)]), \
    np.array([[0,85,255] for i in range(3)]), np.array([[0,85,255] for i in range(4)]), \
    np.array([[255,0,85] for i in range(6)]), np.array([[85,255,0] for i in range(6)]),\
    np.array([[139,0,139] for i in range(12)]), np.array([[220,20,60] for i in range(8)]),],0)
cm_All54 = np.concatenate([cm_body25,cm_body25,cm_body25,cm_body25,cm_body25],0)

# joint order after flip, body, lhand, rhand, face
body118_flip = np.concatenate([np.array([0,1, 5,6,7, 2,3,4, 8, 12,13,14, 9,10,11, 16,15,18,17, 22,23,24, 19,20,21]), \
    25+21+np.arange(21), 25+np.arange(21), \
    25+21*2+np.array([26,25,24,23,22, 21,20,19,18,17, 27,28,29,30, 35,34,33,32,31, 45,44,43,42,47,46, 39,38,37,36,41,40,\
    54,53,52,51,50,49,48, 59,58,57,56,55, 64,63,62,61,60,67,66,65, 69,68])-17],0).astype(np.int32)


kintree_parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16,17, 18, 19, 20, 21],dtype=np.int32)
# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]#, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3*i)
    SMPL_POSE_FLIP_PERM.append(3*i+1)
    SMPL_POSE_FLIP_PERM.append(3*i+2)

openpose25_2_smplx = [np.array([1,2,5,8,9,12]), 12, 9, np.array([1,2,5,8,9,12]), 13,10, np.array([1,2,5,8,9,12]), 14,11, np.array([1,2,5,8,9,12]), 19,22, 1, np.array([1,2,5,8,9,12]), np.array([1,2,5,8,9,12]), np.array([0,1,15,16,17,18]), 5,2,  6,3, 7,4]
openpose25_2_smplx_mask = np.array([isinstance(i,np.ndarray) for i in openpose25_2_smplx])
smplx_joint_idx = np.where(openpose25_2_smplx_mask)[0]
summon_set = []
for i in openpose25_2_smplx:
    if isinstance(i,np.ndarray):
        summon_set.append(i)
summon_set = np.array(summon_set)

openpose25_2_smplx_map= []
for i in openpose25_2_smplx:
    if not isinstance(i,np.ndarray):
        openpose25_2_smplx_map.append(i)
    else:
        openpose25_2_smplx_map.append(0)

h36m_action_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',\
            'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo',\
            'Waiting', 'Walking', 'WalkDog', 'WalkTogether' ]

cmup_action_names = ['haggling1', 'mafia2', 'ultimatum1', 'pizza1']

SMPL_MAJOR_JOINTS = np.array([1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21])

dataset_involved = ['h36m', 'mupots', 'pw3d', 'mpiinf', 'jta','cmup', 'oh','mpiinf_test', 'pw3d_pc', 'pw3d_nc','pw3d_oc','pw3d_vibe', 'pw3d_normal','agora','mini','pw3d_od']
dataset_kp2ds = ['h36m', 'mupots', 'pw3d', 'mpiinf', 'jta','cmup', 'oh','mpiinf_test', 'pw3d_pc', 'pw3d_nc','pw3d_oc','pw3d_vibe', 'pw3d_normal','agora','mini','relativity', 'aich', 'crowdpose','syrip','pw3d_od']
dataset_smpl2lsp = ['h36m', 'cmup','mpiinf','mpiinf_test', 'jta', 'pw3d_nc','pw3d_oc','pw3d_vibe'] #'pw3d', 
MPJAE_ds = ['pw3d_normal', 'h36m']
PVE_ds = ['pw3d_pc', 'pw3d_nc','pw3d_oc','pw3d_vibe', 'pw3d_normal', 'oh','agora','mini']
dataset_depth = ['mupots','agora','mini','pw3d_od']
dataset_relative_depth = ['relativity']
dataset_relative_age = ['relativity', 'agora', 'mupots','mini']
dataset_nokp3ds = ['relativity', 'aich', 'crowdpose']

pretrained_models = {
    'V1+HRNet-50+md':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/3dpw_88_57.8.pkl',
    'V1+HRNet-50+md+CAR':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/pw3d-hc_97.64_76.93_pw3d-vibe_84.85_59.97_86.26_100.39_20.77_oh_38.38_h36m_52.93.pkl',
    'V1+HRNet-50+md+CAR+3DPW':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/pw3d_finetuned_testset_76.02_46.67_92.36_18.76.pkl',
    # 注意：　model/resnet_50.py的预处理的均值方差不一样
    'V1+ResNet-50+ld':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V1_noCAR0.2_pc_103.55_79.69_136.22_oc_98.05_67.03_109.97_nc_95.50_56.75_106.81.pkl',
    'V1+ResNet-50+CAR0.1+ld':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/resnet_cm64_V1_CAR0.1_vibe_91.33_54.97_pc_77.6_oc_66.6_nc_55.6_h36m,mpiinf,coco,mpii,lsp,up,aich.pkl',
    'V1+ResNet-50+CAR0.2+ld':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V1_CAR0.2_pc_103.75_75.76_oc_99.37_67.07_110.70_nc_97.99_57.15_108.26.pkl',
    'V1+ResNet-50+CAR0.3+ld':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V1_CAR0.3_pc_102.97_77.02_134.52_oc_97.78_66.48_108.99_nc_95.05_56.40_105.25.pkl',
    'V1+ResNet-50+md+CAR':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/resnet_cm64_V1_resnet_vibe_89.3_53.5_105.6_h36m,mpiinf,coco,mpii,lsp,oh,posetrack,crowdpose,aich,ochuman,up,jta,muco.pkl',
    'V1+ResNet-50+md+CAR+3DPW':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/resnet_cm64_V1_resnet_vibe_trainmode_test_79.7_49.4_94.7_pw3d0.8,crowdpose0.2.pkl',
    'V1+ResNet-50+md+CAR_2':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/resnet_cm64_V1_resnet_vibe_88.79_52.53_h36m,mpiinf,coco,mpii,lsp,muco,crowdpose.pkl',

    'V1+HRNet-50+md+CAR+AGORA': '/home/sunyu15/Infinity/project_data/romp_data/trained_models/V1_hrnet_agora_ft76_val_108.02_71.79.pkl',
    
    'V6+noRH+noAGORA+CAR': '/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_norelative_noagora_coco,mpii,lsp,crowdpose,muco,h36m.pkl',
    'V6+noRH+AGORA+md+CAR': '/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_norelative_coco,mpii,lsp,crowdpose,muco,agora,h36m.pkl',
    'V6+noRH2+AGORA+md+CAR': '/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_norelative2_coco,mpii,lsp,crowdpose,muco,agora,h36m.pkl',
    'V6+noRH3+AGORA+md+CAR': '/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_norelative3_coco,mpii,lsp,crowdpose,muco,agora,h36m.pkl',
    'V6+RH+AGORA+md+CAR':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_relative_coco,mpii,lsp,crowdpose,muco,relative_human,h36m,syrip.pkl',
    'V6+RH+AGORA+md+CAR-2':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_relative_ft2_coco,mpii,lsp,crowdpose,muco,relative_human,h36m,syrip.pkl',
    'V6+RH+AGORA_md_RH_val0.737_test0.708':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_relative_ct0.12_RH_val0.737_test0.708_msaug_coco,aich,lsp,crowdpose,muco,agora,h36m,relative_human,aist,posetrack,movi.pkl',

    # ablation study: 
    'V6+noRH+noAGORA_relative_age_only': '/home/sunyu15/Infinity/project_data/romp_data/trained_models/hrnet_cm128_V6_hrnet_relative_ba_smil_newPiecewise_cct0.2_nosyrip_noAGORA_relative_age_only.pkl',
    'V6+noRH+noAGORA_relative_depth_only': '/home/sunyu15/Infinity/project_data/romp_data/trained_models/hrnet_cm128_V6_hrnet_relative_ba_smil_newPiecewise_cct0.2_nosyrip_noAGORA_relative_depth_only.pkl',
    'V6+RH+AGORA_val_eval_trans3D':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_trans3D_agoraval_coco,mpii,lsp,crowdpose,muco,agora,h36m.pkl',
    'V6+RH+md+log':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_log_coco,mpii,lsp,crowdpose,muco,agora,h36m.pkl',
    'V6+RH+AGORA_no_depth_encoding':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_no_depth_encoding_coco,mpii,lsp,crowdpose,muco,agora,h36m.pkl',
    'V6+RH+AGORA_no_offsetmap':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/hrnet_cm128_V6_hrnet_relative_wo_offsetmap.pkl',
    'V6+RH+AGORA_no_front-view-condition_when_estimating_bv':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/hrnet_cm128_V6_hrnet_relative_wo_front_view_condition.pkl',
    'V6+RH+AGORA+md+fit3dpw':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/V6_hrnet_cm128_relative_ct0.12_fit3dpw_77.73_46.56_91.34_coco,aich,lsp,crowdpose,muco,pw3d,h36m,relative_human.pkl',
    
    'V7+noRH':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/hrnet_cm128_V7_ft80_h36m,mpiinf,coco,mpii,lsp,crowdpose,muco,relative_human,agora,pw3d.pkl',
    'V7+RH+agora':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/hrnet_cm128_V7_ft6_coco,mpii,lsp,crowdpose,muco,relative_human,h36m,agora,syrip.pkl',
    'V7+RH+agora-2':'/home/sunyu15/Infinity/project_data/romp_data/trained_models/hrnet_cm128_V7_ft11_coco,mpii,lsp,crowdpose,muco,relative_human,h36m,agora,syrip.pkl',
}
#'/home/sunyu15/checkpoints/hrnet_cm128_V6_hrnet_norelative_fitagora_on_gpu0,1,2,3_val/6_249.8342_27.6988_hrnet_cm128_V6_hrnet_norelative_fitagora.pkl'

age_threshold = {'adult': [-0.05,0,0.15], 'teen':[0.15, 0.3, 0.45], 'kid':[0.45, 0.6, 0.75], 'baby':[0.75,0.9,1]}
#{'adult': [-0.04,0,0.08], 'teen':[0.08, 0.18, 0.36], 'kid':[0.36, 0.48, 0.6], 'baby':[0.6,0.78,1]}
relative_depth_types = ['eq', 'cd', 'fd']
relative_age_types = ['adult', 'teen', 'kid', 'baby']

def get_cam3dmap_anchor(FOV, centermap_size):
    depth_level = np.array([1, 10, 20, 100], dtype=np.float32)
    #map_coord_range_each_level = (np.array([4/64., 50/64., 6/64., 4/64.], dtype=np.float32) * args().centermap_size).astype(np.int32)
    map_coord_range_each_level = (np.array([2/64., 25/64., 3/64., 2/64.], dtype=np.float32) * centermap_size).astype(np.int32)
    scale_level = 1/np.tan(np.radians(FOV/2.))/depth_level
    cam3dmap_anchor = []
    scale_cache = 8
    for scale, coord_range in zip(scale_level, map_coord_range_each_level):
        cam3dmap_anchor.append(scale_cache-np.arange(1,coord_range+1)/coord_range*(scale_cache-scale))
        scale_cache = scale
    cam3dmap_anchor = np.concatenate(cam3dmap_anchor)
    return cam3dmap_anchor

# from config import args
# get_cam3dmap_anchor(args().FOV, args().centermap_size)

img_exts = ['.bmp', '.dib', '.jpg', '.jpeg', '.jpe', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.tiff', '.tif', '.sr', '.ras', '.exr', '.hdr', '.pic']
video_exts = ['.mp4', '.avi', '.webm', '.gif']
mesh_color_dict = {'LightCyan': [225,255,255], 'ghostwhite':[248, 248, 255], \
'Azure':[240,255,255],'Cornislk':[255,248,220],'Honeydew':[240,255,240],'LavenderBlush':[255,240,245]}

wardrobe = {# Shirt
            '000':'SMPL_shirt_m_hr.jpg', '001':'SMPL_shirt2_m_hr.jpg', '002':'SMPL_shirt3_m_hr.jpg', '003':'SMPL_shirt4_m_hr.jpg', '004':'SMPL_shirt5_m_hr.jpg', '005':'SMPL_shirt6_m_hr.jpg',\
            '006':'SMPL_shirt7_m_hr.jpg', '007':'SMPL_shirt8_m_hr.jpg', '008':'SMPL_shirt9_m_hr.jpg', '009':'SMPL_shirt10_m_hr.jpg', '0010':'SMPL_shirt11_m_hr.jpg',
            '100':'SMPL_shirt_f_hr.jpg', '101':'SMPL_shirt2_f_hr.jpg', '102':'SMPL_shirt3_f_hr.jpg',
            # T-shirt 
            '010':'SMPL_tshirt_m_hr.jpg','011':'SMPL_tshirt2_m_hr.jpg','012':'SMPL_tshirt_m_lr.jpg',
            '110':'SMPL_tshirt_f_lr.jpg', 
            # long
            '020':'SMPL_long_m_hr.jpg',
            '120':'SMPL_long_f_lr.jpg', 
            # Suit
            '030':'SMPL_suit_m_hr.jpg','031':'SMPL_suit2_m_hr.jpg','032':'SMPL_suit3_m_hr.jpg',
            }
