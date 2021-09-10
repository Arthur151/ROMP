import os

import numpy as np
import pickle as pkl

from .utils import with_zeros, pack, subtract_flat_id
from .utils import axan_to_rot_matrix, rot_matrix_to_axan

class SMPL(object):
    def __init__(self, center_idx=None,  gender='neutral', model_root = '/models'):
        """
        Args:
            center_idx: index of center joint in our computations,
            model_root: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male'
        """
        self.center_idx = center_idx
        self.gender = gender

        if gender == 'neutral':
            self.model_path = os.path.join(model_root, 'SMPL_NEUTRAL.pkl')
        elif gender == 'f':
            self.model_path = os.path.join(model_root, 'SMPL_FEMALE.pkl')
        elif gender == 'm':
            self.model_path = os.path.join(model_root, 'SMPL_MALE.pkl')

        smpl_data = pkl.load(open(self.model_path, 'rb'), encoding = 'latin1')

        self.smpl_data = smpl_data

        self.shapedirs = np.array(smpl_data['shapedirs'])

        self.posedirs = np.array(smpl_data['posedirs'])

        self.v_template = np.array(smpl_data['v_template'])
        self.v_template = np.expand_dims(self.v_template, axis = 0)

        self.J_regressor = np.array(smpl_data['J_regressor'].toarray())

        self.weights = np.array(smpl_data['weights'])

        self.faces =np.array(smpl_data['f'].astype(np.int32))

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']

        parents = list(self.kintree_table[0].tolist())

        self.kintree_parents = parents

        self.num_joints = len(parents)  # 24

    def update(self, pose_axisang, betas , trans=None):
        """
        Args:
        pose_axisang (Nd array (batch_size x 72)): pose parameters in axis-angle representation
        betas (Nd array (batch_size x 10)): if provided, uses given shape parameters
        trans (Nd array (batch_size x 3)): if provided, applies trans to joints and vertices
        """
        # import ipdb
        # ipdb.set_trace()
        batch_size = pose_axisang.shape[0]

        # Convert axis-angle representation to rotation matrix rep. BATCH X 216
        pose_rotmat = axan_to_rot_matrix(pose_axisang)

        # Take out the first rotmat (global rotation)
        root_rot = np.resize(pose_rotmat[:, :9], (batch_size, 3, 3))

        # Take out the remaining rotmats (23 joints)
        pose_rotmat = pose_rotmat[:, 9:]

        # Subtract the identity pose matrix from the current pose matrix
        pose_map = subtract_flat_id(pose_rotmat)

        # v_shaped = v_template + shapedirs * betas
        v_shaped = self.v_template + np.transpose(np.matmul(self.shapedirs, np.transpose(betas, (1, 0))), (2, 0, 1))

        j = np.matmul(self.J_regressor, v_shaped)


        # v_posed = v_shaped + posedirs * pose_map
        # SHAPE: BATCH X 6890 X 3
        v_posed = v_shaped + np.transpose(np.matmul(self.posedirs, np.transpose(pose_map,(1, 0))), (2, 0, 1))

        # Global rigid transformation for each joint is stored in results
        results = []

        # Take the root joint and resize it to BATCH X 3 X 1
        root_j = np.resize(j[:, 0, :], (batch_size, 3, 1))

        # Create a matrix of size BATCH X 4 X 4 for the root joint
        results.append(with_zeros(np.concatenate([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(self.num_joints - 1):
            i_val = int(i + 1)

            # Do the same operation for all the joints in the kinematic chain
            # for each root create BATCH x 4 x 4 matrix which is the global transformation matrix

            joint_rot = np.resize(pose_rotmat[:, (i_val - 1) * 9:i_val * 9], (batch_size, 3, 3))
            joint_j = np.resize(j[:, i_val, :], (batch_size, 3, 1))

            # Find the parent for each joint
            parent = self.kintree_parents[i_val]
            parent_j = np.resize(j[:, parent, :], (batch_size, 3, 1))

            joint_rel_transform = with_zeros(np.concatenate([joint_rot, joint_j - parent_j], 2))

            glob_transf_mat = np.matmul(results[parent], joint_rel_transform)
            results.append(glob_transf_mat)

        # Global transformation matrix for each joint
        # list of 24 matrices - each matrix has dimension B x 4 x 4
        results_global = results

        # Joint positions in global coordinates
        jtr = np.stack(results_global, axis=1)[:, :, :3, 3]

        # Global rotation for each joint - SHAPE: B x 24 x 3 x 3
        results_glb_rot = np.stack(results_global, axis = 1)[:, :, :3, :3]

        # B x 4 x 4 x 24 - This will contain the transformation matrices after the inverse T-pose transformation has
        # been applied
        results2 = np.zeros((batch_size, 4, 4, self.num_joints),  dtype=root_j.dtype )

        # The inverse transformation for rest pose
        # T_k^(-1) = [I - \sum(j_k)]
        #            [0       1    ]
        # the product of G and T_k^(-1) comes out to be G - [0 G(j)]
        #                                                   [0  (0)]

        for i in range(self.num_joints):
            padd_zero = np.zeros(1, dtype=j.dtype)

            joint_j = np.concatenate([j[:, i],  np.tile(np.resize(padd_zero, (1, 1)), (batch_size, 1))], 1)

            tmp = np.matmul(results[i], np.expand_dims(joint_j, axis = 2))

            results2[:, :, :, i] = results[i] - pack(tmp)

        # The transformation matrices multiplied by the weights
        T = np.matmul(results2, np.transpose(self.weights, (1, 0)))

        # The template + blend shapes multiplied by the above product

        rest_shape_h = np.concatenate([np.transpose(v_posed, (0, 2, 1)), np.ones((batch_size, 1, v_posed.shape[1]),
                       dtype=T.dtype)], axis=1)

        verts = np.sum((T * np.expand_dims(rest_shape_h, axis = 1)), axis =2)
        verts = np.transpose(verts, (0, 2, 1))
        verts = verts[:, :, :3]

        if trans is not None:
            jtr = jtr + np.expand_dims(trans, axis = 1)
            verts = verts + np.expand_dims(trans, axis = 1)

        # Vertices and joints in meters
        # Vertices: B x 6890 x 3
        # Jtr: B x 24 x 3
        # global_rot_matrices: B x 24 x 3 x 3
        return verts, jtr, results_glb_rot
