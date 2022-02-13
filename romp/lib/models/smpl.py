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
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os,sys
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.abspath(__file__).replace('models/smpl.py',''))
from config import args

ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints','joints_h36m17', 'joints_smpl24'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)

class VertexJointSelector(nn.Module):

    def __init__(self, vertex_ids=None,
                 use_hands=True,
                 use_feet_keypoints=True, **kwargs):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        face_keyp_idxs = np.array([
            vertex_ids['nose'],
            vertex_ids['reye'],
            vertex_ids['leye'],
            vertex_ids['rear'],
            vertex_ids['lear']], dtype=np.int64)

        extra_joints_idxs = np.concatenate([extra_joints_idxs,
                                            face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int32)

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, tips_idxs])

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints


class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10

    def __init__(self, model_path, J_reg_extra9_path=None, J_reg_h36m17_path=None,\
                 data_struct=None, betas=None, global_orient=None,\
                 body_pose=None, transl=None, dtype=torch.float32, batch_size=1,\
                 joint_mapper=None, gender='neutral', vertex_ids=None, **kwargs):
        ''' SMPL model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.gender = gender
        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))

        super(SMPL, self).__init__()
        self.batch_size = batch_size

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        # The shape components
        shapedirs = data_struct.shapedirs

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(data_struct.v_template),
                                       dtype=dtype))
        if betas is None:
            default_betas = torch.zeros([batch_size, self.NUM_BETAS],dtype=dtype)
        else:
            if 'torch.Tensor' in str(type(betas)):
                default_betas = betas.clone().detach()
            else:
                default_betas = torch.tensor(betas,dtype=dtype)

        self.register_parameter('betas', nn.Parameter(default_betas,
                                                      requires_grad=True))

        
        # The shape components
        shapedirs = shapedirs[:, :, :self.NUM_BETAS]
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        if J_reg_extra9_path is not None:
            J_regressor_extra9 = np.load(J_reg_extra9_path)
            J_regressor_extra9 = to_tensor(to_np(J_regressor_extra9), dtype=dtype)
            self.register_buffer('J_regressor_extra9', J_regressor_extra9)
        else:
            self.register_buffer('J_regressor_extra9', None)

        if J_reg_h36m17_path is not None:
            H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
            J_regressor_h36m17 = np.load(J_reg_h36m17_path)[H36M_TO_J17]
            J_regressor_h36m17 = to_tensor(to_np(J_regressor_h36m17), dtype=dtype)
            self.register_buffer('J_regressor_h36m17', J_regressor_h36m17)
        else:
            self.register_buffer('J_regressor_h36m17', None)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(data_struct.weights), dtype=dtype))

    def create_mean_pose(self, data_struct):
        pass

    @torch.no_grad()
    def reset_params(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def extra_repr(self):
        return 'Number of betas: {}'.format(self.NUM_BETAS)

    def forward(self, batch_size=128, **kwargs):
        person_num = len(kwargs['poses'])
        if person_num>batch_size and batch_size>0:
            result_dict_list = []
            for inds in range(int(np.ceil(person_num/float(batch_size)))):
                batch_data = {k:v[inds*batch_size:(inds+1)*batch_size] if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
                result_dict_list.append(self.single_forward(**batch_data))
            result_dict = {}
            for k in result_dict_list[0].keys():
                result_dict[k] = torch.cat([rdict[k] for rdict in result_dict_list], 0).contiguous()
            return result_dict
        return self.single_forward(**kwargs)

    def single_forward(self, betas=None, poses=None,
                transl=None, return_verts=True, return_full_pose=False,
                **kwargs):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        betas = betas if betas is not None else self.betas
        if isinstance(betas,np.ndarray):
            betas = torch.from_numpy(betas).float()
        if isinstance(poses,np.ndarray):
            poses = torch.from_numpy(poses).float()
        if isinstance(transl,np.ndarray):
            transl = torch.from_numpy(transl).float()
        default_device = self.shapedirs.device
        betas, poses = betas.to(default_device), poses.to(default_device)

        vertices, joints = lbs(betas, poses, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)
        joints_smpl24 = joints.clone()
        joints = self.vertex_joint_selector(vertices, joints)

        outputs = {'verts': vertices, 'j3d':joints, 'joints_smpl24':joints_smpl24}
        
        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:,14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis
            outputs.update({'joints_h36m17':joints_h36m17})


        if self.J_regressor_extra9 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints = torch.cat([joints, vertices2joints(self.J_regressor_extra9, vertices)],1)
            outputs.update({'j3d':joints})
            
            if args().smpl_mesh_root_align:
                # use the Pelvis of most 2D image, not the original Pelvis
                root_trans = joints[:,49].unsqueeze(1)
                joints = joints - root_trans
                vertices =  vertices - root_trans
                joints_smpl24 = joints_smpl24 - root_trans
                outputs.update({'verts': vertices, 'j3d':joints, 'joints_smpl24':joints_smpl24})
        
        if transl is not None:
            outputs = {key:value+transl.unsqueeze(1) for key,value in outputs.items()}

        return outputs


VERTEX_IDS = {
    'smplh': {
        'nose':         332,
        'reye':         6260,
        'leye':         2800,
        'rear':         4071,
        'lear':         583,
        'rthumb':       6191,
        'rindex':       5782,
        'rmiddle':      5905,
        'rring':        6016,
        'rpinky':       6133,
        'lthumb':       2746,
        'lindex':       2319,
        'lmiddle':      2445,
        'lring':        2556,
        'lpinky':       2673,
        'LBigToe':      3216,
        'LSmallToe':    3226,
        'LHeel':        3387,
        'RBigToe':      6617,
        'RSmallToe':    6624,
        'RHeel':        6787
    },
    'smplx': {
        'nose':         9120,
        'reye':         9929,
        'leye':         9448,
        'rear':         616,
        'lear':         6,
        'rthumb':       8079,
        'rindex':       7669,
        'rmiddle':      7794,
        'rring':        7905,
        'rpinky':       8022,
        'lthumb':       5361,
        'lindex':       4933,
        'lmiddle':      5058,
        'lring':        5169,
        'lpinky':       5286,
        'LBigToe':      5770,
        'LSmallToe':    5780,
        'LHeel':        8846,
        'RBigToe':      8463,
        'RSmallToe':    8474,
        'RHeel':        8635
    }
}

def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).contiguous().view(
        batch_size, -1, 3)

    lmk_faces = lmk_faces + torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3).contiguous()[lmk_faces].contiguous().view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)
    dtype = pose.dtype
    posedirs = posedirs.type(dtype)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=J_regressor.device)
    hand_pose = ident.view(1,1,3,3).repeat(batch_size,2,1,1)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose[:,:-6].contiguous().view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3]).type(dtype)
        rot_mats = torch.cat([rot_mats, hand_pose], 1).contiguous()
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).type(dtype)
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs.type(dtype)) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3).type(dtype) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3).type(dtype)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=J_regressor.device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]

    return verts.float(), J_transformed.float()

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    #print(rot_mats.shape, rel_joints.shape,)
    transforms_mat = transform_mat(
        rot_mats.contiguous().view(-1, 3, 3),
        rel_joints.contiguous().view(-1, 3, 1)).contiguous().view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

if __name__ == '__main__':
    import sys, os
    root_dir = os.path.join(os.path.dirname(__file__),'..')
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    import config
    from config import args
    device = torch.device('cuda')  # torch.device('cuda')

    SMPLPY = SMPL(args().smpl_model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path).to(device)
    poses = torch.zeros(1,72).cuda()
    poses[0,18*3+1] = -2.
    results = SMPLPY(betas=torch.zeros(1,10).cuda(), poses=poses)  # Do the thing.

    visualization = True
    if visualization:
        from vedo import *
        verts = results['verts'][0].cpu().numpy()
        faces = np.array(SMPLPY.faces)
        mesh = Mesh([verts, faces]).c('w').alpha(0.8)
        # joints_smpl24 = Points(results['joints_smpl24'][0].cpu().numpy().tolist()).c('red')
        # joints_h36m17 = Points(results['joints_h36m17'][0].cpu().numpy().tolist()).c('blue')
        j3d = Points(results['j3d'][0].cpu().numpy().tolist()).c('green')
        # show(mesh, joints_smpl24, viewup="z", axes=1)
        show(mesh, j3d, viewup="z", axes=1)
        # show(mesh, joints_h36m17, viewup="z", axes=1)
