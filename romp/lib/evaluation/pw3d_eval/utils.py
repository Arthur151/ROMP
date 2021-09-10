import numpy as np
import quaternion


def rot_matrix_to_axan(data):
    """
    Converts rotation matrices to axis angles
    :param data: Rotation matrices. Shape: (Persons, Seq, 24, 3, 3)
    :return: Axis angle representation of inpute matrices. Shape: (Persons, Seq, 24, 3)
    """
    aa = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(data))
    return aa


def axan_to_rot_matrix(data):
    """
    Converts the axis angle representation to a rotation matrix.
    :param data: Axis angle. Shape (batch,  24*3).
    :return: an array of shape (num_people, seq_length, 24, 3, 3).
    """
    # reshape to have sensor values explicit
    data_c = np.array(data, copy=True)
    # n = 24
    batch, n = data_c.shape[0], int(data_c.shape[1] / 3)
    data_r = np.reshape(data_c, [batch, n, 3])

    qs = quaternion.from_rotation_vector(data_r)
    rot = np.array(quaternion.as_rotation_matrix(qs))
    # size Batch x 24 x 3 x 3

    # check this
    rot = np.resize(rot, (batch, 24 * 3 * 3))
    # finally we get Batch X 24*3*3
    return rot


def with_zeros(data):
    """
    Appends a [0, 0, 0, 1] vector to all the 3 X 4 matrices in the batch
    Args:
        data: matrix shape Batch X 3 X 4
    Returns: matrix shape Batch X 4 X 4

    """
    batch_size = data.shape[0]
    padding = np.array([0.0, 0.0, 0.0, 1.0])

    # This creates a list of data and a padding array with size Batch X 1 X 4

    concat_list = [data, np.tile(np.resize(padding, (1, 1, 4)), (batch_size, 1, 1))]
    cat_res = np.concatenate(concat_list, axis=1)
    return cat_res


def pack(data):
    """
    changes a matrix of size B x 4 x 1 to matrix of size B x 4 x 4 where all the additional values are zero
    This is useful for multiplying the global transform with the inverse of the pose transform
    Args:
        data: BATCH x 4 x 1
    Returns:

    """
    batch_size = data.shape[0]
    padding = np.zeros((batch_size, 4, 3))
    pack_list = [padding, data]
    pack_res = np.concatenate(pack_list, 2)
    return pack_res


def subtract_flat_id(rot_mats):
    """
    does R(\theta) - R(\theta*)
    R(\theta*) is a contatenation of identity matrices
    Args:
        rot_mats: shape: BATCH X 207
    Returns:

    """
    # Subtracts identity as a flattened tensor
    id_flat = np.eye(3, dtype=rot_mats.dtype)
    id_flat = np.resize(id_flat, (1, 9))
    id_flat = np.tile(id_flat, (rot_mats.shape[0], 23))
    results = rot_mats - id_flat
    return results
