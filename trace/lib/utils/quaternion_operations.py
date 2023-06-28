import torch
import math

def q_mul(q1, q2):
    """
    Multiply quaternion q1 with q2.
    Expects two equally-sized tensors of shape [*, 4], where * denotes any number of dimensions.
    Returns q1*q2 as a tensor of shape [*, 4].
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4
    original_shape = q1.shape

    # Compute outer product
    terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def wrap_angle(theta):
    """
    Helper method: Wrap the angles of the input tensor to lie between -pi and pi.
    Odd multiples of pi are wrapped to +pi (as opposed to -pi).
    """
    pi_tensor = torch.ones_like(theta, device=theta.device) * math.pi
    result = ((theta + pi_tensor) % (2 * pi_tensor)) - pi_tensor
    result[result.eq(-pi_tensor)] = math.pi

    return result


def q_angle(q):
    """
    Determine the rotation angle of given quaternion tensors of shape [*, 4].
    Return as tensor of shape [*, 1]
    """
    assert q.shape[-1] == 4

    q = q_normalize(q)
    q_re, q_im = torch.split(q, [1, 3], dim=-1)
    norm = torch.linalg.norm(q_im, dim=-1).unsqueeze(dim=-1)
    angle = 2.0 * torch.atan2(norm, q_re)

    return wrap_angle(angle)


def q_normalize(q):
    """
    Normalize the coefficients of a given quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
    return  torch.div(q, norm[:, None])  # q_norm = q / ||q||


def q_conjugate(q):
    """
    Returns the complex conjugate of the input quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    conj = torch.tensor([1, -1, -1, -1], device=q.device)  # multiplication coefficients per element
    return q * conj.expand_as(q)



# these parameters can be tuned!
LAMBDA_ROT = 1 / math.pi  # divide by maxmimum possible rotation angle (pi)
# for LAMBDA_TRANS, assume that translation coeffs. are normalized in 3D eucl. space
LAMBDA_TRANS = 1 / (2 * math.sqrt(3))  # divide by maximum possible translation (2 * unit cube diagonal)

def dq_distance(dq_pred, dq_real):
    '''
    Calculates the screw motion parameters between dual quaternion representations of the given poses pose_pred/real.
    This screw motion describes the "shortest" rigid transformation between dq_pred and dq_real.
    A combination of that transformation's screw axis translation magnitude and rotation angle can be used as a metric.
    => "Distance" between two dual quaternions: weighted sum of screw motion axis magnitude and rotation angle.
    '''

    dq_pred, dq_real = dq_normalize(dq_pred), dq_normalize(dq_real)
    dq_pred_inv = dq_quaternion_conjugate(dq_pred)  # inverse is quat. conj. because it's normalized
    dq_diff = dq_mul(dq_pred_inv, dq_real)
    _, _, theta, d = dq_to_screw(dq_diff)
    distances = LAMBDA_ROT * torch.abs(theta) + LAMBDA_TRANS * torch.abs(d)
    return torch.mean(distances)

def quaternion_difference(q1, q2):
    q1_normed, q2_normed = q_normalize(q1), q_normalize(q2)
    q1_inv = q_conjugate(q1_normed)  # inverse is quat. conj. because it's normalized
    difference = q_mul(q1_inv, q2_normed)
    return difference
