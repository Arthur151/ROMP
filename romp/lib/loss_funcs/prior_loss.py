from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sys
import os
import time
import pickle
import numpy as np

from config import args

DEFAULT_DTYPE = torch.float32

class Interperlation_penalty(nn.Module):
    def __init__(self, faces_tensor, df_cone_height = 0.5,  point2plane=False, penalize_outside=True, max_collisions=8,\
        part_segm_fn=None): # os.path.join(config.project_dir,"model/smplx/smplx_parts_segm.pkl")
        super(Interperlation_penalty, self).__init__()

        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
            sigma=df_cone_height, point2plane=point2plane,
            vectorized=True, penalize_outside=penalize_outside).cuda()
        self.coll_loss_weight = 1.0
        self.search_tree = BVH(max_collisions=max_collisions).cuda()
        self.body_model_faces = faces_tensor

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            self.tri_filtering_module = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents).cuda()
    def forward(self,vertices):
        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        batch_size = vertices.shape[0]
        triangles = torch.index_select(vertices, 1,
            self.body_model_faces).view(batch_size, -1, 3, 3)

        with torch.no_grad():
            collision_idxs = self.search_tree(triangles)

        # Remove unwanted collisions
        if self.tri_filtering_module is not None:
            collision_idxs = self.tri_filtering_module(collision_idxs)

        if collision_idxs.ge(0).sum().item() > 0:
            pen_loss = torch.sum(
                self.coll_loss_weight *
                self.pen_distance(triangles, collision_idxs))
        return pen_loss

def vposer_valid():
    vposer, pose_embedding = [None, ] * 2

    pose_embedding = torch.zeros([batch_size, 32],
                                 dtype=dtype, device=device,
                                 requires_grad=True)

    vposer_ckpt = osp.expandvars(vposer_ckpt)
    vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
    vposer = vposer.to(device=device)
    vposer.eval()

    body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                 dtype=dtype)

    with torch.no_grad():
        pose_embedding.fill_(0)

    body_params = list(body_model.parameters())

    final_params = list(
        filter(lambda x: x.requires_grad, body_params))

    if use_vposer:
        final_params.append(pose_embedding)
    result['body_pose'] = pose_embedding.detach().cpu().numpy()
    body_pose = vposer.decode(
            pose_embedding,
            output_type='aa').view(1, -1)

def create_prior(prior_type, **kwargs):
    if prior_type == 'gmm':
        prior = MaxMixturePrior(**kwargs)
    elif prior_type == 'l2':
        return L2Prior(**kwargs)
    elif prior_type == 'angle':
        return SMPLifyAnglePrior(**kwargs)
    elif prior_type == 'none' or prior_type is None:
        # Don't use any pose prior
        def no_prior(*args, **kwargs):
            return 0.0
        prior = no_prior
    else:
        raise ValueError('Prior {}'.format(prior_type) + ' is not implemented')
    return prior

def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return (torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2).sum(dim=-1)


class SMPLifyAnglePrior(nn.Module):
    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()

        # Indices for the roration angle of
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)

        angle_prior_signs = np.array([1, -1, -1, -1],
                                     dtype=np.float32 if dtype == torch.float32
                                     else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs,
                                         dtype=dtype)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        ''' Returns the angle prior loss for the given pose
        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        '''
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] *
                         self.angle_prior_signs).pow(2)


class L2Prior(nn.Module):
    def __init__(self, dtype=DEFAULT_DTYPE, reduction='sum', **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        return torch.sum(module_input.pow(2))


class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder=args().smpl_model_path,
                 num_gaussians=8, dtype=DEFAULT_DTYPE, epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        assert os.path.exists(full_gmm_fn),print('The path to the mixture prior {} does not exist'.format(full_gmm_fn))

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        param_num = pose.shape[1]
        diff_from_mean = pose.unsqueeze(dim=1) - self.means[:,:param_num]

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions[:,:param_num,:param_num], diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)

class MultiLossFactory(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.num_stages = 1

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in [True]
                ]
            )
        self.heatmaps_loss_factor = [1.]

        self.ae_loss = \
            nn.ModuleList(
                [
                    AELoss('exp') if with_ae_loss else None
                    for with_ae_loss in [True]
                ]
            )
        self.push_loss_factor = [0.001]
        self.pull_loss_factor = [0.001]

    def forward(self, outputs, heatmaps, masks, joints):
        # forward check
        self._forward_check(outputs, heatmaps, masks, joints)

        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmaps[idx], masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                push_loss, pull_loss = self.ae_loss[idx](
                    tags_pred, joints[idx]
                )
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)

        return heatmaps_losses, push_losses, pull_losses

    def _forward_check(self, outputs, heatmaps, masks, joints):
        assert isinstance(outputs, list), \
            'outputs should be a list, got {} instead.'.format(type(outputs))
        assert isinstance(heatmaps, list), \
            'heatmaps should be a list, got {} instead.'.format(type(heatmaps))
        assert isinstance(masks, list), \
            'masks should be a list, got {} instead.'.format(type(masks))
        assert isinstance(joints, list), \
            'joints should be a list, got {} instead.'.format(type(joints))
        assert len(outputs) == self.num_stages, \
            'len(outputs) and num_stages should been same, got {} vs {}.'.format(len(outputs), self.num_stages)
        assert len(outputs) == len(heatmaps), \
            'outputs and heatmaps should have same length, got {} vs {}.'.format(len(outputs), len(heatmaps))
        assert len(outputs) == len(masks), \
            'outputs and masks should have same length, got {} vs {}.'.format(len(outputs), len(masks))
        assert len(outputs) == len(joints), \
            'outputs and joints should have same length, got {} vs {}.'.format(len(outputs), len(joints))
        assert len(outputs) == len(self.heatmaps_loss), \
            'outputs and heatmaps_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.heatmaps_loss))
        assert len(outputs) == len(self.ae_loss), \
            'outputs and ae_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.ae_loss))


if __name__ == '__main__':
    GMM = MaxMixturePrior()
    result = GMM(torch.rand(16,63), torch.rand(16,10))
    print(result.sum(-1))
    print(angle_prior(torch.rand(16,63)).sum(dim=-1))